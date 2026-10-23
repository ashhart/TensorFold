//! GLM's projections in the kernels' reduction order vs float64 (worst column reported); narrower calls = widest bits.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const glm = @import("glm_cuda");
const Gpu = check.Gpu;
const Rig = qr.RigT;
const bf = qr.helpers.bfOf;

const b16 = glm.b16;
const glue = glm.glue;

/// RESULT for one check from each column's worst fraction of its tolerance: the worst column named, all within 1.
pub fn worstCol(r: *Rig, name: []const u8, col: []const f64) !void {
    var most: f64 = 0;
    var at: usize = 0;
    for (col, 0..) |q, j| if (q > most) {
        most = q;
        at = j;
    };
    r.worst = @max(r.worst, most);
    std.debug.print("RESULT {s}: worst column {d} at {e:.3} of its tolerance\n", .{ name, at, most });
    try check.expect(most <= 1, "{s} within its float64 tolerance", .{name});
}

/// Every element within its tolerance of `want`; RESULT names the worst column (n a row) and its worst fraction.
pub fn closeCols(r: *Rig, name: []const u8, ptr: u64, f32_out: bool, want: []const f64, tol: []const f64, n: usize) !void {
    const raw = try r.back(ptr, want.len * @as(usize, if (f32_out) 4 else 2));
    const col = try r.a.alloc(f64, n);
    @memset(col, 0);
    for (want, tol, 0..) |w, t, i| {
        const g: f64 = if (f32_out) std.mem.bytesToValue(f32, raw[4 * i ..][0..4]) else bf(std.mem.bytesToValue(u16, raw[2 * i ..][0..2]));
        const q = @abs(g - w) / t;
        if (!(q <= col[i % n])) col[i % n] = if (std.math.isNan(q)) std.math.inf(f64) else q;
    }
    try worstCol(r, name, col);
}

/// The first `bytes` of a narrower call equal the widest call's.
pub fn sameRows(r: *Rig, name: []const u8, wide: u64, narrow: u64, bytes: usize) !void {
    try check.sameBytes(name, try r.back(narrow, bytes), try r.back(wide, bytes));
}

/// y = x w^T in float64 over bf16 rows; bound: output rounding (2^-8 |y| for bf16) + fp32 (K + 8) 2^-24 of sum |x w|.
pub fn dots(r: *Rig, x: []const u16, w: []const u16, m: usize, n: usize, k: usize, f32_out: bool) !struct { want: []f64, tol: []f64 } {
    const want = try r.a.alloc(f64, m * n);
    const tol = try r.a.alloc(f64, m * n);
    const wd = try r.a.alloc(f64, n * k);
    for (wd, w) |*d, v| d.* = bf(v);
    const xd = try r.a.alloc(f64, k);
    for (0..m) |row| {
        for (xd, x[row * k ..][0..k]) |*d, v| d.* = bf(v);
        for (0..n) |j| {
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (xd, wd[j * k ..][0..k]) |a, b| {
                sum += a * b;
                abs += @abs(a * b);
            }
            want[row * n + j] = sum;
            const chain = abs * @as(f64, @floatFromInt(k + 8)) * std.math.pow(f64, 2, -24);
            tol[row * n + j] = (if (f32_out) 0 else @abs(sum) * std.math.pow(f64, 2, -8)) + chain + 1e-30;
        }
    }
    return .{ .want = want, .tol = tol };
}

/// bf16 rows uploaded one element on, so they start 2 bytes past 16: every launch takes the unstaged kernels.
fn shifted(r: *Rig, x: []const u16) !u64 {
    const s = try r.a.alloc(u16, x.len + 1);
    s[0] = 0;
    @memcpy(s[1..], x);
    return (try r.dev(std.mem.sliceAsBytes(s))) + 2;
}

const Shape = struct { n: usize, k: usize };
const decode_rows = [_]usize{ 1, 3, 16, 17, 40, 127 };
const wide_rows = 300;

/// b16.matmul at 300 rows vs float64; unaligned x (bmm_rows), decode row counts (bmm_dtile, bmm_decode) vs its bytes.
fn matmuls(r: *Rig, set: glue.Set, s: Shape) !void {
    const m = wide_rows;
    const x = try r.bfs(m * s.k, 1.0);
    const w = try r.bfs(s.n * s.k, 0.05);
    const xs = [2]u64{ try r.dev(std.mem.sliceAsBytes(x)), try shifted(r, x) };
    const dw = try r.dev(std.mem.sliceAsBytes(w));
    const part = try r.zeros(b16.partBytes(decode_rows[decode_rows.len - 1], s.n, s.k));
    var name: [128]u8 = undefined;
    for ([_]bool{ false, true }) |wide| {
        const eb: usize = if (wide) 4 else 2;
        const ref = try dots(r, x, w, m, s.n, s.k, wide);
        const full = try r.zeros(m * s.n * eb);
        try b16.matmul(set, r.s, xs[0], s.k, dw, full, part, m, s.n, s.k, wide);
        const kind = if (wide) "fp32" else "bf16";
        try closeCols(r, try std.fmt.bufPrint(&name, "b16 matmul {d} rows x [{d}, {d}] (K in {d} slices), {s} out", .{ m, s.n, s.k, b16.splitK(s.n, s.k), kind }), full, wide, ref.want, ref.tol, s.n);
        const o = try r.zeros(m * s.n * eb);
        try b16.matmul(set, r.s, xs[1], s.k, dw, o, part, m, s.n, s.k, wide);
        try sameRows(r, try std.fmt.bufPrint(&name, "b16 matmul [{d}, {d}] {s}: unaligned rows equal the staged tiles'", .{ s.n, s.k, kind }), full, o, m * s.n * eb);
        for (decode_rows) |dm| for (xs, [_][]const u8{ "aligned", "unaligned" }) |xp, how| {
            const d = try r.zeros(dm * s.n * eb);
            try b16.matmul(set, r.s, xp, s.k, dw, d, part, dm, s.n, s.k, wide);
            try sameRows(r, try std.fmt.bufPrint(&name, "b16 matmul [{d}, {d}] {s}: {d} {s} rows equal the {d}-row call's", .{ s.n, s.k, kind, dm, how, m }), full, d, dm * s.n * eb);
        };
    }
    check.pass("b16 matmul [{d}, {d}] ({d} slices): bf16 and fp32 out within float64 tolerance, every entry and row count's bits equal", .{ s.n, s.k, b16.splitK(s.n, s.k) });
}

/// glue.router (8 K slices) at 300 rows vs float64, fewer rows and unaligned x vs its bytes; glue.select vs float64.
fn routers(r: *Rig, set: glue.Set) !void {
    const m = wide_rows;
    const d = 4096;
    const ne = 256;
    const top_k = 8;
    const scale: f32 = 2.5;
    const x = try r.bfs(m * d, 1.0);
    const w = try r.bfs(ne * d, 0.02);
    const xs = [2]u64{ try r.dev(std.mem.sliceAsBytes(x)), try shifted(r, x) };
    const dw = try r.dev(std.mem.sliceAsBytes(w));
    const part = try r.zeros(glue.router_ks * m * ne * 4);
    const full = try r.zeros(m * ne * 4);
    try glue.router(set, r.s, xs[0], d, dw, full, part, m, d, ne);
    const ref = try dots(r, x, w, m, ne, d, true);
    var name: [128]u8 = undefined;
    try closeCols(r, try std.fmt.bufPrint(&name, "router {d} rows x [{d}, {d}] (K in {d} slices)", .{ m, ne, d, glue.router_ks }), full, true, ref.want, ref.tol, ne);
    for ([_]usize{ 1, 3, 63, 64, 100, m }) |dm| for (xs, [_][]const u8{ "aligned", "unaligned" }) |xp, how| {
        const o = try r.zeros(dm * ne * 4);
        try glue.router(set, r.s, xp, d, dw, o, part, dm, d, ne);
        try sameRows(r, try std.fmt.bufPrint(&name, "router: {d} {s} rows equal the {d}-row call's", .{ dm, how, m }), full, o, dm * ne * 4);
    };
    check.pass("router [{d}, {d}]: within float64 tolerance, router_tile and router_part bits equal at every row count", .{ ne, d });

    const bias = try r.a.alloc(f32, ne);
    for (bias) |*b| b.* = (r.rng.float(f32) * 2 - 1) * 0.05;
    const pick = try r.zeros(m * (top_k + 1) * 4);
    const wts = try r.zeros(m * (top_k + 1) * 4);
    try glue.select(set, r.s, full, try r.dev(std.mem.sliceAsBytes(bias)), pick, wts, m, top_k, ne, scale, true);
    const logits = try r.back(full, m * ne * 4);
    const got_pick = try r.back(pick, m * (top_k + 1) * 4);
    const got_wts = try r.back(wts, m * (top_k + 1) * 4);
    var swaps: usize = 0;
    var most: f64 = 0;
    const score = try r.a.alloc(f64, ne);
    const order = try r.a.alloc(usize, ne);
    for (0..m) |row| {
        for (score, 0..) |*sc, e| sc.* = 1 / (1 + @exp(-@as(f64, std.mem.bytesToValue(f32, logits[4 * (row * ne + e) ..][0..4]))));
        for (order, 0..) |*o, e| o.* = e;
        const Ctx = struct { s: []const f64, b: []const f32 };
        std.mem.sort(usize, order, Ctx{ .s = score, .b = bias }, struct {
            fn lt(c: Ctx, a: usize, b: usize) bool {
                const ca = c.s[a] + c.b[a];
                const cb = c.s[b] + c.b[b];
                return ca > cb or (ca == cb and a < b);
            }
        }.lt);
        var total: f64 = 0;
        for (0..top_k) |i| total += score[@intCast(std.mem.bytesToValue(i32, got_pick[4 * (row * (top_k + 1) + i) ..][0..4]))];
        for (0..top_k + 1) |i| {
            const at = 4 * (row * (top_k + 1) + i);
            const id: usize = @intCast(std.mem.bytesToValue(i32, got_pick[at..][0..4]));
            const wt: f64 = std.mem.bytesToValue(f32, got_wts[at..][0..4]);
            if (i == top_k) {
                try check.expect(id == ne and wt == 1, "select row {d}: the shared expert last at weight 1", .{row});
                continue;
            }
            if (id != order[i]) { // a near tie in float64 may swap: the scores' approximate exp is within 2^-20
                const gap = @abs(score[id] + bias[id] - score[order[i]] - bias[order[i]]);
                try check.expect(gap <= std.math.pow(f64, 2, -20), "select row {d} pick {d}: {d} where float64 ranks {d} (gap {e})", .{ row, i, id, order[i], gap });
                swaps += 1;
            }
            const want = score[id] / total * scale;
            most = @max(most, @abs(wt - want) / (@abs(want) * std.math.pow(f64, 2, -16)));
        }
    }
    r.worst = @max(r.worst, most);
    std.debug.print("RESULT select {d} rows, top {d} of {d}: weights at {e:.3} of 2^-16 relative, {d} near-tie swaps\n", .{ m, top_k, ne, most, swaps });
    try check.expect(most <= 1, "select weights within 2^-16 of float64", .{});
    check.pass("select top-{d} of {d}: picks follow the float64 ranking (near ties aside), weights within float64 tolerance", .{ top_k, ne });
}

/// `glm-ref`: the checks above on synthetic inputs (seeded), no checkpoint and no golden.
pub fn run(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(593);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer mod.unload();
    const ks = try glue.Kernels.load(mod);
    const set: glue.Set = .{ .glue = &ks };
    for ([_]Shape{ .{ .n = 512, .k = 4096 }, .{ .n = 1280, .k = 1536 }, .{ .n = 1024, .k = 128 } }) |sh| try matmuls(&r, set, sh);
    try routers(&r, set);
    try @import("glm_ref_attn.zig").dense(&r, set);
    try @import("glm_ref_attn.zig").sparse(&r, set);
    try @import("glm_ref_fp8.zig").run(&r, set);
    try @import("glm_ref_hc.zig").mixing(&r, set);
    try @import("glm_ref_kda.zig").chains(&r);
    var ops = try cuda.Module.load(gpu.d, cuda.kernels.glm_ops);
    defer ops.unload();
    try @import("glm_ref_index.zig").selection(&r, set, ops);
    try @import("glm_ref_hc.zig").gammas(&r, set); // last: its draws from r.rng follow every check above
    check.pass("glm-ref: every projection within {e:.3} of its float64 tolerance", .{r.worst});
}
