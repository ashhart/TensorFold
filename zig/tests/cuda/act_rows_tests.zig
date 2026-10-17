//! act_rows.cu's SwiGLU, output gate, group sums and DFlash2's dynamic conv against float64, bits at every row count.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rows = @import("norm_rows_tests.zig");

const aref = core.act_ref;
const counts = rows.counts;
const most = rows.most;
const Rig = rows.Rig;
const bf16 = rows.bf16;
const bf = rows.bf;

fn swiglu(r: *Rig, g: *const cuda.act_rows.ActRows, n: usize) !void {
    const gu = r.bfs(most * 2 * n, 3); // [gate | up] rows
    const d = try r.dev(std.mem.sliceAsBytes(gu));
    var os: [counts.len]u64 = undefined;
    var xss: [counts.len]u64 = undefined;
    for (counts, 0..) |m, c| {
        os[c] = try r.out(m * n * 2);
        xss[c] = try r.out(m * n / 64 * 4);
        try g.swiglu(r.s, d, 2 * n, d + 2 * n, 2 * n, os[c], xss[c], m, n);
    }
    var name: [64]u8 = undefined;
    const label = try std.fmt.bufPrint(&name, "swiglu {d}", .{n});
    const o = try r.back(u16, os[counts.len - 1], most * n);
    const xs = try r.back(f32, xss[counts.len - 1], most * n / 64);
    for (0..most) |row| for (0..n) |i| {
        const w = aref.swiglu(bf(gu[row * 2 * n + i]), bf(gu[row * 2 * n + n + i]));
        try r.within(label, bf(o[row * n + i]), w, aref.tolerance(w));
    };
    try r.sums(label, o, xs, most, n);
    for (counts[0 .. counts.len - 1], 0..) |m, c|
        try check.sameBytes(label, std.mem.sliceAsBytes(try r.back(u16, os[c], m * n)), std.mem.sliceAsBytes(o[0 .. m * n]));
}

fn outputGate(r: *Rig, g: *const cuda.act_rows.ActRows, heads: usize, d: usize) !void {
    const n = heads * d;
    const o = r.bfs(most * n, 1);
    const qg = r.bfs(most * 2 * n, 3); // [q_h | gate_h] a head
    const do_ = try r.dev(std.mem.sliceAsBytes(o));
    const dq = try r.dev(std.mem.sliceAsBytes(qg));
    var os: [counts.len]u64 = undefined;
    var xss: [counts.len]u64 = undefined;
    for (counts, 0..) |m, c| {
        os[c] = try r.out(m * n * 2);
        xss[c] = try r.out(m * n / 64 * 4);
        try g.outputGate(r.s, do_, dq + 2 * d, 2 * n, 2 * d, os[c], xss[c], m, heads, d);
    }
    var name: [64]u8 = undefined;
    const label = try std.fmt.bufPrint(&name, "output gate {d} heads of {d}", .{ heads, d });
    const got = try r.back(u16, os[counts.len - 1], most * n);
    const xs = try r.back(f32, xss[counts.len - 1], most * n / 64);
    for (0..most) |row| for (0..heads) |h| for (0..d) |e| {
        const w = bf(o[row * n + h * d + e]) * aref.sigmoid(bf(qg[row * 2 * n + h * 2 * d + d + e]));
        try r.within(label, bf(got[row * n + h * d + e]), w, aref.tolerance(w));
    };
    try r.sums(label, got, xs, most, n);
    for (counts[0 .. counts.len - 1], 0..) |m, c|
        try check.sameBytes(label, std.mem.sliceAsBytes(try r.back(u16, os[c], m * n)), std.mem.sliceAsBytes(got[0 .. m * n]));
}

fn groupSums(r: *Rig, g: *const cuda.act_rows.ActRows, k: usize) !void {
    const ld = k + 64;
    const x = r.bfs(most * ld, 2);
    const d = try r.dev(std.mem.sliceAsBytes(x));
    const xsd = try r.out(most * k / 64 * 4);
    try g.groupSums(r.s, d, ld, xsd, most, k);
    const xs = try r.back(f32, xsd, most * k / 64);
    const dense = try r.a.alloc(u16, most * k);
    for (0..most) |row| @memcpy(dense[row * k ..][0..k], x[row * ld ..][0..k]);
    try r.sums("group sums", dense, xs, most, k);
}

fn dconv(r: *Rig, g: *const cuda.act_rows.ActRows, d: usize, gs: usize, seg: usize, branch: usize, residual: bool) !void {
    const x = r.bfs(most * d, 1);
    const dyn = r.bfs(most * 4 * (d / gs), 0.3);
    const base = r.bfs(4 * d, 0.7);
    const res = r.bfs(most * d, 1);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const ddyn = try r.dev(std.mem.sliceAsBytes(dyn));
    const dbase = try r.dev(std.mem.sliceAsBytes(base));
    const dres = try r.dev(std.mem.sliceAsBytes(res));
    var os: [counts.len]u64 = undefined;
    for (counts, 0..) |m, c| {
        os[c] = try r.out(m * d * 2);
        try g.dconv(r.s, dx, ddyn, dbase, if (residual) dres else null, os[c], m, d, gs, branch, seg);
    }
    var name: [64]u8 = undefined;
    const label = try std.fmt.bufPrint(&name, "dconv d {d} gs {d} seg {d} branch {d}{s}", .{ d, gs, seg, branch, if (residual) " +res" else "" });
    const got = try r.back(u16, os[counts.len - 1], most * d);
    const half = std.math.pow(f64, 2, -8);
    for (0..most) |row| for (0..d) |i| {
        const grp = i / gs;
        const k0 = bf(bf16(@as(f32, @floatCast(bf(base[(branch * 2) * d + i]))) + @as(f32, @floatCast(bf(dyn[((row * 2 + branch) * 2) * (d / gs) + grp])))));
        const k1 = bf(bf16(@as(f32, @floatCast(bf(base[(branch * 2 + 1) * d + i]))) + @as(f32, @floatCast(bf(dyn[((row * 2 + branch) * 2 + 1) * (d / gs) + grp])))));
        const xv = bf(x[row * d + i]);
        const pv = if (row % seg > 0) bf(x[(row - 1) * d + i]) else 0;
        const y = aref.dconv(xv, pv, k0, k1);
        const yt = (@abs(xv * k0) + @abs(pv * k1)) * std.math.pow(f64, 2, -22);
        if (residual) {
            const rv = bf(res[row * d + i]);
            try r.within(label, bf(got[row * d + i]), rv + y, @abs(rv + y) * half + @abs(y) * half + yt + 1e-30);
        } else try r.within(label, bf(got[row * d + i]), y, @abs(y) * half + yt + 1e-30);
    };
    for (counts[0 .. counts.len - 1], 0..) |m, c|
        try check.sameBytes(label, std.mem.sliceAsBytes(try r.back(u16, os[c], m * d)), std.mem.sliceAsBytes(got[0 .. m * d]));
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(27);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = try cuda.Stream.init(gpu.d, false), .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
        r.s.deinit();
    }
    var g = try cuda.act_rows.ActRows.load(gpu.d);
    defer g.unload();
    try swiglu(&r, &g, 17408); // Qwen3.8-27B's MLP
    try swiglu(&r, &g, 192);
    try outputGate(&r, &g, 24, 256); // the 27B's attention output gate
    try outputGate(&r, &g, 3, 64);
    try groupSums(&r, &g, 5120);
    try dconv(&r, &g, 5120, 16, 16, 0, false); // DFlash2's block conv (groups of 16)
    try dconv(&r, &g, 5120, 16, 16, 1, true);
    try dconv(&r, &g, 256, 64, 7, 0, true);
    check.pass("act rows: swiglu, output gate, dconv within float64 tolerance (worst {d:.3}), group sums of the stored outputs, every row count's bits equal", .{r.worst});
}
