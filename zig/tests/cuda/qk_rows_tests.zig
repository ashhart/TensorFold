//! qk_rows.cu against core/qk_ref.zig: q, k within float64 tolerance, v and the caches exact, bits at any row count.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");

const counts = [_]usize{ 1, 3, 17, 300 };
const most = counts[counts.len - 1];
const positions = 4096;

fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn bf(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

const Case = struct { h: usize, hk: usize, dim: usize, rot: usize, norm: bool };

fn run1(gpu: check.Gpu, a: std.mem.Allocator, s: cuda.Stream, g: *const cuda.qk_rows.QkRows, rng: std.Random, c: Case, worst: *f64) !void {
    const heads = c.h + 2 * c.hk;
    const width = heads * c.dim;
    const xs = try a.alloc(u16, most * width);
    for (xs, 0..) |*h, i| h.* = bf16(rng.floatNorm(f32) * (if (i % 37 == 0) @as(f32, 8) else 1));
    const nw = try a.alloc(f32, 2 * c.dim);
    for (nw) |*w| w.* = 0.5 + rng.float(f32);
    const tc = try a.alloc(f32, positions * @max(c.rot, 1));
    const ts = try a.alloc(f32, positions * @max(c.rot, 1));
    for (0..positions) |p| for (0..c.rot) |i| {
        const j = i % (c.rot / 2);
        const ang = @as(f64, @floatFromInt(p)) / std.math.pow(f64, 10000, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(c.rot)));
        tc[p * c.rot + i] = @floatCast(@cos(ang));
        ts[p * c.rot + i] = @floatCast(@sin(ang));
    };
    const len = 600;
    const pos = try a.alloc(i32, most);
    const slot = try a.alloc(i32, most);
    for (pos, slot, 0..) |*p, *sl, i| {
        p.* = @intCast((i * 7919 + 13) % positions);
        sl.* = @intCast(i % 2);
    }
    // distinct (slot, pos % len) a row, so no row's cache entry overwrites another's
    for (0..most) |i| pos[i] = @intCast((i / 2) % len + len * (i % 5));
    var bufs: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(gpu.gpa);
    }
    const up = struct {
        fn f(gp: check.Gpu, l: *std.ArrayList(cuda.DeviceBuffer), bytes: []const u8) !u64 {
            const b = try cuda.DeviceBuffer.fromHost(gp.d, bytes);
            try l.append(gp.gpa, b);
            return b.ptr;
        }
        fn z(gp: check.Gpu, l: *std.ArrayList(cuda.DeviceBuffer), st: cuda.Stream, n: usize) !u64 {
            const b = try cuda.DeviceBuffer.alloc(gp.d, n);
            try l.append(gp.gpa, b);
            try b.fill8(0xff, st.handle);
            return b.ptr;
        }
    };
    const dx = try up.f(gpu, &bufs, std.mem.sliceAsBytes(xs));
    const dnw = try up.f(gpu, &bufs, std.mem.sliceAsBytes(nw));
    const dc = try up.f(gpu, &bufs, std.mem.sliceAsBytes(tc));
    const ds = try up.f(gpu, &bufs, std.mem.sliceAsBytes(ts));
    const dp = try up.f(gpu, &bufs, std.mem.sliceAsBytes(pos));
    const dsl = try up.f(gpu, &bufs, std.mem.sliceAsBytes(slot));
    const shape: cuda.qk_rows.Shape = .{ .norms = if (c.norm) dnw else null, .cos = if (c.rot > 0) dc else null, .sin = if (c.rot > 0) ds else null, .rot = c.rot };
    const cache_bytes = 2 * len * c.hk * c.dim * 2;
    const outs = try a.alloc(cuda.qk_rows.Out, counts.len);
    for (counts, outs) |m, *o| {
        o.* = .{ .q = try up.z(gpu, &bufs, s, m * c.h * c.dim * 2), .k = try up.z(gpu, &bufs, s, m * c.hk * c.dim * 2), .v = try up.z(gpu, &bufs, s, m * c.hk * c.dim * 2), .kc = try up.z(gpu, &bufs, s, cache_bytes), .vc = try up.z(gpu, &bufs, s, cache_bytes), .len = len };
        try g.rows(s, dx, .{ .h = c.h, .hk = c.hk, .dim = c.dim }, shape, dp, dsl, o.*, m);
    }
    try s.synchronize();
    const o = outs[counts.len - 1];
    const back = struct {
        fn f(gp: check.Gpu, al: std.mem.Allocator, ptr: u64, n: usize) ![]u16 {
            const raw = try al.alloc(u16, n);
            const b: cuda.DeviceBuffer = .{ .d = gp.d, .ptr = ptr, .len = n * 2 };
            try b.download(0, std.mem.sliceAsBytes(raw));
            return raw;
        }
    };
    const gq = try back.f(gpu, a, o.q, most * c.h * c.dim);
    const gk = try back.f(gpu, a, o.k, most * c.hk * c.dim);
    const gv = try back.f(gpu, a, o.v, most * c.hk * c.dim);
    const gkc = try back.f(gpu, a, o.kc.?, cache_bytes / 2);
    const gvc = try back.f(gpu, a, o.vc.?, cache_bytes / 2);
    const xv = try a.alloc(f64, c.dim);
    const wq = try a.alloc(f64, c.dim);
    const wk = try a.alloc(f64, c.dim);
    for (wq, wk, 0..) |*p, *q, i| {
        p.* = nw[i];
        q.* = nw[c.dim + i];
    }
    const cv = try a.alloc(f64, @max(c.rot, 1));
    const sv = try a.alloc(f64, @max(c.rot, 1));
    const y = try a.alloc(f64, c.dim);
    var q_most: f64 = 0;
    for (0..most) |r| {
        const p: usize = @intCast(pos[r]);
        for (0..c.rot) |i| {
            cv[i] = tc[p * c.rot + i];
            sv[i] = ts[p * c.rot + i];
        }
        const at = @as(usize, @intCast(slot[r])) * len + p % len;
        for (0..heads) |hd| {
            for (xv, 0..) |*v, i| v.* = bf(xs[r * width + hd * c.dim + i]);
            if (hd >= c.h + c.hk) { // v: exact copies, in the output and the cache
                const k = hd - c.h - c.hk;
                for (0..c.dim) |i| {
                    try check.expect(gv[(r * c.hk + k) * c.dim + i] == xs[r * width + hd * c.dim + i], "qk rows: v copied", .{});
                    try check.expect(gvc[(at * c.hk + k) * c.dim + i] == xs[r * width + hd * c.dim + i], "qk rows: v cached", .{});
                }
                continue;
            }
            const is_q = hd < c.h;
            core.qk_ref.head(xv, if (c.norm) (if (is_q) wq else wk) else null, 1e-6, if (c.rot > 0) cv else null, if (c.rot > 0) sv else null, c.rot, y);
            for (y, 0..) |want, i| {
                const got = if (is_q) bf(gq[(r * c.h + hd) * c.dim + i]) else bf(gk[(r * c.hk + hd - c.h) * c.dim + i]);
                const tol = (@abs(want) + 2 * @abs(xv[i]) + 1e-3) * std.math.pow(f64, 2, -8);
                q_most = @max(q_most, @abs(got - want) / tol);
                if (!is_q) try check.expect(gkc[(at * c.hk + hd - c.h) * c.dim + i] == gk[(r * c.hk + hd - c.h) * c.dim + i], "qk rows: k cached as output", .{});
            }
        }
    }
    worst.* = @max(worst.*, q_most);
    // the same heads from [q | gate] rows and a padded k buffer, no v: the same q, k and k cache bits
    const qg = try a.alloc(u16, most * c.h * 2 * c.dim);
    const kb = try a.alloc(u16, most * (c.hk * c.dim + 64));
    @memset(qg, 0x7fc0); // the gate halves: NaN, never read
    @memset(kb, 0x7fc0);
    for (0..most) |r| {
        for (0..c.h) |hd| @memcpy(qg[(r * c.h + hd) * 2 * c.dim ..][0..c.dim], xs[r * width + hd * c.dim ..][0..c.dim]);
        @memcpy(kb[r * (c.hk * c.dim + 64) ..][0 .. c.hk * c.dim], xs[r * width + c.h * c.dim ..][0 .. c.hk * c.dim]);
    }
    const src: cuda.qk_rows.Src = .{ .q = try up.f(gpu, &bufs, std.mem.sliceAsBytes(qg)), .q_ld = @intCast(c.h * 2 * c.dim), .q_hs = @intCast(2 * c.dim), .k = try up.f(gpu, &bufs, std.mem.sliceAsBytes(kb)), .k_ld = @intCast(c.hk * c.dim + 64), .k_hs = @intCast(c.dim) };
    const o2: cuda.qk_rows.Out = .{ .q = try up.z(gpu, &bufs, s, most * c.h * c.dim * 2), .k = try up.z(gpu, &bufs, s, most * c.hk * c.dim * 2), .v = 0, .kc = try up.z(gpu, &bufs, s, cache_bytes), .len = len };
    try g.rowsFrom(s, src, .{ .h = c.h, .hk = c.hk, .dim = c.dim }, shape, dp, dsl, o2, most);
    try s.synchronize();
    try check.sameBytes("qk rows from strided heads: q", std.mem.sliceAsBytes(try back.f(gpu, a, o2.q, most * c.h * c.dim)), std.mem.sliceAsBytes(gq));
    try check.sameBytes("qk rows from strided heads: k", std.mem.sliceAsBytes(try back.f(gpu, a, o2.k, most * c.hk * c.dim)), std.mem.sliceAsBytes(gk));
    try check.sameBytes("qk rows from strided heads: k cache", std.mem.sliceAsBytes(try back.f(gpu, a, o2.kc.?, cache_bytes / 2)), std.mem.sliceAsBytes(gkc));
    // head-major outputs [heads, rows, dim] (strides): the packed call's bits, rearranged
    {
        const o4: cuda.qk_rows.Out = .{ .q = try up.z(gpu, &bufs, s, most * c.h * c.dim * 2), .k = try up.z(gpu, &bufs, s, most * c.hk * c.dim * 2), .v = try up.z(gpu, &bufs, s, most * c.hk * c.dim * 2), .len = len, .q_ld = c.dim, .q_hs = most * c.dim, .kv_ld = c.dim, .kv_hs = most * c.dim };
        try g.rows(s, dx, .{ .h = c.h, .hk = c.hk, .dim = c.dim }, shape, dp, dsl, o4, most);
        try s.synchronize();
        const hq = try back.f(gpu, a, o4.q, most * c.h * c.dim);
        const hk_ = try back.f(gpu, a, o4.k, most * c.hk * c.dim);
        const hv = try back.f(gpu, a, o4.v, most * c.hk * c.dim);
        for (0..most) |r| {
            for (0..c.h) |hd| try check.sameBytes("qk rows head-major: q", std.mem.sliceAsBytes(hq[(hd * most + r) * c.dim ..][0..c.dim]), std.mem.sliceAsBytes(gq[(r * c.h + hd) * c.dim ..][0..c.dim]));
            for (0..c.hk) |hd| {
                try check.sameBytes("qk rows head-major: k", std.mem.sliceAsBytes(hk_[(hd * most + r) * c.dim ..][0..c.dim]), std.mem.sliceAsBytes(gk[(r * c.hk + hd) * c.dim ..][0..c.dim]));
                try check.sameBytes("qk rows head-major: v", std.mem.sliceAsBytes(hv[(hd * most + r) * c.dim ..][0..c.dim]), std.mem.sliceAsBytes(gv[(r * c.hk + hd) * c.dim ..][0..c.dim]));
            }
        }
    }
    // bf16 norm weights at their own addresses and RoPE from inv (angle = pos * inv in fp32): float64 tolerance
    if (c.norm) {
        const wqb = try a.alloc(u16, c.dim);
        const wkb = try a.alloc(u16, c.dim);
        for (wqb, wkb, 0..) |*p, *q, i| {
            p.* = bf16(nw[i]);
            q.* = bf16(nw[c.dim + i]);
        }
        const half = @max(c.rot / 2, 1);
        const inv = try a.alloc(f32, half);
        for (inv, 0..) |*v, j| v.* = @floatCast(1 / std.math.pow(f64, 10000, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(@max(c.rot, 1)))));
        const sh3: cuda.qk_rows.Shape = .{ .norms = try up.f(gpu, &bufs, std.mem.sliceAsBytes(wqb)), .norm_k = try up.f(gpu, &bufs, std.mem.sliceAsBytes(wkb)), .norms_bf16 = true, .inv = if (c.rot > 0) try up.f(gpu, &bufs, std.mem.sliceAsBytes(inv)) else null, .rot = c.rot };
        const o3: cuda.qk_rows.Out = .{ .q = try up.z(gpu, &bufs, s, most * c.h * c.dim * 2), .k = try up.z(gpu, &bufs, s, most * c.hk * c.dim * 2), .v = try up.z(gpu, &bufs, s, most * c.hk * c.dim * 2), .len = len };
        try g.rows(s, dx, .{ .h = c.h, .hk = c.hk, .dim = c.dim }, sh3, dp, dsl, o3, most);
        try s.synchronize();
        const q3 = try back.f(gpu, a, o3.q, most * c.h * c.dim);
        const k3 = try back.f(gpu, a, o3.k, most * c.hk * c.dim);
        for (wq, wk, 0..) |*p, *q, i| {
            p.* = bf(wqb[i]);
            q.* = bf(wkb[i]);
        }
        for (0..most) |r| {
            const p: usize = @intCast(pos[r]);
            for (0..c.rot) |i| {
                const ang: f64 = @as(f32, @floatFromInt(p)) * inv[i % half];
                cv[i] = @cos(ang);
                sv[i] = @sin(ang);
            }
            for (0..c.h + c.hk) |hd| {
                for (xv, 0..) |*v, i| v.* = bf(xs[r * width + hd * c.dim + i]);
                const is_q = hd < c.h;
                core.qk_ref.head(xv, if (is_q) wq else wk, 1e-6, if (c.rot > 0) cv else null, if (c.rot > 0) sv else null, c.rot, y);
                for (y, 0..) |want, i| {
                    const got = if (is_q) bf(q3[(r * c.h + hd) * c.dim + i]) else bf(k3[(r * c.hk + hd - c.h) * c.dim + i]);
                    q_most = @max(q_most, @abs(got - want) / ((@abs(want) + 2 * @abs(xv[i]) + 1e-3) * std.math.pow(f64, 2, -8)));
                }
            }
        }
        worst.* = @max(worst.*, q_most);
    }
    try check.expect(q_most <= 1, "qk rows {d}/{d}x{d}, rot {d}{s}: within float64 tolerance (worst {d:.3})", .{ c.h, c.hk, c.dim, c.rot, if (c.norm) ", normed" else "", q_most });
    for (counts[0 .. counts.len - 1], outs[0 .. counts.len - 1]) |m, oo| {
        try check.sameBytes("qk rows: q at a smaller row count", std.mem.sliceAsBytes(try back.f(gpu, a, oo.q, m * c.h * c.dim)), std.mem.sliceAsBytes(gq[0 .. m * c.h * c.dim]));
        try check.sameBytes("qk rows: k at a smaller row count", std.mem.sliceAsBytes(try back.f(gpu, a, oo.k, m * c.hk * c.dim)), std.mem.sliceAsBytes(gk[0 .. m * c.hk * c.dim]));
    }
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try cuda.qk_rows.QkRows.load(gpu.d);
    defer g.unload();
    var prng = std.Random.DefaultPrng.init(31);
    var worst: f64 = 0;
    const a = arena.allocator();
    try run1(gpu, a, s, &g, prng.random(), .{ .h = 48, .hk = 4, .dim = 128, .rot = 128, .norm = true }, &worst); // Kolibri, sliding
    try run1(gpu, a, s, &g, prng.random(), .{ .h = 48, .hk = 4, .dim = 128, .rot = 0, .norm = true }, &worst); // Kolibri, full
    try run1(gpu, a, s, &g, prng.random(), .{ .h = 8, .hk = 2, .dim = 64, .rot = 32, .norm = false }, &worst); // partial rotary
    try run1(gpu, a, s, &g, prng.random(), .{ .h = 16, .hk = 2, .dim = 128, .rot = 128, .norm = false }, &worst); // a warp a head, no norm
    try run1(gpu, a, s, &g, prng.random(), .{ .h = 24, .hk = 4, .dim = 256, .rot = 64, .norm = true }, &worst); // Qwen3.8-27B
    check.pass("qk rows: q and k within float64 tolerance (worst {d:.3}), v and the caches exact, bits at any row count", .{worst});
}
