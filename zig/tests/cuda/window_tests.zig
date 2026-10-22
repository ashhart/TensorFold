//! window_attention.cu against core/attention_ref.zig: decode rows over a ring of keys, bits at any row count.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const dump = @import("byte_dump.zig");

const counts = [_]usize{ 1, 3, 17, 64 };
const most = counts[counts.len - 1];

fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn bf(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

const Case = struct { h: usize, hk: usize, d: usize, win: usize, ring: usize, streams: usize, top: usize };

fn run1(gpu: check.Gpu, a: std.mem.Allocator, s: cuda.Stream, g: *const cuda.window_attention.WindowAttention, rng: std.Random, c: Case, worst: *f64) !void {
    const gq = c.h / c.hk;
    const qn = most * c.h * c.d;
    const cn = c.streams * c.ring * c.hk * c.d;
    const qs = try a.alloc(u16, qn);
    const ks = try a.alloc(u16, cn);
    const vs = try a.alloc(u16, cn);
    for (qs) |*x| x.* = bf16(rng.floatNorm(f32));
    for (ks) |*x| x.* = bf16(rng.floatNorm(f32));
    for (vs) |*x| x.* = bf16(rng.floatNorm(f32));
    const pos = try a.alloc(i32, most);
    const slot = try a.alloc(i32, most);
    for (pos, slot, 0..) |*p, *sl, i| {
        p.* = @intCast(if (i % 4 == 0) i % 40 else (i * 7919) % c.top); // short histories and wrapped rings both
        sl.* = @intCast(i % c.streams);
    }
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
    };
    const dq = try up.f(gpu, &bufs, std.mem.sliceAsBytes(qs));
    const dk = try up.f(gpu, &bufs, std.mem.sliceAsBytes(ks));
    const dv = try up.f(gpu, &bufs, std.mem.sliceAsBytes(vs));
    const dp = try up.f(gpu, &bufs, std.mem.sliceAsBytes(pos));
    const dsl = try up.f(gpu, &bufs, std.mem.sliceAsBytes(slot));
    const scale: f32 = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(c.d))));
    const w: cuda.window_attention.Window = .{ .h = c.h, .hk = c.hk, .d = c.d, .win = c.win, .ring = c.ring, .scale = scale };
    const scratch = try cuda.DeviceBuffer.alloc(gpu.d, cuda.window_attention.WindowAttention.scratchBytes(w, most));
    try bufs.append(gpu.gpa, scratch);
    const outs = try a.alloc(cuda.DeviceBuffer, counts.len);
    for (counts, outs) |m, *o| {
        o.* = try cuda.DeviceBuffer.alloc(gpu.d, m * c.h * c.d * 2);
        try bufs.append(gpu.gpa, o.*);
        try o.fill8(0xff, s.handle);
        try g.decode(s, dq, dk, dv, o.ptr, dp, dsl, w, scratch.ptr, m);
    }
    try s.synchronize();
    const wide = try a.alloc(u16, qn);
    try outs[counts.len - 1].download(0, std.mem.sliceAsBytes(wide));
    try dump.output(a, "window", "decode", "rows", &.{ most, c.h, c.d }, .bf16, c, std.mem.sliceAsBytes(wide));
    // per (row, key head): the window's keys gathered from the ring into dense rows, then each query head
    const kd = try a.alloc(f64, c.top * c.d);
    const vd = try a.alloc(f64, c.top * c.d);
    const keys = try a.alloc(usize, c.top);
    const q = try a.alloc(f64, c.d);
    const want = try a.alloc(f64, c.d);
    const tol = try a.alloc(f64, c.d);
    var q_most: f64 = 0;
    for (0..most) |r| {
        const p: usize = @intCast(pos[r]);
        const lo = if (c.win > 0 and p + 1 > c.win) p + 1 - c.win else 0;
        const nk = p + 1 - lo;
        for (0..c.hk) |kh| {
            for (0..nk) |j| {
                const row = @as(usize, @intCast(slot[r])) * c.ring + (lo + j) % c.ring;
                for (0..c.d) |e| {
                    kd[j * c.d + e] = bf(ks[(row * c.hk + kh) * c.d + e]);
                    vd[j * c.d + e] = bf(vs[(row * c.hk + kh) * c.d + e]);
                }
                keys[j] = j;
            }
            for (0..gq) |hh| {
                const hd = kh * gq + hh;
                for (q, 0..) |*x, e| x.* = bf(qs[(r * c.h + hd) * c.d + e]);
                core.attention_ref.head(q, kd, vd, keys[0..nk], c.d, scale, want, tol);
                for (want, tol, 0..) |x, t, e| q_most = @max(q_most, @abs(bf(wide[(r * c.h + hd) * c.d + e]) - x) / t);
            }
        }
    }
    worst.* = @max(worst.*, q_most);
    try check.expect(q_most <= 1, "window decode h {d}/{d} d {d} win {d}: within float64 tolerance (worst {d:.3})", .{ c.h, c.hk, c.d, c.win, q_most });
    for (counts[0 .. counts.len - 1], outs[0 .. counts.len - 1]) |m, o| {
        const got = try a.alloc(u16, m * c.h * c.d);
        try o.download(0, std.mem.sliceAsBytes(got));
        try dump.output(a, "window", "decode", "rows", &.{ m, c.h, c.d }, .bf16, c, std.mem.sliceAsBytes(got));
        try check.sameBytes("window decode: a row count's bits equal the widest call's", std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(wide[0 .. m * c.h * c.d]));
    }
}

const PCase = struct { h: usize, hk: usize, win: usize, ring: usize, p0: usize, n: usize };

/// Prompt rows p0 .. p0 + n: float64 tolerance, a shorter call's bits, a later chunk's bits (from p0 + n / 2).
fn prompt1(gpu: check.Gpu, a: std.mem.Allocator, s: cuda.Stream, g: *const cuda.window_attention.WindowAttention, rng: std.Random, c: PCase, worst: *f64) !void {
    const d = 128;
    const qn = c.n * c.h * d;
    const crow = if (c.ring > 0) c.ring else c.p0 + c.n; // ring 0: a row a position
    const cn = crow * c.hk * d;
    const qs = try a.alloc(u16, qn);
    const ks = try a.alloc(u16, cn);
    const vs = try a.alloc(u16, cn);
    for (qs) |*x| x.* = bf16(rng.floatNorm(f32));
    for (ks) |*x| x.* = bf16(rng.floatNorm(f32));
    for (vs) |*x| x.* = bf16(rng.floatNorm(f32));
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
    const dq = try up.f(gpu, &bufs, std.mem.sliceAsBytes(qs));
    const dk = try up.f(gpu, &bufs, std.mem.sliceAsBytes(ks));
    const dv = try up.f(gpu, &bufs, std.mem.sliceAsBytes(vs));
    const scale: f32 = @floatCast(1 / @sqrt(@as(f64, d)));
    const w: cuda.window_attention.Window = .{ .h = c.h, .hk = c.hk, .d = d, .win = c.win, .ring = c.ring, .scale = scale };
    const full = try up.z(gpu, &bufs, s, qn * 2);
    try g.prompt(s, dq, dk, dv, full, c.p0, c.n, w);
    const pre_n = 17;
    const pre = try up.z(gpu, &bufs, s, pre_n * c.h * d * 2);
    try g.prompt(s, dq, dk, dv, pre, c.p0, pre_n, w);
    const half = c.n / 2;
    const late = try up.z(gpu, &bufs, s, (c.n - half) * c.h * d * 2);
    try g.prompt(s, dq + half * c.h * d * 2, dk, dv, late, c.p0 + half, c.n - half, w);
    try s.synchronize();
    const got = try a.alloc(u16, qn);
    const fb: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = full, .len = qn * 2 };
    try fb.download(0, std.mem.sliceAsBytes(got));
    try dump.output(a, "window", "prompt", "full", &.{ c.n, c.h, d }, .bf16, c, std.mem.sliceAsBytes(got));
    const gq = c.h / c.hk;
    const top = c.p0 + c.n;
    const kd = try a.alloc(f64, @min(top, c.win + 0) * d + top * d);
    const vd = try a.alloc(f64, kd.len);
    const keys = try a.alloc(usize, top);
    const q = try a.alloc(f64, d);
    const want = try a.alloc(f64, d);
    const tol = try a.alloc(f64, d);
    var q_most: f64 = 0;
    var r: usize = 0;
    while (r < c.n) : (r += 7) { // every seventh row, and the last
        const p = c.p0 + r;
        const lo = if (c.win > 0 and p + 1 > c.win) p + 1 - c.win else 0;
        const nk = p + 1 - lo;
        for (0..c.hk) |kh| {
            for (0..nk) |j| {
                const row = if (c.ring > 0) (lo + j) % c.ring else lo + j;
                for (0..d) |e| {
                    kd[j * d + e] = bf(ks[(row * c.hk + kh) * d + e]);
                    vd[j * d + e] = bf(vs[(row * c.hk + kh) * d + e]);
                }
                keys[j] = j;
            }
            for (0..gq) |hh| {
                const hd = kh * gq + hh;
                for (q, 0..) |*x, e| x.* = bf(qs[(r * c.h + hd) * d + e]);
                core.attention_ref.head(q, kd, vd, keys[0..nk], d, scale, want, tol);
                for (want, tol, 0..) |x, t, e| q_most = @max(q_most, @abs(bf(got[(r * c.h + hd) * d + e]) - x) / t);
            }
        }
        if (r + 7 >= c.n and r != c.n - 1) r = c.n - 1 - 7;
    }
    worst.* = @max(worst.*, q_most);
    try check.expect(q_most <= 1, "window prompt h {d}/{d} win {d} ring {d} p0 {d} n {d}: within float64 tolerance (worst {d:.3})", .{ c.h, c.hk, c.win, c.ring, c.p0, c.n, q_most });
    const pb: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = pre, .len = pre_n * c.h * d * 2 };
    const pg = try a.alloc(u16, pre_n * c.h * d);
    try pb.download(0, std.mem.sliceAsBytes(pg));
    try dump.output(a, "window", "prompt", "prefix", &.{ pre_n, c.h, d }, .bf16, c, std.mem.sliceAsBytes(pg));
    try check.sameBytes("window prompt: a shorter call's rows", std.mem.sliceAsBytes(pg), std.mem.sliceAsBytes(got[0 .. pre_n * c.h * d]));
    const lb: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = late, .len = (c.n - half) * c.h * d * 2 };
    const lg = try a.alloc(u16, (c.n - half) * c.h * d);
    try lb.download(0, std.mem.sliceAsBytes(lg));
    try dump.output(a, "window", "prompt", "suffix", &.{ c.n - half, c.h, d }, .bf16, c, std.mem.sliceAsBytes(lg));
    try check.sameBytes("window prompt: a later chunk's rows", std.mem.sliceAsBytes(lg), std.mem.sliceAsBytes(got[half * c.h * d ..]));
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try cuda.window_attention.WindowAttention.load(gpu.d);
    defer g.unload();
    var prng = std.Random.DefaultPrng.init(37);
    var worst: f64 = 0;
    const a = arena.allocator();
    try run1(gpu, a, s, &g, prng.random(), .{ .h = 48, .hk = 4, .d = 128, .win = 513, .ring = 640, .streams = 2, .top = 3000 }, &worst); // Kolibri
    try run1(gpu, a, s, &g, prng.random(), .{ .h = 48, .hk = 4, .d = 128, .win = 513, .ring = 600, .streams = 2, .top = 3000 }, &worst); // the ring's end inside a key tile
    try run1(gpu, a, s, &g, prng.random(), .{ .h = 8, .hk = 2, .d = 64, .win = 100, .ring = 128, .streams = 3, .top = 1000 }, &worst);
    try run1(gpu, a, s, &g, prng.random(), .{ .h = 16, .hk = 1, .d = 128, .win = 0, .ring = 300, .streams = 1, .top = 300 }, &worst); // no window
    try prompt1(gpu, a, s, &g, prng.random(), .{ .h = 48, .hk = 4, .win = 513, .ring = 2624, .p0 = 0, .n = 600 }, &worst); // Kolibri, first chunk
    try prompt1(gpu, a, s, &g, prng.random(), .{ .h = 48, .hk = 4, .win = 513, .ring = 2624, .p0 = 2300, .n = 900 }, &worst); // the ring wraps
    try prompt1(gpu, a, s, &g, prng.random(), .{ .h = 16, .hk = 2, .win = 0, .ring = 0, .p0 = 100, .n = 333 }, &worst); // no window, 8 heads a block
    check.pass("window attention: decode and prompt rows within float64 tolerance (worst {d:.3}), bits at any row count and chunk", .{worst});
}
