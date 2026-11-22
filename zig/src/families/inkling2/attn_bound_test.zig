//! The kernels' order on the host against the float64 formula, inside the page's bound: fixture, seams, pressure.
const std = @import("std");
const attn = @import("attn.zig");
const order = @import("attn_order.zig");
const bound = @import("attn_bound.zig");
const fixture = @import("attn_test.zig");
const bf16_round = @import("bf16_round");

const t = std.testing;

/// The worst error over bound of a call, before and after the output's rounding; each element is inside its bound.
fn worst(a: std.mem.Allocator, s: attn.Shape, in: attn.Inputs) ![2]f64 {
    const n = s.rows * s.heads * s.hd;
    const raw = try a.alloc(f32, n);
    defer a.free(raw);
    const out = try a.alloc(f32, n);
    defer a.free(out);
    try order.attendOrder(s, in, raw, .{ .round_out = false });
    try order.attendOrder(s, in, out, .{});
    var w: [2]f64 = .{ 0, 0 };
    for (0..s.rows) |row| for (0..s.heads) |h| {
        const hb = try bound.head(a, s, in, row, h);
        defer hb.free(a);
        for (0..s.hd) |d| {
            const i = (row * s.heads + h) * s.hd + d;
            const r0 = @abs(@as(f64, raw[i]) - hb.want[d]) / hb.limit(d);
            const r1 = @abs(@as(f64, out[i]) - hb.want[d]) / hb.limitOut(d);
            if (!(r0 <= 1) or !(r1 <= 1)) {
                std.debug.print("outside the bound: row {d} head {d} d {d}: {d} vs {d}, ratios {e} {e}\n", .{ row, h, d, raw[i], hb.want[d], r0, r1 });
                return error.OutsideBound;
            }
            w[0] = @max(w[0], r0);
            w[1] = @max(w[1], r1);
        }
    };
    return w;
}

fn report(name: []const u8, w: [2]f64) void {
    std.debug.print("attn bound {s}: worst error/bound {e:.2} (fp32), {e:.2} (bf16 output)\n", .{ name, w[0], w[1] });
}

test "bound: the fixture" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (try fixture.cases(a), 0..) |c, i| {
        var buf: [32]u8 = undefined;
        report(try std.fmt.bufPrint(&buf, "fixture case {d}", .{i}), try worst(a, c.shape, c.in));
    }
}

/// A call's arrays from a spec: bf16 values from a fixed generator, amplitudes per array.
const Spec = struct {
    rows: usize = 1,
    pos0: usize,
    heads: usize = 2,
    kv_heads: usize = 1,
    hd: usize = 16,
    drel: usize = 4,
    rext: usize = 64,
    window: usize = 0,
    q: f32 = 0.5,
    k: f32 = 0.5,
    v: f32 = 1,
    rel: f32 = 0.5,
    proj: f32 = 0.5,
    sink: f32 = 0,
    gate: f32 = 1,
    rise: f32 = 0, // keys' first coordinate climbs to this over the cache, q's is 1: the maximum rises
    alt: bool = false, // values alternate in sign by position
    seed: u64 = 1,
};

const Built = struct {
    s: attn.Shape,
    in: attn.Inputs,

    fn free(b: Built, a: std.mem.Allocator) void {
        inline for (.{ b.in.q, b.in.k, b.in.v, b.in.rel, b.in.proj, b.in.sink, b.in.gate }) |x| a.free(x);
    }
};

fn fill(a: std.mem.Allocator, seed: *u64, n: usize, amp: f32) ![]f32 {
    const out = try a.alloc(f32, n);
    for (out) |*v| {
        seed.* = seed.* *% 1103515245 +% 12345;
        const r = @as(f32, @floatFromInt((seed.* >> 16) & 0x7fff)) / 16384.0 - 1.0;
        v.* = bf16_round.round32(r * amp);
    }
    return out;
}

fn build(a: std.mem.Allocator, p: Spec) !Built {
    const s: attn.Shape = .{ .rows = p.rows, .pos0 = p.pos0, .heads = p.heads, .kv_heads = p.kv_heads, .hd = p.hd, .drel = p.drel, .rext = p.rext, .window = p.window };
    var seed = p.seed;
    const keys = p.pos0 + p.rows;
    const in: attn.Inputs = .{
        .q = try fill(a, &seed, p.rows * p.heads * p.hd, p.q),
        .k = try fill(a, &seed, keys * p.kv_heads * p.hd, p.k),
        .v = try fill(a, &seed, keys * p.kv_heads * p.hd, p.v),
        .rel = try fill(a, &seed, p.rows * p.heads * p.drel, p.rel),
        .proj = try fill(a, &seed, p.drel * p.rext, p.proj),
        .sink = try fill(a, &seed, p.rows * p.heads, p.sink),
        .gate = try fill(a, &seed, p.rows * p.heads * p.hd, p.gate),
    };
    const q: []f32 = @constCast(in.q);
    const k: []f32 = @constCast(in.k);
    const v: []f32 = @constCast(in.v);
    if (p.rise != 0) {
        for (0..p.rows * p.heads) |i| q[i * p.hd] = 1;
        for (0..keys * p.kv_heads) |i| k[i * p.hd] = bf16_round.round32(p.rise * @as(f32, @floatFromInt(i / p.kv_heads)) / @as(f32, @floatFromInt(keys)));
    }
    if (p.alt) for (0..keys * p.kv_heads) |i| {
        for (v[i * p.hd ..][0..p.hd]) |*x| x.* = if (i / p.kv_heads % 2 == 0) @abs(x.*) + 1 else -@abs(x.*) - 1;
    };
    return .{ .s = s, .in = in };
}

fn run(name: []const u8, p: Spec) !void {
    const b = try build(t.allocator, p);
    defer b.free(t.allocator);
    report(name, try worst(t.allocator, b.s, b.in));
}

test "bound: global seams" {
    inline for (.{ 62, 63, 64, 1022, 1023, 1024, 16382, 16383, 16384, 20000 }) |pos| {
        var buf: [40]u8 = undefined;
        try run(try std.fmt.bufPrint(&buf, "global pos {d}", .{pos}), .{ .pos0 = pos, .rise = 6, .seed = pos });
    }
    try run("global 3 rows, rows on both sides of a seam", .{ .pos0 = 1022, .rows = 3, .heads = 4, .kv_heads = 2, .rise = 6 });
}

test "bound: sliding seams" {
    inline for (.{ 100, 255, 256, 511, 700, 1023, 1024, 3000 }) |pos| {
        var buf: [40]u8 = undefined;
        try run(try std.fmt.bufPrint(&buf, "sliding pos {d}", .{pos}), .{ .pos0 = pos, .window = 1024, .rext = 1024, .rise = 6, .seed = pos });
    }
    try run("sliding window 200, 3 rows", .{ .pos0 = 300, .rows = 3, .window = 200, .rext = 200, .rise = 4 });
    // 64 rows in one block (one head a group): the last row's window starts a tile after the first row's
    try run("sliding window 200, 64 rows in a block", .{ .pos0 = 1000, .rows = 64, .heads = 1, .rext = 200, .window = 200, .rise = 4 });
}

test "bound: head 128" {
    try run("head 128, global, pos 1500", .{ .pos0 = 1500, .hd = 128, .drel = 16, .rext = 1024, .heads = 6, .rise = 4 });
    try run("head 128, sliding, pos 1500", .{ .pos0 = 1500, .hd = 128, .drel = 16, .rext = 1024, .window = 1024, .heads = 6, .rise = 4 });
}

test "bound: cancellation in P.V" {
    try run("cancellation", .{ .pos0 = 2000, .q = 0, .rel = 0, .alt = true, .v = 4 });
}

test "bound: large score gap" {
    try run("large score gap", .{ .pos0 = 3000, .q = 6, .k = 6, .rel = 0, .hd = 16 });
}

test "bound: P on a half-step" {
    // the bias b (a bf16 value) makes the other key's weight e^b against 1: pick the b whose e^b is nearest a half-step
    var best: f32 = -1;
    var gap: f64 = 1;
    var h: u16 = 0xbc00; // -0.0078125 .. -8
    while (h < 0xc100) : (h += 1) {
        const b = bf16_round.value(h);
        const p: f64 = @exp(@as(f64, b));
        const step = std.math.pow(f64, 2, @floor(std.math.log2(p)) - 7);
        const x = p / step;
        const d = @abs(x - @floor(x) - 0.5);
        if (d < gap) {
            gap = d;
            best = b;
        }
    }
    std.debug.print("attn bound half-step: bias {d} gives e^b within {e:.2} of a step from the half-step\n", .{ best, gap });
    const s: attn.Shape = .{ .rows = 1, .pos0 = 1, .heads = 1, .kv_heads = 1, .hd = 1, .drel = 1, .rext = 1, .window = 0 };
    const in: attn.Inputs = .{ .q = &.{0}, .k = &.{ 0, 0 }, .v = &.{ 1, -1 }, .rel = &.{1}, .proj = &.{best}, .sink = &.{-20}, .gate = &.{20} };
    report("half-step", try worst(t.allocator, s, in));
}

test "bound: dominant sink" {
    try run("dominant sink", .{ .pos0 = 1500, .sink = 0, .q = 0, .rel = 0 });
    var p: Spec = .{ .pos0 = 1500, .q = 0.1, .k = 0.1, .rel = 0 };
    p.sink = 8;
    try run("dominant sink 8", p);
}

test "bound: tiny P" {
    const sh: attn.Shape = .{ .rows = 1, .pos0 = 2, .heads = 1, .kv_heads = 1, .hd = 1, .drel = 1, .rext = 2, .window = 0 };
    // bias -88 at distance 0 and -86 at distance 1 against 0 at distance 2: weights 2^-126.96 and 2^-124.1 (relative)
    const in: attn.Inputs = .{ .q = &.{0}, .k = &.{ 0, 0, 0 }, .v = &.{ 1, 2, 3 }, .rel = &.{1}, .proj = &.{ -88, -86 }, .sink = &.{-20}, .gate = &.{20} };
    report("tiny P", try worst(t.allocator, sh, in));
}

/// One key at position 0, hd 1: q and k zero, the score 0, so the value `v` passes through the gate's sigmoid.
fn single(v: f32, gate: f32, sink: f32) struct { s: attn.Shape, in: attn.Inputs } {
    const s: attn.Shape = .{ .rows = 1, .pos0 = 0, .heads = 1, .kv_heads = 1, .hd = 1, .drel = 1, .rext = 1, .window = 0 };
    const Static = struct {
        var vs: [1]f32 = undefined;
        var g: [1]f32 = undefined;
        var sk: [1]f32 = undefined;
    };
    Static.vs[0] = v;
    Static.g[0] = gate;
    Static.sk[0] = sink;
    return .{ .s = s, .in = .{ .q = &.{0}, .k = &.{0}, .v = &Static.vs, .rel = &.{0}, .proj = &.{0}, .sink = &Static.sk, .gate = &Static.g } };
}

test "bound: the domain ends where fp32 overflows or the sigmoid leaves the normal range" {
    const a = t.allocator;
    // e^87 = 2^125.5 < 2^126, e^88 = 2^126.96, e^88.5 = 2^127.68, e^89 = 2^128.4 > FLT_MAX: -87 is the last gate inside
    const ok = single(1, -87, -100);
    (try bound.head(a, ok.s, ok.in, 0, 0)).free(a);
    for ([_]f32{ -88, -88.5, -89, -inf32 }) |g| {
        const c = single(1, g, -100);
        try t.expectError(error.AttnDomain, bound.head(a, c.s, c.in, 0, 0));
    }
    // a dot past 2^120: q and k of 2^70
    var c = single(1, 0, -100);
    var q = [1]f32{0x1p70};
    var k = [1]f32{0x1p70};
    c.in.q = &q;
    c.in.k = &k;
    try t.expectError(error.AttnDomain, bound.head(a, c.s, c.in, 0, 0));
    // a non-finite value at a key another row of the block loads, though it is masked for this row
    const two: attn.Shape = .{ .rows = 2, .pos0 = 0, .heads = 1, .kv_heads = 1, .hd = 1, .drel = 1, .rext = 1, .window = 0 };
    const masked: attn.Inputs = .{ .q = &.{ 0, 0 }, .k = &.{ 0, 0 }, .v = &.{ 1, inf32 }, .rel = &.{ 0, 0 }, .proj = &.{0}, .sink = &.{ -100, -100 }, .gate = &.{ 0, 0 } };
    try t.expectError(error.AttnDomain, bound.head(a, two, masked, 0, 0));
    // a non-finite value at a key the call loads, a non-finite tap, a sink past 2^120
    c = single(inf32, 0, -100);
    try t.expectError(error.AttnDomain, bound.head(a, c.s, c.in, 0, 0));
    c = single(1, 0, 0x1p121);
    try t.expectError(error.AttnDomain, bound.head(a, c.s, c.in, 0, 0));
    c = single(1, 0, -100);
    c.in.proj = &.{-inf32};
    c.in.rel = &.{1};
    try t.expectError(error.AttnDomain, bound.head(a, c.s, c.in, 0, 0));
}

const inf32 = std.math.inf(f32);

test "bound: the domain ends at 2^20 live keys" {
    const a = t.allocator;
    inline for (.{ .{ (1 << 20) - 1, true }, .{ 1 << 20, false } }) |c| {
        const b = try build(a, .{ .pos0 = c[0], .heads = 1, .hd = 1, .drel = 1, .rext = 1, .q = 0, .k = 0, .rel = 0, .proj = 0, .gate = 0 });
        defer b.free(a);
        if (c[1]) {
            (try bound.head(a, b.s, b.in, 0, 0)).free(a);
        } else {
            try t.expectError(error.AttnDomain, bound.head(a, b.s, b.in, 0, 0));
        }
    }
}

test "bound: an output below the normal range" {
    // a value through a tiny sigmoid: its last product rounds on the 2^-149 grid, which the scaled floor alone misses
    const a = t.allocator;
    var found: usize = 0;
    for ([_]f32{ -87, -86 }) |gate| {
        var h: u16 = 0x3200; // 2^-27 ... up
        while (h < 0x3500) : (h += 1) {
            const v = bf16_round.value(h);
            const c = single(v, gate, -100);
            var raw: [1]f32 = undefined;
            try order.attendOrder(c.s, c.in, &raw, .{ .round_out = false });
            const hb = try bound.head(a, c.s, c.in, 0, 0);
            defer hb.free(a);
            const err = @abs(@as(f64, raw[0]) - hb.want[0]);
            const without_abs = (1 + 0x1p-20) * (hb.mag[0] * std.math.expm1(hb.big) + hb.sg[0] * hb.floor * @exp(hb.big));
            if (err > without_abs) {
                found += 1;
                try t.expect(err <= hb.limit(0));
                if (found == 1) std.debug.print("attn bound below normal: v {e} gate {d}: error {e:.3}, bound without the absolute term {e:.3}, bound {e:.3}\n", .{ v, gate, err, without_abs, hb.limit(0) });
            }
        }
    }
    std.debug.print("attn bound below normal: {d} of the values exceed the bound without its absolute term and stay inside the bound\n", .{found});
    try t.expect(found > 0);
}

test "bound: where exp(L) overflows the bound is infinite, never NaN" {
    // q = k = 2^50 and v = 0: the model gives 0, float64 gives 0, and mag * expm1(L) is 0 * inf
    const a = t.allocator;
    const s: attn.Shape = .{ .rows = 1, .pos0 = 0, .heads = 1, .kv_heads = 1, .hd = 1, .drel = 1, .rext = 1, .window = 0 };
    const in: attn.Inputs = .{ .q = &.{0x1p50}, .k = &.{0x1p50}, .v = &.{0}, .rel = &.{0}, .proj = &.{0}, .sink = &.{0}, .gate = &.{0} };
    const hb = try bound.head(a, s, in, 0, 0);
    defer hb.free(a);
    var raw: [1]f32 = undefined;
    try order.attendOrder(s, in, &raw, .{ .round_out = false });
    try t.expectEqual(@as(f32, 0), raw[0]);
    try t.expectEqual(@as(f64, 0), hb.want[0]);
    try t.expect(hb.big > 709.8);
    try t.expectEqual(std.math.inf(f64), hb.limit(0));
    try t.expectEqual(std.math.inf(f64), hb.limitOut(0));
    std.debug.print("attn bound overflow of exp(L): L {e:.2}, limit {d}\n", .{ hb.big, hb.limit(0) });
}

test "bound: a table rounded up at the weak key and down at the dominant one" {
    // the first key dominates; the weak second key's table sum 64.2539 rounds up, the dominant one's 64.2461 down
    const a = t.allocator;
    const s: attn.Shape = .{ .rows = 1, .pos0 = 1, .heads = 1, .kv_heads = 1, .hd = 1, .drel = 2, .rext = 2, .window = 0 };
    var best0: f64 = 0;
    var best1: f64 = 0;
    var bestk: f32 = 0;
    var h: u16 = 0xc200; // -32 up
    while (h < 0xc2f0) : (h += 1) {
        const k1 = bf16_round.value(h);
        var kk = [2]f32{ 0, k1 };
        var vv = [2]f32{ 0, 1 };
        const in: attn.Inputs = .{ .q = &.{1}, .k = &kk, .v = &vv, .rel = &.{ 1, 1 }, .proj = &.{ 64, 64, 0.25390625, 0.24609375 }, .sink = &.{-100}, .gate = &.{100} };
        var raw: [1]f32 = undefined;
        var out: [1]f32 = undefined;
        try order.attendOrder(s, in, &raw, .{ .round_out = false });
        try order.attendOrder(s, in, &out, .{});
        const hb = try bound.head(a, s, in, 0, 0);
        defer hb.free(a);
        const r0 = @abs(@as(f64, raw[0]) - hb.want[0]) / hb.limit(0);
        const r1 = @abs(@as(f64, out[0]) - hb.want[0]) / hb.limitOut(0);
        if (r0 > best0) {
            best0 = r0;
            bestk = k1;
        }
        best1 = @max(best1, r1);
    }
    std.debug.print("attn bound table adversary: worst error/bound {e:.3} (fp32) at k {d}, {e:.3} (bf16 output)\n", .{ best0, bestk, best1 });
    try t.expect(best0 <= 1 and best1 <= 1);
    try t.expect(best0 > 0.5);
}
