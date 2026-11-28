//! The kernels' order on the host against values written out by hand in float64, one rounding point or seam at a time.
const std = @import("std");
const attn = @import("attn.zig");
const order = @import("attn_order.zig");
const bf16_round = @import("bf16_round");

const t = std.testing;
const inf = std.math.inf(f32);
const nan = std.math.nan(f32);
const none: f32 = -100; // a sink that adds 2^-145 to a sum and a gate that leaves the sigmoid at 1

/// hd 1, one head: the keys' scores are `k` (q is 1), no bias, a sink of -100 and a gate of 100 unless given.
const Mini = struct {
    k: []const f32,
    v: []const f32,
    window: usize = 0,
    sink: f32 = none,
    gate: f32 = 100,
    rows: usize = 1,
    heads: usize = 1,
    kv_heads: usize = 1,
    hd: usize = 1,
    q: []const f32 = &.{1},
    rel: f32 = 0,
    rels: []const f32 = &.{}, // the taps of rel, if more than one
    proj: []const f32 = &.{0},
    rext: usize = 1,

    fn run(m: Mini, out: []f32, round_out: bool) !void {
        const pos = m.k.len / m.kv_heads / m.hd; // keys of the cache
        const s: attn.Shape = .{ .rows = m.rows, .pos0 = pos - m.rows, .heads = m.heads, .kv_heads = m.kv_heads, .hd = m.hd, .drel = @max(1, m.rels.len), .rext = m.rext, .window = m.window };
        var qs: [64]f32 = undefined;
        var sinks: [8]f32 = undefined;
        var gates: [64]f32 = undefined;
        var rels: [64]f32 = undefined;
        for (0..m.rows * m.heads) |i| {
            for (0..m.hd) |d| {
                qs[i * m.hd + d] = m.q[d % m.q.len];
                gates[i * m.hd + d] = m.gate;
            }
            sinks[i] = m.sink;
            if (m.rels.len == 0) {
                rels[i] = m.rel;
            } else {
                for (m.rels, 0..) |r, j| rels[i * m.rels.len + j] = r;
            }
        }
        const in: attn.Inputs = .{ .q = qs[0 .. m.rows * m.heads * m.hd], .k = m.k, .v = m.v, .rel = rels[0 .. m.rows * m.heads * @max(1, m.rels.len)], .proj = m.proj, .sink = sinks[0 .. m.rows * m.heads], .gate = gates[0 .. m.rows * m.heads * m.hd] };
        try order.attendOrder(s, in, out, .{ .round_out = round_out });
    }
};

fn near(want: f64, got: f32) !void {
    try t.expectApproxEqRel(want, @as(f64, got), 1e-5);
}

test "table: bf16 once, a tie to even" {
    // rel 1.5 times proj 1.0078125 = 1.51171875, halfway between 1.5078125 and 1.515625: the even one, 1.515625
    var y: [1]f32 = undefined;
    try (Mini{ .k = &.{ 0, 0 }, .v = &.{ 1, 2 }, .q = &.{0}, .rel = 1.5, .proj = &.{ 1.0078125, 0 }, .rext = 2 }).run(&y, false);
    // scores 0 and 1.515625 log2(e); P = bf16(2^-2.18669) = 0.2197265625 and 1: (0.2197265625 + 2) / 1.2197265625
    try near(1.8198558847077662, y[0]);
}

test "P: bf16 once, in the products and the denominator" {
    // scores 0 and -0.5 log2(e): P = 1 and bf16(e^-0.5) = 0.60546875, y = (1 - 0.60546875) / 1.60546875
    var y: [1]f32 = undefined;
    try (Mini{ .k = &.{ 0, -0.5 }, .v = &.{ 1, -1 } }).run(&y, false);
    try near(0.24574209245742093, y[0]);
}

test "output: bf16 once, after the gate" {
    var y: [1]f32 = undefined;
    try (Mini{ .k = &.{ 0, -0.5 }, .v = &.{ 1, -1 } }).run(&y, true);
    try t.expectEqual(@as(u16, 0x3e7c), bf16_round.bits(y[0])); // 0.24574209 rounds to 0.24609375
    try t.expectEqual(@as(u32, 0), @as(u32, @bitCast(y[0])) & 0xffff);
}

test "score: the dot and the scale stay fp32" {
    // q (1, 1), a key (-2, -33 2^-12): the dot -2.008056640625 is not a bf16 value; scale 1 / sqrt(2); the other key 0
    var y: [2]f32 = undefined;
    try (Mini{ .k = &.{ -2, -0.008056640625, 0, 0 }, .v = &.{ 1, 1, 2, 2 }, .hd = 2, .q = &.{ 1, 1 } }).run(&y, false);
    try near(1.8050314465408805, y[0]);
}

test "sink: at the merge, unrounded" {
    // one key at 0, a sink of -0.5: its term is 2^(-0.5 log2(e)) = 0.60653066, not bf16(0.60653066)
    var y: [1]f32 = undefined;
    try (Mini{ .k = &.{0}, .v = &.{1}, .sink = -0.5 }).run(&y, false);
    try near(0.6224593312018546, y[0]);
}

test "gate: its sigmoid" {
    var y: [1]f32 = undefined;
    try (Mini{ .k = &.{0}, .v = &.{1}, .sink = -0.5, .gate = 0 }).run(&y, false);
    try near(0.3112296656009273, y[0]);
}

test "gate: past -88.7 the kernel's sigmoid is 0 where float64's is not" {
    // expf(89) overflows fp32: 1 / (1 + inf) = 0, so the output is 0; float64 has e^-89 = 2.2273635617957434e-39
    var y: [1]f32 = undefined;
    try (Mini{ .k = &.{0}, .v = &.{1}, .gate = -89 }).run(&y, false);
    try t.expectEqual(@as(f32, 0), y[0]);
    var want: [1]f64 = undefined;
    const s: attn.Shape = .{ .rows = 1, .pos0 = 0, .heads = 1, .kv_heads = 1, .hd = 1, .drel = 1, .rext = 1, .window = 0 };
    const in: attn.Inputs = .{ .q = &.{1}, .k = &.{0}, .v = &.{1}, .rel = &.{0}, .proj = &.{0}, .sink = &.{none}, .gate = &.{-89} };
    try attn.attend(f64, false, s, in, &want);
    try t.expectApproxEqRel(2.2273635617957434e-39, want[0], 1e-9);
}

test "window: exactly `window` keys" {
    var y: [1]f32 = undefined;
    try (Mini{ .k = &.{ 0, 0, 0, 0 }, .v = &.{ 10, 20, 30, 40 }, .window = 2 }).run(&y, false);
    try near(35, y[0]);
    try (Mini{ .k = &.{ 0, 0, 0, 0 }, .v = &.{ 10, 20, 30, 40 }, .window = 3 }).run(&y, false);
    try near(30, y[0]);
}

test "bias: distances under rext only" {
    // distance 0 gets 3, distance 1 none: scores 0 and 3 log2(e), P = bf16(2^-4.328) and 1
    var y: [1]f32 = undefined;
    try (Mini{ .k = &.{ 0, 0 }, .v = &.{ 1, 2 }, .q = &.{0}, .rel = 1, .proj = &.{3} }).run(&y, false);
    try near(1.9525581395348837, y[0]);
}

test "group: the group's key head" {
    var y: [4]f32 = undefined;
    try (Mini{ .k = &.{ 0, 0 }, .v = &.{ 1, 2 }, .heads = 4, .kv_heads = 2 }).run(&y, false);
    try t.expectEqualSlices(f32, &.{ 1, 1, 2, 2 }, &y);
}

/// `n` keys of score -1000 (weight 0) and value 0, with the given positions set.
fn keys(a: std.mem.Allocator, n: usize, at: []const struct { usize, f32, f32 }) !struct { k: []f32, v: []f32 } {
    const k = try a.alloc(f32, n);
    const v = try a.alloc(f32, n);
    @memset(k, -1000);
    @memset(v, 0);
    for (at) |e| {
        k[e[0]] = e[1];
        v[e[0]] = e[2];
    }
    return .{ .k = k, .v = v };
}

fn seam(window: usize, n: usize, at: []const struct { usize, f32, f32 }) !f32 {
    const kv = try keys(t.allocator, n, at);
    defer t.allocator.free(kv.k);
    defer t.allocator.free(kv.v);
    var y: [1]f32 = undefined;
    try (Mini{ .k = kv.k, .v = kv.v, .window = window }).run(&y, false);
    return y[0];
}

test "tile seam: P rounds against its tile's maximum, then rescales" {
    // key 0 at 0, key 1 at -0.5, key 64 at +0.5 (log2(e) each): tile 0 rounds P1, tile 1 rescales by 2^-0.72
    const y = try seam(0, 65, &.{ .{ 0, 0, 1 }, .{ 1, -0.5, -1 }, .{ 64, 0.5, 2 } });
    try near(1.134529258560611, y);
}

test "tile: a weak key rounds against a strong one after it" {
    const y = try seam(0, 41, &.{ .{ 0, -0.5, 1 }, .{ 40, 0, -1 } });
    try near(-0.24574209245742093, y);
}

test "sub-chunk seam: starts from its own maximum" {
    // key 1023 at 0, key 1024 at -0.5: its sub-chunk's P is 1, the fold scales it by 2^-0.72 (not bf16)
    const y = try seam(0, 1025, &.{ .{ 1023, 0, 1 }, .{ 1024, -0.5, -1 } });
    try near(0.24491866240370913, y);
}

test "chunk seam at 256: sliding starts afresh, global does not" {
    const at = [_]struct { usize, f32, f32 }{ .{ 255, 0, 1 }, .{ 256, -0.5, -1 } };
    try near(0.24491866240370913, try seam(1024, 257, &at));
    try near(0.24574209245742093, try seam(0, 257, &at));
}

const big: f32 = 16777216; // 2^24: one more is not an fp32 value, two more are

test "tiles: one chain, ones between 2^24 and -2^24 are absorbed" {
    // the mma chain of a sub-chunk runs on across its tiles: 2^24 + 1 + 1 - 2^24 is 0, the exact sum would be 2
    const y = try seam(0, 193, &.{ .{ 0, 0, big }, .{ 64, 0, 1 }, .{ 128, 0, 1 }, .{ 192, 0, -big } });
    try t.expectEqual(@as(f32, 0), @abs(y));
}

test "sub-chunk: summed alone, then folded" {
    // sub-chunk 1 holds the two ones (their own chain gives 2): 2^24, then 2^24 + 2, then 2; y = 2 / 4 keys
    const y = try seam(0, 2049, &.{ .{ 0, 0, big }, .{ 1024, 0, 1 }, .{ 1600, 0, 1 }, .{ 2048, 0, -big } });
    try near(0.5, y);
}

test "sliding chunk: summed alone, then merged" {
    const y = try seam(1024, 513, &.{ .{ 0, 0, big }, .{ 256, 0, 1 }, .{ 448, 0, 1 }, .{ 512, 0, -big } });
    try near(0.5, y);
}

test "global chunk: its sub-chunks fold first" {
    // chunk 1 holds a one in its first sub-chunk and one in its last (folded to 2); chunk 0 holds 2^24, chunk 2 -2^24
    const y = try seam(0, 32769, &.{ .{ 0, 0, big }, .{ 16384, 0, 1 }, .{ 32000, 0, 1 }, .{ 32768, 0, -big } });
    try near(0.5, y);
}

test "NaN: a non-finite V at a masked key a block loads" {
    // rows at positions 1 and 2 share a block: the key at 2 is loaded for the row at 1 and masked there, P = 0, 0 x inf
    var y: [2]f32 = undefined;
    try (Mini{ .k = &.{ 0, 0, 0 }, .v = &.{ 1, 2, inf }, .rows = 2 }).run(&y, false);
    try t.expect(std.math.isNan(y[0]));
    try t.expect(std.math.isInf(y[1]));
}

test "NaN: a masked key no block loads is harmless" {
    // window 2, one row at 4 keeps keys 3 and 4: its block loads from key 3, so the infinite value at 1 is never read
    var y: [1]f32 = undefined;
    try (Mini{ .k = &.{ 0, 0, 0, 0, 0 }, .v = &.{ 1, inf, 1, 1, 1 }, .window = 2 }).run(&y, false);
    try near(1, y[0]);
}

test "NaN: a masked key's K is replaced by -inf" {
    var y: [2]f32 = undefined;
    try (Mini{ .k = &.{ 0, 0, nan }, .v = &.{ 1, 2, 3 }, .rows = 2 }).run(&y, false);
    try t.expect(std.math.isFinite(y[0]));
    try t.expect(std.math.isNan(y[1])); // the same key is live for the row at 2
    try (Mini{ .k = &.{ 0, 0, inf }, .v = &.{ 1, 2, 3 }, .rows = 2 }).run(&y, false);
    try t.expect(std.math.isFinite(y[0]));
}

test "NaN: below the window, a key another row's block loads" {
    // window 2, rows at 3 and 4 in one block: the row at 4 sees keys 3 and 4, the block loads from key 2
    var y: [2]f32 = undefined;
    try (Mini{ .k = &.{ 0, 0, 0, 0, 0 }, .v = &.{ 1, 1, inf, 1, 1 }, .rows = 2, .window = 2 }).run(&y, false);
    try t.expect(std.math.isInf(y[0])); // key 2 is live for the row at 3
    try t.expect(std.math.isNan(y[1])); // and masked for the row at 4, whose block loads it
}

test "NaN: live keys and the sink" {
    var y: [1]f32 = undefined;
    try (Mini{ .k = &.{ 0, 0 }, .v = &.{ inf, 2 } }).run(&y, false);
    try t.expect(std.math.isInf(y[0]));
    try (Mini{ .k = &.{ nan, 0 }, .v = &.{ 1, 2 } }).run(&y, false);
    try t.expect(std.math.isNan(y[0]));
    try (Mini{ .k = &.{ 0, 0 }, .v = &.{ 1, 2 }, .sink = nan }).run(&y, false);
    try t.expect(std.math.isNan(y[0]));
    try (Mini{ .k = &.{ 0, 0 }, .v = &.{ 1, 2 }, .sink = inf }).run(&y, false);
    try t.expect(std.math.isNan(y[0]));
    try (Mini{ .k = &.{ 0, 0 }, .v = &.{ 1, 2 }, .sink = -inf }).run(&y, false);
    try near(1.5, y[0]);
}

test "NaN: the gate" {
    var y: [1]f32 = undefined;
    try (Mini{ .k = &.{0}, .v = &.{1}, .gate = inf }).run(&y, false);
    try near(1, y[0]);
    try (Mini{ .k = &.{0}, .v = &.{1}, .gate = -inf }).run(&y, false);
    try t.expectEqual(@as(f32, 0), y[0]);
    try (Mini{ .k = &.{0}, .v = &.{1}, .gate = nan }).run(&y, false);
    try t.expect(std.math.isNan(y[0]));
}

test "sub-chunks: fold in order" {
    const y = try seam(0, 3073, &.{ .{ 0, 0, big }, .{ 1024, 0, 1 }, .{ 2048, 0, 1 }, .{ 3072, 0, -big } });
    try t.expectEqual(@as(f32, 0), @abs(y));
}

test "sliding chunks: merge in order" {
    const y = try seam(1024, 769, &.{ .{ 0, 0, big }, .{ 256, 0, 1 }, .{ 512, 0, 1 }, .{ 768, 0, -big } });
    try t.expectEqual(@as(f32, 0), @abs(y));
}

test "global chunks: merge in order" {
    const y = try seam(0, 49153, &.{ .{ 0, 0, big }, .{ 16384, 0, 1 }, .{ 32768, 0, 1 }, .{ 49152, 0, -big } });
    try t.expectEqual(@as(f32, 0), @abs(y));
}

test "table: taps in order" {
    // keys 0 and 1 at distance 1 and 0; the key at distance 0 gets bias 0 (absorbed), so both weigh 1
    var y: [1]f32 = undefined;
    try (Mini{ .k = &.{ 0, 0 }, .v = &.{ 1, 2 }, .q = &.{0}, .rels = &.{ big, 1, -big }, .proj = &.{ 1, 0, 1, 0, 1, 0 }, .rext = 2 }).run(&y, false);
    try near(1.5, y[0]);
}

test "score: one chain across k16 blocks" {
    var k: [66]f32 = @splat(0);
    k[0] = big;
    k[16] = 1;
    k[32] = -big;
    var v: [66]f32 = @splat(1);
    @memset(v[33..], 2);
    var y: [33]f32 = undefined;
    try (Mini{ .k = &k, .v = &v, .hd = 33 }).run(&y, false);
    try near(1.5, y[0]); // the first key's score is 0 as the second's: equal weights
}

test "shapes: over 8192 keys is accepted, a refusal stays a refusal" {
    const kv = try keys(t.allocator, 20000, &.{ .{ 0, 0, 1 }, .{ 19999, 0, 3 } });
    defer t.allocator.free(kv.k);
    defer t.allocator.free(kv.v);
    var y: [1]f32 = undefined;
    try (Mini{ .k = kv.k, .v = kv.v }).run(&y, false);
    try near(2, y[0]);
    var z: [3]f32 = undefined;
    try t.expectError(error.AttnShape, (Mini{ .k = &.{ 0, 0 }, .v = &.{ 1, 2 }, .heads = 3, .kv_heads = 2 }).run(&z, false));
}
