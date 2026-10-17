//! draft/dspark.zig's chain lanes against one-at-a-time sums (run in draft.zig's tests: the chain needs lanes).
const std = @import("std");
const dspark = @import("dspark.zig");
const Markov = dspark.Markov;
const Params = dspark.Params;
const bf16 = dspark.bf16;
const toBf16 = dspark.toBf16;
const markovZ = dspark.markovZ;
const chain = dspark.chain;
const pick = dspark.pick;

extern "c" fn exp(x: f64) f64;

/// The chain's sums one candidate / one row at a time (the code before the lanes): the exactness tests' reference.
const reference = struct {
    fn bias(m: Markov, prev: u32, id: u32) f32 {
        const a = m.w1[@as(usize, prev) * m.rank ..][0..m.rank];
        const b = m.w2[@as(usize, id) * m.rank ..][0..m.rank];
        var acc: f32 = 0.0;
        for (a, b) |x, y| acc += bf16(x) * bf16(y);
        return acc;
    }

    fn confidence(m: Markov, hid: []const f32, prev: u32) f32 {
        const cw = m.conf orelse return 0.5;
        var acc: f32 = 0.0;
        for (hid, cw[0..hid.len]) |h, w| acc += h * w;
        const me = m.w1[@as(usize, prev) * m.rank ..][0..m.rank];
        for (me, cw[hid.len..][0..m.rank]) |x, w| acc += bf16(x) * w;
        return @floatCast(1.0 / (1.0 + exp(-@as(f64, acc))));
    }
};

test "the chain's lanes: markovZ and the confidences bit for bit the one-at-a-time sums (the served shape and ragged ones)" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x4101_5eed);
    const rnd = prng.random();
    const vocab = 700;
    for ([_][3]usize{ .{ 256, 64, 4096 }, .{ 256, 64, 5120 }, .{ 13, 21, 37 }, .{ 1, 1, 1 } }) |shape| {
        const rank = shape[0];
        const c = shape[1];
        const D = shape[2];
        const w1 = try a.alloc(u16, vocab * rank);
        defer a.free(w1);
        const w2 = try a.alloc(u16, vocab * rank);
        defer a.free(w2);
        const cw = try a.alloc(f32, D + rank);
        defer a.free(cw);
        for (w1) |*x| x.* = toBf16(rnd.floatNorm(f32) * 0.3);
        for (w2) |*x| x.* = toBf16(rnd.floatNorm(f32) * 0.3);
        for (cw) |*x| x.* = rnd.floatNorm(f32) * 0.05;
        const m: Markov = .{ .w1 = w1, .w2 = w2, .rank = @intCast(rank), .conf = cw };
        const n = 5;
        const ids = try a.alloc(i32, n * c);
        defer a.free(ids);
        const base = try a.alloc(f32, n * c);
        defer a.free(base);
        const hid = try a.alloc(f32, n * D);
        defer a.free(hid);
        for (ids, 0..) |*x, i| x.* = if (i % 9 == 4) -1 else rnd.intRangeLessThan(i32, 0, vocab);
        for (base) |*x| x.* = rnd.floatNorm(f32) * 4.0;
        for (hid) |*x| x.* = bf16(toBf16(rnd.floatNorm(f32)));
        // markovZ against the reference bias, for several anchors
        const z = try a.alloc(f32, c);
        defer a.free(z);
        for (0..8) |_| {
            const prev = rnd.intRangeLessThan(u32, 0, vocab);
            for (0..n) |i| {
                markovZ(m, ids[i * c ..][0..c], base[i * c ..][0..c], prev, z);
                for (ids[i * c ..][0..c], base[i * c ..][0..c], z) |id, b, got| {
                    const want = if (id < 0) -std.math.inf(f32) else b + reference.bias(m, prev, @intCast(id));
                    try std.testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(got)));
                }
            }
            for (0..n) |i| try std.testing.expectEqual(@as(u32, @bitCast(reference.confidence(m, hid[i * D ..][0..D], prev))), @as(u32, @bitCast(m.confidence(hid[i * D ..][0..D], prev))));
        }
        // the whole chain, greedy and keyed, against the reference's rule
        for ([_]bool{ false, true }) |sampled| {
            const params: Params = if (sampled) .{ .seed = 77, .pos0 = 1000, .sampled = true, .temperature = 0.7, .top_p = 0.9, .top_k = 20 } else .{ .pos0 = 1000 };
            var drafts: [n]u32 = undefined;
            var conf: [n]f32 = undefined;
            try chain(a, m, ids, base, c, 3, params, hid, &drafts, &conf);
            var prev: u32 = 3;
            for (0..n) |i| {
                const row = ids[i * c ..][0..c];
                for (row, base[i * c ..][0..c], z) |id, b, *o| o.* = if (id < 0) -std.math.inf(f32) else b + reference.bias(m, prev, @intCast(id));
                const tok = (try pick(a, row, z, params.pos0 + i, params.sampling())) orelse prev;
                try std.testing.expectEqual(tok, drafts[i]);
                try std.testing.expectEqual(@as(u32, @bitCast(reference.confidence(m, hid[i * D ..][0..D], prev))), @as(u32, @bitCast(conf[i])));
                prev = tok;
            }
        }
    }
}
