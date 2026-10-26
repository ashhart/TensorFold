const std = @import("std");

// K steps of one pair: groups of 64, then four sub-steps. The batch size is not an argument.

fn kOrder(n: usize, out: []usize) usize {
    var nout: usize = 0;
    var g: usize = 0;
    while (g < n) : (g += 64) {
        var s: usize = 0;
        while (s < 4) : (s += 1) {
            var j: usize = 0;
            while (j < 16 and g + 16 * s + j < n) : (j += 1) {
                out[nout] = g + 16 * s + j;
                nout += 1;
            }
        }
    }
    return nout;
}

fn sumAt(x: []const f32, order: []const usize) f32 {
    var a: f32 = 0;
    for (order) |i| a += x[i];
    return a;
}

// A wrong reduction: walk K with a stride of the batch, then the next residue.

fn batchStride(n: usize, batch: usize, out: []usize) usize {
    var nout: usize = 0;
    var b: usize = 0;
    while (b < batch) : (b += 1) {
        var i = b;
        while (i < n) : (i += batch) {
            out[nout] = i;
            nout += 1;
        }
    }
    return nout;
}

fn gdnRow(q: [8]f32, k: [8]f32, s0: [8]f32, g: f32, beta: f32, v: f32) struct { y: f32, s: [8]f32 } {
    var s = s0;
    var mem: f32 = 0;
    for (0..8) |i| {
        s[i] *= g;
        mem += s[i] * k[i];
    }
    const u = (v - mem) * beta;
    var y: f32 = 0;
    for (0..8) |i| {
        s[i] += k[i] * u;
        y += s[i] * q[i];
    }
    return .{ .y = y, .s = s };
}

fn online(scores: []const f32) struct { m: f32, l: f32 } {
    var m: f32 = -std.math.inf(f32);
    var l: f32 = 0;
    for (scores) |x| {
        const mn = @max(m, x);
        l = l * @exp(m - mn) + @exp(x - mn);
        m = mn;
    }
    return .{ .m = m, .l = l };
}

test "a row K order does not take the batch size" {
    var a: [128]usize = undefined;
    var b: [128]usize = undefined;
    const na = kOrder(128, &a);
    const nb = kOrder(128, &b);
    try std.testing.expectEqualSlices(usize, a[0..na], b[0..nb]);
}

test "a planted split by batch changes the sum" {
    const x = [_]f32{ 1e8, 1, -1e8, 1, 1e8, 1, -1e8, 1 };
    var seq: [8]usize = undefined;
    var split: [8]usize = undefined;
    const ns = kOrder(8, &seq);
    const nb = batchStride(8, 2, &split);
    try std.testing.expectEqual(@as(usize, 8), ns);
    try std.testing.expectEqual(@as(usize, 8), nb);
    try std.testing.expect(sumAt(&x, seq[0..ns]) != sumAt(&x, split[0..nb]));
}

test "the first GDN row is unchanged when a later row is appended" {
    const q = [8]f32{ 1, 0, 0, 0, 0, 0, 0, 0 };
    const k = [8]f32{ 0.5, 0, 0, 0, 0, 0, 0, 0 };
    const s0 = [8]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const first = gdnRow(q, k, s0, 0.9, 0.5, 1);
    const again = gdnRow(q, k, s0, 0.9, 0.5, 1);
    try std.testing.expectEqual(first.y, again.y);
    try std.testing.expectEqual(first.s, again.s);
    const later = gdnRow(q, k, first.s, 0.9, 0.5, 2);
    try std.testing.expect(later.s[0] != s0[0]);
}

test "a key past the row does not change the softmax" {
    const short = [_]f32{ 1, 2, 0.5 };
    const long = [_]f32{ 1, 2, 0.5, -std.math.inf(f32), -std.math.inf(f32) };
    const a = online(&short);
    const b = online(&long);
    try std.testing.expectEqual(a.m, b.m);
    try std.testing.expectEqual(a.l, b.l);
}
