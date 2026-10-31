//! tf-idol-birefnet's gates against the float64 ruler, each within twice torch's own distance from it.
const std = @import("std");
const ref = @import("reference.zig");

pub const Match = ref.Match;

/// Ours and torch's against the ruler in rows of `width`: worst row and 1 - cosine within twice torch's.
pub const Verdict = struct { ours: Match, torch: Match, pass: bool };

pub fn gate(ours: []const f32, torch: []const f32, ruler: []const f32, width: usize) Verdict {
    const o = match(ours, ruler, width);
    const t = match(torch, ruler, width);
    return .{ .ours = o, .torch = t, .pass = ref.within(o, t) };
}

/// reference.match, 1 - cos as |a/|a| - b/|b||^2 / 2: a dot product cannot resolve 1e-12 and turns negative.
pub fn match(ours: []const f32, theirs: []const f32, width: usize) Match {
    var m = ref.match(ours, theirs, width);
    var aa: f64 = 0;
    var bb: f64 = 0;
    for (ours, theirs) |x, y| {
        aa += @as(f64, x) * x;
        bb += @as(f64, y) * y;
    }
    const na = @sqrt(@max(aa, 1e-300));
    const nb = @sqrt(@max(bb, 1e-300));
    var d: f64 = 0;
    for (ours, theirs) |x, y| {
        const e = @as(f64, x) / na - @as(f64, y) / nb;
        d += e * e;
    }
    m.cosine = 1 - d / 2;
    return m;
}

/// Pixels (of `channels` bytes) where two 8-bit pictures differ, and the largest difference.
pub const Diff = struct { pixels: usize, max: u8 };

pub fn diff(a: []const u8, b: []const u8, channels: usize) Diff {
    std.debug.assert(a.len == b.len and a.len % channels == 0);
    var d: Diff = .{ .pixels = 0, .max = 0 };
    var i: usize = 0;
    while (i < a.len) : (i += channels) {
        var most: u8 = 0;
        for (a[i..][0..channels], b[i..][0..channels]) |x, y| most = @max(most, if (x > y) x - y else y - x);
        if (most > 0) d.pixels += 1;
        d.max = @max(d.max, most);
    }
    return d;
}

/// 8-bit against the ruler: differing pixels within twice torch's, largest difference within torch's (or one).
pub fn withinDiff(ours: Diff, torch: Diff) bool {
    return ours.pixels <= 2 * torch.pixels and ours.max <= @max(torch.max, 1);
}

/// The picture: pixels and largest difference within twice torch's (one alpha level moves a pixel up to two).
pub fn withinPicture(ours: Diff, torch: Diff) bool {
    return ours.pixels <= 2 * torch.pixels and ours.max <= 2 * @max(torch.max, 1);
}

test "a planted 2% error in one logits row fails the gate" {
    const a = std.testing.allocator;
    const n = 64 * 1024;
    const ruler = try a.alloc(f32, n);
    defer a.free(ruler);
    const torch = try a.alloc(f32, n);
    defer a.free(torch);
    const ours = try a.alloc(f32, n);
    defer a.free(ours);
    var rng = std.Random.DefaultPrng.init(3);
    for (ruler, torch, ours) |*r, *t, *o| {
        r.* = 8 * rng.random().floatNorm(f32);
        t.* = r.* * (1 + 1e-5 * rng.random().floatNorm(f32));
        o.* = r.* * (1 + 1e-6 * rng.random().floatNorm(f32));
    }
    try std.testing.expect(gate(ours, torch, ruler, 1024).pass);
    for (ours[17 * 1024 ..][0..1024]) |*o| o.* *= 1.02;
    try std.testing.expect(!gate(ours, torch, ruler, 1024).pass);
}

test "a planted mismatch in the alpha or the picture fails its gate" {
    const base = [6]u8{ 0, 10, 200, 254, 128, 3 };
    var ruler: [24]u8 = undefined;
    for (&ruler, 0..) |*r, i| r.* = base[i % 6];
    var torch = ruler;
    torch[1] += 1;
    torch[7] += 1;
    var ours = ruler;
    ours[2] += 1;
    try std.testing.expect(withinDiff(diff(&ours, &ruler, 1), diff(&torch, &ruler, 1)));
    ours[5] += 2;
    try std.testing.expect(!withinDiff(diff(&ours, &ruler, 1), diff(&torch, &ruler, 1)));
    ours[5] -= 2;
    for ([_]usize{ 8, 9, 10, 11 }) |i| ours[i] +%= 1;
    try std.testing.expect(!withinDiff(diff(&ours, &ruler, 1), diff(&torch, &ruler, 1)));
    try std.testing.expect(withinPicture(.{ .pixels = 27, .max = 2 }, .{ .pixels = 26, .max = 1 }));
    try std.testing.expect(!withinPicture(.{ .pixels = 27, .max = 3 }, .{ .pixels = 26, .max = 1 }));
    try std.testing.expect(!withinPicture(.{ .pixels = 53, .max = 1 }, .{ .pixels = 26, .max = 1 }));
    // three channels a pixel: one pixel, however many of its channels differ
    try std.testing.expectEqual(@as(usize, 1), diff(&[_]u8{ 1, 2, 3 }, &[_]u8{ 2, 3, 4 }, 3).pixels);
}
