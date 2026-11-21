//! Kolibri-1's forward without a GPU: the ring's size, the RoPE table's words, a page's bytes.
const std = @import("std");
const fwd = @import("forward.zig");
const Model = fwd.Model;
const ringSize = fwd.ringSize;
const prompt_rows = fwd.prompt_rows;

test "the ring holds a prompt chunk and a window, in whole key tiles" {
    try std.testing.expectEqual(@as(usize, 8768), ringSize(prompt_rows, 513));
    try std.testing.expect(ringSize(prompt_rows, 513) - 513 >= prompt_rows);
}

test "the f64 angle table against core's words: 19 of 64 differ, first at 7" {
    const rope_freq = @import("core").rope_freq;
    // The site's f64 frequency rounded to fp32 (the word a move would store); it keeps the f64 value for the angle.
    const theta: f64 = 1e4; // config.zig's default rope_theta
    const dh = 128;
    const site_words = [64]u32{
        0x3f800000, 0x3f5dafd7, 0x3f3ff911, 0x3f263de0, 0x3f0ff59a, 0x3ef953cf, 0x3ed7e89b, 0x3ebaf81a,
        0x3ea1e89b, 0x3e8c3504, 0x3e72d424, 0x3e5247ed, 0x3e361887, 0x3e1db040, 0x3e088d77, 0x3dec7fd5,
        0x3dcccccd, 0x3db15978, 0x3d99940e, 0x3d84fe4d, 0x3d6655c3, 0x3d47763f, 0x3d2cba15, 0x3d159348,
        0x3d0186e2, 0x3ce054d2, 0x3cc24350, 0x3ca8398b, 0x3c91ad39, 0x3c7c4d33, 0x3c5a7bf1, 0x3c3d3311,
        0x3c23d70a, 0x3c0de12d, 0x3bf5b9b0, 0x3bd4ca14, 0x3bb8449c, 0x3b9f91cc, 0x3b8a2e77, 0x3b6f520d,
        0x3b4f3e37, 0x3b33770f, 0x3b1b690d, 0x3b06946f, 0x3ae91528, 0x3ac9d75c, 0x3aaec98e, 0x3a975c0e,
        0x3a83126f, 0x3a6301e2, 0x3a44948c, 0x3a2a3b43, 0x3a136a16, 0x39ff4fad, 0x39dd1726, 0x39bf74d7,
        0x39a5cb5f, 0x398f9272, 0x3978a814, 0x395753e5, 0x393a7753, 0x39217917, 0x390bd472, 0x38f22ce3,
    };
    const core_words = [64]u32{
        0x3f800000, 0x3f5dafd7, 0x3f3ff911, 0x3f263de0, 0x3f0ff59a, 0x3ef953cf, 0x3ed7e89b, 0x3ebaf81b,
        0x3ea1e89b, 0x3e8c3504, 0x3e72d423, 0x3e5247ed, 0x3e361887, 0x3e1db040, 0x3e088d77, 0x3dec7fd6,
        0x3dcccccd, 0x3db15978, 0x3d99940d, 0x3d84fe4d, 0x3d6655c2, 0x3d47763f, 0x3d2cba15, 0x3d159348,
        0x3d0186e3, 0x3ce054d2, 0x3cc2434f, 0x3ca8398b, 0x3c91ad39, 0x3c7c4d33, 0x3c5a7bf2, 0x3c3d3311,
        0x3c23d70a, 0x3c0de12d, 0x3bf5b9b0, 0x3bd4ca15, 0x3bb8449c, 0x3b9f91cc, 0x3b8a2e77, 0x3b6f520e,
        0x3b4f3e38, 0x3b33770f, 0x3b1b690d, 0x3b06946f, 0x3ae91528, 0x3ac9d75c, 0x3aaec98e, 0x3a975c0e,
        0x3a83126f, 0x3a6301e2, 0x3a44948c, 0x3a2a3b44, 0x3a136a16, 0x39ff4fac, 0x39dd1725, 0x39bf74d7,
        0x39a5cb60, 0x398f9272, 0x3978a815, 0x395753e4, 0x393a7753, 0x39217916, 0x390bd472, 0x38f22ce2,
    };
    var site: [64]f32 = undefined;
    var core: [64]f32 = undefined;
    for (&site, 0..) |*f, j| {
        const frac = @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(dh));
        f.* = @floatCast(1.0 / std.math.pow(f64, theta, frac));
    }
    rope_freq.inverseFrequencies(theta, dh, &core);
    var differ: usize = 0;
    var first: ?usize = null;
    var max_ulp: u32 = 0;
    for (site, core, 0..) |s, c, j| {
        try std.testing.expectEqual(site_words[j], @as(u32, @bitCast(s)));
        try std.testing.expectEqual(core_words[j], @as(u32, @bitCast(c)));
        const a: u32 = @bitCast(s);
        const b: u32 = @bitCast(c);
        if (a != b) {
            differ += 1;
            if (first == null) first = j;
            max_ulp = @max(max_ulp, if (a > b) a - b else b - a);
        }
    }
    try std.testing.expectEqual(@as(usize, 19), differ);
    try std.testing.expectEqual(@as(?usize, 7), first);
    try std.testing.expectEqual(@as(u32, 1), max_ulp);
}

test "a page's bytes: the pool's layout and the kernels' stride agree" {
    const config = @import("config.zig");
    var why: config.Why = .{};
    var m: Model = undefined;
    m.c = try config.parse(std.testing.allocator, config.pinned, &why);
    m.context = 32768;
    m.ring = ringSize(prompt_rows, m.c.window);
    try std.testing.expectEqual(m.pageBytes(), m.layout().pageBytes());
    try std.testing.expectEqual(@as(u32, 10), m.layout().group().layers);
}
