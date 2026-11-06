//! Dory's CUDA RoPE table on core's rope_freq: transformers' words at both thetas, beside the form it replaced.
const std = @import("std");
const rope_freq = @import("core").rope_freq;
const config = @import("config.zig");

const testing = std.testing;

/// The replaced table (cuda_model.zig before #1852), kept as a reference: f32 theta, Zig's f32 pow, fp32 reciprocal.
fn replaced(theta: f64, head_dim: usize, out: []f32) void {
    const d: f32 = @floatFromInt(head_dim);
    for (out, 0..) |*v, j| v.* = 1 / std.math.pow(f32, @floatCast(theta), @as(f32, @floatFromInt(2 * j)) / d);
}

const Diff = struct { differ: usize, ulps: u32 };

/// Prints both tables' words and values side by side, then the largest absolute and relative difference.
fn sideBySide(label: []const u8, old: []const f32, new: []const f32) Diff {
    var r: Diff = .{ .differ = 0, .ulps = 0 };
    var abs: f64 = 0;
    var rel: f64 = 0;
    std.debug.print("{s}\n   i  old                         new                         ulps\n", .{label});
    for (old, new, 0..) |o, n, i| {
        const a: u32 = @bitCast(o);
        const b: u32 = @bitCast(n);
        const u = if (a > b) a - b else b - a;
        const d = @abs(@as(f64, n) - o);
        if (u != 0) r.differ += 1;
        r.ulps = @max(r.ulps, u);
        abs = @max(abs, d);
        rel = @max(rel, d / @abs(@as(f64, o)));
        const sign = if (b > a) "+" else if (b < a) "-" else " ";
        std.debug.print("{d:>4}  0x{x:0>8} {e:<16}  0x{x:0>8} {e:<16}  {s}{d}\n", .{ i, a, o, b, n, sign, u });
    }
    std.debug.print("{s}: {d} of {d} words differ, largest {d} ulp, abs {e:.3}, rel {e:.3}\n", .{
        label, r.differ, old.len, r.ulps, abs, rel,
    });
    return r;
}

fn expectWords(want: []const u32, got: []const f32) !void {
    for (want, got) |w, g| try testing.expectEqual(w, @as(u32, @bitCast(g)));
}

// transformers' form in exact arithmetic, rn32(1 / rn32(theta^rn32(2i / 256))): it gives rope_freq.zig's torch words
const sliding_words = [128]u32{
    0x3f800000, 0x3f6e39f8, 0x3f5dafd7, 0x3f4e4bad, 0x3f3ff911, 0x3f32a506, 0x3f263de0, 0x3f1ab32b,
    0x3f0ff59a, 0x3f05f6ef, 0x3ef953cf, 0x3ee8045f, 0x3ed7e89b, 0x3ec8eb24, 0x3ebaf81b, 0x3eadfcff,
    0x3ea1e89b, 0x3e96aaea, 0x3e8c3504, 0x3e827909, 0x3e72d423, 0x3e61f835, 0x3e5247ed, 0x3e43ae7c,
    0x3e361887, 0x3e297409, 0x3e1db040, 0x3e12bd91, 0x3e088d77, 0x3dfe24e0, 0x3dec7fd6, 0x3ddc1466,
    0x3dcccccd, 0x3dbe94c6, 0x3db15978, 0x3da50956, 0x3d99940d, 0x3d8eea6b, 0x3d84fe4d, 0x3d778513,
    0x3d6655c2, 0x3d5657e4, 0x3d47763f, 0x3d399d19, 0x3d2cba15, 0x3d20bc1d, 0x3d159348, 0x3d0b30cc,
    0x3d0186e3, 0x3cf11177, 0x3ce054d2, 0x3cd0c1a8, 0x3cc2434f, 0x3cb4c691, 0x3ca8398b, 0x3c9c8b97,
    0x3c91ad39, 0x3c879008, 0x3c7c4d33, 0x3c6ac8e7, 0x3c5a7bf2, 0x3c4b50b3, 0x3c3d3311, 0x3c301052,
    0x3c23d70a, 0x3c187705, 0x3c0de12d, 0x3c040779, 0x3bf5b9b0, 0x3be4aa46, 0x3bd4ca15, 0x3bc6040f,
    0x3bb8449c, 0x3bab7983, 0x3b9f91cc, 0x3b947dae, 0x3b8a2e77, 0x3b80967d, 0x3b6f520e, 0x3b5eb47a,
    0x3b4f3e38, 0x3b40dac5, 0x3b33770f, 0x3b270153, 0x3b1b690d, 0x3b109edb, 0x3b06946f, 0x3afa78f0,
    0x3ae91528, 0x3ad8e673, 0x3ac9d75c, 0x3abbd3ed, 0x3aaec98e, 0x3aa2a6f7, 0x3a975c0e, 0x3a8cd9db,
    0x3a83126f, 0x3a73f1a3, 0x3a6301e2, 0x3a533f28, 0x3a44948c, 0x3a36ee9e, 0x3a2a3b44, 0x3a1e69a5,
    0x3a136a16, 0x3a092e02, 0x39ff4fac, 0x39ed95e2, 0x39dd1725, 0x39cdbd96, 0x39bf74d7, 0x39b229fa,
    0x39a5cb60, 0x399a489e, 0x398f9272, 0x39859aa9, 0x3978a815, 0x39676492, 0x395753e4, 0x394860c1,
    0x393a7753, 0x392d8529, 0x39217916, 0x39164324, 0x390bd472, 0x39021f2b, 0x38f22ce2, 0x38e15c91,
};

const full_words = [128]u32{
    0x3f800000, 0x3f61b5fc, 0x3f470165, 0x3f2f75b1, 0x3f1ab32b, 0x3f086571, 0x3ef0843c, 0x3ed40f33,
    0x3ebaf81b, 0x3ea4d8f8, 0x3e9157e1, 0x3e80258f, 0x3e61f835, 0x3e473bca, 0x3e2fa92d, 0x3e1ae090,
    0x3e088d77, 0x3df0cace, 0x3dd44d6c, 0x3dbb2ef7, 0x3da50956, 0x3d918287, 0x3d804b29, 0x3d623a83,
    0x3d47763f, 0x3d2fdcb8, 0x3d1b0e01, 0x3d08b588, 0x3cf11177, 0x3cd48bb6, 0x3cbb65e3, 0x3ca539c4,
    0x3c91ad39, 0x3c8070ce, 0x3c627ce5, 0x3c47b0c6, 0x3c301052, 0x3c1b3b80, 0x3c08dda5, 0x3bf15831,
    0x3bd4ca15, 0x3bbb9cdf, 0x3ba56a3f, 0x3b91d7f7, 0x3b80967d, 0x3b62bf59, 0x3b47eb5e, 0x3b3043fb,
    0x3b1b690d, 0x3b0905ce, 0x3af19f02, 0x3ad50885, 0x3abbd3ed, 0x3aa59ac8, 0x3a9202c3, 0x3a80bc38,
    0x3a6301e2, 0x3a482606, 0x3a3077b3, 0x3a1b96a6, 0x3a092e02, 0x39f1e5e9, 0x39d54706, 0x39bc0b09,
    0x39a5cb60, 0x39922d9a, 0x3980e1fe, 0x3963447d, 0x394860c1, 0x3930ab7a, 0x391bc44e, 0x39095642,
    0x38f22ce2, 0x38d5859b, 0x38bc4236, 0x38a5fc05, 0x3892587e, 0x388107cf, 0x3863872c, 0x38489b8d,
    0x3830df52, 0x381bf202, 0x38097e8f, 0x37f273f1, 0x37d5c441, 0x37bc7973, 0x37a62cb9, 0x3792836f,
    0x37812dab, 0x3763c9f0, 0x3748d669, 0x37311337, 0x371c1fc4, 0x3709a6e7, 0x36f2bb16, 0x36d602fa,
    0x36bcb0c1, 0x36a65d7b, 0x3692ae6d, 0x36815393, 0x36640cc6, 0x36491157, 0x3631472c, 0x361c4d93,
    0x3609cf4a, 0x35f30250, 0x35d641c6, 0x35bce81e, 0x35a68e4c, 0x3592d977, 0x35817985, 0x35644fb1,
    0x35494c57, 0x35317b31, 0x351c7b70, 0x3509f7bb, 0x34f3499d, 0x34d680a4, 0x34bd1f8c, 0x34a6bf2b,
    0x3493048e, 0x34819f83, 0x346492ae, 0x34498767, 0x3431af44, 0x341ca95a, 0x340a2036, 0x33f390ff,
};

test "Dory RoPE: the pinned config's tables are transformers' words, printed beside the fp32-pow form they replace" {
    const text = try config.pinnedText(testing.allocator);
    defer testing.allocator.free(text);
    var why: config.Why = .{};
    const c = try config.parse(testing.allocator, text, &why);
    try testing.expectEqual(@as(usize, 256), c.head_dim);
    const Case = struct { label: []const u8, theta: f64, want: *const [128]u32, diff: Diff };
    const cases = [_]Case{
        .{
            .label = "Dory sliding (theta 1e4, head dim 256)",
            .theta = c.theta_sliding,
            .want = &sliding_words,
            .diff = .{ .differ = 63, .ulps = 4 },
        },
        .{
            .label = "Dory full (theta 1e7, head dim 256)",
            .theta = c.theta_full,
            .want = &full_words,
            .diff = .{ .differ = 72, .ulps = 5 },
        },
    };
    for (cases) |case| {
        var new: [128]f32 = undefined;
        rope_freq.inverseFrequencies(case.theta, c.head_dim, &new);
        try expectWords(case.want, &new);
        var old: [128]f32 = undefined;
        replaced(case.theta, c.head_dim, &old);
        const got = sideBySide(case.label, &old, &new);
        try testing.expectEqual(case.diff, got);
    }
}
