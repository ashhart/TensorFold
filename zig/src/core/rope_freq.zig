//! transformers' default RoPE inverse frequencies in fp32, 1 / theta^(f32(2i) / dim), each step rounded as torch's.
const std = @import("std");

/// out[i] = 1 / theta^(2i / dim), i < out.len: an fp32 exponent, theta^e rounded once to fp32, an fp32 reciprocal.
pub fn inverseFrequencies(theta: f64, dim: usize, out: []f32) void {
    for (out, 0..) |*f, i| {
        const e: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(dim));
        const p: f32 = @floatCast(std.math.pow(f64, theta, e));
        f.* = 1.0 / p;
    }
}

/// transformers' "proportional" RoPE (Gemma 4's global layers): the first `rotated` pairs over `dim`, zeros after.
pub fn proportional(theta: f64, dim: usize, rotated: usize, out: []f32) void {
    const n = @min(rotated, out.len);
    inverseFrequencies(theta, dim, out[0..n]);
    @memset(out[n..], 0);
}

fn expectWords(want: []const u32, got: []const f32) !void {
    for (want, got) |w, g| try std.testing.expectEqual(w, @as(u32, @bitCast(g)));
}

// transformers 5.22.1's default rope init, torch 2.15.0 CPU: fp32 words, the same on its default, AVX2, AVX-512 paths
test "theta 1e7 over 64 rotated dims (Qwen3.5 and Flash Next's partial 0.25 of 256) equals transformers' words" {
    const torch = [32]u32{
        0x3f800000, 0x3f1ab32b, 0x3ebaf81b, 0x3e61f835, 0x3e088d77, 0x3da50956, 0x3d47763f, 0x3cf11177,
        0x3c91ad39, 0x3c301052, 0x3bd4ca15, 0x3b80967d, 0x3b1b690d, 0x3abbd3ed, 0x3a6301e2, 0x3a092e02,
        0x39a5cb60, 0x394860c1, 0x38f22ce2, 0x3892587e, 0x3830df52, 0x37d5c441, 0x37812dab, 0x371c1fc4,
        0x36bcb0c1, 0x36640cc6, 0x3609cf4a, 0x35a68e4c, 0x35494c57, 0x34f3499d, 0x3493048e, 0x3431af44,
    };
    var got: [32]f32 = undefined;
    inverseFrequencies(1e7, 64, &got);
    try expectWords(&torch, &got);
}

test "theta 5e6 over 64 rotated dims (Spark2.5's full layers) equals transformers' words" {
    const torch = [32]u32{
        0x3f800000, 0x3f1e165d, 0x3ec33f3a, 0x3e712429, 0x3e14e963, 0x3db7ea1a, 0x3d63251b, 0x3d0c44bf,
        0x3cad3d5e, 0x3c55f605, 0x3c042088, 0x3ba32f3e, 0x3b498ad3, 0x3af8ea93, 0x3a99b686, 0x3a3dd829,
        0x39ea77ff, 0x3990ca8b, 0x3932d34f, 0x38dcdc14, 0x38886320, 0x38287230, 0x37d00a62, 0x37807896,
        0x371eab4b, 0x36c3f72a, 0x36720755, 0x361575ab, 0x35b8975d, 0x3563fb18, 0x350cc8e3, 0x34ade091,
    };
    var got: [32]f32 = undefined;
    inverseFrequencies(5e6, 64, &got);
    try expectWords(&torch, &got);
}

test "theta 1e4 over 36 (a Qwen-VL tower's 72-wide head, spatial_dim 36) equals transformers' words" {
    const torch = [18]u32{
        0x3f800000, 0x3f1977cc, 0x3eb800d6, 0x3e5c9d35, 0x3e044133, 0x3d9e91b6, 0x3d3e1e95, 0x3ce3f280, 0x3c88a69b,
        0x3c23d70a, 0x3bc47060, 0x3b6b8631, 0x3b0d3169, 0x3aa94938, 0x3a4af7f3, 0x39f35a5c, 0x3991e2e1, 0x392ee9bf,
    };
    var got: [18]f32 = undefined;
    inverseFrequencies(1e4, 36, &got);
    try expectWords(&torch, &got);
}

test "Gemma 4's proportional global table (theta 1e6, 64 of 256 pairs over 512) equals transformers' scalar words" {
    // torch's default path; its AVX2 and AVX-512 pow give word 29 one ulp up (0x3e561912); 0x3e561911 is correct
    const torch = [64]u32{
        0x3f800000, 0x3f728cf8, 0x3f65ced3, 0x3f59bc10, 0x3f4e4bad, 0x3f437523, 0x3f39305c, 0x3f2f75b1,
        0x3f263de0, 0x3f1d8209, 0x3f153ba8, 0x3f0d6492, 0x3f05f6ef, 0x3efdda64, 0x3ef0843c, 0x3ee3e172,
        0x3ed7e89b, 0x3ecc90c7, 0x3ec1d182, 0x3eb7a2c7, 0x3eadfcff, 0x3ea4d8f8, 0x3e9c2fe1, 0x3e93fb45,
        0x3e8c3504, 0x3e84d752, 0x3e7bb964, 0x3e6e7fde, 0x3e61f835, 0x3e561911, 0x3e4ad998, 0x3e403165,
        0x3e361887, 0x3e2c8776, 0x3e23770f, 0x3e1ae090, 0x3e12bd91, 0x3e0b0801, 0x3e03ba20, 0x3df99cf8,
        0x3dec7fd6, 0x3de01313, 0x3dd44d6c, 0x3dc92617, 0x3dbe94c6, 0x3db49196, 0x3dab150f, 0x3da2181d,
        0x3d99940d, 0x3d918287, 0x3d89dd84, 0x3d829f52, 0x3d778513, 0x3d6a8417, 0x3d5e3201, 0x3d5285a1,
        0x3d47763f, 0x3d3cfb9e, 0x3d330dec, 0x3d29a5c2, 0x3d20bc1d, 0x3d184a56, 0x3d104a21, 0x3d08b588,
    };
    var got: [256]f32 = undefined;
    proportional(1e6, 512, 64, &got);
    try expectWords(&torch, got[0..64]);
    for (got[64..]) |v| try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(v)));
}

test "planted: the fp32-pow form misses transformers' words for theta 1e7 over 64 (15 of 32, up to 5 ulp)" {
    var want: [32]f32 = undefined;
    inverseFrequencies(1e7, 64, &want);
    var differ: usize = 0;
    var most: u32 = 0;
    for (want, 0..) |w, i| {
        const f32pow: f32 = 1.0 / std.math.pow(f32, 1e7, @as(f32, @floatFromInt(2 * i)) / 64.0);
        const a: u32 = @bitCast(w);
        const b: u32 = @bitCast(f32pow);
        if (a != b) differ += 1;
        most = @max(most, if (a > b) a - b else b - a);
    }
    try std.testing.expectEqual(@as(usize, 15), differ);
    try std.testing.expectEqual(@as(u32, 5), most);
}
