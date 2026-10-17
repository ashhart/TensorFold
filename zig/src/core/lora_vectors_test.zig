//! MLX quantizer vectors made independently (a float32 reference of the published algorithm) on synthetic groups.
const std = @import("std");
const host4 = @import("affine4_host.zig");
const lm = @import("lora_merge.zig");

// Nine groups: division vs reciprocal (two), unrounded scale and bias, a tie, both edges, flat, tiny.
const inputs =
    \\bb2b 3d9b 3c3a 3c48 bd2d bc98 3ca9 bd04 3cc9 3d88 3d0a 3c4e bd13 3c44 3c35 bd06 bcdc 3dc3 bb8e 3b2b 3d4e 3c1c bb8d bc63 bb19 bcc9 b9ee bcdb 3c2b bccc bdac 3d04 bce9 3c69 3c29 3d9e 3d4f 3a64 3da0 bd41 3d2a bd48 3c38 3cf4 3d0e 3d12 bd5f 3c9e bd0f bd06 3d59 bdff bda2 3ce7 3d43 bd29 3d88 bbec bceb 3d41 bd3b 3d46 bc3f bcb5
    \\3ce7 3da2 3d0e 3ac1 3d21 bdce 3d03 bbc0 bb33 3d25 3ccb bd7d 3c03 bd65 b9c4 3d83 3d33 bd48 ba53 bcf7 bc11 bca7 bc70 bcc1 3d85 3c5e 3dac 3d8d bd56 bdee 3d9c 3d96 3cf6 bc1e 3dc3 bd84 bd49 3d05 bc90 bd15 3d24 baea 3cea bdbe 3c36 bda8 3d51 3d9a bd99 3db1 bd47 bd7e 3dee bd90 bd61 3c85 3cd3 bc80 3d2f 3acc 3bb8 3d7f 3d4d 3de2
    \\bd4b bd8b bd0e 3d0b 3cad 3d3e 3d4d bd5c 3d0a bd9b 3cf3 3c29 bb88 bbad bb4e 3c6d bc46 bbed bd42 3c9a 3d63 b9fd 3c8b 3d79 bd56 bcba 3cef bd44 3c8a bd9f bdb0 bc9e 3c6e bd9c 3d19 bd99 bc60 3d58 bd2d 3cfc bcc5 3d95 bcb6 3d20 bcdf bd51 be0b 3c20 3d22 3d65 3ba2 3c94 3d90 bcb4 3cfe bbd8 bdc7 bda7 3ca0 bd4c 3d3a bd30 3c86 3b2f
    \\c150 4188 c150 c0c0 3f80 4100 4170 c110 c000 40a0 4140 c140 c0a0 4000 4110 4180 c100 bf80 40c0 4150 c130 c080 4040 4120 4188 c0e0 0000 40e0 4160 c120 c040 4080 4130 c150 c0c0 3f80 4100 4170 c110 c000 40a0 4140 c140 c0a0 4000 4110 4180 c100 bf80 40c0 4150 c130 c080 4040 4120 4188 c0e0 0000 40e0 4160 c120 c040 4080 4130
    \\c170 c160 c150 c140 c130 c120 c110 c100 c0e0 c0c0 c0a0 c080 c040 c000 bf80 0000 c170 c160 c150 c140 c130 c120 c110 c100 c0e0 c0c0 c0a0 c080 c040 c000 bf80 0000 c170 c160 c150 c140 c130 c120 c110 c100 c0e0 c0c0 c0a0 c080 c040 c000 bf80 0000 c170 c160 c150 c140 c130 c120 c110 c100 c0e0 c0c0 c0a0 c080 c040 c000 bf80 0000
    \\0000 3f80 4000 4040 4080 40a0 40c0 40e0 4100 4110 4120 4130 4140 4150 4160 4170 0000 3f80 4000 4040 4080 40a0 40c0 40e0 4100 4110 4120 4130 4140 4150 4160 4170 0000 3f80 4000 4040 4080 40a0 40c0 40e0 4100 4110 4120 4130 4140 4150 4160 4170 0000 3f80 4000 4040 4080 40a0 40c0 40e0 4100 4110 4120 4130 4140 4150 4160 4170
    \\be80 3d80 3ec0 3f30 3f80 be00 3e40 3f00 3f50 3f90 0000 3ea0 3f20 3f70 be40 3e00 3ee0 3f40 3f88 bd80 3e80 3f10 3f60 be80 3d80 3ec0 3f30 3f80 be00 3e40 3f00 3f50 3f90 0000 3ea0 3f20 3f70 be40 3e00 3ee0 3f40 3f88 bd80 3e80 3f10 3f60 be80 3d80 3ec0 3f30 3f80 be00 3e40 3f00 3f50 3f90 0000 3ea0 3f20 3f70 be40 3e00 3ee0 3f40
    \\bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00 bf00
    \\b089 3109 0000 3089 3109 0000 3089 3109 0000 3089 3109 0000 3089 3109 0000 3089 3109 0000 3089 3109 0000 3089 3109 0000 3089 3109 0000 3089 3109 0000 3089 3109 0000 3089 3109 0000 3089 3109 0000 3089 3109 0000 3089 3109 0000 3089 3109 0000 3089 3109 0000 3089 3109 0000 3089 3109 0000 3089 3109 0000 3089 3109 0000 3089
;
const expected =
    \\7a86aae9 7aa6abeb 89ad99f7 b37a7979 6f9dfaa7 a5ccba5c 6cb30d76 78c6c78e 3c63 bdff
    \\86f58636 48c7c658 a999a8b5 33fc3274 b96bc196 34e7e685 7cd0cb2d 05488596 bc6e 3dee
    \\6edcd756 baaabc4c ebaeb799 944b6c86 c7e94d4b b068d8f8 ac8fbaed ab7d6b43 3c5e be0b
    \\e158cf0f 148cf36a 47bf26ad 7be259d0 ae158cf3 d148cf36 047bf26a 37be259d bff2 4188
    \\76543210 fedcba98 76543210 fedcba98 76543210 fedcba98 76543210 fedcba98 3f80 c170
    \\89abcdef 01234567 89abcdef 01234567 89abcdef 01234567 89abcdef 01234567 bf80 4170
    \\7ad158bf be259c03 f369d147 37ad158b 7be259c0 bf369d14 037ad158 47be259c bdc0 3f90
    \\00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 b3d7 bf00
    \\00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 b3d7 0000
;

fn hexWords(comptime T: type, line: []const u8, out: []T) !void {
    var it = std.mem.tokenizeScalar(u8, line, ' ');
    var i: usize = 0;
    while (it.next()) |tok| : (i += 1) out[i] = try std.fmt.parseInt(T, tok, 16);
    try std.testing.expectEqual(out.len, i);
}

/// The same algorithm with the step multiplied by 1/15 instead of divided: the variant the vectors must tell apart.
fn reciprocal(values: *const [64]f32, words: *[8]u32, scale: *u16, bias: *u16) void {
    const lo = std.mem.min(f32, values);
    const hi = std.mem.max(f32, values);
    const step = @max((hi - lo) * (1.0 / 15.0), 1e-7);
    const low_edge = @abs(lo) > @abs(hi);
    const signed = if (low_edge) step else -step;
    const edge = if (low_edge) lo else hi;
    const q0 = @round(edge / signed);
    const s: f32 = if (q0 == 0) signed else edge / q0;
    const b: f32 = if (q0 == 0) 0 else edge;
    scale.* = host4.bf16of(s);
    bias.* = host4.bf16of(b);
    @memset(words, 0);
    for (values, 0..) |x, i| words[i / 8] |= @as(u32, @intFromFloat(std.math.clamp(@round((x - b) / s), 0, 15))) << @intCast(4 * (i % 8));
}

test "the quantizer reproduces the independently produced words, scales and biases, and the reciprocal variant does not" {
    var in_lines = std.mem.tokenizeScalar(u8, inputs, '\n');
    var ex_lines = std.mem.tokenizeScalar(u8, expected, '\n');
    var group: usize = 0;
    var reciprocal_misses: usize = 0;
    while (in_lines.next()) |line| : (group += 1) {
        var bits: [64]u16 = undefined;
        try hexWords(u16, line, &bits);
        var values: [64]f32 = undefined;
        for (&values, bits) |*v, b| v.* = host4.f32of(b);
        var want: [10]u32 = undefined;
        try hexWords(u32, ex_lines.next().?, &want);
        var words: [8]u32 = undefined;
        var scale: [1]u16 = undefined;
        var bias: [1]u16 = undefined;
        try std.testing.expect(host4.quantizeMlx(64, &values, &words, &scale, &bias) == null);
        try std.testing.expectEqualSlices(u32, want[0..8], &words);
        try std.testing.expectEqual(@as(u16, @intCast(want[8])), scale[0]);
        try std.testing.expectEqual(@as(u16, @intCast(want[9])), bias[0]);
        var rw: [8]u32 = undefined;
        var rs: u16 = undefined;
        var rb: u16 = undefined;
        reciprocal(&values, &rw, &rs, &rb);
        if (!std.mem.eql(u32, &rw, &words) or rs != scale[0] or rb != bias[0]) reciprocal_misses += 1;
    }
    try std.testing.expectEqual(@as(usize, 9), group);
    try std.testing.expect(reciprocal_misses >= 2);
}

// A merge: 2 x 128 dense, a rank-2 adapter, alpha / r = 0.1; the reference sums in float64, rounds to f32, quantizes.
const merge_dense =
    \\bcd1 3d53 bc91 3bca bc15 bd99 3be9 bcd7 bc4c bcd7 3e14 bcf4 3d86 3c90 bd87 b9d2 3c4b 3da1 bda8 bda3 bbfd bd61 3d8c 3d0d bca1 3cfd 3d1a 3cf7 3dcf bd6f bd88 3d67 bc39 3cd6 bdd3 3be3 3ccb bcf8 bd24 3c02 3d12 bd18 3c65 3c99 3e19 bc52 bd0c 3d5a ba86 bd01 3ce1 bda3 3cd8 3d38 3d4c bc31 3d88 bbb2 bd52 3b0f bd47 3c66 3ce9 3d98 bd95 3ce5 bccf 3cbf bd30 bd2f 3dce 3c0b bd39 bdaf bbdf 3d95 3ce6 3d6d bdb0 bb41 3d2d bbd6 bc84 bd12 bc3a be02 3c8c 3d62 3b9d bac4 3d2f baa3 3de8 3d30 bda9 bb0e 3cb8 bce2 bd28 3d67 bd7d bdb7 3d4f bd4c 3ddd 3cbe bdac 3d99 bd40 3dd0 3d15 3ca5 3d1d bd2c 3c8e be12 bdb3 be17 3b74 3c67 3c9f bde8 bd0e 3d2f 3c10 3d08 bd70 3d64
    \\bd44 bd4e 3d7e 3dc3 bc99 3d01 3d8c bd19 3d01 3d36 3d47 3d89 3b5b bd75 bb4e bc91 3c88 bcec 3ca9 bc9b 3d80 bbda bd8a bded 3cc6 3d12 3d83 3d89 bca0 bd2d 3c51 3c49 3dac 3c95 bbee bd8c 3bca bd74 bac3 bc0b 3d8e bd92 3d5c bcc0 3d57 bc93 3d00 bd29 bddc ba57 bca2 bd20 bdda be0f 3cc6 3d5b 3d87 3d09 bae2 3d48 bc8b 3d30 3cb5 3dd4 bd8b bd50 3caa bd31 bcc0 3c62 bce5 bcc0 3c9e 3c9c bd5b bd4b 3d87 bd0e 3d15 3c33 3d03 3b61 bd35 3c92 bc08 3d78 3de2 3cc9 baf7 bdb6 bccb bdb0 3c63 bddc 3d2b 3b81 bdac bcc2 39eb 3d89 bd27 bda3 3bdd bdb5 bdad bbad 3ccd bc54 3d34 bb9c 3d42 3d54 bd82 bde0 3ccf bd52 bcf8 3d01 3d57 3d70 3d9f 3ccb 3c4c bcda bd03 bcb8 bc48 3bd2
;
const merge_a =
    \\bf2f 3ec0 3e87 be36 be9c 3f2e be48 3f3e 3e08 3d61 bf3a 3f08 3e2d bf7c 3f14 3f43 bec1 3f15 bd80 bedd 3f28 bd0a bf19 be85 3e0c be7b bd89 bf11 be6e bdb3 3f14 bf33 3efb bed1 3f1a bdd4 3b3f bd9e be57 3eeb 3eb5 3d9e bda5 bddf bf02 3f69 3f95 3f30 3f0b bf68 bf48 be56 bf2b be1e be22 bed5 beca bdb6 be84 be9e bf1c bd27 bf10 3df5 3e10 bea9 3e7e bd1b be91 3f29 bda9 be45 be3d 3d67 bf3d be56 3e3f 3d4d 3f08 bef2 3e74 3e10 3ef5 3edf be60 3d5b bea0 3d99 3ef4 be84 3d0d 3f91 3cec bf20 bf1d 3ed5 3eca bf09 3c81 befe 3dc2 be74 bf27 3e6d 3e2f 3e48 beba 3f57 bf21 3eb4 3f5e beed bf16 3ce9 bf3f bf0d bcb3 3d8a be67 bd25 bcad bf78 3f10 bf1f bdfe 3e7e bdcf 3f5a
    \\3f04 bef5 bf29 beb5 3eed 3f71 3d87 bf9f 3d4c bf0c 3e04 3e90 3eab bf23 3e53 3e51 bf4a 3b72 be79 3cfc 3e5c be9a 3f0b 3f2b be8f 3cff bc83 3f2e 3dea bebe be4d bec3 3f07 3eec 3d92 bdde 3aac bd83 3f21 3f4a 3e95 bf09 be65 3f16 bc2d be44 3f24 3f07 bee1 be6e bbdd bde6 3f2f 3dd6 be92 3eb6 be3b be89 3f3d 3e34 bdc4 be77 3ce5 3e9b bee3 3f47 bd37 3e4f beca bca3 be1e 3e3d 3e0a beb3 3e3b 3f07 3f5b bde2 3d59 be8d 3f22 be9e 3dcf bd9f bd37 bee0 3eaf be00 3e51 becc be57 3f13 3e8c bdf0 3e83 3f00 3c3d becd 3f8c be86 bf4b 3eaf bed7 3e47 bca4 bf96 3f3d bebe bf1a bdf1 be83 3d8b 3f20 bf16 be02 3f33 be59 be05 3e23 3dda 3ef4 be28 bf31 bdf8 bf31 3d0f be73 3d92
;
const merge_b =
    \\be94 3ddb 3ee6 bf00
;
const merge_expected =
    \\c8e99b79 ad76b0ba 64cadd68 5ec3677a 9ab89f7a 7cb078b7 9666d69a 578b8b95 bc98 3e2b
    \\9d5696a4 83bac936 ba18567a 85be7a88 5b44c679 a8d6a58d 99032a5b a598b539 3c9a be1a
    \\cca6ff83 9a59dadb 139d7b9b 9c879cab 87695a8d 7cbb6c7d ab107676 e9c7a6bc 3c9b be2f
    \\67c85b26 9d7b32c9 bff8c7ba 9909583a 3929d574 bfacc7d4 dbc81941 c9896e8d 3c6c be05
;

fn bf16Values(comptime lines: []const u8, out: []f32) !void {
    var it = std.mem.tokenizeAny(u8, lines, " \n");
    var i: usize = 0;
    while (it.next()) |tok| : (i += 1) out[i] = host4.f32of(try std.fmt.parseInt(u16, tok, 16));
    try std.testing.expectEqual(out.len, i);
}

test "a merge with a non-power-of-two scale reproduces the independently produced words, scales and biases" {
    var dense: [256]f32 = undefined;
    var a: [256]f32 = undefined;
    var b: [4]f32 = undefined;
    try bf16Values(merge_dense, &dense);
    try bf16Values(merge_a, &a);
    try bf16Values(merge_b, &b);
    var words: [32]u32 = undefined;
    var scales: [4]u16 = undefined;
    var biases: [4]u16 = undefined;
    var bad: usize = 0;
    try lm.mergeDense(std.testing.allocator, 64, 2, 128, &dense, .{ .a = &a, .b = &b, .r = 2, .scale = 0.1 }, .{ .words = &words, .scales = &scales, .biases = &biases }, &bad);
    var lines = std.mem.tokenizeScalar(u8, merge_expected, '\n');
    var g: usize = 0;
    while (lines.next()) |line| : (g += 1) {
        var want: [10]u32 = undefined;
        try hexWords(u32, line, &want);
        try std.testing.expectEqualSlices(u32, want[0..8], words[g * 8 ..][0..8]);
        try std.testing.expectEqual(@as(u16, @intCast(want[8])), scales[g]);
        try std.testing.expectEqual(@as(u16, @intCast(want[9])), biases[g]);
    }
    try std.testing.expectEqual(@as(usize, 4), g);
}
