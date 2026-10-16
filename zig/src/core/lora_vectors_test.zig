//! MLX quantizer vectors made independently (a float32 reference of the published algorithm) on synthetic groups.
const std = @import("std");
const host4 = @import("affine4_host.zig");

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
