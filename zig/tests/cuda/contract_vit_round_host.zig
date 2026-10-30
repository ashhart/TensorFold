//! Maintained registration and tie witness for RADIO's fused BF16 epilogue.
const std = @import("std");
const core = @import("core");
const Glue = @import("contract_vit_glue.zig").Glue;
const t = std.testing;

test "RADIO rounding: the omitted bias-round control is routed to the shapes gate" {
    var glue: Glue = .{ .rig = undefined, .k = undefined, .act = .none, .residual = true, .in_place = false, .wrong_bias_round = true };
    try t.expectEqual(@as(?core.contract.gates.Gate, .shapes), glue.kernel().control);
}

test "RADIO rounding: the finite tie fixture distinguishes removing the intermediate BF16 round" {
    const bf = core.bf16_round;
    const acc: f32 = 1.00390625;
    const bias: f32 = 0;
    const residual: f32 = -1;
    const rounded = bf.bits(@as(f32, @bitCast(@as(u32, bf.bits(acc + bias)) << 16)) + residual);
    const wrong = bf.bits((acc + bias) + residual);
    try t.expectEqual(@as(u16, 0), rounded);
    try t.expectEqual(@as(u16, 0x3b80), wrong);
    try t.expect(rounded != wrong);
}
