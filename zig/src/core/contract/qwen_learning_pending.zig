//! Dense 27B learning/image launch configurations. Zero usage is pending, never a compiler receipt.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "27B tf_train_zero", .image = "train", .symbol = "tf_train_zero", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_q27_gemm_q4w_bf", .image = "qwen27_train", .symbol = "tf_q27_gemm_q4w_bf", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_q27_gemm_q4w", .image = "qwen27_train", .symbol = "tf_q27_gemm_q4w", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_q27_gemm_q4_bf", .image = "qwen27_train", .symbol = "tf_q27_gemm_q4_bf", .threads = 128, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_q27_gemm_q4", .image = "qwen27_train", .symbol = "tf_q27_gemm_q4", .threads = 128, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_sum_parts", .image = "attn_train", .symbol = "tf_sum_parts", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_q27_fit_step", .image = "qwen27_train", .symbol = "tf_q27_fit_step", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_q27_zero_rows", .image = "qwen27_train", .symbol = "tf_q27_zero_rows", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_train_softmax", .image = "train", .symbol = "tf_train_softmax", .threads = 1024, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_train_rms_back", .image = "train", .symbol = "tf_train_rms_back", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_axpy", .image = "attn_train", .symbol = "tf_axpy", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_q27_add_bf16", .image = "qwen27_train", .symbol = "tf_q27_add_bf16", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_q27_cols", .image = "qwen27_train", .symbol = "tf_q27_cols", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_q27_pocket_rows", .image = "qwen27_train", .symbol = "tf_q27_pocket_rows", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_q27_pocket_back", .image = "qwen27_train", .symbol = "tf_q27_pocket_back", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_q27_pocket_grad", .image = "qwen27_train", .symbol = "tf_q27_pocket_grad", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_q27_swiglu_back", .image = "qwen27_train", .symbol = "tf_q27_swiglu_back", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_attn_prep", .image = "attn_train", .symbol = "tf_attn_prep", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_attn_gate_back", .image = "attn_train", .symbol = "tf_attn_gate_back", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_attn_q", .image = "attn_train", .symbol = "tf_attn_q", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_train_attn_kv", .image = "train", .symbol = "tf_train_attn_kv", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_attn_prep_back", .image = "attn_train", .symbol = "tf_attn_prep_back", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_q27_fit_dots", .image = "qwen27_train", .symbol = "tf_q27_fit_dots", .threads = 256, .dynamic = 20480 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_q27_fit_dots_tree", .image = "qwen27_train", .symbol = "tf_q27_fit_dots_tree", .threads = 256, .dynamic = 20480 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_gdn_front", .image = "gdn_train", .symbol = "tf_gdn_front", .threads = 128, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_gdn_scan_fwd", .image = "gdn_train", .symbol = "tf_gdn_scan_fwd", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_gdn_norm_back", .image = "gdn_train", .symbol = "tf_gdn_norm_back", .threads = 128, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_gdn_scan_back", .image = "gdn_train", .symbol = "tf_gdn_scan_back", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_gdn_qk_back", .image = "gdn_train", .symbol = "tf_gdn_qk_back", .threads = 128, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_gdn_gates_back", .image = "gdn_train", .symbol = "tf_gdn_gates_back", .threads = 64, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_gdn_conv_back", .image = "gdn_train", .symbol = "tf_gdn_conv_back", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_gdn_dequant", .image = "gdn_train", .symbol = "tf_gdn_dequant", .threads = 256, .dynamic = 0 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_train_narrow", .image = "train", .symbol = "tf_train_narrow", .threads = 256 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
    .{ .launch = .{ .name = "27B tf_train_gemm", .image = "train", .symbol = "tf_train_gemm", .threads = 128 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
};
test "Trainer admission: every missing dense learning resource receipt keeps qualification red" {
    const std = @import("std");
    for (records) |record| try std.testing.expect(try registry.checkCuda(std.testing.allocator, &.{record}, true) > 0);
}
