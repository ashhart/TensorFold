//! CUDA lane nvcc 13.0 receipts; sm121 refreshed with 13.0.88. Other SMs and later compilers keep their selection.
const CudaRecord = @import("registry.zig").CudaRecord;
const none = @import("registry.zig").none;
pub const records = [_]CudaRecord{
    .{ .launch = .{ .name = "qwen35 tf_q35_gdn_slots_1", .symbol = "tf_q35_gdn_slots_1", .image = "qwen35", .threads = 128 }, .usage = .{ .{ .regs = 56, .static = 0, .bound = 128 }, .{ .regs = 55, .static = 0, .bound = 128 }, .{ .regs = 55, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 4096, .bound = 128 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35 tf_q35_gdn_slots_2", .symbol = "tf_q35_gdn_slots_2", .image = "qwen35", .threads = 128 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 8192, .bound = 128 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35 tf_q35_gdn_slots_4", .symbol = "tf_q35_gdn_slots_4", .image = "qwen35", .threads = 128 }, .usage = .{ .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 56, .static = 0, .bound = 128 }, .{ .regs = 84, .static = 16384, .bound = 128 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35 tf_q35_gdn_slots_8", .symbol = "tf_q35_gdn_slots_8", .image = "qwen35", .threads = 128 }, .usage = .{ .{ .regs = 96, .static = 0, .bound = 128 }, .{ .regs = 96, .static = 0, .bound = 128 }, .{ .regs = 96, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 80, .static = 16384, .bound = 128 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35 tf_q35_gdn_slots_16", .symbol = "tf_q35_gdn_slots_16", .image = "qwen35", .threads = 128 }, .usage = .{ .{ .regs = 150, .static = 0, .bound = 128 }, .{ .regs = 149, .static = 0, .bound = 128 }, .{ .regs = 149, .static = 0, .bound = 128 }, .{ .regs = 108, .static = 0, .bound = 128 }, .{ .regs = 80, .static = 32768, .bound = 128 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35 tf_q35_gdn_chain", .symbol = "tf_q35_gdn_chain", .image = "qwen35", .threads = 128 }, .usage = .{ .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 56, .static = 0, .bound = 128 }, .{ .regs = 56, .static = 0, .bound = 128 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35 tf_q35_rope", .symbol = "tf_q35_rope", .image = "qwen35", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 16384, .bound = 256 }, .{ .regs = 47, .static = 16384, .bound = 256 }, .{ .regs = 47, .static = 16384, .bound = 256 }, .{ .regs = 34, .static = 16384, .bound = 256 }, .{ .regs = 34, .static = 512, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35 tf_q35_attn_partial", .symbol = "tf_q35_attn_partial", .image = "qwen35", .threads = 128, .dynamic = 65536, .opt_in = true }, .usage = .{ .{ .regs = 48, .static = 0, .bound = 0 }, .{ .regs = 48, .static = 0, .bound = 0 }, .{ .regs = 48, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 168, .static = 11648, .bound = 128 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35 tf_q35_attn_rows", .symbol = "tf_q35_attn_rows", .image = "qwen35", .threads = 256, .dynamic = 65536, .opt_in = true }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 244, .static = 6912, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35 tf_q35_attn_merge", .symbol = "tf_q35_attn_merge", .image = "qwen35", .threads = 32 }, .usage = .{ .{ .regs = 40, .static = 0, .bound = 32 }, .{ .regs = 38, .static = 0, .bound = 32 }, .{ .regs = 37, .static = 0, .bound = 32 }, .{ .regs = 42, .static = 0, .bound = 32 }, .{ .regs = 42, .static = 0, .bound = 32 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35_kv8 tf_q35_rope8", .symbol = "tf_q35_rope8", .image = "qwen35_kv8", .threads = 256 }, .usage = .{ .{ .regs = 40, .static = 16416, .bound = 256 }, .{ .regs = 41, .static = 16416, .bound = 256 }, .{ .regs = 42, .static = 16416, .bound = 256 }, .{ .regs = 40, .static = 16416, .bound = 256 }, .{ .regs = 38, .static = 544, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35_kv8 tf_q35_attn_partial8", .symbol = "tf_q35_attn_partial8", .image = "qwen35_kv8", .threads = 128 }, .usage = .{ .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 56, .static = 0, .bound = 0 }, .{ .regs = 34, .static = 0, .bound = 0 }, .{ .regs = 197, .static = 44672, .bound = 128 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35_kv8 tf_q35_attn_rows8", .symbol = "tf_q35_attn_rows8", .image = "qwen35_kv8", .threads = 256 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 243, .static = 39936, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35_keep tf_q35k_apply", .symbol = "tf_q35k_apply", .image = "qwen35_keep", .threads = 128 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 128 }, .{ .regs = 40, .static = 0, .bound = 128 }, .{ .regs = 40, .static = 0, .bound = 128 }, .{ .regs = 40, .static = 0, .bound = 128 }, .{ .regs = 40, .static = 0, .bound = 128 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35_keep tf_q35k_compact", .symbol = "tf_q35k_compact", .image = "qwen35_keep", .threads = 256 }, .usage = .{ .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 17, .static = 0, .bound = 256 }, .{ .regs = 17, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 36, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35_kv8 tf_q35k_compact8", .symbol = "tf_q35k_compact8", .image = "qwen35_kv8", .threads = 256 }, .usage = .{ .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 }, .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 28, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "w4a8_prefill tf_w4a8_quantize", .symbol = "tf_w4a8_quantize", .image = "w4a8_prefill", .threads = 128 }, .usage = none },
    .{ .launch = .{ .name = "w4a8_prefill tf_w4a8_prefill", .symbol = "tf_w4a8_prefill", .image = "w4a8_prefill", .threads = 256, .dynamic = 55296, .opt_in = true }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 165, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35_draft tf_q35d_gather_rows", .symbol = "tf_q35d_gather_rows", .image = "qwen35_draft", .threads = 256 }, .usage = .{ .{ .regs = 20, .static = 0, .bound = 0 }, .{ .regs = 20, .static = 0, .bound = 0 }, .{ .regs = 20, .static = 0, .bound = 0 }, .{ .regs = 20, .static = 0, .bound = 0 }, .{ .regs = 20, .static = 0, .bound = 0 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35 tf_q35_gdn_pre", .symbol = "tf_q35_gdn_pre", .image = "qwen35", .threads = 32 }, .usage = .{ .{ .regs = 50, .static = 0, .bound = 32 }, .{ .regs = 52, .static = 0, .bound = 32 }, .{ .regs = 52, .static = 0, .bound = 32 }, .{ .regs = 40, .static = 0, .bound = 32 }, .{ .regs = 38, .static = 0, .bound = 32 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35 tf_q35_conv_commit", .symbol = "tf_q35_conv_commit", .image = "qwen35", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35 tf_q35_tap", .symbol = "tf_q35_tap", .image = "qwen35", .threads = 256 }, .usage = .{ .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 26, .static = 0, .bound = 256 }, .{ .regs = 26, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35_keep tf_q35k_conv", .symbol = "tf_q35k_conv", .image = "qwen35_keep", .threads = 256 }, .usage = .{ .{ .regs = 20, .static = 0, .bound = 0 }, .{ .regs = 20, .static = 0, .bound = 0 }, .{ .regs = 20, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35_draft tf_q35d_dyn_conv_batch", .symbol = "tf_q35d_dyn_conv_batch", .image = "qwen35_draft", .threads = 256 }, .usage = .{ .{ .regs = 29, .static = 0, .bound = 0 }, .{ .regs = 28, .static = 0, .bound = 0 }, .{ .regs = 28, .static = 0, .bound = 0 }, .{ .regs = 36, .static = 0, .bound = 0 }, .{ .regs = 38, .static = 0, .bound = 0 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35_draft tf_q35d_norm_rope_batch", .symbol = "tf_q35d_norm_rope_batch", .image = "qwen35_draft", .threads = 32 }, .usage = .{ .{ .regs = 30, .static = 0, .bound = 32 }, .{ .regs = 32, .static = 0, .bound = 32 }, .{ .regs = 32, .static = 0, .bound = 32 }, .{ .regs = 36, .static = 0, .bound = 32 }, .{ .regs = 36, .static = 0, .bound = 32 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35_draft tf_q35d_mask_rows", .symbol = "tf_q35d_mask_rows", .image = "qwen35_draft", .threads = 256 }, .usage = .{ .{ .regs = 14, .static = 0, .bound = 0 }, .{ .regs = 16, .static = 0, .bound = 0 }, .{ .regs = 16, .static = 0, .bound = 0 }, .{ .regs = 16, .static = 0, .bound = 0 }, .{ .regs = 16, .static = 0, .bound = 0 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35_draft tf_q35d_attn_partial", .symbol = "tf_q35d_attn_partial", .image = "qwen35_draft", .threads = 128 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 114, .static = 3616, .bound = 128 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35_draft tf_q35d_attn_merge", .symbol = "tf_q35d_attn_merge", .image = "qwen35_draft", .threads = 32 }, .usage = .{ .{ .regs = 37, .static = 0, .bound = 32 }, .{ .regs = 37, .static = 0, .bound = 32 }, .{ .regs = 37, .static = 0, .bound = 32 }, .{ .regs = 34, .static = 0, .bound = 32 }, .{ .regs = 34, .static = 0, .bound = 32 } }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35_bf16_rows tf_q35_embed_bf16", .symbol = "tf_q35_embed_bf16", .image = "qwen35_bf16_rows", .threads = 256 }, .usage = .{
        .{ .regs = 16, .static = 0, .bound = 256 },
        .{ .regs = 16, .static = 0, .bound = 256 },
        .{ .regs = 16, .static = 0, .bound = 256 },
        .{ .regs = 40, .static = 0, .bound = 256 },
        .{ .regs = 40, .static = 0, .bound = 256 },
    }, .later_pending = true },
    .{ .launch = .{ .name = "qwen35_bf16_rows tf_q35_dense_bf16", .symbol = "tf_q35_dense_bf16", .image = "qwen35_bf16_rows", .threads = 256 }, .usage = .{
        .{ .regs = 34, .static = 32, .bound = 256 },
        .{ .regs = 40, .static = 32, .bound = 256 },
        .{ .regs = 40, .static = 32, .bound = 256 },
        .{ .regs = 42, .static = 32, .bound = 256 },
        .{ .regs = 42, .static = 32, .bound = 256 },
    }, .later_pending = true },
};

test "sm121 receipts leave unmeasured launches and later compiler selections pending" {
    const std = @import("std");
    const reg = @import("registry.zig");
    for (records) |r| {
        if (std.mem.eql(u8, r.launch.symbol, "tf_w4a8_quantize")) {
            try std.testing.expectEqual(none, r.usage);
            continue;
        }
        if (std.mem.eql(u8, r.launch.symbol, "tf_q35_attn_rows") or
            std.mem.eql(u8, r.launch.symbol, "tf_q35_attn_rows8") or
            std.mem.eql(u8, r.launch.symbol, "tf_w4a8_prefill") or
            std.mem.eql(u8, r.launch.symbol, "tf_q35d_attn_partial"))
        {
            for (r.usage[0..4]) |u| try std.testing.expectEqual(@as(u32, 0), u.regs);
            try std.testing.expect(r.usage[4].regs != 0);
            const later = reg.usageFor(r, .{ .major = 13, .minor = 2 });
            if (std.mem.eql(u8, r.launch.symbol, "tf_q35d_attn_partial")) {
                try std.testing.expectEqual(@as(u32, 50), later[4].regs);
                try std.testing.expectEqual(@as(u32, 32), later[4].bound);
            } else try std.testing.expectEqual(none, later);
        }
    }
}
