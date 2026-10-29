//! nvcc 13.0 five-SM usage for separate whole-window and paged prompt ABIs.
const std = @import("std");
const reg = @import("registry.zig");
pub const records = [_]reg.CudaRecord{
    .{ .launch = .{ .name = "tf_q35_attn_partial_whole", .symbol = "tf_q35_attn_partial_whole", .image = "qwen35", .threads = 128, .dynamic = 65536, .opt_in = true, .min_sm = 80 }, .usage = .{
        .{ .regs = 167, .static = 11648, .bound = 128 },
        .{ .regs = 167, .static = 11648, .bound = 128 },
        .{ .regs = 167, .static = 11648, .bound = 128 },
        .{ .regs = 168, .static = 11648, .bound = 128 },
        .{ .regs = 168, .static = 11648, .bound = 128 },
    }, .later_pending = true },
    .{ .launch = .{ .name = "tf_q35_attn_rows_whole", .symbol = "tf_q35_attn_rows_whole", .image = "qwen35", .threads = 256, .dynamic = 65536, .opt_in = true, .min_sm = 80 }, .usage = .{
        .{ .regs = 254, .static = 6912, .bound = 256 },
        .{ .regs = 254, .static = 6912, .bound = 256 },
        .{ .regs = 254, .static = 6912, .bound = 256 },
        .{ .regs = 244, .static = 6912, .bound = 256 },
        .{ .regs = 244, .static = 6912, .bound = 256 },
    }, .later_pending = true },
    .{ .launch = .{ .name = "tf_q35_attn_partial8_whole", .symbol = "tf_q35_attn_partial8_whole", .image = "qwen35_kv8", .threads = 128, .dynamic = 0, .opt_in = false, .min_sm = 80 }, .usage = .{
        .{ .regs = 199, .static = 44672, .bound = 128 },
        .{ .regs = 199, .static = 44672, .bound = 128 },
        .{ .regs = 208, .static = 44672, .bound = 128 },
        .{ .regs = 197, .static = 44672, .bound = 128 },
        .{ .regs = 197, .static = 44672, .bound = 128 },
    }, .later_pending = true },
    .{ .launch = .{ .name = "tf_q35_attn_rows8_whole", .symbol = "tf_q35_attn_rows8_whole", .image = "qwen35_kv8", .threads = 256, .dynamic = 0, .opt_in = false, .min_sm = 80 }, .usage = .{
        .{ .regs = 255, .static = 39936, .bound = 256 },
        .{ .regs = 255, .static = 39936, .bound = 256 },
        .{ .regs = 255, .static = 39936, .bound = 256 },
        .{ .regs = 243, .static = 39936, .bound = 256 },
        .{ .regs = 243, .static = 39936, .bound = 256 },
    }, .later_pending = true },
    .{ .launch = .{ .name = "_ZN20tf_prefill_attention19hpattn_whole_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifi", .symbol = "_ZN20tf_prefill_attention19hpattn_whole_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifi", .image = "prefill_attention", .threads = 256, .dynamic = 98304, .opt_in = true, .min_sm = 80 }, .usage = .{
        .{ .regs = 254, .static = 0, .bound = 256 },
        .{ .regs = 254, .static = 0, .bound = 256 },
        .{ .regs = 254, .static = 0, .bound = 256 },
        .{ .regs = 255, .static = 0, .bound = 256 },
        .{ .regs = 255, .static = 0, .bound = 256 },
    }, .later_pending = true },
    .{ .launch = .{ .name = "_ZN20tf_prefill_attention20hpattn8_whole_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifi", .symbol = "_ZN20tf_prefill_attention20hpattn8_whole_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifi", .image = "prefill_attention", .threads = 256, .dynamic = 98304, .opt_in = true, .min_sm = 80 }, .usage = .{
        .{ .regs = 255, .static = 0, .bound = 256 },
        .{ .regs = 255, .static = 0, .bound = 256 },
        .{ .regs = 255, .static = 0, .bound = 256 },
        .{ .regs = 255, .static = 0, .bound = 256 },
        .{ .regs = 255, .static = 0, .bound = 256 },
    }, .later_pending = true },
    .{ .launch = .{ .name = "_ZN20tf_prefill_attention13hpattn_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifiPKixi", .symbol = "_ZN20tf_prefill_attention13hpattn_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifiPKixi", .image = "prefill_attention", .threads = 256, .dynamic = 98304, .opt_in = true, .min_sm = 80 }, .usage = .{
        .{ .regs = 253, .static = 0, .bound = 256 },
        .{ .regs = 253, .static = 0, .bound = 256 },
        .{ .regs = 253, .static = 0, .bound = 256 },
        .{ .regs = 231, .static = 0, .bound = 256 },
        .{ .regs = 231, .static = 0, .bound = 256 },
    }, .later_pending = true },
    .{ .launch = .{ .name = "_ZN20tf_prefill_attention14hpattn8_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifiPKixi", .symbol = "_ZN20tf_prefill_attention14hpattn8_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifiPKixi", .image = "prefill_attention", .threads = 256, .dynamic = 98304, .opt_in = true, .min_sm = 80 }, .usage = .{
        .{ .regs = 255, .static = 0, .bound = 256 },
        .{ .regs = 255, .static = 0, .bound = 256 },
        .{ .regs = 255, .static = 0, .bound = 256 },
        .{ .regs = 255, .static = 0, .bound = 256 },
        .{ .regs = 255, .static = 0, .bound = 256 },
    }, .later_pending = true },
};

test "whole attention receipts fit five SMs and leave later compilers pending" {
    for (records, 0..) |r, i| {
        var found: usize = 0;
        for (reg.all_cuda) |active| if (std.mem.eql(u8, active.launch.symbol, r.launch.symbol) and std.mem.eql(u8, active.launch.image, r.launch.image)) {
            try std.testing.expectEqualDeep(r, active);
            found += 1;
        };
        try std.testing.expectEqual(@as(usize, 1), found);
        for (records[0..i]) |other| try std.testing.expect(!std.mem.eql(u8, other.launch.symbol, r.launch.symbol));
        var missing: usize = 0;
        for (r.usage) |u| if (u.regs == 0) {
            missing += 1;
        };
        try std.testing.expectEqual(@as(usize, 0), missing);
        try std.testing.expectEqual(@as(usize, 0), try reg.checkCuda(std.testing.allocator, &.{r}, true));
        for ([_]?@import("limits.zig").Nvcc{ null, .{ .major = 13, .minor = 0 }, .{ .major = 13, .minor = 1 } }) |compiler|
            try std.testing.expectEqual(r.usage, reg.usageFor(r, compiler));
        for ([_]@import("limits.zig").Nvcc{ .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 }, .{ .major = 14, .minor = 0 } }) |compiler|
            try std.testing.expectEqual(reg.none, reg.usageFor(r, compiler));
        var unmeasured = r;
        unmeasured.usage = reg.none;
        try std.testing.expectEqual(@as(usize, 5), try reg.checkCuda(std.testing.allocator, &.{unmeasured}, true));
    }
}
