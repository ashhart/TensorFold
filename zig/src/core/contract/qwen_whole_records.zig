//! Separate whole-window and paged prompt ABIs await their own five-SM compiler receipts.
const std = @import("std");
const reg = @import("registry.zig");
const bf16_whole = "_ZN20tf_prefill_attention19hpattn_whole_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifi";
const fp8_whole = "_ZN20tf_prefill_attention20hpattn8_whole_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifi";
const bf16_pages = "_ZN20tf_prefill_attention13hpattn_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifiPKixi";
const fp8_pages = "_ZN20tf_prefill_attention14hpattn8_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifiPKixi";
pub const records = blk: {
    var result: [8]reg.CudaRecord = undefined;
    for (.{ "tf_q35_attn_partial_whole", "tf_q35_attn_rows_whole", "tf_q35_attn_partial8_whole", "tf_q35_attn_rows8_whole", bf16_whole, fp8_whole, bf16_pages, fp8_pages }, 0..) |symbol, i| {
        result[i] = .{ .launch = .{ .name = symbol, .symbol = symbol, .image = if (i < 2) "qwen35" else if (i < 4) "qwen35_kv8" else "prefill_attention", .threads = if (i < 4 and i % 2 == 0) 128 else 256, .dynamic = if (i < 2) 65536 else if (i >= 4) 98304 else 0, .opt_in = i < 2 or i >= 4, .min_sm = 80 }, .usage = reg.none, .later_pending = true };
    }
    break :blk result;
};

test "whole attention registrations pin distinct real symbols and fail closed without five-SM receipts" {
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
        try std.testing.expectEqual(missing, try reg.checkCuda(std.testing.allocator, &.{r}, true));
        var unmeasured = r;
        unmeasured.usage = reg.none;
        try std.testing.expectEqual(@as(usize, 5), try reg.checkCuda(std.testing.allocator, &.{unmeasured}, true));
    }
}
