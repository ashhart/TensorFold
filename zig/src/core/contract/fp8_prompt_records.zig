//! Native FP8 prompt launch receipts from nvcc 13.5 for sm_120a and sm_121a.
const std = @import("std");
const r = @import("registry.zig");
const limits = @import("limits.zig");

fn entry(comptime symbol: []const u8, comptime registers: u32, comptime shared: u32) r.CudaRecord {
    const used: limits.Usage = .{ .regs = registers, .static = shared, .bound = 256 };
    const none: limits.Usage = .{ .regs = 0, .static = 0 };
    return .{
        .launch = .{ .name = "native FP8 " ++ symbol, .symbol = symbol, .image = "fp8_prefill", .threads = 256, .min_sm = 120 },
        .usage = .{ none, none, none, used, used },
        .first_nvcc = .{ .major = 13, .minor = 5 },
    };
}

pub const records = [_]r.CudaRecord{
    entry("tf_fp8_prompt_quant", 16, 0),
    entry("tf_affine4_fp8_quant", 27, 0),
    entry("tf_fp8_row_quant", 22, 64),
    entry("tf_fp8_prompt", 92, 49152),
    entry("tf_fp8_prompt_f32", 92, 49152),
    entry("tf_fp8_row_prompt", 92, 49152),
    entry("tf_fp8_row_prompt_f32", 92, 49152),
    entry("tf_affine4_fp8_prompt", 107, 18432),
    entry("tf_affine4_fp8_prompt_f32", 107, 18432),
};

test "FP8 prompt records qualify only the measured compiler and Blackwell profiles" {
    for (records) |record| {
        try std.testing.expectEqual(r.none, r.usageFor(record, null));
        try std.testing.expectEqual(r.none, r.usageFor(record, .{ .major = 13, .minor = 3 }));
        try std.testing.expectEqual(record.usage, r.usageFor(record, .{ .major = 13, .minor = 5 }));
        for (record.usage[0..3]) |usage| try std.testing.expectEqual(@as(u32, 0), usage.regs);
    }
    try std.testing.expectEqual(@as(usize, 0), try r.checkCuda(std.testing.allocator, &records, true));
}
