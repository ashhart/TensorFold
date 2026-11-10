//! Optional FP8 tile launches; compile-resource receipts must be filled before qualification.
const std = @import("std");
const r = @import("registry.zig");

fn entry(comptime symbol: []const u8, comptime dynamic: u32) r.CudaRecord {
    return .{
        .launch = .{ .name = "native FP8 tile " ++ symbol, .symbol = symbol, .image = "fp8_prefill_tiles", .threads = 256, .dynamic = dynamic, .opt_in = dynamic > 49152, .min_sm = 120 },
        .usage = r.none,
        .geometry_pending = true,
        .first_nvcc = .{ .major = 13, .minor = 5 },
    };
}

pub const records = [_]r.CudaRecord{
    entry("tf_fp8_tile128x64", 49152),
    entry("tf_fp8_tile128x64_f32", 49152),
    entry("tf_fp8_tile128x64_row", 49152),
    entry("tf_fp8_tile128x64_row_f32", 49152),
    entry("tf_fp8_tile128x128", 65536),
    entry("tf_fp8_tile128x128_f32", 65536),
    entry("tf_fp8_tile128x128_row", 65536),
    entry("tf_fp8_tile128x128_row_f32", 65536),
    entry("tf_fp8_tile64x64", 32768),
    entry("tf_fp8_tile64x64_f32", 32768),
    entry("tf_fp8_tile64x64_row", 32768),
    entry("tf_fp8_tile64x64_row_f32", 32768),
};

test "FP8 prompt optional records keep shared memory opt-in and unique symbols" {
    for (records, 0..) |record, i| {
        try std.testing.expectEqual(@as(u32, 256), record.launch.threads);
        try std.testing.expectEqual(record.launch.dynamic > 49152, record.launch.opt_in);
        for (records[i + 1 ..]) |other| try std.testing.expect(!std.mem.eql(u8, record.launch.symbol, other.launch.symbol));
    }
}
