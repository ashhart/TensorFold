//! Wide row launches reserve their 64 threads in source; no measured compiler limits are inferred.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = blk: {
    var out: [36]M = undefined;
    var i: usize = 0;
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| for (.{ "f32", "bf16" }) |sum| {
        for (.{ "", "_relu2", "_strided" }) |style| {
            out[i] = .{ .name = std.fmt.comptimePrint("row projection wide tf_row_projection_wide{s}_q{d}_{s}", .{ style, bits, sum }), .threads = 64, .tg_memory = 0, .reserved = 64 };
            i += 1;
        }
    };
    break :blk out;
};

test "wide row launch declarations cover all new symbols and fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 36), records.len);
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
