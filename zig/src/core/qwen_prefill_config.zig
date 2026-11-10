//! Build-time CUDA prompt capacity; verification and context limits remain independently bounded.
const std = @import("std");
pub const default_rows: usize = 4096;
pub const max_rows = resolve(@import("qwen_prefill_options").max_rows) catch
    @compileError("qwen-prefill-max-rows must be 4096 or 8192");
pub const warm_rows = [_]usize{ 128, 512, 1024, max_rows - 512, max_rows };

pub fn resolve(requested: ?usize) !usize {
    const rows = requested orelse default_rows;
    if (rows != 4096 and rows != 8192) return error.InvalidQwenPrefillLimit;
    return rows;
}
pub fn admitLearning(enabled: bool) !void {
    if (max_rows > default_rows and enabled) return error.WidePrefillLearningUnsupported;
}
pub fn admitPockets(enabled: bool) !void {
    if (max_rows > default_rows and enabled) return error.WidePrefillPocketsUnsupported;
}

test "Qwen prefill config: default and explicit capacities are closed" {
    try std.testing.expectEqual(@as(usize, 4096), try resolve(null));
    try std.testing.expectEqual(@as(usize, 4096), try resolve(4096));
    try std.testing.expectEqual(@as(usize, 8192), try resolve(8192));
    for ([_]usize{ 0, 1024, 4097, 8193, 16384 }) |rows|
        try std.testing.expectError(error.InvalidQwenPrefillLimit, resolve(rows));
    try std.testing.expect(max_rows == 4096 or max_rows == 8192);
    try std.testing.expectEqual(max_rows, warm_rows[warm_rows.len - 1]);
    try std.testing.expectEqual(max_rows - 512, warm_rows[warm_rows.len - 2]);
}
test "Qwen prefill config: wide builds reject incompatible learners and pockets" {
    try admitLearning(false);
    try admitPockets(false);
    if (max_rows == 8192) {
        try std.testing.expectError(error.WidePrefillLearningUnsupported, admitLearning(true));
        try std.testing.expectError(error.WidePrefillPocketsUnsupported, admitPockets(true));
    } else {
        try admitLearning(true);
        try admitPockets(true);
        try std.testing.expectEqualSlices(usize, &.{ 128, 512, 1024, 3584, 4096 }, &warm_rows);
    }
}
