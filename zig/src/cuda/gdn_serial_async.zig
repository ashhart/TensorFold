//! Optional asynchronous BF16 staging around the existing serial GDN arithmetic.
const std = @import("std");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const serial = @import("gdn_serial_tiles.zig");
pub const Serial = serial.Serial;
pub const Choice = serial.Choice;
fn choice(comptime rows: usize, comptime steps: usize) Choice {
    return .{ .rows = rows, .steps = steps, .symbol = std.fmt.comptimePrint("tf_gdn_async_{d}_{d}", .{ rows, steps }) };
}
pub const choices = [_]Choice{ choice(32, 16), choice(32, 32), choice(64, 16), choice(64, 32) };
pub fn select(rows: usize, steps: usize) !Choice {
    for (choices) |c| if (c.rows == rows and c.steps == steps) return c;
    return error.UnsupportedGdnAsyncTile;
}
pub fn sharedBytes(rows: usize, steps: usize) !usize {
    _ = try select(rows, steps);
    return 2 * steps * (512 + 2 * rows + 6);
}
pub fn load(d: *const Driver, image: []const u8, rows: usize, steps: usize) !Serial {
    const config = try select(rows, steps);
    var module = try Module.load(d, image);
    errdefer module.unload();
    return .{ .module = module, .function = try module.function(config.symbol), .config = config };
}
test "GDN async staging keeps 16-byte copy alignment and bounded static shared memory" {
    for (choices) |c| {
        const bytes = try sharedBytes(c.rows, c.steps);
        try std.testing.expectEqual(@as(usize, 0), bytes % 16);
        try std.testing.expect(bytes < 48 * 1024);
        try std.testing.expectEqual(@as(usize, 0), (c.steps * 16) % (2 * c.rows));
        try std.testing.expectEqual(@as(usize, 0), (c.steps * (c.rows / 8)) % (2 * c.rows));
    }
    try std.testing.expectEqual(@as(usize, 18624), try sharedBytes(32, 16));
    try std.testing.expectEqual(@as(usize, 37248), try sharedBytes(32, 32));
    try std.testing.expectEqual(@as(usize, 20672), try sharedBytes(64, 16));
    try std.testing.expectEqual(@as(usize, 41344), try sharedBytes(64, 32));
    try std.testing.expectError(error.UnsupportedGdnAsyncTile, select(16, 16));
}
