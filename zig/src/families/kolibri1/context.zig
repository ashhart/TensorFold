//! Logical context plus bounded verification reserve; arithmetic shared by native pricing and lane admission.
const std = @import("std");
pub const reserve = 16;

pub fn capacity(window: usize) !usize {
    if (window == 0) return error.BadContext;
    const with_reserve = try std.math.add(usize, window, reserve);
    return (try std.math.add(usize, with_reserve, 1023)) / 1024 * 1024;
}

pub fn fits(physical: usize, prompt: usize, reply: usize) bool {
    return prompt <= physical and reply <= physical - prompt and reserve <= physical - prompt - reply;
}

test "Kolibri context: logical context boundary fits only with the producer's verification reserve" {
    const physical = try capacity(262144);
    try std.testing.expectEqual(@as(usize, 263168), physical);
    try std.testing.expect(fits(physical, 262080, 64));
    try std.testing.expect(!fits(262144, 262080, 64));
    try std.testing.expect(fits(physical, physical - reserve - 1, 1));
    try std.testing.expect(!fits(physical, physical - reserve, 1));
}

test "Kolibri context: native context pricing checks zero, overflow and padded versus unpadded windows" {
    try std.testing.expectEqual(@as(usize, 65536), try capacity(65000));
    try std.testing.expectEqual(@as(usize, 66560), try capacity(65536));
    try std.testing.expectError(error.BadContext, capacity(0));
    try std.testing.expectError(error.Overflow, capacity(std.math.maxInt(usize)));
    try std.testing.expectError(error.Overflow, capacity(std.math.maxInt(usize) - reserve));
    try std.testing.expect(!fits(std.math.maxInt(usize), std.math.maxInt(usize), 1));
}
