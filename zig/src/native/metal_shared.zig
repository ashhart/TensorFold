//! What the Metal open paths share: the GPU's chip class and name, its prompt chunk, the model's window, GiB bytes.
const std = @import("std");
const mtl = @import("metal");
const Allocator = std.mem.Allocator;

/// The chip class gate entries name ("apple-m5" for an Apple M5 Max); null without an Apple GPU.
pub fn chip(a: Allocator) ?[]const u8 {
    const device = mtl.Device.init() catch return null;
    defer device.deinit();
    const gen = generation(std.mem.span(device.name())) orelse return null;
    return std.fmt.allocPrint(a, "apple-m{d}", .{gen}) catch null;
}

/// The name of the system GPU the Metal engines open ("Apple M5 Max"); null without one.
pub fn deviceName(a: Allocator, _: ?u32) ?[]const u8 {
    const device = mtl.Device.init() catch return null;
    defer device.deinit();
    return a.dupe(u8, std.mem.span(device.name())) catch null;
}

/// The M generation in a Metal device name ("Apple M5 Max": 5).
fn generation(name: []const u8) ?u32 {
    var words = std.mem.tokenizeScalar(u8, name, ' ');
    if (!std.ascii.eqlIgnoreCase(words.next() orelse return null, "apple")) return null;
    const word = words.next() orelse return null;
    if (word.len < 2 or std.ascii.toLower(word[0]) != 'm') return null;
    const gen = std.fmt.parseInt(u32, word[1..], 10) catch return null;
    return if (gen >= 1) gen else null;
}

/// Python's prompt chunk: 8,192 rows with tensor units (M5 on) where memory allows, else its 2,048.
pub fn prefillStep(device: mtl.Device) u32 {
    const gen = generation(std.mem.span(device.name())) orelse 0;
    return if (gen >= 5 and device.maxWorkingSet() >= 48 << 30) 8192 else 2048;
}

/// A GiB count's bytes (the flags allow 0 up to below 2^34).
pub fn gibBytes(g: f64) u64 {
    return @intFromFloat(g * (1 << 30));
}

test "chip classes from Metal device names" {
    try std.testing.expectEqual(@as(?u32, 5), generation("Apple M5 Max"));
    try std.testing.expectEqual(@as(?u32, 12), generation("Apple M12"));
    try std.testing.expectEqual(@as(?u32, null), generation("AMD Radeon Pro"));
    try std.testing.expectEqual(@as(?u32, null), generation("Apple Mx"));
}
