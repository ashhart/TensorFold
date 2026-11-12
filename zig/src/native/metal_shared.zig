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

/// The model's window (config.json's max_position_embeddings, text_config's first), 0 when it names none.
pub fn modelContext(a: Allocator, io: std.Io, dir: []const u8) i64 {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return 0;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return 0;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return 0;
    if (doc != .object) return 0;
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    const limit = text.object.get("max_position_embeddings") orelse doc.object.get("max_position_embeddings") orelse return 0;
    return if (limit == .integer and limit.integer > 0) limit.integer else 0;
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
