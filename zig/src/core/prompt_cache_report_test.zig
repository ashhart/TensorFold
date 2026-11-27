//! Pressure wording, levels and rate limits on the prompt cache's per-request report.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const report = @import("prompt_cache_report.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;

fn expectContains(line: report.Line, level: report.Level, text: []const u8) !void {
    try std.testing.expectEqual(level, line.level);
    try std.testing.expect(std.mem.indexOf(u8, line.text, text) != null);
}

test "memory pressure: a kept-zero report warns once, reminds at 60 seconds and clears at info" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{ .gpa = gpa };
    var store = pc.Store.init(gpa, fake.snapshots(), .{}, 1 << 20);
    defer store.deinit();
    var buf: [1024]u8 = undefined;

    const normal = try report.renderAt(&store, 12, 0, 0, 0, &buf);
    try expectContains(normal, .info, "kept 0;");
    try std.testing.expect(std.mem.indexOf(u8, normal.text, "memory pressure") == null);

    store.hold = true;
    try expectContains(try report.renderAt(&store, 12, 0, 0, std.time.ns_per_s, &buf), .warning, "kept 0;");
    try expectContains(try report.renderAt(&store, 12, 0, 0, 30 * std.time.ns_per_s, &buf), .info, "kept 0;");
    const reminder = try report.renderAt(&store, 12, 0, 0, 61 * std.time.ns_per_s, &buf);
    try std.testing.expectEqual(report.Level.warning, reminder.level);
    try std.testing.expectEqualStrings("prompt cache: memory pressure still prevents new prompt states", reminder.text);
    try expectContains(try report.renderAt(&store, 12, 0, 0, 62 * std.time.ns_per_s, &buf), .info, "kept 0;");

    store.hold = false;
    const cleared = try report.renderAt(&store, 12, 0, 0, 63 * std.time.ns_per_s, &buf);
    try std.testing.expectEqual(report.Level.info, cleared.level);
    try std.testing.expectEqualStrings("prompt cache: memory pressure cleared; new prompt states are kept again", cleared.text);
    try expectContains(try report.renderAt(&store, 12, 0, 0, 64 * std.time.ns_per_s, &buf), .info, "kept 0;");

    store.hold = true;
    try expectContains(try report.renderAt(&store, 12, 0, 0, 65 * std.time.ns_per_s, &buf), .warning, "memory pressure: new prompt states are not kept");
}
