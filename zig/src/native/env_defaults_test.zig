//! The documented TF_* defaults the native engines read from the environment, pinned on the host.
const std = @import("std");
const api = @import("engine_api");
const env = @import("sys").env;
const setting = @import("cuda_device.zig").setting;
const drama_host = @import("drama_host.zig");
const drama_open = @import("drama_open.zig");

fn set(name: [:0]const u8, value: ?[:0]const u8) !void {
    const rc = if (value) |v| env.setenv(name, v, 1) else env.unsetenv(name);
    try std.testing.expectEqual(@as(c_int, 0), rc);
}

/// One variable's value when the test began, put back when it ends.
const Saved = struct {
    name: [:0]const u8,
    value: ?[:0]u8,

    fn take(name: [:0]const u8) !Saved {
        const raw = std.c.getenv(name) orelse return .{ .name = name, .value = null };
        return .{ .name = name, .value = try std.testing.allocator.dupeSentinel(u8, std.mem.span(raw), 0) };
    }

    fn restore(s: Saved) void {
        if (s.value) |v| {
            _ = env.setenv(s.name, v, 1);
            std.testing.allocator.free(v);
        } else _ = env.unsetenv(s.name);
    }
};

test "env defaults: TF_DRAMA_PREFILL_ROWS unset prefills one row and a count inside 1 through 1024 is taken" {
    const a = std.testing.allocator;
    const rows = try Saved.take("TF_DRAMA_PREFILL_ROWS");
    defer rows.restore();
    const device = try Saved.take("TF_CUDA_DEVICE");
    defer device.restore();
    try set("TF_DRAMA_PREFILL_ROWS", null);
    try set("TF_CUDA_DEVICE", null);
    var problem: []const u8 = "";
    try std.testing.expectEqual(@as(u32, 1), (drama_host.Options{}).prefill_rows);
    try std.testing.expectEqual(@as(?u32, 1), try setting(a, null, "TF_DRAMA_PREFILL_ROWS", (drama_host.Options{}).prefill_rows, &problem));
    try set("TF_DRAMA_PREFILL_ROWS", "128");
    try std.testing.expectEqual(@as(?u32, 128), try setting(a, null, "TF_DRAMA_PREFILL_ROWS", (drama_host.Options{}).prefill_rows, &problem));
}

test "env defaults: TF_DRAMA_PREFILL_ROWS outside 1 through 1024 is refused by name before any checkpoint" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const rows = try Saved.take("TF_DRAMA_PREFILL_ROWS");
    defer rows.restore();
    const device = try Saved.take("TF_CUDA_DEVICE");
    defer device.restore();
    try set("TF_CUDA_DEVICE", null);
    var problem: []const u8 = "";
    for ([_][:0]const u8{ "0", "1025" }) |value| {
        try set("TF_DRAMA_PREFILL_ROWS", value);
        try std.testing.expectEqual(@as(?api.Opened, null), try drama_open.open(arena.allocator(), a, std.testing.io, .{ .dir = "", .model_type = "drama3" }, &problem));
        try std.testing.expectEqualStrings("TF_DRAMA_PREFILL_ROWS must be 1 through 1024", problem);
    }
    try set("TF_DRAMA_PREFILL_ROWS", "rows");
    try std.testing.expectEqual(@as(?api.Opened, null), try drama_open.open(arena.allocator(), a, std.testing.io, .{ .dir = "", .model_type = "drama3" }, &problem));
    try std.testing.expectEqualStrings("TF_DRAMA_PREFILL_ROWS=\"rows\": expected whole prompt rows a chunk, 1 through 1024 (checked when a Drama 3 checkpoint opens)", problem);
}

test "env defaults: TF_DRAMA_REFERENCE_CACHE unset keeps no voices and more than 16 is refused by name" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var saved: [3]Saved = undefined;
    for (&saved, [_][:0]const u8{ "TF_DRAMA_REFERENCE_CACHE", "TF_DRAMA_PREFILL_ROWS", "TF_CUDA_DEVICE" }) |*s, name| s.* = try Saved.take(name);
    defer for (saved) |s| s.restore();
    for (saved) |s| try set(s.name, null);
    var problem: []const u8 = "";
    try std.testing.expectEqual(@as(?u32, 0), try setting(a, null, "TF_DRAMA_REFERENCE_CACHE", (drama_host.Options{}).reference_entries, &problem));
    try set("TF_DRAMA_REFERENCE_CACHE", "17");
    try std.testing.expectEqual(@as(?api.Opened, null), try drama_open.open(arena.allocator(), a, std.testing.io, .{ .dir = "", .model_type = "drama3" }, &problem));
    try std.testing.expectEqualStrings("TF_DRAMA_REFERENCE_CACHE must be 0 (off) through 16 entries", problem);
}
