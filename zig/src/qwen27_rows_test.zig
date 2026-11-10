//! The production prompt-row policies over real owned lane streams, without a device.
const std = @import("std");
const rows = @import("families/qwen27/prompt_rows.zig");
const policy = @import("native/qwen27_prompt_rows.zig");
const Stream = @import("core/lanes/stream.zig").Stream;
const a = std.testing.allocator;

test "qwen27 rows: every qualified chip sizes its dense frame to the unchanged prompt width" {
    for ([_]u32{ 13, 14, 15, 16, 17, 18 }, [_]u32{ 128, 128, 128, 128, 2048, 512 }) |generation, expected| {
        const geometry = @import("families/qwen27/attention_plan.zig").Geometry{ .shared_prefix = generation >= 17 };
        const chosen = rows.frameRequested(generation, false, null, geometry);
        try std.testing.expectEqual(expected, chosen);
        try std.testing.expectEqual(geometry.promptRows(rows.requested(generation, false, null)), geometry.promptRows(chosen));
    }
}

test "qwen27 rows: deferred loading keeps the real geometry and the M5 initial frame" {
    try std.testing.expectEqual(@as(u32, 128), rows.frameRequested(0, false, null, .{ .shared_prefix = false }));
    try std.testing.expectEqual(@as(u32, 512), rows.frameRequested(0, false, null, .{}));
    try std.testing.expectEqual(@as(u32, 2048), rows.frameRequested(17, false, null, .{}));
    try std.testing.expectEqual(@as(u32, 512), rows.frameRequested(15, false, null, .{}));
}

test "qwen27 rows: frame overrides and routed defaults stay exact on every chip" {
    for ([_]u32{ 0, 13, 14, 15, 16, 17, 18 }) |generation| {
        const geometry = @import("families/qwen27/attention_plan.zig").Geometry{ .shared_prefix = generation >= 17 };
        try std.testing.expectEqual(@as(u32, 1024), rows.frameRequested(generation, true, null, geometry));
        for ([_]bool{ false, true }) |routed| {
            for ([_][]const u8{ "16", "128", "512", "1024", "2048" }, [_]u32{ 16, 128, 512, 1024, 2048 }) |value, wanted| {
                try std.testing.expectEqual(wanted, rows.frameRequested(generation, routed, value, geometry));
            }
        }
    }
}

test "qwen27 rows: only the measured generation widens its default" {
    for ([_]u32{ 0, 13, 14, 15, 16, 18 }) |generation| try std.testing.expectEqual(@as(u32, 512), rows.requested(generation, false, null));
    try std.testing.expectEqual(@as(u32, 2048), rows.requested(17, false, null));
}

test "qwen27 rows: with experts the default is 1,024 on every chip, and overrides stay exact" {
    for ([_]u32{ 0, 13, 16, 17, 18 }) |generation| {
        try std.testing.expectEqual(@as(u32, 1024), rows.requested(generation, true, null));
        try std.testing.expectEqual(@as(u32, 1024), rows.requested(generation, true, "15"));
        try std.testing.expectEqual(@as(u32, 512), rows.requested(generation, true, "512"));
    }
}

test "qwen27 rows: valid overrides stay exact and invalid ones use the chip default" {
    for ([_]u32{ 0, 13, 17, 18 }) |generation| {
        const valid = [_][]const u8{ "16", "128", "512", "1024", "2048" };
        for (valid, [_]u32{ 16, 128, 512, 1024, 2048 }) |text, want| try std.testing.expectEqual(want, rows.requested(generation, false, text));
        for ([_][]const u8{ "", "0", "15", "17", "2049", "4096", "-16", "4294967296", " 512", "no" }) |text| {
            try std.testing.expectEqual(@as(u32, if (generation == 17) 2048 else 512), rows.requested(generation, false, text));
        }
    }
}

const Backend = struct {
    runner: struct { slots: usize = 3 } = .{},
    chunk: usize = 2048,
    owners: [3]?*Stream = @splat(null),
    fills: std.AutoHashMapUnmanaged(*Stream, bool) = .empty,
};

test "qwen27 rows: a decoder arriving between chunks caps the next one and leaving widens it" {
    var prompt = try Stream.init(a, .{ .id = "prompt", .prompt = &.{ 1, 2 }, .max_new = 4, .drafts = false });
    defer prompt.deinit(a);
    var other = try Stream.init(a, .{ .id = "other", .prompt = &.{ 1, 2 }, .max_new = 4, .drafts = false });
    defer other.deinit(a);
    var b: Backend = .{ .owners = .{ &prompt, null, null } };
    defer b.fills.deinit(a);
    try b.fills.put(a, &prompt, true);
    try std.testing.expectEqual(@as(usize, 2048), policy.choose(&b, &prompt, 0));
    b.owners[1] = &other;
    try b.fills.put(a, &other, true);
    try std.testing.expectEqual(@as(usize, 2048), policy.choose(&b, &prompt, 0));
    try std.testing.expect(b.fills.remove(&other));
    try std.testing.expectEqual(@as(usize, 512), policy.choose(&b, &prompt, 0));
    other.paused = true;
    try std.testing.expectEqual(@as(usize, 2048), policy.choose(&b, &prompt, 0));
    other.paused = false;
    try std.testing.expectEqual(@as(usize, 512), policy.choose(&b, &prompt, 0));
    other.finished = true;
    try std.testing.expectEqual(@as(usize, 2048), policy.choose(&b, &prompt, 0));
    other.finished = false;
    for ([_]usize{ 16, 128, 256, 512, 1024, 2048 }) |capacity| {
        b.chunk = capacity;
        try std.testing.expectEqual(@min(capacity, @as(usize, 512)), policy.choose(&b, &prompt, 0));
    }
    b.chunk = 2048;
    b.owners[1] = null;
    b.owners[2] = &other;
    b.runner.slots = 2;
    try std.testing.expectEqual(@as(usize, 2048), policy.choose(&b, &prompt, 0));
    b.runner.slots = 3;
    try std.testing.expectEqual(@as(usize, 512), policy.choose(&b, &prompt, 0));
    b.owners[2] = null;
    try std.testing.expectEqual(@as(usize, 2048), policy.choose(&b, &prompt, 0));
}

fn never(_: *anyopaque) bool {
    return false;
}

test "qwen27 rows: a background pass narrows its chunks with depth and a foreground one does not" {
    var prompt = try Stream.init(a, .{ .id = "prompt", .prompt = &.{ 1, 2 }, .max_new = 4, .drafts = false });
    defer prompt.deinit(a);
    var b: Backend = .{ .owners = .{ &prompt, null, null } };
    defer b.fills.deinit(a);
    try b.fills.put(a, &prompt, true);
    try std.testing.expectEqual(@as(usize, 2048), policy.choose(&b, &prompt, 120000)); // a foreground pass
    prompt.yield_check = .{ .ptr = &b, .check = never };
    for ([_]usize{ 0, 16384 }) |at| try std.testing.expectEqual(@as(usize, 2048), policy.choose(&b, &prompt, at));
    try std.testing.expectEqual(@as(usize, 1664), policy.choose(&b, &prompt, 20000)); // 1677, a whole 128 rows
    try std.testing.expectEqual(@as(usize, 1024), policy.choose(&b, &prompt, 32768));
    try std.testing.expectEqual(@as(usize, 256), policy.choose(&b, &prompt, 131072));
    try std.testing.expectEqual(@as(usize, 256), policy.choose(&b, &prompt, 262144)); // never below 256
    b.chunk = 128; // nor wider than the chunk
    try std.testing.expectEqual(@as(usize, 128), policy.choose(&b, &prompt, 131072));
    b.chunk = 2048;
    var other = try Stream.init(a, .{ .id = "other", .prompt = &.{ 1, 2 }, .max_new = 4, .drafts = false });
    defer other.deinit(a);
    b.owners[1] = &other; // a decoder beside it caps the chunk first
    try std.testing.expectEqual(@as(usize, 512), policy.choose(&b, &prompt, 20000));
    try std.testing.expectEqual(@as(usize, 256), policy.choose(&b, &prompt, 131072));
}

const floor = @import("native/qwen27_prompt_floor.zig");
const Load = struct {
    calls: usize = 0,
    fn open(l: *@This(), value: ?[]const u8, why: *[]const u8) !void {
        try floor.environment(a, value, why);
        l.calls += 1;
    }
};

test "qwen27 rows: a below-floor setting refuses before the loader with its value, origin and remedy" {
    var loader: Load = .{};
    for ([_][]const u8{ "0", "1", "16", "17", "32", "64", "112", "127" }) |value| {
        var why: []const u8 = "";
        try std.testing.expectError(error.PromptRowsTooSmall, loader.open(value, &why));
        defer a.free(why);
        try std.testing.expect(std.mem.indexOf(u8, why, "at least 128 prompt rows") != null);
        const origin = try std.fmt.allocPrint(a, "TF_PROMPT_ROWS supplied {s}.", .{value});
        defer a.free(origin);
        try std.testing.expect(std.mem.indexOf(u8, why, origin) != null);
        try std.testing.expect(std.mem.indexOf(u8, why, "unset it to use the device default") != null);
    }
    try std.testing.expectEqual(@as(usize, 0), loader.calls);
}

test "qwen27 rows: exactly the serving floor reaches the loader and keeps the requested capacity" {
    var loader: Load = .{};
    var why: []const u8 = "unchanged";
    try loader.open("128", &why);
    try loader.open(null, &why);
    try std.testing.expectEqual(@as(usize, 2), loader.calls);
    try std.testing.expectEqualStrings("unchanged", why);
    try std.testing.expectEqual(@as(u32, 128), rows.requested(13, false, "128"));
    try std.testing.expectEqual(@as(u32, 16), rows.requested(13, false, "16"));
}

test "qwen27 rows: a caller-created narrow frame names frame capacity rather than an unrelated environment" {
    var why: []const u8 = "";
    try std.testing.expectError(error.PromptRowsTooSmall, floor.admit(a, 64, .frame, &why));
    defer a.free(why);
    try std.testing.expectEqualStrings("native Qwen prompt serving needs at least 128 prompt rows; model.frame.capacity supplied 64. Create the model frame with at least 128 rows before attaching native lanes.", why);
    try floor.check(128);
}

test "qwen27 rows: malformed and over-cap native settings refuse before loading" {
    var loader: Load = .{};
    for ([_][]const u8{ "no", " 512", "129", "2049", "4096", "4294967296" }) |value| {
        var why: []const u8 = "";
        try std.testing.expectError(error.InvalidEnvironment, loader.open(value, &why));
        defer a.free(why);
        try std.testing.expect(std.mem.indexOf(u8, why, "TF_PROMPT_ROWS=") != null);
        try std.testing.expect(std.mem.indexOf(u8, why, "128 to 2048") != null);
    }
    try std.testing.expectEqual(@as(usize, 0), loader.calls);
}

test "qwen27 rows: empty or whitespace-only settings are unset and reach the loader" {
    var loader: Load = .{};
    var why: []const u8 = "unchanged";
    for ([_][]const u8{ "", " ", "\t\n" }) |value| try loader.open(value, &why);
    try std.testing.expectEqual(@as(usize, 3), loader.calls);
    try std.testing.expectEqualStrings("unchanged", why);
}
