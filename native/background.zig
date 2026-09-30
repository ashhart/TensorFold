const std = @import("std");

pub fn requested(body: std.json.Value, chat: bool) bool {
    if (body != .object) return false;
    if (body.object.get("priority")) |priority| if (priority == .string and std.mem.eql(u8, priority.string, "background")) return true;
    if (!chat) return false;
    if (body.object.get("tools")) |tools| if (tools == .array and tools.array.items.len > 0) return false;
    const messages = body.object.get("messages") orelse return false;
    if (messages != .array or messages.array.items.len == 0) return false;
    const first = messages.array.items[0];
    if (first != .object) return false;
    const role = first.object.get("role") orelse return false;
    const text = first.object.get("content") orelse return false;
    if (role != .string or !std.mem.eql(u8, role.string, "system") or text != .string) return false;
    var size: usize = 0;
    for (messages.array.items) |message| {
        if (message != .object) return false;
        const content = message.object.get("content") orelse return false;
        if (content != .string) return false;
        size +|= std.unicode.utf8CountCodepoints(content.string) catch return false;
    }
    return size < 4096 and std.ascii.findIgnoreCase(text.string, "title") != null;
}

pub fn Queue(comptime T: type) type {
    return struct {
        const Self = @This();
        // Eight waiting clients plus the eight active jobs that may be preempted.
        entries: [16]T = undefined,
        len: usize = 0,
        closed: bool = false,
        mutex: std.Io.Mutex = .init,
        condition: std.Io.Condition = .init,

        pub fn put(q: *Self, io: std.Io, item: T, resumed: bool) bool {
            q.mutex.lockUncancelable(io);
            defer q.mutex.unlock(io);
            if (q.closed or (!resumed and q.len >= 8)) return false;
            std.debug.assert(q.len < q.entries.len);
            q.entries[q.len] = item;
            q.len += 1;
            q.condition.signal(io);
            return true;
        }
        fn remove(q: *Self, at: usize) T {
            const item = q.entries[at];
            std.mem.copyForwards(T, q.entries[at .. q.len - 1], q.entries[at + 1 .. q.len]);
            q.len -= 1;
            return item;
        }
        pub fn take(q: *Self, io: std.Io, wait: bool, foreground_only: bool) error{Closed}!?T {
            q.mutex.lockUncancelable(io);
            defer q.mutex.unlock(io);
            while (q.len == 0) {
                if (q.closed) return error.Closed;
                if (!wait) return null;
                q.condition.waitUncancelable(io, &q.mutex);
            }
            for (q.entries[0..q.len], 0..) |item, i| if (!item.background) return q.remove(i);
            return if (foreground_only) null else q.remove(0);
        }
        pub fn foreground(q: *Self, io: std.Io) ?T {
            q.mutex.lockUncancelable(io);
            defer q.mutex.unlock(io);
            for (q.entries[0..q.len]) |item| if (!item.background) return item;
            return null;
        }
        pub fn removeIf(q: *Self, io: std.Io, predicate: *const fn (T) bool) ?T {
            q.mutex.lockUncancelable(io);
            defer q.mutex.unlock(io);
            for (q.entries[0..q.len], 0..) |item, i| if (predicate(item)) return q.remove(i);
            return null;
        }
        pub fn close(q: *Self, io: std.Io) void {
            q.mutex.lockUncancelable(io);
            defer q.mutex.unlock(io);
            q.closed = true;
            q.condition.broadcast(io);
        }
    };
}

pub const Replay = struct {
    end: usize = 0,
    at: usize = 0,
    pub fn restart(r: *Replay, delivered: []const u8) void {
        r.* = .{ .end = delivered.len };
    }
    pub fn fresh(r: *Replay, delivered: []const u8, piece: []const u8) ![]const u8 {
        const count = @min(r.end - r.at, piece.len);
        if (!std.mem.eql(u8, delivered[r.at..][0..count], piece[0..count])) return error.BackgroundReplayDiverged;
        r.at += count;
        return piece[count..];
    }
    pub fn token(expected: []const u32, position: usize, actual: i32) !void {
        if (position < expected.len and (actual < 0 or expected[position] != @as(u32, @intCast(actual)))) return error.BackgroundReplayDiverged;
    }
};

test "replay suppresses delivered bytes across new chunk and UTF-8 boundaries" {
    var replay = Replay{};
    const old = "a🌍tool";
    replay.restart(old);
    try std.testing.expectEqualStrings("", try replay.fresh(old, old[0..3]));
    try std.testing.expectEqualStrings("", try replay.fresh(old, old[3..]));
    try std.testing.expectEqualStrings("new", try replay.fresh(old, "new"));
    replay.restart(old);
    try std.testing.expectEqualStrings("new", try replay.fresh(old, "a🌍toolnew"));
    replay.restart(old);
    try std.testing.expectError(error.BackgroundReplayDiverged, replay.fresh(old, "b"));
    try Replay.token(&.{ 1, 2 }, 0, 1);
    try std.testing.expectError(error.BackgroundReplayDiverged, Replay.token(&.{ 1, 2 }, 1, 3));
    try Replay.token(&.{1}, 1, 2);
}

test "foreground bypasses queued backgrounds and interrupted jobs retain capacity" {
    const Item = struct { background: bool, id: usize };
    var queue = Queue(Item){};
    const io = std.testing.io;
    for (0..7) |id| try std.testing.expect(queue.put(io, .{ .background = true, .id = id }, false));
    try std.testing.expect(queue.put(io, .{ .background = false, .id = 7 }, false));
    try std.testing.expect(!queue.put(io, .{ .background = false, .id = 8 }, false));
    for (8..16) |id| try std.testing.expect(queue.put(io, .{ .background = true, .id = id }, true));
    try std.testing.expectEqual(@as(usize, 7), queue.foreground(io).?.id);
    try std.testing.expectEqual(@as(usize, 7), (try queue.take(io, false, true)).?.id);
    try std.testing.expectEqual(null, try queue.take(io, false, true));
    for (0..7) |id| try std.testing.expectEqual(id, (try queue.take(io, false, false)).?.id);
    for (8..16) |id| try std.testing.expectEqual(id, (try queue.take(io, false, false)).?.id);
    try std.testing.expectEqual(null, try queue.take(io, false, false));
    queue.close(io);
    try std.testing.expectError(error.Closed, queue.take(io, true, false));
    try std.testing.expect(!queue.put(io, .{ .background = false, .id = 0 }, true));
}

test "title requests require short text-only chat without tools" {
    const cases = .{
        .{ true, true, "{\"messages\":[{\"role\":\"system\",\"content\":\"Choose a TITLE\"}]}" },
        .{ false, false, "{\"messages\":[{\"role\":\"system\",\"content\":\"Choose a title\"}]}" },
        .{ false, true, "{\"messages\":[{\"role\":\"user\",\"content\":\"Choose a title\"}]}" },
        .{ false, true, "{\"tools\":[{}],\"messages\":[{\"role\":\"system\",\"content\":\"Choose a title\"}]}" },
        .{ false, true, "{\"messages\":[{\"role\":\"system\",\"content\":\"title\"},{\"role\":\"user\",\"content\":[] }]}" },
        .{ true, false, "{\"priority\":\"background\"}" },
    };
    inline for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, case[2], .{});
        defer parsed.deinit();
        try std.testing.expectEqual(case[0], requested(parsed.value, case[1]));
    }
    const title = "{\"messages\":[{\"role\":\"system\",\"content\":\"title\"},{\"role\":\"user\",\"content\":\"\"}]}";
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, title, .{});
    defer parsed.deinit();
    const content = parsed.value.object.get("messages").?.array.items[1].object.getPtr("content").?;
    var text: [4090 * 4 + 1]u8 = undefined;
    for (0..4090) |i| @memcpy(text[i * 4 ..][0..4], "🌍");
    text[text.len - 1] = 'x';
    content.* = .{ .string = text[0 .. text.len - 1] };
    try std.testing.expect(requested(parsed.value, true));
    content.* = .{ .string = &text };
    try std.testing.expect(!requested(parsed.value, true));
}
