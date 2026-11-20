//! Bounded typed-event journals for background Responses; replay copies events while the owner is locked.
const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;

pub const max_bytes = 1 << 20;

pub const Log = struct {
    gpa: Allocator,
    arena: std.heap.ArenaAllocator,
    events: std.ArrayList(*json.Object) = .empty,
    last: ?u64 = null,
    truncated: bool = false,

    pub fn init(gpa: Allocator) Log {
        return .{ .gpa = gpa, .arena = .init(gpa) };
    }

    pub fn deinit(log: *Log) void {
        log.arena.deinit();
        log.* = init(log.gpa);
    }

    pub fn charge(log: *const Log) usize {
        return log.arena.queryCapacity();
    }

    pub fn record(log: *Log, payload: *json.Object) Allocator.Error!void {
        const raw = payload.get("sequence_number") orelse return;
        const sequence = std.fmt.parseInt(u64, raw.int, 10) catch return;
        log.last = sequence;
        if (log.truncated) return;
        const a = log.arena.allocator();
        const copied = try json.deepCopy(a, .{ .object = payload });
        try log.events.append(a, copied.object);
        if (log.charge() <= max_bytes) return;
        log.truncated = true;
        log.arena.deinit();
        log.arena = .init(log.gpa);
        log.events = .empty;
    }

    pub fn copyAfter(log: *const Log, a: Allocator, after: ?u64, queued_base: ?*json.Object) ![]*json.Object {
        if (log.truncated) return error.StaleCursor;
        const last = log.last orelse if (queued_base != null) 0 else return error.StaleCursor;
        if (after) |cursor| if (cursor > last) return error.InvalidCursor;
        var copied: std.ArrayList(*json.Object) = .empty;
        if (log.events.items.len == 0) {
            const base = queued_base.?;
            const created = try json.newObject(a);
            try created.put(a, "type", .{ .string = "response.created" });
            try created.put(a, "sequence_number", try json.intValue(a, 0));
            try created.put(a, "response", try json.deepCopy(a, .{ .object = base }));
            if (after == null) try copied.append(a, created);
            return copied.toOwnedSlice(a);
        }
        for (log.events.items) |event| {
            const raw = event.get("sequence_number").?;
            const sequence = std.fmt.parseInt(u64, raw.int, 10) catch unreachable;
            if (after) |cursor| if (sequence <= cursor) continue;
            try copied.append(a, (try json.deepCopy(a, .{ .object = event })).object);
        }
        return copied.toOwnedSlice(a);
    }
};

test "background Responses: journal replays an exclusive cursor and rejects a future one" {
    var route: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer route.deinit();
    const a = route.allocator();
    var log: Log = .init(std.testing.allocator);
    defer log.deinit();
    for (0..3) |i| {
        const event = try json.newObject(a);
        try event.put(a, "type", .{ .string = "response.output_text.delta" });
        try event.put(a, "sequence_number", try json.intValue(a, i));
        try event.put(a, "delta", .{ .string = "x" });
        try log.record(event);
    }
    const suffix = try log.copyAfter(a, 1, null);
    try std.testing.expectEqual(@as(usize, 1), suffix.len);
    try std.testing.expectEqualStrings("2", suffix[0].get("sequence_number").?.int);
    try std.testing.expectError(error.InvalidCursor, log.copyAfter(a, 4, null));
}

test "background Responses: a queued journal has typed liveness without output" {
    var route: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer route.deinit();
    const a = route.allocator();
    var log: Log = .init(std.testing.allocator);
    defer log.deinit();
    const base = try json.newObject(a);
    try base.put(a, "status", .{ .string = "queued" });
    const events = try log.copyAfter(a, null, base);
    try std.testing.expectEqual(@as(usize, 1), events.len);
    try std.testing.expectEqualStrings("response.created", events[0].get("type").?.string);
    const after = try log.copyAfter(a, 0, base);
    try std.testing.expectEqual(@as(usize, 0), after.len);
}

test "background Responses: an oversized journal reports a stale cursor instead of a false complete replay" {
    var route: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer route.deinit();
    const a = route.allocator();
    var log: Log = .init(std.testing.allocator);
    defer log.deinit();
    const event = try json.newObject(a);
    try event.put(a, "sequence_number", try json.intValue(a, 0));
    const text = try a.alloc(u8, max_bytes + 1);
    @memset(text, 'x');
    try event.put(a, "delta", .{ .string = text });
    try log.record(event);
    try std.testing.expect(log.truncated);
    try std.testing.expectError(error.StaleCursor, log.copyAfter(a, null, null));
}
