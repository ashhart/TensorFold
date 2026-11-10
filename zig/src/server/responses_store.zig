//! The Responses store: the newest finished responses with their messages, and the chains they continue.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const pyrepr = @import("pyrepr.zig");
const translate_mod = @import("responses_translate.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

const Entry = struct {
    arena: std.heap.ArenaAllocator,
    id: []const u8,
    response: Value,
    parent: ?[]const u8,
    messages: []Value,
    size: usize,
};

/// The newest stored responses, in memory, for GET, DELETE and ``previous_response_id``.
pub const Store = struct {
    gpa: Allocator,
    mutex: std.Io.Mutex = .init,
    entries: std.ArrayList(*Entry) = .empty, // oldest first
    bytes: usize = 0,
    limit: usize = 1024,
    max_bytes: usize = 256 * 1024 * 1024,

    pub fn deinit(s: *Store) void {
        for (s.entries.items) |e| free(s.gpa, e);
        s.entries.deinit(s.gpa);
    }

    fn free(gpa: Allocator, e: *Entry) void {
        var arena = e.arena;
        arena.deinit();
        gpa.destroy(e);
    }

    fn find(s: *Store, id: []const u8) ?usize {
        for (s.entries.items, 0..) |e, i| if (std.mem.eql(u8, e.id, id)) return i;
        return null;
    }

    /// Keeps a finished response with the messages it added; the oldest go past 1,024 entries or 256 MiB.
    pub fn put(s: *Store, io: std.Io, cx: *Cx, response: Value, added: []const Value) !void {
        const from_output = try translate_mod.messages(cx, response.get("output").?.array);
        const all = try std.mem.concat(cx.a, Value, &.{ added, from_output });
        return s.insert(io, cx, response, all, true);
    }

    /// Keeps a finished response with its whole conversation and no stored parent (its public fields stay as sent).
    pub fn putWhole(s: *Store, io: std.Io, cx: *Cx, response: Value, messages: []Value) !void {
        return s.insert(io, cx, response, messages, false);
    }

    /// One entry for ``response`` with these messages; ``chained`` links it to its ``previous_response_id``.
    fn insert(s: *Store, io: std.Io, cx: *Cx, response: Value, messages: []Value, chained: bool) !void {
        const e = try s.gpa.create(Entry);
        e.arena = .init(s.gpa);
        errdefer free(s.gpa, e);
        const a = e.arena.allocator();
        e.response = try json.deepCopy(a, response);
        e.id = e.response.get("id").?.string;
        const parent = e.response.get("previous_response_id");
        e.parent = if (chained and parent != null and parent.? == .string) parent.?.string else null;
        e.messages = (try json.deepCopy(a, .{ .array = messages })).array;
        e.size = (try json.stringify(cx.a, e.response, .{})).len + (try json.stringify(cx.a, .{ .array = e.messages }, .{})).len;
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        if (s.find(e.id)) |i| {
            const old = s.entries.orderedRemove(i);
            s.bytes -= old.size;
            free(s.gpa, old);
        }
        try s.entries.append(s.gpa, e);
        s.bytes += e.size;
        while (s.entries.items.len > 0 and (s.entries.items.len > s.limit or s.bytes > s.max_bytes)) {
            const old = s.entries.orderedRemove(0);
            s.bytes -= old.size;
            free(s.gpa, old);
        }
    }

    pub fn get(s: *Store, io: std.Io, a: Allocator, id: []const u8) !?Value {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        const i = s.find(id) orelse return null;
        return try json.deepCopy(a, s.entries.items[i].response);
    }

    pub fn delete(s: *Store, io: std.Io, id: []const u8) bool {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        const i = s.find(id) orelse return false;
        const old = s.entries.orderedRemove(i);
        s.bytes -= old.size;
        free(s.gpa, old);
        return true;
    }
};

/// Every message of a stored response's chain, oldest first (the chain stays newest in the store).
pub const History = struct {
    store: *Store,
    io: std.Io,

    pub fn conversation(h: History, cx: *Cx, rid: Value) errors.Refused![]Value {
        if (rid != .string) return cx.refuse("previous_response_id must be a string");
        return switch (try h.walk(cx.a, rid.string)) {
            .found => |messages| messages,
            .missing => |id| cx.fail(.request, "previous response {s} is not stored here (store: false, deleted, or evicted): send the conversation's items instead", .{try pyrepr.repr(cx.a, .{ .string = id })}),
        };
    }

    /// The chain's messages, or null when any response of it is not stored (or memory ran out).
    pub fn lookup(h: History, cx: *Cx, id: []const u8) ?[]Value {
        return switch (h.walk(cx.a, id) catch return null) {
            .found => |messages| messages,
            .missing => null,
        };
    }

    const Walk = union(enum) { found: []Value, missing: []const u8 };

    /// The chain from ``id`` back, copied into ``a``, or a copy of the first id it misses.
    fn walk(h: History, a: Allocator, id: []const u8) Allocator.Error!Walk {
        const s = h.store;
        s.mutex.lockUncancelable(h.io);
        defer s.mutex.unlock(h.io);
        var chain: std.ArrayList([]Value) = .empty;
        var at: ?[]const u8 = id;
        while (at) |x| {
            const i = s.find(x) orelse return .{ .missing = try a.dupe(u8, x) };
            const e = s.entries.orderedRemove(i);
            s.entries.append(s.gpa, e) catch {};
            try chain.append(a, e.messages);
            at = e.parent;
        }
        var out: std.ArrayList(Value) = .empty;
        var i = chain.items.len;
        while (i > 0) {
            i -= 1;
            for (chain.items[i]) |m| try out.append(a, try json.deepCopy(a, m));
        }
        return .{ .found = out.items };
    }
};

test "a whole response keeps its own conversation and no stored parent, and a lookup miss is null" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var cx: Cx = .{ .a = a };
    var s: Store = .{ .gpa = std.testing.allocator };
    defer s.deinit();
    const h: History = .{ .store = &s, .io = io };
    const user = (try json.parse(a, "{\"role\":\"user\",\"content\":\"x\"}")).ok;
    var one = [_]Value{user};
    try s.put(io, &cx, (try json.parse(a, "{\"id\":\"resp_p\",\"output\":[]}")).ok, &one);
    try s.put(io, &cx, (try json.parse(a, "{\"id\":\"resp_d\",\"previous_response_id\":\"resp_p\",\"output\":[]}")).ok, &one);
    var three = [_]Value{ user, user, user };
    try s.putWhole(io, &cx, (try json.parse(a, "{\"id\":\"resp_c\",\"previous_response_id\":\"resp_p\",\"output\":[]}")).ok, &three);
    try std.testing.expectEqual(@as(usize, 2), h.lookup(&cx, "resp_d").?.len);
    try std.testing.expectEqual(@as(usize, 3), h.lookup(&cx, "resp_c").?.len);
    try std.testing.expectEqualStrings("resp_p", (try s.get(io, a, "resp_c")).?.get("previous_response_id").?.string);
    try std.testing.expect(h.lookup(&cx, "resp_x") == null);
    try std.testing.expectError(error.Refused, h.conversation(&cx, .{ .string = "resp_x" }));
    try std.testing.expectEqual(@as(usize, 3), (try h.conversation(&cx, .{ .string = "resp_c" })).len);
}
