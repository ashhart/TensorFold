//! A Responses socket's connection-local cache: the latest finished response of each lane, under two byte budgets.
const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;

pub const connection_budget = 64 * 1024 * 1024;
pub const global_budget = 512 * 1024 * 1024;

/// One lane's latest response: its id and the complete message list a continuation needs, in its own arena.
pub const Entry = struct {
    arena: std.heap.ArenaAllocator,
    id: []const u8,
    lane: []const u8,
    messages: []json.Value,
    /// The JSON text length of ``messages``: what the entry counts against both budgets.
    size: usize,
};

pub const Cache = struct {
    gpa: Allocator,
    /// The server-wide count every socket's cache adds to and takes from (``Server.ws_cache_bytes``).
    global: *std.atomic.Value(usize),
    /// Lane name ("" for the default lane) to its entry, oldest first.
    lanes: std.array_hash_map.String(*Entry) = .empty,
    bytes: usize = 0,
    connection_budget: usize = connection_budget,
    global_budget: usize = global_budget,

    /// Caches ``messages`` as ``lane``'s latest response ``id``; false when it cannot fit, and the lane is then empty.
    pub fn put(c: *Cache, lane: []const u8, id: []const u8, messages: []const json.Value) Allocator.Error!bool {
        const entry = try copy(c.gpa, lane, id, messages);
        if (c.lanes.get(lane)) |old| c.drop(old);
        while (!c.fits(entry.size)) {
            if (c.lanes.count() == 0 or c.hopeless(entry.size)) {
                destroy(c.gpa, entry);
                return false;
            }
            c.drop(c.lanes.values()[0]);
        }
        c.lanes.put(c.gpa, entry.lane, entry) catch |e| {
            c.release(entry.size);
            destroy(c.gpa, entry);
            return e;
        };
        return true;
    }

    /// The cached response ``id`` in any lane.
    pub fn find(c: *const Cache, id: []const u8) ?*Entry {
        for (c.lanes.values()) |e| if (std.mem.eql(u8, e.id, id)) return e;
        return null;
    }

    /// Frees the cached response ``id``, when it is cached.
    pub fn evict(c: *Cache, id: []const u8) void {
        if (c.find(id)) |e| c.drop(e);
    }

    /// Frees every entry and returns its bytes to the server-wide count.
    pub fn deinit(c: *Cache) void {
        while (c.lanes.count() > 0) c.drop(c.lanes.values()[0]);
        c.lanes.deinit(c.gpa);
    }

    /// Reserves ``size`` on both budgets, or reserves nothing.
    fn fits(c: *Cache, size: usize) bool {
        if (size > c.connection_budget - @min(c.bytes, c.connection_budget)) return false;
        var now = c.global.load(.monotonic);
        while (true) {
            if (size > c.global_budget - @min(now, c.global_budget)) return false;
            now = c.global.cmpxchgWeak(now, now + size, .monotonic, .monotonic) orelse break;
        }
        c.bytes += size;
        return true;
    }

    /// Whether ``size`` cannot fit even with every entry of this connection freed, so evicting would gain nothing.
    fn hopeless(c: *const Cache, size: usize) bool {
        const others = c.global.load(.monotonic) -| c.bytes;
        return size > c.connection_budget or size > c.global_budget -| others;
    }

    fn release(c: *Cache, size: usize) void {
        c.bytes -= size;
        _ = c.global.fetchSub(size, .monotonic);
    }

    fn drop(c: *Cache, e: *Entry) void {
        _ = c.lanes.orderedRemove(e.lane);
        c.release(e.size);
        destroy(c.gpa, e);
    }
};

/// A deep copy of the response into a new entry with its own arena.
fn copy(gpa: Allocator, lane: []const u8, id: []const u8, messages: []const json.Value) Allocator.Error!*Entry {
    const entry = try gpa.create(Entry);
    entry.arena = .init(gpa);
    errdefer destroy(gpa, entry);
    const a = entry.arena.allocator();
    entry.lane = try a.dupe(u8, lane);
    entry.id = try a.dupe(u8, id);
    entry.messages = try a.alloc(json.Value, messages.len);
    for (messages, entry.messages) |m, *slot| slot.* = try json.deepCopy(a, m);
    entry.size = textSize(entry.messages);
    return entry;
}

fn destroy(gpa: Allocator, e: *Entry) void {
    e.arena.deinit();
    gpa.destroy(e);
}

/// The length of ``json.dumps(messages)``, counted without building the text.
fn textSize(messages: []json.Value) usize {
    var buf: [256]u8 = undefined;
    var counter: std.Io.Writer.Discarding = .init(&buf);
    json.write(&counter.writer, .{ .array = messages }, .{}) catch unreachable;
    return @intCast(counter.fullCount());
}

const testing = std.testing;

/// A message list whose JSON text is ``n`` bytes long: one string of ``n`` - 4 bytes.
fn sized(a: Allocator, n: usize) ![]json.Value {
    const text = try a.alloc(u8, n - 4);
    @memset(text, 'm');
    const out = try a.alloc(json.Value, 1);
    out[0] = .{ .string = text };
    return out;
}

test "ws_cache: a lane's new response replaces its entry, and find, evict and deinit free bytes" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var global: std.atomic.Value(usize) = .init(0);
    var c: Cache = .{ .gpa = testing.allocator, .global = &global };
    try testing.expect(try c.put("", "resp_1", try sized(a, 100)));
    try testing.expectEqual(@as(usize, 100), c.find("resp_1").?.size);
    try testing.expect(try c.put("", "resp_2", try sized(a, 40)));
    try testing.expect(c.find("resp_1") == null);
    try testing.expectEqualStrings("resp_2", c.find("resp_2").?.id);
    try testing.expectEqual(@as(usize, 40), global.load(.monotonic));
    try testing.expect(try c.put("side", "resp_3", try sized(a, 60)));
    try testing.expectEqualStrings("side", c.find("resp_3").?.lane);
    try testing.expectEqual(@as(usize, 100), c.bytes);
    c.evict("resp_2");
    c.evict("resp_missing");
    try testing.expect(c.find("resp_2") == null);
    try testing.expectEqual(@as(usize, 60), global.load(.monotonic));
    try testing.expect(try c.put("", "resp_4", try sized(a, 10)));
    c.deinit();
    try testing.expectEqual(@as(usize, 0), global.load(.monotonic));
}

test "ws_cache: entries are deep copies that outlive the caller's memory" {
    var global: std.atomic.Value(usize) = .init(0);
    var c: Cache = .{ .gpa = testing.allocator, .global = &global };
    defer c.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    const parsed = try json.parseText(arena.allocator(), "[{\"role\":\"user\",\"content\":\"hi\"}]");
    try testing.expect(try c.put("", "resp_1", parsed.ok.array));
    arena.deinit();
    const e = c.find("resp_1").?;
    try testing.expectEqualStrings("hi", e.messages[0].get("content").?.string);
    try testing.expectEqual(@as(usize, "[{\"role\": \"user\", \"content\": \"hi\"}]".len), e.size);
}

test "ws_cache: the connection budget evicts other lanes oldest first, else refuses" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var global: std.atomic.Value(usize) = .init(0);
    var c: Cache = .{ .gpa = testing.allocator, .global = &global, .connection_budget = 100 };
    defer c.deinit();
    try testing.expect(try c.put("a", "r_a", try sized(a, 30)));
    try testing.expect(try c.put("b", "r_b", try sized(a, 30)));
    try testing.expect(try c.put("c", "r_c", try sized(a, 30)));
    try testing.expect(try c.put("a", "r_a2", try sized(a, 30)));
    try testing.expect(try c.put("d", "r_d", try sized(a, 30)));
    try testing.expect(c.find("r_b") == null);
    try testing.expect(c.find("r_c") != null and c.find("r_a2") != null and c.find("r_d") != null);
    try testing.expect(try c.put("e", "r_e", try sized(a, 70)));
    try testing.expect(c.find("r_c") == null and c.find("r_a2") == null and c.find("r_d") != null);
    try testing.expect(!try c.put("d", "r_big", try sized(a, 101)));
    try testing.expect(c.find("r_d") == null and c.find("r_big") == null and c.find("r_e") != null);
    try testing.expectEqual(@as(usize, 70), c.bytes);
    try testing.expect(try c.put("f", "r_f", try sized(a, 40)));
    try testing.expect(c.find("r_e") == null and c.find("r_f") != null);
    try testing.expectEqual(@as(usize, 40), global.load(.monotonic));
}

test "ws_cache: the server-wide budget counts every socket's entries" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var global: std.atomic.Value(usize) = .init(0);
    var one: Cache = .{ .gpa = testing.allocator, .global = &global, .global_budget = 100 };
    var two: Cache = .{ .gpa = testing.allocator, .global = &global, .global_budget = 100 };
    try testing.expect(try one.put("", "r_1", try sized(a, 60)));
    try testing.expect(!try two.put("", "r_2", try sized(a, 50)));
    try testing.expect(try two.put("", "r_3", try sized(a, 40)));
    try testing.expectEqual(@as(usize, 100), global.load(.monotonic));
    try testing.expect(try two.put("x", "r_4", try sized(a, 40)));
    try testing.expect(two.find("r_3") == null and one.find("r_1") != null);
    one.deinit();
    try testing.expectEqual(@as(usize, 40), global.load(.monotonic));
    two.deinit();
    try testing.expectEqual(@as(usize, 0), global.load(.monotonic));
}
