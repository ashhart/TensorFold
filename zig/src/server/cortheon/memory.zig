//! What a reply's client never saw (server tool rounds, Cortheon notes), kept to splice back in front of it next turn.
const std = @import("std");
const json = @import("json");
const Value = json.Value;
const Allocator = std.mem.Allocator;

const max_entries = 512;
const max_bytes: usize = 32 << 20;

pub const Store = struct {
    gpa: Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    map: std.AutoArrayHashMapUnmanaged(u64, []u8) = .empty, // insertion order is age order
    bytes: usize = 0,
    checked: std.AutoArrayHashMapUnmanaged(u64, void) = .empty, // user turns a Cortheon check already ran on

    pub fn deinit(s: *Store) void {
        for (s.map.values()) |v| s.gpa.free(v);
        s.map.deinit(s.gpa);
        s.checked.deinit(s.gpa);
    }

    /// Keeps `hidden` (a JSON array of messages) for the reply `key` names, oldest entries dropped past the caps.
    pub fn put(s: *Store, key: u64, hidden: []const u8) void {
        if (hidden.len > max_bytes / 4) return;
        const copy = s.gpa.dupe(u8, hidden) catch return;
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        if (s.map.fetchSwapRemove(key)) |old| {
            s.bytes -= old.value.len;
            s.gpa.free(old.value);
        }
        while (s.map.count() > 0 and (s.map.count() >= max_entries or s.bytes + copy.len > max_bytes)) {
            const oldest = s.map.values()[0];
            s.bytes -= oldest.len;
            s.gpa.free(oldest);
            s.map.orderedRemoveAt(0);
        }
        s.map.put(s.gpa, key, copy) catch {
            s.gpa.free(copy);
            return;
        };
        s.bytes += copy.len;
    }

    /// The hidden messages kept for `key`, copied into `a`.
    pub fn get(s: *Store, a: Allocator, key: u64) ?[]const u8 {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        const v = s.map.get(key) orelse return null;
        return a.dupe(u8, v) catch null;
    }

    /// True the first time a check runs for the user turn `key`; false after.
    pub fn firstCheck(s: *Store, key: u64) bool {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        if (s.checked.contains(key)) return false;
        if (s.checked.count() >= 4 * max_entries) s.checked.orderedRemoveAt(0);
        s.checked.put(s.gpa, key, {}) catch return false;
        return true;
    }
};

fn role(m: Value) []const u8 {
    return m.strField("role") orelse "";
}

/// A message's words: its content string, or its text parts joined with nothing between them.
pub fn text(a: Allocator, m: Value) Allocator.Error![]const u8 {
    const c = m.get("content") orelse return "";
    if (c == .string) return c.string;
    if (c != .array) return "";
    var buf: std.ArrayList(u8) = .empty;
    var one: ?[]const u8 = null;
    for (c.array) |part| {
        if (part != .object) continue;
        const kind = part.get("type") orelse continue;
        if (kind != .string or !std.mem.eql(u8, kind.string, "text")) continue;
        const t = part.get("text") orelse continue;
        if (t != .string) continue;
        if (one == null and buf.items.len == 0) {
            one = t.string;
            continue;
        }
        if (buf.items.len == 0) if (one) |s| try buf.appendSlice(a, s);
        one = null;
        try buf.appendSlice(a, t.string);
    }
    return if (buf.items.len > 0) buf.items else one orelse "";
}

/// Folds `s` into `h` with runs of whitespace as one space and the ends trimmed, as clients reshape whitespace.
fn foldText(h: *std.hash.Wyhash, s: []const u8) void {
    var space = false;
    var started = false;
    for (s) |ch| {
        if (std.ascii.isWhitespace(ch)) {
            space = started;
            continue;
        }
        if (space) h.update(" ");
        space = false;
        started = true;
        h.update(&.{ch});
    }
    h.update("\x00");
}

/// A call's arguments in one canonical form: parsed and written compactly, else the raw text.
fn foldArguments(a: Allocator, h: *std.hash.Wyhash, args: ?Value) void {
    const v = args orelse return h.update("\x00");
    const parsed: Value = if (v == .string) switch (json.parseText(a, v.string) catch return h.update(v.string)) {
        .ok => |x| x,
        .err => return h.update(v.string),
    } else v;
    const canon = json.stringify(a, sorted(a, parsed) catch return h.update("\x00"), .{ .ascii = false, .compact = true }) catch return h.update("\x00");
    h.update(canon);
    h.update("\x00");
}

/// `v` with every object's keys in byte order, so key order never changes a key.
fn sorted(a: Allocator, v: Value) Allocator.Error!Value {
    switch (v) {
        .array => |items| {
            const out = try a.alloc(Value, items.len);
            for (items, out) |x, *slot| slot.* = try sorted(a, x);
            return .{ .array = out };
        },
        .object => |o| {
            const keys = try a.dupe([]const u8, o.keys());
            std.mem.sort([]const u8, keys, {}, struct {
                fn less(_: void, x: []const u8, y: []const u8) bool {
                    return std.mem.order(u8, x, y) == .lt;
                }
            }.less);
            const out = try json.newObject(a);
            for (keys) |k| try out.put(a, k, try sorted(a, o.get(k).?));
            return .{ .object = out };
        },
        else => return v,
    }
}

/// The key of the assistant reply at `i`: the user turn before it, its prose, and its first call.
pub fn replyKey(a: Allocator, messages: []const Value, i: usize) Allocator.Error!u64 {
    var h: std.hash.Wyhash = .init(0x636f727468656f6e);
    var user: []const u8 = "";
    var j = i;
    while (j > 0) {
        j -= 1;
        if (std.mem.eql(u8, role(messages[j]), "user")) {
            user = try text(a, messages[j]);
            break;
        }
    }
    foldText(&h, user);
    foldText(&h, try text(a, messages[i]));
    if (messages[i].get("tool_calls")) |calls| if (calls == .array and calls.array.len > 0) {
        const f = calls.array[0].get("function") orelse Value.null;
        h.update(f.strField("name") orelse "");
        h.update("\x00");
        foldArguments(a, &h, f.get("arguments"));
    };
    return h.final();
}

/// The key of the latest user turn: its text and position, so a check runs once per turn.
pub fn userKey(a: Allocator, messages: []const Value) Allocator.Error!u64 {
    var h: std.hash.Wyhash = .init(0x75736572);
    var n: usize = 0;
    for (messages) |m| if (std.mem.eql(u8, role(m), "user")) {
        n += 1;
    };
    var j = messages.len;
    while (j > 0) {
        j -= 1;
        if (std.mem.eql(u8, role(messages[j]), "user")) {
            foldText(&h, try text(a, messages[j]));
            break;
        }
    }
    h.update(std.mem.asBytes(&n));
    return h.final();
}

/// `messages` with every earlier reply's hidden messages back in front of it.
pub fn splice(a: Allocator, store: *Store, messages: []const Value) Allocator.Error![]const Value {
    var out: std.ArrayList(Value) = .empty;
    var found = false;
    for (messages, 0..) |m, i| {
        if (std.mem.eql(u8, role(m), "assistant")) if (store.get(a, try replyKey(a, messages, i))) |stored| {
            switch (json.parseText(a, stored) catch return error.OutOfMemory) {
                .ok => |v| if (v == .array) {
                    if (!found) try out.appendSlice(a, messages[0..i]);
                    found = true;
                    try out.appendSlice(a, v.array);
                },
                .err => {},
            }
        };
        if (found) try out.append(a, m);
    }
    return if (found) out.items else messages;
}

fn msg(a: Allocator, r: []const u8, content: []const u8) !Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = r });
    try o.put(a, "content", .{ .string = content });
    return .{ .object = o };
}

test "a kept exchange goes back in front of its reply, whitespace and argument order aside" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var store: Store = .{ .gpa = std.testing.allocator, .io = std.testing.io };
    defer store.deinit();
    const turn = [_]Value{ try msg(a, "user", "What changed in Zig 0.17?"), try msg(a, "assistant", "It added  std.Io.\n") };
    store.put(try replyKey(a, &turn, 1), "[{\"role\": \"tool\", \"content\": \"notes\"}]");
    const later = [_]Value{ try msg(a, "system", "s"), try msg(a, "user", "What changed in Zig 0.17?"), try msg(a, "assistant", " It added std.Io."), try msg(a, "user", "And 0.16?") };
    const out = try splice(a, &store, &later);
    try std.testing.expectEqual(@as(usize, 5), out.len);
    try std.testing.expectEqualStrings("tool", out[2].strField("role").?);
    try std.testing.expectEqualStrings("assistant", out[3].strField("role").?);
    const other = [_]Value{ try msg(a, "user", "Different question"), try msg(a, "assistant", "It added std.Io.") };
    try std.testing.expectEqual(@as(usize, 2), (try splice(a, &store, &other)).len);
    const call_a = (try json.parseText(a, "{\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [{\"id\": \"c1\", \"function\": {\"name\": \"bash\", \"arguments\": \"{\\\"b\\\": 1, \\\"a\\\": 2}\"}}]}")).ok;
    const call_b = (try json.parseText(a, "{\"role\": \"assistant\", \"content\": null, \"tool_calls\": [{\"id\": \"zz\", \"function\": {\"name\": \"bash\", \"arguments\": {\"a\": 2, \"b\": 1}}}]}")).ok;
    const u = try msg(a, "user", "run it");
    try std.testing.expectEqual(try replyKey(a, &.{ u, call_a }, 1), try replyKey(a, &.{ u, call_b }, 1));
}

fn textPart(a: Allocator, words: []const u8) !Value {
    const p = try json.newObject(a);
    try p.put(a, "type", .{ .string = "text" });
    try p.put(a, "text", .{ .string = words });
    return .{ .object = p };
}

fn imagePart(a: Allocator) !Value {
    const url = try json.newObject(a);
    try url.put(a, "url", .{ .string = "https://images.example/pic.png" });
    const p = try json.newObject(a);
    try p.put(a, "type", .{ .string = "image_url" });
    try p.put(a, "image_url", .{ .object = url });
    return .{ .object = p };
}

fn contentMessage(a: Allocator, r: []const u8, parts: []const Value) !Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = r });
    const owned = try a.alloc(Value, parts.len);
    @memcpy(owned, parts);
    try o.put(a, "content", .{ .array = owned });
    return .{ .object = o };
}

test "text joins well-formed text parts in order with nothing between them" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const message = try contentMessage(a, "user", &.{
        try textPart(a, "PART-ONE-"),
        try textPart(a, "PART-TWO-"),
    });
    try std.testing.expectEqualStrings("PART-ONE-PART-TWO-", try text(a, message));
}

test "an image part adds no words beside the text parts" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const message = try contentMessage(a, "user", &.{
        try textPart(a, "PART-ONE-"),
        try imagePart(a),
        try textPart(a, "PART-TWO-"),
    });
    const words = try text(a, message);
    try std.testing.expectEqualStrings("PART-ONE-PART-TWO-", words);
    try std.testing.expect(std.mem.indexOf(u8, words, "https://images.example/pic.png") == null);
}

test "a content-part user turn keys the same reply and check as its joined string" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const words = "What changed in Zig 0.17?";
    const parts = try contentMessage(a, "user", &.{
        try textPart(a, "What changed in "),
        try textPart(a, "Zig 0.17?"),
    });
    const plain = try msg(a, "user", words);
    const assistant = try msg(a, "assistant", "It added std.Io.");
    try std.testing.expectEqual(try replyKey(a, &.{ plain, assistant }, 1), try replyKey(a, &.{ parts, assistant }, 1));
    try std.testing.expectEqual(try userKey(a, &.{plain}), try userKey(a, &.{parts}));
}

test "the store drops its oldest entries and checks run once a turn" {
    var store: Store = .{ .gpa = std.testing.allocator, .io = std.testing.io };
    defer store.deinit();
    for (0..max_entries + 10) |k| store.put(k, "[]");
    try std.testing.expect(store.map.count() <= max_entries);
    try std.testing.expect(store.get(std.testing.allocator, 0) == null);
    try std.testing.expect(store.firstCheck(7));
    try std.testing.expect(!store.firstCheck(7));
}
