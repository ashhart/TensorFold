//! Conversations a lesson's questions are asked after (slide_context.zig), written once and kept in the fact graph.
const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;

/// A user's line and the model's reply to it.
pub const Turn = [2][]const u8;

/// A kept conversation: its name and its turns.
pub const Talk = struct { name: []const u8, turns: []const Turn };

/// The conversations kept, their text held by the graph's arena.
pub const Talks = struct {
    list: std.ArrayList(Talk) = .empty,

    pub fn deinit(t: *Talks, gpa: Allocator) void {
        t.list.deinit(gpa);
    }

    /// Conversation `name` copied into `a`; null when none is kept.
    pub fn find(t: *const Talks, a: Allocator, name: []const u8) !?[]const Turn {
        for (t.list.items) |x| if (std.mem.eql(u8, x.name, name)) {
            const out = try a.alloc(Turn, x.turns.len);
            for (out, x.turns) |*o, turn| o.* = .{ try a.dupe(u8, turn[0]), try a.dupe(u8, turn[1]) };
            return out;
        };
        return null;
    }

    /// `turns` kept as conversation `name`, copied into `arena`.
    pub fn keep(t: *Talks, gpa: Allocator, arena: Allocator, name: []const u8, turns: []const Turn) !void {
        const out = try arena.alloc(Turn, turns.len);
        for (out, turns) |*o, x| o.* = .{ try arena.dupe(u8, x[0]), try arena.dupe(u8, x[1]) };
        try t.list.append(gpa, .{ .name = try arena.dupe(u8, name), .turns = out });
    }

    /// As the graph file holds them: [{"name", "turns": [[line, reply], ...]}].
    pub fn value(t: *const Talks, a: Allocator) !json.Value {
        const out = try a.alloc(json.Value, t.list.items.len);
        for (t.list.items, out) |x, *v| {
            const o = try json.newObject(a);
            try o.put(a, "name", .{ .string = x.name });
            const turns = try a.alloc(json.Value, x.turns.len);
            for (x.turns, turns) |turn, *tv| tv.* = try pair(a, turn);
            try o.put(a, "turns", .{ .array = turns });
            v.* = .{ .object = o };
        }
        return .{ .array = out };
    }

    /// The graph file's `talks`, their strings borrowed from where it was parsed (`a`).
    pub fn read(t: *Talks, gpa: Allocator, a: Allocator, v: ?json.Value) !void {
        const xs = (v orelse return).items() orelse return;
        for (xs) |x| {
            var turns: std.ArrayList(Turn) = .empty;
            for ((x.field("turns") orelse continue).items() orelse continue) |turn| if (turn.items()) |p| {
                if (p.len == 2) try turns.append(a, .{ p[0].str() orelse "", p[1].str() orelse "" });
            };
            try t.list.append(gpa, .{ .name = x.strField("name") orelse continue, .turns = turns.items });
        }
    }
};

/// A line and its reply as a JSON pair.
pub fn pair(a: Allocator, turn: Turn) !json.Value {
    const out = try a.alloc(json.Value, 2);
    out[0] = .{ .string = turn[0] };
    out[1] = .{ .string = turn[1] };
    return .{ .array = out };
}

test "conversations and a fact's chain turn persist across a reopen of the graph" {
    const Graph = @import("slide_graph.zig").Graph;
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fs.path.join(gpa, &.{ ".zig-cache", "tmp", &tmp.sub_path, "graph.json" });
    defer gpa.free(path);
    var g = Graph.init(gpa, io, path);
    const id = g.add("My cat is called Pip.", "chat", 100);
    g.mark(id, .learned);
    g.chained(id, .{ "What is my cat called?", "Your cat is called Pip." });
    g.keepTalk("check", &.{ .{ "Hi there.", "Hello! How can I help?" }, .{ "Tell me a joke.", "Why did the chicken cross the road?" } });
    g.deinit();

    var again = Graph.init(gpa, io, path);
    defer again.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const talk = (try again.talk(a, "check")).?;
    try std.testing.expectEqual(@as(usize, 2), talk.len);
    try std.testing.expectEqualStrings("Why did the chicken cross the road?", talk[1][1]);
    try std.testing.expect(try again.talk(a, "task") == null);
    const chain = try again.chains(a, 2);
    try std.testing.expectEqual(@as(usize, 1), chain.len);
    try std.testing.expectEqualStrings("Your cat is called Pip.", chain[0][1]);
}
