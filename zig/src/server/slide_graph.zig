//! The Sliding Weights fact graph: what the learner was taught and whether the weights recall it; never the weights.
const std = @import("std");
const json = @import("json.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const State = enum { queued, learning, learned, missed };

pub const Fact = struct { id: u32, text: []const u8, source: []const u8, at: i64, state: State };

/// Words of five letters or more that say nothing about a fact, so they never link two facts.
const common = [_][]const u8{ "about", "after", "again", "being", "could", "every", "other", "should", "still", "their", "there", "these", "those", "under", "until", "where", "which", "while", "would" };

/// Facts in teaching order, saved to `path` after every change (null: memory only); ids are never reused.
pub const Graph = struct {
    gpa: Allocator,
    io: Io,
    path: ?[]const u8,
    mutex: Io.Mutex = .init,
    arena: std.heap.ArenaAllocator,
    facts: std.ArrayList(Fact) = .empty,
    next: u32 = 1,

    /// A graph that starts from `path`'s file when it reads, else empty.
    pub fn init(gpa: Allocator, io: Io, path: ?[]const u8) Graph {
        var g: Graph = .{ .gpa = gpa, .io = io, .path = path, .arena = .init(gpa) };
        g.load() catch {};
        return g;
    }

    pub fn deinit(g: *Graph) void {
        g.facts.deinit(g.gpa);
        g.arena.deinit();
    }

    /// A newly taught fact, queued; returns its id (0 when memory ran out).
    pub fn add(g: *Graph, text: []const u8, source: []const u8, at: i64) u32 {
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        const a = g.arena.allocator();
        const fact: Fact = .{ .id = g.next, .text = a.dupe(u8, text) catch return 0, .source = a.dupe(u8, source) catch return 0, .at = at, .state = .queued };
        g.facts.append(g.gpa, fact) catch return 0;
        g.next += 1;
        g.save();
        return fact.id;
    }

    pub fn mark(g: *Graph, id: u32, state: State) void {
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        for (g.facts.items) |*f| if (f.id == id) {
            f.state = state;
            break;
        };
        g.save();
    }

    /// Forget every fact and remove the file; the weights keep what they learned.
    pub fn clear(g: *Graph) void {
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        g.facts.clearRetainingCapacity();
        if (g.path) |p| Io.Dir.cwd().deleteFile(g.io, p) catch {};
    }

    /// {"next": N, "facts": [{"id", "text", "source", "at", "state", "links"}]}; links join facts sharing a key word.
    pub fn render(g: *Graph, a: Allocator) ![]u8 {
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        return g.document(a);
    }

    fn document(g: *Graph, a: Allocator) ![]u8 {
        const facts = try a.alloc(json.Value, g.facts.items.len);
        for (g.facts.items, facts) |f, *out| {
            var links: std.ArrayList(json.Value) = .empty;
            for (g.facts.items) |other| if (other.id != f.id and shareWord(f.text, other.text)) try links.append(a, try json.intValue(a, other.id));
            const o = try json.newObject(a);
            try o.put(a, "id", try json.intValue(a, f.id));
            try o.put(a, "text", .{ .string = f.text });
            try o.put(a, "source", .{ .string = f.source });
            try o.put(a, "at", try json.intValue(a, f.at));
            try o.put(a, "state", .{ .string = @tagName(f.state) });
            try o.put(a, "links", .{ .array = links.items });
            out.* = .{ .object = o };
        }
        const root = try json.newObject(a);
        try root.put(a, "next", try json.intValue(a, g.next));
        try root.put(a, "facts", .{ .array = facts });
        return json.stringify(a, .{ .object = root }, .{ .compact = true });
    }

    /// Write the file through a renamed temporary; a failed write keeps the old file (called under the lock).
    fn save(g: *Graph) void {
        const path = g.path orelse return;
        var scratch = std.heap.ArenaAllocator.init(g.gpa);
        defer scratch.deinit();
        const a = scratch.allocator();
        const body = g.document(a) catch return;
        const tmp = std.fmt.allocPrint(a, "{s}.tmp", .{path}) catch return;
        if (std.fs.path.dirname(path)) |dir| Io.Dir.cwd().createDirPath(g.io, dir) catch return;
        Io.Dir.cwd().writeFile(g.io, .{ .sub_path = tmp, .data = body }) catch return;
        Io.Dir.cwd().rename(tmp, Io.Dir.cwd(), path, g.io) catch return;
    }

    fn load(g: *Graph) !void {
        const path = g.path orelse return;
        const a = g.arena.allocator();
        const bytes = try Io.Dir.cwd().readFileAlloc(g.io, path, a, .limited(1 << 26));
        const root = switch (try json.parse(a, bytes)) {
            .ok => |v| v,
            .err => return error.BadGraph,
        };
        for ((root.field("facts") orelse return error.BadGraph).items() orelse return error.BadGraph) |f| {
            const state = std.meta.stringToEnum(State, f.strField("state") orelse "") orelse .queued;
            try g.facts.append(g.gpa, .{ .id = try int(u32, f.field("id")), .text = f.strField("text") orelse "", .source = f.strField("source") orelse "", .at = try int(i64, f.field("at")), .state = state });
        }
        g.next = try int(u32, root.field("next"));
    }
};

fn int(comptime T: type, v: ?json.Value) !T {
    const x = v orelse return error.BadGraph;
    if (x != .int) return error.BadGraph;
    return std.fmt.parseInt(T, x.int, 10);
}

/// Whether two facts share a word of five or more letters that is not on the common list, ignoring case.
fn shareWord(x: []const u8, y: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, x, " \t\n.,;:!?()[]\"'");
    while (it.next()) |w| {
        if (w.len < 5 or isCommon(w)) continue;
        var other = std.mem.tokenizeAny(u8, y, " \t\n.,;:!?()[]\"'");
        while (other.next()) |v| if (std.ascii.eqlIgnoreCase(w, v)) return true;
    }
    return false;
}

fn isCommon(w: []const u8) bool {
    for (common) |c| if (std.ascii.eqlIgnoreCase(w, c)) return true;
    return false;
}

test "facts persist across a reopen, link by a shared key word, and clear without reusing ids" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fs.path.join(gpa, &.{ ".zig-cache", "tmp", &tmp.sub_path, "graph.json" });
    defer gpa.free(path);
    var g = Graph.init(gpa, io, path);
    const a = g.add("My company, Larkspur Robotics, sells the Heron X2.", "file:company.md", 100);
    const b = g.add("Larkspur Robotics made 4.7 million pounds in Q3.", "file:company.md", 101);
    const c = g.add("My favourite colour is teal.", "chat", 102);
    g.mark(a, .learned);
    g.mark(c, .missed);
    g.deinit();

    var again = Graph.init(gpa, io, path);
    defer again.deinit();
    try std.testing.expectEqual(@as(usize, 3), again.facts.items.len);
    try std.testing.expectEqual(State.learned, again.facts.items[0].state);
    try std.testing.expectEqual(State.missed, again.facts.items[2].state);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = switch (try json.parse(arena.allocator(), try again.render(arena.allocator()))) {
        .ok => |v| v,
        .err => return error.BadGraph,
    };
    const facts = doc.field("facts").?.items().?;
    try std.testing.expectEqual(@as(usize, 1), facts[0].field("links").?.items().?.len);
    try std.testing.expectEqualStrings(try std.fmt.allocPrint(arena.allocator(), "{d}", .{b}), facts[0].field("links").?.items().?[0].int);
    try std.testing.expectEqual(@as(usize, 0), facts[2].field("links").?.items().?.len);
    again.clear();
    try std.testing.expectEqual(c + 1, again.add("My dog is called Pickle.", "chat", 103));
}
