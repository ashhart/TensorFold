//! A node's share of an EXLR8 release: halves [16 r / n, 16 (r + 1) / n) of every expert, and the record a pull leaves.

const std = @import("std");
const Io = std.Io;

pub const halves = 16;
/// The file a node pull writes beside the release's own files once every half and whole file has checked out.
pub const record_name = "exlr8-node.json";

/// Rank `rank` of `ranks` nodes; every node holds the same intermediate channels of every expert, shared included.
pub const Node = struct {
    rank: u8 = 0,
    ranks: u8 = 1,

    /// "R/N", with N dividing 16 so each node holds whole halves.
    pub fn parse(text: []const u8) !Node {
        const slash = std.mem.indexOfScalar(u8, text, '/') orelse return error.BadExlr8Node;
        const r = std.fmt.parseInt(u8, text[0..slash], 10) catch return error.BadExlr8Node;
        const n = std.fmt.parseInt(u8, text[slash + 1 ..], 10) catch return error.BadExlr8Node;
        return init(r, n);
    }

    pub fn init(rank: u8, ranks: u8) !Node {
        if (ranks == 0 or ranks > halves or halves % ranks != 0 or rank >= ranks) return error.BadExlr8Node;
        return .{ .rank = rank, .ranks = ranks };
    }

    pub fn first(n: Node) usize {
        return halves / @as(usize, n.ranks) * n.rank;
    }

    pub fn end(n: Node) usize {
        return n.first() + halves / @as(usize, n.ranks);
    }

    pub fn owns(n: Node, h: usize) bool {
        return h >= n.first() and h < n.end();
    }
};

/// What a node pull fetched and checked, as exlr8-node.json records it.
pub const Record = struct {
    repo: []const u8,
    revision: []const u8,
    set_id: []const u8,
    profile: []const u8,
    node: Node,
    /// Trellis bytes of the node's halves, every one equal to its xxhash64.
    expert_bytes: u64,
    /// Files fetched whole, every one equal to its published sha256.
    whole_files: u32,

    pub fn write(r: Record, io: Io, gpa: std.mem.Allocator, dir: []const u8) !void {
        var text: Io.Writer.Allocating = .init(gpa);
        defer text.deinit();
        try std.json.Stringify.value(.{
            .format = "exlr8-node-v1",
            .repo = r.repo,
            .revision = r.revision,
            .set_id = r.set_id,
            .profile = r.profile,
            .rank = r.node.rank,
            .ranks = r.node.ranks,
            .halves = [2]usize{ r.node.first(), r.node.end() },
            .expert_bytes = r.expert_bytes,
            .whole_files = r.whole_files,
            .checks = "every half's xxhash64 against the index; every whole file's sha256 against hashes/",
        }, .{ .whitespace = .indent_2 }, &text.writer);
        try text.writer.writeByte('\n');
        const path = try std.fs.path.join(gpa, &.{ dir, record_name });
        defer gpa.free(path);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = text.written() });
    }

    /// The record in `dir`, or error.NotAnExlr8Node; strings live in `arena`.
    pub fn read(io: Io, arena: std.mem.Allocator, dir: []const u8) !Record {
        const path = try std.fs.path.join(arena, &.{ dir, record_name });
        const text = Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(1 << 20)) catch |e| switch (e) {
            error.FileNotFound => return error.NotAnExlr8Node,
            else => return e,
        };
        return parse(arena, text);
    }

    pub fn parse(arena: std.mem.Allocator, text: []const u8) !Record {
        const v = std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{}) catch return error.NotAnExlr8Node;
        if (v != .object) return error.NotAnExlr8Node;
        const o = v.object;
        if (!std.mem.eql(u8, str(o, "format") orelse "", "exlr8-node-v1")) return error.NotAnExlr8Node;
        const rank = std.math.cast(u8, int(o, "rank") orelse return error.NotAnExlr8Node) orelse return error.NotAnExlr8Node;
        const ranks = std.math.cast(u8, int(o, "ranks") orelse return error.NotAnExlr8Node) orelse return error.NotAnExlr8Node;
        return .{
            .repo = str(o, "repo") orelse return error.NotAnExlr8Node,
            .revision = str(o, "revision") orelse return error.NotAnExlr8Node,
            .set_id = str(o, "set_id") orelse return error.NotAnExlr8Node,
            .profile = str(o, "profile") orelse return error.NotAnExlr8Node,
            .node = try Node.init(rank, ranks),
            .expert_bytes = int(o, "expert_bytes") orelse return error.NotAnExlr8Node,
            .whole_files = std.math.cast(u32, int(o, "whole_files") orelse return error.NotAnExlr8Node) orelse return error.NotAnExlr8Node,
        };
    }
};

fn str(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = o.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

fn int(o: std.json.ObjectMap, key: []const u8) ?u64 {
    const v = o.get(key) orelse return null;
    return if (v == .integer and v.integer >= 0) @intCast(v.integer) else null;
}

/// The release's weight set ID: sha256 over "<path> <size> <sha256>\n" of every weight file, sorted by path.
pub fn setId(lines: []const []const u8) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (lines) |l| {
        h.update(l);
        h.update("\n");
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    return d;
}

test "a node holds whole halves; only splits of 16 parse" {
    const n = try Node.parse("1/2");
    try std.testing.expectEqual(@as(usize, 8), n.first());
    try std.testing.expectEqual(@as(usize, 16), n.end());
    try std.testing.expect(n.owns(8) and n.owns(15) and !n.owns(7));
    const all = try Node.parse("0/1");
    try std.testing.expect(all.owns(0) and all.owns(15));
    try std.testing.expectEqual(@as(usize, 4), (try Node.parse("3/4")).end() - (try Node.parse("3/4")).first());
    for ([_][]const u8{ "2/2", "0/3", "0/0", "1", "a/2", "0/32" }) |t| try std.testing.expectError(error.BadExlr8Node, Node.parse(t));
}

test "a node record reads back what a pull wrote" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const want: Record = .{ .repo = "o/m", .revision = "abc", .set_id = "x", .profile = "home", .node = try Node.init(1, 2), .expert_bytes = 147 << 30, .whole_files = 170 };
    var text: Io.Writer.Allocating = .init(std.testing.allocator);
    defer text.deinit();
    try std.json.Stringify.value(.{ .format = "exlr8-node-v1", .repo = want.repo, .revision = want.revision, .set_id = want.set_id, .profile = want.profile, .rank = 1, .ranks = 2, .expert_bytes = want.expert_bytes, .whole_files = want.whole_files }, .{}, &text.writer);
    const got = try Record.parse(arena.allocator(), text.written());
    try std.testing.expectEqualStrings("home", got.profile);
    try std.testing.expectEqual(want.node, got.node);
    try std.testing.expectEqual(want.expert_bytes, got.expert_bytes);
    try std.testing.expectError(error.NotAnExlr8Node, Record.parse(arena.allocator(), "{\"format\":\"other\"}"));
}

test "the set ID hashes newline-ended lines in the order given" {
    const d = setId(&.{ "a 1 x", "b 2 y" });
    var want: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("a 1 x\nb 2 y\n", &want, .{});
    try std.testing.expectEqualSlices(u8, &want, &d);
}
