//! Conversation notes and the fact graph are private files: owner-only, and never written through a planted link.
const std = @import("std");
const platform = @import("sys");
const json = @import("json");
const compact_mem = @import("compact_mem.zig");
const server = @import("server.zig");
const Graph = @import("slide_graph.zig").Graph;
const testing = std.testing;
const io = testing.io;

fn absolute(a: std.mem.Allocator, tmp: testing.TmpDir, name: []const u8) ![]u8 {
    const cwd = try std.process.currentPathAlloc(io, a);
    return std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}/{s}", .{ cwd, tmp.sub_path, name });
}

/// The file's mode bits, and that it is a plain file a planted link did not redirect.
fn modeOf(path: []const u8) !u32 {
    const st = try std.Io.Dir.cwd().statFile(io, path, .{ .follow_symlinks = false });
    try testing.expectEqual(std.Io.File.Kind.file, st.kind);
    return @intCast(st.permissions.toMode() & 0o777);
}

test "security: a conversation note is owner-only and is not written through a planted link" {
    if (platform.windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try absolute(a, tmp, "notes");
    const victim = try absolute(a, tmp, "victim.txt");
    try tmp.dir.writeFile(io, .{ .sub_path = "victim.txt", .data = "untouched" });

    var srv: server.Server = undefined;
    srv.config = .{ .served_name = "test", .model_ids = &.{"test"}, .compact_memory = dir };
    srv.io = io;
    const msgs = (try json.parse(a, "[{\"role\":\"user\",\"content\":\"hello\"}]")).ok.items().?;
    const key = try compact_mem.conversationKey(a, msgs);

    // the names the old writer used for its temporaries, planted as links to another file
    try tmp.dir.createDirPath(io, "notes");
    for ([_][]const u8{ ".md.tmp", ".json.tmp" }) |ext| {
        const link = try std.fmt.allocPrint(a, "notes/{s}{s}", .{ key, ext });
        try tmp.dir.symLink(io, victim, link, .{});
    }
    compact_mem.storeMem(&srv, a, msgs, 1, "a private summary", 0);

    try testing.expectEqualStrings("untouched", try tmp.dir.readFileAlloc(io, "victim.txt", a, .limited(64)));
    for ([_][]const u8{ ".md", ".json" }) |ext| {
        const path = try std.fmt.allocPrint(a, "{s}/{s}{s}", .{ dir, key, ext });
        try testing.expectEqual(@as(u32, 0o600), try modeOf(path));
    }
    const note = try std.fmt.allocPrint(a, "notes/{s}.md", .{key});
    try testing.expectEqualStrings("a private summary", try tmp.dir.readFileAlloc(io, note, a, .limited(64)));
    try testing.expectEqualStrings("a private summary", compact_mem.loadMem(&srv, a, msgs, 0).?.note);
    try testing.expect(compact_mem.loadMem(&srv, a, msgs, 0x5a17) == null); // a salted request reads no shared note
    const other = (try json.parse(a, "[{\"role\":\"user\",\"content\":\"salted\"}]")).ok.items().?;
    compact_mem.storeMem(&srv, a, other, 1, "a salted summary", 0x5a17); // nor writes one another salt would read
    const salted = try std.fmt.allocPrint(a, "notes/{s}.md", .{try compact_mem.conversationKey(a, other)});
    try testing.expectError(error.FileNotFound, tmp.dir.readFileAlloc(io, salted, a, .limited(64)));
}

test "security: a folder the note writer makes is the owner's alone" {
    if (platform.windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try absolute(a, tmp, "fresh/notes");
    var srv: server.Server = undefined;
    srv.config = .{ .served_name = "test", .model_ids = &.{"test"}, .compact_memory = dir };
    srv.io = io;
    const msgs = (try json.parse(a, "[{\"role\":\"user\",\"content\":\"hello\"}]")).ok.items().?;
    compact_mem.storeMem(&srv, a, msgs, 1, "note", 0);
    const st = try std.Io.Dir.cwd().statFile(io, dir, .{});
    try testing.expectEqual(@as(u32, 0), @as(u32, @intCast(st.permissions.toMode() & 0o077)));
}

test "security: the fact graph is owner-only and is not written through a planted link" {
    if (platform.windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try absolute(a, tmp, "graph.json");
    const victim = try absolute(a, tmp, "victim.txt");
    try tmp.dir.writeFile(io, .{ .sub_path = "victim.txt", .data = "untouched" });
    try tmp.dir.symLink(io, victim, "graph.json.tmp", .{});

    var g = Graph.init(testing.allocator, io, path);
    _ = g.add("My sister is called Ana.", "chat", 100);
    try g.persist();
    g.deinit();

    try testing.expectEqualStrings("untouched", try tmp.dir.readFileAlloc(io, "victim.txt", a, .limited(64)));
    try testing.expectEqual(@as(u32, 0o600), try modeOf(path));
    var again = Graph.init(testing.allocator, io, path);
    defer again.deinit();
    try testing.expectEqual(@as(usize, 1), again.facts.items.len);
}
