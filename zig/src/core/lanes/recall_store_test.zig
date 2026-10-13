//! The store on disk: reloads, refusal by name, torn tails, bad ids, the cap, dedup, read-only, emptying.
const std = @import("std");
const store = @import("recall_store.zig");
const rr = @import("recall_record.zig");
const Store = store.Store;
const gpa = std.testing.allocator;

const vocab: u32 = 5000;

fn identity(seed: u8, name: []const u8) store.Identity {
    return .{ .digest = @splat(seed), .vocab = vocab, .name = name };
}

/// A fresh directory under the test's tmp dir, as a path the store opens.
const Dir = struct {
    tmp: std.testing.TmpDir,
    path: []u8,

    fn init() !Dir {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        return .{ .tmp = tmp, .path = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/recall", .{tmp.sub_path}) };
    }

    fn deinit(d: *Dir) void {
        gpa.free(d.path);
        d.tmp.cleanup();
    }
};

fn open(d: *const Dir, id: store.Identity, o: struct { cap: u64 = 64 << 20, read_only: bool = false }) !*Store {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var problem: []const u8 = "";
    const opts: store.Options = .{ .dir = d.path, .cap_bytes = o.cap, .read_only = o.read_only };
    return (try Store.open(gpa, std.testing.io, id, opts, &problem, arena.allocator())) orelse error.TestRefused;
}

fn ids(n: usize, seed: u64) ![]u32 {
    const out = try gpa.alloc(u32, n);
    var rng = std.Random.DefaultPrng.init(seed);
    for (out) |*t| t.* = rng.random().uintLessThan(u32, vocab);
    return out;
}

test "a record reloads to the same lookups, and a request read again adds nothing" {
    var d = try Dir.init();
    defer d.deinit();
    const prompt = try ids(300, 1);
    defer gpa.free(prompt);
    const reply = try ids(200, 2);
    defer gpa.free(reply);
    var held: u32 = 0;
    {
        const s = try open(&d, identity(1, "model-a"), .{});
        defer s.close();
        s.readPrompt(prompt, 0);
        s.readReply(prompt, reply);
        held = s.shared.record.held();
        try std.testing.expectEqual(@as(u64, 300 + 64 + 200), s.added);
        s.readPrompt(prompt, 0); // the same request again: nothing new
        s.readReply(prompt, reply);
        try std.testing.expectEqual(@as(u64, 564), s.added);
        try std.testing.expectEqual(@as(u64, 564), s.held);
    }
    const s = try open(&d, identity(1, "model-b"), .{}); // another model with the same tokenizer shares it
    defer s.close();
    try std.testing.expectEqual(held, s.shared.record.held());
    const m = s.shared.record.longest(prompt);
    try std.testing.expectEqual(@as(u32, rr.max_match), m.len);
    var out: [4]u32 = undefined;
    try std.testing.expectEqualSlices(u32, reply[0..4], s.shared.record.copy(m.at, &out));
}

test "a growing conversation adds each turn's new ids and the overlaps around them" {
    var d = try Dir.init();
    defer d.deinit();
    const s = try open(&d, identity(2, "m"), .{});
    defer s.close();
    const turn1 = try ids(400, 3);
    defer gpa.free(turn1);
    s.readPrompt(turn1, 0);
    const new2 = try ids(100, 4);
    defer gpa.free(new2);
    const turn2 = try std.mem.concat(gpa, u32, &.{ turn1, new2 });
    defer gpa.free(turn2);
    s.readPrompt(turn2, 0); // no resume point, as after a restart: the record still holds the first 400
    try std.testing.expectEqual(@as(u64, 400 + 64 + 100), s.added);
    const new3 = try ids(50, 5);
    defer gpa.free(new3);
    const turn3 = try std.mem.concat(gpa, u32, &.{ turn2, new3 });
    defer gpa.free(turn3);
    s.readPrompt(turn3, @intCast(turn2.len)); // resumed at the second turn's end: the overlap and the new 50
    try std.testing.expectEqual(@as(u64, 564 + 64 + 50), s.added);
}

test "a record written under another tokenizer is refused by name" {
    var d = try Dir.init();
    defer d.deinit();
    (try open(&d, identity(3, "Writer-7B"), .{})).close();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var problem: []const u8 = "";
    const opts: store.Options = .{ .dir = d.path, .cap_bytes = 64 << 20 };
    const reader = try Store.open(gpa, std.testing.io, identity(4, "Reader-27B"), opts, &problem, arena.allocator());
    try std.testing.expect(reader == null);
    try std.testing.expect(std.mem.indexOf(u8, problem, "Writer-7B") != null);
    try std.testing.expect(std.mem.indexOf(u8, problem, "Reader-27B") != null);
}

test "a torn tail and an id past the vocabulary are cut at reload, and writing goes on after them" {
    var d = try Dir.init();
    defer d.deinit();
    const a = try ids(100, 6);
    defer gpa.free(a);
    {
        const s = try open(&d, identity(5, "m"), .{});
        defer s.close();
        s.readPrompt(a, 0);
    }
    var buf: [1100]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buf, "{s}/{x:0>16}.ids", .{ d.path, @as(u64, 0) }, 0);
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .APPEND = true }, @as(std.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    const torn = [_]u32{ 7, 8, 9 }; // a segment whose mark never landed
    _ = std.c.write(fd, std.mem.sliceAsBytes(&torn).ptr, 12);
    _ = std.c.close(fd);
    const b = try ids(80, 7);
    defer gpa.free(b);
    {
        const s = try open(&d, identity(5, "m"), .{});
        defer s.close();
        try std.testing.expectEqual(@as(u32, 101), s.shared.record.held());
        s.readPrompt(b, 0);
    }
    const s = try open(&d, identity(5, "m"), .{});
    defer s.close();
    try std.testing.expectEqual(@as(u32, 101 + 81), s.shared.record.held());
    try std.testing.expect(s.shared.record.longest(b[0..40]).len > 0);
    const bad = [_]u32{ 1, 2, vocab + 3, rr.mark };
    const fd2 = std.c.open(path, .{ .ACCMODE = .WRONLY, .APPEND = true }, @as(std.c.mode_t, 0));
    _ = std.c.write(fd2, std.mem.sliceAsBytes(&bad).ptr, 16);
    _ = std.c.close(fd2);
    const again = try open(&d, identity(5, "m"), .{ .read_only = true });
    defer again.close();
    try std.testing.expectEqual(@as(u32, 182), again.shared.record.held());
}

test "past the cap the oldest chunk goes, file and all" {
    var d = try Dir.init();
    defer d.deinit();
    const s = try open(&d, identity(6, "m"), .{ .cap = 2 * store.chunk_bytes });
    defer s.close();
    const first = try ids(1000, 8);
    defer gpa.free(first);
    s.readPrompt(first, 0);
    const bulk = try ids(rr.chunk_tokens * 2 + 1000, 9);
    defer gpa.free(bulk);
    s.readPrompt(bulk, 0);
    try std.testing.expectEqual(@as(usize, 2), s.shared.record.chunks.items.len);
    try std.testing.expectEqual(@as(u32, 0), s.shared.record.longest(first[0..20]).len);
    var buf: [1100]u8 = undefined;
    const oldest = try std.fmt.bufPrintSentinel(&buf, "{s}/{x:0>16}.ids", .{ d.path, @as(u64, 0) }, 0);
    try std.testing.expect(std.c.access(oldest, 0) != 0);
    try std.testing.expectEqual(@as(u32, rr.max_match), s.shared.record.longest(bulk[0 .. bulk.len - 1]).len);
}

test "read-only writes nothing and cannot empty; a second writer reads; empty starts over" {
    var d = try Dir.init();
    defer d.deinit();
    const a = try ids(100, 10);
    defer gpa.free(a);
    const w = try open(&d, identity(7, "m"), .{});
    w.readPrompt(a, 0);
    const second = try open(&d, identity(7, "m"), .{}); // the lock is held: it reads only
    try std.testing.expect(!second.writes);
    second.readPrompt(a, 0);
    try std.testing.expectEqual(@as(u64, 0), second.added);
    second.close();
    w.close();
    const r = try open(&d, identity(7, "m"), .{ .read_only = true });
    const b = try ids(100, 11);
    defer gpa.free(b);
    r.readPrompt(b, 0);
    try std.testing.expectEqual(@as(u64, 0), r.added);
    try std.testing.expectError(error.ReadOnly, r.empty());
    r.close();
    const e = try open(&d, identity(7, "m"), .{});
    try std.testing.expectEqual(@as(u32, 101), e.shared.record.held());
    try e.empty();
    try std.testing.expectEqual(@as(u32, 0), e.shared.record.held());
    e.readPrompt(b, 0);
    e.close();
    const after = try open(&d, identity(7, "m"), .{});
    defer after.close();
    try std.testing.expectEqual(@as(u32, 101), after.shared.record.held());
    try std.testing.expect(after.shared.record.longest(b[0..30]).len > 0);
    try std.testing.expectEqual(@as(u32, 0), after.shared.record.longest(a[0..30]).len);
}

test "an empty that cannot remove a chunk file says which, keeps nothing in memory, adds nothing until one succeeds" {
    var d = try Dir.init();
    defer d.deinit();
    const s = try open(&d, identity(8, "m"), .{});
    defer s.close();
    const a = try ids(300, 12);
    defer gpa.free(a);
    s.readPrompt(a, 0);
    var buf: [1100]u8 = undefined;
    const stuck = try std.fmt.bufPrintSentinel(&buf, "{s}/{x:0>16}.ids", .{ d.path, @as(u64, 0xff) }, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.mkdir(stuck, 0o700)); // a directory: no unlink removes it
    var inner: [1200]u8 = undefined;
    const keep = try std.fmt.bufPrintSentinel(&inner, "{s}/x", .{stuck}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.mkdir(keep, 0o700));
    try std.testing.expectError(error.Unlinked, s.empty());
    try std.testing.expect(std.mem.indexOf(u8, s.why(), "00000000000000ff.ids") != null);
    try std.testing.expect(s.shared.record.held() == 0 and s.added == 0 and s.held == 0 and s.stats().left);
    s.readPrompt(a, 0);
    try std.testing.expectEqual(@as(u64, 0), s.added); // nothing goes in while a file is left
    _ = std.c.rmdir(keep);
    _ = std.c.rmdir(stuck);
    try s.empty();
    try std.testing.expect(!s.stats().left);
    s.readPrompt(a, 0);
    try std.testing.expectEqual(@as(u64, 300), s.added);
}
