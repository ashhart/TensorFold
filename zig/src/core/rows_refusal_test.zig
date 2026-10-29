//! A disk state its family refuses while it scatters: forgotten after one recall; a failing link keeps it.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const prefetch = @import("rows_prefetch.zig");
const ahead = @import("rows_prefetch_test.zig");
const Fake = @import("rows_fake.zig").Fake;
const fresh = @import("rows_fake.zig").fresh;
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const Allocator = std.mem.Allocator;
const gpa = std.testing.allocator;
const t = std.testing;

const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };
const head = 600; // tokens both conversations share, so their first block is one file

fn root(buf: []u8, tag: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-refusal-{s}-{d}", .{ tag, platform.c.getpid() }) catch unreachable;
}

fn options(dir: []const u8) rows.Options {
    return .{ .sync_writes = true, .root = dir, .identity = 7, .cap = 1 << 34, .floor = 0, .daily = 1 << 40, .lookahead = 1 };
}

/// A server's store over `dir`: the family, its disk states and the RAM store (in place: the store points at `f`).
const Server = struct {
    f: Fake = .{ .gpa = gpa },
    r: *rows.Rows = undefined,
    s: pc.Store = undefined,

    fn open(x: *Server, dir: []const u8) !void {
        x.r = try rows.Rows.open(gpa, x.f.snapshots(), options(dir));
        x.s = pc.Store.init(gpa, x.f.snapshots(), rules, 1 << 30);
        x.s.disk = x.r;
    }

    fn close(x: *Server) void {
        x.s.deinit();
        x.r.deinit();
    }

    fn request(x: *Server, a: Allocator, p: []const u32, history: u32) !struct { from: u32, sum: u64 } {
        x.f.prompt = p;
        const plan = try x.s.begin(a, p, history, &.{}, x.f.starts, null, x.f.spans);
        return .{ .from = plan.from, .sum = x.f.pass(&x.s, plan) };
    }
};

/// Conversation `c`'s first turn: the shared head, then 300 tokens of its own.
fn turn(a: Allocator, c: u32) ![]u32 {
    const out = try a.alloc(u32, head + 300);
    for (out, 0..) |*x, i| x.* = if (i < head) @intCast(1000 + i * 7 % 997) else @intCast(50_000 + c * 10_000 + i);
    return out;
}

/// Conversations 1 and 2 written under `dir`, each a state at 800 tokens; their next turns.
fn seeded(a: Allocator, dir: []const u8) ![2][]u32 {
    var x: Server = .{};
    try x.open(dir);
    defer x.close();
    var out: [2][]u32 = undefined;
    for (&out, 1..) |*next, c| {
        const first = try turn(a, @intCast(c));
        _ = try x.request(a, first, 800);
        while (rows.write.step(x.r)) {}
        next.* = try std.mem.concat(a, u32, &.{ first[0..801], &[_]u32{ 4, 4 } });
    }
    try t.expectEqual(@as(usize, 2), x.r.states.items.len);
    return out;
}

fn exists(r: *const rows.Rows, name: rows.chain.Name, ext: []const u8) bool {
    var buf: [1200]u8 = undefined;
    const p = rows.file.path(&buf, r.dir, &rows.chain.hex(name), ext, false) catch return false;
    return platform.c.access(p, 0) == 0;
}

/// Which file the family fails on, with what, and which of the two states are kept afterwards.
const Case = struct { piece: enum { shared, own }, err: anyerror, mine: bool, other: bool };

const cases = [_]Case{
    .{ .piece = .shared, .err = error.SnapshotSize, .mine = false, .other = false }, // a head block both states name
    .{ .piece = .shared, .err = error.BadKeptRange, .mine = false, .other = false },
    .{ .piece = .own, .err = error.DiskWords, .mine = false, .other = true }, // the state's own file
    .{ .piece = .own, .err = error.KeptWordsDiffer, .mine = false, .other = true },
    .{ .piece = .own, .err = error.ScatterFailed, .mine = true, .other = true }, // a link failing: the disk is fine
    .{ .piece = .shared, .err = error.OutOfMemory, .mine = true, .other = true },
};

/// State `i`'s read started and finished.
fn started(p: *prefetch.Prefetch, r: *rows.Rows, i: usize) !void {
    const name = p.start(r, i) orelse return error.NotReadAhead;
    try t.expectEqual(@as(?prefetch.Phase, .ready), ahead.settled(p, name));
}

/// Conversation 1's next turn recalled with its copy failing as `c` says, both disk states read ahead first or not.
fn recalled(a: Allocator, tag: []const u8, c: Case, read_ahead: bool) !void {
    var buf: [128]u8 = undefined;
    const dir = root(&buf, tag);
    defer rmTree(dir);
    const turns = try seeded(a, dir);
    var x: Server = .{};
    try x.open(dir);
    defer x.close();
    const mine = x.r.best(turns[0], &.{}, &.{}, 0).?;
    var plan = try prefetch.Plan.of(x.r, gpa, &x.r.states.items[mine]);
    defer plan.deinit(gpa);
    const own = plan.pieces[plan.pieces.len - 1];
    try t.expectEqual(rows.file.Kind.state, own.kind);
    const shared = plan.pieces[0];
    try t.expectEqual(rows.file.Kind.block, shared.kind);
    const theirs = x.r.best(turns[1], &.{}, &.{}, 0).?;
    const other = x.r.states.items[theirs].name;
    const f = if (read_ahead) try prefetch.make(x.r, t.io, .{}) else null;
    if (f) |p| {
        try started(p, x.r, mine);
        try started(p, x.r, theirs);
    }
    x.f.scatters = .{ .after = if (c.piece == .own) plan.pieces.len - 1 else 0, .err = c.err };
    // the failed read leaves the turn to prefill what it would have restored
    const got = try x.request(a, turns[0], 802);
    try t.expectEqual(@as(u32, 0), got.from);
    try t.expectEqual(fresh(turns[0]), got.sum);
    try t.expectEqual(@as(u64, 0), x.r.counts.reads);
    try t.expectEqual(@as(u64, if (c.mine) 0 else 1), x.r.counts.failed);
    try t.expectEqual(@as(usize, @intFromBool(c.mine)) + @intFromBool(c.other), x.r.states.items.len);
    try t.expectEqual(c.mine, x.r.best(turns[0], &.{}, &.{}, 0) != null);
    try t.expectEqual(c.mine, exists(x.r, own.name, ".st"));
    try t.expectEqual(c.other, exists(x.r, other, ".st"));
    try t.expectEqual(c.mine or c.other, exists(x.r, shared.name, ".blk"));
    if (f) |p| {
        const k = ahead.counts(p);
        try t.expectEqual(@as(u64, 2), k.made);
        try t.expectEqual(@as(u64, 1), k.used);
        // a forgotten state's read is freed there and then, and counted cancelled if nobody used it
        try t.expectEqual(@as(u64, @intFromBool(!c.mine)) + @intFromBool(!c.other), k.freed);
        try t.expectEqual(@as(u64, @intFromBool(!c.other)), k.cancelled);
        try t.expectEqual(c.mine, ahead.settled(p, own.name) != null);
        try t.expectEqual(c.other, ahead.settled(p, other) != null);
    }
}

test "a state its family refuses while it scatters is forgotten after one recall; a failing link keeps it" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    for (cases, 0..) |c, i| {
        var tag: [16]u8 = undefined;
        try recalled(arena.allocator(), std.fmt.bufPrint(&tag, "sync{d}", .{i}) catch unreachable, c, false);
    }
}

test "prefetch: a read-ahead state its family refuses is forgotten after one recall, its read freed once" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    for (cases, 0..) |c, i| {
        var tag: [16]u8 = undefined;
        try recalled(arena.allocator(), std.fmt.bufPrint(&tag, "ahead{d}", .{i}) catch unreachable, c, true);
    }
}
