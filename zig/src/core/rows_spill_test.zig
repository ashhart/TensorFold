//! --spill-gib's tests: states evicted from RAM written to the rows store, read back equal, bounded, checked, cut off.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const Fake = @import("rows_fake.zig").Fake;
const fresh = @import("rows_fake.zig").fresh;
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const Allocator = std.mem.Allocator;

const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };

fn root(buf: []u8, tag: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-spill-{s}-{d}", .{ tag, platform.c.getpid() }) catch unreachable;
}

fn options(dir: []const u8, spill: rows.Spill) rows.Options {
    return .{ .root = dir, .identity = 7, .cap = 1 << 34, .floor = 0, .daily = 1 << 40, .lookahead = 1, .spill = spill };
}

/// A conversation's prompt: a head shared by every conversation `c`, then `n` tokens of its own.
fn prompt(a: Allocator, head: u32, c: u32, n: u32) ![]u32 {
    const out = try a.alloc(u32, head + n);
    for (out, 0..) |*t, i| t.* = if (i < head) @intCast(1000 + i * 7 % 997) else @intCast(50_000 + c * 10_000 + i);
    return out;
}

/// One request through a store with `f`: its plan, then a pass from it.
fn request(a: Allocator, s: *pc.Store, f: *Fake, p: []const u32, history: u32) !struct { from: u32, sum: u64 } {
    f.prompt = p;
    const plan = try s.begin(a, p, history, &.{}, f.starts, null, f.spans);
    return .{ .from = plan.from, .sum = f.pass(s, plan) };
}

/// A RAM budget that holds one state of `at` tokens, not two.
fn one(f: *Fake, at: u32) u64 {
    return Fake.snapshots(f).vtable.bytes(f, at) + 1000;
}

/// Files under `dir` still named `.part` (a write that never finished).
fn partFiles(dir: [:0]const u8) usize {
    const d = platform.c.opendir(dir) orelse return 0;
    defer _ = platform.c.closedir(d);
    var n: usize = 0;
    while (platform.c.readdir(d)) |ent| n += @intFromBool(std.mem.endsWith(u8, std.mem.sliceTo(&ent.name, 0), ".part"));
    return n;
}

test "spill: a state evicted from RAM goes to disk, and its next turn resumes from it equal to a fresh pass" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "resume");
    defer rmTree(dir);
    const first = try prompt(a, 0, 1, 700);
    const again = try std.mem.concat(a, u32, &.{ first[0..601], &[_]u32{ 1, 2 } });
    {
        var f: Fake = .{ .gpa = gpa };
        var s = pc.Store.init(gpa, f.snapshots(), rules, one(&f, 600));
        const r = try rows.attach(gpa, &s, options(dir, .evicted));
        defer r.deinit();
        defer s.deinit();
        _ = try request(a, &s, &f, first, 600);
        try std.testing.expect(!rows.write.pending(r) and !rows.write.step(r)); // nothing written behind
        try std.testing.expectEqual(@as(usize, 0), r.states.items.len);
        _ = try request(a, &s, &f, try prompt(a, 0, 2, 700), 600); // the first conversation's state leaves RAM
        try std.testing.expectEqual(@as(usize, 1), r.states.items.len);
        try std.testing.expectEqual(@as(u64, 1), r.counts.writes);
        const got = try request(a, &s, &f, again, 602);
        try std.testing.expectEqual(@as(u32, 600), got.from);
        try std.testing.expect(f.restored_ok);
        try std.testing.expectEqual(fresh(again), got.sum);
        try std.testing.expectEqual(@as(u64, 1), r.counts.reads);
    }
    var f: Fake = .{ .gpa = gpa }; // a restart reads the spilled state back too
    var s = pc.Store.init(gpa, f.snapshots(), rules, one(&f, 600));
    const r = try rows.attach(gpa, &s, options(dir, .evicted));
    defer r.deinit();
    defer s.deinit();
    const got = try request(a, &s, &f, again, 602);
    try std.testing.expectEqual(@as(u32, 600), got.from);
    try std.testing.expect(f.restored_ok);
    try std.testing.expectEqual(fresh(again), got.sum);
}

test "spill: --checkpoint-slots keeps that many states in RAM, and the oldest conversation goes to disk first" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "slots");
    defer rmTree(dir);
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
    var o = options(dir, .evicted);
    o.slots = 2;
    const r = try rows.attach(gpa, &s, o);
    defer r.deinit();
    defer s.deinit();
    try std.testing.expectEqual(@as(usize, 2), s.most);
    var talks: [3][]u32 = undefined;
    for (&talks, 1..) |*p, c| {
        p.* = try prompt(a, 0, @intCast(c), 700);
        _ = try request(a, &s, &f, p.*, 600);
        try std.testing.expect(s.entries.items.len <= 2);
    }
    try std.testing.expectEqual(@as(usize, 1), r.states.items.len);
    try std.testing.expect(r.best(talks[0], &.{}, &.{}, 0) != null);
    const again = try std.mem.concat(a, u32, &.{ talks[0][0..601], &[_]u32{ 3, 4 } });
    const got = try request(a, &s, &f, again, 602);
    try std.testing.expectEqual(@as(u32, 600), got.from);
    try std.testing.expect(f.restored_ok);
    try std.testing.expectEqual(fresh(again), got.sum);
    try std.testing.expect(s.entries.items.len <= 2);
}

test "spill: past --spill-gib the oldest spilled state goes, even one that saves more tokens a byte" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "cap");
    defer rmTree(dir);
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), rules, one(&f, 2600));
    const r = try rows.attach(gpa, &s, options(dir, .evicted));
    defer r.deinit();
    defer s.deinit();
    // two conversations share a long head (few bytes only theirs), then three plain ones
    const talks = [_]struct { p: []u32, h: u32 }{
        .{ .p = try prompt(a, 2048, 1, 600), .h = 2600 },
        .{ .p = try prompt(a, 2048, 2, 600), .h = 2600 },
        .{ .p = try prompt(a, 0, 3, 1500), .h = 1400 },
        .{ .p = try prompt(a, 0, 4, 1500), .h = 1400 },
        .{ .p = try prompt(a, 0, 5, 1500), .h = 1400 },
    };
    for (talks[0..4]) |t| _ = try request(a, &s, &f, t.p, t.h);
    try std.testing.expectEqual(@as(usize, 3), r.states.items.len);
    const third = r.states.items[r.best(talks[2].p, &.{}, &.{}, 0).?];
    r.cap = r.held() + third.own + third.rows - 1; // the fourth's state fits only once something goes
    _ = try request(a, &s, &f, talks[4].p, talks[4].h);
    try std.testing.expect(r.held() <= r.cap);
    try std.testing.expect(r.best(talks[0].p, &.{}, &.{}, 0) == null); // the oldest
    for (talks[1..4]) |t| try std.testing.expect(r.best(t.p, &.{}, &.{}, 0) != null);
}

test "spill: a garbled spilled file is refused, forgotten, and the turn prefills its tokens again" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "garbled");
    defer rmTree(dir);
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), rules, one(&f, 600));
    const r = try rows.attach(gpa, &s, options(dir, .evicted));
    defer r.deinit();
    defer s.deinit();
    const first = try prompt(a, 0, 1, 700);
    _ = try request(a, &s, &f, first, 600);
    _ = try request(a, &s, &f, try prompt(a, 0, 2, 700), 600);
    try std.testing.expectEqual(@as(usize, 1), r.states.items.len);
    var it = r.blocks.keyIterator();
    var path: [1200]u8 = undefined;
    const file = try rows.file.path(&path, r.dir, &rows.chain.hex(it.next().?.*), ".blk", false);
    const fd = platform.c.open(file, .{ .ACCMODE = .RDWR }, @as(platform.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    try std.testing.expectEqual(@as(isize, 1), platform.c.pwrite(fd, "X", 1, 100));
    _ = platform.c.close(fd);
    const again = try std.mem.concat(a, u32, &.{ first[0..601], &[_]u32{ 5, 6 } });
    const got = try request(a, &s, &f, again, 602);
    try std.testing.expectEqual(@as(u32, 0), got.from);
    try std.testing.expectEqual(fresh(again), got.sum);
    try std.testing.expectEqual(@as(u64, 1), r.counts.failed);
    try std.testing.expect(r.best(again, &.{}, &.{}, 0) == null);
}

test "spill: a spill cut off midway leaves no state a later turn or a restart reads" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "cut");
    defer rmTree(dir);
    const first = try prompt(a, 0, 1, 700);
    const again = try std.mem.concat(a, u32, &.{ first[0..601], &[_]u32{ 7, 8 } });
    var zdir: [1200]u8 = undefined;
    {
        var f: Fake = .{ .gpa = gpa, .gathers = 2 }; // two blocks go out, then the copies fail
        var s = pc.Store.init(gpa, f.snapshots(), rules, one(&f, 600));
        const r = try rows.attach(gpa, &s, options(dir, .evicted));
        defer r.deinit();
        defer s.deinit();
        _ = try request(a, &s, &f, first, 600);
        _ = try request(a, &s, &f, try prompt(a, 0, 2, 700), 600);
        try std.testing.expectEqual(@as(u64, 1), r.counts.failed);
        try std.testing.expectEqual(@as(usize, 0), r.states.items.len);
        try std.testing.expectEqual(@as(usize, 0), r.blocks.count()); // the blocks it wrote went with it
        try std.testing.expectEqual(@as(usize, 0), partFiles(r.dir));
        f.gathers = null;
        const got = try request(a, &s, &f, again, 602);
        try std.testing.expectEqual(@as(u32, 0), got.from);
        try std.testing.expectEqual(fresh(again), got.sum);
        const stray = try std.fmt.bufPrintSentinel(&zdir, "{s}/{s}.st.part", .{ r.dir, &rows.chain.hex(rows.chain.stem(@as(rows.chain.Name, @splat(9)))) }, 0);
        const fd = platform.c.open(stray, .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(platform.c.mode_t, 0o600));
        try std.testing.expect(fd >= 0);
        _ = platform.c.close(fd); // as a server killed mid-write leaves it
    }
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), rules, one(&f, 600));
    const r = try rows.attach(gpa, &s, options(dir, .evicted));
    defer r.deinit();
    defer s.deinit();
    try std.testing.expectEqual(@as(usize, 0), partFiles(r.dir));
    const got = try request(a, &s, &f, again, 602);
    try std.testing.expectEqual(@as(u32, 0), got.from);
    try std.testing.expectEqual(fresh(again), got.sum);
}

test "spill with --learn-conversations: an entry evicted before its write is finished, not given up" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "finish");
    defer rmTree(dir);
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), rules, one(&f, 600));
    const r = try rows.attach(gpa, &s, options(dir, .finish));
    defer r.deinit();
    defer s.deinit();
    const first = try prompt(a, 0, 5, 700);
    _ = try request(a, &s, &f, first, 600);
    try std.testing.expect(rows.write.pending(r)); // written behind, as --learn-conversations writes
    _ = rows.write.step(r);
    try std.testing.expect(r.job != null);
    _ = try request(a, &s, &f, try prompt(a, 0, 6, 700), 600); // the first state leaves RAM mid-write
    try std.testing.expectEqual(@as(usize, 1), r.states.items.len);
    try std.testing.expect(r.best(first, &.{}, &.{}, 0) != null);
    while (rows.write.step(r)) {}
    try std.testing.expectEqual(@as(usize, 2), r.states.items.len);
}

test "without --spill-gib nothing changes: an unwritten evicted state is dropped and the RAM bound is the family's" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "off");
    defer rmTree(dir);
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), rules, one(&f, 600));
    const r = try rows.attach(gpa, &s, options(dir, .off));
    defer r.deinit();
    defer s.deinit();
    try std.testing.expectEqual(@as(usize, std.math.maxInt(usize)), s.most);
    const first = try prompt(a, 0, 1, 700);
    _ = try request(a, &s, &f, first, 600);
    _ = try request(a, &s, &f, try prompt(a, 0, 2, 700), 600); // evicted before any idle write
    try std.testing.expectEqual(@as(usize, 0), r.states.items.len);
    try std.testing.expect(rows.write.pending(r));
    while (rows.write.step(r)) {}
    try std.testing.expectEqual(@as(usize, 1), r.states.items.len);
    try std.testing.expect(r.best(first, &.{}, &.{}, 0) == null);
}
