//! KV isolation across sessions: a spilled and dropped state, a retired page, and a disjoint session admitted between.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const Fake = @import("rows_fake.zig").Fake;
const fresh = @import("rows_fake.zig").fresh;
const Pooled = @import("prompt_cache_fake.zig").Fake;
const pooledFresh = @import("prompt_cache_test.zig").fresh;
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const Allocator = std.mem.Allocator;

const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };

fn root(buf: []u8, tag: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-isolation-{s}-{d}", .{ tag, platform.c.getpid() }) catch unreachable;
}

/// Session `c`'s prompt of `n` tokens: no token is shared with another session's, and every length is the same.
fn session(a: Allocator, c: u32, n: u32) ![]u32 {
    const out = try a.alloc(u32, n);
    for (out, 0..) |*t, i| t.* = @intCast(50_000 + c * 10_000 + i);
    return out;
}

/// The session's next turn: its first 601 tokens, then two new ones.
fn next(a: Allocator, p: []const u32) ![]u32 {
    return std.mem.concat(a, u32, &.{ p[0..601], &[_]u32{ 1, 2 } });
}

/// One request: its plan, then a pass from it (`f.restored_ok` holds whether every restore equalled a fresh pass).
fn request(a: Allocator, s: *pc.Store, f: *Fake, p: []const u32, history: u32) !struct { from: u32, sum: u64 } {
    f.prompt = p;
    const plan = try s.begin(a, p, history, &.{}, f.starts, null, f.spans);
    return .{ .from = plan.from, .sum = f.pass(s, plan) };
}

/// A RAM budget that holds one state of `at` tokens, not two.
fn one(f: *Fake, at: u32) u64 {
    return Fake.snapshots(f).vtable.bytes(f, at) + 1000;
}

fn blockFiles(dir: [:0]const u8) usize {
    const d = platform.c.opendir(dir) orelse return 0;
    defer _ = platform.c.closedir(d);
    var n: usize = 0;
    while (platform.c.readdir(d)) |ent| n += @intFromBool(std.mem.endsWith(u8, std.mem.sliceTo(&ent.name, 0), ".blk"));
    return n;
}

test "rows isolation: B admitted after A is spilled and dropped never resumes A's rows; A resumes exactly its own" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "spill");
    defer rmTree(dir);
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), rules, one(&f, 600));
    const r = try rows.attach(gpa, &s, .{ .sync_writes = true, .root = dir, .identity = 7, .cap = 1 << 34, .floor = 0, .daily = 1 << 40, .lookahead = 1, .spill = .evicted });
    defer r.deinit();
    defer s.deinit();
    // A and B share no token, and keep states at the same positions
    const sa = try session(a, 1, 700);
    const sb = try session(a, 2, 700);
    _ = try request(a, &s, &f, sa, 600);
    const b1 = try request(a, &s, &f, sb, 600); // A leaves RAM for disk
    try std.testing.expectEqual(@as(usize, 1), r.states.items.len);
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    try std.testing.expectEqualSlices(u32, sb[0..600], s.entries.items[0].tokens[0..600]);
    // (1) B's first turn and its next resume nothing of A's: B prefills, then resumes its own RAM state
    try std.testing.expectEqual(@as(u32, 0), b1.from);
    try std.testing.expectEqual(fresh(sb), b1.sum);
    const b_next = try next(a, sb);
    const b2 = try request(a, &s, &f, b_next, 602);
    try std.testing.expectEqual(@as(u32, 600), b2.from);
    try std.testing.expect(f.restored_ok);
    try std.testing.expectEqual(fresh(b_next), b2.sum);
    try std.testing.expectEqual(@as(u64, 0), r.counts.reads); // A's spilled file was not read for B
    // (2) A's next turn resumes A's spilled state, equal to a fresh pass of A's own tokens
    const a_next = try next(a, sa);
    const a2 = try request(a, &s, &f, a_next, 602);
    try std.testing.expectEqual(@as(u32, 600), a2.from);
    try std.testing.expect(f.restored_ok);
    try std.testing.expectEqual(fresh(a_next), a2.sum);
    try std.testing.expectEqual(@as(u64, 1), r.counts.reads);
}

test "rows isolation: after A's spilled state and blocks are retired and B's are written, A's lookup takes nothing" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "retire");
    defer rmTree(dir);
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), rules, one(&f, 600));
    const r = try rows.attach(gpa, &s, .{ .sync_writes = true, .root = dir, .identity = 7, .cap = 1 << 34, .floor = 0, .daily = 1 << 40, .lookahead = 1, .spill = .evicted });
    defer r.deinit();
    defer s.deinit();
    const sa = try session(a, 1, 700);
    const sb = try session(a, 2, 700);
    _ = try request(a, &s, &f, sa, 600);
    _ = try request(a, &s, &f, sb, 600); // A on disk
    try std.testing.expectEqual(@as(usize, 1), r.states.items.len);
    try std.testing.expect(blockFiles(r.dir) > 0);
    r.remove(0); // A's state retired: its file and blocks go
    try std.testing.expectEqual(@as(usize, 0), r.states.items.len);
    try std.testing.expectEqual(@as(usize, 0), blockFiles(r.dir));
    _ = try request(a, &s, &f, try session(a, 3, 700), 600); // B's state takes the disk A's left
    try std.testing.expectEqual(@as(usize, 1), r.states.items.len);
    try std.testing.expect(blockFiles(r.dir) > 0);
    // (3) A's prefix matches no state's tokens: nothing is read or restored, and A prefills its own tokens
    const a_next = try next(a, sa);
    try std.testing.expectEqual(@as(?usize, null), r.best(a_next, &.{}, &.{}, 0));
    const reads = r.counts.reads;
    const got = try request(a, &s, &f, a_next, 602);
    try std.testing.expectEqual(@as(u32, 0), got.from);
    try std.testing.expectEqual(reads, r.counts.reads);
    try std.testing.expect(f.restored_ok);
    try std.testing.expectEqual(fresh(a_next), got.sum);
}

test "rows isolation: storage A's dropped state left spare and B's save reused is never resumed for A's prefix" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Pooled = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.pooled(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const sa = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18 };
    const sb = [_]u32{ 21, 22, 23, 24, 25, 26, 27, 28 };
    f.at, f.sum = .{ 4, pooledFresh(sa[0..4]) };
    try std.testing.expect(s.keep(&sa, 4, null, &.{}, &.{}));
    s.remove(0); // A's page retired: its storage stays spare
    try std.testing.expectEqual(@as(u64, 94), s.spare());
    f.at, f.sum = .{ 4, pooledFresh(sb[0..4]) };
    try std.testing.expect(s.keep(&sb, 4, null, &.{}, &.{})); // B's state reuses that page
    try std.testing.expectEqual(@as(u64, 0), s.spare());
    // (3) the lookup for A's prefix compares tokens first: B's entry at the same position is not restored
    const l = try s.lookup(a, &sa, 4, &.{}, &.{}, &.{});
    try std.testing.expect(l.entry == null);
    f.at, f.sum = .{ 999, 0 }; // a restore would overwrite these
    const plan = try s.begin(a, &sa, 4, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 0), plan.from);
    try std.testing.expectEqual(@as(u32, 999), f.at);
    try std.testing.expectEqual(pooledFresh(&sa), f.pass(&s, &sa, plan));
}
