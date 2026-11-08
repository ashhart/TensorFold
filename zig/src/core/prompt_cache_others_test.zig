//! Which kept states go when a new one needs room: main's order, pinned before the keep-others rule.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Store = pc.Store;
const Allocator = std.mem.Allocator;
const t = std.testing;

/// A family whose state at `at` takes `at` MiB; it writes down each state it drops as conversation << 16 | at.
const Family = struct {
    gpa: Allocator,
    session: u32 = 0,
    live: usize = 0,
    dropped: std.ArrayList(u32) = .empty,

    fn snapshots(f: *Family) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }
    fn of(ptr: *anyopaque) *Family {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytes(_: *anyopaque, at: u32) u64 {
        return @as(u64, at) << 20;
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f = of(ptr);
        const st = try f.gpa.create(u32);
        st.* = f.session << 16 | at;
        f.live += 1;
        return st;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const f = of(ptr);
        const st: *u32 = @ptrCast(@alignCast(saved));
        f.dropped.append(f.gpa, st.*) catch @panic("test drop list");
        f.gpa.destroy(st);
        f.live -= 1;
    }
    fn deinit(f: *Family) void {
        f.dropped.deinit(f.gpa);
    }
};

/// Conversation `id`'s first `len` tokens (a later turn's prompt extends an earlier one's).
fn talk(buf: []u32, id: u32, len: usize) []const u32 {
    for (buf[0..len], 0..) |*x, i| x.* = id * 10_000 + @as(u32, @intCast(i));
    return buf[0..len];
}

/// One request of conversation `id`: begin (restore and room), then a keep at each planned mark, as a pass does.
fn turn(f: *Family, s: *Store, a: Allocator, id: u32, len: usize) !void {
    var buf: [4096]u32 = undefined;
    const prompt = talk(&buf, id, len);
    f.session = id;
    const p = try s.begin(a, prompt, @intCast(len - 2), &.{}, &.{}, null, &.{});
    for (p.marks) |m| _ = s.keep(prompt, m, null, &.{}, &.{});
}

/// `turn` through lookup and resumed, the lane host's path: the restored entry is borrowed while room is made.
fn served(f: *Family, s: *Store, a: Allocator, id: u32, len: usize) !void {
    var buf: [4096]u32 = undefined;
    const prompt = talk(&buf, id, len);
    f.session = id;
    const l = try s.lookup(a, prompt, @intCast(len - 2), &.{}, &.{}, &.{});
    if (l.entry) |e| s.resumed(e, prompt, true);
    for (l.marks) |m| _ = s.keep(prompt, m, null, &.{}, &.{});
}

fn sumHeld(s: *const Store) u64 {
    var n: u64 = 0;
    for (s.entries.items) |e| n += e.bytes;
    return n;
}

test "keep others: off, a scripted run of three conversations evicts in main's order" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Family = .{ .gpa = t.allocator };
    defer f.deinit();
    var s = Store.init(t.allocator, f.snapshots(), .{ .min_prompt = 0 }, 100 << 20);
    defer s.deinit();
    try turn(&f, &s, a, 1, 30); // A keeps 28 MiB
    try turn(&f, &s, a, 2, 40); // B 38
    try turn(&f, &s, a, 3, 25); // C 23: 89 held
    try turn(&f, &s, a, 1, 34); // A's 32 replaces its 28
    try turn(&f, &s, a, 4, 60); // D's 58 pushes others out
    try served(&f, &s, a, 2, 45); // B again, through lookup
    try turn(&f, &s, a, 1, 50); // A's 48
    // A's lookup makes room while its own 28 is borrowed, so B's 38 goes first; D takes A's 28 (superseded), then C
    try t.expectEqualSlices(u32, &.{ 2 << 16 | 38, 1 << 16 | 28, 3 << 16 | 23, 1 << 16 | 32, 4 << 16 | 58 }, f.dropped.items);
    try t.expectEqual(@as(usize, 2), s.entries.items.len);
    try t.expectEqual(@as(u64, 5), s.counts.evicted);
    try t.expectEqual(@as(u64, 0), s.counts.refused);
    try t.expectEqual(@as(u64, (43 + 48) << 20), s.held);
    try t.expectEqual(sumHeld(&s), s.held);
    try t.expectEqual(s.entries.items.len, f.live);
}

test "keep others: off, a seeded run of 400 turns drops states in main's order" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Family = .{ .gpa = t.allocator };
    defer f.deinit();
    var s = Store.init(t.allocator, f.snapshots(), .{ .min_prompt = 0 }, 160 << 20);
    defer s.deinit();
    s.most = 5;
    var prng: std.Random.DefaultPrng = .init(2346);
    const r = prng.random();
    var lens: [8]usize = @splat(0);
    var ids: [8]u32 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    for (0..400) |_| {
        const k = r.uintLessThan(usize, lens.len);
        if (lens[k] == 0 or r.uintLessThan(u32, 12) == 0) { // a new conversation in this seat
            ids[k] += 8;
            lens[k] = 8 + r.uintLessThan(usize, 40);
        } else lens[k] += 2 + r.uintLessThan(usize, 30);
        if (lens[k] > 400) lens[k] = 8;
        if (r.boolean()) try turn(&f, &s, a, ids[k], lens[k]) else try served(&f, &s, a, ids[k], lens[k]);
    }
    try t.expectEqual(@as(usize, 246), f.dropped.items.len);
    try t.expectEqual(@as(u64, 1744666703333870265), std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(f.dropped.items)));
    try t.expectEqual(@as(u64, 246), s.counts.evicted);
    try t.expectEqual(@as(u64, 153), s.counts.refused);
    try t.expectEqual(@as(u64, 247), s.counts.kept);
    try t.expectEqual(@as(u64, 74), s.counts.hits);
    try t.expectEqual(@as(u64, 146 << 20), s.held);
    try t.expectEqual(sumHeld(&s), s.held);
    try t.expectEqual(s.entries.items.len, f.live);
}
