//! Which kept states go when a new one needs room: main's order with the keep-others rule off, and the rule on.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const note = @import("prompt_cache_note.zig");
const Store = pc.Store;
const Allocator = std.mem.Allocator;
const t = std.testing;

/// A family whose state at `at` takes `at` MiB; it writes down each state it drops as conversation << 16 | at.
const Family = struct {
    gpa: Allocator,
    session: u32 = 0,
    live: usize = 0,
    dropped: std.ArrayList(u32) = .empty,
    pool: u64 = 0, // `pages`: a fixed pool of this many bytes, a dropped state's back as spare, no trim (as GLM's)
    used: u64 = 0,

    fn snapshots(f: *Family) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }
    fn pages(f: *Family, pool: u64) pc.Snapshots {
        f.pool = pool;
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop, .spare = spare, .reuses = reuses } };
    }
    fn spare(ptr: *anyopaque) u64 {
        const f = of(ptr);
        return f.pool - f.used;
    }
    fn reuses(ptr: *anyopaque, at: u32) bool {
        return spare(ptr) >= bytes(ptr, at);
    }
    fn of(ptr: *anyopaque) *Family {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytes(_: *anyopaque, at: u32) u64 {
        return @as(u64, at) << 20;
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f = of(ptr);
        if (f.pool > 0 and f.pool - f.used < bytes(ptr, at)) return error.PoolFull;
        const st = try f.gpa.create(u32);
        st.* = f.session << 16 | at;
        f.live += 1;
        if (f.pool > 0) f.used += bytes(ptr, at);
        return st;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const f = of(ptr);
        const st: *u32 = @ptrCast(@alignCast(saved));
        f.dropped.append(f.gpa, st.*) catch @panic("test drop list");
        if (f.pool > 0) f.used -= bytes(ptr, @truncate(st.* & 0xffff));
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

/// A, B and C keep 28, 38 and 23 MiB of a 100 MiB budget: 89 held, 11 free.
fn three(f: *Family, s: *Store, a: Allocator) !void {
    try turn(f, s, a, 1, 30);
    try turn(f, s, a, 2, 40);
    try turn(f, s, a, 3, 25);
}

/// Whether `three`'s states all stay: nothing dropped, 89 MiB held, and each still resumes its conversation.
fn survivors(f: *const Family, s: *Store) bool {
    var buf: [64]u32 = undefined;
    if (f.dropped.items.len != 0 or s.entries.items.len != 3 or f.live != 3 or s.held != 89 << 20) return false;
    for ([_][2]u32{ .{ 1, 28 }, .{ 2, 38 }, .{ 3, 23 } }) |c| {
        const e = s.find(talk(&buf, c[0], 60), &.{}, &.{}) orelse return false;
        if (e.at != c[1]) return false;
    }
    return true;
}

test "keep others: on, a state that fits only by evicting another conversation's is refused, once, and all stay" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Family = .{ .gpa = t.allocator };
    defer f.deinit();
    var s = Store.init(t.allocator, f.snapshots(), .{ .min_prompt = 0 }, 100 << 20);
    defer s.deinit();
    try t.expect(!s.keep_others); // off unless the host sets it
    s.keep_others = true;
    try three(&f, &s, a);
    var heard: note.Heard = .{};
    note.sink = &heard;
    defer note.sink = null;
    try turn(&f, &s, a, 1, 50); // A's 48 fits only with B's or C's room: its own 28 is not enough
    try t.expect(survivors(&f, &s));
    try t.expectEqualStrings("kept nothing at 48 tokens: 48 MiB does not fit beside 3 kept states; 11 MiB of the 100 MiB budget free (TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS=1)\n", heard.text());
    try served(&f, &s, a, 4, 60); // a new conversation through the lane host's path: its lookup evicts nothing either
    try t.expect(survivors(&f, &s));
    try t.expectEqual(@as(usize, 2), std.mem.count(u8, heard.text(), "does not fit beside"));
    try t.expect(std.mem.indexOf(u8, heard.text(), "kept nothing at 58 tokens: 58 MiB does not fit") != null);
    s.most = 3; // 5 MiB fits the budget, but only in another conversation's slot
    try turn(&f, &s, a, 5, 7);
    try t.expect(survivors(&f, &s));
    try t.expectEqual(@as(u64, 3), s.counts.refused);
    try t.expectEqual(@as(u64, 0), s.counts.evicted);
    try t.expectEqual(@as(u64, 3), s.counts.kept);
}

test "keep others: on, a conversation's longer state still replaces its own older one, on both paths" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Family = .{ .gpa = t.allocator };
    defer f.deinit();
    var s = Store.init(t.allocator, f.snapshots(), .{ .min_prompt = 0 }, 100 << 20);
    defer s.deinit();
    s.keep_others = true;
    try three(&f, &s, a);
    try turn(&f, &s, a, 1, 34); // A's 32 in place of its 28: 93 held
    try t.expectEqualSlices(u32, &.{1 << 16 | 28}, f.dropped.items);
    try served(&f, &s, a, 2, 45); // B's 43 in place of its 38, its 38 borrowed while the lookup made room
    try t.expectEqualSlices(u32, &.{ 1 << 16 | 28, 2 << 16 | 38 }, f.dropped.items);
    var buf: [64]u32 = undefined;
    for ([_][2]u32{ .{ 1, 32 }, .{ 2, 43 }, .{ 3, 23 } }) |c| try t.expectEqual(c[1], s.find(talk(&buf, c[0], 60), &.{}, &.{}).?.at);
    try t.expectEqual(@as(u64, 98 << 20), s.held);
    try t.expectEqual(@as(u64, 0), s.counts.refused);
}

test "keep others: on, a pool family's spare pages take its own replacement, and free bytes add up throughout" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Family = .{ .gpa = t.allocator };
    defer f.deinit();
    var s = Store.init(t.allocator, f.pages(100 << 20), .{ .min_prompt = 0 }, 100 << 20);
    defer s.deinit();
    s.keep_others = true;
    try three(&f, &s, a);
    for ([_]void{ {}, {} }) |_| { // before and after a refusal: 89 held, 11 spare, nothing else free
        try t.expectEqual(@as(u64, 89 << 20), s.held);
        try t.expectEqual(sumHeld(&s), s.held);
        try t.expectEqual(f.used, s.held);
        try t.expectEqual(@as(u64, 11 << 20), s.spare());
        try t.expectEqual(@as(u64, 0), s.room());
        try turn(&f, &s, a, 1, 50);
    }
    try t.expectEqual(@as(u64, 2), s.counts.refused);
    try t.expectEqual(@as(usize, 0), f.dropped.items.len);
    try turn(&f, &s, a, 1, 34); // A's 28 goes back to the pool, and its 32 reuses the spare pages
    try t.expectEqualSlices(u32, &.{1 << 16 | 28}, f.dropped.items);
    try t.expectEqual(@as(u64, 93 << 20), s.held);
    try t.expectEqual(sumHeld(&s), s.held);
    try t.expectEqual(f.used, s.held);
    try t.expectEqual(@as(u64, 7 << 20), s.spare());
    try t.expectEqual(s.budget, s.held + s.spare() + s.room());
    try t.expectEqual(@as(u64, 2), s.counts.refused);
}

test "keep others: the control, with the rule off the same request loses a survivor" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Family = .{ .gpa = t.allocator };
    defer f.deinit();
    var s = Store.init(t.allocator, f.snapshots(), .{ .min_prompt = 0 }, 100 << 20);
    defer s.deinit();
    try three(&f, &s, a);
    try t.expect(survivors(&f, &s));
    try turn(&f, &s, a, 1, 50);
    try t.expect(!survivors(&f, &s));
    try t.expectEqualSlices(u32, &.{2 << 16 | 38}, f.dropped.items); // B's: A's own 28 is borrowed while it makes room
}

test "keep others: only exactly 1 turns the rule on" {
    try t.expect(pc.others.wanted("1"));
    for ([_]?[]const u8{ null, "", "0", "true", "on", " 1", "1 ", "01" }) |raw| try t.expect(!pc.others.wanted(raw));
}
