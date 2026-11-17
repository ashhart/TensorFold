//! The miss reasons: every cause code derived at lookup, read-only, and replies unchanged.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const miss = @import("prompt_cache_miss.zig");
const fake = @import("prompt_cache_fake.zig");
const engine_status = @import("engine_status.zig");

const gpa = std.testing.allocator;

fn tokens(n: usize, seed: u32) []u32 {
    const t = gpa.alloc(u32, n) catch unreachable;
    for (t, 0..) |*x, i| x.* = seed + @as(u32, @intCast(i));
    return t;
}

test "miss reasons: the names table matches the enum, in order" {
    inline for (engine_status.Status.PromptCache.miss_reason_names, 0..) |name, i|
        try std.testing.expectEqualStrings(name, @tagName(@as(miss.Reason, @fromBackingInt(@intCast(i)))));
}

test "miss reasons: a store with nothing says cold, and a short prompt says too_short" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var f: fake.Fake = .{ .gpa = gpa };
    var store = pc.Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 4, .tail = true, .min_prompt = 8, .min_gap = 2, .lookahead = 1 }, 1024);
    defer store.deinit();
    const short = tokens(4, 1);
    defer gpa.free(short);
    const l = try store.lookup(arena.allocator(), short, 0, &.{}, &.{}, &.{});
    try std.testing.expectEqual(miss.Reason.too_short, l.miss.?.reason);
    const prompt = tokens(12, 1);
    defer gpa.free(prompt);
    const l2 = try store.lookup(arena.allocator(), prompt, 0, &.{}, &.{}, &.{});
    try std.testing.expectEqual(miss.Reason.cold, l2.miss.?.reason);
    try std.testing.expectEqual(@as(u64, 1), store.miss_counts[@backingInt(miss.Reason.cold)]);
}

test "miss reasons: a token change before the last user start is prompt_changed, after it reply_changed" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var f: fake.Fake = .{ .gpa = gpa };
    var store = pc.Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 4, .tail = true, .min_prompt = 0, .min_gap = 2, .lookahead = 1 }, 1024);
    defer store.deinit();
    const prompt = tokens(12, 1);
    defer gpa.free(prompt);
    const p1 = try plan(arena.allocator(), &store, prompt);
    _ = f.pass(&store, prompt, p1);
    // the same path, one token changed at 3 (before the last user start at 8): the request's prompt changed
    const edited = tokens(12, 1);
    defer gpa.free(edited);
    edited[3] = 999;
    const starts_user = [_]u32{8};
    const l = try store.lookupUser(arena.allocator(), edited, 12, 0, 0, 0, &.{}, &starts_user, &.{});
    try std.testing.expectEqual(miss.Reason.prompt_changed, l.miss.?.reason);
    try std.testing.expectEqual(@as(u32, 3), l.miss.?.at);
    try std.testing.expectEqual(@as(u32, 4), l.miss.?.kept); // tokens are seed+i: position 3 held 4
    try std.testing.expectEqual(@as(u32, 999), l.miss.?.request);
    // a change at the last user start is the earlier reply's rendering, not an earlier message
    const edited2 = tokens(12, 1);
    defer gpa.free(edited2);
    edited2[8] = 777;
    const l2 = try store.lookupUser(arena.allocator(), edited2, 12, 0, 0, 0, &.{}, &starts_user, &.{});
    try std.testing.expectEqual(miss.Reason.reply_changed, l2.miss.?.reason);
    try std.testing.expectEqual(@as(u32, 8), l2.miss.?.at);
}

test "miss reasons: a request ending inside a longer kept state says ends_inside" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var f: fake.Fake = .{ .gpa = gpa };
    var store = pc.Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 4, .tail = true, .min_prompt = 0, .min_gap = 2, .lookahead = 1 }, 1024);
    defer store.deinit();
    const long = tokens(12, 5);
    defer gpa.free(long);
    const p1 = try plan(arena.allocator(), &store, long);
    _ = f.pass(&store, long, p1);
    const l = try store.lookup(arena.allocator(), long[0..6], 0, &.{}, &.{}, &.{});
    try std.testing.expectEqual(miss.Reason.ends_inside, l.miss.?.reason);
    try std.testing.expectEqual(@as(u32, 6), l.miss.?.shared_len);
}

test "miss reasons: decode rows the kept pass did not prefill say rows_changed" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var f: fake.Fake = .{ .gpa = gpa };
    var store = pc.Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 4, .tail = true, .min_prompt = 0, .min_gap = 2, .lookahead = 1 }, 1024);
    defer store.deinit();
    const prompt = tokens(12, 7);
    defer gpa.free(prompt);
    const p1 = try plan(arena.allocator(), &store, prompt);
    _ = f.pass(&store, prompt, p1);
    // the same tokens kept under one decode span, looked up under another
    const spans = [_][2]u32{.{ 0, 12 }};
    const other = [_][2]u32{.{ 2, 12 }};
    _ = store.keep(prompt, 8, null, &.{}, &spans);
    const l = try store.lookup(arena.allocator(), prompt, 0, &.{}, &.{}, &other);
    try std.testing.expectEqual(miss.Reason.rows_changed, l.miss.?.reason);
}

test "miss reasons: a state gone for room says evicted, shed says shed, cleared says cleared" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var f: fake.Fake = .{ .gpa = gpa };
    var store = pc.Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 4, .tail = true, .min_prompt = 0, .min_gap = 2, .lookahead = 1 }, 150);
    defer store.deinit();
    const a_prompt = tokens(12, 11);
    defer gpa.free(a_prompt);
    const b_prompt = tokens(12, 211);
    defer gpa.free(b_prompt);
    const p1 = try plan(arena.allocator(), &store, a_prompt);
    _ = f.pass(&store, a_prompt, p1);
    const p2 = try plan(arena.allocator(), &store, b_prompt); // a second keep under a tiny budget evicts the first
    _ = f.pass(&store, b_prompt, p2);
    const l = try store.lookup(arena.allocator(), a_prompt, 0, &.{}, &.{}, &.{});
    try std.testing.expectEqual(miss.Reason.evicted, l.miss.?.reason);
    // shed: a kept state leaves under a zero watermark, and its own miss says so
    const pb = try plan(arena.allocator(), &store, b_prompt); // consumes lookup(a)'s reservation
    _ = f.pass(&store, b_prompt, pb);
    const shed = @import("prompt_cache_shed.zig");
    _ = shed.shed(&store, 0);
    const l2 = try store.lookup(arena.allocator(), b_prompt, 0, &.{}, &.{}, &.{});
    try std.testing.expectEqual(miss.Reason.shed, l2.miss.?.reason);
    // cleared: a weight change takes the last states, and the ring says cleared, not cold
    const c_prompt = tokens(12, 411);
    defer gpa.free(c_prompt);
    const pc3 = try plan(arena.allocator(), &store, c_prompt); // consumes lookup(b)'s reservation
    _ = f.pass(&store, c_prompt, pc3);
    store.clear();
    const l3 = try store.lookup(arena.allocator(), c_prompt, 0, &.{}, &.{}, &.{});
    try std.testing.expectEqual(miss.Reason.cleared, l3.miss.?.reason);
}

test "miss reasons: a refused keep says not_kept, a failed restore says restore_failed, a full resume says none" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var f: fake.Fake = .{ .gpa = gpa };
    var store = pc.Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 4, .tail = true, .min_prompt = 0, .min_gap = 2, .lookahead = 1 }, 0);
    defer store.deinit();
    const prompt = tokens(12, 21);
    defer gpa.free(prompt);
    _ = store.keep(prompt, 8, null, &.{}, &.{}); // a zero budget refuses the keep: the ring records it
    const l = try store.lookup(arena.allocator(), prompt, 0, &.{}, &.{}, &.{});
    try std.testing.expectEqual(miss.Reason.not_kept, l.miss.?.reason);
    // a restore the family refused: the chosen state's failure is the reply's reason
    var b: fake.Fake = .{ .gpa = gpa, .fail_restore = true };
    var store2 = pc.Store.init(gpa, b.snapshots(), .{ .planned = true, .grid = 4, .tail = true, .min_prompt = 0, .min_gap = 2, .lookahead = 1 }, 1024);
    defer store2.deinit();
    const kept = tokens(12, 31);
    defer gpa.free(kept);
    const p2 = try plan(arena.allocator(), &store2, kept);
    _ = b.pass(&store2, kept, p2);
    const l2 = try plan(arena.allocator(), &store2, kept);
    try std.testing.expectEqual(@as(u32, 0), l2.from);
    try std.testing.expectEqual(miss.Reason.restore_failed, l2.miss.?.reason);
    // a resume with nothing further available says none
    var c: fake.Fake = .{ .gpa = gpa };
    var store3 = pc.Store.init(gpa, c.snapshots(), .{ .planned = true, .grid = 4, .tail = true, .min_prompt = 0, .min_gap = 2, .lookahead = 1 }, 1024);
    defer store3.deinit();
    const full = tokens(12, 41);
    defer gpa.free(full);
    const p3 = try plan(arena.allocator(), &store3, full);
    _ = c.pass(&store3, full, p3);
    const l3 = try plan(arena.allocator(), &store3, full);
    try std.testing.expectEqual(@as(u32, 8), l3.from);
    try std.testing.expectEqual(miss.Reason.none, l3.miss.?.reason);
    try std.testing.expectEqual(@as(u32, 8), l3.miss.?.resumed_len);
}

test "miss reasons: the field is diagnostics only — every lookup's marks and restores are unchanged" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var f: fake.Fake = .{ .gpa = gpa };
    var store = pc.Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 4, .tail = true, .min_prompt = 0, .min_gap = 2, .lookahead = 1 }, 1024);
    defer store.deinit();
    const prompt = tokens(12, 51);
    defer gpa.free(prompt);
    const p1 = try plan(arena.allocator(), &store, prompt);
    _ = f.pass(&store, prompt, p1);
    // the derivation is read-only: the same lookup twice restores the same state and says the same thing
    const l1 = try store.lookup(arena.allocator(), prompt, 0, &.{}, &.{}, &.{});
    const l2 = try store.lookup(arena.allocator(), prompt, 0, &.{}, &.{}, &.{});
    try std.testing.expectEqual(l1.entry.?.at, l2.entry.?.at);
    try std.testing.expectEqual(@as(u32, 8), l2.entry.?.at);
    try std.testing.expectEqual(miss.Reason.none, l1.miss.?.reason);
    try std.testing.expectEqual(miss.Reason.none, l2.miss.?.reason);
    try std.testing.expectEqual(l1.miss.?.shared_len, l2.miss.?.shared_len);
}

fn plan(a: std.mem.Allocator, s: *pc.Store, prompt: []const u32) !pc.Plan {
    return s.begin(a, prompt, 0, &.{}, &.{}, null, &.{});
}

fn dbg(s: *pc.Store, tag: []const u8) void {
    std.debug.print("DBG {s}: entries={d} held={d} ring:", .{ tag, s.entries.items.len, s.held });
    for (s.ring.records) |rec| if (rec.key != 0) std.debug.print(" [{d}:{s}]", .{ rec.len, @tagName(rec.cause) });
    std.debug.print("\n", .{});
}
