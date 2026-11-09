//! Mutable backend rows are captured while readable, before allocation-free expiry reclaims their location.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const fake = @import("rows_fake.zig");
const waits = @import("rows_test_wait.zig");
const notes = @import("prompt_cache_note.zig");
const a = std.testing.allocator;

fn transient(_: *anyopaque, saved: pc.Saved) bool {
    const k: *fake.Kept = @ptrCast(@alignCast(saved));
    return k.at >= 900;
}

fn run(sync: bool, spill: rows.Spill) !void {
    const tmp = if (platform.c.getenv("TMPDIR")) |p| std.mem.span(p) else ".";
    const root = try std.fmt.allocPrint(a, "{s}/tf-expiry-{s}-{any}-{d}", .{ tmp, @tagName(spill), sync, platform.c.getpid() });
    defer a.free(root);
    defer @import("prompt_imprint_test.zig").rmTree(root);
    var f: fake.Fake = .{ .gpa = a };
    var image = f.snapshots().vtable.rows.?.*;
    image.transient = transient;
    var table = f.snapshots().vtable.*;
    table.rows = &image;
    var s = pc.Store.init(a, .{ .ptr = &f, .vtable = &table }, .{ .lookahead = 1, .min_gap = 1, .min_prompt = 0 }, 1 << 30);
    const r = try rows.attach(a, &s, .{ .root = root, .identity = 7, .cap = 1 << 30, .daily = 1 << 30, .floor = 0, .lookahead = 1, .spill = spill, .sync_writes = sync });
    defer r.deinit();
    defer s.deinit();
    var tokens: [1200]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 1);
    f.prompt = &tokens;
    f.at = 600;
    try std.testing.expect(s.keep(&tokens, 600, null, &.{}, &.{}));
    if (spill == .off) try waits.published(r);
    s.remove(0); // the old depth reaches disk through ordinary eviction
    try waits.published(r);
    // No new LRU eviction occurs: only the family invalidates the next state.
    f.at = 900;
    try std.testing.expect(s.keep(&tokens, 900, null, &.{}, &.{}));
    const e = s.entries.items[0];
    const before = f.gathered;
    s.expired(e.saved); // must neither gather nor allocate here
    try std.testing.expectEqual(before, f.gathered);
    const k: *fake.Kept = @ptrCast(@alignCast(e.saved));
    @memset(k.img, 0xa5); // the shared pool reused its rows, without keeping any old contents
    s.collectExpired();
    try waits.published(r);
    try std.testing.expectEqual(@as(u64, 2), r.counts.writes);
    try std.testing.expectEqual(@as(u32, 900), r.states.items[r.best(&tokens, &.{}, &.{}, 0).?].at);
    f.gathers = 0; // a recall must use the disk's staged bytes, never an old borrowed image
    const plan = try s.begin(a, &tokens, 0, &.{}, &.{}, null, &.{});
    defer a.free(plan.marks);
    try std.testing.expectEqual(@as(u32, 900), plan.from);
    _ = f.pass(&s, plan);
    try std.testing.expect(f.restored_ok);
    try std.testing.expectEqual(fake.fresh(&tokens), f.sum);
    // A deterministic target's next tokens depend on all prompt rows, so compare a complete reply vector.
    var resumed = f.sum;
    var fresh = fake.fresh(&tokens);
    for (0..32) |_| {
        const got: u32 = @truncate(resumed ^ (resumed >> 32));
        const want: u32 = @truncate(fresh ^ (fresh >> 32));
        try std.testing.expectEqual(want, got);
        resumed = resumed *% 31 +% got;
        fresh = fresh *% 31 +% want;
    }
}

test "expiry spill: the latest family-expired depth restores its complete image and fresh reply" {
    for ([_]bool{ true, false }) |sync| for ([_]rows.Spill{ .evicted, .finish, .off }) |spill| try run(sync, spill);
}

test "expiry spill: an unreadable image reports one refusal before allocation-free expiry" {
    const tmp = if (platform.c.getenv("TMPDIR")) |p| std.mem.span(p) else ".";
    const root = try std.fmt.allocPrint(a, "{s}/tf-expiry-fail-{d}", .{ tmp, platform.c.getpid() });
    defer a.free(root);
    defer @import("prompt_imprint_test.zig").rmTree(root);
    for ([_]bool{ true, false }) |sync| {
        var f: fake.Fake = .{ .gpa = a, .gathers = 0 };
        var image = f.snapshots().vtable.rows.?.*;
        image.transient = transient;
        var table = f.snapshots().vtable.*;
        table.rows = &image;
        var s = pc.Store.init(a, .{ .ptr = &f, .vtable = &table }, .{ .lookahead = 1, .min_prompt = 0 }, 1 << 30);
        const r = try rows.attach(a, &s, .{ .root = root, .identity = 17 + @as(u64, @intFromBool(sync)), .cap = 1 << 30, .daily = 1 << 30, .floor = 0, .spill = .evicted, .sync_writes = sync });
        defer r.deinit();
        defer s.deinit();
        var heard: notes.Heard = .{};
        notes.sink = &heard;
        defer notes.sink = null;
        const tokens: [1000]u32 = @splat(7);
        f.prompt = &tokens;
        f.at = 900;
        try std.testing.expect(s.keep(&tokens, 900, null, &.{}, &.{}));
        s.expired(s.entries.items[0].saved);
        s.collectExpired();
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, heard.text(), "900 tokens"));
        try std.testing.expect(std.mem.indexOf(u8, heard.text(), "GatherFailed") != null);
        try std.testing.expectEqual(@as(usize, 0), r.states.items.len);
    }
}

test "expiry spill: allocation refusal while queuing a mutable image is named once" {
    const tmp = if (platform.c.getenv("TMPDIR")) |p| std.mem.span(p) else ".";
    const root = try std.fmt.allocPrint(a, "{s}/tf-expiry-oom-{d}", .{ tmp, platform.c.getpid() });
    defer a.free(root);
    defer @import("prompt_imprint_test.zig").rmTree(root);
    var f: fake.Fake = .{ .gpa = a };
    var image = f.snapshots().vtable.rows.?.*;
    image.transient = transient;
    var table = f.snapshots().vtable.*;
    table.rows = &image;
    var s = pc.Store.init(a, .{ .ptr = &f, .vtable = &table }, .{ .lookahead = 1, .min_prompt = 0 }, 1 << 30);
    defer {
        const disk = s.disk;
        s.deinit();
        if (disk) |r| r.deinit();
    }
    const tokens: [1000]u32 = @splat(7);
    f.prompt = &tokens;
    f.at = 900;
    try std.testing.expect(s.keep(&tokens, 900, null, &.{}, &.{}));
    const r = try rows.attach(a, &s, .{ .root = root, .identity = 21, .cap = 1 << 30, .daily = 1 << 30, .floor = 0, .planned = true, .spill = .evicted });
    // Both allocations in the queue path: the chunk starts, then the queue's backing storage.
    for (0..2) |index| {
        var heard: notes.Heard = .{};
        notes.sink = &heard;
        defer notes.sink = null;
        var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = index });
        r.gpa = failing.allocator();
        rows.write.queued(r, s.entries.items[0], &.{ 256, 512 });
        r.gpa = a;
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, heard.text(), "900 tokens"));
        try std.testing.expect(std.mem.indexOf(u8, heard.text(), "OutOfMemory") != null);
        try std.testing.expectEqual(@as(usize, 0), r.queue.items.len);
    }
}
