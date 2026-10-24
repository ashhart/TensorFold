//! Erasure covers saved conversations after their RAM entry leaves, and cancels writes that could restore them.
const std = @import("std");
const libc = @import("sys").c;
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const Fake = @import("rows_fake.zig").Fake;
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const t = std.testing;

test "erase saved conversation removes disk-only states and pending writes across restart" {
    const gpa = t.allocator;
    var buf: [128]u8 = undefined;
    const dir = try std.fmt.bufPrint(&buf, "/tmp/tf-rows-erase-{d}", .{libc.getpid()});
    defer rmTree(dir);
    const options: rows.Options = .{ .root = dir, .identity = 7, .cap = 1 << 30, .floor = 0, .daily = 1 << 30, .lookahead = 1 };
    var bad: [700]u32 = undefined;
    for (&bad, 0..) |*v, i| v.* = @intCast(1000 + i);
    const span = [_]u32{ 80, 81, 82, 83 };
    @memcpy(bad[350..354], &span);
    var good = bad;
    @memcpy(good[350..354], &[_]u32{ 90, 91, 92, 93 });
    var pending = bad;
    pending[550] = 9999;
    var f: Fake = .{ .gpa = gpa };
    {
        const r = try rows.Rows.open(gpa, f.snapshots(), options);
        defer r.deinit();
        var s = pc.Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 }, 1 << 25);
        defer s.deinit();
        s.disk = r;
        for ([_][]const u32{ &bad, &good }) |prompt| {
            f.prompt = prompt;
            f.at = 600;
            try t.expect(s.keep(prompt, 600, null, &.{}, &.{}));
            while (rows.write.step(r)) {}
        }
        try t.expectEqual(@as(usize, 2), r.states.items.len);
        s.remove(0);
        f.prompt = &pending;
        f.at = 600;
        try t.expect(s.keep(&pending, 600, null, &.{}, &.{}));
        try t.expect(rows.write.step(r));
        try t.expect(r.job != null);
        const erased = s.forgetSpan(&span, &.{});
        try t.expect(erased.complete());
        try t.expectEqual(@as(usize, 1), s.entries.items.len);
        try t.expect(r.job == null);
        try t.expectEqual(@as(usize, 0), r.queue.items.len);
        try t.expectEqual(@as(usize, 1), r.states.items.len);
        try t.expect(r.best(&bad, &.{}, &.{}, 0) == null);
        try t.expect(r.best(&good, &.{}, &.{}, 0) != null);
        while (rows.write.step(r)) {}
    }
    const reopened = try rows.Rows.open(gpa, f.snapshots(), options);
    defer reopened.deinit();
    try t.expectEqual(@as(usize, 1), reopened.states.items.len);
    try t.expect(reopened.best(&bad, &.{}, &.{}, 0) == null);
    try t.expect(reopened.best(&pending, &.{}, &.{}, 0) == null);
    try t.expect(reopened.best(&good, &.{}, &.{}, 0) != null);
}

test "erase saved conversation reports a failed unlink and never claims its remaining copy clear" {
    const gpa = t.allocator;
    var buf: [128]u8 = undefined;
    const dir = try std.fmt.bufPrint(&buf, "/tmp/tf-rows-erase-fault-{d}", .{libc.getpid()});
    defer rmTree(dir);
    const prompt = [_]u32{ 1, 2, 80, 81, 82, 83, 7, 8, 9, 10 };
    var f: Fake = .{ .gpa = gpa, .prompt = &prompt, .at = 8 };
    const r = try rows.Rows.open(gpa, f.snapshots(), .{
        .root = dir,
        .identity = 7,
        .cap = 1 << 30,
        .floor = 0,
        .daily = 1 << 30,
        .lookahead = 1,
    });
    defer r.deinit();
    var s = pc.Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 }, 1 << 25);
    defer s.deinit();
    s.disk = r;
    try t.expect(s.keep(&prompt, 8, null, &.{}, &.{}));
    while (rows.write.step(r)) {}
    var path_buf: [1200]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buf, "{s}/{s}.st", .{
        r.dir, std.fmt.bytesToHex(rows.chain.stem(r.states.items[0].name), .lower),
    }, 0);
    var parked_buf: [1200]u8 = undefined;
    const parked = try std.fmt.bufPrintSentinel(&parked_buf, "{s}.saved", .{path}, 0);
    try t.expectEqual(@as(c_int, 0), libc.rename(path, parked));
    try t.expectEqual(@as(c_int, 0), libc.mkdir(path, 0o700));
    const failed = s.forgetSpan(&.{ 80, 81, 82, 83 }, &.{});
    try t.expect(failed.failed > 0);
    try t.expect(!failed.complete());
    try t.expectEqual(@as(usize, 1), r.states.items.len);
    try t.expectEqual(@as(c_int, 0), libc.rmdir(path));
    try t.expectEqual(@as(c_int, 0), libc.rename(parked, path));
    const retried = s.forgetSpan(&.{ 80, 81, 82, 83 }, &.{});
    try t.expect(!retried.complete());
    try t.expectEqual(@as(usize, 0), r.states.items.len);
}

test "erase saved conversation allocation failure preserves the disk state and reports incomplete" {
    const gpa = t.allocator;
    var buf: [128]u8 = undefined;
    const dir = try std.fmt.bufPrint(&buf, "/tmp/tf-rows-erase-oom-{d}", .{libc.getpid()});
    defer rmTree(dir);
    var prompt: [700]u32 = undefined;
    for (&prompt, 0..) |*v, i| v.* = @intCast(i + 1000);
    const span = prompt[350..354];
    var f: Fake = .{ .gpa = gpa, .prompt = &prompt, .at = 600 };
    const r = try rows.Rows.open(gpa, f.snapshots(), .{
        .root = dir,
        .identity = 7,
        .cap = 1 << 30,
        .floor = 0,
        .daily = 1 << 30,
        .lookahead = 1,
    });
    defer r.deinit();
    var s = pc.Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 }, 1 << 25);
    defer s.deinit();
    s.disk = r;
    try t.expect(s.keep(&prompt, 600, null, &.{}, &.{}));
    while (rows.write.step(r)) {}
    s.remove(0);
    var failing: std.testing.FailingAllocator = .init(gpa, .{ .fail_index = 0 });
    r.gpa = failing.allocator();
    const result = s.forgetSpan(span, &.{});
    r.gpa = gpa;
    try t.expect(failing.has_induced_failure);
    try t.expect(result.failed > 0);
    try t.expect(!result.complete());
    try t.expectEqual(@as(usize, 1), r.states.items.len);
    try t.expect(r.best(&prompt, &.{}, &.{}, 0) != null);
}
