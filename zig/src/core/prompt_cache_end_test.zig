//! ROUND-502: the pass's own end as a kept state -- class-checked when it is saved and again when a later
//! turn's suffix is measured, so a turn that adds a tensor-tile chunk resumes at the end and a small turn
//! still falls back to the tile tail.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Fake = @import("prompt_cache_test.zig").Fake;
const fresh = @import("prompt_cache_test.zig").fresh;
const Store = pc.Store;

fn rules(end_mark: bool) pc.Rules {
    return .{ .planned = true, .grid = 256, .tail = true, .tail_align = 128, .min_prompt = 0, .end_mark = end_mark };
}

fn prompt(buf: []u32) []const u32 {
    for (buf, 0..) |*t, i| t.* = @intCast(1 + (i * 7) % 9000);
    return buf;
}

test "the pass's end is kept, and a tensor-tile turn resumes there and equals a fresh pass" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), rules(true), 1 << 20);
    defer s.deinit();
    var b1: [320]u32 = undefined;
    const p1 = prompt(&b1);
    var plan = try s.begin(a, p1, @intCast(p1.len), &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 256, 320 }, plan.marks); // the tile tail and the end
    try std.testing.expectEqual(fresh(p1), f.pass(&s, p1, plan));
    try std.testing.expectEqual(@as(u32, 256), s.find(p1, &.{}, &.{}).?.at); // an end never resumes its own prompt

    var b2: [480]u32 = undefined;
    const p2 = prompt(&b2);
    plan = try s.begin(a, p2, @intCast(p2.len), &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 320), plan.from); // the end (320), not the tile tail (256)
    try std.testing.expectEqual(fresh(p2), f.pass(&s, p2, plan));
}

test "a small turn falls back to the tile tail: the end's 20 rows are the shared-body class" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), rules(true), 1 << 20);
    defer s.deinit();
    var b1: [320]u32 = undefined;
    const p1 = prompt(&b1);
    _ = f.pass(&s, p1, try s.begin(a, p1, @intCast(p1.len), &.{}, &.{}, null, &.{}));
    var b2: [480]u32 = undefined;
    const p2 = prompt(&b2);
    _ = f.pass(&s, p2, try s.begin(a, p2, @intCast(p2.len), &.{}, &.{}, null, &.{}));

    var b3: [500]u32 = undefined;
    const p3 = prompt(&b3);
    const plan = try s.begin(a, p3, @intCast(p3.len), &.{}, &.{}, null, &.{});
    // ROUND-506: the resumed pass kept a tail at 384 (96 rows before 480), so the 20-row turn resumes there
    // (116 rows, class 2) instead of the end two turns back (180 rows).
    try std.testing.expectEqual(@as(u32, 384), plan.from);
    try std.testing.expectEqual(fresh(p3), f.pass(&s, p3, plan));
}

test "a suffix past one chunk falls back to the grid state" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), rules(true), 1 << 20);
    defer s.deinit();
    var b1: [320]u32 = undefined;
    const p1 = prompt(&b1);
    _ = f.pass(&s, p1, try s.begin(a, p1, @intCast(p1.len), &.{}, &.{}, null, &.{}));
    var b2: [700]u32 = undefined;
    const p2 = prompt(&b2);
    _ = f.pass(&s, p2, try s.begin(a, p2, @intCast(p2.len), &.{}, &.{}, null, &.{}));

    var b3: [800]u32 = undefined;
    const p3 = prompt(&b3);
    const plan = try s.begin(a, p3, @intCast(p3.len), &.{}, &.{}, null, &.{});
    // ROUND-506: the resumed pass kept its last tile at 512 -- grid-aligned (512 % 256 == 0), so it is a grid
    // state and the 288-row suffix is one chunk. The 256 grid is no longer the nearest; 512 serves.
    try std.testing.expectEqual(@as(u32, 512), plan.from);
    try std.testing.expectEqual(fresh(p3), f.pass(&s, p3, plan));
}

test "without the flag the plan is the shipped one" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), rules(false), 1 << 20);
    defer s.deinit();
    var b1: [320]u32 = undefined;
    const p1 = prompt(&b1);
    const plan = try s.begin(a, p1, @intCast(p1.len), &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{256}, plan.marks);
    _ = f.pass(&s, p1, plan);
    var b2: [480]u32 = undefined;
    const p2 = prompt(&b2);
    const p = try s.begin(a, p2, @intCast(p2.len), &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 256), p.from);
}

test "the real grid: the 21,966 -> 22,705 -> 22,742 conversation plans one chunk a turn" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 15_872, .tail = true, .tail_align = 128, .min_prompt = 0, .end_mark = true }, 1 << 22);
    defer s.deinit();
    const p1 = try a.alloc(u32, 21_966);
    defer a.free(p1);
    for (p1, 0..) |*t, i| t.* = @intCast(1 + (i * 7) % 9000);
    var plan = try s.begin(a, p1, @intCast(p1.len), &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 21_888, 21_966 }, plan.marks); // cold: the tile tail and the end
    try std.testing.expectEqual(fresh(p1), f.pass(&s, p1, plan));

    const p2 = try a.alloc(u32, 22_705);
    defer a.free(p2);
    for (p2, 0..) |*t, i| t.* = @intCast(1 + (i * 7) % 9000);
    plan = try s.begin(a, p2, @intCast(p2.len), &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 21_966), plan.from); // the end, not 21,888
    // ROUND-506: the resumed pass also plans its last tile as a SNAPSHOT, 177 rows before its end (22,528),
    // so the next turn can start there without cutting this pass.
    try std.testing.expectEqualSlices(u32, &.{ 22_528, 22_705 }, plan.marks);
    try std.testing.expectEqual(fresh(p2), f.pass(&s, p2, plan));

    const p3 = try a.alloc(u32, 22_742);
    defer a.free(p3);
    for (p3, 0..) |*t, i| t.* = @intCast(1 + (i * 7) % 9000);
    plan = try s.begin(a, p3, @intCast(p3.len), &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 22_528), plan.from); // 37 rows are the shared class; the tail is 214 rows
    try std.testing.expectEqual(fresh(p3), f.pass(&s, p3, plan));
}
