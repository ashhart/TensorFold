//! The bank's durable state: a compaction interrupted before its journal goes replays nothing twice.
const std = @import("std");
const safety = @import("safety_test.zig");
const Bank = @import("bank.zig").Bank;
const store = @import("store.zig");
const t = std.testing;

fn bankDir(tmp: *std.testing.TmpDir) ![:0]u8 {
    return tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
}

test "bank durable: a crash after a compaction's base and before its journal delete replays no unit twice" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try bankDir(&tmp);
    defer t.allocator.free(dir);
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const b = try safety.taught(t.allocator, dir, &mock, true); // base written: the next unit goes to the journal
    defer b.deinit();
    const more = [_]u32{ 40, 41, 42, 43 };
    _ = try b.teachChunk(.{ .parent_id = 8, .body_ids = &more, .gate_ids = &more, .is_small_fact = true });
    _ = try b.save();
    try tmp.dir.copyFile(store.delta_file_name, tmp.dir, "journal.kept", t.io, .{});
    _ = try b.saveFull(); // the compaction: both units in the base, then the journal deleted
    try tmp.dir.rename("journal.kept", tmp.dir, store.delta_file_name, t.io); // as if that delete never happened
    const back = try Bank.init(t.allocator, t.io, dir, mock.backend());
    defer back.deinit();
    try t.expect(try back.loadIfPresent());
    try t.expectEqual(@as(usize, 2), back.units.items.len);
}

/// Puts a non-empty directory where the bank must write or delete `name`: every write or delete of it fails.
fn block(tmp: *std.testing.TmpDir, name: []const u8) !void {
    try tmp.dir.createDirPath(t.io, name);
    var d = try tmp.dir.openDir(t.io, name, .{});
    defer d.close(t.io);
    (try d.createFile(t.io, "x", .{})).close(t.io);
}

fn unblock(tmp: *std.testing.TmpDir, name: []const u8) !void {
    try tmp.dir.deleteTree(t.io, name);
}

test "bank durable: a forget whose journal write fails leaves the bank as it was, and one that succeeds outlives a restart" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try bankDir(&tmp);
    defer t.allocator.free(dir);
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const b = try safety.taught(t.allocator, dir, &mock, true);
    defer b.deinit();
    try block(&tmp, store.delta_file_name);
    if (b.forgetDurable(7)) |_| return error.TestUnexpectedResult else |_| {}
    try t.expectEqual(@as(usize, 1), b.units.items.len); // the reply said failure: the fact still answers
    try unblock(&tmp, store.delta_file_name);
    try t.expectEqual(@as(u32, 1), try b.forgetDurable(7));
    try t.expectEqual(@as(usize, 0), b.units.items.len);
    const back = try Bank.init(t.allocator, t.io, dir, mock.backend());
    defer back.deinit();
    try t.expect(try back.loadIfPresent());
    try t.expect(!back.hasUnits()); // the deletion the reply reported is the one on disk
}

test "bank durable: a compacting forget whose base write fails puts back every unit and every revived belief" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try bankDir(&tmp);
    defer t.allocator.free(dir);
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const b = try safety.taught(t.allocator, dir, &mock, false); // no base yet: a forget writes one whole
    defer b.deinit();
    const more = [_]u32{ 40, 41, 42, 43 };
    _ = try b.teachChunk(.{ .parent_id = 8, .body_ids = &more, .gate_ids = &more, .is_small_fact = true });
    try t.expectEqual(@as(u32, 1), b.supersedeParent(8, 7)); // 7 replaced 8: forgetting 7 revives 8
    const ids = [2]u32{ b.units.items[0].unit_id, b.units.items[1].unit_id };
    try block(&tmp, store.file_name);
    if (b.forgetDurable(7)) |_| return error.TestUnexpectedResult else |_| {}
    try t.expectEqual(@as(usize, 2), b.units.items.len);
    try t.expectEqual(ids, [2]u32{ b.units.items[0].unit_id, b.units.items[1].unit_id });
    try t.expectEqual(@as(u32, 7), b.units.items[1].gate.superseded_by);
    try unblock(&tmp, store.file_name);
    try t.expectEqual(@as(u32, 1), try b.forgetDurable(7));
    try t.expectEqual(@as(usize, 1), b.units.items.len);
    try t.expectEqual(@as(u32, 0), b.units.items[0].gate.superseded_by);
}

test "bank durable: a clear whose files cannot go keeps every unit and says so" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try bankDir(&tmp);
    defer t.allocator.free(dir);
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const b = try safety.taught(t.allocator, dir, &mock, true);
    defer b.deinit();
    try block(&tmp, store.delta_file_name);
    if (b.clearDurable()) |_| return error.TestUnexpectedResult else |_| {}
    try t.expectEqual(@as(usize, 1), b.units.items.len);
    try unblock(&tmp, store.delta_file_name);
    try t.expectEqual(@as(u32, 1), try b.clearDurable());
    try t.expect(!b.hasUnits());
}
