//! Lesson publication is one transaction across live units, replacement gates, and durable frames.
const std = @import("std");
const safety = @import("safety_test.zig");
const Bank = @import("bank.zig").Bank;
const store = @import("store.zig");
const t = std.testing;

fn bankDir(tmp: *std.testing.TmpDir) ![:0]u8 {
    return tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
}

fn routes(b: anytype, mock: *safety.Mock, ids: []const u32) !bool {
    const snap = (try b.routeAndCompose(ids)) orelse return false;
    mock.backend().dropSnap(mock, snap.saved);
    return true;
}

const journal_backup = "living-weights.delta.saved";

fn blockJournal(tmp: *std.testing.TmpDir) !void {
    std.Io.Dir.rename(tmp.dir, store.delta_file_name, tmp.dir, journal_backup, t.io) catch |e| {
        if (e != error.FileNotFound) return e;
    };
    try tmp.dir.createDirPath(t.io, store.delta_file_name);
    var d = try tmp.dir.openDir(t.io, store.delta_file_name, .{});
    defer d.close(t.io);
    (try d.createFile(t.io, "x", .{})).close(t.io);
}

fn unblockJournal(tmp: *std.testing.TmpDir) !void {
    try tmp.dir.deleteTree(t.io, store.delta_file_name);
    try std.Io.Dir.rename(tmp.dir, journal_backup, tmp.dir, store.delta_file_name, t.io);
}

fn failDirectorySync(_: std.Io, _: []const u8) anyerror!void {
    return error.AccessDenied;
}

test "bank lesson: a failed multi-unit replacement restores live state and leaves reload unchanged" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try bankDir(&tmp);
    defer t.allocator.free(dir);
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const b = try safety.taught(t.allocator, dir, &mock, true);
    defer b.deinit();
    const old_ids = [_]u32{ 40, 41, 42, 43 };
    _ = try b.teachChunk(.{ .parent_id = 8, .body_ids = &old_ids, .gate_ids = &old_ids, .is_small_fact = true });
    _ = try b.save();
    try t.expectEqual(@as(u32, 1), try b.supersedeDurable(8, 7));
    const old_uid = [2]u32{ b.units.items[0].unit_id, b.units.items[1].unit_id };
    const old_version = b.version;
    const new_a = [_]u32{ 50, 51, 52, 53 };
    const new_b = [_]u32{ 60, 61, 62, 63 };
    _ = try b.teachChunk(.{ .parent_id = 9, .body_ids = &new_a, .gate_ids = &new_a, .is_small_fact = true, .staged = true });
    _ = try b.teachChunk(.{ .parent_id = 9, .body_ids = &new_b, .gate_ids = &new_b, .is_small_fact = true, .staged = true });
    const staged_next_uid = b.next_unit_id;
    const staged_next_epoch = b.next_epoch;

    try blockJournal(&tmp);
    if (b.commitLesson(9, &.{7}, 8)) |_| return error.TestUnexpectedResult else |_| {}
    try t.expectEqual(@as(usize, 2), b.units.items.len);
    try t.expectEqual(old_uid, [2]u32{ b.units.items[0].unit_id, b.units.items[1].unit_id });
    try t.expectEqual(@as(u32, 0), b.units.items[0].gate.superseded_by);
    try t.expectEqual(@as(u32, 7), b.units.items[1].gate.superseded_by);
    try t.expectEqual(@as(usize, 0), b.staged.items.len);
    try t.expectEqual(old_version, b.version);
    try t.expectEqual(staged_next_uid, b.next_unit_id);
    try t.expectEqual(staged_next_epoch, b.next_epoch);
    try unblockJournal(&tmp);

    const back = try Bank.init(t.allocator, t.io, dir, mock.backend());
    try t.expect(try back.loadIfPresent());
    defer back.deinit();
    try t.expectEqual(@as(u32, 8), back.maxParentId()); // no unit from the failed parent 9 was replayed
    try t.expectEqual(@as(usize, 2), back.units.items.len);
    try t.expectEqual(old_uid, [2]u32{ back.units.items[0].unit_id, back.units.items[1].unit_id });
    try t.expectEqual(@as(u32, 7), back.units.items[1].gate.superseded_by);
}

test "bank lesson: directory-sync failure after batch rename keeps the committed replacement" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try bankDir(&tmp);
    defer t.allocator.free(dir);
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const b = try safety.taught(t.allocator, dir, &mock, true);
    defer b.deinit();
    const new_a = [_]u32{ 50, 51, 52, 53 };
    _ = try b.teachChunk(.{ .parent_id = 8, .body_ids = &new_a, .gate_ids = &new_a, .is_small_fact = true, .staged = true });
    _ = try b.teachChunk(.{ .parent_id = 8, .body_ids = &new_a, .gate_ids = &new_a, .is_small_fact = true, .staged = true });

    try t.expectEqual(@as(u32, 2), try b.commitLessonWithSync(8, &.{7}, null, failDirectorySync));
    try t.expectEqual(@as(usize, 2), b.units.items.len);
    for (b.units.items) |u| try t.expectEqual(@as(u32, 8), u.parent_id);
    try t.expectEqual(b.unflushed_from, b.units.items.len);
    const ask = [_]u32{ 50, 51, 52 };
    try t.expect(try routes(b, &mock, &ask));

    const back = try Bank.init(t.allocator, t.io, dir, mock.backend());
    defer back.deinit();
    try t.expect(try back.loadIfPresent());
    try t.expectEqual(@as(usize, 2), back.units.items.len);
    for (back.units.items) |u| try t.expectEqual(@as(u32, 8), u.parent_id);
    try t.expect(try routes(back, &mock, &ask));
}

test "bank consolidate: a post-rename sync error keeps the pruned candidate live and reloadable" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try bankDir(&tmp);
    defer t.allocator.free(dir);
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const b = try safety.taught(t.allocator, dir, &mock, true);
    defer b.deinit();
    const old_ids = [_]u32{ 40, 41, 42, 43 };
    _ = try b.teachChunk(.{ .parent_id = 8, .body_ids = &old_ids, .gate_ids = &old_ids, .is_small_fact = true });
    _ = try b.save();
    try t.expectEqual(@as(u32, 1), try b.supersedeDurable(8, 7));
    var outcome: @import("lesson.zig").ConsolidateOutcome = .{};
    try t.expectError(error.AccessDenied, @import("lesson.zig").consolidateWithSync(b, true, false, failDirectorySync, &outcome));
    try t.expect(outcome.committed);
    try t.expectEqual(@as(u32, 1), outcome.report.pruned_superseded);
    try t.expectEqual(@as(usize, 1), b.units.items.len);
    try t.expectEqual(@as(u32, 7), b.units.items[0].parent_id);

    const back = try Bank.init(t.allocator, t.io, dir, mock.backend());
    defer back.deinit();
    try t.expect(try back.loadIfPresent());
    try t.expectEqual(@as(usize, 1), back.units.items.len);
    try t.expectEqual(@as(u32, 7), back.units.items[0].parent_id);
}

test "bank supersede: a failed non-standing save preserves compacted skill ownership" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try bankDir(&tmp);
    defer t.allocator.free(dir);
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const b = try safety.taught(t.allocator, dir, &mock, true);
    defer b.deinit();
    const pol1 = [_]u32{ 50, 51, 52 };
    const pol2 = [_]u32{ 53, 54, 55 };
    _ = try b.teachChunk(.{ .parent_id = 10, .body_ids = &pol1, .gate_ids = &.{}, .is_standing = true });
    _ = try b.teachChunk(.{ .parent_id = 11, .body_ids = &pol2, .gate_ids = &.{}, .is_standing = true });
    _ = try b.consolidate(false, true);
    try t.expectEqual(@as(usize, 2), b.skill_rec.len);
    try t.expectEqual(@as(u32, 2), b.skill_policy_count);

    const skill_values = try t.allocator.alloc(f32, b.skill_rec.len * b.skill_rec[0].delta.len);
    defer t.allocator.free(skill_values);
    for (b.skill_rec, 0..) |layer, r| {
        const start = r * layer.delta.len;
        @memcpy(skill_values[start .. start + layer.delta.len], layer.delta);
    }
    var old_gates: [3]u32 = undefined;
    for (b.units.items, 0..) |unit, i| old_gates[i] = unit.gate.superseded_by;
    const old_version = b.version;
    const base_backup = "living-weights.safetensors.saved";
    try std.Io.Dir.rename(tmp.dir, store.file_name, tmp.dir, base_backup, t.io);
    try tmp.dir.createDirPath(t.io, store.file_name);
    var blocked = try tmp.dir.openDir(t.io, store.file_name, .{});
    defer blocked.close(t.io);
    (try blocked.createFile(t.io, "x", .{})).close(t.io);

    if (b.supersedeDurable(7, 12)) |_| return error.TestUnexpectedResult else |_| {}
    try t.expectEqual(@as(usize, 3), b.units.items.len);
    for (b.units.items, 0..) |unit, i| try t.expectEqual(old_gates[i], unit.gate.superseded_by);
    try t.expectEqual(old_version, b.version);
    try t.expectEqual(@as(usize, 2), b.skill_rec.len);
    try t.expectEqual(@as(u32, 2), b.skill_policy_count);
    for (b.skill_rec, 0..) |layer, r| {
        const start = r * layer.delta.len;
        try t.expectEqualSlices(f32, skill_values[start .. start + layer.delta.len], layer.delta);
    }
}

test "bank lesson: an early setup allocation failure discards the staged lesson" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try bankDir(&tmp);
    defer t.allocator.free(dir);
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    var failing = t.FailingAllocator.init(t.allocator, .{});
    const b = try safety.taught(failing.allocator(), dir, &mock, false);
    defer b.deinit();
    const ids = [_]u32{ 40, 41, 42, 43 };
    _ = try b.teachChunk(.{ .parent_id = 8, .body_ids = &ids, .gate_ids = &ids, .is_small_fact = true, .staged = true });
    try t.expectEqual(@as(usize, 1), b.staged.items.len);

    failing.fail_index = failing.alloc_index;
    try t.expectError(error.OutOfMemory, b.commitLesson(8, &.{}, null));
    failing.fail_index = std.math.maxInt(usize);
    try t.expectEqual(@as(usize, 1), b.units.items.len);
    try t.expectEqual(@as(usize, 0), b.staged.items.len);
    _ = try b.teachChunk(.{ .parent_id = 8, .body_ids = &ids, .gate_ids = &ids, .is_small_fact = true, .staged = true });
    try t.expectEqual(@as(u32, 1), try b.commitLesson(8, &.{}, null));
    try t.expectEqual(@as(usize, 2), b.units.items.len);
    try t.expectEqual(@as(usize, 0), b.staged.items.len);
}
