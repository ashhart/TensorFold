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
