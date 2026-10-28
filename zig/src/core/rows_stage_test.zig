//! Staged writes that share blocks: a block still being written is charged and gathered once, and a state waits for it.
const std = @import("std");
const rows = @import("rows_store.zig");
const fixture = @import("rows_async_fixture.zig");
const Fixture = fixture.Fixture;
const Hold = fixture.Gate;
const t = std.testing;

/// A prompt that shares its first 513 tokens with `fixture.prompt(17)`, so their first two blocks are the same files.
fn sibling(salt: u32) [1100]u32 {
    var p = fixture.prompt(salt);
    const base = fixture.prompt(17);
    @memcpy(p[0..513], base[0..513]);
    return p;
}

fn pair() struct { first: [1100]u32, second: [1100]u32 } {
    return .{ .first = fixture.prompt(17), .second = sibling(18) };
}

test "rows stage: a block still being written is charged and gathered once" {
    const x = try Fixture.open("stage-share", .finish, false, 1 << 20);
    defer x.close();
    var hold: Hold = .{};
    defer hold.release.store(true, .release);
    x.r.file_io = hold.io();
    const p = pair();
    try x.keep(&p.first, 900);
    try Hold.wait(&hold.entered);
    try x.keep(&p.second, 900);
    try t.expectEqual(@as(usize, 2), x.r.worker.?.pending());
    const charged = x.r.worker.?.reserved();
    hold.release.store(true, .release);
    try x.settle();
    try t.expectEqual(@as(usize, 2), x.r.states.items.len);
    try t.expectEqual(x.r.held(), charged);
    try t.expectEqual(x.r.blocks.count() + 2, x.f.gathered);
    const saved = try rows.read.read(x.r, 1);
    defer x.f.snapshots().vtable.drop(&x.f, saved);
    try x.f.snapshots().vtable.restore(&x.f, null, saved);
    try t.expect(x.f.restored_ok);
}

test "rows stage: a state whose earlier job failed to write its block is not published" {
    const x = try Fixture.open("stage-lost", .finish, false, 1 << 20);
    defer x.close();
    var hold: Hold = .{ .mode = .first_fails };
    defer hold.release.store(true, .release);
    x.r.file_io = hold.io();
    const p = pair();
    try x.keep(&p.first, 900);
    try Hold.wait(&hold.entered);
    try x.keep(&p.second, 900);
    try t.expectEqual(@as(usize, 2), x.r.worker.?.pending());
    hold.release.store(true, .release);
    try x.settle();
    try t.expectEqual(@as(usize, 0), x.r.states.items.len);
    try t.expectEqual(@as(usize, 0), fixture.files(x.r.dir, ".st"));
    try t.expectEqual(@as(usize, 0), fixture.files(x.r.dir, ".blk"));
    try t.expectEqual(@as(usize, 0), fixture.files(x.r.dir, ".part"));
    try t.expectEqual(@as(u64, 2), x.r.failures.count);
    try t.expectEqual(error.NoSpaceLeft, x.r.failures.records[0].?.err);
    try t.expectEqual(error.MissingBlock, x.r.failures.records[1].?.err);
    try t.expectEqual(@as(u64, 2), x.r.counts.failed);
    try t.expectEqual(@as(u64, 0), x.r.counts.writes);
    try t.expectEqual(@as(u64, 0), x.r.admission.reserved);
}

test "rows stage: a block its writer failed to write is written by the next save, though a job that skipped it is in flight" {
    const x = try Fixture.open("stage-chain", .finish, false, 1 << 20);
    defer x.close();
    var failing: Hold = .{ .mode = .first_fails };
    defer failing.release.store(true, .release);
    var held: Hold = .{};
    defer held.release.store(true, .release);
    x.r.file_io = failing.io();
    const first = fixture.prompt(17);
    const second = sibling(18);
    const third = sibling(19);
    try x.keep(&first, 900);
    try Hold.wait(&failing.entered);
    try x.keep(&second, 900);
    x.r.worker.?.io = held.io(); // what the worker writes after the first job waits on `held`
    failing.release.store(true, .release);
    try Hold.wait(&held.entered);
    rows.write.reap(x.r);
    x.r.admission.retry_at = 0; // the back-off the first failure set has run out
    try t.expectEqual(@as(usize, 1), x.r.worker.?.pending());
    try x.keep(&third, 900);
    try t.expectEqual(@as(usize, 2), x.r.worker.?.pending());
    held.release.store(true, .release);
    try x.settle();
    try t.expectEqual(@as(usize, 1), x.r.states.items.len);
    try t.expectEqual(error.NoSpaceLeft, x.r.failures.records[0].?.err);
    try t.expectEqual(error.MissingBlock, x.r.failures.records[1].?.err);
    try t.expectEqual(@as(u64, 2), x.r.counts.failed);
    try t.expectEqual(@as(usize, 0), fixture.files(x.r.dir, ".part"));
    const saved = try rows.read.read(x.r, 0);
    defer x.f.snapshots().vtable.drop(&x.f, saved);
    try x.f.snapshots().vtable.restore(&x.f, null, saved);
    try t.expect(x.f.restored_ok);
}

test "rows stage: a job a fence cancelled promises no block, so the next save writes its own" {
    const x = try Fixture.open("stage-fence", .finish, false, 1 << 20);
    defer x.close();
    var hold: Hold = .{};
    defer hold.release.store(true, .release);
    x.r.file_io = hold.io();
    const p = pair();
    try x.keep(&p.first, 900);
    try Hold.wait(&hold.entered);
    rows.write.fence(x.r);
    try x.keep(&p.second, 900);
    hold.release.store(true, .release);
    try x.settle();
    try t.expectEqual(@as(usize, 1), x.r.states.items.len);
    try t.expectEqual(@as(u64, 1), x.r.counts.failed);
    try t.expectEqual(error.Cancelled, x.r.failures.records[0].?.err);
    const saved = try rows.read.read(x.r, 0);
    defer x.f.snapshots().vtable.drop(&x.f, saved);
    try x.f.snapshots().vtable.restore(&x.f, null, saved);
    try t.expect(x.f.restored_ok);
}
