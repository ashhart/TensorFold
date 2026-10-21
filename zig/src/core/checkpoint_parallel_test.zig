//! Serial and bounded parallel readers must agree on whole files, projections and every persisted record.
const std = @import("std");
const Content = @import("checkpoint_identity.zig").Content;
const fixture = @import("digest_fixture.zig");
const t = std.testing;

test "checkpoint parallel: worker limits reserve serving CPUs and bound storage and buffers" {
    try t.expectEqual(@as(usize, 0), Content.workerCount(20, 2, 0, 8));
    try t.expectEqual(@as(usize, 1), Content.workerCount(1, 2, 9, 8));
    try t.expectEqual(@as(usize, 2), Content.workerCount(4, 2, 9, 8));
    try t.expectEqual(@as(usize, 3), Content.workerCount(20, 2, 9, 3));
    try t.expectEqual(@as(usize, 4), Content.workerCount(20, 2, 4, 8));
    try t.expectEqual(@as(usize, 8), Content.workerCount(128, 2, 100, 128));
}

test "checkpoint parallel: serial software and parallel auto produce equal records and whole and split identities" {
    if (!@import("digest_key.zig").persistent) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const a = t.allocator;
    const io = t.io;
    try fixture.write(a, io, tmp.dir, 8, 1 << 20);
    const path = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(path);
    const serial = try std.fs.path.join(a, &.{ path, "serial" });
    defer a.free(serial);
    const parallel = try std.fs.path.join(a, &.{ path, "parallel" });
    defer a.free(parallel);
    const before = try (try Content.startWithOptions(a, io, path, false, serial, .{ .storage_workers = 1, .hash_mode = .software })).finishWithStats();
    const after = try (try Content.startWithOptions(a, io, path, false, parallel, .{})).finishWithStats();
    try t.expectEqual(before.digest, after.digest);
    try t.expectEqual(before.bytes_read, after.bytes_read);
    try t.expect(before.bytes_read > 8 << 20);
    try t.expectEqual(@as(usize, 0), before.cache_hits);
    try t.expectEqual(@as(usize, 0), after.cache_hits);
    try fixture.equalRecords(a, io, serial, parallel, 8);
    const shared_before = try (try Content.startWithOptions(a, io, path, true, serial, .{ .storage_workers = 1 })).finishWithStats();
    const shared_after = try (try Content.startWithOptions(a, io, path, true, parallel, .{})).finishWithStats();
    try t.expectEqual(shared_before.digest, shared_after.digest);
    try t.expectEqual(@as(usize, 8), shared_after.cache_hits);
    try t.expectEqual(@as(u64, 0), shared_after.bytes_read);
    try t.expectEqual(@as(u64, 0), shared_before.bytes_read);
    try t.expect(!std.mem.eql(u8, &before.digest, &shared_after.digest));
}
