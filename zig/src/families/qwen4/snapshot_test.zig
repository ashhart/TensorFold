//! Native snapshot packing round-trips actual frame-owner bytes across capacities and fails closed before writes.
const std = @import("std");
const cfg = @import("config.zig");
const support = @import("state_test.zig");
const State = @import("state.zig").State(support.Backend);
const Snap = @import("snapshot.zig").Snapshot(support.Backend);
const a = std.testing.allocator;
fn prefix(s: *State) ![]u8 {
    const v = try s.current(.{ .bytes = s.layout.state_bytes });
    return v.buffer[v.offset..][0..v.len];
}
test "Qwen4 prompt state: exact frame and KV prefix, complete pooled blocks, different destination capacity" {
    var p: cfg.Problem = .{};
    var c = try cfg.parse(a, @embedFile("fixtures/tiny-dense.json"), &p);
    defer c.deinit();
    var b: support.Backend = .{};
    var s = try State.open(a, &b, &c, 32, 4, std.math.maxInt(u64));
    defer s.close() catch unreachable;
    var dest = try State.open(a, &b, &c, 64, 4, std.math.maxInt(u64));
    defer dest.close() catch unreachable;
    try s.reset();
    try s.begin(3);
    try support.complete(&s, &.{ 1, 2, 3 });
    try s.keep(3);
    const l = s.layout.layers[1];
    const d = dest.layout.layers[1];
    for ([_]@import("state_layout.zig").Range{ l.key, l.value, l.index, l.pooled }, 0..) |range, i| @memset(s.buffer[range.offset..][0..range.bytes], @intCast(41 + i));
    const charge = try Snap.charge(&s);
    var snap = try Snap.open(a, &s, @splat(1), charge);
    defer snap.close() catch unreachable;
    try snap.capture(&s, @splat(1));
    try snap.restore(&dest, @splat(1));
    try std.testing.expectEqualSlices(u8, try prefix(&s), try prefix(&dest));
    try std.testing.expectEqualSlices(u32, s.history(), dest.history());
    const lengths = [_]usize{ 3 * c.kv_heads * c.head_dim * 2, 3 * c.kv_heads * c.head_dim * 2, 3 * c.index_dim * 2, 3 / c.index_ratio * c.index_dim * 2 };
    for ([_]@import("state_layout.zig").Range{ d.key, d.value, d.index, d.pooled }, lengths, 0..) |range, n, i| {
        for (dest.buffer[range.offset..][0..n]) |byte| try std.testing.expectEqual(@as(u8, @intCast(41 + i)), byte);
        try std.testing.expectEqual(@as(u8, 0xcc), dest.buffer[range.offset + n]);
    }
    try std.testing.expectEqual(charge, snap.charged);
    try dest.begin(1);
    try support.complete(&dest, &.{4});
    try dest.keep(1);
    try std.testing.expectEqual(@as(usize, 4), dest.pos);
}
test "Qwen4 prompt state: budget, identity and unpublished snapshots refuse without allocation or mutation" {
    var p: cfg.Problem = .{};
    var c = try cfg.parse(a, @embedFile("fixtures/tiny-dense.json"), &p);
    defer c.deinit();
    var b: support.Backend = .{};
    var s = try State.open(a, &b, &c, 32, 4, std.math.maxInt(u64));
    defer s.close() catch unreachable;
    try s.reset();
    const allocations = b.allocs;
    try std.testing.expectError(error.Qwen4SnapshotBudget, Snap.open(a, &s, @splat(1), 0));
    try std.testing.expectEqual(allocations, b.allocs);
    var snap = try Snap.open(a, &s, @splat(1), std.math.maxInt(u64));
    try std.testing.expectError(error.Qwen4SnapshotPhase, snap.restore(&s, @splat(1)));
    try snap.capture(&s, @splat(1));
    const copies = b.copies;
    try std.testing.expectError(error.Qwen4SnapshotIdentity, snap.restore(&s, @splat(2)));
    try std.testing.expectEqual(copies, b.copies);
    b.fail_copy = true;
    try std.testing.expectError(error.PlantedCopy, snap.capture(&s, @splat(1)));
    b.fail_copy = false;
    try std.testing.expectError(error.Qwen4SnapshotPhase, snap.restore(&s, @splat(1)));
    try snap.capture(&s, @splat(1));
    b.fail_drain = true;
    try std.testing.expectError(error.PlantedDrain, snap.close());
    try std.testing.expectEqual(@as(usize, 2), b.live);
    b.fail_drain = false;
    try snap.close();
    try std.testing.expectEqual(@as(usize, 1), b.live);
}
test "Qwen4 prompt state: a full prompt includes its own last logits, with no target replay or stale result" {
    var p: cfg.Problem = .{};
    var c = try cfg.parse(a, @embedFile("fixtures/tiny-dense.json"), &p);
    defer c.deinit();
    var b: support.Backend = .{};
    var s = try State.open(a, &b, &c, 32, 4, std.math.maxInt(u64));
    defer s.close() catch unreachable;
    try s.reset();
    try s.begin(2);
    try support.complete(&s, &.{ 3, 5 });
    try s.keep(2);
    const Tensor = @import("arena.zig").Arena(support.Backend).Tensor;
    const buffer = try b.allocate(c.vocab * 2);
    defer b.release(buffer);
    const last: Tensor = .{ .buffer = buffer, .offset = 0, .rows = 1, .width = c.vocab, .kind = .bf16 };
    @memset(buffer, 0x3f);
    var snap = try Snap.open(a, &s, @splat(4), try Snap.charge(&s));
    defer snap.close() catch unreachable;
    try snap.capture(&s, @splat(4));
    const before = b.copies;
    try std.testing.expectError(error.Qwen4SnapshotLogits, snap.restoreWithLogits(&s, @splat(4), last));
    try std.testing.expectEqual(before, b.copies);
    try snap.captureWithLogits(&s, @splat(4), last);
    @memset(buffer, 0);
    try s.reset();
    try snap.restoreWithLogits(&s, @splat(4), last);
    try std.testing.expectEqual(@as(usize, 2), s.pos);
    try std.testing.expectEqualSlices(u32, &.{ 3, 5 }, s.history());
    for (buffer) |byte| try std.testing.expectEqual(@as(u8, 0x3f), byte);
}
