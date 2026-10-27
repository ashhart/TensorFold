const std = @import("std");
const shared = @import("exl3_shared_kv.zig");

const geometry: shared.Geometry = .{ .lanes = 3, .layers = 2, .blocks = 8, .slots = 3, .row_bytes = 4, .window_rows = 4, .work_words = 8 };
const first: shared.Key = .{ .layer = 0, .block = 0 };
const second: shared.Key = .{ .layer = 1, .block = 0 };
const third: shared.Key = .{ .layer = 0, .block = 1 };
const fourth: shared.Key = .{ .layer = 1, .block = 1 };

fn store(arena: *shared.Arena, sequence: shared.Sequence, keys: []const shared.Key, value: u8) !void {
    const lease = try arena.begin(sequence);
    const assignments = try arena.prepare(lease, keys);
    try arena.submit(lease);
    var payload: [32]u8 = @splat(value);
    for (assignments, 0..) |assignment, index| if (!assignment.hit) try arena.writeFetched(lease, index, &payload);
    try arena.completeFetch(lease);
    try arena.validate();
    try arena.completeConsumers(lease);
}

test "exl3: shared KV cross-lane eviction invalidates the evicted generation and layer" {
    var arena = try shared.Arena.init(std.testing.allocator, geometry);
    defer arena.deinit() catch unreachable;
    const sequence_a = try arena.register(0);
    const sequence_b = try arena.register(1);
    const sequence_c = try arena.register(2);
    try store(&arena, sequence_a, &.{first}, 0x11);
    try store(&arena, sequence_b, &.{first}, 0x22);
    try store(&arena, sequence_a, &.{second}, 0x33);
    const other_slot = (try arena.slotFor(sequence_b, first)).?;
    try store(&arena, sequence_c, &.{first}, 0x44);
    try std.testing.expectEqual(null, try arena.slotFor(sequence_a, first));
    try std.testing.expectEqual(other_slot, (try arena.slotFor(sequence_b, first)).?);
    try std.testing.expect((try arena.slotFor(sequence_a, second)) != null);
    try store(&arena, sequence_a, &.{ second, third }, 0x55);
    try std.testing.expectEqual(null, try arena.slotFor(sequence_b, first));
    try std.testing.expect((try arena.slotFor(sequence_c, first)) != null);
    try arena.validate();
}

test "exl3: shared KV reset release and lane reuse never alias old generations" {
    var arena = try shared.Arena.init(std.testing.allocator, geometry);
    defer arena.deinit() catch unreachable;
    const previous = try arena.register(0);
    const other = try arena.register(1);
    try store(&arena, previous, &.{ first, second }, 0xa1);
    try store(&arena, other, &.{first}, 0xb2);
    const foreign_slot = (try arena.slotFor(other, first)).?;
    const foreign_bytes = try std.testing.allocator.dupe(u8, arena.payload);
    defer std.testing.allocator.free(foreign_bytes);
    const current = try arena.reset(previous);
    try std.testing.expect(current.generation > previous.generation);
    try std.testing.expectError(error.SharedKvSequence, arena.begin(previous));
    try std.testing.expectError(error.SharedKvSequence, arena.release(previous));
    try std.testing.expectEqual(null, try arena.slotFor(current, first));
    try std.testing.expectEqual(foreign_slot, (try arena.slotFor(other, first)).?);
    try std.testing.expectEqualSlices(u8, foreign_bytes, arena.payload);
    try store(&arena, current, &.{first}, 0xc3);
    try arena.release(current);
    const reused = try arena.register(0);
    try std.testing.expect(reused.generation > current.generation);
    try std.testing.expectEqual(null, try arena.slotFor(reused, first));
    try std.testing.expectEqual(foreign_slot, (try arena.slotFor(other, first)).?);
    try arena.validate();
}

test "exl3: shared KV planning pins later hits deduplicates and cancels without mutation" {
    var arena = try shared.Arena.init(std.testing.allocator, geometry);
    defer arena.deinit() catch unreachable;
    const sequence = try arena.register(0);
    try store(&arena, sequence, &.{ first, second, third }, 0x15);
    const hit_slot = (try arena.slotFor(sequence, first)).?;
    const lease = try arena.begin(sequence);
    const plan = try arena.prepare(lease, &.{ fourth, first, fourth });
    try std.testing.expectEqual(@as(usize, 2), plan.len);
    try std.testing.expect(plan[0].slot != hit_slot);
    try std.testing.expect(plan[1].hit);
    try std.testing.expectEqual(hit_slot, plan[1].slot);
    const clock = arena.clock;
    try arena.cancel(lease);
    try std.testing.expectEqual(clock, arena.clock);
    try std.testing.expectEqual(null, try arena.slotFor(sequence, fourth));
    try std.testing.expect((try arena.slotFor(sequence, second)) != null);
    try arena.validate();
    const another = try arena.begin(sequence);
    try std.testing.expectError(error.SharedKvCapacity, arena.prepare(another, &.{ first, second, third, fourth }));
    try std.testing.expectError(error.SharedKvKey, arena.prepare(another, &.{.{ .layer = 2, .block = 0 }}));
    try arena.cancel(another);
    try arena.validate();
}

test "exl3: shared KV lease spans fetch and consumers and cannot be released by stale tickets" {
    var arena = try shared.Arena.init(std.testing.allocator, geometry);
    defer arena.deinit() catch unreachable;
    const sequence = try arena.register(0);
    const other = try arena.register(1);
    const lease = try arena.begin(sequence);
    _ = try arena.prepare(lease, &.{first});
    try arena.submit(lease);
    try std.testing.expectError(error.SharedKvBusy, arena.begin(other));
    try std.testing.expectError(error.SharedKvBusy, arena.reset(other));
    try std.testing.expectError(error.SharedKvBusy, arena.release(sequence));
    try std.testing.expectError(error.SharedKvBusy, arena.deinit());
    try std.testing.expectError(error.SharedKvPending, arena.cancel(lease));
    try std.testing.expectError(error.SharedKvFetchIncomplete, arena.completeFetch(lease));
    try std.testing.expectEqual(null, try arena.slotFor(sequence, first));
    try arena.validate();
    const payload: [32]u8 = @splat(0x65);
    try std.testing.expectError(error.SharedKvPayload, arena.writeFetched(lease, 0, payload[0..31]));
    try arena.writeFetched(lease, 0, &payload);
    try arena.completeFetch(lease);
    try std.testing.expectEqualSlices(u8, &payload, try arena.read(lease, first));
    try std.testing.expectError(error.SharedKvBusy, arena.begin(other));
    try arena.completeConsumers(lease);
    const next = try arena.begin(other);
    try std.testing.expectError(error.SharedKvLease, arena.cancel(lease));
    try std.testing.expectError(error.SharedKvLease, arena.completeConsumers(lease));
    try arena.cancel(next);
}

test "exl3: shared KV submission and consumer failures retain ownership until drained" {
    for ([_]bool{ false, true }) |consumer_failure| {
        var arena = try shared.Arena.init(std.testing.allocator, geometry);
        defer arena.deinit() catch unreachable;
        const sequence = try arena.register(0);
        const other = try arena.register(1);
        try store(&arena, other, &.{first}, 0x12);
        const foreign = (try arena.slotFor(other, first)).?;
        const lease = try arena.begin(sequence);
        _ = try arena.prepare(lease, &.{second});
        try arena.submit(lease);
        const payload: [32]u8 = @splat(0x78);
        try arena.writeFetched(lease, 0, &payload);
        if (consumer_failure) try arena.completeFetch(lease);
        try arena.failSubmitted(lease);
        try std.testing.expectError(error.SharedKvBusy, arena.begin(other));
        try std.testing.expectError(error.SharedKvBusy, arena.reset(sequence));
        try std.testing.expectError(error.SharedKvPending, arena.cancel(lease));
        try std.testing.expectError(error.SharedKvPhase, arena.completeConsumers(lease));
        try arena.recoverAfterDrain(lease);
        try std.testing.expectError(error.SharedKvSequenceFailed, arena.begin(sequence));
        try std.testing.expectEqual(null, try arena.slotFor(sequence, second));
        try std.testing.expectEqual(foreign, (try arena.slotFor(other, first)).?);
        const reset = try arena.reset(sequence);
        try store(&arena, reset, &.{second}, 0x23);
        try arena.validate();
    }
}

test "exl3: shared KV owned-entry snapshots isolate foreign bytes and remap to a new generation" {
    var arena = try shared.Arena.init(std.testing.allocator, geometry);
    defer arena.deinit() catch unreachable;
    const source = try arena.register(0);
    const other = try arena.register(1);
    try store(&arena, source, &.{first}, 0xa1);
    try store(&arena, source, &.{second}, 0xa2);
    const saving = try arena.begin(source);
    var image = try arena.saveOwned(std.testing.allocator, saving);
    defer image.deinit(std.testing.allocator);
    try arena.cancel(saving);
    try std.testing.expectEqual(@as(usize, 2), image.entries.len);
    try arena.release(source);
    try store(&arena, other, &.{ first, third }, 0xb1);
    const destination = try arena.register(2);
    const foreign = (try arena.slotFor(other, third)).?;
    const size = try geometry.blockBytes();
    const foreign_bytes = try std.testing.allocator.dupe(u8, arena.payload[@as(usize, foreign) * size ..][0..size]);
    defer std.testing.allocator.free(foreign_bytes);
    const restoring = try arena.begin(destination);
    try arena.restoreOwned(restoring, &image);
    for (image.entries, 0..) |entry, index| {
        try std.testing.expectEqualSlices(u8, image.payload[index * size ..][0..size], try arena.read(restoring, entry.key));
        const slot = (try arena.slotFor(destination, entry.key)).?;
        try std.testing.expectEqual(destination.generation, arena.owners[slot].generation);
        try std.testing.expectEqual(destination.lane, arena.owners[slot].lane);
    }
    try std.testing.expectEqual(foreign, (try arena.slotFor(other, third)).?);
    try std.testing.expectEqualSlices(u8, foreign_bytes, arena.payload[@as(usize, foreign) * size ..][0..size]);
    try arena.completeConsumers(restoring);
    try arena.validate();
    const first_slot = (try arena.slotFor(destination, first)).?;
    const second_slot = (try arena.slotFor(destination, second)).?;
    try std.testing.expect(arena.stamps[first_slot] < arena.stamps[second_slot]);
    try std.testing.expect(first_slot != image.entries[0].source_slot or second_slot != image.entries[1].source_slot);
    const image_copy = try std.testing.allocator.dupe(u8, image.payload);
    defer std.testing.allocator.free(image_copy);
    try store(&arena, destination, &.{third}, 0xd3);
    const again = try arena.begin(destination);
    try arena.restoreOwned(again, &image);
    try std.testing.expectEqual(null, try arena.slotFor(destination, third));
    try arena.completeConsumers(again);
    try std.testing.expectEqualSlices(u8, image_copy, image.payload);
    try arena.validate();
}

test "exl3: shared KV empty owned snapshot clears only destination residency" {
    var arena = try shared.Arena.init(std.testing.allocator, geometry);
    defer arena.deinit() catch unreachable;
    const sequence = try arena.register(0);
    const other = try arena.register(1);
    const saving = try arena.begin(sequence);
    var image = try arena.saveOwned(std.testing.allocator, saving);
    defer image.deinit(std.testing.allocator);
    try arena.cancel(saving);
    try store(&arena, sequence, &.{first}, 0x13);
    try store(&arena, other, &.{second}, 0x24);
    const restoring = try arena.begin(sequence);
    try arena.restoreOwned(restoring, &image);
    try arena.completeConsumers(restoring);
    try std.testing.expectEqual(null, try arena.slotFor(sequence, first));
    try std.testing.expect((try arena.slotFor(other, second)) != null);
    try arena.validate();
}

test "exl3: shared KV snapshot malformed versions geometry lengths duplicates and overflow are preflight errors" {
    var arena = try shared.Arena.init(std.testing.allocator, geometry);
    defer arena.deinit() catch unreachable;
    const sequence = try arena.register(0);
    try store(&arena, sequence, &.{ first, second }, 0x13);
    const saving = try arena.begin(sequence);
    var image = try arena.saveOwned(std.testing.allocator, saving);
    defer image.deinit(std.testing.allocator);
    try arena.cancel(saving);
    const restoring = try arena.begin(sequence);
    var bad = image;
    bad.version = 1;
    try std.testing.expectError(error.SharedKvSnapshotVersion, arena.restoreOwned(restoring, &bad));
    bad = image;
    bad.geometry.row_bytes *= 2;
    try std.testing.expectError(error.SharedKvSnapshotGeometry, arena.restoreOwned(restoring, &bad));
    bad = image;
    bad.payload = bad.payload[1..];
    try std.testing.expectError(error.SharedKvSnapshotLength, arena.restoreOwned(restoring, &bad));
    const original = image.entries[1];
    image.entries[1] = image.entries[0];
    try std.testing.expectError(error.SharedKvSnapshotDuplicate, arena.restoreOwned(restoring, &image));
    image.entries[1] = original;
    image.entries[1].stamp = image.clock + 1;
    try std.testing.expectError(error.SharedKvSnapshotEntry, arena.restoreOwned(restoring, &image));
    image.entries[1] = original;
    arena.clock = std.math.maxInt(u64);
    try std.testing.expectError(error.Overflow, arena.restoreOwned(restoring, &image));
    try std.testing.expectEqual(shared.Phase.acquired, arena.active.?.phase);
    try arena.cancel(restoring);
    try arena.validate();
}

test "exl3: shared KV byte accounting charges once and rejects overflow and non-BF16 rows" {
    const rank: shared.Geometry = .{ .lanes = 8, .layers = 12, .blocks = 50005, .slots = 16384, .row_bytes = 1024, .window_rows = 6148, .work_words = 4 * 256 * 512 };
    const bytes = try rank.accounting();
    try std.testing.expectEqual(@as(usize, 128 * 1024 * 1024), bytes.payload);
    try std.testing.expectEqual(bytes.shared + bytes.sequence * 8, bytes.rank);
    try std.testing.expectEqual(@as(usize, 940441600), try rank.duplicatedLegacyBytes());
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(shared.Owner));
    var bad = rank;
    bad.row_bytes = std.math.maxInt(usize) - 1;
    try std.testing.expectError(error.Overflow, bad.accounting());
    bad = rank;
    bad.window_rows = std.math.maxInt(usize);
    try std.testing.expectError(error.Overflow, bad.accounting());
    bad = rank;
    bad.work_words = std.math.maxInt(usize);
    try std.testing.expectError(error.Overflow, bad.accounting());
    bad = rank;
    bad.row_bytes = 3;
    try std.testing.expectError(error.SharedKvGeometry, bad.accounting());
    bad = rank;
    bad.slots = std.math.maxInt(u32);
    try std.testing.expectError(error.SharedKvGeometry, bad.accounting());
}

test "exl3: shared KV generation lease serial and clock exhaustion fail without wraparound" {
    var arena = try shared.Arena.init(std.testing.allocator, geometry);
    defer arena.deinit() catch unreachable;
    var sequence = try arena.register(0);
    arena.lanes[0].generation = std.math.maxInt(u64);
    sequence.generation = std.math.maxInt(u64);
    try std.testing.expectError(error.Overflow, arena.reset(sequence));
    try arena.release(sequence);
    try std.testing.expectError(error.Overflow, arena.register(0));
    const other = try arena.register(1);
    arena.serial = std.math.maxInt(u64);
    try std.testing.expectError(error.Overflow, arena.begin(other));
    arena.serial = 0;
    const lease = try arena.begin(other);
    _ = try arena.prepare(lease, &.{first});
    arena.clock = std.math.maxInt(u64);
    try std.testing.expectError(error.Overflow, arena.submit(lease));
    try std.testing.expectEqual(shared.Phase.planned, arena.active.?.phase);
    try arena.cancel(lease);
    try arena.validate();
}

fn allocationCase(allocator: std.mem.Allocator) !void {
    var arena = try shared.Arena.init(allocator, geometry);
    defer arena.deinit() catch unreachable;
}

fn snapshotAllocationCase(allocator: std.mem.Allocator) !void {
    var arena = try shared.Arena.init(std.testing.allocator, geometry);
    defer arena.deinit() catch unreachable;
    const sequence = try arena.register(0);
    try store(&arena, sequence, &.{first}, 0x47);
    const lease = try arena.begin(sequence);
    defer arena.cancel(lease) catch unreachable;
    var image = try arena.saveOwned(allocator, lease);
    defer image.deinit(allocator);
    try image.validate(geometry);
}

test "exl3: shared KV allocation and snapshot allocation failures retain lease and clean ownership" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationCase, .{});
    try std.testing.checkAllAllocationFailures(std.testing.allocator, snapshotAllocationCase, .{});
}
