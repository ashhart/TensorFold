//! Flash Next EXL3 offload tests: the VRAM line states, LRU and layer tagging.

const std = @import("std");
const hub = @import("exl3_offload.zig");
const vram_ = @import("exl3_offload_vram.zig");

const shape = hub.shape;
const shared_id = hub.shared_id;
const Cache = hub.Cache;
const Decision = hub.Decision;
const Policy = hub.Policy;
const Line = hub.Line;
const Vram = hub.Vram;
const Owned = hub.Owned;
const Copy = hub.Copy;
const init = hub.init;
const find = hub.find;
const planRound = hub.planRound;
const swap = hub.swap;
const installProfile = hub.installProfile;
const onDevice = hub.onDevice;
const residentAt = hub.residentAt;
const touch = hub.touch;
const stage = hub.stage;
const stageKeeping = hub.stageKeeping;
const schedule = hub.schedule;
const reap = hub.reap;
const reapAt = hub.reapAt;
const vramIndexAt = hub.vramIndexAt;
const vramIndex = vram_.vramIndex;
const observe = hub.observe;
const allOwned = hub.allOwned;
const seedHottest = hub.seedHottest;
const fillOwned = hub.fillOwned;
const fillOwnedLayers = hub.fillOwnedLayers;
const scheduleUsed = hub.scheduleUsed;
const scheduleUsedAt = hub.scheduleUsedAt;
const lineOf = hub.lineOf;
const lineIds = hub.lineIds;
const stagedCopies = hub.stagedCopies;

fn benchPolicy(slot_n: u16, expert_bytes: u64, copy_bytes: u64) Policy {
    return .{
        .slots = slot_n,
        .expert_bytes = expert_bytes,
        .copy_bytes = copy_bytes,
    };
}

test "a staged copy is not on device until reap, and LRU drops the older resident" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);

    try std.testing.expectEqual(@as(usize, 0), try stage(&vram, 4));
    try std.testing.expectEqual(@as(usize, 1), try stage(&vram, 7));
    try std.testing.expect(!onDevice(vram, 4));
    try std.testing.expectError(error.NotInflight, reap(&vram, 9, 1));
    try reap(&vram, 4, 1);
    try reap(&vram, 7, 2);
    try std.testing.expect(onDevice(vram, 4));
    try std.testing.expect(touch(&vram, 7, 5));
    try std.testing.expect(onDevice(vram, 4));

    // 4 was reaped at 1 and not touched again. 7 was touched at 5. 4 is the victim.
    try std.testing.expectEqual(@as(usize, 0), try stage(&vram, 8));
    try std.testing.expect(!onDevice(vram, 8));
    try std.testing.expectEqual(Line.inflight, line[0]);
    try std.testing.expectEqual(@as(u16, 7), slots[1]);
    try reap(&vram, 8, 6);
    try std.testing.expect(onDevice(vram, 8));
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 4));
}

test "an in-flight line is not reused, and a full in-flight set is busy" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    _ = try stage(&vram, 1);
    _ = try stage(&vram, 2);
    try std.testing.expectError(error.Busy, stage(&vram, 3));
    try std.testing.expectEqual(@as(u16, 1), slots[0]);
    try std.testing.expectEqual(@as(u16, 2), slots[1]);
    try std.testing.expectError(error.Shared, stage(&vram, shared_id));
}

test "schedule touches residents, stages the hottest miss, and will not evict a hit" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    var decisions: [4]Decision = undefined;
    var scratch: [4]u16 = undefined;
    const policy = benchPolicy(2, 100, 100);

    const first = [_]u16{ 7, shared_id, 4, 7 };
    try schedule(&vram, &first, policy, 3, &decisions, &scratch);
    try std.testing.expectEqual(Decision.hit, decisions[1]);
    try std.testing.expectEqual(Decision.copy, decisions[0]);
    try std.testing.expectEqual(Decision.copy, decisions[3]);
    try std.testing.expectEqual(Decision.cpu, decisions[2]);
    try std.testing.expect(!onDevice(vram, 7));
    try std.testing.expectEqual(Line.inflight, line[0]);
    try std.testing.expectEqual(@as(u16, 7), slots[0]);
    try std.testing.expectEqual(Line.empty, line[1]);

    try reap(&vram, 7, 4);
    const held = [_]u16{7};
    try schedule(&vram, &held, policy, 9, decisions[0..1], &scratch);
    try std.testing.expectEqual(Decision.hit, decisions[0]);
    try std.testing.expectEqual(@as(u32, 9), stamp[0]);

    // Both lines are residents this step asked for, so the miss cannot take either.
    _ = try stage(&vram, 3);
    try reap(&vram, 3, 10);
    const protected = [_]u16{ 7, 3, 8 };
    try schedule(&vram, &protected, benchPolicy(2, 100, 10_000), 11, decisions[0..3], &scratch);
    try std.testing.expectEqual(Decision.hit, decisions[0]);
    try std.testing.expectEqual(Decision.hit, decisions[1]);
    try std.testing.expectEqual(Decision.cpu, decisions[2]);
    try std.testing.expect(onDevice(vram, 7));
    try std.testing.expect(onDevice(vram, 3));
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 8));
}

test "a zero copy budget stages nothing, and an older resident is the one that moves" {
    var slots = [_]u16{ 4, 7 };
    var line = [_]Line{ .resident, .resident };
    var stamp = [_]u32{ 1, 5 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    slots = .{ 4, 7 };
    line = .{ .resident, .resident };
    stamp = .{ 1, 5 };
    var decisions: [1]Decision = undefined;
    var scratch: [2]u16 = undefined;

    const miss = [_]u16{9};
    try schedule(&vram, &miss, benchPolicy(2, 0, 10_000), 6, &decisions, &scratch);
    try std.testing.expectEqual(Decision.cpu, decisions[0]);
    try std.testing.expectEqual(@as(u16, 4), slots[0]);
    try std.testing.expectEqual(Line.resident, line[0]);

    try schedule(&vram, &miss, benchPolicy(2, 100, 100), 6, &decisions, &scratch);
    try std.testing.expectEqual(Decision.copy, decisions[0]);
    try std.testing.expectEqual(@as(u16, 9), slots[0]);
    try std.testing.expectEqual(Line.inflight, line[0]);
    try std.testing.expect(onDevice(vram, 7));
    try std.testing.expect(!onDevice(vram, 9));
}

test "residentAt distinguishes the same expert id on two layers" {
    var slots = [_]u16{ 7, 7 };
    var line = [_]Line{ .resident, .resident };
    var stamp = [_]u32{ 1, 2 };
    var layer = [_]u8{ 0, 40 };
    const vram = Vram.init(&slots, &line, &stamp, &layer);
    slots = .{ 7, 7 };
    line = .{ .resident, .resident };
    stamp = .{ 1, 2 };
    layer = .{ 0, 40 };

    try std.testing.expect(residentAt(vram, 0, 7));
    try std.testing.expect(residentAt(vram, 40, 7));
    try std.testing.expect(!residentAt(vram, 1, 7));
    try std.testing.expect(!residentAt(vram, 0, 8));
    // Id-only lookup still aliases. Generate does not call it on this table.
    try std.testing.expect(onDevice(vram, 7));

    line[1] = .inflight;
    try std.testing.expect(residentAt(vram, 0, 7));
    try std.testing.expect(!residentAt(vram, 40, 7));
}

test "reapAt promotes one inflight layer and leaves the other inflight" {
    var slots = [_]u16{ 7, 7 };
    var line = [_]Line{ .inflight, .inflight };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 40 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    slots = .{ 7, 7 };
    line = .{ .inflight, .inflight };
    stamp = .{ 0, 0 };
    layer = .{ 0, 40 };

    try reapAt(&vram, 0, 7, 3);
    try std.testing.expect(residentAt(vram, 0, 7));
    try std.testing.expectEqual(Line.inflight, line[1]);
    try std.testing.expect(!residentAt(vram, 40, 7));
    try std.testing.expectEqual(@as(u32, 3), stamp[0]);

    // The other layer's copy is still inflight, then promotable on its own reap.
    try reapAt(&vram, 40, 7, 4);
    try std.testing.expect(residentAt(vram, 40, 7));
    try std.testing.expect(residentAt(vram, 0, 7));
    try std.testing.expectEqual(@as(u32, 4), stamp[1]);
    try std.testing.expectEqual(@as(u32, 3), stamp[0]);

    // Layer 0 is already resident.
    line[1] = .inflight;
    stamp[1] = 0;
    try std.testing.expectError(error.NotInflight, reap(&vram, 7, 9));
    try std.testing.expectEqual(Line.resident, line[0]);
    try std.testing.expectEqual(Line.inflight, line[1]);
    try std.testing.expectEqual(@as(u32, 3), stamp[0]);
    try reapAt(&vram, 40, 7, 9);
    try std.testing.expect(residentAt(vram, 40, 7));
    try std.testing.expectEqual(@as(u32, 9), stamp[1]);

    // An inflight copy that exists only on layer 40 is not promoted by `reap`.
    slots = .{ 7, Cache.empty_id };
    line = .{ .inflight, .empty };
    stamp = .{ 0, 0 };
    layer = .{ 40, 0 };
    try std.testing.expectError(error.NotInflight, reap(&vram, 7, 1));
    try std.testing.expectEqual(Line.inflight, line[0]);
    try std.testing.expect(!residentAt(vram, 40, 7));
    try reapAt(&vram, 40, 7, 2);
    try std.testing.expect(residentAt(vram, 40, 7));
    try std.testing.expectEqual(@as(u8, 40), layer[0]);
}

test "stageKeeping tags a newly assigned line as layer 0" {
    var slots = [_]u16{4};
    var line = [_]Line{.resident};
    var stamp = [_]u32{1};
    var layer = [_]u8{40};
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    slots = .{4};
    line = .{.resident};
    stamp = .{1};
    layer = .{40};

    try std.testing.expectEqual(@as(usize, 0), try stageKeeping(&vram, 9, &.{}));
    try std.testing.expectEqual(@as(u16, 9), slots[0]);
    try std.testing.expectEqual(@as(u8, 0), layer[0]);
    try std.testing.expectEqual(Line.inflight, line[0]);
    try std.testing.expect(vramIndexAt(vram, 0, 9) != null);
    try std.testing.expectEqual(@as(?usize, null), vramIndexAt(vram, 40, 9));
    try std.testing.expect(slots[0] != shared_id);
}
