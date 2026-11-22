//! Flash Next EXL3 offload tests: the first fill of the VRAM lines.

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

test "observed counts seed empty lines and a later round keeps the hot id" {
    var counts: [shape.experts]u32 = @splat(0);
    try observe(&counts, &.{ 1, shared_id, 1, 3 });
    try std.testing.expectEqual(@as(u32, 2), counts[1]);
    try std.testing.expectEqual(@as(u32, 1), counts[3]);
    try observe(&counts, &.{1});
    try std.testing.expectEqual(@as(u32, 3), counts[1]);
    var short: [1]u32 = .{9};
    try std.testing.expectError(error.Length, observe(&short, &.{1}));
    try std.testing.expectEqual(@as(u32, 9), short[0]);

    var seeded_counts: [shape.experts]u32 = @splat(0);
    seeded_counts[3] = 10;
    seeded_counts[1] = 10;
    seeded_counts[8] = 2;
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    const n = try seedHottest(&vram, &seeded_counts, 2);
    try std.testing.expectEqual(@as(u16, 2), n);
    try std.testing.expectEqual(@as(u16, 1), slots[0]);
    try std.testing.expectEqual(@as(u16, 3), slots[1]);
    try std.testing.expectEqual(Line.inflight, line[0]);
    try std.testing.expectEqual(Line.inflight, line[1]);
    try std.testing.expect(!onDevice(vram, 1));
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 8));
    try reap(&vram, 1, 1);
    try std.testing.expect(onDevice(vram, 1));

    var wide: [3]u16 = .{ Cache.empty_id, Cache.empty_id, Cache.empty_id };
    var wide_line: [3]Line = .{ .empty, .empty, .empty };
    var wide_stamp: [3]u32 = .{ 0, 0, 0 };
    var wide_layer: [3]u8 = .{ 0, 0, 0 };
    var wide_vram = Vram.init(&wide, &wide_line, &wide_stamp, &wide_layer);
    try std.testing.expectError(error.Length, seedHottest(&wide_vram, counts[0..3], 1));
    try std.testing.expectEqual(Line.empty, wide_line[0]);
}

test "a rank fills its own lines and does not copy the other rank's expert" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    var counts: [shape.experts]u32 = @splat(0);
    counts[400] = 100;
    counts[7] = 3;
    const rank0 = Owned{ .lo = 0, .hi = 256 };
    const n = try fillOwned(&vram, &counts, rank0);
    try std.testing.expectEqual(@as(u16, 2), n);
    try std.testing.expectEqual(@as(u16, 7), slots[0]);
    try std.testing.expectEqual(@as(u16, 0), slots[1]);
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 400));
    try std.testing.expectEqual(Line.inflight, line[0]);
    try reap(&vram, 7, 1);
    try reap(&vram, 0, 1);

    var decisions: [2]Decision = undefined;
    var scratch: [2]u16 = undefined;
    const asked = [_]u16{ 400, 9 };
    try scheduleUsed(&vram, &asked, &counts, benchPolicy(2, 100, 10_000), 4, &decisions, &scratch, rank0);
    try std.testing.expectEqual(Decision.cpu, decisions[0]);
    try std.testing.expectEqual(Decision.copy, decisions[1]);
    try std.testing.expect(onDevice(vram, 7));
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 400));
    try std.testing.expect(vramIndex(vram, 9) != null);
}

test "fillOwnedLayers stripes an explicit line count and skips the other rank" {
    const wide: u16 = @intCast(shape.layers + 2);
    var slots: [shape.layers + 2]u16 = @splat(Cache.empty_id);
    var line: [shape.layers + 2]Line = @splat(.empty);
    var stamp: [shape.layers + 2]u32 = @splat(0);
    var layer: [shape.layers + 2]u8 = @splat(0);
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    var counts: [shape.layers][shape.experts]u32 = @splat(@splat(0));
    const rank0 = Owned{ .lo = 0, .hi = 256 };
    var short: [2][shape.experts]u32 = @splat(@splat(0));
    try std.testing.expectError(error.Length, fillOwnedLayers(&vram, wide, &short, rank0));
    try std.testing.expectError(error.Length, fillOwnedLayers(&vram, wide, &counts, .{ .lo = 4, .hi = 2 }));
    try std.testing.expectEqual(Line.empty, line[0]);

    const n = try fillOwnedLayers(&vram, wide, &counts, rank0);
    try std.testing.expectEqual(wide, n);
    // Id, then layer: slot 1 is layer 1's lowest id, not layer 0's second id.
    try std.testing.expectEqual(@as(u16, 0), slots[0]);
    try std.testing.expectEqual(@as(u8, 0), layer[0]);
    try std.testing.expectEqual(@as(u16, 0), slots[1]);
    try std.testing.expectEqual(@as(u8, 1), layer[1]);
    try std.testing.expectEqual(@as(u16, 0), slots[shape.layers - 1]);
    try std.testing.expectEqual(@as(u8, @intCast(shape.layers - 1)), layer[shape.layers - 1]);
    try std.testing.expectEqual(@as(u16, 1), slots[shape.layers]);
    try std.testing.expectEqual(@as(u8, 0), layer[shape.layers]);
    try std.testing.expectEqual(@as(u16, 1), slots[shape.layers + 1]);
    try std.testing.expectEqual(@as(u8, 1), layer[shape.layers + 1]);
    var on_layer0: usize = 0;
    for (slots, layer) |id, lay| {
        try std.testing.expect(id < 256);
        try std.testing.expect(id != shared_id);
        if (lay == 0) on_layer0 += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), on_layer0);

    // Fewer lines than layers: the remainder is layers 0..3, one lowest id each.
    var few_slots: [4]u16 = @splat(Cache.empty_id);
    var few_line: [4]Line = @splat(.empty);
    var few_stamp: [4]u32 = @splat(0);
    var few_layer: [4]u8 = @splat(0);
    var few = Vram.init(&few_slots, &few_line, &few_stamp, &few_layer);
    try std.testing.expectEqual(@as(u16, 4), try fillOwnedLayers(&few, 4, &counts, rank0));
    try std.testing.expectEqual([_]u16{ 0, 0, 0, 0 }, few_slots);
    try std.testing.expectEqual([_]u8{ 0, 1, 2, 3 }, few_layer);

    // Rank 1. A hotter id outside Owned is not staged. Layer 0 keeps its hottest owned id.
    var rank_slots: [4]u16 = @splat(Cache.empty_id);
    var rank_line: [4]Line = @splat(.empty);
    var rank_stamp: [4]u32 = @splat(0);
    var rank_layer: [4]u8 = @splat(0);
    var rank = Vram.init(&rank_slots, &rank_line, &rank_stamp, &rank_layer);
    counts[0][1] = 999;
    counts[0][300] = 5;
    const rank1 = Owned{ .lo = 256, .hi = 512 };
    try std.testing.expectEqual(@as(u16, 4), try fillOwnedLayers(&rank, 4, &counts, rank1));
    try std.testing.expect(vramIndexAt(rank, 0, 300) != null);
    try std.testing.expect(vramIndexAt(rank, 1, 256) != null);
    try std.testing.expect(vramIndexAt(rank, 2, 256) != null);
    try std.testing.expect(vramIndexAt(rank, 3, 256) != null);
    try std.testing.expectEqual(@as(?usize, null), vramIndex(rank, 1));
    for (rank_slots) |id| {
        try std.testing.expect(id >= 256 and id < 512);
        try std.testing.expect(id != shared_id);
    }
}

test "fillOwnedLayers keeps a hot id inside the stripe and does not give its layer a second line" {
    const lines: u16 = @intCast(shape.layers);
    var slots: [shape.layers]u16 = @splat(Cache.empty_id);
    var line: [shape.layers]Line = @splat(.empty);
    var stamp: [shape.layers]u32 = @splat(0);
    var layer: [shape.layers]u8 = @splat(0);
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    var counts: [shape.layers][shape.experts]u32 = @splat(@splat(0));
    counts[0][9] = 10;
    counts[0][8] = 9;
    const n = try fillOwnedLayers(&vram, lines, &counts, .{ .lo = 0, .hi = 256 });
    try std.testing.expectEqual(lines, n);
    try std.testing.expect(vramIndexAt(vram, 0, 9) != null);
    try std.testing.expectEqual(@as(?usize, null), vramIndexAt(vram, 0, 8));
    try std.testing.expect(vramIndexAt(vram, 1, 0) != null);
    try std.testing.expect(vramIndexAt(vram, 40, 0) != null);
    var on_layer0: usize = 0;
    for (layer) |lay| {
        if (lay == 0) on_layer0 += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), on_layer0);
}
