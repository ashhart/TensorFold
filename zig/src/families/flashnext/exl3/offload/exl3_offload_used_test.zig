//! Flash Next EXL3 offload tests: count-ordered scheduling and line addressing.

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

test "a high count keeps an old stamp and a one-off is the victim" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    slots = .{ 1, 4 };
    line = .{ .resident, .resident };
    stamp = .{ 1, 100 };
    var counts: [shape.experts]u32 = @splat(0);
    counts[1] = 50;
    counts[4] = 1;
    var decisions: [1]Decision = undefined;
    var scratch: [2]u16 = undefined;

    const miss = [_]u16{9};
    try scheduleUsed(&vram, &miss, &counts, benchPolicy(2, 100, 100), 6, &decisions, &scratch, allOwned());
    try std.testing.expectEqual(Decision.copy, decisions[0]);
    try std.testing.expectEqual(@as(u16, 1), slots[0]);
    try std.testing.expectEqual(Line.resident, line[0]);
    try std.testing.expectEqual(@as(u16, 9), slots[1]);
    try std.testing.expectEqual(Line.inflight, line[1]);
    try std.testing.expect(onDevice(vram, 1));
    try std.testing.expect(!onDevice(vram, 9));
}

test "zero lifetime counts fall back to stamp LRU" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    slots = .{ 4, 7 };
    line = .{ .resident, .resident };
    stamp = .{ 1, 5 };
    var counts: [shape.experts]u32 = @splat(0);
    var decisions: [1]Decision = undefined;
    var scratch: [2]u16 = undefined;

    const miss = [_]u16{9};
    try scheduleUsed(&vram, &miss, &counts, benchPolicy(2, 100, 100), 6, &decisions, &scratch, allOwned());
    try std.testing.expectEqual(@as(u16, 9), slots[0]);
    try std.testing.expectEqual(Line.inflight, line[0]);
    try std.testing.expect(onDevice(vram, 7));
}

test "lifetime count beats this round's appearances" {
    var slots = [_]u16{Cache.empty_id};
    var line = [_]Line{.empty};
    var stamp = [_]u32{0};
    var layer = [_]u8{0};
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    var counts: [shape.experts]u32 = @splat(0);
    counts[8] = 5;
    const ids = [_]u16{ 3, 3, 3, 3, 8 };
    var decisions: [ids.len]Decision = undefined;
    var scratch: [2]u16 = undefined;
    try scheduleUsed(&vram, &ids, &counts, benchPolicy(1, 100, 100), 1, &decisions, &scratch, allOwned());
    try std.testing.expectEqual(Decision.cpu, decisions[0]);
    try std.testing.expectEqual(Decision.copy, decisions[4]);
    try std.testing.expectEqual(@as(u16, 8), slots[0]);
    try std.testing.expectEqual(Line.inflight, line[0]);
}

test "a resident named this step is kept even when its count is low" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    slots = .{ 1, 4 };
    line = .{ .resident, .resident };
    stamp = .{ 1, 100 };
    var counts: [shape.experts]u32 = @splat(0);
    counts[4] = 1;
    const ids = [_]u16{ 1, 4, 9 };
    var decisions: [3]Decision = undefined;
    var scratch: [2]u16 = undefined;
    try scheduleUsed(&vram, &ids, &counts, benchPolicy(2, 100, 100), 6, &decisions, &scratch, allOwned());
    try std.testing.expectEqual(Decision.hit, decisions[0]);
    try std.testing.expectEqual(Decision.hit, decisions[1]);
    try std.testing.expectEqual(Decision.cpu, decisions[2]);
    try std.testing.expect(onDevice(vram, 1));
    try std.testing.expect(onDevice(vram, 4));
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 9));
}

test "a keep on layer 0 does not protect that id on layer 40" {
    var slots = [_]u16{ 7, 7, 3 };
    var line = [_]Line{ .resident, .resident, .resident };
    var stamp = [_]u32{ 4, 4, 4 };
    var layer = [_]u8{ 0, 40, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    slots = .{ 7, 7, 3 };
    line = .{ .resident, .resident, .resident };
    stamp = .{ 4, 4, 4 };
    layer = .{ 0, 40, 0 };

    var counts: [shape.layers][shape.experts]u32 = @splat(@splat(0));
    counts[0][7] = 10;
    counts[40][7] = 1;
    counts[0][3] = 8;
    const ids = [_]u16{ 7, 9 };
    var decisions: [2]Decision = undefined;
    var scratch: [2]u16 = undefined;
    try scheduleUsedAt(&vram, 0, &ids, &counts, benchPolicy(3, 100, 100), 8, &decisions, &scratch, allOwned());

    try std.testing.expectEqual(Decision.hit, decisions[0]);
    try std.testing.expectEqual(Decision.copy, decisions[1]);
    try std.testing.expect(residentAt(vram, 0, 7));
    try std.testing.expect(residentAt(vram, 0, 3));
    try std.testing.expect(!residentAt(vram, 40, 7));
    try std.testing.expect(vramIndexAt(vram, 0, 9) != null);
    try std.testing.expectEqual(@as(u32, 8), stamp[0]);
    try std.testing.expectEqual(@as(u32, 4), stamp[2]);
    try std.testing.expectEqual(@as(u16, 7), slots[0]);
    try std.testing.expectEqual(@as(u16, 3), slots[2]);
}

test "scheduleUsed is the layer-0 wrapper and is not the multi-layer path" {
    var slots = [_]u16{ 1, 4 };
    var line = [_]Line{ .resident, .resident };
    var stamp = [_]u32{ 1, 50 };
    var layer = [_]u8{ 0, 40 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    slots = .{ 1, 4 };
    line = .{ .resident, .resident };
    stamp = .{ 1, 50 };
    layer = .{ 0, 40 };

    // The flat vector says id 4 is hot.
    var flat: [shape.experts]u32 = @splat(0);
    flat[1] = 1;
    flat[4] = 100;
    var decisions: [1]Decision = undefined;
    var scratch: [2]u16 = undefined;
    const miss = [_]u16{9};
    try scheduleUsed(&vram, &miss, &flat, benchPolicy(2, 100, 100), 6, &decisions, &scratch, allOwned());
    try std.testing.expectEqual(Decision.copy, decisions[0]);
    try std.testing.expectEqual(@as(u16, 1), slots[0]);
    try std.testing.expectEqual(@as(u8, 0), layer[0]);
    try std.testing.expectEqual(@as(u16, 9), slots[1]);
    try std.testing.expectEqual(@as(u8, 0), layer[1]);
    try std.testing.expect(!residentAt(vram, 40, 4));

    slots = .{ 1, 4 };
    line = .{ .resident, .resident };
    stamp = .{ 1, 50 };
    layer = .{ 0, 40 };
    var counts: [shape.layers][shape.experts]u32 = @splat(@splat(0));
    counts[0][1] = 1;
    counts[40][4] = 100;
    try scheduleUsedAt(&vram, 0, &miss, &counts, benchPolicy(2, 100, 100), 7, &decisions, &scratch, allOwned());
    try std.testing.expectEqual(@as(u16, 9), slots[0]);
    try std.testing.expectEqual(@as(u8, 0), layer[0]);
    try std.testing.expectEqual(@as(u16, 4), slots[1]);
    try std.testing.expectEqual(@as(u8, 40), layer[1]);
    try std.testing.expect(residentAt(vram, 40, 4));
    try std.testing.expect(!residentAt(vram, 0, 1));
}

test "scheduleUsedAt does not stage another rank or the shared expert" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    var counts: [shape.layers][shape.experts]u32 = @splat(@splat(0));
    counts[3][400] = 50;
    counts[3][9] = 1;
    const ids = [_]u16{ 400, shared_id, 9 };
    var decisions: [3]Decision = undefined;
    var scratch: [4]u16 = undefined;
    const rank0 = Owned{ .lo = 0, .hi = 256 };
    const policy = benchPolicy(2, 100, 10_000);
    try std.testing.expectError(error.Length, scheduleUsedAt(&vram, @intCast(shape.layers), &ids, &counts, policy, 4, &decisions, &scratch, rank0));
    try std.testing.expectEqual(Line.empty, line[0]);

    try scheduleUsedAt(&vram, 3, &ids, &counts, policy, 4, &decisions, &scratch, rank0);
    try std.testing.expectEqual(Decision.cpu, decisions[0]);
    try std.testing.expectEqual(Decision.hit, decisions[1]);
    try std.testing.expectEqual(Decision.copy, decisions[2]);
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 400));
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, shared_id));
    try std.testing.expect(!residentAt(vram, 3, 9));
    try std.testing.expect(vramIndexAt(vram, 3, 9) != null);
    try std.testing.expectEqual(@as(u16, 9), slots[0]);
    try std.testing.expectEqual(@as(u8, 3), layer[0]);
    try std.testing.expectEqual(Line.empty, line[1]);
    try std.testing.expect(slots[0] != shared_id and slots[1] != shared_id);
}

test "lineIds maps hits to their line and everything else to -1, stagedCopies lists each staged line once" {
    var slots: [3]u16 = undefined;
    var line: [3]Line = undefined;
    var stamp: [3]u32 = undefined;
    var layer: [3]u8 = undefined;
    var v = Vram.init(&slots, &line, &stamp, &layer);
    var counts: [shape.layers][shape.experts]u32 = @splat(@splat(0));
    const pol = Policy{ .slots = 3, .expert_bytes = 10, .copy_bytes = 20 };
    var ids = [_]u16{ 7, 9, 9, 11, 512 };
    var dec: [5]Decision = undefined;
    var scratch: [8]u16 = undefined;
    try observe(&counts[2], &ids);
    try scheduleUsedAt(&v, 2, &ids, &counts, pol, 1, &dec, &scratch, allOwned());
    // 3 unique misses (9 twice is hottest), budget for two copies: 9 and 7 copy, 11 cpu, 512 hit.
    try std.testing.expectEqual(Decision.copy, dec[0]);
    try std.testing.expectEqual(Decision.copy, dec[1]);
    try std.testing.expectEqual(Decision.copy, dec[2]);
    try std.testing.expectEqual(Decision.cpu, dec[3]);
    try std.testing.expectEqual(Decision.hit, dec[4]);
    var copies: [4]Copy = undefined;
    const n = try stagedCopies(v, 2, &ids, &dec, &copies);
    try std.testing.expectEqual(@as(usize, 2), n);
    var ln: [5]i32 = undefined;
    try lineIds(v, 2, &ids, &dec, &ln);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, -1, -1, -1 }, &ln);
    try std.testing.expectError(error.Length, stagedCopies(v, 2, &ids, &dec, copies[0..1]));
    for (copies[0..n]) |c| try reapAt(&v, 2, c.id, 2);
    try scheduleUsedAt(&v, 2, &ids, &counts, pol, 3, &dec, &scratch, allOwned());
    try lineIds(v, 2, &ids, &dec, &ln);
    try std.testing.expectEqual(@as(i32, @intCast(lineOf(v, 2, 7).?)), ln[0]);
    try std.testing.expectEqual(@as(i32, @intCast(lineOf(v, 2, 9).?)), ln[1]);
    try std.testing.expectEqual(ln[1], ln[2]);
    try std.testing.expect(ln[3] == -1 and ln[4] == -1);
    try std.testing.expectEqual(@as(?usize, null), lineOf(v, 3, 7));
}
