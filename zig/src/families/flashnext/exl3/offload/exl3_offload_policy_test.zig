//! Flash Next EXL3 offload tests: the slot cache planner, swap and profile install.

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

test "resident ids are hits and planRound does not change slots" {
    var slots = [_]u16{ 4, 1, 9 };
    var freq = [_]u32{ 3, 8, 1 };
    const cache = init(&slots, &freq);
    slots = .{ 4, 1, 9 };
    freq = .{ 3, 8, 1 };
    const before_slots = slots;
    const before_freq = freq;

    const ids = [_]u16{ 4, shared_id, 1, 9, 4 };
    var decisions: [5]Decision = undefined;
    try planRound(cache, &ids, benchPolicy(3, 100, 10_000), &decisions, &[_]u16{});

    for (decisions) |d| try std.testing.expectEqual(Decision.hit, d);
    try std.testing.expectEqual(before_slots, slots);
    try std.testing.expectEqual(before_freq, freq);
    try std.testing.expectEqual(@as(?usize, 1), find(cache, 1));
    try std.testing.expectEqual(@as(?usize, null), find(cache, shared_id));
}

test "misses past the copy budget are cpu and slots stay put" {
    var slots = [_]u16{42};
    var freq = [_]u32{6};
    const cache = init(&slots, &freq);
    slots[0] = 42;
    freq[0] = 6;
    const before_slots = slots;
    const before_freq = freq;

    // Counts: 5 x4, 1 x3, 8 x1, 3 x1. Budget 250/100 = 2, so 5 and 1 are copied.
    const ids = [_]u16{ 8, 5, 1, 5, 3, 5, 1, 42, 5, 1 };
    var decisions: [ids.len]Decision = undefined;
    var scratch: [8]u16 = undefined;
    try planRound(cache, &ids, benchPolicy(1, 100, 250), &decisions, &scratch);

    const want = [_]Decision{ .cpu, .copy, .copy, .copy, .cpu, .copy, .copy, .hit, .copy, .copy };
    try std.testing.expectEqualSlices(Decision, &want, &decisions);
    try std.testing.expectEqual(before_slots, slots);
    try std.testing.expectEqual(before_freq, freq);
}

test "duplicate miss ids share one decision" {
    var slots: [1]u16 = undefined;
    var freq: [1]u32 = undefined;
    const cache = init(&slots, &freq);
    const ids = [_]u16{ 6, 2, 6 };
    var decisions: [3]Decision = undefined;
    var scratch: [4]u16 = undefined;
    // One copy. 6 appears twice, 2 once, so both 6s are copy and 2 is cpu.
    try planRound(cache, &ids, benchPolicy(1, 100, 100), &decisions, &scratch);
    try std.testing.expectEqual(Decision.copy, decisions[0]);
    try std.testing.expectEqual(Decision.cpu, decisions[1]);
    try std.testing.expectEqual(Decision.copy, decisions[2]);
}

test "equal miss counts break ties toward the lower id" {
    var slots: [0]u16 = .{};
    var freq: [0]u32 = .{};
    const cache = init(&slots, &freq);
    // 10 x5, 6 x4, 2 x4. One copy would be 10; two copies are 10 and 2, not 6.
    const ids = [_]u16{ 6, 10, 2, 6, 10, 2, 10, 6, 2, 10, 6, 10, 2 };
    var decisions: [ids.len]Decision = undefined;
    var scratch: [4]u16 = undefined;
    try planRound(cache, &ids, benchPolicy(0, 50, 100), &decisions, &scratch);
    for (ids, decisions) |id, d| {
        const want: Decision = if (id == 10 or id == 2) .copy else .cpu;
        try std.testing.expectEqual(want, d);
    }
}

test "shared id is a hit and is never stored" {
    var slots: [4]u16 = @splat(Cache.empty_id);
    var freq: [4]u32 = @splat(0);
    var cache = init(&slots, &freq);
    slots[0] = 7;
    freq[0] = 2;

    const ids = [_]u16{ shared_id, 4, shared_id };
    var decisions: [3]Decision = undefined;
    var scratch: [4]u16 = undefined;
    try planRound(cache, &ids, benchPolicy(4, 100, 0), &decisions, &scratch);
    try std.testing.expectEqual(Decision.hit, decisions[0]);
    try std.testing.expectEqual(Decision.cpu, decisions[1]);
    try std.testing.expectEqual(Decision.hit, decisions[2]);
    try std.testing.expectEqual(@as(u16, 7), slots[0]);

    const requested = [_]u16{ shared_id, 4, shared_id, 9 };
    const installed = try swap(&cache, &requested, 4, benchPolicy(4, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 2), installed);
    try std.testing.expectEqual(@as(?usize, null), find(cache, shared_id));
    for (slots) |id| try std.testing.expect(id != shared_id);

    var hist: [shape.experts]u32 = @splat(0);
    hist[4] = 3;
    installProfile(&cache, &hist);
    try std.testing.expectEqual(@as(?usize, null), find(cache, shared_id));
    for (slots) |id| try std.testing.expect(id != shared_id);
}

test "round 2 installs nothing and round 4 installs min of cap and missing" {
    var slots: [8]u16 = @splat(Cache.empty_id);
    var freq: [8]u32 = @splat(0);
    var cache = init(&slots, &freq);
    const requested = [_]u16{ 3, 1, 3, 8 };
    var scratch: [8]u16 = undefined;
    var policy = benchPolicy(8, 1, 0);
    policy.swap_every = 4;

    try std.testing.expectEqual(@as(u16, 0), try swap(&cache, &requested, 0, policy, &scratch));
    try std.testing.expectEqual(@as(u16, 0), try swap(&cache, &requested, 2, policy, &scratch));
    try std.testing.expectEqual(@as(u16, Cache.empty_id), slots[0]);
    try std.testing.expectEqual(@as(u32, 0), freq[0]);

    // Three unique misses, under the cap.
    const n = try swap(&cache, &requested, 4, policy, &scratch);
    try std.testing.expectEqual(@as(u16, 3), n);
    try std.testing.expectEqual(@as(u16, 3), slots[0]);
    try std.testing.expectEqual(@as(u32, 2), freq[0]);
    try std.testing.expectEqual(@as(u16, 1), slots[1]);
    try std.testing.expectEqual(@as(u16, 8), slots[2]);
    try std.testing.expectEqual(@as(u16, Cache.empty_id), slots[3]);
}

test "a cold line is replaced by the missing id" {
    var slots = [_]u16{41};
    var freq = [_]u32{0};
    var cache = init(&slots, &freq);
    slots[0] = 41;
    freq[0] = 0;
    var scratch: [2]u16 = undefined;
    const requested = [_]u16{9};
    const n = try swap(&cache, &requested, 4, benchPolicy(1, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 1), n);
    try std.testing.expectEqual(@as(u16, 9), slots[0]);
    try std.testing.expectEqual(@as(u32, 1), freq[0]);
    try std.testing.expectEqual(@as(?usize, null), find(cache, 41));
}

test "eviction ties take the later slot" {
    var slots = [_]u16{ 30, 10, 20 };
    var freq = [_]u32{ 4, 4, 4 };
    var cache = init(&slots, &freq);
    slots = .{ 30, 10, 20 };
    freq = .{ 4, 4, 4 };
    var scratch: [2]u16 = undefined;
    const n = try swap(&cache, &[_]u16{7}, 4, benchPolicy(3, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 1), n);
    // Higher id would have dropped 30. Higher slot index drops 20.
    try std.testing.expectEqual([_]u16{ 30, 10, 7 }, slots);
    try std.testing.expectEqual(@as(u32, 1), freq[2]);
}

test "a resident named this round is heated before victims are chosen" {
    var slots = [_]u16{ 2, 1 };
    var freq = [_]u32{ 0, 0 };
    var cache = init(&slots, &freq);
    slots = .{ 2, 1 };
    freq = .{ 0, 0 };
    var scratch: [2]u16 = undefined;
    // Without the heat, both freqs stay 0 and the later slot (id 1) would be the victim.
    const n = try swap(&cache, &[_]u16{ 1, 99 }, 4, benchPolicy(2, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 1), n);
    try std.testing.expectEqual([_]u16{ 99, 1 }, slots);
    try std.testing.expectEqual(@as(u32, 1), freq[0]);
    try std.testing.expectEqual(@as(u32, 1), freq[1]);
}

test "resident frequency adds this round's request count" {
    var slots = [_]u16{ 5, Cache.empty_id };
    var freq = [_]u32{ 4, 0 };
    var cache = init(&slots, &freq);
    slots = .{ 5, Cache.empty_id };
    freq = .{ 4, 0 };
    var scratch: [2]u16 = undefined;
    const requested = [_]u16{ 5, 5, 5, 4 };
    const n = try swap(&cache, &requested, 4, benchPolicy(2, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 1), n);
    try std.testing.expectEqual(@as(u16, 5), slots[0]);
    try std.testing.expectEqual(@as(u32, 7), freq[0]);
    try std.testing.expectEqual(@as(u16, 4), slots[1]);
    try std.testing.expectEqual(@as(u32, 1), freq[1]);
}

test "empty lines are filled before a cold resident is evicted" {
    var slots = [_]u16{ 30, Cache.empty_id, 40 };
    var freq = [_]u32{ 5, 0, 1 };
    var cache = init(&slots, &freq);
    slots = .{ 30, Cache.empty_id, 40 };
    freq = .{ 5, 0, 1 };
    var scratch: [4]u16 = undefined;
    const n = try swap(&cache, &[_]u16{ 7, 8 }, 4, benchPolicy(3, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 2), n);
    try std.testing.expectEqual([_]u16{ 30, 7, 8 }, slots);
    try std.testing.expectEqual(@as(u32, 5), freq[0]);
}

test "a candidate placed this call is not evicted by a later one" {
    var slots = [_]u16{ 1, 2, 3, 4 };
    var freq = [_]u32{ 1, 50, 2, 40 };
    var cache = init(&slots, &freq);
    slots = .{ 1, 2, 3, 4 };
    freq = .{ 1, 50, 2, 40 };
    var scratch: [4]u16 = undefined;
    const n = try swap(&cache, &[_]u16{ 10, 11, 12 }, 4, benchPolicy(4, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 3), n);
    try std.testing.expectEqual([_]u16{ 10, 2, 11, 12 }, slots);
    try std.testing.expectEqual(@as(u32, 50), freq[1]);
}

test "installProfile keeps the hottest routed ids" {
    var slots: [3]u16 = @splat(shared_id);
    var freq = [_]u32{ 9, 9, 9 };
    var cache = init(&slots, &freq);
    var hist: [shape.experts]u32 = @splat(0);
    hist[10] = 1;
    hist[4] = 7;
    hist[8] = 7;
    hist[2] = 3;
    installProfile(&cache, &hist);
    try std.testing.expectEqual([_]u16{ 4, 8, 2 }, slots);
    try std.testing.expectEqual([_]u32{ 7, 7, 3 }, freq);
    try std.testing.expectEqual(@as(?usize, null), find(cache, shared_id));

    var wide_slots: [513]u16 = undefined;
    var wide_freq: [513]u32 = undefined;
    var wide = init(&wide_slots, &wide_freq);
    var flat: [shape.experts]u32 = @splat(0);
    flat[0] = 1;
    installProfile(&wide, &flat);
    for (wide_slots[0..shape.experts], 0..) |id, i| {
        try std.testing.expectEqual(@as(u16, @intCast(i)), id);
    }
    try std.testing.expectEqual(Cache.empty_id, wide_slots[512]);
    try std.testing.expectEqual(@as(u32, 1), wide_freq[0]);
    try std.testing.expectEqual(@as(u32, 0), wide_freq[1]);
    try std.testing.expectEqual(@as(?usize, null), find(wide, shared_id));
}

test "swap installs at most the cap even when slots and misses are larger" {
    var slots: [200]u16 = @splat(Cache.empty_id);
    var freq: [200]u32 = @splat(0);
    var cache = init(&slots, &freq);
    var requested: [200]u16 = undefined;
    for (&requested, 0..) |*id, i| id.* = @intCast(i);
    var scratch: [200]u16 = undefined;
    const n = try swap(&cache, &requested, 4, benchPolicy(200, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 96), n);
    for (slots[0..96], 0..) |id, i| try std.testing.expectEqual(@as(u16, @intCast(i)), id);
    try std.testing.expectEqual(Cache.empty_id, slots[96]);
    try std.testing.expectEqual(@as(u32, 1), freq[0]);
    try std.testing.expectEqual(@as(u32, 0), freq[96]);
}

test "an empty slot table misses every routed id and installs nothing" {
    var slots: [0]u16 = .{};
    var freq: [0]u32 = .{};
    var cache = init(&slots, &freq);
    const ids = [_]u16{ shared_id, 4, 4, 8 };
    var decisions: [4]Decision = undefined;
    var scratch: [2]u16 = undefined;
    const policy = benchPolicy(0, 10, 10);
    try planRound(cache, &ids, policy, &decisions, &scratch);
    try std.testing.expectEqual(Decision.hit, decisions[0]);
    try std.testing.expectEqual(Decision.copy, decisions[1]);
    try std.testing.expectEqual(Decision.copy, decisions[2]);
    try std.testing.expectEqual(Decision.cpu, decisions[3]);

    try std.testing.expectEqual(@as(u16, 0), try swap(&cache, &ids, 4, policy, &scratch));
    try std.testing.expectEqual(@as(usize, 0), slots.len);
}

test "zero expert bytes copies nothing" {
    var slots: [0]u16 = .{};
    var freq: [0]u32 = .{};
    const cache = init(&slots, &freq);
    const ids = [_]u16{ 1, 2 };
    var decisions: [2]Decision = undefined;
    var scratch: [2]u16 = undefined;
    try planRound(cache, &ids, benchPolicy(0, 0, 1000), &decisions, &scratch);
    try std.testing.expectEqual(Decision.cpu, decisions[0]);
    try std.testing.expectEqual(Decision.cpu, decisions[1]);
}

test "a short scratch or a short decision buffer is refused without writes" {
    var slots = [_]u16{1};
    var freq = [_]u32{5};
    var cache = init(&slots, &freq);
    slots[0] = 1;
    freq[0] = 5;

    var decisions: [1]Decision = undefined;
    try std.testing.expectError(error.Length, planRound(cache, &[_]u16{ 1, 2 }, benchPolicy(1, 1, 1), &decisions, &[_]u16{}));
    try std.testing.expectEqual(@as(u16, 1), slots[0]);
    try std.testing.expectEqual(@as(u32, 5), freq[0]);

    var one: [2]Decision = undefined;
    try std.testing.expectError(error.Scratch, planRound(cache, &[_]u16{ 2, 3 }, benchPolicy(1, 1, 1), &one, &[_]u16{}));
    try std.testing.expectEqual(@as(u16, 1), slots[0]);

    var tiny: [1]u16 = undefined;
    try std.testing.expectError(error.Scratch, swap(&cache, &[_]u16{ 2, 3 }, 4, benchPolicy(1, 1, 0), &tiny));
    try std.testing.expectEqual(@as(u16, 1), slots[0]);
    try std.testing.expectEqual(@as(u32, 5), freq[0]);

    var off = benchPolicy(1, 1, 0);
    off.swap_every = 0;
    var scratch: [2]u16 = undefined;
    try std.testing.expectEqual(@as(u16, 0), try swap(&cache, &[_]u16{2}, 4, off, &scratch));
    try std.testing.expectEqual(@as(u32, 5), freq[0]);
}
