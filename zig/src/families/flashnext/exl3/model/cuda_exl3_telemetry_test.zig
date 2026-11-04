const std = @import("std");
const cuda = @import("cuda");
const planner = @import("../dplan/cuda_exl3_dplan.zig");
const pool = @import("../offload/cuda_exl3_pool.zig");
const telemetry = @import("exl3_telemetry.zig");
const Fake = @import("cuda_exl3_fake_driver_test.zig").Fake;

test "exl3: replay device sampling reads executed counters without pulling ownership or synchronizing" {
    var driver = Fake.driver();
    var resident: pool.Pool = undefined;
    resident.lines.pinned_bank = null;
    resident.telemetry_device = .{ .plans = 5, .hit_picks = 20, .copies = 3 };
    resident.telemetry_valid = true;
    var pools = [_]*pool.Pool{&resident};
    var state: [planner.stateWords(1)]u32 = @splat(0);
    const executed: planner.Misc = .{ .now = 104, .rounds = 4, .hit_slots = 30, .miss_slots = 10, .copies = 8, .evictions = 2, .overflow = 0, .inplace = 1, .recent_cursor = 9, .recent_filled = 8 };
    const offset = planner.miscOffset(1) * 4;
    @memcpy(std.mem.sliceAsBytes(&state)[offset..][0..@sizeOf(planner.Misc)], std.mem.asBytes(&executed));
    var seen = [_]planner.Misc{std.mem.zeroes(planner.Misc)};
    var device: planner.DevPlan = undefined;
    device.owner = .device;
    device.pools = &pools;
    var pool_states = [_]planner.PoolState{try planner.PoolState.init(1, 0)};
    device.pool_states = &pool_states;
    device.max_rows = 1;
    device.state = .{ .d = &driver, .ptr = @intFromPtr(&state), .len = @sizeOf(@TypeOf(state)), .borrowed = true };
    device.seen = &seen;
    const syncs = Fake.sync_calls;
    const first = try device.samplePoolAfterConsumers(&resident);
    const again = try device.samplePoolAfterConsumers(&resident);
    try std.testing.expectEqualDeep(first, again);
    try std.testing.expectEqual(@as(u64, 9), first.plans);
    try std.testing.expectEqual(@as(u64, 50), first.hit_picks);
    try std.testing.expectEqual(@as(u64, 11), first.copies);
    try std.testing.expectEqual(@as(u64, 1), first.inplace_experts);
    try std.testing.expectEqual(planner.Owner.device, device.owner);
    try std.testing.expectEqual(@as(u32, 0), seen[0].copies);
    try std.testing.expectEqual(syncs, Fake.sync_calls);
    resident.telemetry_device = first;
    seen[0] = executed;
    device.owner = .host;
    try std.testing.expectEqualDeep(first, try device.samplePoolAfterConsumers(&resident));
    device.owner = .device;
    try std.testing.expectEqualDeep(first, try device.samplePoolAfterConsumers(&resident));
}

test "exl3: ragged planner telemetry and MTP lookup use each pool's state bounds" {
    var driver = Fake.driver();
    const counts = [_]usize{ 10, 37, 12 };
    var residents: [counts.len]pool.Pool = undefined;
    var pools: [counts.len]*pool.Pool = undefined;
    var pool_states: [counts.len]planner.PoolState = undefined;
    var seen: [counts.len]planner.Misc = @splat(std.mem.zeroes(planner.Misc));
    var words: usize = 0;
    for (counts, 0..) |lines, index| {
        residents[index].opt = .{ .layer = if (index == 2) 0 else @intCast(index), .lines = lines };
        residents[index].telemetry_valid = true;
        residents[index].telemetry_device = .{};
        pools[index] = &residents[index];
        pool_states[index] = try planner.PoolState.init(lines, words);
        words = pool_states[index].endWords();
    }
    const state = try std.testing.allocator.alloc(u32, words);
    defer std.testing.allocator.free(state);
    const mirror = try std.testing.allocator.alignedAlloc(u8, .@"16", words * 4);
    defer std.testing.allocator.free(mirror);
    @memset(state, 0);
    for (pool_states, 0..) |entry, index| {
        const misc: *planner.Misc = @ptrCast(entry.slice(state)[planner.miscOffset(entry.lines)..][0..10]);
        misc.* = std.mem.zeroes(planner.Misc);
        misc.rounds = @intCast(index + 1);
        misc.hit_slots = @intCast(10 * (index + 1));
        misc.copies = @intCast(index + 2);
    }
    var device: planner.DevPlan = undefined;
    device.d = &driver;
    device.owner = .device;
    device.pools = &pools;
    device.pool_states = &pool_states;
    device.max_rows = 16;
    device.state = .{ .d = &driver, .ptr = @intFromPtr(state.ptr), .len = state.len * 4, .borrowed = true };
    device.mirror = .{ .d = &driver, .bytes = mirror };
    device.stream = null;
    device.seen = &seen;
    const syncs = Fake.sync_calls;
    for (&residents, 0..) |*resident, index| {
        try std.testing.expectEqual(@as(?usize, index), device.indexOf(resident));
        const sample = try device.samplePoolAfterConsumers(resident);
        try std.testing.expectEqual(@as(u64, index + 1), sample.plans);
        try std.testing.expectEqual(@as(u64, 10 * (index + 1)), sample.hit_picks);
        try std.testing.expectEqual(@as(u64, index + 2), sample.copies);
    }
    var outside: pool.Pool = undefined;
    outside.opt = .{ .layer = 0, .lines = 18 };
    outside.telemetry_valid = true;
    outside.telemetry_device = .{ .plans = 7 };
    try std.testing.expectEqual(@as(?usize, null), device.indexOf(&outside));
    try std.testing.expectEqualDeep(outside.telemetry_device, try device.samplePoolAfterConsumers(&outside));
    const totals = try device.peek();
    try std.testing.expectEqual(@as(u32, 6), totals.rounds);
    try std.testing.expectEqual(@as(u32, 60), totals.hit_slots);
    try std.testing.expectEqual(@as(u32, 9), totals.copies);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(state), mirror);
    try std.testing.expectEqual(syncs, Fake.sync_calls);
}

test "exl3: device counter widening handles one u32 rollover before summation" {
    var previous = std.mem.zeroes(planner.Misc);
    previous.hit_slots = std.math.maxInt(u32) - 2;
    var current = previous;
    current.hit_slots = 4;
    const delta = telemetry.Device.pending(current, previous);
    try std.testing.expectEqual(@as(u64, 7), delta.hit_picks);
}

test "exl3: ring demand counts routed pick appearances separately from per-chunk expert requests and speculative copies" {
    var stats: @import("../prefill/cuda_exl3_prefill_ring.zig").Stats = .{};
    stats.observeDemand(true, 17);
    stats.observeDemand(false, 31);
    stats.observeDemand(false, 0);
    try std.testing.expectEqual(@as(u64, 17), stats.hit_picks);
    try std.testing.expectEqual(@as(u64, 31), stats.miss_picks);
    try std.testing.expectEqual(@as(u64, 1), stats.hit_expert_requests);
    try std.testing.expectEqual(@as(u64, 1), stats.miss_expert_requests);
    try std.testing.expectEqual(@as(u64, 0), stats.copies);
    try std.testing.expectEqual(@as(u64, 0), stats.dma_requested_bytes);
}

test "exl3: aggregate telemetry finds attached pools deduplicates MTP aliases and separates committed device plans" {
    var resident: pool.Pool = undefined;
    resident.lines.pinned_bank = null;
    resident.telemetry_id = 991;
    resident.telemetry_valid = true;
    resident.telemetry_device = .{ .plans = 5, .hit_picks = 20, .miss_picks = 10, .copies = 3, .inplace_experts = 1 };
    resident.stats = .{ .rounds = 7, .hit_slots = 25, .miss_slots = 13, .misses = 5 };
    resident.lines.stats = .{ .copies = 2, .bytes = 96 };
    const payload: [16]u8 = @splat(0);
    var sources = [_][3][]const u8{.{ &payload, &payload, &payload }};
    var widths = [_][3]i32{.{ 1, 1, 1 }};
    resident.src = .{ .routed = &sources };
    resident.k2 = &widths;
    var experts: @import("../pack/cuda_exl3_load.zig").Experts = undefined;
    experts.pool = &resident;
    var engine: @import("../../cuda_prompt.zig").Engine = undefined;
    engine.exl3_prefill = .{};
    engine.exl3_chain = .{};
    engine.telemetry = .{};
    engine.exl3_dplan = .{};
    engine.n_layers = 1;
    engine.layers[0].exl3 = &experts;
    const collector = @import("cuda_exl3_telemetry.zig");
    var snapshot = try collector.sampleAfterConsumers(std.testing.allocator, &engine, &engine.layers[0], &.{}, .{ .rank = 0, .world = 2 }, .all_rank_consumers_joined);
    defer snapshot.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), snapshot.expert_pools);
    try std.testing.expectEqual(@as(u64, 2), snapshot.experts.host_plans);
    try std.testing.expectEqual(@as(u64, 5), snapshot.experts.host_hit_picks);
    try std.testing.expectEqual(@as(u64, 3), snapshot.experts.host_miss_picks);
    try std.testing.expectEqual(@as(u64, 2), snapshot.experts.host_mandatory_copies);
    try std.testing.expectEqual(@as(u64, 5), snapshot.experts.device_plans);
    try std.testing.expectEqual(@as(u64, 1), snapshot.experts.device_inplace_experts);
    try std.testing.expectEqual(@as(u64, 144), snapshot.experts.mapped_copy_payload_min_bytes);
    try std.testing.expectEqual(@as(u64, 144), snapshot.experts.mapped_copy_payload_max_bytes);
    try std.testing.expectEqual(@as(u64, 96), snapshot.experts.pool_dma_requested_bytes);
}
