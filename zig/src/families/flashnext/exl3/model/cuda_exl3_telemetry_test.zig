const std = @import("std");
const cuda = @import("cuda");
const pool = @import("../offload/cuda_exl3_pool.zig");
const pager = @import("../kv/cuda_exl3_kvpage.zig");
const telemetry = @import("exl3_telemetry.zig");
const Fake = @import("../offload/cuda_exl3_shared_kv_test.zig").Fake;

test "exl3: raw KV counters preserve unsigned high bits and restore advances telemetry revision" {
    var driver = Fake.driver();
    var tables: [12 * (2 + 8) + 2]u32 = @splat(0);
    tables[24 + 3] = 0x80000001;
    tables[24 + 4] = 0xfffffff0;
    tables[24 + 5] = 0x90000002;
    tables[24] = 0x80000010;
    var paged: pager.Paged = undefined;
    paged.d = &driver;
    paged.shared_arena = null;
    paged.cfg = .{ .slots = 1 };
    paged.wr = 4096;
    paged.nblocks = 1;
    paged.tabs = .{ .d = &driver, .ptr = @intFromPtr(&tables), .len = @sizeOf(@TypeOf(tables)), .borrowed = true };
    paged.rebases = @splat(0);
    paged.written = @splat(0);
    const stats = try paged.stats(0);
    try std.testing.expectEqual(@as(u64, 0x80000001), stats.misses);
    try std.testing.expectEqual(@as(u64, 0xfffffff0), stats.needed);
    try std.testing.expectEqual(@as(u64, 0x90000002), stats.calls);
    try std.testing.expectEqual(@as(u32, 0x80000010), try paged.sampleClockAfterConsumers());
    paged.telemetry_revision = 7;
    paged.publishSnapshot(.{ .layout = undefined, .win0 = @splat(0), .rebases = @splat(0), .written = @splat(0) });
    try std.testing.expectEqual(@as(u64, 8), paged.telemetry_revision);
}

test "exl3: shared telemetry samples the arena clock without completion waits" {
    const shared = @import("../offload/cuda_exl3_shared_kv.zig");
    var driver = Fake.driver();
    const arena = try shared.Arena.initStorage(std.testing.allocator, &driver, .{ .cap = 8192, .lanes = 2, .window = 2052, .rows = 1, .slots = 3 });
    defer arena.destroy() catch unreachable;
    const clock: *u32 = @ptrFromInt(arena.tables.ptr + arena.layout.clock);
    clock.* = 524288;
    var paged: pager.Paged = undefined;
    paged.d = &driver;
    paged.shared_arena = arena;
    const syncs = Fake.sync_calls;
    try std.testing.expectEqual(@as(u32, 524288), try paged.sampleClockAfterConsumers());
    clock.* += 1;
    try std.testing.expectEqual(@as(u32, 524289), try paged.sampleClockAfterConsumers());
    try std.testing.expectEqual(syncs, Fake.sync_calls);
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
    engine.shared_arena = null;
    engine.telemetry = .{};
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
