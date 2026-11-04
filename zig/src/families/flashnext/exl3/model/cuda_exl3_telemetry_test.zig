const std = @import("std");
const cuda = @import("cuda");
const pool = @import("../offload/cuda_exl3_pool.zig");
const telemetry = @import("exl3_telemetry.zig");

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
