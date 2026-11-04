const std = @import("std");
const prompt = @import("../../cuda_prompt.zig");
const load = @import("../../cuda_prompt_load.zig");
const pool = @import("../offload/cuda_exl3_pool.zig");
const pass = @import("cuda_exl3_pass.zig");
const prefill = @import("../prefill/cuda_exl3_prefill.zig");
const weights = @import("../../cuda_prompt_wt.zig");
pub const counters = @import("exl3_telemetry.zig");
pub const Completion = enum { all_rank_consumers_joined };

pub fn sampleAfterConsumers(allocator: std.mem.Allocator, engine: *prompt.Engine, mtp: ?*const load.Layer, sequences: []const *prompt.Seq, rank: counters.Rank, completion: Completion) !counters.Snapshot {
    _ = completion;
    try rank.validate();
    var hash = std.hash.Wyhash.init(@intFromPtr(engine));
    var totals: counters.Experts = .{};
    var attached: [weights.layers_n + 1]*@import("../pack/cuda_exl3_load.zig").Experts = undefined;
    const attached_count = pass.offloadedPools(engine.layers[0..engine.n_layers], mtp, &attached);
    var layers: []counters.LayerSample = if (engine.telemetry.layers_enabled) try allocator.alloc(counters.LayerSample, attached_count) else @constCast(&.{});
    errdefer allocator.free(layers);
    var seen: [weights.layers_n + 1]*pool.Pool = undefined;
    var seen_count: usize = 0;
    for (attached[0..attached_count]) |experts| {
        const resident = experts.pool orelse continue;
        var duplicate = false;
        for (seen[0..seen_count]) |previous| if (previous == resident) {
            duplicate = true;
            break;
        };
        if (duplicate) continue;
        if (resident.telemetry_id == 0) return error.TelemetryOwnerChanged;
        if (!resident.telemetry_valid) return error.TelemetryCounterRegression;
        seen[seen_count] = resident;
        seen_count += 1;
        hash.update(std.mem.asBytes(&resident.telemetry_id));
        const device = resident.telemetry_device;
        var minimum: u64 = std.math.maxInt(u64);
        var maximum: u64 = 0;
        for (resident.src.routed, resident.k2) |source, widths| {
            if (widths[0] == 0) continue;
            const bytes = try std.math.add(u64, try std.math.add(u64, source[0].len, source[1].len), source[2].len);
            minimum = @min(minimum, bytes);
            maximum = @max(maximum, bytes);
        }
        if (minimum == std.math.maxInt(u64)) {
            if (device.copies != 0) return error.TelemetryInconsistentCounters;
            minimum = 0;
        }
        const host = try counters.difference(counters.Device, resident.telemetry_device, .{ .plans = resident.stats.rounds, .hit_picks = resident.stats.hit_slots, .miss_picks = resident.stats.miss_slots, .copies = resident.stats.misses, .evictions = resident.stats.evictions, .inplace_experts = resident.telemetry_device.inplace_experts, .overflow_plans = resident.telemetry_device.overflow_plans });
        const contribution: counters.Experts = .{
            .host_plans = host.plans,
            .host_hit_picks = host.hit_picks,
            .host_miss_picks = host.miss_picks,
            .host_mandatory_copies = host.copies,
            .host_prefetch_copies = resident.stats.prefetches,
            .pool_dma_copies = resident.lines.stats.copies,
            .pool_dma_requested_bytes = resident.lines.stats.bytes,
            .device_plans = device.plans,
            .device_hit_picks = device.hit_picks,
            .device_miss_picks = device.miss_picks,
            .device_copies = device.copies,
            .device_inplace_experts = device.inplace_experts,
            .device_overflow_plans = device.overflow_plans,
            .mapped_copy_payload_min_bytes = try std.math.mul(u64, device.copies, minimum),
            .mapped_copy_payload_max_bytes = try std.math.mul(u64, device.copies, maximum),
        };
        try counters.add(counters.Experts, &totals, contribution);
        if (engine.telemetry.layers_enabled) {
            var layer_sample: counters.LayerSample = .{ .owner = resident.telemetry_id, .kind = .mtp, .layer = 0, .lines = resident.opt.lines, .counts = counters.LayerCounts.fromExperts(contribution) };
            for (engine.layers[0..engine.n_layers], 0..) |layer, decoder_index| if (layer.exl3) |decoder| {
                if (decoder.pool == resident) {
                    layer_sample.kind = .decoder;
                    layer_sample.layer = decoder_index;
                    break;
                }
            };
            layers[seen_count - 1] = layer_sample;
        }
    }
    for (sequences, 0..) |sequence, index| for (sequences[0..index]) |previous| if (previous == sequence) return error.TelemetryDuplicateOwner;
    const ring = prefill.ringTelemetry(engine);
    if (ring) |observed| {
        totals.ring_dma_copies = observed.copies;
        totals.ring_dma_requested_bytes = observed.dma_requested_bytes;
        totals.ring_unused_copies = observed.unused_copies;
        totals.ring_hit_picks = observed.hit_picks;
        totals.ring_miss_picks = observed.miss_picks;
        totals.ring_hit_expert_requests = observed.hit_expert_requests;
        totals.ring_miss_expert_requests = observed.miss_expert_requests;
        if (observed.attribution) |by_source| @import("../prefill/cuda_exl3_prefill_attribution.zig").publish(by_source, &totals);
    }
    const sampled_layers = if (engine.telemetry.layers_enabled) try allocator.realloc(layers, seen_count) else layers;
    return .{ .rank = rank, .scope = hash.final(), .ring_owner = if (ring) |observed| observed.owner else 0, .experts = totals, .expert_pools = seen_count, .layers = sampled_layers };
}

test "exl3: cache telemetry pure host contracts" {
    _ = counters;
    _ = @import("cuda_exl3_telemetry_test.zig");
    _ = @import("../../cuda_prompt_telemetry.zig");
}
