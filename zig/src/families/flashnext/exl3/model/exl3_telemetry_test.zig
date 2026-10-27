const std = @import("std");
const telemetry = @import("exl3_telemetry.zig");
const expectEqual = std.testing.expectEqual;
const expectError = std.testing.expectError;

fn sample() telemetry.KvSample {
    return .{ .owner = 12, .revision = 3, .layer = 0, .empty = false, .epoch = 100, .clock = 100, .requests = 40, .misses = 20, .calls = 10, .overflow = false, .max_blocks = 1024, .block_bytes = 8192, .rebases = 1, .written = 4096 };
}

test "exl3: KV telemetry unsigned wrap retains block denominator and BF16 payload units" {
    var before = sample();
    before.requests = std.math.maxInt(u32) - 3;
    before.misses = std.math.maxInt(u32) - 1;
    var after = before;
    after.epoch += 2;
    after.clock += 2;
    after.requests = 6;
    after.misses = 2;
    after.calls += 2;
    const result = try before.delta(after, false);
    try expectEqual(@as(u64, 10), result.block_requests);
    try expectEqual(@as(u64, 4), result.block_misses);
    try expectEqual(@as(u64, 32768), result.logical_fetch_bytes);
    try expectEqual(@as(u64, 2), result.sparse_calls);
}

test "exl3: KV telemetry rejects restore reset recycle and ambiguous wrap intervals" {
    const before = sample();
    var after = before;
    after.revision += 1;
    after.requests += 1;
    try expectError(error.TelemetryKvReset, before.delta(after, false));
    try expectError(error.TelemetryKvReset, before.delta(after, true));
    after = before;
    after.owner += 1;
    try expectError(error.TelemetryOwnerChanged, before.delta(after, false));
    after = before;
    after.clock -= 1;
    try expectError(error.TelemetryCounterRegression, before.delta(after, false));
    after = before;
    after.clock += 1 << 22;
    try expectError(error.TelemetryKvIntervalTooLong, before.delta(after, false));
    after = before;
    after.overflow = true;
    try expectError(error.TelemetryKvOverflow, before.delta(after, false));
    after = before;
    after.calls += 1;
    try expectError(error.TelemetryInconsistentCounters, before.delta(after, false));
}

test "exl3: KV initial empty reset needs explicit opt in and exactly one revision" {
    var before = sample();
    before.empty = true;
    before.epoch = 0;
    before.clock = 0;
    before.requests = 0;
    before.misses = 0;
    before.calls = 0;
    var after = before;
    after.revision += 1;
    after.epoch = 10;
    after.clock = 10;
    after.requests = 90;
    after.misses = 60;
    after.calls = 4;
    try expectError(error.TelemetryKvReset, before.delta(after, false));
    try expectEqual(@as(u64, 60), (try before.delta(after, true)).block_misses);
    after.revision += 1;
    try expectError(error.TelemetryKvReset, before.delta(after, true));
}

test "exl3: fresh shared lane at high arena clock uses completed boundary delta not last layer epoch" {
    var before = sample();
    before.empty = true;
    before.epoch = 0;
    before.clock = 524288;
    before.max_blocks = 8192;
    before.requests = 0;
    before.misses = 0;
    before.calls = 0;
    var after = before;
    after.clock += 1;
    after.epoch = after.clock;
    after.requests = 512;
    after.misses = 512;
    after.calls = 1;
    try std.testing.expect(@as(u64, after.epoch - before.epoch) * after.max_blocks >= @as(u64, 1) << 32);
    const result = try before.delta(after, false);
    try expectEqual(@as(u64, 512), result.block_requests);
    try expectEqual(@as(u64, 512 * 8192), result.logical_fetch_bytes);
    after.revision += 1;
    try expectError(error.TelemetryKvReset, before.delta(after, false));
    try expectEqual(@as(u64, 512), (try before.delta(after, true)).block_misses);
}

test "exl3: restored historical layer epoch cannot replace boundary clock or weaken wrap checks" {
    var before = sample();
    before.epoch = 900000;
    before.clock = 20;
    before.requests = std.math.maxInt(u32) - 3;
    before.misses = std.math.maxInt(u32) - 1;
    var after = before;
    after.clock = 21;
    after.epoch = 21;
    after.requests = 6;
    after.misses = 2;
    after.calls += 1;
    const result = try before.delta(after, false);
    try expectEqual(@as(u64, 10), result.block_requests);
    try expectEqual(@as(u64, 4), result.block_misses);
    after.clock = before.clock - 1;
    try expectError(error.TelemetryCounterRegression, before.delta(after, false));
    after.clock = before.clock + (1 << 22);
    try expectError(error.TelemetryKvIntervalTooLong, before.delta(after, false));
    before.clock = std.math.maxInt(u32);
    after.clock = 0;
    try expectError(error.TelemetryCounterRegression, before.delta(after, false));
}

test "exl3: checked telemetry aggregation exceeds u32 without overflow or partial mutation" {
    var totals: telemetry.Experts = .{ .device_hit_picks = std.math.maxInt(u32) };
    try telemetry.add(telemetry.Experts, &totals, .{ .device_hit_picks = 10 });
    try expectEqual(@as(u64, std.math.maxInt(u32)) + 10, totals.device_hit_picks);
    totals.device_hit_picks = std.math.maxInt(u64);
    const before = totals;
    try expectError(error.Overflow, telemetry.add(telemetry.Experts, &totals, .{ .host_plans = 5, .device_hit_picks = 1 }));
    try std.testing.expectEqualDeep(before, totals);
    try expectError(error.TelemetryCounterRegression, telemetry.difference(telemetry.Experts, totals, .{}));
    try expectEqual(@as(?f64, null), try telemetry.fraction(0, 0));
    try expectEqual(@as(?f64, 0.25), try telemetry.fraction(1, 3));
    try expectError(error.Overflow, telemetry.fraction(std.math.maxInt(u64), 1));
}

test "exl3: invalid device telemetry cannot fail normal ownership transfer or publish partial totals" {
    const Misc = struct { now: u32 = 0, rounds: u32 = 0, hit_slots: u32 = 0, miss_slots: u32 = 0, copies: u32 = 0, evictions: u32 = 0, overflow: u32 = 0, inplace: u32 = 0 };
    var total: telemetry.Device = .{ .copies = 10 };
    var valid = true;
    telemetry.Device.record(&total, &valid, Misc{ .rounds = 1, .hit_slots = 11 }, Misc{}, 10);
    try std.testing.expect(!valid);
    try expectEqual(@as(u64, 10), total.copies);
    telemetry.Device.record(&total, &valid, Misc{ .rounds = 1, .copies = 1 }, Misc{}, 10);
    try expectEqual(@as(u64, 10), total.copies);
    try expectError(error.TelemetryExpertIntervalTooLong, telemetry.Device.pendingBounded(Misc{ .rounds = 1 << 30 }, Misc{}, 10));
}

test "exl3: phase report stable schema rank and model-row denominators" {
    var values = [_]telemetry.KvSample{sample()};
    var before: telemetry.Snapshot = .{ .rank = .{ .rank = 1, .world = 2 }, .scope = 7, .kv = &values };
    var after = before;
    after.experts.device_hit_picks = 40;
    after.experts.device_miss_picks = 10;
    const report = try telemetry.Report.between(before, after, .{ .phase = .decode, .model_rows = 16, .steps = 8 });
    const text = try report.formatAlloc(std.testing.allocator);
    defer std.testing.allocator.free(text);
    try std.testing.expect(std.mem.startsWith(u8, text, "EXL3-CACHE v=1 rank=1 world=2 phase=decode rows=16 steps=8 expert_pools=0 kv_layers=1 cache=expert "));
    try std.testing.expect(std.mem.indexOf(u8, text, "device_hit_picks=40 device_miss_picks=10") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "cache=bf16-kv block_requests=0 block_misses=0 sparse_calls=0 logical_fetch_bytes=0") != null);
    try expectError(error.TelemetryDenominator, telemetry.Report.between(before, after, .{ .phase = .prefill, .model_rows = 0, .steps = 1 }));
    after.rank.rank = 0;
    try expectError(error.TelemetryOwnerChanged, telemetry.Report.between(before, after, .{ .phase = .decode, .model_rows = 1, .steps = 1 }));
    after = before;
    after.scope += 1;
    try expectError(error.TelemetryOwnerChanged, telemetry.Report.between(before, after, .{ .phase = .decode, .model_rows = 1, .steps = 1 }));
    after = before;
    after.ring_owner = 13;
    _ = try telemetry.Report.between(before, after, .{ .phase = .prefill, .model_rows = 1, .steps = 1 });
    before.ring_owner = 12;
    try expectError(error.TelemetryOwnerChanged, telemetry.Report.between(before, after, .{ .phase = .prefill, .model_rows = 1, .steps = 1 }));
}

test "exl3: per-layer deltas sum to aggregate counters and overlapping reports claim counters once" {
    const rank: telemetry.Rank = .{ .rank = 0, .world = 2 };
    var before_layers = [_]telemetry.LayerSample{.{ .owner = 11, .kind = .decoder, .layer = 4, .lines = 20 }};
    var after_layers = before_layers;
    after_layers[0].counts.host_miss_picks = 5;
    after_layers[0].counts.host_mandatory_copies = 2;
    after_layers[0].counts.device_miss_picks = 3;
    after_layers[0].counts.device_copies = 1;
    var before: telemetry.Snapshot = .{ .rank = rank, .scope = 42, .kv = &.{}, .layers = &before_layers, .expert_pools = 1 };
    var after = before;
    after.layers = &after_layers;
    after.experts.host_miss_picks = 5;
    after.experts.host_mandatory_copies = 2;
    after.experts.device_miss_picks = 3;
    after.experts.device_copies = 1;
    var cursor: telemetry.LayerCursor(2) = .{};
    const first = try cursor.claim(std.testing.allocator, before, after);
    defer std.testing.allocator.free(first);
    const report = try telemetry.Report.between(before, after, .{ .phase = .decode, .model_rows = 1, .steps = 1 });
    try std.testing.expectEqual(report.experts.host_miss_picks, first[0].counts.host_miss_picks);
    try std.testing.expectEqual(report.experts.host_mandatory_copies, first[0].counts.host_mandatory_copies);
    try std.testing.expectEqual(report.experts.device_miss_picks, first[0].counts.device_miss_picks);
    try std.testing.expectEqual(report.experts.device_copies, first[0].counts.device_copies);

    var concurrent_before_layers = after_layers;
    var concurrent_after_layers = after_layers;
    concurrent_before_layers[0].counts.host_miss_picks = 3;
    concurrent_before_layers[0].counts.host_mandatory_copies = 1;
    concurrent_before_layers[0].counts.device_miss_picks = 2;
    concurrent_before_layers[0].counts.device_copies = 0;
    concurrent_after_layers[0].counts.host_miss_picks = 8;
    concurrent_after_layers[0].counts.host_mandatory_copies = 4;
    concurrent_after_layers[0].counts.device_miss_picks = 6;
    concurrent_after_layers[0].counts.device_copies = 2;
    before.layers = &concurrent_before_layers;
    after.layers = &concurrent_after_layers;
    before.experts = .{ .host_miss_picks = 5, .host_mandatory_copies = 2, .device_miss_picks = 3, .device_copies = 1 };
    after.experts = .{ .host_miss_picks = 10, .host_mandatory_copies = 5, .device_miss_picks = 7, .device_copies = 3 };
    const second = try cursor.claim(std.testing.allocator, before, after);
    defer std.testing.allocator.free(second);
    const second_report = try telemetry.Report.between(before, after, .{ .phase = .decode, .model_rows = 1, .steps = 1 });
    try std.testing.expectEqual(second_report.experts.host_miss_picks, second[0].counts.host_miss_picks);
    try std.testing.expectEqual(second_report.experts.host_mandatory_copies, second[0].counts.host_mandatory_copies);
    try std.testing.expectEqual(second_report.experts.device_miss_picks, second[0].counts.device_miss_picks);
    try std.testing.expectEqual(second_report.experts.device_copies, second[0].counts.device_copies);
    try std.testing.expectEqual(@as(u64, 10), first[0].counts.host_miss_picks + second[0].counts.host_miss_picks);
}
