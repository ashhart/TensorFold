const std = @import("std");

pub const Phase = enum { prefill, decode, verify, unknown };
pub const Rank = struct {
    rank: usize,
    world: usize,

    pub fn validate(owner: Rank) !void {
        if (owner.world == 0 or owner.rank >= owner.world) return error.TelemetryRank;
    }
};

var next_identity: std.atomic.Value(u64) = .init(0);

pub fn identity() !u64 {
    var previous = next_identity.load(.monotonic);
    while (true) {
        const next = try std.math.add(u64, previous, 1);
        previous = next_identity.cmpxchgWeak(previous, next, .monotonic, .monotonic) orelse return next;
    }
}

pub const Device = struct {
    plans: u64 = 0,
    hit_picks: u64 = 0,
    miss_picks: u64 = 0,
    copies: u64 = 0,
    evictions: u64 = 0,
    inplace_experts: u64 = 0,
    overflow_plans: u64 = 0,

    pub fn pending(current: anytype, seen: @TypeOf(current)) Device {
        return .{ .plans = current.rounds -% seen.rounds, .hit_picks = current.hit_slots -% seen.hit_slots, .miss_picks = current.miss_slots -% seen.miss_slots, .copies = current.copies -% seen.copies, .evictions = current.evictions -% seen.evictions, .inplace_experts = current.inplace -% seen.inplace, .overflow_plans = current.overflow -% seen.overflow };
    }

    pub fn pendingBounded(current: anytype, seen: @TypeOf(current), max_picks: u64) !Device {
        const delta = pending(current, seen);
        const bound = try std.math.mul(u64, delta.plans, max_picks);
        if (max_picks == 0 or bound >= @as(u64, 1) << 32) return error.TelemetryExpertIntervalTooLong;
        if (try std.math.add(u64, delta.hit_picks, delta.miss_picks) > bound or try std.math.add(u64, delta.copies, delta.inplace_experts) > bound or delta.overflow_plans > delta.plans) return error.TelemetryInconsistentCounters;
        return delta;
    }

    pub fn record(total: *Device, valid: *bool, current: anytype, seen: @TypeOf(current), max_picks: u64) void {
        if (!valid.*) return;
        const delta = pendingBounded(current, seen, max_picks) catch {
            valid.* = false;
            return;
        };
        add(Device, total, delta) catch {
            valid.* = false;
        };
    }
};

pub fn add(comptime Counters: type, total: *Counters, contribution: Counters) !void {
    var result = total.*;
    inline for (@typeInfo(Counters).@"struct".field_names) |field| @field(result, field) = try std.math.add(u64, @field(result, field), @field(contribution, field));
    total.* = result;
}

pub fn difference(comptime Counters: type, before: Counters, after: Counters) !Counters {
    var result: Counters = undefined;
    inline for (@typeInfo(Counters).@"struct".field_names) |field| {
        if (@field(after, field) < @field(before, field)) return error.TelemetryCounterRegression;
        @field(result, field) = @field(after, field) - @field(before, field);
    }
    return result;
}

pub const Experts = struct {
    host_plans: u64 = 0,
    host_hit_picks: u64 = 0,
    host_miss_picks: u64 = 0,
    host_mandatory_copies: u64 = 0,
    host_prefetch_copies: u64 = 0,
    pool_dma_copies: u64 = 0,
    pool_dma_requested_bytes: u64 = 0,
    device_plans: u64 = 0,
    device_hit_picks: u64 = 0,
    device_miss_picks: u64 = 0,
    device_copies: u64 = 0,
    device_inplace_experts: u64 = 0,
    device_overflow_plans: u64 = 0,
    mapped_copy_payload_min_bytes: u64 = 0,
    mapped_copy_payload_max_bytes: u64 = 0,
    ring_dma_copies: u64 = 0,
    ring_dma_requested_bytes: u64 = 0,
    ring_unused_copies: u64 = 0,
    ring_hit_picks: u64 = 0,
    ring_miss_picks: u64 = 0,
    ring_hit_expert_requests: u64 = 0,
    ring_miss_expert_requests: u64 = 0,
    ring_demand_copies: u64 = 0,
    ring_demand_requested_bytes: u64 = 0,
    ring_demand_unused_copies: u64 = 0,
    ring_demand_unused_requested_bytes: u64 = 0,
    ring_initial_prefetch_copies: u64 = 0,
    ring_initial_prefetch_requested_bytes: u64 = 0,
    ring_initial_prefetch_unused_copies: u64 = 0,
    ring_initial_prefetch_unused_requested_bytes: u64 = 0,
    ring_cross_layer_prefetch_copies: u64 = 0,
    ring_cross_layer_prefetch_requested_bytes: u64 = 0,
    ring_cross_layer_prefetch_unused_copies: u64 = 0,
    ring_cross_layer_prefetch_unused_requested_bytes: u64 = 0,
};

pub const Snapshot = struct {
    rank: Rank,
    scope: u64,
    ring_owner: u64 = 0,
    experts: Experts = .{},
    expert_pools: usize = 0,
    layers: []LayerSample = &.{},

    pub fn deinit(snapshot: *Snapshot, allocator: std.mem.Allocator) void {
        allocator.free(snapshot.layers);
        snapshot.* = undefined;
    }
};

pub const LayerCounts = struct {
    host_plans: u64 = 0,
    host_hit_picks: u64 = 0,
    host_miss_picks: u64 = 0,
    host_mandatory_copies: u64 = 0,
    host_prefetch_copies: u64 = 0,
    pool_dma_copies: u64 = 0,
    pool_dma_requested_bytes: u64 = 0,
    device_plans: u64 = 0,
    device_hit_picks: u64 = 0,
    device_miss_picks: u64 = 0,
    device_copies: u64 = 0,
    device_inplace_experts: u64 = 0,
    device_overflow_plans: u64 = 0,
    mapped_copy_payload_min_bytes: u64 = 0,
    mapped_copy_payload_max_bytes: u64 = 0,

    pub fn fromExperts(experts: Experts) LayerCounts {
        var result: LayerCounts = undefined;
        inline for (@typeInfo(LayerCounts).@"struct".field_names) |field| @field(result, field) = @field(experts, field);
        return result;
    }
};

pub const LayerSample = struct {
    owner: u64,
    kind: enum { decoder, mtp },
    layer: usize,
    lines: usize,
    counts: LayerCounts = .{},

    fn samePool(first: LayerSample, last: LayerSample) bool {
        return first.owner != 0 and first.owner == last.owner and first.kind == last.kind and first.layer == last.layer and first.lines == last.lines;
    }
};

pub fn LayerCursor(comptime capacity: usize) type {
    return struct {
        lock: std.atomic.Mutex = .unlocked,
        scope: u64 = 0,
        rank: Rank = .{ .rank = 0, .world = 1 },
        samples: [capacity]LayerSample = undefined,
        len: usize = 0,

        pub fn claim(cursor: *@This(), allocator: std.mem.Allocator, before: Snapshot, after: Snapshot) ![]LayerSample {
            while (!cursor.lock.tryLock()) std.atomic.spinLoopHint();
            defer cursor.lock.unlock();
            if (before.layers.len != after.layers.len or after.layers.len > capacity) return error.TelemetryOwnerChanged;
            if (!std.meta.eql(before.rank, after.rank) or before.scope != after.scope) return error.TelemetryOwnerChanged;
            if (cursor.len != 0 and (cursor.scope != after.scope or !std.meta.eql(cursor.rank, after.rank) or cursor.len != after.layers.len)) return error.TelemetryOwnerChanged;
            const result = try allocator.alloc(LayerSample, after.layers.len);
            errdefer allocator.free(result);
            for (before.layers, after.layers, result, 0..) |first, last, *delta, index| {
                if (!first.samePool(last)) return error.TelemetryOwnerChanged;
                for (after.layers[0..index]) |previous| if (previous.owner == last.owner) return error.TelemetryDuplicateOwner;
                if (cursor.len != 0 and !cursor.samples[index].samePool(last)) return error.TelemetryOwnerChanged;
                delta.* = last;
                delta.counts = try difference(LayerCounts, first.counts, last.counts);
            }
            for (after.layers, 0..) |last, index| {
                cursor.samples[index] = last;
            }
            cursor.scope = after.scope;
            cursor.rank = after.rank;
            cursor.len = after.layers.len;
            return result;
        }
    };
}

pub const Interval = struct {
    phase: Phase,
    model_rows: u64,
    steps: u64,
};

pub fn fraction(misses: u64, hits: u64) !?f64 {
    const denominator = try std.math.add(u64, misses, hits);
    return if (denominator == 0) null else @as(f64, @floatFromInt(misses)) / @as(f64, @floatFromInt(denominator));
}

pub const Report = struct {
    rank: Rank,
    interval: Interval,
    experts: Experts,
    expert_pools: usize = 0,
    layers: []const LayerSample = &.{},

    pub fn write(report: Report, writer: *std.Io.Writer) !void {
        try report.header(writer, "expert");
        inline for (@typeInfo(Experts).@"struct".field_names) |field| try writer.print(" {s}={d}", .{ field, @field(report.experts, field) });
        try writer.writeByte('\n');
        for (report.layers) |layer| {
            try report.header(writer, "expert-layer");
            try writer.print(" pool={d} kind={s} layer={d} lines={d}", .{ layer.owner, @tagName(layer.kind), layer.layer, layer.lines });
            inline for (@typeInfo(LayerCounts).@"struct".field_names) |field| try writer.print(" {s}={d}", .{ field, @field(layer.counts, field) });
            try writer.writeByte('\n');
        }
    }

    fn header(report: Report, writer: *std.Io.Writer, cache: []const u8) !void {
        try writer.print("EXL3-CACHE v=1 rank={d} world={d} phase={s} rows={d} steps={d} expert_pools={d} cache={s}", .{ report.rank.rank, report.rank.world, @tagName(report.interval.phase), report.interval.model_rows, report.interval.steps, report.expert_pools, cache });
    }

    pub fn formatAlloc(report: Report, allocator: std.mem.Allocator) ![]u8 {
        var writer: std.Io.Writer.Allocating = .init(allocator);
        errdefer writer.deinit();
        try report.write(&writer.writer);
        return writer.toOwnedSlice();
    }

    pub fn between(before: Snapshot, after: Snapshot, interval: Interval) !Report {
        try before.rank.validate();
        try after.rank.validate();
        if (!std.meta.eql(before.rank, after.rank) or before.scope != after.scope or before.expert_pools != after.expert_pools) return error.TelemetryOwnerChanged;
        if (before.ring_owner != 0 and before.ring_owner != after.ring_owner) return error.TelemetryOwnerChanged;
        if (interval.model_rows == 0 or interval.steps == 0) return error.TelemetryDenominator;
        const experts = try difference(Experts, before.experts, after.experts);
        if (experts.device_overflow_plans != 0) return error.TelemetryExpertOverflow;
        return .{ .rank = after.rank, .interval = interval, .experts = experts, .expert_pools = after.expert_pools };
    }
};

test "exl3: telemetry host contracts" {
    _ = @import("exl3_telemetry_test.zig");
}
