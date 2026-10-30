const std = @import("std");
const cache = @import("exl3/model/cuda_exl3_telemetry.zig");
const counters = cache.counters;

pub const State = struct {
    enabled: bool = false,
    layers_enabled: bool = false,
    layer_cursor: counters.LayerCursor(@import("cuda_prompt_wt.zig").layers_n + 1) = .{},
    rank: counters.Rank = .{ .rank = 0, .world = 1 },
    forward_phase: counters.Phase = .unknown,

    pub fn fromEnv(rank: usize, world: usize) State {
        const enabled = enabledFromEnv();
        const state: State = .{ .enabled = enabled, .layers_enabled = enabled and layersEnabledFromEnv(), .rank = .{ .rank = rank, .world = world } };
        if (state.enabled) std.log.warn("EXL3-CACHE-SAMPLING v=1 rank={d} world={d} optin_boundary_sync=1 bytes=logical_payload_not_pcie", .{ rank, world });
        return state;
    }

    pub fn forwardPhase(state: State, fed: usize) counters.Phase {
        return if (state.forward_phase != .unknown) state.forward_phase else if (fed == 0) .prefill else .unknown;
    }

    pub fn enter(state: *State, phase: counters.Phase) Scope {
        const previous = state.forward_phase;
        state.forward_phase = phase;
        return .{ .state = state, .previous = previous };
    }
};

pub const Scope = struct {
    state: *State,
    previous: counters.Phase,

    pub fn restore(scope: Scope) void {
        scope.state.forward_phase = scope.previous;
    }
};

pub fn forwardArg(state: State, fed: usize) u64 {
    if (!state.enabled) return 0;
    return switch (state.forwardPhase(fed)) {
        .unknown => 0,
        .prefill => 1,
        .decode => 2,
        .verify => 3,
    };
}

pub fn phaseFromArg(value: u64) !counters.Phase {
    return switch (value) {
        0 => .unknown,
        1 => .prefill,
        2 => .decode,
        3 => .verify,
        else => error.BadRankFrame,
    };
}

pub fn Operations(comptime Engine: type, comptime Seq: type, comptime Hooks: type, comptime underlying: anytype) type {
    return struct {
        pub fn complete(engine: *Engine) !void {
            if (engine.collective) |collective| return collective.finish(engine.stream.*);
            try engine.stream.synchronize();
        }

        pub fn forward(engine: *Engine, comptime Tri: type, tri: Tri, sequence: *Seq, ids: []const u32, hooks: Hooks) !void {
            if (!engine.telemetry.enabled) return underlying.forward(engine, Tri, tri, sequence, ids, hooks);
            const phase = engine.telemetry.forwardPhase(sequence.fed);
            const sequences = [_]*Seq{sequence};
            var before = try begin(engine, &sequences, phase, complete);
            defer if (before) |*sample| sample.deinit(engine.gpa);
            underlying.forward(engine, Tri, tri, sequence, ids, hooks) catch |failure| {
                failed(engine, phase, failure);
                return failure;
            };
            if (before) |sample| try finish(engine, &sequences, sample, phase, ids.len, complete);
        }

        pub fn round(engine: *Engine, comptime Tri: type, tri: Tri, sequences: []const *Seq, ids: []const u32, out: u64) !void {
            if (!engine.telemetry.enabled) return underlying.round(engine, Tri, tri, sequences, ids, out);
            var before = try begin(engine, sequences, .decode, complete);
            defer if (before) |*sample| sample.deinit(engine.gpa);
            underlying.round(engine, Tri, tri, sequences, ids, out) catch |failure| {
                failed(engine, .decode, failure);
                return failure;
            };
            if (before) |sample| try finish(engine, sequences, sample, .decode, sequences.len, complete);
        }
    };
}

pub fn enabledFromEnv() bool {
    const value = std.c.getenv("TF_FLASHNEXT_CACHE_TELEMETRY") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

pub fn layersEnabledFromEnv() bool {
    const value = std.c.getenv("TF_FLASHNEXT_EXL3_LAYER_TELEMETRY") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

pub fn failed(engine: anytype, phase: counters.Phase, failure: anyerror) void {
    std.log.warn("EXL3-CACHE-INVALID v=1 rank={d} world={d} phase={s} reason={s}", .{ engine.telemetry.rank.rank, engine.telemetry.rank.world, @tagName(phase), @errorName(failure) });
}

pub fn begin(engine: anytype, sequences: anytype, phase: counters.Phase, comptime complete: anytype) !?counters.Snapshot {
    return beginWithSampler(engine, sequences, phase, complete, cache.sampleAfterConsumers);
}

pub fn beginWithSampler(engine: anytype, sequences: anytype, phase: counters.Phase, comptime complete: anytype, comptime sample: anytype) !?counters.Snapshot {
    if (!engine.telemetry.enabled) return null;
    complete(engine) catch |failure| {
        failed(engine, phase, failure);
        return failure;
    };
    return sample(engine.gpa, engine, null, sequences, engine.telemetry.rank, cache.Completion.all_rank_consumers_joined) catch |failure| {
        failed(engine, phase, failure);
        return null;
    };
}

pub fn finish(engine: anytype, sequences: anytype, before: counters.Snapshot, phase: counters.Phase, rows: usize, comptime complete: anytype) !void {
    return finishWithSampler(engine, sequences, before, phase, rows, complete, cache.sampleAfterConsumers, publish);
}

pub fn finishWithSampler(engine: anytype, sequences: anytype, before: counters.Snapshot, phase: counters.Phase, rows: usize, comptime complete: anytype, comptime sample: anytype, comptime emit: anytype) !void {
    complete(engine) catch |failure| {
        failed(engine, phase, failure);
        return failure;
    };
    var after = sample(engine.gpa, engine, null, sequences, engine.telemetry.rank, cache.Completion.all_rank_consumers_joined) catch |failure| {
        failed(engine, phase, failure);
        return;
    };
    defer after.deinit(engine.gpa);
    var report = counters.Report.between(before, after, .{ .phase = phase, .model_rows = rows, .steps = 1 }) catch |failure| {
        failed(engine, phase, failure);
        return;
    };
    const layers = if (engine.telemetry.layers_enabled) engine.telemetry.layer_cursor.claim(engine.gpa, before, after) catch |failure| {
        failed(engine, phase, failure);
        return;
    } else &.{};
    defer engine.gpa.free(layers);
    report.layers = layers;
    emit(engine, report) catch |failure| {
        failed(engine, phase, failure);
        return;
    };
}

fn publish(engine: anytype, report: counters.Report) !void {
    const text = try report.formatAlloc(engine.gpa);
    defer engine.gpa.free(text);
    std.debug.print("{s}", .{text});
}

test "exl3: serving telemetry disabled adds no completion or readback and opt in samples only completed boundaries" {
    const Fake = struct {
        gpa: std.mem.Allocator,
        telemetry: State = .{ .rank = .{ .rank = 1, .world = 2 } },
        completions: u64 = 0,
        reads: u64 = 0,
        records: u64 = 0,

        fn complete(engine: *@This()) !void {
            engine.completions += 1;
        }

        fn sample(_: std.mem.Allocator, engine: *@This(), _: @TypeOf(null), _: @TypeOf(null), rank: counters.Rank, _: cache.Completion) !counters.Snapshot {
            try std.testing.expect(engine.completions > engine.reads);
            engine.reads += 1;
            return .{ .rank = rank, .scope = 1, .experts = .{ .device_copies = engine.reads } };
        }

        fn emit(engine: *@This(), report: counters.Report) !void {
            try std.testing.expectEqual(counters.Phase.decode, report.interval.phase);
            try std.testing.expectEqual(@as(u64, 2), report.interval.model_rows);
            try std.testing.expectEqual(@as(u64, 1), report.experts.device_copies);
            try std.testing.expectEqual(@as(usize, 1), report.rank.rank);
            engine.records += 1;
        }
    };
    var engine: Fake = .{ .gpa = std.testing.allocator };
    try std.testing.expect((try beginWithSampler(&engine, null, .decode, Fake.complete, Fake.sample)) == null);
    try std.testing.expectEqual(@as(u64, 0), engine.completions);
    try std.testing.expectEqual(@as(u64, 0), engine.reads);
    engine.telemetry.enabled = true;
    var first = (try beginWithSampler(&engine, null, .decode, Fake.complete, Fake.sample)).?;
    defer first.deinit(engine.gpa);
    try finishWithSampler(&engine, null, first, .decode, 2, Fake.complete, Fake.sample, Fake.emit);
    try std.testing.expectEqual(@as(u64, 2), engine.completions);
    try std.testing.expectEqual(@as(u64, 2), engine.reads);
    try std.testing.expectEqual(@as(u64, 1), engine.records);
}

test "exl3: TP forward metadata has explicit bounded phase mapping independent of token count" {
    const rank_engine = @import("cuda_rank_engine.zig");
    const wire = @import("cuda_rank_wire.zig");
    const logits = @import("cuda_prompt.zig").batch.Logits;
    const phases = [_]counters.Phase{ .unknown, .prefill, .decode, .verify };
    var sender: State = .{ .enabled = true };
    var worker: State = .{ .enabled = true, .forward_phase = .verify };
    for (phases, 0..) |phase, index| {
        sender.forward_phase = phase;
        for ([_]u32{ 4, 8192 }) |bytes| {
            const header: wire.Header = .{ .kind = .forward, .object = 17, .arg = forwardArg(sender, 16384), .bytes = bytes, .flags = @backingInt(logits.last) };
            try std.testing.expectEqual(@as(u64, index), header.arg);
            const frame = try header.encode();
            const received = try wire.Header.decode(&frame);
            try rank_engine.shape(received);
            {
                const scope = worker.enter(try phaseFromArg(received.arg));
                defer scope.restore();
                try std.testing.expectEqual(sender.forwardPhase(16384), worker.forwardPhase(16384));
            }
            try std.testing.expectEqual(counters.Phase.verify, worker.forward_phase);
        }
    }
    sender.enabled = false;
    sender.forward_phase = .prefill;
    try std.testing.expectEqual(@as(u64, 0), forwardArg(sender, 0));
    try rank_engine.shape(.{ .kind = .forward, .object = 17, .bytes = 4 });
    for ([_]u64{ 4, 255, std.math.maxInt(u64) }) |invalid_arg| {
        try std.testing.expectError(error.BadRankFrame, phaseFromArg(invalid_arg));
        try std.testing.expectError(error.BadRankFrame, rank_engine.shape(.{ .kind = .forward, .object = 17, .arg = invalid_arg, .bytes = 4 }));
    }
    try std.testing.expectError(error.BadRankFrame, rank_engine.shape(.{ .kind = .verify, .object = 17, .arg = 3, .bytes = 4 }));
    try std.testing.expectError(error.BadRankFrame, rank_engine.shape(.{ .kind = .round, .arg = 1, .bytes = 12 }));
}

test "exl3: forward and verify phase scopes restore nested caller state on error" {
    const Call = struct {
        fn fail(state: *State, phase: counters.Phase) !void {
            const scope = state.enter(phase);
            defer scope.restore();
            try std.testing.expectEqual(phase, state.forward_phase);
            return error.ScopedOperationFailed;
        }
    };
    for ([_]counters.Phase{ .unknown, .prefill, .decode }) |previous| {
        var state: State = .{ .forward_phase = previous };
        const parent = state.enter(.prefill);
        try std.testing.expectError(error.ScopedOperationFailed, Call.fail(&state, .verify));
        try std.testing.expectEqual(counters.Phase.prefill, state.forward_phase);
        try std.testing.expectError(error.ScopedOperationFailed, Call.fail(&state, try phaseFromArg(1)));
        try std.testing.expectEqual(counters.Phase.prefill, state.forward_phase);
        parent.restore();
        try std.testing.expectEqual(previous, state.forward_phase);
        try std.testing.expect(!state.enabled);
    }
}

test "exl3: extracted disabled forward and round wrappers call underlying operations without joins" {
    const prompt = @import("cuda_prompt.zig");
    const cuda = @import("cuda");
    const Fake = @import("exl3/model/cuda_exl3_fake_driver_test.zig").Fake;
    const Calls = struct {
        var forwards: usize = 0;
        var rounds: usize = 0;

        fn forward(_: *prompt.Engine, comptime Tri: type, _: Tri, _: *prompt.Seq, _: []const u32, _: prompt.batch.Hooks) !void {
            forwards += 1;
        }

        fn round(_: *prompt.Engine, comptime Tri: type, _: Tri, _: []const *prompt.Seq, _: []const u32, _: u64) !void {
            rounds += 1;
        }
    };
    Calls.forwards = 0;
    Calls.rounds = 0;
    var driver = Fake.driver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var engine: prompt.Engine = undefined;
    engine.telemetry = .{};
    engine.gpa = std.testing.allocator;
    engine.collective = null;
    engine.stream = &stream;
    var sequence: prompt.Seq = undefined;
    sequence.fed = 9000;
    const ops = Operations(prompt.Engine, prompt.Seq, prompt.batch.Hooks, .{ .forward = Calls.forward, .round = Calls.round });
    const syncs = Fake.sync_calls;
    try ops.forward(&engine, void, {}, &sequence, &.{123}, .{});
    try ops.round(&engine, void, {}, &.{&sequence}, &.{123}, 0);
    try std.testing.expectEqual(@as(usize, 1), Calls.forwards);
    try std.testing.expectEqual(@as(usize, 1), Calls.rounds);
    try std.testing.expectEqual(syncs, Fake.sync_calls);
}
