//! Reusable batch-composition exactness: a device family supplies a Runner and caller-owned observation buffers.
const std = @import("std");

pub const max_streams = 8;
pub const no_round = std.math.maxInt(u64);
pub const finish_join = std.math.maxInt(u32);
pub const Reason = @import("stream.zig").Reason;
pub const Phase = enum { before, mid, finish };
pub const Stop = enum { length, natural };

/// One deterministic schedule; shrinking retains the reference and the prefix of these peer arrays.
pub const Plan = struct {
    cohort: u8,
    joins: [max_streams]u32,
    peer_max_new: [max_streams]u32,
    reference_max_new: u32,
    restore_at: u32,
    seed: u64,
    phase: Phase,
    stop: Stop,
    drafts: bool,
    keyed: bool,
};

/// Runner output owns no memory: it fills counts and copies into these caller-owned buffers before returning.
pub const Observation = struct {
    tokens: []u32,
    rng_positions: []u64,
    token_count: usize = 0,
    rng_count: usize = 0,
    finished: bool = false,
    reason: Reason = .none,
    accepted: u64 = 0,
    cached: u32 = 0,
    overlap: bool = false,
    join_rounds: [max_streams]u64 = @splat(no_round),
    leave_rounds: [max_streams]u64 = @splat(no_round),

    pub fn reset(o: *Observation) void {
        const tokens = o.tokens;
        const positions = o.rng_positions;
        o.* = .{ .tokens = tokens, .rng_positions = positions };
    }
};

/// Execute one Plan; reject short buffers and copy reference tokens, draw positions and actual events before returning.
pub const Runner = struct {
    ptr: *anyopaque,
    callback: *const fn (*anyopaque, Plan, *Observation) anyerror!void,

    pub fn run(r: Runner, plan: Plan, out: *Observation) !void {
        out.reset();
        return r.callback(r.ptr, plan, out);
    }
};

const Difference = union(enum) {
    token: struct { index: usize, solo: u32, batch: u32 },
    rng: struct { index: usize, solo: u64, batch: u64 },
    reason: struct { solo: Reason, batch: Reason },

    fn toError(d: Difference) anyerror {
        return switch (d) {
            .token => error.TokenMismatch,
            .rng => error.RngPositionMismatch,
            .reason => error.StopReasonMismatch,
        };
    }
};

const absent_u32 = std.math.maxInt(u32);
const absent_u64 = std.math.maxInt(u64);

fn compare(solo: *const Observation, batch: *const Observation) ?Difference {
    const tokens = @min(solo.token_count, batch.token_count);
    for (0..tokens) |i| if (solo.tokens[i] != batch.tokens[i]) {
        return .{ .token = .{ .index = i, .solo = solo.tokens[i], .batch = batch.tokens[i] } };
    };
    if (solo.token_count != batch.token_count) return .{ .token = .{
        .index = tokens,
        .solo = if (tokens < solo.token_count) solo.tokens[tokens] else absent_u32,
        .batch = if (tokens < batch.token_count) batch.tokens[tokens] else absent_u32,
    } };
    const positions = @min(solo.rng_count, batch.rng_count);
    for (0..positions) |i| if (solo.rng_positions[i] != batch.rng_positions[i]) {
        return .{ .rng = .{ .index = i, .solo = solo.rng_positions[i], .batch = batch.rng_positions[i] } };
    };
    if (solo.rng_count != batch.rng_count) return .{ .rng = .{
        .index = positions,
        .solo = if (positions < solo.rng_count) solo.rng_positions[positions] else absent_u64,
        .batch = if (positions < batch.rng_count) batch.rng_positions[positions] else absent_u64,
    } };
    if (solo.reason != batch.reason) return .{ .reason = .{ .solo = solo.reason, .batch = batch.reason } };
    return null;
}

fn replay(plan: Plan) void {
    const n: usize = plan.cohort;
    std.debug.print(
        "lane invariance replay: cohort={d} phase={s} joins={any} peer_max_new={any} reference_max_new={d} " ++
            "restore_at={d} drafts={} keyed={} stop={s} seed={d}\n",
        .{
            plan.cohort,            @tagName(plan.phase), plan.joins[0..n], plan.peer_max_new[0..n],
            plan.reference_max_new, plan.restore_at,      plan.drafts,      plan.keyed,
            @tagName(plan.stop),    plan.seed,
        },
    );
}

fn report(plan: Plan, d: Difference) void {
    switch (d) {
        .token => |m| std.debug.print(
            "lane invariance token mismatch at {d}: solo={d} batch={d} (absent={d})\n",
            .{ m.index, m.solo, m.batch, absent_u32 },
        ),
        .rng => |m| std.debug.print(
            "lane invariance RNG-position mismatch at {d}: solo={d} batch={d} (absent={d})\n",
            .{ m.index, m.solo, m.batch, absent_u64 },
        ),
        .reason => |m| std.debug.print(
            "lane invariance stop-reason mismatch: solo={s} batch={s}\n",
            .{ m.solo.name(), m.batch.name() },
        ),
    }
    std.debug.print("smallest confirmed failing schedule follows; join {d} means after the reference finishes\n", .{finish_join});
    replay(plan);
}

fn evidenceName(e: anyerror) []const u8 {
    return @errorName(e);
}

fn evidenceFailure(plan: Plan, e: anyerror) anyerror {
    std.debug.print("lane invariance missing or invalid evidence: {s}\n", .{evidenceName(e)});
    replay(plan);
    return e;
}

fn requireEvidence(plan: Plan, solo: *const Observation, batch: *const Observation) !void {
    const expected: Reason = if (plan.stop == .natural) .stop else .length;
    if (!solo.finished or !batch.finished) return error.ReplyNotFinished;
    if (solo.reason != expected or batch.reason != expected) return error.UnexpectedStopOutcome;
    if (plan.drafts and (solo.accepted == 0 or batch.accepted == 0)) return error.MissingDraftAcceptance;
    if (solo.rng_count != solo.token_count or batch.rng_count != batch.token_count) return error.MissingRngPositions;
    if (plan.restore_at > 0 and (solo.cached != plan.restore_at or batch.cached != plan.restore_at)) {
        return error.MissingPrefixRestoration;
    }
    for (0..plan.cohort) |i| {
        if (batch.join_rounds[i] == no_round or batch.leave_rounds[i] == no_round) {
            return error.MissingJoinLeaveEvidence;
        }
        if (i > 0 and plan.phase != .finish and batch.join_rounds[i] != plan.joins[i]) {
            return error.JoinRoundMismatch;
        }
        if (i > 0 and plan.phase == .finish and batch.join_rounds[i] < batch.leave_rounds[0]) {
            return error.FinishJoinWasNotObserved;
        }
        if (i > 0 and plan.phase == .before and batch.join_rounds[0] <= batch.join_rounds[i]) {
            return error.BeforeJoinWasNotObserved;
        }
    }
    if (plan.cohort > 1) {
        const peer_leave = batch.leave_rounds[1];
        const reference_leave = batch.leave_rounds[0];
        const valid_leave = switch (plan.phase) {
            .before => peer_leave < reference_leave,
            .mid => peer_leave > 1 and peer_leave < reference_leave,
            .finish => peer_leave >= reference_leave,
        };
        if (!valid_leave) return error.LeavePhaseMismatch;
    }
    if (plan.phase != .finish and plan.cohort > 1 and !batch.overlap) return error.MissingObservedOverlap;
}

/// Runs one plan, shrinking a mismatch over the deterministic peer prefix before printing its exact replay.
pub fn checkPlan(runner: Runner, plan: Plan, solo: *Observation, batch: *Observation) !void {
    if (solo == batch) return error.NeedDistinctObservations;
    if (plan.cohort < 2 or plan.cohort > max_streams) return error.BadCohort;
    var reference_plan = plan;
    reference_plan.cohort = 1;
    reference_plan.joins[0] = 0;
    runner.run(reference_plan, solo) catch |e| {
        std.debug.print("lane invariance solo runner failed: {s}\n", .{@errorName(e)});
        replay(reference_plan);
        return e;
    };
    var smallest = plan;
    runner.run(smallest, batch) catch |e| {
        std.debug.print("lane invariance batch runner failed: {s}\n", .{@errorName(e)});
        replay(smallest);
        return e;
    };
    if (compare(solo, batch)) |initial| {
        var failure = initial;
        if (plan.cohort > 2) {
            var two = plan;
            two.cohort = 2;
            runner.run(two, batch) catch |e| {
                std.debug.print("lane invariance shrink runner failed: {s}\n", .{@errorName(e)});
                replay(two);
                return e;
            };
            if (compare(solo, batch)) |found| {
                smallest = two;
                failure = found;
            } else if (plan.cohort == 8) {
                var four = plan;
                four.cohort = 4;
                runner.run(four, batch) catch |e| {
                    std.debug.print("lane invariance shrink runner failed: {s}\n", .{@errorName(e)});
                    replay(four);
                    return e;
                };
                if (compare(solo, batch)) |found| {
                    smallest = four;
                    failure = found;
                }
            }
        }
        report(smallest, failure);
        return failure.toError();
    }
    requireEvidence(plan, solo, batch) catch |e| return evidenceFailure(plan, e);
}

/// Smallest-first deterministic matrix: every cohort, join phase, draft/sampling/restore mode and stop outcome.
pub fn makePlan(cohort: u8, phase: Phase, drafts: bool, keyed: bool, restored: bool, stop: Stop) Plan {
    var p: Plan = .{
        .cohort = cohort,
        .joins = @splat(0),
        .peer_max_new = .{ 16, 4, 16, 7, 12, 2, 9, 18 },
        .reference_max_new = 16,
        .restore_at = if (restored) 2 else 0,
        .seed = 0x6b69_6d69_3133,
        .phase = phase,
        .stop = stop,
        .drafts = drafts,
        .keyed = keyed,
    };
    const join: u32 = switch (phase) {
        .before => 0,
        .mid => 1,
        .finish => finish_join,
    };
    for (1..max_streams) |i| p.joins[i] = join;
    p.joins[0] = if (phase == .before) 1 else 0;
    p.peer_max_new[1] = switch (phase) {
        .before => 8,
        .mid => 7,
        .finish => 18,
    };
    return p;
}

/// Family entry point; see Runner above and `invariance_fake.zig` for a complete no-allocation caller setup.
pub fn check(runner: Runner, solo: *Observation, batch: *Observation) !void {
    for ([_]u8{ 2, 4, 8 }) |cohort| {
        for ([_]Phase{ .before, .mid, .finish }) |phase| {
            for ([_]bool{ false, true }) |drafts| {
                for ([_]bool{ false, true }) |keyed| {
                    for ([_]bool{ false, true }) |restored| {
                        for ([_]Stop{ .length, .natural }) |stop| {
                            try checkPlan(runner, makePlan(cohort, phase, drafts, keyed, restored, stop), solo, batch);
                        }
                    }
                }
            }
        }
    }
}
