//! Family use: call check with two fixed Observations and Runner{.ptr = &family, .callback = Family.run}.
const std = @import("std");
const lanes = @import("lanes.zig");
const inv = lanes.invariance;
const fixtures = @import("drafted_test.zig");
const be = lanes.backend;
const Stream = lanes.Stream;
const gpa = std.testing.allocator;

const Recorder = struct {
    target: lanes.fake.Fake,
    trace: std.ArrayList(u64) = .empty,
    reference: ?*Stream = null,
    overlap: bool = false,
    vt: be.Backend.VTable = undefined,

    fn init() Recorder {
        return .{ .target = .{ .gpa = gpa } };
    }

    fn deinit(r: *Recorder) void {
        r.trace.deinit(gpa);
        r.target.deinit();
    }

    fn self(ptr: *anyopaque) *Recorder {
        const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return @fieldParentPtr("target", target);
    }

    fn backend(r: *Recorder) be.Backend {
        r.vt = r.target.backend().vtable.*;
        r.vt.first = first;
        r.vt.queue = queue;
        r.vt.verify = verify;
        return .{ .ptr = &r.target, .vtable = &r.vt, .chunk_invariant = true };
    }

    fn note(r: *Recorder, s: *Stream, position: u64) !void {
        if (r.reference == s) try r.trace.append(gpa, position);
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const r = self(ptr);
        try r.note(s, position);
        return r.target.backend().first(s, position);
    }

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const r = self(ptr);
        try r.note(s, position);
        return r.target.backend().queue(s, feed, position);
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const r = self(ptr);
        var has_reference = false;
        for (windows) |w| {
            if (w.stream == r.reference) {
                has_reference = true;
                for (w.positions) |position| try r.trace.append(gpa, position);
            }
        }
        if (has_reference and windows.len > 1) r.overlap = true;
        return r.target.backend().verify(windows, out);
    }
};

const ids = [inv.max_streams][]const u8{ "reference", "peer1", "peer2", "peer3", "peer4", "peer5", "peer6", "peer7" };

fn samplingFor(plan: inv.Plan, index: usize) ?lanes.Sampling {
    if (!plan.keyed) return null;
    return .{ .seed = plan.seed +% index, .temperature = 1.0, .top_p = 0.95 };
}

fn naturalEos(plan: inv.Plan) u32 {
    var history: [40]u32 = undefined;
    const prompt = fixtures.p1;
    @memcpy(history[0..prompt.len], &prompt);
    var length = prompt.len;
    var emitted: [24]u32 = undefined;
    for (&emitted) |*token| {
        token.* = lanes.fake.next(history[0..length], samplingFor(plan, 0), length);
        history[length] = token.*;
        length += 1;
    }
    for (14..emitted.len) |i| {
        var unique = true;
        for (emitted[0..i]) |earlier| unique = unique and earlier != emitted[i];
        if (unique) return emitted[i];
    }
    return emitted[emitted.len - 1];
}

fn markLeaves(streams: []Stream, leaves: []u64, round: u64) void {
    for (streams, leaves) |*s, *leave| {
        if (s.finished and leave.* == inv.no_round) leave.* = round;
    }
}

fn addDue(
    round: u64,
    e: *lanes.Engine,
    plan: inv.Plan,
    streams: []Stream,
    joins: []u64,
) !void {
    for (streams, joins, 0..) |*s, *joined, i| {
        if (joined.* != inv.no_round) continue;
        const wanted = plan.joins[i];
        const due = if (wanted == inv.finish_join) streams[0].finished else round >= wanted;
        if (due) {
            joined.* = round;
            try e.addStream(s);
        }
    }
}

const Fault = enum { none, token, rng, reason, no_draft, no_restore, no_overlap, no_join, bad_before_join, early_leave, token_nonmonotonic };

const FakeRunner = struct {
    fault: Fault = .none,
    last_cohort: u8 = 0,
    seen_cohorts: u16 = 0,

    fn gate(r: *FakeRunner) inv.Runner {
        return .{ .ptr = r, .callback = run };
    }

    fn applyFault(r: *FakeRunner, plan: inv.Plan, o: *inv.Observation) void {
        if (plan.cohort == 1) return;
        switch (r.fault) {
            .none => {},
            .token => if (o.token_count > 0) {
                o.tokens[0] +%= 1;
            },
            .rng => if (o.rng_count > 0) {
                o.rng_positions[o.rng_count - 1] +%= 1;
            },
            .reason => o.reason = if (o.reason == .stop) .length else .stop,
            .no_draft => o.accepted = 0,
            .no_restore => o.cached = 0,
            .no_overlap => o.overlap = false,
            .no_join => o.join_rounds[1] = inv.no_round,
            .bad_before_join => o.join_rounds[0] = 0,
            .early_leave => o.leave_rounds[1] = 0,
            .token_nonmonotonic => if (plan.cohort != 4 and o.token_count > 0) {
                o.tokens[0] +%= 1;
            },
        }
    }

    fn run(ptr: *anyopaque, plan: inv.Plan, out: *inv.Observation) anyerror!void {
        const r: *FakeRunner = @ptrCast(@alignCast(ptr));
        r.last_cohort = plan.cohort;
        r.seen_cohorts |= @as(u16, 1) << @intCast(plan.cohort);
        var recorder = Recorder.init();
        defer recorder.deinit();
        var head: fixtures.FakeDrafter = .{ .facts_ = .{ .depth = 6, .step_ms = 0.5, .batched = true } };
        defer head.deinit();
        var wrapped = try lanes.drafted.Drafted.init(gpa, recorder.backend(), head.drafter());
        defer wrapped.deinit();
        var model = fixtures.base();
        model.gpu_tokens = false;
        model.lone_copies = false;
        if (plan.drafts) model = wrapped.facts(model);
        var cfg = try lanes.Config.init(gpa, model, 16, 15);
        defer cfg.deinit(gpa);
        cfg.timed = false;
        var clock: lanes.fake.FixedClock = .{};
        // Native fake drafts use restored target history, not the external fixture's unseen-prefix placeholders.
        const backend = recorder.backend();
        var streams: [inv.max_streams]Stream = undefined;
        var proposers: [inv.max_streams]lanes.SuffixLookup = undefined;
        var saved: [inv.max_streams]?*anyopaque = @splat(null);
        var made: usize = 0;
        var made_proposers: usize = 0;
        defer for (saved) |state| if (state) |s| recorder.target.drop(s);
        defer {
            for (streams[0..made]) |*s| s.deinit(gpa);
            for (proposers[0..made_proposers]) |*p| p.deinit();
        }
        var engine = lanes.Engine.init(gpa, &cfg, backend, clock.clock());
        defer engine.deinit();
        const n: usize = plan.cohort;
        const eos = [_]u32{naturalEos(plan)};
        for (0..n) |i| {
            var prompt = fixtures.p1;
            if (i > 0) prompt[0] += @intCast(i * 3);
            if (plan.restore_at > 0) {
                var early = try Stream.init(gpa, .{
                    .id = ids[i],
                    .prompt = prompt[0..plan.restore_at],
                    .max_new = 1,
                });
                defer early.deinit(gpa);
                try recorder.target.backend().prefill(&early);
                defer recorder.target.backend().release(&early);
                saved[i] = try recorder.target.save(&early, plan.restore_at);
            }
            proposers[i] = try lanes.SuffixLookup.init(gpa, .{ .min_match = 4 });
            made_proposers += 1;
            const limit = if (i == 0) plan.reference_max_new else plan.peer_max_new[i];
            streams[i] = try Stream.init(gpa, .{
                .id = ids[i],
                .prompt = &prompt,
                .max_new = limit,
                .eos = if (i == 0 and plan.stop == .natural) &eos else &.{},
                .sampling = samplingFor(plan, i),
                .drafts = plan.drafts,
                .proposer = proposers[i].proposer(),
                .reuse = .{ .saved = saved[i], .at = plan.restore_at },
            });
            made += 1;
        }
        recorder.reference = &streams[0];
        var joins: [inv.max_streams]u64 = @splat(inv.no_round);
        defer {
            for (streams[0..made], joins[0..made]) |*s, joined| {
                if (joined != inv.no_round and !s.finished) engine.discard(s);
            }
        }
        var leaves: [inv.max_streams]u64 = @splat(inv.no_round);
        var round: u64 = 0;
        try addDue(round, &engine, plan, streams[0..n], joins[0..n]);
        markLeaves(streams[0..n], leaves[0..n], round);
        var complete = false;
        while (round < 128 and !complete) : (round += 1) {
            complete = true;
            for (0..n) |i| complete = complete and joins[i] != inv.no_round and leaves[i] != inv.no_round;
            if (complete) break;
            const emitted_before = streams[0].emitted().len;
            const trace_before = recorder.trace.items.len;
            try engine.step();
            const committed = streams[0].emitted().len - emitted_before;
            const kept = trace_before + committed;
            if (recorder.trace.items.len < kept) return error.MissingCommittedPositions;
            // One verify window records accepted target positions before tentative rejects; keep committed rows only.
            recorder.trace.items.len = kept;
            try addDue(round + 1, &engine, plan, streams[0..n], joins[0..n]);
            markLeaves(streams[0..n], leaves[0..n], round + 1);
        }
        if (!complete) return error.ScheduleStalled;
        const emitted = streams[0].emitted();
        if (out.tokens.len < emitted.len or out.rng_positions.len < recorder.trace.items.len) {
            return error.ObservationTooSmall;
        }
        @memcpy(out.tokens[0..emitted.len], emitted);
        @memcpy(out.rng_positions[0..recorder.trace.items.len], recorder.trace.items);
        out.token_count = emitted.len;
        out.rng_count = recorder.trace.items.len;
        out.finished = streams[0].finished;
        out.reason = streams[0].reason;
        out.accepted = streams[0].accepted;
        out.cached = streams[0].cached;
        out.overlap = recorder.overlap;
        out.join_rounds = joins;
        out.leave_rounds = leaves;
        r.applyFault(plan, out);
    }
};

fn observation(tokens: []u32, positions: []u64) inv.Observation {
    return .{ .tokens = tokens, .rng_positions = positions };
}

fn expectControl(fault: Fault, plan: inv.Plan, expected: anyerror) !void {
    var tokens_a: [64]u32 = undefined;
    var tokens_b: [64]u32 = undefined;
    var positions_a: [128]u64 = undefined;
    var positions_b: [128]u64 = undefined;
    var solo = observation(&tokens_a, &positions_a);
    var batch = observation(&tokens_b, &positions_b);
    var runner: FakeRunner = .{ .fault = fault };
    try std.testing.expectError(expected, inv.checkPlan(runner.gate(), plan, &solo, &batch));
}

test "lane invariance: fake target dynamic cohorts" {
    var tokens_a: [64]u32 = undefined;
    var tokens_b: [64]u32 = undefined;
    var positions_a: [128]u64 = undefined;
    var positions_b: [128]u64 = undefined;
    var solo = observation(&tokens_a, &positions_a);
    var batch = observation(&tokens_b, &positions_b);
    var runner: FakeRunner = .{};
    try inv.check(runner.gate(), &solo, &batch);
}

test "lane invariance: planted mismatches evidence and cohort shrinking" {
    const plain = inv.makePlan(2, .before, false, true, false, .length);
    try expectControl(.token, plain, error.TokenMismatch);
    try expectControl(.rng, plain, error.RngPositionMismatch);
    try expectControl(.rng, inv.makePlan(2, .mid, true, true, false, .length), error.RngPositionMismatch);
    try expectControl(.reason, inv.makePlan(2, .before, false, true, false, .natural), error.StopReasonMismatch);
    try expectControl(.no_draft, inv.makePlan(2, .before, true, true, false, .length), error.MissingDraftAcceptance);
    try expectControl(.no_restore, inv.makePlan(2, .before, false, true, true, .length), error.MissingPrefixRestoration);
    try expectControl(.no_overlap, plain, error.MissingObservedOverlap);
    try expectControl(.no_join, plain, error.MissingJoinLeaveEvidence);
    try expectControl(.bad_before_join, plain, error.BeforeJoinWasNotObserved);
    try expectControl(.early_leave, inv.makePlan(2, .mid, false, true, false, .length), error.LeavePhaseMismatch);

    var tokens_a: [64]u32 = undefined;
    var tokens_b: [64]u32 = undefined;
    var positions_a: [128]u64 = undefined;
    var positions_b: [128]u64 = undefined;
    var solo = observation(&tokens_a, &positions_a);
    var batch = observation(&tokens_b, &positions_b);
    var runner: FakeRunner = .{ .fault = .token_nonmonotonic };
    const large = inv.makePlan(8, .before, false, true, false, .length);
    try std.testing.expectError(error.TokenMismatch, inv.checkPlan(runner.gate(), large, &solo, &batch));
    try std.testing.expectEqual(@as(u8, 2), runner.last_cohort);
    const wanted = (@as(u16, 1) << 8) | (@as(u16, 1) << 2);
    try std.testing.expectEqual(wanted, runner.seen_cohorts & wanted);
    try std.testing.expectEqual(@as(u16, 0), runner.seen_cohorts & (@as(u16, 1) << 4));
}
