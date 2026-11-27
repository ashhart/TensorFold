//! End-to-end host tests for Inkling 2 Mini SuffixLookup copy drafting, cost gating, parity, reuse, and cancellation.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const fx = @import("inkling2_copy_fixture.zig");
const backend = @import("../families/inkling2/metal/backend.zig");
const dspark = @import("../families/inkling2/metal/dspark.zig");
const st = @import("../families/inkling2/metal/state.zig");

const gpa = fx.gpa;
const io = fx.io;
const Harness = fx.Harness;
const Reply = fx.Reply;
const PauseGate = fx.PauseGate;
const runOne = fx.runOne;
const expectParity = fx.expectParity;

test "Inkling2 SuffixLookup copy drafting fires on repeated 8-gram, saves verify calls, and keeps T=0 and T=1.0 parity" {
    const prompt = [_]u32{ 1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17 };
    const samplings = [_]lanes.Sampling{
        .{ .seed = 1, .temperature = 0.0 },
        .{ .seed = 42, .temperature = 1.0, .top_k = 8, .top_p = 0.95 },
    };
    for (samplings, 0..) |sampling, idx| {
        var plain_h: Harness = undefined;
        try plain_h.open(.{ .drafts = false, .mismatch_at = prompt.len + 10 });
        defer plain_h.deinit();
        try std.testing.expect(!plain_h.host.engine().info().draft_sources.copies);

        var copy_h: Harness = undefined;
        try copy_h.open(.{ .drafts = true, .mismatch_at = prompt.len + 10 });
        defer copy_h.deinit();
        try std.testing.expect(copy_h.host.engine().info().draft_sources.copies);

        const req: api.Request = .{ .prompt = &prompt, .max_tokens = 20, .sampling = sampling, .drafts = true, .logprobs = 2 };
        var plain_r = try runOne(&plain_h, @intCast(idx * 2 + 1), &req);
        defer plain_r.deinit();
        var copy_r = try runOne(&copy_h, @intCast(idx * 2 + 2), &req);
        defer copy_r.deinit();

        try expectParity(&plain_r, &copy_r);
        try std.testing.expectEqual(@as(u64, 0), plain_r.drafted);
        try std.testing.expect(copy_r.drafted > 0);
        try std.testing.expect(copy_r.accepted > 0);
        try std.testing.expect(copy_h.target.verify_calls.load(.acquire) < plain_h.target.verify_calls.load(.acquire));
    }
}

test "Inkling2 SuffixLookup rejects short coincidental matches under copy_match=8 and honors stricter min_match" {
    const match4 = [_]u32{ 1, 30, 31, 32, 33, 34, 35, 36, 37, 9, 30, 31, 32, 33 };
    const match7 = [_]u32{ 1, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 9, 30, 31, 32, 33, 34, 35, 36 };
    const match8 = [_]u32{ 1, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 9, 30, 31, 32, 33, 34, 35, 36, 37 };
    const match10 = [_]u32{ 1, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 9, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39 };

    var plain_h: Harness = undefined;
    try plain_h.open(.{ .drafts = false });
    defer plain_h.deinit();
    var h_default: Harness = undefined;
    try h_default.open(.{ .drafts = true, .min_match = 4 });
    defer h_default.deinit();
    var h_strict: Harness = undefined;
    try h_strict.open(.{ .drafts = true, .min_match = 10 });
    defer h_strict.deinit();

    var r4 = try runOne(&h_default, 1, &.{ .prompt = &match4, .max_tokens = 3, .sampling = .{ .seed = 0, .temperature = 0 } });
    defer r4.deinit();
    try std.testing.expectEqual(@as(u64, 0), r4.drafted);

    var r7 = try runOne(&h_default, 2, &.{ .prompt = &match7, .max_tokens = 1, .sampling = .{ .seed = 0, .temperature = 0 } });
    defer r7.deinit();
    try std.testing.expectEqual(@as(u64, 0), r7.drafted);

    var r8_plain = try runOne(&plain_h, 3, &.{ .prompt = &match8, .max_tokens = 4, .sampling = .{ .seed = 0, .temperature = 0 } });
    defer r8_plain.deinit();
    var r8_def = try runOne(&h_default, 4, &.{ .prompt = &match8, .max_tokens = 4, .sampling = .{ .seed = 0, .temperature = 0 } });
    defer r8_def.deinit();
    var r8_strict = try runOne(&h_strict, 5, &.{ .prompt = &match8, .max_tokens = 2, .sampling = .{ .seed = 0, .temperature = 0 } });
    defer r8_strict.deinit();
    try expectParity(&r8_plain, &r8_def);
    try std.testing.expect(r8_def.drafted > 0 and r8_def.accepted > 0);
    try std.testing.expectEqual(@as(u64, 0), r8_strict.drafted);

    var r10_plain = try runOne(&plain_h, 6, &.{ .prompt = &match10, .max_tokens = 4, .sampling = .{ .seed = 0, .temperature = 0 } });
    defer r10_plain.deinit();
    var r10_strict = try runOne(&h_strict, 7, &.{ .prompt = &match10, .max_tokens = 4, .sampling = .{ .seed = 0, .temperature = 0 } });
    defer r10_strict.deinit();
    try expectParity(&r10_plain, &r10_strict);
    try std.testing.expect(r10_strict.drafted > 0 and r10_strict.accepted > 0);
}

test "Inkling2 SuffixLookup preserves parity across multi-stream concurrent batches (N=2)" {
    const p1 = [_]u32{ 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 2, 10, 11, 12, 13, 14, 15, 16, 17 };
    const p2 = [_]u32{ 4, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 3, 20, 21, 22, 23, 24, 25, 26, 27 };

    var solo_h: Harness = undefined;
    try solo_h.open(.{ .drafts = false, .lanes_count = 1 });
    defer solo_h.deinit();
    var batch_h: Harness = undefined;
    try batch_h.open(.{ .drafts = true, .lanes_count = 2 });
    defer batch_h.deinit();
    try std.testing.expect(batch_h.host.engine().info().draft_sources.copies);

    const req1: api.Request = .{ .prompt = &p1, .max_tokens = 16, .sampling = .{ .seed = 7, .temperature = 0.0 }, .logprobs = 2 };
    const req2: api.Request = .{ .prompt = &p2, .max_tokens = 16, .sampling = .{ .seed = 99, .temperature = 1.0, .top_k = 8, .top_p = 0.95 }, .logprobs = 2 };

    var s1 = try runOne(&solo_h, 1, &req1);
    defer s1.deinit();
    var s2 = try runOne(&solo_h, 2, &req2);
    defer s2.deinit();

    var c1: Reply = .{};
    defer c1.deinit();
    var c2: Reply = .{};
    defer c2.deinit();
    try batch_h.host.engine().submit(10, &req1, c1.sink());
    try batch_h.host.engine().submit(11, &req2, c2.sink());
    try std.testing.expectEqual(api.Reason.length, try c1.wait());
    try std.testing.expectEqual(api.Reason.length, try c2.wait());

    try expectParity(&s1, &c1);
    try expectParity(&s2, &c2);
    try std.testing.expect(c1.drafted > 0 and c2.drafted > 0);
}

test "Inkling2 SuffixLookup works identically with warm prompt cache prefix reuse" {
    const prefix = [_]u32{ 2, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
    const turn1 = prefix ++ [_]u32{ 8, 9 };
    const turn2 = prefix ++ [_]u32{ 7, 10, 11, 12, 13, 14, 15, 16, 17 };

    var plain_h: Harness = undefined;
    try plain_h.open(.{ .drafts = false, .cache_budget = 1 << 20 });
    defer plain_h.deinit();
    var copy_h: Harness = undefined;
    try copy_h.open(.{ .drafts = true, .cache_budget = 1 << 20 });
    defer copy_h.deinit();

    var p1 = try runOne(&plain_h, 1, &.{ .prompt = &turn1, .max_tokens = 4, .history_len = prefix.len, .sampling = .{ .seed = 0, .temperature = 0 } });
    defer p1.deinit();
    var c1 = try runOne(&copy_h, 1, &.{ .prompt = &turn1, .max_tokens = 4, .history_len = prefix.len, .sampling = .{ .seed = 0, .temperature = 0 } });
    defer c1.deinit();

    var p2 = try runOne(&plain_h, 2, &.{ .prompt = &turn2, .max_tokens = 12, .history_len = prefix.len, .sampling = .{ .seed = 0, .temperature = 0 } });
    defer p2.deinit();
    var c2 = try runOne(&copy_h, 2, &.{ .prompt = &turn2, .max_tokens = 12, .history_len = prefix.len, .sampling = .{ .seed = 0, .temperature = 0 } });
    defer c2.deinit();

    try std.testing.expectEqual(@as(?u32, prefix.len), c2.cached);
    try expectParity(&p2, &c2);
    try std.testing.expect(c2.drafted > 0 and c2.accepted > 0);
}

test "Inkling2 SuffixLookup request cancellation rolls back mid-copy without leaks" {
    const prompt = [_]u32{ 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 6, 10, 11, 12, 13, 14, 15, 16, 17 };
    var gate: PauseGate = .{ .arm_after_verify = .init(2) };

    var h: Harness = undefined;
    try h.open(.{ .drafts = true });
    defer h.deinit();
    h.target.gate = &gate;

    var r1: Reply = .{};
    defer r1.deinit();
    try h.host.engine().submit(1, &.{ .prompt = &prompt, .max_tokens = 128, .sampling = .{ .seed = 0, .temperature = 0 } }, r1.sink());
    while (!gate.entered.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    h.host.engine().cancel(1);
    gate.release.store(true, .release);

    const reason = try r1.wait();
    try std.testing.expectEqual(api.Reason.cancelled, reason);
    try std.testing.expectEqual(@as(usize, 0), h.target.streams.count());

    h.target.gate = null;
    var r2 = try runOne(&h, 2, &.{ .prompt = &prompt, .max_tokens = 8, .sampling = .{ .seed = 0, .temperature = 0 } });
    defer r2.deinit();
    try std.testing.expect(r2.drafted > 0);
}

test "Inkling2 Drafted DSpark skips holds on 12-gram copy rounds via copy_first while absorbing kept rows" {
    const copy_prompt = [_]u32{
        1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    };
    const plain_prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 40, 41, 42 };

    var plain_h: Harness = undefined;
    try plain_h.open(.{ .drafts = false });
    defer plain_h.deinit();
    var h: Harness = undefined;
    try h.open(.{ .drafts = true, .with_dspark = true });
    defer h.deinit();

    const info = h.host.engine().info().draft_sources;
    try std.testing.expect(info.copies);
    try std.testing.expect(info.drafter != null);

    // Phase A: open may run one initial hold; subsequent copy rounds absorb kept rows without hold().
    var p_copy = try runOne(&plain_h, 1, &.{ .prompt = &copy_prompt, .max_tokens = 20, .sampling = .{ .seed = 0, .temperature = 0 } });
    defer p_copy.deinit();
    var d_copy = try runOne(&h, 1, &.{ .prompt = &copy_prompt, .max_tokens = 20, .sampling = .{ .seed = 0, .temperature = 0 } });
    defer d_copy.deinit();
    try expectParity(&p_copy, &d_copy);
    try std.testing.expect(d_copy.drafted > 0);
    try std.testing.expect(h.drafter.?.absorb_calls.load(.acquire) >= 2);
    const holds_after_copies = h.drafter.?.hold_calls.load(.acquire);
    try std.testing.expect(holds_after_copies <= 1);

    // Phase B: non-repeating prompt has no copy ahead, so DSpark resumes hold() across decode rounds.
    var p_nocopy = try runOne(&plain_h, 2, &.{ .prompt = &plain_prompt, .max_tokens = 12, .sampling = .{ .seed = 0, .temperature = 0 } });
    defer p_nocopy.deinit();
    var d_nocopy = try runOne(&h, 2, &.{ .prompt = &plain_prompt, .max_tokens = 12, .sampling = .{ .seed = 0, .temperature = 0 } });
    defer d_nocopy.deinit();
    try expectParity(&p_nocopy, &d_nocopy);
    try std.testing.expect(h.drafter.?.hold_calls.load(.acquire) > holds_after_copies + 1);
}

test "Inkling2 copy_guard skips unprofitable copies on expensive verify and keeps fast high-hit copies" {
    const MissProposer = struct {
        const fake_copy = [_]u32{ 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 49 };
        fn propose(_: *anyopaque, _: []const u32, most: i64) anyerror![]const u32 {
            const n: usize = @intCast(@min(@as(i64, fake_copy.len), @max(0, most)));
            return fake_copy[0..n];
        }
        fn lastMatch(_: *anyopaque) i64 {
            return 16;
        }
        const vtable: lanes.proposer.Proposer.VTable = .{ .propose = propose, .last_match = lastMatch };
        fn make(ctx: *anyopaque, _: std.mem.Allocator, _: []const u32, _: []const u32) anyerror!lanes.proposer.Proposer {
            return .{ .ptr = ctx, .vtable = &vtable };
        }
        fn free(_: *anyopaque, _: lanes.proposer.Proposer) void {}
    };
    const startup_costs = blk: {
        var c: [16]lanes.config.Cost = undefined;
        for (&c, 1..) |*x, w| x.* = .{ .width = @intCast(w), .ms = 5 + 0.2 * @as(f64, @floatFromInt(w)) };
        break :blk c;
    };
    var dummy: u8 = 0;
    const miss_factory: @FieldType(api.LaneHost, "proposers") = .{ .ctx = &dummy, .make = MissProposer.make, .free = MissProposer.free };
    const base_prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

    // Expensive verify curve (M3-like long context): copy_guard prices unprofitable copies off.
    var plain_h: Harness = undefined;
    try plain_h.open(.{ .drafts = false });
    defer plain_h.deinit();
    var guarded_h: Harness = undefined;
    try guarded_h.open(.{ .drafts = true, .window_costs = &startup_costs, .step_ms = .{ 50, 40 }, .proposers = miss_factory });
    defer guarded_h.deinit();
    var unguarded_h: Harness = undefined;
    try unguarded_h.open(.{ .drafts = true, .window_costs = &startup_costs, .step_ms = .{ 50, 40 }, .copy_guard_override = false, .proposers = miss_factory });
    defer unguarded_h.deinit();

    const miss_req: api.Request = .{ .prompt = &base_prompt, .max_tokens = 64, .sampling = .{ .seed = 0, .temperature = 0 }, .logprobs = 2 };
    var r_plain = try runOne(&plain_h, 1, &miss_req);
    defer r_plain.deinit();
    var r_guarded = try runOne(&guarded_h, 2, &miss_req);
    defer r_guarded.deinit();
    var r_unguarded = try runOne(&unguarded_h, 3, &miss_req);
    defer r_unguarded.deinit();

    try expectParity(&r_plain, &r_guarded);
    try expectParity(&r_plain, &r_unguarded);
    try std.testing.expect(r_guarded.drafted * 3 < r_unguarded.drafted);
    try std.testing.expect(guarded_h.target.verify_rows.load(.acquire) < unguarded_h.target.verify_rows.load(.acquire));

    // Fast verify curve with genuine >= 8-token repeating span: copy_guard keeps drafting and saves verify calls.
    const hit_prompt = [_]u32{ 1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17 };
    var fast_plain: Harness = undefined;
    try fast_plain.open(.{ .drafts = false });
    defer fast_plain.deinit();
    var fast_guarded: Harness = undefined;
    try fast_guarded.open(.{ .drafts = true, .window_costs = &startup_costs, .step_ms = .{ 10, 0.2 } });
    defer fast_guarded.deinit();

    const hit_req: api.Request = .{ .prompt = &hit_prompt, .max_tokens = 24, .sampling = .{ .seed = 0, .temperature = 0 }, .logprobs = 2 };
    var fp = try runOne(&fast_plain, 4, &hit_req);
    defer fp.deinit();
    var fg = try runOne(&fast_guarded, 5, &hit_req);
    defer fg.deinit();
    try expectParity(&fp, &fg);
    try std.testing.expect(fg.drafted > 0 and fg.accepted > 0);
    try std.testing.expect(fast_guarded.target.verify_calls.load(.acquire) < fast_plain.target.verify_calls.load(.acquire));
}

test "Inkling2 own_copy_rates trims breaking copy streams in shared batches while keeping parity" {
    const MissProposer = struct {
        const fake_copy = [_]u32{ 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 49 };
        fn propose(_: *anyopaque, _: []const u32, most: i64) anyerror![]const u32 {
            const n: usize = @intCast(@min(@as(i64, fake_copy.len), @max(0, most)));
            return fake_copy[0..n];
        }
        fn lastMatch(_: *anyopaque) i64 {
            return 16;
        }
        const vtable: lanes.proposer.Proposer.VTable = .{ .propose = propose, .last_match = lastMatch };
        fn make(ctx: *anyopaque, _: std.mem.Allocator, _: []const u32, _: []const u32) anyerror!lanes.proposer.Proposer {
            return .{ .ptr = ctx, .vtable = &vtable };
        }
        fn free(_: *anyopaque, _: lanes.proposer.Proposer) void {}
    };
    const w_costs = blk: {
        var c: [16]lanes.config.Cost = undefined;
        for (&c, 1..) |*x, w| x.* = .{ .width = @intCast(w), .ms = 10 + 3.0 * @as(f64, @floatFromInt(w - 1)) };
        break :blk c;
    };
    const s_costs = [_]lanes.config.Cost{ .{ .width = 24, .ms = 79 }, .{ .width = 32, .ms = 103 } };
    var dummy: u8 = 0;
    const miss_factory: @FieldType(api.LaneHost, "proposers") = .{ .ctx = &dummy, .make = MissProposer.make, .free = MissProposer.free };
    const p1 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const p2 = [_]u32{ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

    var solo_h: Harness = undefined;
    try solo_h.open(.{ .drafts = false, .lanes_count = 1 });
    defer solo_h.deinit();
    var shared_h: Harness = undefined;
    try shared_h.open(.{ .drafts = true, .lanes_count = 2, .window_costs = &w_costs, .shared_costs = &s_costs, .step_ms = .{ 10, 3.0 }, .proposers = miss_factory });
    defer shared_h.deinit();

    const req1: api.Request = .{ .prompt = &p1, .max_tokens = 32, .sampling = .{ .seed = 11, .temperature = 1.0, .top_k = 8, .top_p = 0.95 }, .logprobs = 2 };
    const req2: api.Request = .{ .prompt = &p2, .max_tokens = 32, .sampling = .{ .seed = 29, .temperature = 1.0, .top_k = 8, .top_p = 0.95 }, .logprobs = 2 };

    var s1 = try runOne(&solo_h, 1, &req1);
    defer s1.deinit();
    var s2 = try runOne(&solo_h, 2, &req2);
    defer s2.deinit();

    var c1: Reply = .{};
    defer c1.deinit();
    var c2: Reply = .{};
    defer c2.deinit();
    try shared_h.host.engine().submit(10, &req1, c1.sink());
    try shared_h.host.engine().submit(11, &req2, c2.sink());
    try std.testing.expectEqual(api.Reason.length, try c1.wait());
    try std.testing.expectEqual(api.Reason.length, try c2.wait());

    try expectParity(&s1, &c1);
    try expectParity(&s2, &c2);
    // Without own_copy_rates, 32 rounds * 15 rows = 480 drafts per stream; with it, misses trim grants sharply.
    try std.testing.expect(c1.drafted + c2.drafted < 32 * 8);
}

test "Inkling2 SuffixLookup proposeAfter graft compatibility leaks zero bytes" {
    var lookup = try lanes.SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer lookup.deinit();
    const p = lookup.proposer();
    const history = [_]u32{ 10, 11, 12, 13, 14, 15, 16, 17, 1, 10, 11, 12 };
    try std.testing.expectEqual(@as(usize, 0), (try p.propose(&history, 8)).len);
    const grafted = try p.proposeAfter(&history, &.{13}, 8);
    try std.testing.expect(grafted.len >= 4);
    try std.testing.expectEqualSlices(u32, &.{ 14, 15, 16, 17 }, grafted[0..4]);
    try std.testing.expect(p.lastMatch() >= 4);
}

test "Inkling native startup draft_sources reports copies with --drafts, clears them with --no-drafts, and keeps DSpark --drafter" {
    const Stub = struct {
        fn prefill(_: *anyopaque, _: *lanes.Stream) !void {}
        fn first(_: *anyopaque, _: *lanes.Stream, _: u64) !u64 {
            return 0;
        }
        fn queue(_: *anyopaque, _: *lanes.Stream, _: lanes.backend.Feed, _: u64) !u64 {
            return 0;
        }
        fn read(_: *anyopaque, _: u64) !u32 {
            return 0;
        }
        fn verify(_: *anyopaque, _: []const lanes.backend.Window, _: []lanes.backend.Verified) !void {}
        fn keep(_: *anyopaque, _: []const lanes.backend.Window, _: []const []const u32) !void {}
        fn draft(_: *anyopaque, _: []const lanes.backend.DraftRequest) !void {}
        fn release(_: *anyopaque, _: *lanes.Stream) void {}
        const backend_impl: lanes.backend.Backend = .{
            .keep_full_lone = true,
            .ptr = undefined,
            .vtable = &.{
                .prefill = prefill,
                .first = first,
                .queue = queue,
                .read = read,
                .verify = verify,
                .keep = keep,
                .draft = draft,
                .release = release,
            },
        };
    };
    const StubDrafter = struct {
        fn taps(_: *anyopaque) []const u32 {
            return &.{0};
        }
        fn facts(_: *anyopaque) lanes.drafter.Facts {
            return .{ .depth = 7, .step_ms = 0, .copy_first = true, .copy_match = 12, .plain_guard = false };
        }
        fn openStream(_: *anyopaque, _: *lanes.Stream) !void {}
        fn absorb(_: *anyopaque, _: []const lanes.drafter.Absorb) !void {}
        fn hold(_: *anyopaque, _: []const lanes.drafter.Hold) !void {}
        fn held(_: *anyopaque, _: []const *lanes.Stream, _: []const []u32) !void {}
        fn releaseStream(_: *anyopaque, _: *lanes.Stream) void {}
        const vtable: lanes.drafter.Drafter.VTable = .{
            .taps = taps,
            .facts = facts,
            .open = openStream,
            .absorb = absorb,
            .hold = hold,
            .held = held,
            .release = releaseStream,
        };
    };

    var dummy: u8 = 0;
    var clock: lanes.backend.WallClock = .{ .io = std.testing.io };
    var target_b = Stub.backend_impl;
    target_b.ptr = &dummy;

    // 1) Verify DSpark Drafter facts directly enable copy_first.
    var ds: dspark.Drafter = .{ .gpa = gpa, .target = undefined, .weights = undefined, .pipelines = undefined, .sampler = undefined, .scratch = undefined, .capacity = 64 };
    const ds_facts = ds.drafter().facts();
    try std.testing.expect(ds_facts.copy_first);
    try std.testing.expectEqual(@as(u32, 7), ds_facts.depth);
    try std.testing.expectEqual(@as(i64, 12), ds_facts.copy_match);
    try std.testing.expect(!ds_facts.plain_guard);

    // 2) Test --drafts vs --no-drafts across 1 and 2 lanes, and --drafts + DSpark --drafter.
    for ([_]u32{ 1, 2 }) |stream_count| {
        for ([_]bool{ true, false }) |drafts_on| {
            var m: backend.Metal = .{ .gpa = gpa, .model = undefined, .options = .{ .capacity = 64, .streams = stream_count, .drafts = drafts_on, .budget = 0 }, .scratch = undefined, .prompt = undefined, .tokens = &.{}, .values = &.{} };
            for (0..st.window_rows) |i| m.costs[i] = .{ .width = @intCast(i + 1), .ms = 10 + 0.5 * @as(f64, @floatFromInt(i)) };
            m.timed = st.window_rows;
            const mf = m.facts();
            try std.testing.expectEqual(drafts_on, mf.copy_drafts);
            try std.testing.expectEqual(drafts_on, mf.copy_guard);
            try std.testing.expectEqual(drafts_on, mf.own_copy_rates);
            try std.testing.expectEqual(@as(i64, 8), mf.copy_match);

            var cfg = try lanes.Config.init(gpa, mf, st.window_rows, st.window_rows - 1);
            defer cfg.deinit(gpa);
            try std.testing.expectEqual(@as(u32, if (drafts_on) st.window_rows - 1 else 0), cfg.max_copy);
            try std.testing.expectEqual(drafts_on, cfg.copy_guard);
            try std.testing.expectEqual(drafts_on, cfg.own_copy_rates);
            try std.testing.expectEqual(@as(i64, 8), cfg.enter_match);
            try std.testing.expect(!cfg.family_costs.empty());

            var core = lanes.Engine.init(gpa, &cfg, target_b, clock.clock());
            defer core.deinit();
            var host = api.LaneHost.init(gpa, std.testing.io, &core, .{
                .lanes = stream_count,
                .context_window = 64,
                .prefill_step = st.prompt_rows,
                .structures = true,
                .logprobs = true,
                .draft_sources = .{ .drafter = null },
            });
            host.stop();
            const ds_info = host.engine().info().draft_sources;
            try std.testing.expectEqual(drafts_on, ds_info.copies);
            try std.testing.expect(!ds_info.mtp);
            try std.testing.expect(ds_info.drafter == null);
        }
    }

    var m: backend.Metal = .{ .gpa = gpa, .model = undefined, .options = .{ .capacity = 64, .streams = 2, .drafts = true, .budget = 0 }, .scratch = undefined, .prompt = undefined, .tokens = &.{}, .values = &.{} };
    var drafted = try lanes.drafted.Drafted.init(gpa, target_b, .{ .ptr = &dummy, .vtable = &StubDrafter.vtable });
    defer drafted.deinit();
    const df = drafted.facts(m.facts());
    try std.testing.expect(df.copy_drafts);
    try std.testing.expect(df.copy_guard);
    try std.testing.expect(df.own_copy_rates);
    try std.testing.expect(df.copy_first);
    try std.testing.expectEqual(@as(i64, 12), df.copy_match);
    try std.testing.expectEqual(@as(u32, 7), df.drafts);

    var cfg = try lanes.Config.init(gpa, df, st.window_rows, st.window_rows - 1);
    defer cfg.deinit(gpa);
    var core = lanes.Engine.init(gpa, &cfg, drafted.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, std.testing.io, &core, .{
        .lanes = 2,
        .context_window = 64,
        .prefill_step = st.prompt_rows,
        .structures = true,
        .logprobs = true,
        .draft_sources = .{ .drafter = .{ .kind = "DSpark", .name = "test-dspark" } },
    });
    host.stop();
    const with_drafter = host.engine().info().draft_sources;
    try std.testing.expect(with_drafter.copies);
    try std.testing.expect(!with_drafter.mtp);
    try std.testing.expect(with_drafter.drafter != null);
    try std.testing.expectEqualStrings("DSpark", with_drafter.drafter.?.kind);
    try std.testing.expectEqualStrings("test-dspark", with_drafter.drafter.?.name);
}
