//! Actual lane-core begin/commit/done callbacks with a CPU target and recording CUDA events.
const std = @import("std");
const lanes = @import("lanes");
const cuda = @import("cuda");
const timing = @import("qwen35").native.round_timing;
const Rig = @import("attribution_fake").Rig;
const be = lanes.backend;
const t = std.testing;
const Target = struct {
    fake: lanes.fake.Fake,
    rig: Rig,
    trace: timing.Trace,
    table: be.Backend.VTable = undefined,
    fn backend(self: *Target) be.Backend {
        self.table = self.fake.backend().vtable.*;
        self.table.verify = verify;
        return .{ .ptr = &self.fake, .vtable = &self.table, .round_observer = self.trace.observer() };
    }
    fn verify(ptr: *anyopaque, windows: []const be.Window, output: []be.Verified) !void {
        const self: *Target = @fieldParentPtr("fake", @as(*lanes.fake.Fake, @ptrCast(@alignCast(ptr))));
        const scope = cuda.round_attribution.scope(.forward);
        defer scope.end();
        var rows: usize = 0;
        for (windows) |w| rows += w.rows();
        try self.rig.timing.frames.append(t.allocator, .{ .correlation = scope.correlation, .live_rows = rows, .windows = windows.len, .shape_bucket = timing.Shape.init(rows, windows.len, true, false, 1).total, .shape_key = 1, .mt2 = false });
        try self.rig.kernel();
        _ = self.rig.driver.api.cuStreamSynchronize(null);
        try self.fake.backend().verify(windows, output);
    }
};
test "round attribution core: complete, clipped and shared commits come from actual acceptance" {
    if (comptime !@import("profile").round_attribution.enabled) return error.SkipZigTest;
    for ([_]usize{ 1, 2 }) |count| for ([_]u8{ 0, 1, 2, 3 }) |mode| {
        const rig = try Rig.init();
        defer rig.deinit();
        const id: [64]u8 = @splat('a');
        rig.timing.identity = &id;
        var target: Target = .{ .fake = .{ .gpa = t.allocator }, .rig = rig, .trace = .{ .extra = rig.timing, .legacy = false } };
        defer target.fake.deinit();
        var cfg = try lanes.Config.init(t.allocator, .{ .exact_width = 16, .hidden_rows = true, .streams_exact = true, .batch_rows = 128, .max_streams = 8 }, 16, 15);
        defer cfg.deinit(t.allocator);
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(t.allocator, &cfg, target.backend(), clock.clock());
        defer core.deinit();
        var streams: [2]lanes.Stream = undefined;
        var initialized: usize = 0;
        defer for (streams[0..initialized]) |*s| s.deinit(t.allocator);
        var before: [2]usize = undefined;
        for (streams[0..count], 0..) |*s, i| {
            s.* = try lanes.Stream.init(t.allocator, .{ .id = if (i == 0) "alpha" else "beta", .prompt = &.{ 1, 2, 3 }, .max_new = if (mode == 1) 2 else 20, .eos = &.{} });
            initialized += 1;
            try core.addStream(s);
            before[i] = s.emitted().len;
            if (mode < 2) {
                try s.force.appendSlice(t.allocator, &.{ 11, 12, 13, 14 });
            } else {
                var history: std.ArrayList(u32) = .empty;
                defer history.deinit(t.allocator);
                try history.appendSlice(t.allocator, s.prompt());
                try history.append(t.allocator, s.pending.?);
                const guesses = try t.allocator.alloc(u32, 3);
                for (guesses) |*guess| {
                    guess.* = if (mode == 2) 0 else lanes.fake.next(history.items, s.sampling, history.items.len);
                    try history.append(t.allocator, guess.*);
                }
                s.next = .{ .count = 3, .tokens = guesses };
            }
        }
        try core.step();
        try t.expect(!rig.timing.active);
        try t.expect(rig.timing.problem == null);
        try t.expectEqual(count, rig.timing.requests.items.len);
        try t.expectEqual(count, rig.timing.windows.items.len);
        for (streams[0..count], rig.timing.windows.items, 0..) |*s, w, i| {
            try t.expectEqualSlices(u32, s.emitted()[before[i]..], w.committed_ids);
            try t.expectEqual(s.cache_len, w.cached_after);
            try t.expectEqual(s.pending, w.pending_after);
            try t.expectEqual(if (mode < 2) @as(usize, 0) else w.kept_path.len - 1, w.connected_accepted_depth);
            try t.expectEqual(w.offered_rows - w.kept_path.len, w.rolled_back_rows);
            try t.expectEqual(if (mode < 2) @as(usize, 0) else w.offered_rows - 1, w.proposal_nodes);
            if (mode == 2) try t.expect(w.rolled_back_rows > 0);
            if (mode == 3) try t.expect(w.connected_accepted_depth > 0);
        }
    };
}

test "attribution review: legacy frame compiles attribution out" {
    if (comptime @import("profile").round_attribution.enabled) return error.SkipZigTest;
    const rig = try Rig.init();
    defer rig.deinit();
    try rig.timing.frames.append(t.allocator, .{ .correlation = 1, .live_rows = 123, .windows = 1, .shape_bucket = 1, .shape_key = 1, .mt2 = false });
    var legacy: timing.Trace = .{ .extra = rig.timing };
    legacy.frame(0, 1, 4, 4, 1);
    try t.expectEqual(@as(usize, 123), rig.timing.frames.items[0].live_rows);
}
test "attribution review: legacy done compiles attribution out" {
    if (comptime @import("profile").round_attribution.enabled) return error.SkipZigTest;
    const rig = try Rig.init();
    defer rig.deinit();
    var legacy: timing.Trace = .{ .extra = rig.timing };
    const waits = rig.waitCount();
    legacy.observer().done(&legacy, &.{});
    try t.expectEqual(waits, rig.waitCount());
}

test "attribution review: MTP absorption enters the proposal scope" {
    if (comptime !@import("profile").round_attribution.enabled) return error.SkipZigTest;
    const rig = try Rig.init();
    defer rig.deinit();
    rig.timing.begin();
    var mtp: @import("qwen35").mtp.Mtp = undefined;
    mtp.io = t.io;
    mtp.trees = false;
    mtp.batch = false;
    try mtp.drafter().vtable.absorb(&mtp, &.{});
    try t.expectEqual(@as(usize, 1), rig.timing.host.items.len);
    try t.expectEqual(cuda.round_attribution.model.Phase.proposal, rig.timing.host.items[0].phase);
    try t.expectEqual(cuda.round_attribution.model.Phase.other, rig.timing.phase);
}

const pc = lanes.policy_clock;
const Slow = struct {
    ms: f64 = 1,
    fn start(_: *anyopaque) void {}
    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 {
        return @as(*Slow, @ptrCast(@alignCast(ptr))).ms;
    }
    fn clock(s: *Slow) be.Clock {
        return .{ .ptr = s, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }
};
test "attribution review: slow tracing cannot contaminate replayed adaptive cost or grants" {
    var slow: Slow = .{};
    var tape: pc.Tape = .{ .a = t.allocator, .base = slow.clock() };
    defer tape.deinit();
    tape.clock().start();
    const original = tape.clock().elapsedMs(.cost);
    slow.ms = 1000;
    const contaminated = slow.clock().elapsedMs(.cost);
    try t.expect(original != contaminated);
    tape.replay = tape.recorded.items;
    tape.clock().start();
    const fixed = tape.clock().elapsedMs(.cost);
    try t.expectEqual(original, fixed);
    var cfg = try lanes.Config.init(t.allocator, .{ .exact_width = 16, .hidden_rows = true, .drafts = 3, .window_costs = &.{ .{ .width = 1, .ms = 1 }, .{ .width = 2, .ms = 1 }, .{ .width = 3, .ms = 1 }, .{ .width = 4, .ms = 1 } } }, 16, 15);
    defer cfg.deinit(t.allocator);
    const costs = [_]f64{ original, contaminated, fixed };
    var grants: [3]i64 = undefined;
    for (costs, &grants) |cost, *grant| {
        var fake: lanes.fake.Fake = .{ .gpa = t.allocator };
        defer fake.deinit();
        var core = lanes.Engine.init(t.allocator, &cfg, fake.backend(), slow.clock());
        defer core.deinit();
        try core.rule.observeCost(0, 1, false, null);
        try core.rule.observeCost(3, cost, false, null);
        var state: ?@import("lanes").depth.State = null;
        defer if (state) |*s| s.deinit(t.allocator);
        const who: @import("lanes").depth.Who = .{ .state = &state, .draft_room = 4, .finished = false, .forced = false, .sampled = true };
        grant.* = try core.rule.headDepth(&who, null, .{ .ptr = &slow, .copy = struct {
            fn no(_: *anyopaque, _: *const @import("lanes").depth.Who) bool {
                return false;
            }
        }.no });
    }
    try t.expect(grants[0] != grants[1]);
    try t.expectEqual(grants[0], grants[2]);
    try t.expect(tape.complete());
    try t.expectError(error.PolicyClockTapeExhausted, tape.take(.{ .kind = .elapsed }));
    tape.at = 0;
    try t.expectError(error.PolicyClockWorkloadChanged, tape.take(.{ .kind = .verify, .key = 42 }));
}

const PolicyTarget = struct {
    fake: lanes.fake.Fake,
    slow: *Slow,
    tracing: bool,
    shapes: std.ArrayList(u64) = .empty,
    table: be.Backend.VTable = undefined,
    fn backend(p: *PolicyTarget) be.Backend {
        p.table = p.fake.backend().vtable.*;
        p.table.verify = verify;
        return .{ .ptr = &p.fake, .vtable = &p.table };
    }
    fn verify(ptr: *anyopaque, windows: []const be.Window, output: []be.Verified) !void {
        const p: *PolicyTarget = @fieldParentPtr("fake", @as(*lanes.fake.Fake, @ptrCast(@alignCast(ptr))));
        if (p.tracing) p.slow.ms = 1000;
        var rows: usize = 0;
        for (windows) |w| rows += w.rows();
        pc.frame(timing.Shape.init(rows, windows.len, true, false, 1).key());
        try p.shapes.append(t.allocator, pc.verifyKey(windows));
        try p.fake.backend().verify(windows, output);
    }
};
const PolicyResult = struct {
    events: []const pc.Event,
    shapes: []u64,
    tokens: []u32,
    drafted: u64,
    accepted: u64,
    fn deinit(r: PolicyResult) void {
        t.allocator.free(r.events);
        t.allocator.free(r.shapes);
        t.allocator.free(r.tokens);
    }
};
fn policyRun(tracing: bool, replay: ?[]const pc.Event, use_tape: bool, streams: usize) !PolicyResult {
    var slow: Slow = .{};
    var tape: pc.Tape = .{ .a = t.allocator, .base = slow.clock(), .replay = replay };
    defer tape.deinit();
    var target: PolicyTarget = .{ .fake = .{ .gpa = t.allocator }, .slow = &slow, .tracing = tracing };
    defer target.fake.deinit();
    defer target.shapes.deinit(t.allocator);
    var cfg = try lanes.Config.init(t.allocator, .{ .exact_width = 16, .hidden_rows = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 3, .copy_drafts = false, .window_costs = &.{ .{ .width = 1, .ms = 1 }, .{ .width = 2, .ms = 1 }, .{ .width = 3, .ms = 1 }, .{ .width = 4, .ms = 1 } } }, 16, 15);
    defer cfg.deinit(t.allocator);
    var core = lanes.Engine.init(t.allocator, &cfg, target.backend(), if (use_tape) tape.clock() else slow.clock());
    defer core.deinit();
    var ss: [4]lanes.Stream = undefined;
    var live: usize = 0;
    defer for (ss[0..live]) |*s| s.deinit(t.allocator);
    const ids = [_][]const u8{ "a", "b", "c", "d" };
    for (ss[0..streams], 0..) |*s, i| {
        s.* = try lanes.Stream.init(t.allocator, .{ .id = ids[i], .prompt = &.{ 1, 2, 3 }, .max_new = 64, .eos = &.{} });
        live += 1;
        try core.addStream(s);
    }
    for (0..12) |_| try core.step();
    if (use_tape) try t.expect(tape.complete());
    var tokens: std.ArrayList(u32) = .empty;
    errdefer tokens.deinit(t.allocator);
    for (ss[0..streams]) |s| try tokens.appendSlice(t.allocator, s.emitted());
    return .{ .events = try t.allocator.dupe(pc.Event, tape.recorded.items), .shapes = try t.allocator.dupe(u64, target.shapes.items), .tokens = try tokens.toOwnedSlice(t.allocator), .drafted = core.drafted, .accepted = core.accepted };
}
test "attribution review: policy replay pins actual lone and shared costs windows tokens and counters" {
    if (comptime !@import("profile").round_attribution.enabled) return error.SkipZigTest;
    for ([_]usize{ 1, 4 }) |streams| {
        const original = try policyRun(false, null, true, streams);
        defer original.deinit();
        const fixed = try policyRun(true, original.events, true, streams);
        defer fixed.deinit();
        try t.expectEqualSlices(u64, original.shapes, fixed.shapes);
        try t.expectEqualSlices(u32, original.tokens, fixed.tokens);
        try t.expectEqual(original.drafted, fixed.drafted);
        try t.expectEqual(original.accepted, fixed.accepted);
        if (streams == 1) {
            const old = try policyRun(true, null, false, streams);
            defer old.deinit();
            try t.expect(!std.mem.eql(u64, original.shapes, old.shapes));
            try t.expect(original.drafted != old.drafted);
        }
    }
}
