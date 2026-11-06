//! Counted host scheduling checks for family facts, with history-dependent draws through a long zero-row interval.
const std = @import("std");
const lanes = @import("lanes.zig");
const fixtures = @import("drafted_test.zig");
const be = lanes.backend;
const a = std.testing.allocator;

const Counted = struct {
    target: lanes.fake.Fake = .{ .gpa = a },
    first: ?*lanes.Stream = null,
    proposals: usize = 0,
    skipped: usize = 0,

    fn draft(ptr: *anyopaque, reqs: []const be.DraftRequest) !void {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        const c: *Counted = @fieldParentPtr("target", f);
        for (reqs) |r| {
            var q = r;
            if (r.stream != c.first) q.depth = 0;
            if (q.stream == c.first) {
                if (q.depth > 0) c.proposals += 1 else c.skipped += 1;
            }
            try f.backend().vtable.draft(ptr, &.{q});
            if (q.stream == c.first and q.depth > 0) {
                const l = f.lanes.get(q.stream).?;
                var history: std.ArrayList(u32) = .empty;
                defer history.deinit(a);
                if (q.rows) |path| {
                    try history.appendSlice(a, l.history.items[0..l.base]);
                    for (path) |row| try history.append(a, l.rows.items[row]);
                } else try history.appendSlice(a, l.history.items);
                var oracle = try lanes.Stream.init(a, .{ .id = "fresh-head", .prompt = history.items, .max_new = 32, .sampling = q.stream.sampling });
                defer oracle.deinit(a);
                var fresh: lanes.fake.Fake = .{ .gpa = a };
                defer fresh.deinit();
                try fresh.backend().prefill(&oracle);
                var request = q;
                request.stream = &oracle;
                request.rows = null;
                request.first = if (q.first) |feed| .{ .value = switch (feed) {
                    .value => |v| v,
                    .handle => |h| try f.backend().read(h),
                } } else .{ .value = q.follow[q.follow.len - 1] };
                try fresh.backend().draft(&.{request});
                try std.testing.expectEqualSlices(u32, fresh.lanes.get(&oracle).?.held.items, f.lanes.get(q.stream).?.held.items);
            }
        }
    }
    fn capacity(ptr: *anyopaque, s: *lanes.Stream) u32 {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        const c: *Counted = @fieldParentPtr("target", f);
        return if (s == c.first) 4 else 0;
    }
};

const Result = struct { wasted: usize, skipped: usize, resumed: bool };

fn run(policy: bool, external: bool, seed: ?u64, peers: bool, drafts: bool) !struct { result: Result, tokens: []u32 } {
    var counted: Counted = .{};
    defer counted.target.deinit();
    var table = counted.target.backend().vtable.*;
    table.draft = Counted.draft;
    table.draft_capacity = Counted.capacity;
    const native: be.Backend = .{ .ptr = &counted.target, .vtable = &table };
    var strict: fixtures.Strict = .{ .inner = native };
    defer strict.deinit();
    var head: fixtures.FakeDrafter = .{ .facts_ = .{ .depth = 4, .step_ms = 0, .plain_guard = false, .batched = true } };
    defer head.deinit();
    var wrapped = try lanes.drafted.Drafted.init(a, strict.backend(), head.drafter());
    defer wrapped.deinit();
    var costs: [32]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, r| c.* = .{ .width = @intCast(r), .ms = if (r <= 16) 1 else 1000 };
    var m: lanes.Model = .{ .exact_width = 16, .mtp = drafts, .speculate = drafts, .speculate_early = false, .drafts = 4, .draft_streams = true, .allocate_shared_drafts = policy, .window_costs = &costs, .hidden_rows = true, .batch_rows = 32, .max_streams = 16, .lone_copies = false };
    if (external) {
        m = wrapped.facts(m);
        m.allocate_shared_drafts = policy;
    }
    var cfg = try lanes.Config.init(a, m, 16, 15);
    defer cfg.deinit(a);
    cfg.timed = false;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, if (external) wrapped.backend() else native, clock.clock());
    defer e.deinit();
    var streams: [16]lanes.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    const count: usize = if (peers) streams.len else 1;
    for (streams[0..count], 0..) |*s, i| {
        s.* = try lanes.Stream.init(a, .{ .id = "skip", .prompt = &.{ 7, 13, 29 }, .max_new = if (i == 0) 260 else 180, .drafts = drafts, .sampling = if (seed) |v| .{ .seed = v, .temperature = 1.0, .top_p = 0.95 } else null });
        made += 1;
        if (i == 0) counted.first = s;
        try e.addStream(s);
    }
    const before = if (external) head.holds else counted.proposals;
    const before_scores = head.proposed_absorbs;
    try e.step();
    if (external and peers and drafts and policy) try std.testing.expectEqual(before_scores, head.proposed_absorbs);
    const wasted = (if (external) head.holds else counted.proposals) - before;
    if (peers and drafts) try std.testing.expectEqual(@as(?u32, 0), streams[0].granted);
    const paused = if (external) head.holds else counted.proposals;
    var skipped: usize = 0;
    if (peers) while (!streams[1].finished) {
        try e.step();
        if ((if (external) head.holds else counted.proposals) == paused) skipped += 1;
    };
    while (e.activeCount() > 0) try e.step();
    for (streams[1..made]) |*s| try std.testing.expectEqualSlices(u32, streams[0].emitted()[0..s.emitted().len], s.emitted());
    const resumed = (if (external) head.holds else counted.proposals) > paused;
    return .{ .result = .{ .wasted = wasted, .skipped = skipped, .resumed = resumed }, .tokens = try a.dupe(u32, streams[0].emitted()) };
}

/// The real family's policy must opt in; the control uses the previous scheduling and must count unused proposals.
pub fn check(model: lanes.Model, external: bool) !void {
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var head: fixtures.FakeDrafter = .{};
    defer head.deinit();
    var wrapped = try lanes.drafted.Drafted.init(a, target.backend(), head.drafter());
    defer wrapped.deinit();
    const facts = if (external) wrapped.facts(model) else model;
    try std.testing.expect(facts.allocate_shared_drafts);
    for ([_]?u64{ null, 1234 }) |seed| {
        const old = try run(false, external, seed, true, true);
        defer a.free(old.tokens);
        const shared = try run(facts.allocate_shared_drafts, external, seed, true, true);
        defer a.free(shared.tokens);
        const solo = try run(true, external, seed, false, true);
        defer a.free(solo.tokens);
        const plain = try run(true, false, seed, false, false);
        defer a.free(plain.tokens);
        try std.testing.expect(old.result.wasted > 0);
        try std.testing.expectEqual(@as(usize, 0), shared.result.wasted);
        try std.testing.expect(shared.result.skipped >= 128 and shared.result.resumed);
        try std.testing.expectEqualSlices(u32, plain.tokens, old.tokens);
        try std.testing.expectEqualSlices(u32, plain.tokens, shared.tokens);
        try std.testing.expectEqualSlices(u32, plain.tokens, solo.tokens);
    }
}
