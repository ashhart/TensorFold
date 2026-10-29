//! Counted shared-drafter scheduling and hybrid windows through the real lane planner, without a GPU.
const std = @import("std");
const lanes = @import("lanes");
const a = std.testing.allocator;
const be = lanes.backend;

const Counted = struct {
    fake: lanes.fake.Fake = .{ .gpa = a },
    first: ?*lanes.Stream = null,
    proposals: usize = 0,
    fn capacity(ptr: *anyopaque, s: *lanes.Stream) u32 {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        const c: *Counted = @fieldParentPtr("fake", f);
        return if (s == c.first) 15 else 0;
    }
    fn draft(ptr: *anyopaque, reqs: []const be.DraftRequest) !void {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        const c: *Counted = @fieldParentPtr("fake", f);
        for (reqs) |r| {
            var q = r;
            if (r.stream != c.first) q.depth = 0;
            if (q.depth > 0) {
                c.proposals += 1;
            }
            try f.backend().vtable.draft(ptr, &.{q});
        }
    }
};

fn shared(policy: bool) !struct { wasted: usize, resumed: bool } {
    var costs: [32]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, r| c.* = .{ .width = @intCast(r), .ms = if (r <= 16) 1 else 1000 };
    var cfg = try lanes.Config.init(a, .{ .exact_width = 32, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 15, .draft_probabilities = true, .draft_streams = true, .allocate_shared_drafts = policy, .window_costs = &costs, .hidden_rows = true, .batch_rows = 32, .max_streams = 16, .lone_copies = false }, 32, 15);
    defer cfg.deinit(a);
    cfg.timed = false;
    var counted: Counted = .{};
    defer counted.fake.deinit();
    var table = counted.fake.backend().vtable.*;
    table.draft = Counted.draft;
    table.draft_capacity = Counted.capacity;
    var backend = counted.fake.backend();
    backend.vtable = &table;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, backend, clock.clock());
    defer e.deinit();
    var ss: [16]lanes.Stream = undefined;
    var made: usize = 0;
    defer for (ss[0..made]) |*s| s.deinit(a);
    for (&ss, 0..) |*s, i| {
        s.* = try lanes.Stream.init(a, .{ .id = "counted", .prompt = &.{ 7, 13, 29 }, .max_new = if (i == 0) 24 else 6 });
        made += 1;
        if (i == 0) counted.first = s;
        try e.addStream(s);
    }
    const before = counted.proposals;
    try e.step();
    try std.testing.expectEqual(@as(?u32, 0), ss[0].granted);
    const wasted = counted.proposals - before;
    const paused = counted.proposals;
    try e.step();
    if (policy) try std.testing.expectEqual(paused, counted.proposals);
    while (!ss[1].finished) try e.step();
    const resumed = counted.proposals > paused;
    while (e.activeCount() > 0) try e.step();
    return .{ .wasted = wasted, .resumed = resumed };
}

test "qwen27 window: zero allocated draft rows skip counted proposal passes and resume when rows return" {
    const old = try shared(false);
    try std.testing.expect(old.wasted > 0);
    const got = try shared(true);
    try std.testing.expectEqual(@as(usize, 0), got.wasted);
    try std.testing.expect(got.resumed);
}

test "qwen27 window: context copy joins a 15-node tree only within the target window" {
    for ([_]u32{ 16, 32, 64 }) |width| for ([_]bool{ false, true }) |copies| {
        var cfg = try lanes.Config.init(a, .{ .exact_width = width, .mtp = true, .speculate = true, .drafts = 15, .context_copies = copies, .lone_copies = copies, .batch_rows = width }, width, 15);
        defer cfg.deinit(a);
        var fake: lanes.fake.Fake = .{ .gpa = a };
        defer fake.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var e = lanes.Engine.init(a, &cfg, fake.backend(), clock.clock());
        defer e.deinit();
        var s = try lanes.Stream.init(a, .{ .id = "hybrid", .prompt = &.{ 1, 2, 3 }, .max_new = 64 });
        defer s.deinit(a);
        const tree = try a.alloc(u32, 15);
        const parents = try a.alloc(i32, 15);
        for (tree, parents, 0..) |*t, *p, i| {
            t.* = @intCast(i + 1);
            p.* = -1;
        }
        s.next = .{ .count = 15, .tokens = tree, .parents = parents };
        var copy: [48]u32 = undefined;
        for (&copy, 0..) |*t, i| t.* = @intCast(31 + i);
        const p = try lanes.windows.plan(&e, &s, if (copies) .{ .tokens = &copy } else null);
        try std.testing.expectEqual(@as(usize, if (copies and width > 16) width - 1 else 15), p.count());
        try std.testing.expectEqual(copies and width > 16, p.context_copy);
        try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, p.tokens[0..3]);
        if (p.context_copy) try std.testing.expectEqualSlices(u32, copy[0 .. width - 16], p.tokens[15..]);
    };
}

fn copiedReply(drafts: bool, copies: bool, width: u32) !struct { tokens: []u32, widest: usize } {
    var cfg = try lanes.Config.init(a, .{ .exact_width = width, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 15, .draft_probabilities = true, .context_copies = copies, .lone_copies = copies, .batch_rows = width }, width, 15);
    defer cfg.deinit(a);
    cfg.timed = false;
    var pattern: [64]u32 = undefined;
    for (&pattern, 0..) |*t, i| t.* = @intCast(11 + i);
    var target: lanes.fake.Fake = .{ .gpa = a, .cycle_after = 0, .pattern = &pattern };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var prompt: [128]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = pattern[i % pattern.len];
    var lookup = try lanes.SuffixLookup.init(a, .{ .min_match = 3 });
    defer lookup.deinit();
    var s = try lanes.Stream.init(a, .{ .id = "copy-equality", .prompt = &prompt, .max_new = 96, .drafts = drafts, .proposer = if (copies) lookup.proposer() else null });
    defer s.deinit(a);
    try e.addStream(&s);
    var widest: usize = 0;
    while (e.activeCount() > 0) {
        try e.step();
        widest = @max(widest, target.last_rows);
    }
    return .{ .tokens = try a.dupe(u32, s.emitted()), .widest = widest };
}

test "qwen27 window: real 32 and 64-row head-plus-copy rounds keep draft-false target tokens" {
    const plain = try copiedReply(false, false, 64);
    defer a.free(plain.tokens);
    const tree = try copiedReply(true, false, 64);
    defer a.free(tree.tokens);
    try std.testing.expect(tree.widest <= 16);
    try std.testing.expectEqualSlices(u32, plain.tokens, tree.tokens);
    for ([_]u32{ 32, 64 }) |width| {
        const both = try copiedReply(true, true, width);
        defer a.free(both.tokens);
        try std.testing.expectEqual(@as(usize, width), both.widest);
        try std.testing.expectEqualSlices(u32, plain.tokens, both.tokens);
    }
}

fn atWidth(width: u32, seed: ?u64) !struct { tokens: []u32, widest: usize } {
    var cfg = try lanes.Config.init(a, .{ .exact_width = width, .mtp = width > 1, .speculate = true, .speculate_early = false, .drafts = width - 1, .draft_prior = &.{1}, .draft_probabilities = true, .batch_rows = width }, width, width - 1);
    defer cfg.deinit(a);
    cfg.timed = false;
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var s = try lanes.Stream.init(a, .{ .id = "all-widths", .prompt = &.{ 7, 13, 29 }, .max_new = 80, .drafts = width > 1, .sampling = if (seed) |v| .{ .seed = v, .temperature = 0.8 } else null });
    defer s.deinit(a);
    try e.addStream(&s);
    var widest: usize = 0;
    while (e.activeCount() > 0) {
        try e.step();
        widest = @max(widest, target.last_rows);
    }
    return .{ .tokens = try a.dupe(u32, s.emitted()), .widest = widest };
}

test "qwen27 window: every actual width 1 through 64 keeps serial greedy and keyed target tokens" {
    for ([_]?u64{ null, 17 }) |seed| {
        const plain = try atWidth(1, seed);
        defer a.free(plain.tokens);
        for (2..65) |width| {
            const got = try atWidth(@intCast(width), seed);
            defer a.free(got.tokens);
            try std.testing.expectEqual(width, got.widest);
            try std.testing.expectEqualSlices(u32, plain.tokens, got.tokens);
        }
    }
}
