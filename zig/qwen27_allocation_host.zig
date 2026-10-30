//! Counted shared-drafter scheduling and hybrid windows through the real lane planner, without a GPU.
const std = @import("std");
const lanes = @import("lanes");
const a = std.testing.allocator;
const be = lanes.backend;

const Counted = struct {
    fake: lanes.fake.Fake = .{ .gpa = a },
    first: ?*lanes.Stream = null,
    proposals: usize = 0,
    taps: [512]u32 = undefined,
    tap_len: usize = 0,
    head: [512]u32 = undefined,
    head_len: usize = 0,
    ready: bool = false,
    zero_rounds: usize = 0,
    long_flushes: usize = 0,
    drop_kept: bool = false,
    fn absorb(c: *Counted, history: []const u32) !void {
        const common = @min(c.tap_len, history.len);
        try std.testing.expectEqualSlices(u32, c.taps[0..common], history[0..common]);
        if (history.len > c.taps.len) return error.TooManyFixtureTaps;
        @memcpy(c.taps[0..history.len], history);
        c.tap_len = history.len;
    }
    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        const c: *Counted = @fieldParentPtr("fake", f);
        try f.backend().vtable.keep(ptr, windows, paths);
        for (windows) |w| if (w.stream == c.first and !c.drop_kept) try c.absorb(f.lanes.get(w.stream).?.history.items);
    }
    fn prepare(c: *Counted, r: be.DraftRequest) !void {
        const l = c.fake.lanes.get(r.stream).?;
        var history: [512]u32 = undefined;
        const base = if (r.rows != null) l.base else l.history.items.len;
        @memcpy(history[0..base], l.history.items[0..base]);
        if (!c.ready) {
            @memcpy(c.head[0..base], history[0..base]);
            c.head_len = base;
            c.ready = true;
        }
        var n = base;
        if (r.rows) |path| for (path) |row| {
            history[n] = l.rows.items[row];
            n += 1;
        };
        if (c.tap_len == 0) try c.absorb(history[0..base]);
        if (r.depth == 0) {
            c.zero_rounds += 1;
            return;
        }
        if (c.tap_len != base and c.tap_len != n) return error.MissingCommittedTaps;
        try c.absorb(history[0..n]);
        const old = c.head_len;
        if (@import("src/families/qwen27/dflash/catchup.zig").fresh(c.ready, 0, c.head_len, c.tap_len, 256)) |part| {
            try std.testing.expect(part.rows <= 128);
            try std.testing.expectEqual(c.head_len, part.first);
            @memcpy(c.head[c.head_len..c.tap_len], c.taps[c.head_len..c.tap_len]);
        } else {
            if (c.ready and c.tap_len - old > 128) c.long_flushes += 1;
            @memcpy(c.head[0..c.tap_len], c.taps[0..c.tap_len]);
        }
        c.ready = true;
        c.head_len = c.tap_len;
        try std.testing.expectEqualSlices(u32, c.taps[0..c.tap_len], c.head[0..c.head_len]);
    }
    fn compare(c: *Counted, r: be.DraftRequest) !void {
        if (r.depth == 0) return;
        var oracle = try lanes.Stream.init(a, .{ .id = "never-skipped", .prompt = c.head[0..c.head_len], .max_new = 32, .sampling = r.stream.sampling });
        defer oracle.deinit(a);
        var f: lanes.fake.Fake = .{ .gpa = a };
        defer f.deinit();
        try f.backend().vtable.prefill(&f, &oracle);
        const pending = if (r.first) |feed| switch (feed) {
            .value => |v| v,
            .handle => |h| try c.fake.backend().vtable.read(&c.fake, h),
        } else r.follow[r.follow.len - 1];
        var q = r;
        q.stream = &oracle;
        q.rows = null;
        q.first = .{ .value = pending };
        q.follow = &.{};
        try f.backend().vtable.draft(&f, &.{q});
        try std.testing.expectEqualSlices(u32, f.lanes.get(&oracle).?.held.items, c.fake.lanes.get(r.stream).?.held.items);
    }
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
            if (q.stream == c.first) try c.prepare(q);
            if (q.depth > 0) {
                c.proposals += 1;
            }
            try f.backend().vtable.draft(ptr, &.{q});
            if (q.stream == c.first) try c.compare(q);
        }
    }
};

fn shared(policy: bool, others: u32, seed: ?u64, drop_kept: bool) !struct { wasted: usize, resumed: bool, skipped: usize, flushes: usize } {
    var costs: [32]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, r| c.* = .{ .width = @intCast(r), .ms = if (r <= 16) 1 else 1000 };
    var cfg = try lanes.Config.init(a, .{ .exact_width = 32, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 15, .draft_probabilities = true, .draft_streams = true, .allocate_shared_drafts = policy, .window_costs = &costs, .hidden_rows = true, .batch_rows = 32, .max_streams = 16, .lone_copies = false }, 32, 15);
    defer cfg.deinit(a);
    cfg.timed = false;
    var counted: Counted = .{ .drop_kept = drop_kept };
    defer counted.fake.deinit();
    var table = counted.fake.backend().vtable.*;
    table.draft = Counted.draft;
    table.keep = Counted.keep;
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
        s.* = try lanes.Stream.init(a, .{ .id = "counted", .prompt = &.{ 7, 13, 29 }, .max_new = if (i == 0) (if (others > 6) others + 40 else 24) else others, .sampling = if (seed) |v| .{ .seed = v, .temperature = 0.8 } else null });
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
    for (&ss) |*s| {
        const plain = try serialReply(s.max_new, seed);
        defer a.free(plain);
        try std.testing.expectEqualSlices(u32, plain, s.emitted());
    }
    return .{ .wasted = wasted, .resumed = resumed, .skipped = counted.zero_rounds, .flushes = counted.long_flushes };
}

test "qwen27 window: zero allocated draft rows skip counted proposal passes and resume when rows return" {
    const old = try shared(false, 6, null, false);
    try std.testing.expect(old.wasted > 0);
    const got = try shared(true, 6, null, false);
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

test "qwen27 window: a copy round clears head chances before later shared pricing" {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 4, .drafts = 3, .allocate_shared_drafts = true, .batch_rows = 4 }, 4, 3);
    defer cfg.deinit(a);
    var fake: lanes.fake.Fake = .{ .gpa = a };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, fake.backend(), clock.clock());
    defer e.deinit();
    var s = try lanes.Stream.init(a, .{ .id = "copied", .prompt = &.{1}, .max_new = 8 });
    defer s.deinit(a);
    try s.draft_chances.appendSlice(a, &.{ 0.9, 0.7 });
    var plans = [_]lanes.windows.Plan{.{ .stream = &s, .position = 1, .kind = .copy, .tokens = &.{ 2, 3 } }};
    try lanes.windows.allocate(&e, &plans);
    try std.testing.expectEqual(@as(usize, 0), s.draft_chances.items.len);
}

fn serialReply(max_new: u32, seed: ?u64) ![]u32 {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 1, .batch_rows = 1 }, 1, 0);
    defer cfg.deinit(a);
    var fake: lanes.fake.Fake = .{ .gpa = a };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, fake.backend(), clock.clock());
    defer e.deinit();
    var s = try lanes.Stream.init(a, .{ .id = "counted", .prompt = &.{ 7, 13, 29 }, .max_new = max_new, .drafts = false, .sampling = if (seed) |v| .{ .seed = v, .temperature = 0.8 } else null });
    defer s.deinit(a);
    try e.addStream(&s);
    while (e.activeCount() > 0) try e.step();
    return a.dupe(u32, s.emitted());
}

test "qwen27 window: long zero-grant tap backlog flushes and resumes the never-skipped proposals" {
    const got = try shared(true, 200, null, false);
    try std.testing.expect(got.skipped > 128);
    try std.testing.expect(got.flushes > 0);
    try std.testing.expect(got.resumed);
    try std.testing.expectError(error.MissingCommittedTaps, shared(true, 200, null, true));
    const sampled = try shared(true, 6, 17, false);
    try std.testing.expect(sampled.resumed);
}
