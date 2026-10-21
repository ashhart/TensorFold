//! Real lane and external-drafter controls for copy precedence and transition back to neural drafting.
const std = @import("std");
const cfg = @import("config.zig");
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const fake = @import("fake.zig");
const win = @import("windows.zig");
const fixtures = @import("drafted_test.zig");
const Drafted = @import("drafted.zig").Drafted;
const Suffix = @import("proposer.zig").SuffixLookup;
const a = std.testing.allocator;

fn model() cfg.Model {
    var m: cfg.Model = .{ .exact_width = 16, .batch_rows = 32, .hidden_rows = true, .context_copies = true, .copy_match = 3 };
    m.root_copies = true;
    return m;
}

test "root copy: context continuation takes precedence over an already held neural tree" {
    var conf = try cfg.Config.init(a, model(), 16, 15);
    defer conf.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var core = Engine.init(a, &conf, target.backend(), clock.clock());
    defer core.deinit();
    core.alone = true;
    var stream = try Stream.init(a, .{ .id = "copy", .prompt = &.{ 1, 2, 3 }, .max_new = 32 });
    defer stream.deinit(a);
    stream.pending = 4;
    stream.next = .{ .count = 2, .tokens = try a.dupe(u32, &.{ 77, 88 }), .parents = try a.dupe(i32, &.{ -1, -1 }) };
    const p = try win.plan(&core, &stream, .{ .tokens = &.{ 5, 6, 7 } });
    try std.testing.expectEqual(win.Kind.copy, p.kind);
    try std.testing.expectEqualSlices(u32, &.{ 5, 6, 7 }, p.tokens);
    try std.testing.expect(p.parents == null and p.held == 0);
}

test "root copy: neural holds stop while real committed rows are absorbed and resume on rejection" {
    var prompt: [120]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(11 + i % 3);
    var target: fake.Fake = .{ .gpa = a, .probe_prompt = prompt.len, .cycle_after = 0 };
    defer target.deinit();
    var head: fixtures.FakeDrafter = .{};
    defer head.deinit();
    var strict: fixtures.Strict = .{ .inner = target.backend() };
    defer strict.deinit();
    var backend = strict.backend();
    backend.first_before_draft = true;
    const Stale = struct {
        fn tree(_: *anyopaque, _: *Stream, gpa: std.mem.Allocator) !?@import("stream.zig").Held {
            return .{ .count = 1, .tokens = try gpa.dupe(u32, &.{96}) };
        }
    };
    var drafter = head.drafter();
    var table = drafter.vtable.*;
    table.tree = Stale.tree;
    drafter.vtable = &table;
    var wrapped = try Drafted.init(a, backend, drafter);
    defer wrapped.deinit();
    var conf = try cfg.Config.init(a, wrapped.facts(model()), 16, 15);
    defer conf.deinit(a);
    var clock: fake.FixedClock = .{};
    var core = Engine.init(a, &conf, wrapped.backend(), clock.clock());
    defer core.deinit();
    var suffix = try Suffix.init(a, .{ .min_match = 3 });
    defer suffix.deinit();
    var stream = try Stream.init(a, .{ .id = "copy", .prompt = &prompt, .max_new = 80, .proposer = suffix.proposer() });
    defer stream.deinit(a);
    try @import("open.zig").add(&core, &stream);
    try std.testing.expectEqual(@as(u32, 0), head.holds);
    try std.testing.expectEqual(@as(u32, 0), stream.next.?.count);
    for (0..2) |_| try core.step();
    try std.testing.expectEqual(stream.cache_len - prompt.len, head.live_rows);
    try std.testing.expect(head.live_rows > 0);
    try std.testing.expectEqual(@as(u32, 0), head.holds);
    for (stream.emitted(), 0..) |t, i| try std.testing.expectEqual(@as(u32, @intCast(11 + i % 3)), t);
    target.pattern = &.{ 14, 15, 16 };
    const before = stream.emitted().len;
    try core.step();
    try std.testing.expect(head.holds > 0);
    try std.testing.expectEqual(before + 1, stream.emitted().len);
    while (!stream.finished) {
        try core.step();
        try std.testing.expectEqual(stream.cache_len - prompt.len, head.live_rows);
    }
    for (stream.emitted()[before..], before..) |t, i| try std.testing.expectEqual(@as(u32, @intCast(14 + i % 3)), t);
    try std.testing.expectEqual(@as(u32, 0), strict.refused);
}

fn free(out: *fixtures.Out) void {
    for (out.tokens) |t| a.free(t);
    a.free(out.tokens);
    out.head.deinit();
}

test "root copy: seeded, restored, plain and shared external-drafter paths keep the same target tokens" {
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 3, 1, 4, 1 };
    for ([_]bool{ false, true }) |sampled| {
        const sampling: ?@import("sampling.zig").Sampling = if (sampled) .{ .seed = 19, .temperature = 0.7 } else null;
        var plain = try fixtures.run(&.{.{ .prompt = &prompt, .max_new = 64, .sampling = sampling, .drafts = false }}, .plain, false);
        defer free(&plain);
        for ([_]u32{ 0, 8 }) |restored| for ([_]bool{ false, true }) |batched| {
            var out = try fixtures.run(&.{
                .{ .prompt = &prompt, .max_new = 64, .sampling = sampling, .root_copies = true, .restored = restored },
                .{ .prompt = &prompt, .max_new = 64, .sampling = sampling, .root_copies = true, .restored = restored },
            }, .wrapped, batched);
            defer free(&out);
            for (out.tokens) |ids| try std.testing.expectEqualSlices(u32, plain.tokens[0], ids);
        };
    }
}

test "root copy: a headless pipelined stream retries copies after its one-token rejection fallback" {
    var prompt: [120]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(11 + i % 3);
    var m = model();
    m.gpu_tokens = true;
    var conf = try cfg.Config.init(a, m, 16, 15);
    defer conf.deinit(a);
    var target: fake.Fake = .{ .gpa = a, .probe_prompt = prompt.len, .cycle_after = 0 };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var core = Engine.init(a, &conf, target.backend(), clock.clock());
    defer core.deinit();
    var suffix = try Suffix.init(a, .{ .min_match = 3 });
    defer suffix.deinit();
    var stream = try Stream.init(a, .{ .id = "headless", .prompt = &prompt, .max_new = 80, .proposer = suffix.proposer() });
    defer stream.deinit(a);
    try core.addStream(&stream);
    stream.root_copy_miss = true;
    try core.step();
    try std.testing.expect(!stream.root_copy_miss);
    try std.testing.expect((try win.copyProposal(&core, &stream)).tokens.len > 0);
}

test "root copy: shared-round pricing counts the head passes actually performed" {
    var prompt: [120]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(11 + i % 3);
    var costs: [32]cfg.Cost = undefined;
    for (&costs, 1..) |*c, i| c.* = .{ .width = @intCast(i), .ms = 1 + @as(f64, @floatFromInt(i)) * 0.01 };
    var m = model();
    m.mtp = true;
    m.speculate = true;
    m.speculate_early = false;
    m.drafts = 6;
    m.draft_streams = true;
    m.mtp_step_ms = 3;
    m.window_costs = costs[0..16];
    m.shared_costs = &costs;
    var conf = try cfg.Config.init(a, m, 16, 15);
    defer conf.deinit(a);
    var target: fake.Fake = .{ .gpa = a, .probe_prompt = prompt.len, .cycle_after = 0 };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var core = Engine.init(a, &conf, target.backend(), clock.clock());
    defer core.deinit();
    var suffixes: [2]Suffix = undefined;
    var streams: [2]Stream = undefined;
    for (&streams, &suffixes) |*s, *p| {
        p.* = try Suffix.init(a, .{ .min_match = 3 });
        s.* = try Stream.init(a, .{ .id = "copy", .prompt = &prompt, .max_new = 80, .proposer = p.proposer() });
        try core.addStream(s);
    }
    defer for (&streams, &suffixes) |*s, *p| {
        s.deinit(a);
        p.deinit();
    };
    try core.step();
    try std.testing.expect(core.shared_rounds > 0);
    try std.testing.expect(core.rule.greedy_overhead.items.len > 0);
    for (core.rule.greedy_overhead.items) |over| try std.testing.expectEqual(@as(f64, 0), over.head);
}
