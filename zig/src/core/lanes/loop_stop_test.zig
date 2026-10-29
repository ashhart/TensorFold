//! The loop guard's stop policy on the fake target: the reply ends at the fire, drafted, shared and in a grammar.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const grammar = @import("grammar.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const et = @import("engine_test.zig");

const gpa = std.testing.allocator;
const t = std.testing;

const Case = struct { drafts: bool = true, stop: bool = true, grammar: bool = false };
const Out = struct { emitted: []u32, reason: sm.Reason, period: ?u32, steps: usize, grammar_done: bool };

/// Every case on one engine over the looping fake target, stepped until all finish: each one's tokens and end.
fn run(cases: []const Case) ![]Out {
    var cfg = try et.model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = 70, .probe_prompt = et.p2.len };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var streams: [4]sm.Stream = undefined;
    var proposers: [4]SuffixLookup = undefined;
    var classes: [4]et.Classes = undefined;
    var constraints: [4]grammar.Constraint = undefined;
    const n = cases.len;
    for (cases, streams[0..n], proposers[0..n], classes[0..n], constraints[0..n]) |c, *s, *p, *g, *k| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        g.* = .{ .after = 91, .free = true };
        k.* = .init(g.rules(), (fake.vocab + 31) / 32, 91);
        s.* = try sm.Stream.init(gpa, .{ .id = "stop", .prompt = &et.p2, .max_new = 400, .eos = &.{96}, .drafts = c.drafts, .proposer = p.proposer(), .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true, .loop_stop = c.stop, .grammar = if (c.grammar) k else null });
    }
    defer for (streams[0..n], proposers[0..n]) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams[0..n]) |*s| try engine.addStream(s);
    var done: [4]usize = @splat(0);
    var steps: usize = 0;
    while (engine.activeCount() > 0 and steps < 5000) {
        try engine.step();
        steps += 1;
        for (streams[0..n], done[0..n]) |*s, *d| if (s.finished and d.* == 0) {
            d.* = steps;
        };
    }
    const out = try gpa.alloc(Out, n);
    for (out, streams[0..n], done[0..n], constraints[0..n], cases) |*o, *s, d, *k, c| {
        try t.expect(s.finished);
        o.* = .{ .emitted = try gpa.dupe(u32, s.emitted()), .reason = s.reason, .period = s.loop_period, .steps = d, .grammar_done = c.grammar and k.finished() };
    }
    return out;
}

fn free(outs: []Out) void {
    for (outs) |o| gpa.free(o.emitted);
    gpa.free(outs);
}

/// Where the guarded reply on this target forces its close (engine_test.zig's loop guard test).
const fire_end = 329;

test "with the stop policy a firing commit finishes the stream there and forces no close" {
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);
    for (0..sm.loop_warm_in) |i| try tokens.append(gpa, @intCast(1000 + i));
    for (0..257) |_| try tokens.append(gpa, 7);
    var s = try sm.Stream.init(gpa, .{ .id = "loop", .prompt = &.{1}, .max_new = 400, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true, .loop_stop = true });
    defer s.deinit(gpa);
    try t.expectEqual(@as(usize, 321), try s.commit(gpa, tokens.items, &.{}));
    try t.expect(s.finished and s.reason == .stop and s.loop_period.? == 1 and s.force.items.len == 0);
    try t.expectEqual(@as(usize, 0), try s.commit(gpa, &.{8}, &.{}));
}

test "the stop policy ends the reply at the fire: the guarded reply's prefix, drafted equal to plain" {
    const off = try run(&.{.{ .stop = false }});
    defer free(off);
    const on = try run(&.{.{}});
    defer free(on);
    const plain = try run(&.{.{ .drafts = false }});
    defer free(plain);
    try t.expectEqualSlices(u32, &.{ 90, 91, 92 }, off[0].emitted[fire_end..][0..3]); // the default answers after its close
    try t.expectEqualSlices(u32, off[0].emitted[0..fire_end], on[0].emitted);
    try t.expectEqualSlices(u32, on[0].emitted, plain[0].emitted);
    for ([_]Out{ on[0], plain[0] }) |o| {
        try t.expect(o.reason == .stop and o.period.? == 3);
        try t.expect(std.mem.indexOfScalar(u32, o.emitted, 90) == null and std.mem.indexOfScalar(u32, o.emitted, 91) == null);
    }
    try t.expect(on[0].steps < off[0].steps); // its rounds end at the fire: the engine holds the stream no longer
}

test "with the stop policy shared rounds commit what each stream commits alone" {
    const cases = [_]Case{ .{}, .{ .stop = false }, .{ .drafts = false } };
    const together = try run(&cases);
    defer free(together);
    for (cases, together) |c, got| {
        const alone = try run(&.{c});
        defer free(alone);
        try t.expectEqualSlices(u32, alone[0].emitted, got.emitted);
        try t.expectEqual(alone[0].period, got.period);
        try t.expectEqual(alone[0].reason, got.reason);
    }
}

test "with the stop policy a structured reply ends in its reasoning, before its grammar starts" {
    const on = try run(&.{.{ .grammar = true }});
    defer free(on);
    const plain = try run(&.{.{ .grammar = true, .drafts = false }});
    defer free(plain);
    const off = try run(&.{.{ .grammar = true, .stop = false }});
    defer free(off);
    try t.expectEqualSlices(u32, on[0].emitted, plain[0].emitted);
    try t.expectEqualSlices(u32, off[0].emitted[0..fire_end], on[0].emitted);
    try t.expect(on[0].reason == .stop and on[0].period.? == 3 and !on[0].grammar_done);
    try t.expect(std.mem.indexOfScalar(u32, on[0].emitted, 91) == null);
    try t.expect(off[0].grammar_done and std.mem.indexOfScalar(u32, off[0].emitted, 91) != null);
}
