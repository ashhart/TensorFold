//! Production mask policies at the fake device boundary, with the real lane scheduler and grammar rollback.
const std = @import("std");
const lanes = @import("lanes");
const masks = @import("qwen27_masks");
const fake = @import("qwen27_mask_fake");
const draws = @import("qwen_draw");
const a = std.testing.allocator;

test "qwen27 masks: greedy and keyed draws honor allowed bits and reject malformed masks" {
    var logits: [97]u16 = @splat(0);
    logits[96] = 0x4180;
    logits[7] = 0x4000;
    logits[9] = 0x4000;
    var bits: [4]u32 = @splat(0);
    bits[0] = (1 << 7) | (1 << 9);
    try std.testing.expectEqual(@as(u32, 7), try masks.draw(draws.draw, a, &logits, @as(?lanes.Sampling, null), 0, &bits));
    for ([_]u64{ 1, 17, 129 }) |position| {
        const sampling: ?lanes.Sampling = .{ .seed = 123, .temperature = 0.8, .top_k = 1, .top_p = 0.9 };
        const got = try masks.draw(draws.draw, a, &logits, sampling, position, &bits);
        try std.testing.expect(lanes.grammar.allows(&bits, got));
        try std.testing.expectEqual(try draws.draw(a, &logits, sampling, position, &bits), got);
    }
    try std.testing.expectError(error.InvalidQwenMasks, masks.draw(draws.draw, a, &logits, @as(?lanes.Sampling, null), 0, bits[0..3]));
    @memset(&bits, 0);
    try std.testing.expectError(error.NoAllowedQwenToken, masks.draw(draws.draw, a, &logits, @as(?lanes.Sampling, null), 0, &bits));
}

test "qwen27 masks: proposal masks follow each branch and restore the grammar before and after the first commit" {
    for ([_]bool{ false, true }) |committed| {
        var seq = lanes.grammar.Sequence{ .gpa = a, .want = &.{ 9, 10, 11 }, .stop = 96 };
        defer seq.log.deinit(a);
        var c = lanes.grammar.Constraint.init(seq.rules(), 4, null);
        if (committed) try std.testing.expect(try c.follow(9));
        var t = fake.Tree{ .gpa = a, .tokens = try a.dupe(u32, &.{ 10, 96, 11, 42, 96, 10 }), .parents = try a.dupe(i32, &.{ -1, -1, 0, 0, 2, 4 }), .scores = try a.dupe(f64, &.{ -0.4, -9, -0.7, -8, -1.2, -2 }) };
        defer t.deinit();
        try masks.prune(a, @as(?*lanes.grammar.Constraint, &c), if (committed) null else 9, &t);
        try std.testing.expectEqualSlices(u32, &.{ 10, 11, 96 }, t.tokens);
        try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1 }, t.parents);
        try std.testing.expectEqualSlices(f64, &.{ -0.4, -0.7, -1.2 }, t.scores);
        try std.testing.expectEqual(@as(usize, @intFromBool(committed)), seq.at);
        try std.testing.expectEqual(@as(usize, @intFromBool(committed)), seq.log.items.len);
    }
}

fn run(want: []const u32, depth: u32, seed: ?u64) ![]u32 {
    var seq = lanes.grammar.Sequence{ .gpa = a, .want = want, .stop = 96 };
    defer seq.log.deinit(a);
    var c = lanes.grammar.Constraint.init(seq.rules(), 4, null);
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .mtp = true, .speculate = true, .drafts = depth, .hidden_rows = true, .streams_exact = true, .batch_rows = 8, .max_streams = 2, .draft_streams = true }, 8, 7);
    defer cfg.deinit(a);
    var target = fake.Adapter{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var s = try lanes.Stream.init(a, .{ .id = "qwen27", .prompt = &.{ 3, 1, 4, 1 }, .max_new = 512, .sampling = if (seed) |v| .{ .seed = v, .temperature = 0.8 } else null, .grammar = &c, .drafts = depth > 0, .eos = &.{96} });
    defer s.deinit(a);
    try engine.addStream(&s);
    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expect(seq.at > want.len and target.first_masks == 1 and target.row_masks > 0);
    if (depth > 0) try std.testing.expect(s.drafted > 0);
    if (depth > 1) try std.testing.expect(target.filtered > 0);
    return a.dupe(u32, s.emitted());
}

test "qwen27 masks: a required tool call through the lane fake is identical plain and drafted" {
    const text = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"a,b:c {d:e}\"}}</tool_call>";
    var ids: [text.len]u32 = undefined;
    for (&ids, text) |*id, byte| id.* = byte - 32;
    for ([_]?u64{ null, 17, 91 }) |seed| {
        const plain = try run(&ids, 0, seed);
        defer a.free(plain);
        try std.testing.expectEqualSlices(u32, &ids, plain[0..ids.len]);
        try std.testing.expectEqual(@as(usize, ids.len + 1), plain.len);
        try std.testing.expectEqual(@as(u32, 96), plain[ids.len]);
        for ([_]u32{ 1, 4, 7 }) |depth| {
            const drafted = try run(&ids, depth, seed);
            defer a.free(drafted);
            try std.testing.expectEqualSlices(u32, plain, drafted);
        }
    }
}

fn pruneFailures(allocator: std.mem.Allocator) !void {
    var seq = lanes.grammar.Sequence{ .gpa = allocator, .want = &.{ 9, 10, 11 }, .stop = 96 };
    defer seq.log.deinit(allocator);
    var c = lanes.grammar.Constraint.init(seq.rules(), 4, null);
    var t = fake.Tree{ .gpa = a, .tokens = try a.dupe(u32, &.{ 10, 42, 11 }), .parents = try a.dupe(i32, &.{ -1, -1, 0 }), .scores = try a.dupe(f64, &.{ 0, 0, 0 }) };
    defer t.deinit();
    masks.prune(allocator, @as(?*lanes.grammar.Constraint, &c), 9, &t) catch |err| {
        try std.testing.expectEqual(@as(usize, 0), seq.at);
        try std.testing.expectEqual(@as(usize, 0), seq.log.items.len);
        try std.testing.expect(c.active);
        try std.testing.expectEqualSlices(u32, &.{ 10, 42, 11 }, t.tokens);
        return err;
    };
    try std.testing.expectEqual(@as(usize, 0), seq.at);
    try std.testing.expectEqualSlices(u32, &.{ 10, 11 }, t.tokens);
}

test "qwen27 masks: proposal allocation failures restore the matcher and keep the original owned tree" {
    try std.testing.checkAllAllocationFailures(a, pruneFailures, .{});
}

test "qwen27 masks: thinking activation and EOS prune only the constrained suffix of each draft branch" {
    var seq = lanes.grammar.Sequence{ .gpa = a, .want = &.{10}, .stop = 96 };
    defer seq.log.deinit(a);
    var c = lanes.grammar.Constraint.init(seq.rules(), 4, 90);
    var t = fake.Tree{ .gpa = a, .tokens = try a.dupe(u32, &.{ 42, 90, 10, 96, 42, 12 }), .parents = try a.dupe(i32, &.{ -1, 0, 1, 2, 3, 1 }), .scores = try a.dupe(f64, &.{ 0, 0, 0, 0, 0, 0 }) };
    defer t.deinit();
    try masks.prune(a, @as(?*lanes.grammar.Constraint, &c), @as(?u32, 7), &t);
    try std.testing.expectEqualSlices(u32, &.{ 42, 90, 10, 96 }, t.tokens);
    try std.testing.expect(!c.active and seq.at == 0);
}

test "qwen27 masks: shared windows keep each stream's mask and keyed positions independent" {
    const wants = [_][]const u32{ &.{ 9, 10, 11, 12, 13, 14, 15 }, &.{ 21, 22, 23, 24, 25 } };
    for ([_]u32{ 0, 1, 4, 7 }) |depth| {
        var seqs = [_]lanes.grammar.Sequence{
            .{ .gpa = a, .want = wants[0], .stop = 96 }, .{ .gpa = a, .want = wants[1], .stop = 96 },
        };
        defer for (&seqs) |*s| s.log.deinit(a);
        var constraints: [2]lanes.grammar.Constraint = undefined;
        var streams: [2]lanes.Stream = undefined;
        var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .mtp = true, .speculate = true, .drafts = depth, .hidden_rows = true, .streams_exact = true, .batch_rows = 16, .max_streams = 2, .draft_streams = true }, 8, 7);
        defer cfg.deinit(a);
        var target = fake.Adapter{ .gpa = a };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var engine = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
        defer engine.deinit();
        for (&streams, &constraints, &seqs, 0..) |*s, *c, *seq, i| {
            c.* = .init(seq.rules(), 4, null);
            s.* = try lanes.Stream.init(a, .{ .id = "shared", .prompt = if (i == 0) &.{ 1, 2 } else &.{ 5, 6, 7 }, .max_new = 100, .sampling = .{ .seed = 17 + i, .temperature = 0.7 }, .grammar = c, .drafts = depth > 0, .eos = &.{96} });
            try engine.addStream(s);
        }
        defer for (&streams) |*s| s.deinit(a);
        while (engine.activeCount() > 0) try engine.step();
        for (&streams, wants, 0..) |*s, want, i| {
            const solo = try run(want, depth, 17 + i);
            defer a.free(solo);
            try std.testing.expectEqualSlices(u32, solo, s.emitted());
        }
        try std.testing.expectEqual(@as(usize, 2), target.first_masks);
    }
}
