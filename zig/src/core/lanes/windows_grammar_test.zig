//! Structured output on a planned window: a tree shaped as a chain grows by the grammar's own drafts as a chain does.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const grammar = @import("grammar.zig");
const win = @import("windows.zig");
const et = @import("engine_test.zig");

const gpa = std.testing.allocator;

/// A test grammar: the `want` tokens in order, then 96; its own drafts are what is left of `want`.
const Spell = struct {
    want: []const u32,
    at: usize = 0,
    done: bool = false,

    fn rules(s: *Spell) grammar.Rules {
        return .{ .ptr = s, .vtable = &.{ .accept = accept, .rollback = rollback, .terminated = terminated, .fill = fill, .drafts = drafts, .free = free } };
    }

    fn cast(ptr: *anyopaque) *Spell {
        return @ptrCast(@alignCast(ptr));
    }

    fn allows(s: *const Spell, t: u32) bool {
        if (s.done) return false;
        return if (s.at == s.want.len) t == 96 else t == s.want[s.at];
    }

    fn accept(ptr: *anyopaque, t: u32) anyerror!bool {
        const s = cast(ptr);
        if (!s.allows(t)) return false;
        if (s.at == s.want.len) s.done = true else s.at += 1;
        return true;
    }

    fn rollback(ptr: *anyopaque, n: usize) anyerror!void {
        const s = cast(ptr);
        for (0..n) |_| {
            if (s.done) s.done = false else s.at -= 1;
        }
    }

    fn terminated(ptr: *anyopaque) bool {
        return cast(ptr).done;
    }

    fn fill(ptr: *anyopaque, words: []u32) anyerror!void {
        const s = cast(ptr);
        @memset(words, 0);
        for (0..fake.vocab) |t| if (s.allows(@intCast(t))) {
            words[t / 32] |= @as(u32, 1) << @intCast(t % 32);
        };
    }

    fn drafts(ptr: *anyopaque, out: []u32) anyerror!usize {
        const s = cast(ptr);
        if (s.done) return 0;
        const n = @min(out.len, s.want.len - s.at);
        @memcpy(out[0..n], s.want[s.at..][0..n]);
        return n;
    }

    fn free(_: *anyopaque) void {}
};

test "a tree shaped as a chain takes the grammar's own drafts as chain rows, and its window builds" {
    var cfg = try et.model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const prompt = [_]u32{ 3, 1, 4, 1, 5 };
    // a one-node tree, then a two-node chain whose second node the grammar refuses
    for ([_][]const u32{ &.{5}, &.{ 5, 4 } }, [_][]const i32{ &.{-1}, &.{ -1, 0 } }) |tokens, parents| {
        var g: Spell = .{ .want = &.{ 5, 7, 9, 11 } };
        var k: grammar.Constraint = .init(g.rules(), (fake.vocab + 31) / 32, null);
        var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &prompt, .max_new = 8, .eos = &.{96}, .drafts = true, .grammar = &k });
        defer s.deinit(gpa);
        s.pending = 3;
        var p: win.Plan = .{ .stream = &s, .position = prompt.len, .kind = .fill, .tokens = tokens, .parents = parents };
        try win.constrain(&engine, &p);
        try std.testing.expectEqualSlices(u32, &.{ 5, 7, 9, 11 }, p.tokens);
        try std.testing.expect(p.parents == null);
        try std.testing.expectEqual(@as(u32, 2), p.grammar_from);
        try std.testing.expectEqual(@as(usize, 5 * k.words), p.masks.len);
        const w = try win.build(&engine, p, false);
        try std.testing.expectEqualSlices(u64, &.{ 6, 7, 8, 9, 10 }, w.positions);
        try std.testing.expect(w.parents == null);
        try std.testing.expectEqual(@as(usize, 0), g.at);
    }
}

fn sharing(s: *Engine, plans: []win.Plan) !void {
    if (@hasDecl(win, "constrainShared")) return win.constrainShared(s, plans);
    for (plans) |*p| try win.constrain(s, p);
}

test "shared grammar: an uneven spare row rotates across three and eight streams" {
    for ([_]usize{ 3, 8 }) |count| {
        var cfg = try et.model();
        defer cfg.deinit(gpa);
        cfg.batch_rows = @intCast(2 * count + 1);
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer engine.deinit();
        engine.alone = false;
        var spells: [8]Spell = @splat(.{ .want = &.{ 5, 7, 9, 11, 13, 15 } });
        var constraints: [8]grammar.Constraint = undefined;
        var streams: [8]sm.Stream = undefined;
        for (spells[0..count], constraints[0..count], streams[0..count]) |*spell, *constraint, *stream| {
            constraint.* = .init(spell.rules(), (fake.vocab + 31) / 32, null);
            stream.* = try sm.Stream.init(gpa, .{ .id = "shared", .prompt = &.{3}, .max_new = 8, .eos = &.{96}, .drafts = true, .grammar = constraint });
            stream.pending = 3;
        }
        defer for (streams[0..count]) |*stream| stream.deinit(gpa);
        for (0..count) |turn| {
            engine.shared_rounds = turn;
            var plans: [8]win.Plan = undefined;
            for (plans[0..count], streams[0..count]) |*plan, *stream| plan.* = .{ .stream = stream, .position = 1, .kind = .none };
            try sharing(&engine, plans[0..count]);
            var total: usize = 0;
            for (plans[0..count], 0..) |plan, i| {
                const rows: usize = if (i == turn) 3 else 2;
                try std.testing.expect(plan.stream == &streams[i]);
                try std.testing.expectEqual(rows, 1 + plan.count());
                try std.testing.expectEqual(rows * constraints[i].words, plan.masks.len);
                try std.testing.expectEqual(@as(usize, 0), spells[i].at);
                const window = try win.build(&engine, plan, false);
                try std.testing.expectEqual(rows, window.positions.len);
                for (window.positions, 0..) |position, j| try std.testing.expectEqual(@as(u64, 2 + j), position);
                total += rows;
            }
            try std.testing.expectEqual(@as(usize, cfg.batch_rows), total);
        }
    }
}

test "shared grammar: short requests forced windows and family width cap spare growth" {
    var cfg = try et.model();
    defer cfg.deinit(gpa);
    cfg.batch_rows = 9;
    cfg.family_width = 3;
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    engine.alone = false;
    var spells: [3]Spell = @splat(.{ .want = &.{ 5, 7, 9, 11, 13, 15 } });
    var constraints: [3]grammar.Constraint = undefined;
    var streams: [3]sm.Stream = undefined;
    for (&spells, &constraints, &streams, 0..) |*spell, *constraint, *stream, i| {
        constraint.* = .init(spell.rules(), (fake.vocab + 31) / 32, null);
        stream.* = try sm.Stream.init(gpa, .{ .id = "caps", .prompt = &.{3}, .max_new = if (i == 0) 1 else 8, .eos = &.{96}, .drafts = true, .grammar = constraint });
        stream.pending = 3;
    }
    defer for (&streams) |*stream| stream.deinit(gpa);
    var plans = [_]win.Plan{
        .{ .stream = &streams[0], .position = 1, .kind = .none },
        .{ .stream = &streams[1], .position = 1, .kind = .forced, .tokens = &.{ 5, 7 } },
        .{ .stream = &streams[2], .position = 1, .kind = .none },
    };
    try sharing(&engine, &plans);
    try std.testing.expectEqual(@as(usize, 0), plans[0].count());
    try std.testing.expectEqualSlices(u32, &.{ 5, 7 }, plans[1].tokens);
    try std.testing.expectEqual(@as(u32, 0), plans[1].grammar_from);
    try std.testing.expectEqual(@as(usize, 2), plans[2].count());
    streams[2].drafts = false;
    plans[2] = .{ .stream = &streams[2], .position = 1, .kind = .none };
    try sharing(&engine, plans[2..]);
    try std.testing.expectEqual(@as(usize, 0), plans[2].count());
}

fn sharedReply(drafted: bool, count: usize, sampled: bool) !struct { hash: [32]u8, rounds: u64 } {
    var cfg = try et.modelWith(false);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const want = [_]u32{ 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27 };
    var spells: [8]Spell = @splat(.{ .want = &want });
    var constraints: [8]grammar.Constraint = undefined;
    var streams: [8]sm.Stream = undefined;
    for (spells[0..count], constraints[0..count], streams[0..count], 0..) |*spell, *constraint, *stream, i| {
        constraint.* = .init(spell.rules(), (fake.vocab + 31) / 32, null);
        stream.* = try sm.Stream.init(gpa, .{ .id = "reply", .prompt = &.{ 3, 1, 4 }, .max_new = 16, .eos = &.{96}, .drafts = drafted, .grammar = constraint, .sampling = if (sampled) .{ .seed = 57 + i, .temperature = 0.8, .top_k = 7 } else null });
        try engine.addStream(stream);
    }
    defer for (streams[0..count]) |*stream| stream.deinit(gpa);
    var steps: usize = 0;
    while (engine.activeCount() > 0) {
        try engine.step();
        try std.testing.expect(target.last_rows <= cfg.batch_rows);
        steps += 1;
        try std.testing.expect(steps < 64);
    }
    var hash: [32]u8 = undefined;
    for (streams[0..count], 0..) |stream, i| {
        try std.testing.expectEqualSlices(u32, &want, stream.emitted()[0..want.len]);
        try std.testing.expectEqual(@as(usize, want.len + 1), stream.emitted().len);
        try std.testing.expectEqual(@as(u32, 96), stream.emitted()[want.len]);
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        for (stream.emitted()) |token| {
            var bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &bytes, token, .little);
            h.update(&bytes);
        }
        const current = h.finalResult();
        if (i == 0) hash = current else try std.testing.expectEqual(hash, current);
    }
    return .{ .hash = hash, .rounds = target.rounds };
}

test "shared grammar: greedy and seeded complete replies equal plain and solo with bounded rounds" {
    for ([_]bool{ false, true }) |sampled| {
        const solo = try sharedReply(false, 1, sampled);
        const solo_drafted = try sharedReply(true, 1, sampled);
        try std.testing.expectEqual(solo.hash, solo_drafted.hash);
        for ([_]usize{ 2, 4, 8 }) |count| {
            const plain = try sharedReply(false, count, sampled);
            const drafted = try sharedReply(true, count, sampled);
            try std.testing.expectEqual(solo.hash, plain.hash);
            try std.testing.expectEqual(plain.hash, drafted.hash);
            try std.testing.expect(drafted.rounds < plain.rounds);
        }
    }
}
