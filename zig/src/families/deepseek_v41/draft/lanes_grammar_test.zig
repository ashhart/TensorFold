//! Structured output through DeepSeek's lane backend on the CPU twin: grammar replies equal their masked serial decode.
const std = @import("std");
const lanes = @import("lanes");
const twin = @import("twin.zig");
const iface = @import("iface.zig");
const ln = @import("lanes.zig");
const servedCosts = @import("lanes_test.zig").servedCosts;

const gpa = std.testing.allocator;
const grammar = lanes.grammar;

/// A test grammar over the twin's 48 tokens: `length` tokens t with (t + step + r) % 3 == 0, then token 47.
const Steps = struct {
    length: u32,
    r: u32,
    at: u32 = 0,
    done: bool = false,
    const stop: u32 = 47;
    const words = 2;

    fn rules(g: *Steps) grammar.Rules {
        return .{ .ptr = g, .vtable = &.{ .accept = accept, .rollback = rollback, .terminated = terminated, .fill = fill, .drafts = drafts, .free = free } };
    }

    fn cast(ptr: *anyopaque) *Steps {
        return @ptrCast(@alignCast(ptr));
    }

    fn allows(g: *const Steps, t: u32) bool {
        if (g.done) return false;
        if (g.at == g.length) return t == stop;
        return t != stop and (t + g.at + g.r) % 3 == 0;
    }

    fn accept(ptr: *anyopaque, t: u32) anyerror!bool {
        const g = cast(ptr);
        if (!g.allows(t)) return false;
        if (g.at == g.length) g.done = true else g.at += 1;
        return true;
    }

    fn rollback(ptr: *anyopaque, n: usize) anyerror!void {
        const g = cast(ptr);
        for (0..n) |_| {
            if (g.done) g.done = false else g.at -= 1;
        }
    }

    fn terminated(ptr: *anyopaque) bool {
        return cast(ptr).done;
    }

    fn fill(ptr: *anyopaque, out: []u32) anyerror!void {
        const g = cast(ptr);
        @memset(out, 0);
        for (0..out.len * 32) |t| if (g.allows(@intCast(t))) {
            out[t / 32] |= @as(u32, 1) << @intCast(t % 32);
        };
    }

    fn drafts(_: *anyopaque, _: []u32) anyerror!usize {
        return 0;
    }

    fn free(_: *anyopaque) void {}
};

const Case = struct { prompt: []const u32, length: u32, r: u32 = 0, sampling: ?lanes.Sampling = null, drafts: bool = true };

/// The twin alone, one row a step, each row's draw among the tokens the case's grammar allows after the reply so far.
fn serial(cs: Case) ![]u32 {
    const t = try twin.Twin.init(gpa, .{}, 1);
    defer t.deinit();
    const target = t.target();
    var g: Steps = .{ .length = cs.length, .r = cs.r };
    var bits: [Steps.words]u32 = undefined;
    try Steps.fill(&g, &bits);
    try target.firstMask(0, &bits);
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    var tok = try target.prefill(0, cs.prompt, cs.sampling, cs.prompt.len);
    while (true) {
        try out.append(gpa, tok);
        try std.testing.expect(try Steps.accept(&g, tok));
        if (g.done) break;
        try Steps.fill(&g, &bits);
        const at: u64 = cs.prompt.len + out.items.len - 1;
        var c: [1]u32 = undefined;
        var cs_out = [_][]u32{&c};
        try target.window(&.{.{ .slot = 0, .start = at, .tokens = &.{tok}, .parents = null, .draws = &.{at + 1}, .sampling = cs.sampling, .masks = &bits }}, &cs_out);
        tok = c[0];
    }
    target.release(0);
    return out.toOwnedSlice(gpa);
}

const Run = struct {
    out: [][]u32,
    drafted: u64,

    fn deinit(r: Run) void {
        for (r.out) |o| gpa.free(o);
        gpa.free(r.out);
    }
};

/// The cases through the lane engine over DeepSeek's backend on the twin, `slots` streams at once.
fn run(cases: []const Case, slots: u32) !Run {
    const t = try twin.Twin.init(gpa, .{}, slots + 1);
    defer t.deinit();
    const d = try twin.Drafter.init(gpa, t, slots + 1);
    defer d.deinit();
    var c = try servedCosts(gpa);
    defer c.deinit(gpa);
    const x = try ln.Lanes.init(gpa, t.target(), d.pass(), c, .{ .shape = d.shape, .slots = slots });
    defer x.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cfg = try lanes.Config.init(gpa, try x.model(arena.allocator(), 64), 16, 15);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpa, &cfg, x.backend(), clock.clock());
    defer e.deinit();
    x.attach(&e);
    const gs = try gpa.alloc(Steps, cases.len);
    defer gpa.free(gs);
    const cons = try gpa.alloc(grammar.Constraint, cases.len);
    defer gpa.free(cons);
    const streams = try gpa.alloc(lanes.Stream, cases.len);
    defer gpa.free(streams);
    for (cases, gs, cons, streams) |cs, *g, *k, *s| {
        g.* = .{ .length = cs.length, .r = cs.r };
        k.* = .init(g.rules(), Steps.words, null);
        s.* = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = cs.prompt, .max_new = cs.length + 8, .sampling = cs.sampling, .drafts = cs.drafts, .grammar = k });
    }
    defer for (streams) |*s| s.deinit(gpa);
    for (streams) |*s| try e.addStream(s);
    while (e.activeCount() > 0) try e.step();
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| {
        try std.testing.expectEqualStrings("", s.problem);
        try std.testing.expect(s.reason == .stop);
        o.* = try gpa.dupe(u32, s.emitted());
    }
    return .{ .out = out, .drafted = e.drafted };
}

const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4 };
const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5 };
const p3 = [_]u32{ 10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30 };
const warm: lanes.Sampling = .{ .seed = 5, .temperature = 1.0, .top_k = 20, .top_p = 0.95, .min_p = 0.02 };

test "a structured reply equals its masked serial decode, drafted or not, alone or beside three others" {
    for ([_]?lanes.Sampling{ null, warm }) |smp| {
        const cases = [_]Case{ .{ .prompt = &p1, .length = 40, .sampling = smp }, .{ .prompt = &p2, .length = 33, .r = 1, .sampling = smp }, .{ .prompt = &p3, .length = 52, .r = 2, .sampling = smp }, .{ .prompt = &p2, .length = 24, .r = 2, .sampling = smp } };
        var plain: [cases.len]Case = cases;
        for (&plain) |*cs| cs.drafts = false;
        const together = try run(&cases, 4);
        defer together.deinit();
        const undrafted = try run(&plain, 4);
        defer undrafted.deinit();
        try std.testing.expect(together.drafted > 0 and undrafted.drafted == 0);
        for (cases, together.out, undrafted.out) |cs, got, flat| {
            const want = try serial(cs);
            defer gpa.free(want);
            try std.testing.expectEqual(@as(usize, cs.length + 1), want.len);
            try std.testing.expectEqualSlices(u32, want, got);
            try std.testing.expectEqualSlices(u32, want, flat);
            const solo = try run(&.{cs}, 1);
            defer solo.deinit();
            try std.testing.expectEqualSlices(u32, want, solo.out[0]);
        }
    }
}

test "the grammar changes the reply: its first and later draws follow the bits" {
    const cs: Case = .{ .prompt = &p1, .length = 30, .r = 1 };
    const got = try run(&.{cs}, 1);
    defer got.deinit();
    const t = try twin.Twin.init(gpa, .{}, 1);
    defer t.deinit();
    const unmasked = try t.serial(0, cs.prompt, cs.length + 1, null, null);
    defer gpa.free(unmasked);
    try std.testing.expect(!std.mem.eql(u32, unmasked, got.out[0]));
    for (got.out[0][0..cs.length], 0..) |tok, k| try std.testing.expect((tok + k + cs.r) % 3 == 0 and tok != Steps.stop);
    try std.testing.expectEqual(Steps.stop, got.out[0][cs.length]);
}

/// The twin with prompts in pieces as the GPU target runs them; `tail` leaves the last row to the first round.
const Joining = struct {
    t: *twin.Twin,
    pass: iface.Pass,

    fn target(j: *Joining) iface.Target {
        return .{ .ptr = j, .vtable = &.{ .prefill = prefill, .window = window, .keep = keep, .taps = taps, .release = release, .first_mask = firstMask, .begin = begin, .piece = piece, .finish = finish, .tail = tail } };
    }

    fn inner(ptr: *anyopaque) iface.Target {
        const j: *Joining = @ptrCast(@alignCast(ptr));
        return j.t.target();
    }

    fn prefill(ptr: *anyopaque, slot: u32, ids: []const u32, smp: ?lanes.Sampling, draw: u64) anyerror!u32 {
        return inner(ptr).prefill(slot, ids, smp, draw);
    }

    fn window(ptr: *anyopaque, segments: []const iface.Segment, choices: [][]u32) anyerror!void {
        return inner(ptr).window(segments, choices);
    }

    fn keep(ptr: *anyopaque, slot: u32, path: []const u32) anyerror!void {
        return inner(ptr).keep(slot, path);
    }

    fn taps(ptr: *anyopaque, slot: u32) iface.Taps {
        return inner(ptr).taps(slot);
    }

    fn release(ptr: *anyopaque, slot: u32) void {
        inner(ptr).release(slot);
    }

    fn firstMask(ptr: *anyopaque, slot: u32, bits: []const u32) void {
        inner(ptr).firstMask(slot, bits) catch unreachable;
    }

    fn begin(_: *anyopaque, _: u32, _: []const u32) anyerror!iface.Target.Begun {
        return .{};
    }

    fn piece(_: *anyopaque, _: u32, _: []const u32, _: u64, _: u64, _: bool) anyerror!void {}

    fn finish(ptr: *anyopaque, slot: u32, ids: []const u32, smp: ?lanes.Sampling, draw: u64) anyerror!u32 {
        return inner(ptr).prefill(slot, ids, smp, draw);
    }

    fn tail(ptr: *anyopaque, slot: u32, ids: []const u32, smp: ?lanes.Sampling, _: u64) anyerror!?u32 {
        const j: *Joining = @ptrCast(@alignCast(ptr));
        const t = inner(ptr);
        try t.firstMask(slot, &.{}); // the first token comes from the round's window, its bits the lane core's
        _ = try t.prefill(slot, ids[0 .. ids.len - 1], smp, ids.len - 1);
        const tp = t.taps(slot);
        var rows: [64]u32 = undefined;
        for (rows[0..tp.rows], 0..) |*r, i| r.* = @intCast(i);
        try j.pass.ingest(slot, 0, tp, rows[0..tp.rows]);
        return null;
    }
};

/// The cases' prompts in one piece each, then joined (`together`: one draft call, TF_DSV41_JOIN_DRAFTS), then rounds.
fn runJoined(cases: []const Case, slots: u32, together: bool) !Run {
    const t = try twin.Twin.init(gpa, .{}, slots + 1);
    defer t.deinit();
    const d = try twin.Drafter.init(gpa, t, slots + 1);
    defer d.deinit();
    var j: Joining = .{ .t = t, .pass = d.pass() };
    var c = try servedCosts(gpa);
    defer c.deinit(gpa);
    const x = try ln.Lanes.init(gpa, j.target(), d.pass(), c, .{ .shape = d.shape, .slots = slots });
    defer x.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cfg = try lanes.Config.init(gpa, try x.model(arena.allocator(), 64), 16, 15);
    defer cfg.deinit(gpa);
    cfg.join_tail = true;
    cfg.join_drafts = together;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpa, &cfg, x.backend(), clock.clock());
    defer e.deinit();
    x.attach(&e);
    const gs = try gpa.alloc(Steps, cases.len);
    defer gpa.free(gs);
    const cons = try gpa.alloc(grammar.Constraint, cases.len);
    defer gpa.free(cons);
    const streams = try gpa.alloc(lanes.Stream, cases.len);
    defer gpa.free(streams);
    const ptrs = try gpa.alloc(*lanes.Stream, cases.len);
    defer gpa.free(ptrs);
    const errs = try gpa.alloc(?anyerror, cases.len);
    defer gpa.free(errs);
    for (cases, gs, cons, streams, ptrs) |cs, *g, *k, *s, *p| {
        g.* = .{ .length = cs.length, .r = cs.r };
        k.* = .init(g.rules(), Steps.words, null);
        s.* = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = cs.prompt, .max_new = cs.length + 8, .sampling = cs.sampling, .drafts = cs.drafts, .grammar = k });
        p.* = s;
    }
    defer for (streams) |*s| s.deinit(gpa);
    for (streams) |*s| {
        _ = try e.beginStream(s);
        try e.piece(s, 0, s.prompt_len - 1, false);
    }
    if (together) {
        try std.testing.expectEqual(cases.len, try e.finishStreams(ptrs, errs));
        for (errs) |x_| if (x_) |err| return err;
    } else for (ptrs) |s| try e.finishStream(s);
    while (e.activeCount() > 0) try e.step();
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| {
        try std.testing.expectEqualStrings("", s.problem);
        try std.testing.expect(s.reason == .stop);
        o.* = try gpa.dupe(u32, s.emitted());
    }
    return .{ .out = out, .drafted = e.drafted };
}

test "a joined prompt's first round walks its held drafts: replies equal their masked serial decode" {
    for ([_]?lanes.Sampling{ null, warm }) |smp| {
        const cases = [_]Case{ .{ .prompt = &p1, .length = 40, .sampling = smp }, .{ .prompt = &p2, .length = 33, .r = 1, .sampling = smp }, .{ .prompt = &p3, .length = 52, .r = 2, .sampling = smp }, .{ .prompt = &p2, .length = 24, .r = 2, .sampling = smp } };
        for ([_]bool{ true, false }) |together| {
            var wants: [cases.len][]u32 = undefined;
            for (cases, &wants) |cs, *want| want.* = try serial(cs);
            defer for (wants) |want| gpa.free(want);
            for (cases, wants) |cs, want| {
                const solo = try runJoined(&.{cs}, 1, together);
                defer solo.deinit();
                try std.testing.expect(solo.drafted > 0);
                try std.testing.expectEqualSlices(u32, want, solo.out[0]);
            }
            const four = try runJoined(&cases, 4, together);
            defer four.deinit();
            try std.testing.expect(four.drafted > 0);
            for (wants, four.out) |want, got| try std.testing.expectEqualSlices(u32, want, got);
        }
    }
}
