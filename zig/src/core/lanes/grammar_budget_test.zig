//! grammar_budget through the round loop: a test call closes inside tiny budgets, drafted == plain, shared == solo.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const Sampling = @import("sampling.zig").Sampling;
const grammar = @import("grammar.zig");
const model = @import("engine_test.zig").model;

const gpa = std.testing.allocator;
const words = (fake.vocab + 31) / 32;
const stop = 96;

/// A tool call as a DFA (opening 10..15, argument 20..79, `"` `}` `}` or `"}}`, close 84); its budget hook is exact.
const Call = struct {
    const open = 6;
    const content = 6;
    const quote = 7;
    const brace = 8;
    const braces = 9;
    const closed = 10;
    const ended = 11; // its stop token taken

    path: std.ArrayList(u8) = .empty, // the states after each taken token
    budgeted: bool = true, // false: no budget hook at all (the grammar before grammar_budget)
    consulted: usize = 0,

    fn rules(c: *Call) grammar.Rules {
        const with: *const grammar.Rules.VTable = &.{ .accept = accept, .rollback = rollback, .terminated = terminated, .fill = fill, .drafts = drafts, .free = free, .budget = budget };
        const without: *const grammar.Rules.VTable = &.{ .accept = accept, .rollback = rollback, .terminated = terminated, .fill = fill, .drafts = drafts, .free = free };
        return .{ .ptr = c, .vtable = if (c.budgeted) with else without };
    }

    fn cast(ptr: *anyopaque) *Call {
        return @ptrCast(@alignCast(ptr));
    }

    fn at(c: *const Call) u8 {
        return if (c.path.items.len == 0) 0 else c.path.items[c.path.items.len - 1];
    }

    /// The state after `t` from `s`, null when refused.
    fn step(s: u8, t: u32) ?u8 {
        if (s < open) return if (t == 10 + s) s + 1 else null;
        return switch (s) {
            content => if (t >= 20 and t < 80) content else if (t == 81) quote else if (t == 82) braces else null,
            quote => if (t == 83) brace else null,
            brace => if (t == 83) braces else null,
            braces => if (t == 84) closed else null,
            closed => if (t == 85) closed else if (t == stop) ended else null,
            else => null,
        };
    }

    /// Fewest tokens from `s` to a complete call (the stop token then allowed).
    fn need(s: u8) u32 {
        if (s < open) return open - s + 2;
        return switch (s) {
            content => 2,
            quote => 3,
            brace => 2,
            braces => 1,
            else => 0,
        };
    }

    fn accept(ptr: *anyopaque, t: u32) anyerror!bool {
        const c = cast(ptr);
        try c.path.append(gpa, step(c.at(), t) orelse return false);
        return true;
    }

    fn rollback(ptr: *anyopaque, n: usize) anyerror!void {
        const c = cast(ptr);
        c.path.shrinkRetainingCapacity(c.path.items.len - n);
    }

    fn terminated(ptr: *anyopaque) bool {
        return cast(ptr).at() == ended;
    }

    fn fill(ptr: *anyopaque, bits: []u32) anyerror!void {
        @memset(bits, 0);
        for (0..fake.vocab) |t| if (step(cast(ptr).at(), @intCast(t)) != null) {
            bits[t / 32] |= @as(u32, 1) << @intCast(t % 32);
        };
    }

    /// The forced text: each state with one way on, up to the argument or the end.
    fn drafts(ptr: *anyopaque, out: []u32) anyerror!usize {
        var s = cast(ptr).at();
        var n: usize = 0;
        while (n < out.len) : (n += 1) {
            const t: u32 = if (s < open) 10 + s else switch (s) {
                quote, brace => 83,
                braces => 84,
                else => break,
            };
            out[n] = t;
            s = step(s, t).?;
        }
        return n;
    }

    fn budget(ptr: *anyopaque, bits: []u32, left: u32) anyerror!bool {
        const c = cast(ptr);
        if (bits.len == 0 or left == 0) return true;
        c.consulted += 1;
        var keep: [words]u32 = undefined;
        @memcpy(&keep, bits);
        var any = grammar.allows(bits, stop);
        for (0..stop) |t| if (grammar.allows(bits, t)) {
            if (need(step(c.at(), @intCast(t)).?) > left - 1) keep[t / 32] &= ~(@as(u32, 1) << @intCast(t % 32)) else any = true;
        };
        if (any) @memcpy(bits, &keep);
        return true;
    }

    fn free(_: *anyopaque) void {}

    /// The reply's answer is a complete call: the DFA, run over it from the grammar's start, can stop.
    fn complete(reply: []const u32, after: ?u32) bool {
        const from = if (after) |a| (std.mem.indexOfScalar(u32, reply, a) orelse return false) + 1 else 0;
        var s: u8 = 0;
        for (reply[from..]) |t| s = step(s, t) orelse return false;
        return s == closed or s == ended;
    }
};

const Case = struct {
    prompt: []const u32,
    max_new: u32,
    sampling: ?Sampling = null,
    drafts: bool = true,
    budget: bool = true,
    hooked: bool = true, // the grammar has a budget hook (armed or not)
    think_budget: u32 = 0,
    answer_max: u32 = 0,
};

const Run = struct { tokens: [][]u32, drafted: u64, consulted: usize };

/// Every case's reply, admitted together and stepped until done.
fn run(cases: []const Case) !Run {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const n = cases.len;
    const streams = try gpa.alloc(sm.Stream, n);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, n);
    defer gpa.free(proposers);
    const calls = try gpa.alloc(Call, n);
    defer gpa.free(calls);
    const constraints = try gpa.alloc(grammar.Constraint, n);
    defer gpa.free(constraints);
    for (cases, streams, proposers, calls, constraints) |c, *s, *p, *g, *k| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        g.* = .{ .budgeted = c.hooked };
        k.* = .init(g.rules(), words, if (c.think_budget > 0) 91 else null);
        if (c.budget) try k.arm();
        s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{stop}, .sampling = c.sampling, .drafts = c.drafts, .proposer = p.proposer(), .think_budget = c.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .answer_max = c.answer_max, .grammar = k });
    }
    defer for (streams, proposers, calls) |*s, *p, *g| {
        s.deinit(gpa);
        p.deinit();
        g.path.deinit(gpa);
    };
    for (streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    var out: Run = .{ .tokens = try gpa.alloc([]u32, n), .drafted = 0, .consulted = 0 };
    for (out.tokens, streams, calls) |*o, *s, g| {
        o.* = try gpa.dupe(u32, s.emitted());
        out.drafted += s.drafted;
        out.consulted += g.consulted;
    }
    return out;
}

fn release(r: Run) void {
    for (r.tokens) |t| gpa.free(t);
    gpa.free(r.tokens);
}

const prompts = [_][]const u32{ &.{ 3, 1, 4, 1, 5, 9, 2, 6 }, &.{ 2, 7, 1, 8, 2, 8, 1, 8, 2 }, &.{ 1, 6, 1, 8, 0, 3, 3 } };
const sampled: Sampling = .{ .seed = 9, .temperature = 0.8, .top_k = 0, .top_p = 0.95 };

test "grammar budget: inside a tiny budget the call always closes, drafted or not, and without it the reply is cut" {
    var cut: usize = 0;
    for (prompts) |prompt| for ([_]?Sampling{ null, sampled }) |sampling| for ([_]u32{ 8, 9, 10, 12, 16, 23 }) |max_new| {
        for ([_]bool{ false, true }) |drafts| for ([_]bool{ false, true }) |budget| {
            const r = try run(&.{.{ .prompt = prompt, .max_new = max_new, .sampling = sampling, .drafts = drafts, .budget = budget }});
            defer release(r);
            if (budget) try std.testing.expect(Call.complete(r.tokens[0], null)) else cut += @intFromBool(!Call.complete(r.tokens[0], null));
        };
    };
    try std.testing.expect(cut > 0); // the budget closes the calls the plain grammar leaves cut
}

test "grammar budget: with it on, drafted equals plain and shared rounds equal solo ones" {
    for ([_]?Sampling{ null, sampled }) |sampling| for ([_]u32{ 9, 11, 14, 30 }) |max_new| {
        var solo: [prompts.len][]u32 = undefined;
        defer for (solo) |s| gpa.free(s);
        var drafted: u64 = 0;
        for (prompts, &solo) |prompt, *s| {
            const plain = try run(&.{.{ .prompt = prompt, .max_new = max_new, .sampling = sampling, .drafts = false }});
            defer release(plain);
            const with = try run(&.{.{ .prompt = prompt, .max_new = max_new, .sampling = sampling }});
            defer release(with);
            try std.testing.expectEqualSlices(u32, plain.tokens[0], with.tokens[0]);
            drafted += with.drafted;
            s.* = try gpa.dupe(u32, with.tokens[0]);
        }
        try std.testing.expect(drafted > 0);
        var cases: [prompts.len]Case = undefined;
        for (&cases, prompts) |*c, prompt| c.* = .{ .prompt = prompt, .max_new = max_new, .sampling = sampling };
        const shared = try run(&cases);
        defer release(shared);
        for (solo, shared.tokens) |s, got| try std.testing.expectEqualSlices(u32, s, got);
    };
}

test "grammar budget: with thinking on, the answer limit binds from the think end and drafted still equals plain" {
    for ([_]u32{ 4, 7 }) |think| for ([_]u32{ 8, 9, 12 }) |answer| {
        const prompt = prompts[0];
        const plain = try run(&.{.{ .prompt = prompt, .max_new = 60, .drafts = false, .think_budget = think, .answer_max = answer }});
        defer release(plain);
        const drafted = try run(&.{.{ .prompt = prompt, .max_new = 60, .think_budget = think, .answer_max = answer }});
        defer release(drafted);
        try std.testing.expectEqualSlices(u32, plain.tokens[0], drafted.tokens[0]);
        try std.testing.expect(Call.complete(plain.tokens[0], 91));
    };
}

test "grammar budget: off never consults the hook, and the reply is the grammar's without one" {
    for (prompts) |prompt| for ([_]bool{ false, true }) |drafts| {
        const before = try run(&.{.{ .prompt = prompt, .max_new = 14, .drafts = drafts, .budget = false, .hooked = false }});
        defer release(before);
        const off = try run(&.{.{ .prompt = prompt, .max_new = 14, .drafts = drafts, .budget = false }});
        defer release(off);
        try std.testing.expectEqualSlices(u32, before.tokens[0], off.tokens[0]);
        try std.testing.expectEqual(@as(usize, 0), off.consulted);
        const on = try run(&.{.{ .prompt = prompt, .max_new = 14, .drafts = drafts }});
        defer release(on);
        try std.testing.expect(on.consulted > 0);
    };
}

/// A fresh constraint after `pending` and `path`, with thinking on and the answer limit: its next bits (null: none).
fn plainBits(a: std.mem.Allocator, g: *Call, pending: u32, path: []const u32, answer: u32) !?[]u32 {
    var k: grammar.Constraint = .init(g.rules(), words, 91);
    try k.arm();
    k.budget = .{ .max_new = 50, .answer_max = answer, .think_end = 91 };
    _ = try k.follow(pending);
    for (path) |t| try std.testing.expect(try k.follow(t));
    return k.next(a);
}

test "grammar budget: a window crossing the think end masks each row as its plain step does, chain or tree" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const tokens = [_]u32{ 5, 91, 10, 11, 12, 13, 14, 15, 20, 82, 81 };
    const chain = [_]i32{ -1, 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    const tree = [_]i32{ -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8 };
    for ([_][]const i32{ &chain, &tree }) |parents| for ([_]u32{ 9, 10, 30 }) |answer| {
        var g: Call = .{};
        defer g.path.deinit(gpa);
        var k: grammar.Constraint = .init(g.rules(), words, 91);
        try k.arm();
        k.budget = .{ .max_new = 50, .answer_max = answer, .think_end = 91 };
        _ = try k.follow(tokens[0]);
        const w = try k.window(a, tokens[0..parents.len], parents, false, parents.len);
        try std.testing.expectEqual(parents.len, w.rows.len);
        var narrowed = false;
        for (w.rows, 0..) |r, i| {
            var path: std.ArrayList(u32) = .empty;
            var at: i32 = @intCast(r);
            while (at > 0) : (at = parents[@intCast(at)]) try path.insert(a, 0, tokens[@intCast(at)]);
            var h: Call = .{};
            defer h.path.deinit(gpa);
            const row = w.masks[i * words ..][0..words];
            if (try plainBits(a, &h, tokens[0], path.items, answer)) |bits| {
                try std.testing.expectEqualSlices(u32, bits, row);
                var raw: [words]u32 = undefined;
                try Call.fill(&h, &raw);
                narrowed = narrowed or !std.mem.eql(u32, &raw, row);
            } else try std.testing.expect(std.mem.allEqual(u32, row, std.math.maxInt(u32)));
        }
        try std.testing.expectEqual(answer < 30, narrowed); // a tight answer limit binds inside the window
    };
}

test "grammar budget: a grammar without the hook refuses to arm" {
    var c: Call = .{ .budgeted = false };
    var k: grammar.Constraint = .init(c.rules(), words, null);
    try std.testing.expectError(error.GrammarBudgetUnsupported, k.arm());
    try std.testing.expect(k.budget == null);
}
