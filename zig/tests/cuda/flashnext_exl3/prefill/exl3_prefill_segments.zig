//! exl3-prefill mixed|resume: deep and shallow segments side by side, and a state kept at any row then resumed.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("../../check.zig");
const flash = @import("flashnext");
const engine = @import("exl3_prefill_engine.zig");
const Gpu = check.Gpu;
const Eng = engine.Eng;

const xpre = flash.exl3.prefill;
const prompt = flash.prompt;
const vocab = prompt.vocabulary;
const max_segments = 256;
const n_tokens = 24;

/// Shallow segments (under the deep path's 512 rows): a history mark's 7 rows, the longest that stays shallow, another.
const shorts = [_]usize{ 7, 511, 100 };

/// One prompt's cuts: the rows of each forward, and whether it took the deep path.
const Plan = struct {
    len: [max_segments]usize = undefined,
    deep: [max_segments]bool = undefined,
    n: usize = 0,

    fn add(p: *Plan, rows: usize) !void {
        if (p.n == max_segments) return error.TooManySegments;
        p.len[p.n] = rows;
        p.n += 1;
    }

    /// Segments of `step` rows over [from, to): the server's steps (`stop = at + step`), the last one the remainder.
    fn steps(from: usize, to: usize, step: usize) !Plan {
        var p: Plan = .{};
        var at = from;
        while (at < to) : (at += p.len[p.n - 1]) try p.add(@min(step, to - at));
        return p;
    }

    /// Deep segments of `step` rows, a shallow one between, from either; `mark`: the last 7 rows cut.
    fn mixed(n: usize, step: usize, shallow_first: bool, mark: bool) !Plan {
        var p: Plan = .{};
        const end = if (mark) n - shorts[0] else n;
        var at: usize = 0;
        var is_deep = !shallow_first;
        var k: usize = 0;
        while (at < end) : (is_deep = !is_deep) {
            const want = if (is_deep) step else shorts[k % shorts.len];
            k += @intFromBool(!is_deep);
            try p.add(@min(want, end - at));
            at += p.len[p.n - 1];
        }
        if (mark) try p.add(shorts[0]);
        return p;
    }

    fn count(p: *const Plan, deep: bool) usize {
        var c: usize = 0;
        for (p.deep[0..p.n]) |d| c += @intFromBool(d == deep);
        return c;
    }

    fn show(p: *const Plan) void {
        for (p.len[0..p.n], p.deep[0..p.n]) |rows, d| std.debug.print(" {d}{s}", .{ rows, if (d) "d" else "s" });
    }
};

/// What one prompt gave: the last row's logits and the greedy tokens after it.
const Run = struct {
    logits: []u16,
    tokens: [n_tokens]u32 = undefined,

    fn init(a: std.mem.Allocator) !Run {
        return .{ .logits = try a.alloc(u16, vocab) };
    }

    fn take(r: *Run, g: *Eng) !void {
        @memcpy(std.mem.sliceAsBytes(r.logits), g.seq.logits);
        try engine.greedy(g, &r.tokens);
    }

    fn verdict(r: Run, want: Run) struct { logits: bool, tokens: bool } {
        return .{
            .logits = std.mem.eql(u8, std.mem.sliceAsBytes(r.logits), std.mem.sliceAsBytes(want.logits)),
            .tokens = std.mem.eql(u32, &r.tokens, &want.tokens),
        };
    }

    /// Print how this run compares with `want`; true when both the logits bytes and the tokens are equal.
    fn against(r: Run, want: Run, name: []const u8) bool {
        const v = r.verdict(want);
        std.debug.print("; vs {s}: logits {s}, {d} tokens {s}", .{ name, if (v.logits) "bit-equal" else "DIFFERENT", n_tokens, if (v.tokens) "identical" else "DIFFERENT" });
        return v.logits and v.tokens;
    }
};

/// The options the next forwards read: the deep path on or off, its chunk, the cache's grid.
fn setOptions(g: *Eng, deep_on: bool, step: usize, grid: bool) void {
    var o = xpre.optionsFromEnv();
    o.enabled = deep_on;
    o.rows = step;
    o.snap_grid = grid;
    g.e.exl3_prefill.options = o;
}

/// `ids` from `at.*` through the engine, one forward a segment as the server steps a prompt; the last keeps its logits.
fn feed(g: *Eng, ids: []const u32, at: *usize, plan: *Plan) !void {
    for (plan.len[0..plan.n], plan.deep[0..plan.n]) |rows, *deep| {
        const stop = at.* + rows;
        if (stop > ids.len) return error.UnexpectedTensor;
        const before = g.e.exl3_prefill.stats.chunks;
        try g.e.forward(flash.Tri, g.tri(), &g.seq, ids[at.*..stop], .{ .logits = if (stop == ids.len) .last else .none, .prompt = true });
        deep.* = g.e.exl3_prefill.stats.chunks != before;
        at.* = stop;
    }
}

/// A fresh sequence through `plan`'s forwards, then the greedy tokens.
fn fresh(g: *Eng, ids: []const u32, plan: *Plan, out: *Run) !void {
    try g.reset();
    var at: usize = 0;
    try feed(g, ids, &at, plan);
    try out.take(g);
}

/// Every prompt row alone through a decode round, forced to the prompt's tokens (what the server's rounds run).
fn serial(g: *Eng, ids: []const u32, out: *Run) !void {
    try g.reset();
    var buf = try cuda.DeviceBuffer.alloc(g.gpu.d, vocab * 2);
    defer buf.free();
    for (ids) |id| {
        var tok = id;
        const q = [_]*prompt.Seq{&g.seq};
        try g.e.round(flash.Tri, g.tri(), &q, (&tok)[0..1], buf.ptr);
    }
    try out.take(g);
}

fn parseList(text: []const u8, out: []isize) !usize {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |tok| : (n += 1) {
        if (n == out.len) return error.TooManyValues;
        out[n] = try std.fmt.parseInt(isize, tok, 10);
    }
    return n;
}

fn parseLines(rest: []const [:0]const u8, at: usize) !?usize {
    if (rest.len <= at or std.mem.eql(u8, rest[at], "-")) return null;
    return try std.fmt.parseInt(usize, rest[at], 10);
}

/// Shallow segments (under 512 rows) beside deep ones at each step size equal the batched pass and serial decode.
pub fn mixedRun(gpu: Gpu, dir: []const u8, rest: []const [:0]const u8) !void {
    const a = gpu.gpa;
    if (rest.len < 3) return error.MissingArgument;
    const all_ids = try engine.readIds(gpu, rest[0]);
    defer a.free(all_ids);
    const n = try std.fmt.parseInt(usize, rest[1], 10);
    if (n <= shorts[0] or n > all_ids.len) return error.NotEnoughIds;
    var sizes: [8]isize = undefined;
    const n_sizes = try parseList(rest[2], &sizes);
    var max_step: usize = 0;
    for (sizes[0..n_sizes]) |s| {
        if (s < 512) return error.StepBelowDeepRows;
        max_step = @max(max_step, @as(usize, @intCast(s)));
    }
    const g = try Eng.init(gpu, dir, try parseLines(rest, 3), n + 64, @min(max_step, (n + 255) / 256 * 256));
    defer g.deinit();
    const ids = all_ids[0..n];
    var batched = try Run.init(a);
    defer a.free(batched.logits);
    var forced = try Run.init(a);
    defer a.free(forced.logits);
    var got = try Run.init(a);
    defer a.free(got.logits);
    setOptions(g, false, 0, true);
    var whole = try Plan.steps(0, n, n);
    try fresh(g, ids, &whole, &batched);
    try serial(g, ids, &forced);
    std.debug.print("MIXED reference, {d} tokens: forced serial decode of every row", .{n});
    var all_equal = forced.against(batched, "the batched pass");
    std.debug.print("\n", .{});
    var cases: usize = 0;
    const orders = [_]struct { shallow_first: bool, mark: bool }{ .{ .shallow_first = false, .mark = false }, .{ .shallow_first = true, .mark = false }, .{ .shallow_first = false, .mark = true } };
    for (sizes[0..n_sizes]) |size| for (orders) |order| {
        const step: usize = @intCast(size);
        setOptions(g, true, step, true);
        var plan = try Plan.mixed(n, step, order.shallow_first, order.mark);
        try fresh(g, ids, &plan, &got);
        std.debug.print("MIXED {d} tokens, steps of {d} rows, cuts", .{ n, step });
        plan.show();
        const eq_batched = got.against(batched, "the batched pass");
        const eq_serial = got.against(forced, "forced serial decode");
        std.debug.print("; {d} deep, {d} shallow segments\n", .{ plan.count(true), plan.count(false) });
        try check.expect(plan.count(true) > 0 and plan.count(false) > 0, "the cut at steps of {d} rows mixes deep and shallow segments", .{step});
        all_equal = all_equal and eq_batched and eq_serial;
        cases += 1;
    };
    g.e.exl3_prefill.options = null;
    try check.expect(all_equal, "mixed deep and shallow segments equal the batched pass and forced serial decode", .{});
    check.pass("{d} mixed cuts at {d} step sizes equal the batched pass and forced serial decode in logits bytes and tokens", .{ cases, n_sizes });
}

/// A state kept at any row (snap grid off), cold-continued or resumed, equals a fresh prompt; a cut < 0 is rows left.
pub fn resumeRun(gpu: Gpu, dir: []const u8, rest: []const [:0]const u8) !void {
    const a = gpu.gpa;
    if (rest.len < 4) return error.MissingArgument;
    const all_ids = try engine.readIds(gpu, rest[0]);
    defer a.free(all_ids);
    const n = try std.fmt.parseInt(usize, rest[1], 10);
    if (n < 2 or n > all_ids.len) return error.NotEnoughIds;
    var sizes: [8]isize = undefined;
    const n_sizes = try parseList(rest[2], &sizes);
    var cuts: [16]isize = undefined;
    const n_cuts = try parseList(rest[3], &cuts);
    var max_step: usize = 0;
    for (sizes[0..n_sizes]) |s| {
        if (s <= 0) return error.BadStep;
        max_step = @max(max_step, @as(usize, @intCast(s)));
    }
    const g = try Eng.init(gpu, dir, try parseLines(rest, 4), n + 64, @min(max_step, (n + 255) / 256 * 256));
    defer g.deinit();
    const ids = all_ids[0..n];
    var want = try Run.init(a);
    defer a.free(want.logits);
    var cold = try Run.init(a);
    defer a.free(cold.logits);
    var resumed = try Run.init(a);
    defer a.free(resumed.logits);
    var all_equal = true;
    var cases: usize = 0;
    for (sizes[0..n_sizes]) |size| {
        const step: usize = @intCast(size);
        setOptions(g, true, step, true);
        const grid_on = xpre.snapGrid(g.e, g.e.chunk);
        setOptions(g, true, step, false);
        try check.expect(xpre.snapGrid(g.e, g.e.chunk) == 0, "the engine's snap grid is off with the option off", .{});
        std.debug.print("RESUME steps of {d} rows: snap grid {d} by default, 0 (any row) now\n", .{ step, grid_on });
        var whole = try Plan.steps(0, n, step);
        try fresh(g, ids, &whole, &want);
        for (cuts[0..n_cuts]) |c| {
            const mark: usize = if (c < 0) n - @as(usize, @intCast(-c)) else @intCast(c);
            if (mark == 0 or mark >= n) return error.CutOutsidePrompt;
            var head = try Plan.steps(0, mark, step);
            var tail = try Plan.steps(mark, n, step);
            // the cold pass cuts at the mark (the history mark's last chunk), keeps the state, and goes on
            try g.reset();
            var at: usize = 0;
            try feed(g, ids, &at, &head);
            const saved = try flash.snapshot.save(a, g.gpu.d, g.stream, &g.seq, null);
            defer flash.snapshot.drop(a, g.stream, saved);
            try feed(g, ids, &at, &tail);
            try cold.take(g);
            // the resumed pass: nothing of the prompt left on the device but the kept state
            try g.reset();
            try flash.snapshot.restore(g.stream, &g.seq, null, saved);
            try check.expect(g.seq.fed == mark, "the kept state resumes at row {d}", .{mark});
            at = mark;
            try feed(g, ids, &at, &tail);
            try resumed.take(g);
            std.debug.print("RESUME {d} tokens, steps of {d} rows, state at row {d} ({d} past a chunk start, {d} rows after it, last chunk {s})", .{ n, step, mark, mark % step, n - mark, if (tail.deep[tail.n - 1]) "deep" else "shallow" });
            const eq_cold = cold.against(want, "fresh (cold pass with the mark)");
            const eq_resumed = resumed.against(want, "fresh (resumed pass)");
            std.debug.print("\n", .{});
            all_equal = all_equal and eq_cold and eq_resumed;
            cases += 1;
        }
    }
    g.e.exl3_prefill.options = null;
    try check.expect(all_equal, "a state kept at any row, resumed or continued, equals a fresh prompt", .{});
    check.pass("{d} states kept at any row (snap grid off) resume to a fresh prompt's logits bytes and tokens", .{cases});
}
