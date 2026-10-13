//! cuda_slide's recapture while either sequence is bound: each keeps its own sampling rule, when a capture fails too.

const std = @import("std");
const lanes = @import("lanes");
const nemotron = @import("nemotron");
const check = @import("check.zig");

const Sampling = lanes.Sampling;
const Seq = nemotron.state.Seq;

// unlike each other and unlike the captures' placeholder (seed 0, temperature 1)
const rule_a: Sampling = .{ .seed = 7, .temperature = 0.6, .top_k = 40, .top_p = 0.9 };
const rule_b: Sampling = .{ .seed = 1 << 40 | 3, .temperature = 1.3, .top_k = 10, .top_p = 0.8, .min_p = 0.05 };

/// The own sequence's rule, the other's, and which of them is bound when the graphs are captured again.
const Case = struct { name: []const u8, own: ?Sampling, other: ?Sampling, other_bound: bool };

const cases = [_]Case{
    .{ .name = "own greedy, other B, other bound", .own = null, .other = rule_b, .other_bound = true },
    .{ .name = "own A, other B, other bound", .own = rule_a, .other = rule_b, .other_bound = true },
    .{ .name = "own A, other greedy, other bound", .own = rule_a, .other = null, .other_bound = true },
    .{ .name = "own A, other B, own bound", .own = rule_a, .other = rule_b, .other_bound = false },
};

/// MODEL with both draw modes' windows, as cuda_native.open leaves them, and another sequence: each case, then failing.
pub fn run(gpu: check.Gpu, model: []const u8) !void {
    const e = try nemotron.Engine.init(gpu.gpa, gpu.io, gpu.ctx, model, null, .{ .context = 8192, .mtp = false, .graphs = true, .sampling = null, .segments = 1 });
    defer e.deinit();
    try e.captureWindows();
    try e.setSampling(.{ .seed = 0, .temperature = 1.0 });
    try e.captureWindows();
    try e.setSampling(null);
    try check.expect(held(e, 0) == nemotron.state.max_rows and held(e, 1) == nemotron.state.max_rows, "windows held: {d} greedy, {d} sampled", .{ held(e, 0), held(e, 1) });
    const seq = try e.newSeq();
    defer e.freeSeq(seq);
    var bad: usize = 0;
    for (cases) |c| bad += try tally(e, seq, c, false);
    const draw = e.k.draws.draw_fn;
    defer e.k.draws.draw_fn = draw;
    e.k.draws.draw_fn.handle = null; // the sampled set's first capture fails at its draw, after its placeholder is set
    for (cases) |c| bad += try tally(e, seq, c, true);
    e.k.draws.draw_fn = draw;
    bad += try tally(e, seq, cases[0], false); // the captures run whole again once the draws are back
    try check.expect(bad == 0, "recapture: {d} of {d} runs left a sequence another rule", .{ bad, 2 * cases.len + 1 });
    check.pass("recapture: {d} runs, {d} cases captured and failing in the sampled set, each sequence's rule kept", .{ 2 * cases.len + 1, cases.len });
}

/// One run of case `c`: 1 when a check failed (its line printed), 0 when every check passed.
fn tally(e: *nemotron.Engine, seq: *Seq, c: Case, fails: bool) !usize {
    once(e, seq, c, fails) catch |err| {
        if (err != error.TestFailed) return err;
        return 1;
    };
    return 0;
}

/// Case `c` set up, the graphs captured again (`fails`: the sampled set refused), then each sequence's rule read back.
fn once(e: *nemotron.Engine, seq: *Seq, c: Case, fails: bool) !void {
    e.bind(&e.own);
    try e.setSampling(c.own);
    e.bind(seq);
    try e.setSampling(c.other);
    const bound = if (c.other_bound) seq else &e.own;
    const left = if (c.other_bound) &e.own else seq;
    e.bind(bound);
    const how: []const u8 = if (fails) ", the sampled set refused" else "";
    const before = e.stats.captures;
    var failed: ?anyerror = null;
    nemotron.slide.recapture(e) catch |err| {
        failed = err;
    };
    const made = e.stats.captures - before;
    if (!fails) if (failed) |err| return err;
    try check.expect(failed != null or !fails, "recapture {s}{s}: the capture succeeded", .{ c.name, how });
    // a failure comes at the sampled set's first window: the serial graph and the greedy set were captured again
    const want = 1 + held(e, 0) + if (fails) 0 else held(e, 1);
    try check.expect(made == want, "recapture {s}{s}: {d} graphs captured again, expected {d}", .{ c.name, how, made, want });
    try check.expect(e.bound == bound, "recapture {s}{s}: another sequence bound after the capture", .{ c.name, how });
    try same(c, how, "the bound {s} sequence's rule", bound, e, e.sampling);
    try same(c, how, "the {s} sequence's kept rule", left, e, left.sampling);
    e.bind(left);
    try same(c, how, "the {s} sequence's rule once bound", left, e, e.sampling);
    try same(c, how, "the {s} sequence's kept rule", bound, e, bound.sampling);
    e.bind(bound);
    try same(c, how, "the {s} sequence's rule once bound again", bound, e, e.sampling);
    if (failed) |err| {
        check.pass("recapture {s}{s}: {t} after {d} graphs, both rules kept", .{ c.name, how, err, made });
    } else check.pass("recapture {s}: {d} graphs, both rules kept", .{ c.name, made });
}

/// `got` against the rule case `c` gave sequence `s`, the own one or the other.
fn same(c: Case, how: []const u8, comptime what: []const u8, s: *const Seq, e: *const nemotron.Engine, got: ?Sampling) !void {
    const own = s == &e.own;
    const want = if (own) c.own else c.other;
    try check.expect(std.meta.eql(got, want), "recapture {s}{s}: " ++ what ++ " is {any}, expected {any}", .{ c.name, how, if (own) "own" else "other", got, want });
}

/// The window graphs of draw mode `m` the engine holds (0 greedy, 1 sampled).
fn held(e: *const nemotron.Engine, m: usize) usize {
    var n: usize = 0;
    for (e.windows[m]) |g| n += @intFromBool(g != null);
    return n;
}
