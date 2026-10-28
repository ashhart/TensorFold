//! `drafter <drafter dir> <target dir> [--steps N]`: the HIP DFlash2 drafter's chains against the plain target (GPU).

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const qwen35 = @import("qwen3_5");

const Engine = qwen35.engine.Engine;
const Draft = qwen35.draft.Draft;
const Features = lanes.backend.Features;

/// Chat prompts of the kinds DFlash2's README reports (code, math, chat), thinking on as the template's default.
const texts = [_][]const u8{
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and two tests.",
    "A train leaves at 9:15 and travels 180 km at 72 km/h. At what time does it arrive? Explain step by step.",
    "Give me three tips for keeping a small vegetable garden healthy through a hot summer.",
};

fn chat(gpa: std.mem.Allocator, text: []const u8) ![]u8 {
    return std.fmt.allocPrint(gpa, "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n<think>\n", .{text});
}

/// One prompt's plain greedy continuation: `g[0]` the prefill's token, `g[i + 1]` the target's after `g[i]`.
fn decode(io: std.Io, e: *Engine, caches: anytype, prompt: []const u32, g: []u32) !f64 {
    g[0] = try e.prefill(caches, prompt, 0, null, .{ .sampling = null, .position = prompt.len }, null);
    var total: f64 = 0;
    for (0..g.len - 1) |i| {
        const pos = prompt.len + i;
        const rows = [_]Engine.Rows{.{ .caches = caches, .pos = pos, .tokens = g[i..][0..1] }};
        var out: [1]u32 = undefined;
        const t0 = std.Io.Clock.awake.now(io);
        _ = try e.verify(&rows, &.{.{ .sampling = null, .position = pos + 1 }}, &out);
        e.keep(0, 1);
        try e.flush();
        total += ms(io, t0);
        g[i + 1] = out[0];
    }
    return total / @as(f64, @floatFromInt(g.len - 1));
}

fn ms(io: std.Io, t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
}

/// The drafted rounds a chain drafter gives over a greedy continuation: drafts, acceptance and every round's chain.
const Rounds = struct {
    rounds: usize = 0,
    first: usize = 0, // rounds whose first draft is the target's next token
    accepted: usize = 0, // drafts accepted over all rounds
    chains: std.ArrayList(u32) = .empty,
};

/// Rounds as the round loop runs them: absorb the kept rows, hold after the pending token, keep the matching prefix.
fn rounds(gpa: std.mem.Allocator, d: *Draft, s: *lanes.Stream, f0: Features, len: usize, g: []const u32) !Rounds {
    const x = d.drafter();
    const depth = d.c.block - 1;
    var r: Rounds = .{};
    errdefer r.chains.deinit(gpa);
    try x.open(s);
    defer x.release(s);
    var f = f0;
    f.offset = 0;
    f.rows = @intCast(len);
    try x.absorb(&.{.{ .stream = s, .features = f, .start = 0, .follow = g[0..1] }});
    var p = len; // the pending token g[p - len] sits at position p
    while (p - len + depth + 1 <= g.len) {
        const at = p - len;
        try x.hold(&.{.{ .stream = s, .pending = g[at], .position = p + 1, .depth = @intCast(depth) }});
        var held: [16]u32 = undefined;
        try x.held(&.{s}, &.{held[0..depth]});
        try r.chains.appendSlice(gpa, held[0..depth]);
        var k: usize = 0;
        while (k < depth and held[k] == g[at + 1 + k]) k += 1;
        r.rounds += 1;
        r.accepted += k;
        if (k > 0) r.first += 1;
        // the target keeps the accepted drafts and its bonus token: those rows are absorbed next
        f.offset = p;
        f.rows = @intCast(k + 1);
        const kept = g[at + 1 ..][0 .. k + 1];
        try x.absorb(&.{.{ .stream = s, .features = f, .start = p, .follow = kept, .prompt = false }});
        p += k + 1;
    }
    return r;
}

/// Positions and shapes through the contract: a hold behind its context and too few held drafts are refused.
fn refusals(d: *Draft, s: *lanes.Stream, f0: Features) !void {
    const x = d.drafter();
    try x.open(s);
    defer x.release(s);
    var f = f0;
    f.offset = 0;
    f.rows = 4;
    try x.absorb(&.{.{ .stream = s, .features = f, .start = 0, .follow = &.{} }});
    const behind = x.hold(&.{.{ .stream = s, .pending = 1, .position = 6, .depth = 7 }});
    if (behind != error.DraftContextBehind) return error.HoldNotRefused;
    try x.hold(&.{.{ .stream = s, .pending = 1, .position = 5, .depth = 3 }});
    var out: [7]u32 = undefined;
    if (x.held(&.{s}, &.{out[0..4]}) != error.TooFewDrafts) return error.HeldNotRefused;
    try x.held(&.{s}, &.{out[0..3]});
    for (out[0..3]) |t| if (t >= d.c.vocab) return error.DraftOutOfVocabulary;
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len != 2 and !(args.len == 4 and std.mem.eql(u8, args[2], "--steps"))) {
        std.debug.print("usage: tf-qwen35-test drafter <drafter dir> <target dir> [--steps N]\n", .{});
        return error.Usage;
    }
    const steps = if (args.len == 4) try std.fmt.parseInt(usize, args[3], 10) else 160;
    var tok = try qwen35.tokenizer.loadTokenizer(io, gpa, args[1]);
    defer tok.deinit();
    var prompts: [texts.len][]u32 = undefined;
    var longest: usize = 0;
    for (texts, &prompts) |t, *p| {
        const text = try chat(gpa, t);
        defer gpa.free(text);
        p.* = try tok.encode(gpa, text);
        longest = @max(longest, p.len);
    }
    defer for (prompts) |p| gpa.free(p);
    var target = try qwen35.Config.read(gpa, io, args[1]);
    defer target.deinit();
    const c = try qwen35.draft_config.Config.read(gpa, io, args[0], qwen35.draft_config.Target.of(target.spec));
    const policy = (try @import("group.zig").resolve("", 0)).policy;
    const e = try Engine.open(gpa, io, args[1], .{
        .capacity = longest + steps + 64,
        .prompt_rows = longest + 16,
        .streams = 1,
        .batch_rows = 16,
        .policy = policy,
    });
    defer e.deinit();
    try e.setTaps(c.tapLayers());
    const ring_rows = e.model().taps.rows;
    if (longest + steps > ring_rows) return error.RingTooShort;
    const t_load = std.Io.Clock.awake.now(io);
    const d = try Draft.init(gpa, io, e, args[0], 1);
    defer d.deinit();
    const depth: f64 = @floatFromInt(c.block - 1);
    const loaded = "PASS drafter load: {d:.1} s, a draft step {d:.2} ms (a hold of {d} drafts {d:.2} ms)\n";
    std.debug.print(loaded, .{ ms(io, t_load) / 1e3, d.step_ms, c.block - 1, d.step_ms * depth });
    var stream: lanes.Stream = undefined;
    stream.sampling = null;
    var all: Rounds = .{};
    var plain_ms: f64 = 0;
    for (prompts, 0..) |prompt, n| {
        var caches = try e.newCaches(prompt.len + steps + 8);
        defer caches.deinit(gpa);
        const g = try gpa.alloc(u32, steps);
        defer gpa.free(g);
        plain_ms += try decode(io, e, &caches, prompt, g);
        const f: Features = .{
            .buffer = caches.tap.?.base(),
            .rows = 0,
            .row_bytes = @intCast(c.n_taps * c.hidden * 2),
            .space = .device,
            .dtype = if (e.model().act == .bf16) .bf16 else .f16,
        };
        if (n == 0) {
            try refusals(d, &stream, f);
            std.debug.print("PASS drafter refusals: a hold behind its context, more drafts read than held\n", .{});
        }
        d.times = .{};
        var a = try rounds(gpa, d, &stream, f, prompt.len, g);
        defer a.chains.deinit(gpa);
        const t = d.times;
        var b = try rounds(gpa, d, &stream, f, prompt.len, g);
        defer b.chains.deinit(gpa);
        if (a.rounds != b.rounds or !std.mem.eql(u32, a.chains.items, b.chains.items)) return error.DraftsDiffer;
        const per = @as(f64, @floatFromInt(a.accepted)) / @as(f64, @floatFromInt(a.rounds));
        const fmt = "PASS drafter prompt {d} ({d} tokens, {d} steps): {d} rounds, first draft right {d}/{d}, " ++
            "{d:.2} drafts kept a round ({d:.2} tokens), identical rerun; hold {d:.2} ms, absorb {d:.2} ms\n";
        const hold_ms = @as(f64, @floatFromInt(t.hold_ns)) / 1e6 / @as(f64, @floatFromInt(t.holds));
        const absorb_ms = @as(f64, @floatFromInt(t.absorb_ns)) / 1e6 / @as(f64, @floatFromInt(t.absorbs));
        std.debug.print(fmt, .{ n, prompt.len, steps, a.rounds, a.first, a.rounds, per, per + 1, hold_ms, absorb_ms });
        all.rounds += a.rounds;
        all.first += a.first;
        all.accepted += a.accepted;
    }
    const per = @as(f64, @floatFromInt(all.accepted)) / @as(f64, @floatFromInt(all.rounds));
    const plain = plain_ms / @as(f64, @floatFromInt(prompts.len));
    const summary = "drafter acceptance over {d} rounds: first draft right {d:.3}, " ++
        "mean acceptance length {d:.2} (kept drafts + bonus); plain decode {d:.2} ms a token\n";
    const first = @as(f64, @floatFromInt(all.first)) / @as(f64, @floatFromInt(all.rounds));
    std.debug.print(summary, .{ all.rounds, first, per + 1, plain });
    // a drafter blind to the target's taps keeps almost no drafts; a working one keeps several
    if (per + 1 < min_length) return error.LowAcceptance;
}

/// The mean acceptance length under which the check fails.
const min_length = 2.0;
