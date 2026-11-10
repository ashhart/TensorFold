//! `drafter <drafter dir> <target dir> [--steps N]`: the HIP DFlash2 drafter's chains against the plain target (GPU).

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const qwen35 = @import("qwen3_5");

const Engine = qwen35.engine.Engine;
const Draft = qwen35.draft.Draft;
const Features = lanes.backend.Features;
const Hold = lanes.drafter.Hold;
const Absorb = lanes.drafter.Absorb;
const max_streams = qwen35.draft.max_streams;

/// Chat prompts of the kinds DFlash2's README reports (code, math, chat), thinking on as the template's default.
const texts = [_][]const u8{
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and two tests.",
    "A train leaves at 9:15 and travels 180 km at 72 km/h. At what time does it arrive? Explain step by step.",
    "Give me three tips for keeping a small vegetable garden healthy through a hot summer.",
};

fn chat(gpa: std.mem.Allocator, text: []const u8) ![]u8 {
    return std.fmt.allocPrint(gpa, "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n<think>\n", .{text});
}

/// A prompt's continuation drawn by `rule` (null: greedy): `g[0]` the prefill's token, `g[i + 1]` the one after `g[i]`.
fn decode(io: std.Io, e: *Engine, caches: anytype, prompt: []const u32, g: []u32, rule: ?lanes.Sampling) !f64 {
    g[0] = try e.prefill(caches, prompt, 0, null, .{ .sampling = rule, .position = prompt.len }, null);
    var total: f64 = 0;
    for (0..g.len - 1) |i| {
        const pos = prompt.len + i;
        const rows = [_]Engine.Rows{.{ .caches = caches, .pos = pos, .tokens = g[i..][0..1] }};
        var out: [1]u32 = undefined;
        const t0 = std.Io.Clock.awake.now(io);
        _ = try e.verify(&rows, &.{.{ .sampling = rule, .position = pos + 1 }}, &out);
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

/// The target's tap ring in `caches` as the drafter reads it; each absorb sets its first row and row count.
fn features(e: *Engine, c: qwen35.draft_config.Config, caches: *qwen35.state.Caches) Features {
    return .{
        .buffer = caches.tap.?.base(),
        .rows = 0,
        .row_bytes = @intCast(c.n_taps * c.hidden * 2),
        .space = .device,
        .dtype = if (e.model().act == .bf16) .bf16 else .f16,
    };
}

/// The drafted rounds a chain drafter gives over a continuation: drafts, acceptance and every round's chain.
const Rounds = struct {
    rounds: usize = 0,
    first: usize = 0, // rounds whose first draft is the target's next token
    accepted: usize = 0, // drafts accepted over all rounds
    chains: std.ArrayList(u32) = .empty,

    fn same(a: Rounds, b: Rounds) bool {
        return a.rounds == b.rounds and std.mem.eql(u32, a.chains.items, b.chains.items);
    }

    fn add(all: *Rounds, r: Rounds) void {
        all.rounds += r.rounds;
        all.first += r.first;
        all.accepted += r.accepted;
    }

    /// Drafts kept a round.
    fn kept(r: Rounds) f64 {
        return @as(f64, @floatFromInt(r.accepted)) / @as(f64, @floatFromInt(r.rounds));
    }
};

/// One stream's rounds over a continuation: its prompt's length, its target's features and the target's tokens.
const Track = struct { stream: *lanes.Stream, f: Features, len: usize, g: []const u32, r: Rounds = .{} };

/// Rounds as the round loop runs them, the live streams in one hold; returns the holds that ran more than one stream.
fn rounds(gpa: std.mem.Allocator, d: *Draft, tracks: []Track) !usize {
    if (tracks.len > max_streams) return error.TooManyStreams;
    const x = d.drafter();
    const depth = d.c.block - 1;
    for (tracks) |*t| t.r = .{};
    errdefer for (tracks) |*t| t.r.chains.deinit(gpa);
    var opened: usize = 0;
    defer for (tracks[0..opened]) |t| x.release(t.stream);
    var at: [max_streams]usize = undefined; // each stream's pending token g[at] at position len + at
    for (tracks, 0..) |t, i| {
        try x.open(t.stream);
        opened += 1;
        var f = t.f;
        f.offset = 0;
        f.rows = @intCast(t.len);
        try x.absorb(&.{.{ .stream = t.stream, .features = f, .start = 0, .follow = t.g[0..1] }});
        at[i] = 0;
    }
    var holds: [max_streams]Hold = undefined;
    var streams: [max_streams]*lanes.Stream = undefined;
    var chains: [max_streams][16]u32 = undefined;
    var outs: [max_streams][]u32 = undefined;
    var live: [max_streams]usize = undefined;
    var absorbs: [max_streams]Absorb = undefined;
    var shared: usize = 0;
    while (true) {
        var n: usize = 0;
        for (tracks, 0..) |t, i| {
            if (at[i] + depth + 1 > t.g.len) continue;
            const p = t.len + at[i];
            holds[n] = .{
                .stream = t.stream,
                .pending = t.g[at[i]],
                .position = p + 1,
                .depth = @intCast(depth),
                .sampling = t.stream.sampling,
            };
            streams[n] = t.stream;
            outs[n] = chains[n][0..depth];
            live[n] = i;
            n += 1;
        }
        if (n == 0) return shared;
        if (n > 1) shared += 1;
        try x.hold(holds[0..n]);
        try x.held(streams[0..n], outs[0..n]);
        for (live[0..n], outs[0..n], absorbs[0..n]) |i, chain, *a| {
            const t = &tracks[i];
            const next = t.g[at[i] + 1 ..];
            try t.r.chains.appendSlice(gpa, chain);
            var k: usize = 0;
            while (k < depth and chain[k] == next[k]) k += 1;
            t.r.rounds += 1;
            t.r.accepted += k;
            if (k > 0) t.r.first += 1;
            // the target keeps the accepted drafts and its bonus token: those rows are absorbed next
            const p = t.len + at[i];
            var f = t.f;
            f.offset = p;
            // the continuation's last round can lack its bonus token's follower: absorb only the rows it has
            const follow = next[0..@min(k + 1, next.len)];
            f.rows = @intCast(follow.len);
            a.* = .{ .stream = t.stream, .features = f, .start = p, .follow = follow, .prompt = false };
            at[i] += k + 1;
        }
        try x.absorb(absorbs[0..n]);
    }
}

/// A stream's rounds run twice on one stream: the rerun must draft the same chains; the first run's are returned.
fn twice(gpa: std.mem.Allocator, d: *Draft, t: Track) !Rounds {
    var a = [1]Track{t};
    _ = try rounds(gpa, d, &a);
    errdefer a[0].r.chains.deinit(gpa);
    var b = [1]Track{t};
    _ = try rounds(gpa, d, &b);
    defer b[0].r.chains.deinit(gpa);
    if (!a[0].r.same(b[0].r)) return error.DraftsDiffer;
    return a[0].r;
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

/// The seeded sampled run: the target draws each continuation by `rule`, and the drafter's chains draw by it too.
fn sampled(gpa: std.mem.Allocator, io: std.Io, e: *Engine, d: *Draft, prompts: []const []u32, steps: usize) !void {
    const rule: lanes.Sampling = .{ .seed = seed, .temperature = 1.0 };
    var stream: lanes.Stream = undefined;
    stream.sampling = rule;
    var all: Rounds = .{};
    for (prompts, 0..) |prompt, n| {
        var caches = try e.newCaches(prompt.len + steps + 8);
        defer caches.deinit(gpa);
        const g = try gpa.alloc(u32, steps);
        defer gpa.free(g);
        _ = try decode(io, e, &caches, prompt, g, rule);
        var a = try twice(gpa, d, .{ .stream = &stream, .f = features(e, d.c, &caches), .len = prompt.len, .g = g });
        defer a.chains.deinit(gpa);
        const fmt = "PASS drafter sampled prompt {d} (seed {d}, temperature {d:.1}): {d} rounds, " ++
            "first draft right {d}/{d}, {d:.2} drafts kept a round ({d:.2} tokens), identical rerun, chains {x:0>16}\n";
        const hash = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(a.chains.items));
        std.debug.print(fmt, .{ n, seed, rule.temperature, a.rounds, a.first, a.rounds, a.kept(), a.kept() + 1, hash });
        all.add(a);
    }
    const first = @as(f64, @floatFromInt(all.first)) / @as(f64, @floatFromInt(all.rounds));
    const summary = "drafter sampled acceptance over {d} rounds: first draft right {d:.3}, " ++
        "mean acceptance length {d:.2} (kept drafts + bonus)\n";
    std.debug.print(summary, .{ all.rounds, first, all.kept() + 1 });
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
    // frames of two streams at most: a lone stream's holds run frames of one, the pair's frames of two
    const d = try Draft.init(gpa, io, e, args[0], 2);
    defer d.deinit();
    const depth: f64 = @floatFromInt(c.block - 1);
    const loaded = "PASS drafter load: {d:.1} s, a draft step {d:.2} ms (a hold of {d} drafts {d:.2} ms)\n";
    std.debug.print(loaded, .{ ms(io, t_load) / 1e3, d.step_ms, c.block - 1, d.step_ms * depth });
    var streams: [2]lanes.Stream = undefined;
    for (&streams) |*s| s.sampling = null;
    // every prompt's greedy continuation, its caches kept for the pair that drafts two prompts together
    var caches: [texts.len]qwen35.state.Caches = undefined;
    var greedy: [texts.len][]u32 = undefined;
    var made: usize = 0;
    defer for (caches[0..made], greedy[0..made]) |*cs, g| {
        cs.deinit(gpa);
        gpa.free(g);
    };
    var plain_ms: f64 = 0;
    for (prompts, 0..) |prompt, n| {
        greedy[n] = try gpa.alloc(u32, steps);
        caches[n] = e.newCaches(prompt.len + steps + 8) catch |err| {
            gpa.free(greedy[n]);
            return err;
        };
        made += 1;
        plain_ms += try decode(io, e, &caches[n], prompt, greedy[n], null);
    }
    try refusals(d, &streams[0], features(e, c, &caches[0]));
    std.debug.print("PASS drafter refusals: a hold behind its context, more drafts read than held\n", .{});
    var alone: [texts.len]Rounds = undefined;
    var ran: usize = 0;
    defer for (alone[0..ran]) |*r| r.chains.deinit(gpa);
    var all: Rounds = .{};
    for (prompts, 0..) |prompt, n| {
        d.times = .{};
        const f = features(e, c, &caches[n]);
        alone[n] = try twice(gpa, d, .{ .stream = &streams[0], .f = f, .len = prompt.len, .g = greedy[n] });
        ran += 1;
        const a = alone[n];
        const t = d.times;
        const fmt = "PASS drafter prompt {d} ({d} tokens, {d} steps): {d} rounds, first draft right {d}/{d}, " ++
            "{d:.2} drafts kept a round ({d:.2} tokens), identical rerun; hold {d:.2} ms, absorb {d:.2} ms\n";
        const hold_ms = @as(f64, @floatFromInt(t.hold_ns)) / 1e6 / @as(f64, @floatFromInt(t.holds));
        const absorb_ms = @as(f64, @floatFromInt(t.absorb_ns)) / 1e6 / @as(f64, @floatFromInt(t.absorbs));
        const per = a.kept();
        std.debug.print(fmt, .{ n, prompt.len, steps, a.rounds, a.first, a.rounds, per, per + 1, hold_ms, absorb_ms });
        all.add(a);
    }
    const plain = plain_ms / @as(f64, @floatFromInt(prompts.len));
    const summary = "drafter acceptance over {d} rounds: first draft right {d:.3}, " ++
        "mean acceptance length {d:.2} (kept drafts + bonus); plain decode {d:.2} ms a token\n";
    const first = @as(f64, @floatFromInt(all.first)) / @as(f64, @floatFromInt(all.rounds));
    std.debug.print(summary, .{ all.rounds, first, all.kept() + 1, plain });
    // a drafter blind to its taps, or reading its ring a slot behind, keeps fewer drafts than a working one
    if (all.kept() + 1 < min_length) return error.LowAcceptance;
    // prompts 0 and 1 in frames of two streams, in both orders: a stream's chains must not depend on its slot
    var ab: [2]Track = undefined;
    var ba: [2]Track = undefined;
    for (0..2) |n| {
        const f = features(e, c, &caches[n]);
        const t: Track = .{ .stream = &streams[n], .f = f, .len = prompts[n].len, .g = greedy[n] };
        ab[n] = t;
        ba[1 - n] = t;
    }
    const together = try rounds(gpa, d, &ab);
    defer for (&ab) |*t| t.r.chains.deinit(gpa);
    _ = try rounds(gpa, d, &ba);
    defer for (&ba) |*t| t.r.chains.deinit(gpa);
    if (together == 0) return error.NoSharedFrame;
    for (0..2) |n| if (!ab[n].r.same(ba[1 - n].r)) return error.FrameSlotsDiffer;
    for (0..2) |n| {
        const r = ab[n].r;
        const fmt = "PASS drafter two streams, prompt {d}: {d} frames of two, chains equal in either slot; " ++
            "{d} rounds, {d:.2} tokens a round (alone {d} rounds, {d:.2}), chains {s} its chains alone\n";
        const equal = if (r.same(alone[n])) "equal to" else "unlike";
        std.debug.print(fmt, .{ n, together, r.rounds, r.kept() + 1, alone[n].rounds, alone[n].kept() + 1, equal });
    }
    try sampled(gpa, io, e, d, &prompts, steps);
}

/// The mean acceptance length under which the check fails: below a clean run's, above a ring read a slot behind.
const min_length = 4.5;

/// The sampled run's seed.
const seed = 7;
