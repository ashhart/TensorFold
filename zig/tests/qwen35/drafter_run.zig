//! `drafter <drafter dir> <target dir> [--steps N]`: the HIP DFlash2 drafter's chains against the plain target (GPU).

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const qwen35 = @import("qwen3_5");

const Engine = qwen35.engine.Engine;
const Hip = qwen35.hip_lanes.Hip;
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

/// Drafts kept at `depth` a hold, frames truncated to depth + 1 rows and full blocks, each hold at the same context.
const Truncated = struct { rounds: usize = 0, same: usize = 0, short: usize = 0, full: usize = 0 };

/// A greedy walk held twice a round, at `depth` (a frame of depth + 1 rows) and at the block (its first drafts).
fn truncated(d: *Draft, s: *lanes.Stream, f0: Features, len: usize, g: []const u32, depth: usize) !Truncated {
    const x = d.drafter();
    const block = d.c.block - 1;
    var r: Truncated = .{};
    try x.open(s);
    defer x.release(s);
    var f = f0;
    f.offset = 0;
    f.rows = @intCast(len);
    try x.absorb(&.{.{ .stream = s, .features = f, .start = 0, .follow = g[0..1] }});
    var p = len;
    while (p - len + block + 1 <= g.len) {
        const at = p - len;
        var short: [16]u32 = undefined;
        var full: [16]u32 = undefined;
        try x.hold(&.{.{ .stream = s, .pending = g[at], .position = p + 1, .depth = @intCast(depth) }});
        try x.held(&.{s}, &.{short[0..depth]});
        try x.hold(&.{.{ .stream = s, .pending = g[at], .position = p + 1, .depth = @intCast(block) }});
        try x.held(&.{s}, &.{full[0..block]});
        const ks = matching(short[0..depth], g[at + 1 ..]);
        const kf = matching(full[0..depth], g[at + 1 ..]);
        r.rounds += 1;
        r.short += ks;
        r.full += kf;
        if (std.mem.eql(u32, short[0..depth], full[0..depth])) r.same += 1;
        // the walk keeps the full block's drafts, so both holds see the same contexts
        f.offset = p;
        f.rows = @intCast(kf + 1);
        const follow = g[at + 1 ..][0 .. kf + 1];
        try x.absorb(&.{.{ .stream = s, .features = f, .start = p, .follow = follow, .prompt = false }});
        p += kf + 1;
    }
    return r;
}

/// Drafts matching the greedy continuation `g` from the first.
fn matching(drafts: []const u32, g: []const u32) usize {
    var k: usize = 0;
    while (k < drafts.len and drafts[k] == g[k]) k += 1;
    return k;
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
    // features of other rows than the absorb names, and a kept row past them
    f.offset = 0;
    const elsewhere = x.absorb(&.{.{ .stream = s, .features = f, .start = 4, .follow = &.{}, .prompt = false }});
    if (elsewhere != error.FeaturesNotAtStart) return error.StartNotHonoured;
    f.offset = 4;
    const past = x.absorb(&.{.{ .stream = s, .features = f, .start = 4, .follow = &.{}, .rows = &.{ 0, 4 } }});
    if (past != error.BadDraftRows) return error.KeptRowNotRefused;
}

/// Rows a verified window of `wide` rows keeps (a tree's path, out of the first rows), and their count.
const picks = [_]u32{ 0, 5, 10, 15 };
const wide = 16;

/// One stream's chain after the prompt and an absorb at `p` (`ring`: the tap ring the features name).
fn keptChain(d: *Draft, s: *lanes.Stream, f0: Features, ring: u64, p: usize, rows: ?[]const u32, pending: u32) ![7]u32 {
    const x = d.drafter();
    try x.open(s);
    defer x.release(s);
    var f = f0;
    f.buffer = ring;
    f.offset = 0;
    f.rows = @intCast(p);
    try x.absorb(&.{.{ .stream = s, .features = f, .start = 0, .follow = &.{} }});
    // kept rows pick from the window's rows; whole rows are as many of its first
    f.offset = p;
    f.rows = if (rows != null) wide else picks.len;
    try x.absorb(&.{.{ .stream = s, .features = f, .start = p, .follow = &.{}, .rows = rows, .prompt = false }});
    var held: [7]u32 = undefined;
    try x.hold(&.{.{ .stream = s, .pending = pending, .position = p + picks.len + 1, .depth = 7 }});
    try x.held(&.{s}, &.{&held});
    return held;
}

/// Kept rows `picks` of a window draft as the first rows of a ring holding those rows in order, not as its first rows.
fn keptRows(e: *Engine, d: *Draft, s: *lanes.Stream, f: Features, p: usize, pending: u32) !void {
    const t = e.model().taps;
    const row: usize = f.row_bytes;
    var alt = try hip.DeviceBuffer.alloc(&e.driver, t.rows * row);
    defer alt.free();
    try e.stream.synchronize();
    try alt.copyFrom(0, f.buffer, t.rows * row);
    for (picks, 0..) |from, to| try alt.copyFrom(((p + to) % t.rows) * row, f.buffer + ((p + from) % t.rows) * row, row);
    const kept = try keptChain(d, s, f, f.buffer, p, &picks, pending);
    const moved = try keptChain(d, s, f, alt.base(), p, null, pending);
    if (!std.mem.eql(u32, &kept, &moved)) return error.KeptRowsDiffer;
    const first = [_]u32{ 0, 1, 2, 3 };
    const prefix = try keptChain(d, s, f, f.buffer, p, &first, pending);
    const whole = try keptChain(d, s, f, f.buffer, p, null, pending);
    if (!std.mem.eql(u32, &prefix, &whole)) return error.KeptPrefixDiffers;
    // rows the absorb ignored would draft as the first rows: the picks must move the chain to tell
    if (std.mem.eql(u32, &kept, &whole)) return error.KeptRowsVacuous;
}

/// A prompt past the tap ring through the lanes backend and Drafted, as the server runs it; then `features`' bounds.
fn served(gpa: std.mem.Allocator, e: *Engine, d: *Draft, prompt: []const u32) !void {
    const h = try Hip.init(gpa, e);
    defer h.deinit();
    const b = h.backend();
    var x = try lanes.drafted.Drafted.init(gpa, b, d.drafter());
    defer x.deinit();
    var s = try lanes.Stream.init(gpa, .{ .id = "long", .prompt = prompt, .max_new = 4 });
    defer s.deinit(gpa);
    // the whole prompt's rows: the drafter reads its window's newest, still in the ring
    try x.backend().prefill(&s);
    defer x.backend().release(&s);
    const taps = d.c.tapLayers();
    const len: u64 = prompt.len;
    const ring: u64 = e.model().taps.rows;
    const window: u64 = qwen35.engine.tap_window;
    const whole = try b.features(&s, taps, 0, @intCast(len - 1));
    if (whole.offset != 0 or whole.rows != len - 1) return error.FeaturesMoved;
    _ = try b.features(&s, taps, len - window, @intCast(window));
    const refused = [_][2]u64{
        .{ len - ring - 8, 16 }, // its first rows are out of the ring
        .{ 0, 16 }, // all of them are
        .{ len - 4, 5 }, // its last row is not written
        .{ len - 4, 0 }, // no row
    };
    for (refused) |r| {
        const got = b.features(&s, taps, r[0], @intCast(r[1]));
        if (got != error.RowsNotHeld) {
            std.debug.print("FAIL drafter features [{d}, +{d}) of {d} rows in a {d}-row ring: served\n", .{ r[0], r[1], len, ring });
            return error.FeaturesNotBounded;
        }
    }
    if (b.features(&s, taps[1..], 0, 4) != error.TapsMismatch) return error.TapsNotChecked;
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
    if (steps < min_steps) {
        std.debug.print("--steps {d}: at least {d}, so the rounds measure acceptance\n", .{ steps, min_steps });
        return error.Usage;
    }
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
    // a prompt past the tap ring of 1024-row prompt chunks: the served path's eviction
    const long = try gpa.alloc(u32, qwen35.engine.tap_window + qwen35.prefix.chunk + 256);
    defer gpa.free(long);
    for (long, 0..) |*t, i| t.* = prompts[i % prompts.len][(i / prompts.len) % prompts[i % prompts.len].len];
    var target = try qwen35.Config.read(gpa, io, args[1]);
    defer target.deinit();
    const c = try qwen35.draft_config.Config.read(gpa, io, args[0], qwen35.draft_config.Target.of(target.spec));
    const policy = (try @import("group.zig").resolve("", 0)).policy;
    const e = try Engine.open(gpa, io, args[1], .{
        .capacity = @max(longest + steps, long.len + Hip.max_window) + 64,
        .prompt_rows = qwen35.prefix.chunk,
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
    const loaded = "PASS drafter load: {d:.1} s, a draft step {d:.2} ms (past a one-draft hold)\n";
    std.debug.print(loaded, .{ ms(io, t_load) / 1e3, d.step_ms });
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
    var cut: [cut_depths.len]Truncated = .{ .{}, .{} };
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
    std.debug.print("PASS drafter refusals: a hold behind its context, more drafts read than held, " ++
        "features of other rows, a kept row past them\n", .{});
    try keptRows(e, d, &streams[0], features(e, c, &caches[0]), prompts[0].len, greedy[0][4]);
    std.debug.print("PASS drafter kept rows: rows {any} of {d} draft as those rows in order, not as the " ++
        "first rows; a prefix as whole rows\n", .{ picks, wide });
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
        for (&cut, cut_depths) |*sum, depth| {
            const one = try truncated(d, &streams[0], f, prompt.len, greedy[n], depth);
            inline for (.{ "rounds", "same", "short", "full" }) |name| @field(sum, name) += @field(one, name);
        }
    }
    try truncatedHolds(cut);
    if (all.rounds < min_rounds) {
        std.debug.print("FAIL drafter: {d} rounds, too few to judge acceptance (at least {d})\n", .{ all.rounds, min_rounds });
        return error.TooFewRounds;
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
    try served(gpa, e, d, long);
    std.debug.print("PASS drafter served: a {d}-token prompt through the lanes backend and Drafted; features " ++
        "serve its drafter window and refuse rows out of the ring, unwritten or none\n", .{long.len});
}

/// The depths whose truncated frames are checked against the full block.
const cut_depths = [_]usize{ 1, 3 };

/// The acceptance length a truncated frame may lose to the full block.
const max_truncated_loss = 0.05;

/// Truncated frames keep the full block's acceptance, not its chains: the block's rows attend to each other unmasked.
fn truncatedHolds(cut: [cut_depths.len]Truncated) !void {
    for (cut, cut_depths) |r, depth| {
        const n: f64 = @floatFromInt(r.rounds);
        const short = @as(f64, @floatFromInt(r.short)) / n + 1;
        const full = @as(f64, @floatFromInt(r.full)) / n + 1;
        const fmt = "{s} drafter depth {d} truncated frames: {d} rounds, {d} chains as the full block's, " ++
            "acceptance length {d:.3} against {d:.3}\n";
        const lost = short + max_truncated_loss < full;
        std.debug.print(fmt, .{ if (lost) "FAIL" else "PASS", depth, r.rounds, r.same, short, full });
        if (lost) return error.TruncatedFramesLoseAcceptance;
    }
}

/// Greedy steps a prompt at least, and rounds over all prompts at least, for the acceptance to mean anything.
const min_steps = 64;
const min_rounds = 24;

/// The mean acceptance length under which the check fails: below a clean run's, above a ring read a slot behind.
const min_length = 4.5;

/// The sampled run's seed.
const seed = 7;
