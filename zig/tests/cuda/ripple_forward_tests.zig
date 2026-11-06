//! Ripple's forward rounds == one-row steps, bit for bit (ripple-forward): streams, windows, graphs, reuse, taps.
const std = @import("std");
const cuda = @import("cuda");
const rc = @import("ricursive_cuda");
const check = @import("check.zig");

const fwd = rc.model;
const wts = rc.weights;
const fixture = @import("ripple_fixture.zig"); // the release-width target's random weights

pub const config = fixture.config;
pub const build = fixture.build;

/// A stream: its prompt's length and decode tokens (teacher-forced, so every run feeds the same rows).
const Spec = struct { name: []const u8, prompt: usize };

const specs = [_]Spec{
    .{ .name = "A", .prompt = 507 }, // rows across the 512-row page edge
    .{ .name = "B", .prompt = 3066 }, // across the 3,072-row ring's wrap
    .{ .name = "C", .prompt = 1019 }, // across 1,024: a page edge and the window's
    .{ .name = "D", .prompt = 37 },
    .{ .name = "E", .prompt = 4093 }, // a wrapped ring, rows across 4,096 keys (8 to 9 key groups)
};

/// Rounds of (stream, rows): several streams a row each, a verify window, mixes of 20 and 48 rows, one and k rows.
const rounds = [_][]const [2]usize{
    &.{ .{ 0, 1 }, .{ 1, 1 }, .{ 2, 1 }, .{ 3, 1 }, .{ 4, 1 } },
    &.{.{ 1, 8 }},
    &.{ .{ 0, 5 }, .{ 2, 5 }, .{ 3, 5 }, .{ 4, 5 } },
    &.{ .{ 0, 10 }, .{ 1, 10 }, .{ 2, 10 }, .{ 3, 10 }, .{ 4, 8 } },
    &.{ .{ 3, 1 }, .{ 1, 3 } },
};
const round_rows = 48;

fn decodeTokens(j: usize) usize {
    var n: usize = 0;
    for (rounds) |r| for (r) |x| if (x[0] == j) {
        n += x[1];
    };
    return n;
}

const Streams = struct {
    gpa: std.mem.Allocator,
    ids: [specs.len][]u32 = undefined, // the prompt, then the decode tokens
    want: [specs.len][][]u8 = undefined, // one-row steps' logits a decode token
    prompt: [specs.len][]u8 = undefined, // the prompt's last row's logits

    fn init(gpa: std.mem.Allocator, vocab: usize, seed: u64) !Streams {
        var x: Streams = .{ .gpa = gpa };
        var prng = std.Random.DefaultPrng.init(seed);
        for (specs, 0..) |sp, j| {
            x.ids[j] = try gpa.alloc(u32, sp.prompt + decodeTokens(j));
            for (x.ids[j]) |*t| t.* = prng.random().uintLessThan(u32, @intCast(vocab));
            x.want[j] = try gpa.alloc([]u8, decodeTokens(j));
            for (x.want[j]) |*l| l.* = try gpa.alloc(u8, vocab * 2);
            x.prompt[j] = try gpa.alloc(u8, vocab * 2);
        }
        return x;
    }

    fn deinit(x: *Streams) void {
        for (0..specs.len) |j| {
            for (x.want[j]) |l| x.gpa.free(l);
            x.gpa.free(x.want[j]);
            x.gpa.free(x.ids[j]);
            x.gpa.free(x.prompt[j]);
        }
    }
};

/// Equal logits rows, or the first differing logit and how many differ.
fn same(comptime what: []const u8, args: anytype, got: []const u8, want: []const u8) bool {
    if (std.mem.eql(u8, got, want)) return true;
    const g = std.mem.bytesAsSlice(u16, got);
    const w = std.mem.bytesAsSlice(u16, want);
    var diff: usize = 0;
    var first: usize = g.len;
    for (g, w, 0..) |a, b, i| if (a != b) {
        diff += 1;
        first = @min(first, i);
    };
    std.debug.print("  DIFF ", .{});
    std.debug.print(what, args);
    std.debug.print(": {d} of {d} logits differ, first {d}: {x:0>4} for {x:0>4}\n", .{ diff, g.len, first, g[first], w[first] });
    return false;
}

/// A prompt into `slot` from 0 in chunks of up to 2,048 rows (M1's path), its last row through the head.
fn prefill(m: *const fwd.Model, slot: usize, ids: []const u32) !fwd.Seq {
    var seq: fwd.Seq = .{ .slot = slot };
    while (seq.len < ids.len) {
        const end = @min(ids.len, seq.len + fwd.prompt_rows);
        const at = seq.len;
        try m.forward(&seq, ids[at..end], .prompt);
        if (end == ids.len) try m.head(end - at - 1);
    }
    return seq;
}

/// One-row steps of stream j in its reference slot j: the logits each round must give.
fn reference(m: *fwd.Model, x: *Streams, j: usize, logits: []u8) !usize {
    const ids = x.ids[j];
    const n = specs[j].prompt;
    var seq = try prefill(m, j, ids[0..n]);
    try m.download(m.b.logits, x.prompt[j]);
    var fails: usize = 0;
    for (x.want[j], n..) |want, p| {
        try m.forward(&seq, ids[p..][0..1], .decode);
        try m.head(0);
        try m.download(m.b.logits, want);
        if (p != n and p != n + 5) continue;
        // the same row again on four-warp tree blocks (sms 0): its keys are rewritten with the same bits
        const sms = m.gd.tree.sms;
        m.gd.tree.sms = 0;
        seq.len -= 1;
        try m.forward(&seq, ids[p..][0..1], .decode);
        try m.head(0);
        try m.download(m.b.logits, logits);
        m.gd.tree.sms = sms;
        if (same("tree one-warp vs four-warp, stream {s} at {d}", .{ specs[j].name, p }, logits, want)) {
            check.pass("ripple-forward tree one-warp == four-warp blocks: stream {s} at {d} ({d} key groups)", .{ specs[j].name, p, rc.attn.groupsOf(&.{p}) });
        } else {
            std.debug.print("FAIL ripple-forward tree one-warp != four-warp blocks: stream {s} at {d}\n", .{ specs[j].name, p });
            fails += 1;
        }
    }
    return fails;
}

/// The driver's free device bytes (GB10: of its unified memory) after the stream's work.
fn freeBytes(m: *const fwd.Model) !usize {
    try m.s.synchronize();
    var free: usize = 0;
    var total: usize = 0;
    try m.d.check(m.d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
    return free;
}

/// The last round's rows' logits by rowLogits into `out`.
fn roundRows(m: *const fwd.Model, n: usize, out: []u8) !void {
    try m.rowLogits(n);
    try m.download(m.b.rows, out[0 .. n * m.c.vocab * 2]);
}

/// Each of n rows of `got` the same as `want`'s.
fn sameRows(what: []const u8, n: usize, v: usize, got: []const u8, want: []const u8) bool {
    var ok = true;
    for (0..n) |r| ok = same("{s} row {d}", .{ what, r }, got[r * v ..][0..v], want[r * v ..][0..v]) and ok;
    return ok;
}

/// A round from its graph into `got`, then eager over the same caches into `want`; true when every row's logits match.
fn graphVsEager(m: *const fwd.Model, what: []const u8, chains: []const fwd.Chain, ids: []const u32, got: []u8, want: []u8) !bool {
    try m.roundAs(chains, ids, .graph);
    try roundRows(m, ids.len, got);
    try m.roundAs(chains, ids, .exact);
    try roundRows(m, ids.len, want);
    return sameRows(what, ids.len, m.c.vocab * 2, got, want);
}

/// Rounds over the test slots against the references, by rowLogits and by head: a graph, fixed eager, then B1's eager.
fn runRounds(m: *const fwd.Model, x: *Streams, base: usize, done: *[specs.len]usize, logits: []u8, rows: []u8, by_graph: []u8, by_fixed: []u8) !usize {
    const v = m.c.vocab;
    var fails: usize = 0;
    var chains: [specs.len]fwd.Chain = undefined;
    var ids: [round_rows]u32 = undefined;
    var who: [round_rows][2]usize = undefined; // (stream, decode token) a row
    var pos: [round_rows]usize = undefined;
    for (rounds, 0..) |rd, ri| {
        var n: usize = 0;
        for (rd, 0..) |c, ci| {
            const j = c[0];
            const p0 = specs[j].prompt + done[j];
            chains[ci] = .{ .slot = base + j, .p0 = p0, .tokens = c[1] };
            for (0..c[1]) |i| {
                ids[n] = x.ids[j][p0 + i];
                who[n] = .{ j, done[j] + i };
                pos[n] = p0 + i;
                n += 1;
            }
            done[j] += c[1];
        }
        const ch = chains[0..rd.len];
        const free0 = try freeBytes(m);
        try m.roundAs(ch, ids[0..n], .graph);
        try roundRows(m, n, by_graph);
        const graph_mib = (@as(f64, @floatFromInt(free0)) - @as(f64, @floatFromInt(try freeBytes(m)))) / (1 << 20);
        std.debug.print("INFO ripple-forward graph of {d} rows: {d:.1} MiB of device memory (the driver's free bytes before and after its capture), {d} graphs\n", .{ n, graph_mib, m.captured() });
        try m.roundAs(ch, ids[0..n], .fixed);
        try roundRows(m, n, by_fixed);
        try m.roundAs(ch, ids[0..n], .exact);
        try roundRows(m, n, rows);
        var bad: usize = 0;
        for (0..n) |r| {
            const want = x.want[who[r][0]][who[r][1]];
            const name = specs[who[r][0]].name;
            if (!same("round {d} row {d} (stream {s} at {d}) by rowLogits", .{ ri, r, name, pos[r] }, rows[r * v * 2 ..][0 .. v * 2], want)) bad += 1;
            try m.head(r);
            try m.download(m.b.logits, logits);
            if (!same("round {d} row {d} (stream {s} at {d}) by head", .{ ri, r, name, pos[r] }, logits, want)) bad += 1;
        }
        const ng = rc.attn.groupsOf(pos[0..n]);
        const path = if (rc.attn.oneWarp(&m.gd, n, ng)) "one-warp" else "four-warp";
        if (bad == 0) {
            check.pass("ripple-forward round {d}: {d} chains, {d} rows ({s} tree blocks, {d} key groups) == one-row steps", .{ ri, rd.len, n, path, ng });
        } else {
            std.debug.print("FAIL ripple-forward round {d}: {d} chains, {d} rows ({s} tree blocks, {d} key groups): {d} row comparisons differ\n", .{ ri, rd.len, n, path, ng, bad });
            fails += 1;
        }
        const fixed = m.gd.groups();
        const fixed_path = if (rc.attn.oneWarp(&m.gd, n, fixed)) "one-warp" else "four-warp";
        if (sameRows("fixed key groups", n, v * 2, by_fixed, rows)) {
            check.pass("ripple-forward fixed key groups, round {d}: {d} groups ({s}) == the round's own {d} ({s}), eager", .{ ri, fixed, fixed_path, ng, path });
        } else {
            std.debug.print("FAIL ripple-forward fixed key groups, round {d}: {d} groups ({s}) != the round's own {d} ({s})\n", .{ ri, fixed, fixed_path, ng, path });
            fails += 1;
        }
        if (sameRows("graph", n, v * 2, by_graph, rows)) {
            check.pass("ripple-forward graph, round {d}: {d} rows replayed from a graph == the eager round", .{ ri, n });
        } else {
            std.debug.print("FAIL ripple-forward graph, round {d}: {d} rows from a graph != the eager round\n", .{ ri, n });
            fails += 1;
        }
    }
    return fails;
}

/// A replayed round: (slot, p0, rows) a chain.
const Replay = struct { what: []const u8, chains: []const [3]usize };

/// Graphs captured by the rounds above, replayed at other positions and slots (each against eager).
const elsewhere = [_]Replay{
    .{ .what = "the 5-row graph (five one-row chains) as one window in slot 1 at 3,070 (the ring's wrap, a page edge)", .chains = &.{.{ 1, 3070, 5 }} },
    .{ .what = "the 8-row graph (one window) as four 2-row chains: page edges at 512 and 1,024, 4,096 keys, the reused slot", .chains = &.{ .{ 0, 511, 2 }, .{ 2, 1023, 2 }, .{ 4, 4095, 2 }, .{ 6, 52, 2 } } },
    .{ .what = "the one-row graph in slot 8 at 54", .chains = &.{.{ 8, 54, 1 }} },
    .{ .what = "the 4-row graph (two chains) as one window over 4,096 keys of a wrapped ring", .chains = &.{.{ 9, 4094, 4 }} },
    .{ .what = "the 20-row graph (four chains) as one window in slot 4 at 3,060 (the ring's wrap, a page edge)", .chains = &.{.{ 4, 3060, 20 }} },
};

/// Graphs of several row counts replayed in another order than their captures (5, 8, 20, 48, 4, 17, 1).
const any_order = [_]Replay{
    .{ .what = "48 rows", .chains = &.{ .{ 5, 3040, 24 }, .{ 7, 1000, 24 } } },
    .{ .what = "1 row", .chains = &.{.{ 3, 53, 1 }} },
    .{ .what = "17 rows", .chains = &.{.{ 9, 4080, 17 }} },
    .{ .what = "4 rows", .chains = &.{ .{ 0, 500, 1 }, .{ 1, 3080, 3 } } },
    .{ .what = "1 row again", .chains = &.{.{ 2, 1030, 1 }} },
    .{ .what = "8 rows", .chains = &.{.{ 8, 40, 8 }} },
};

/// A replay's chains and `toks`' first rows as its ids; true when the graph's rows equal eager's.
fn replay(m: *const fwd.Model, r: Replay, toks: []const u32, got: []u8, want: []u8) !bool {
    var chains: [specs.len]fwd.Chain = undefined;
    var n: usize = 0;
    for (r.chains, 0..) |c, i| {
        chains[i] = .{ .slot = c[0], .p0 = c[1], .tokens = c[2] };
        n += c[2];
    }
    return graphVsEager(m, r.what, chains[0..r.chains.len], toks[0..n], got, want);
}

/// The tapped layers (columns 0, 1 and 2): sliding, global, sliding.
const tap_layers = [_]u32{ 1, 4, 8 };

/// A slot's whole tap ring into `out`.
fn ringOf(m: *const fwd.Model, slot: usize, out: []u8) !void {
    try m.download(m.tapRing(slot).?, out[0 .. fwd.tap_rows * m.tapRowBytes()]);
}

/// Position p's tap column t in a downloaded ring.
fn tapRow(m: *const fwd.Model, ring: []const u8, p: usize, t: usize) []const u8 {
    const h2 = m.c.hidden * 2;
    return ring[(p % fwd.tap_rows) * m.tapRowBytes() + t * h2 ..][0..h2];
}

/// The ring rows that differ between two downloaded rings.
fn ringDiff(m: *const fwd.Model, got: []const u8, want: []const u8) usize {
    const row = m.tapRowBytes();
    var n: usize = 0;
    for (0..fwd.tap_rows) |r| n += @intFromBool(!std.mem.eql(u8, got[r * row ..][0..row], want[r * row ..][0..row]));
    return n;
}

/// Stream B in a zeroed `slot`: a prompt chunk's last logits, a graph round's and an eager round's (22-token window).
fn plainB(m: *const fwd.Model, x: *const Streams, slot: usize, last: []u8, by_graph: []u8, eager: []u8) !void {
    const n = specs[1].prompt;
    const t = decodeTokens(1);
    try m.reset(slot, m.context);
    _ = try prefill(m, slot, x.ids[1][0..n]);
    try m.download(m.b.logits, last);
    const chain = [_]fwd.Chain{.{ .slot = slot, .p0 = n, .tokens = t }};
    try m.roundAs(&chain, x.ids[1][n..][0..t], .graph);
    try roundRows(m, t, by_graph);
    try m.roundAs(&chain, x.ids[1][n..][0..t], .exact);
    try roundRows(m, t, eager);
}

/// One-row eager steps of `ids` in `slot` from p0.
fn eagerSteps(m: *const fwd.Model, slot: usize, p0: usize, ids: []const u32) !void {
    for (ids, p0..) |id, p| try m.roundAs(&.{.{ .slot = slot, .p0 = p, .tokens = 1 }}, &.{id}, .exact);
}

/// The drafter's taps (slots zeroed first): tap rows equal the residual after each tapped layer; steps 1 to 5 below.
fn taps(m: *fwd.Model, w: *const wts.Weights, x: *const Streams) !usize {
    const a = m.gpa;
    const c = m.c;
    const v2 = c.vocab * 2;
    const h2 = c.hidden * 2;
    const t = decodeTokens(1);
    var fails: usize = 0;
    const bufs = try a.alloc(u8, 2 * (v2 + 2 * t * v2));
    defer a.free(bufs);
    const off = .{ bufs[0..v2], bufs[v2..][0 .. t * v2], bufs[v2 + t * v2 ..][0 .. t * v2] };
    const on = .{ bufs[v2 + 2 * t * v2 ..][0..v2], bufs[2 * v2 + 2 * t * v2 ..][0 .. t * v2], bufs[2 * v2 + 3 * t * v2 ..][0 .. t * v2] };

    // 4 (before): plain logits without taps, the round's graph captured without the tap launches
    if (m.tap != null) return error.TapsAlreadySet;
    try plainB(m, x, 4, off[0], off[1], off[2]);

    try m.setTaps(&tap_layers);
    if (m.captured() != 0) {
        std.debug.print("FAIL ripple-forward taps: setTaps left {d} graphs captured\n", .{m.captured()});
        fails += 1;
    }
    for (0..m.arena.slots) |sl| try m.reset(sl, m.context);
    const ring_bytes = fwd.tap_rows * m.tapRowBytes();
    const rings = try a.alloc(u8, 2 * ring_bytes);
    defer a.free(rings);
    const ring_a = rings[0..ring_bytes];
    const ring_b = rings[ring_bytes..];
    const resid = try a.alloc(u8, fwd.prompt_rows * h2);
    defer a.free(resid);

    // 1 and 3: B's prompt (3,066 rows: 2,048, then 1,018 at 2,048 to 3,065, rows 2,048 to 2,063 then 0 to 1,001)
    var cut: [tap_layers.len]wts.Weights = undefined; // the model ending at each tapped layer (shared buffers)
    for (&cut, tap_layers) |*cw, l| {
        cw.* = w.*;
        cw.layers = w.layers[0 .. l + 1];
    }
    const ids = x.ids[1][0..specs[1].prompt];
    var chunk_bad: usize = 0;
    var wrap_bad: usize = 0;
    var kept_bad: usize = 0;
    var at: usize = 0;
    while (at < ids.len) {
        const end = @min(ids.len, at + fwd.prompt_rows);
        try ringOf(m, 0, ring_a);
        try m.prompt(0, at, ids[at..end]);
        try ringOf(m, 0, ring_b);
        for (0..fwd.tap_rows) |r| {
            const named = (r + fwd.tap_rows - at % fwd.tap_rows) % fwd.tap_rows < end - at;
            const row = m.tapRowBytes();
            if (!named and !std.mem.eql(u8, ring_a[r * row ..][0..row], ring_b[r * row ..][0..row])) kept_bad += 1;
        }
        for (&cut, 0..) |*cw, col| {
            m.w = cw;
            defer m.w = w;
            try m.prompt(1 + col, at, ids[at..end]);
            try m.download(m.b.resid, resid[0 .. (end - at) * h2]);
            for (at..end) |p| if (!std.mem.eql(u8, tapRow(m, ring_b, p, col), resid[(p - at) * h2 ..][0..h2])) {
                chunk_bad += 1;
                wrap_bad += @intFromBool(p >= fwd.tap_rows);
            };
        }
        at = end;
    }
    if (chunk_bad == 0) {
        check.pass("ripple-forward taps 1: B's 2 prompt chunks (3,066 rows), layers 1, 4 and 8 == the residual after each layer (eager)", .{});
    } else {
        std.debug.print("FAIL ripple-forward taps 1: {d} of {d} tap rows differ from the residual after their layer\n", .{ chunk_bad, ids.len * tap_layers.len });
        fails += 1;
    }
    var untouched: usize = 0; // slots 4-9: no chunk named them since the reset
    for (4..m.arena.slots) |sl| {
        try ringOf(m, sl, ring_a);
        untouched += @intFromBool(std.mem.indexOfNone(u8, ring_a, &.{0}) != null);
    }
    if (wrap_bad == 0 and kept_bad == 0 and untouched == 0) {
        check.pass("ripple-forward taps 3: positions 2,064 to 3,065 at pos % 2,064; rows 1,002 to 2,047 and other slots' rings untouched", .{});
    } else {
        std.debug.print("FAIL ripple-forward taps 3: {d} wrapped rows wrong, {d} unnamed rows changed, {d} other slots' rings written\n", .{ wrap_bad, kept_bad, untouched });
        fails += 1;
    }

    // 4 (after): the same logits with taps on, eager and from a graph captured with the tap launches
    try plainB(m, x, 4, on[0], on[1], on[2]);
    const p_ok = same("taps on: B's prompt", .{}, on[0], off[0]);
    const g_ok = sameRows("taps on: graph round", t, v2, on[1], off[1]);
    const e_ok = sameRows("taps on: eager round", t, v2, on[2], off[2]);
    if (p_ok and g_ok and e_ok) {
        check.pass("ripple-forward taps 4: plain logits with taps on == taps off: B's prompt, a {d}-row round eager and from a graph", .{t});
    } else {
        std.debug.print("FAIL ripple-forward taps 4: taps change plain logits (same: prompt {d}, graph round {d}, eager round {d})\n", .{ @intFromBool(p_ok), @intFromBool(g_ok), @intFromBool(e_ok) });
        fails += 1;
    }

    // 2: two streams, 16 rows each, one across the ring's wrap (2,056 to 2,071), the other at 100 to 115
    var prng = std.Random.DefaultPrng.init(0x7461_7073);
    var toks: [3 * 32]u32 = undefined;
    for (&toks) |*tk| tk.* = prng.random().uintLessThan(u32, @intCast(c.vocab));
    const lens = [2]usize{ 2056, 100 };
    for ([_]usize{ 5, 7 }) |sl| _ = try prefill(m, sl, x.ids[4][0..lens[0]]);
    for ([_]usize{ 6, 8 }) |sl| _ = try prefill(m, sl, x.ids[0][0..lens[1]]);
    // the 32-row graph captured first in slots 4 and 9 (positions 0 to 15), then replayed below
    try m.reset(4, m.context);
    try m.roundAs(&.{ .{ .slot = 4, .p0 = 0, .tokens = 16 }, .{ .slot = 9, .p0 = 0, .tokens = 16 } }, toks[0..32], .graph);
    const graphs = m.captured();
    try m.roundAs(&.{ .{ .slot = 7, .p0 = lens[0], .tokens = 16 }, .{ .slot = 8, .p0 = lens[1], .tokens = 16 } }, toks[0..32], .graph);
    try eagerSteps(m, 5, lens[0], toks[0..16]);
    try eagerSteps(m, 6, lens[1], toks[16..32]);
    var round_bad: usize = 0;
    for ([_][2]usize{ .{ 7, 5 }, .{ 8, 6 } }) |pair| {
        try ringOf(m, pair[0], ring_a);
        try ringOf(m, pair[1], ring_b);
        round_bad += ringDiff(m, ring_a, ring_b);
    }
    if (round_bad == 0 and m.captured() == graphs) {
        check.pass("ripple-forward taps 2: a 2-stream round of 16 rows a stream from a graph (one across the wrap) == one-row eager steps, whole rings", .{});
    } else {
        std.debug.print("FAIL ripple-forward taps 2: {d} ring rows differ; graphs {d} before the replay, {d} after\n", .{ round_bad, graphs, m.captured() });
        fails += 1;
    }

    // 5: a verify round at 2,072 and 116 keeps 5 and 11 rows; the next 16-row rounds start at 2,077 and 127
    const p1 = [2]usize{ lens[0] + 16, lens[1] + 16 };
    const kept = [2]usize{ 5, 11 };
    try m.roundAs(&.{ .{ .slot = 7, .p0 = p1[0], .tokens = 16 }, .{ .slot = 8, .p0 = p1[1], .tokens = 16 } }, toks[32..64], .graph);
    try m.roundAs(&.{ .{ .slot = 7, .p0 = p1[0] + kept[0], .tokens = 16 }, .{ .slot = 8, .p0 = p1[1] + kept[1], .tokens = 16 } }, toks[64..96], .graph);
    try eagerSteps(m, 5, p1[0], toks[32..][0..kept[0]]);
    try eagerSteps(m, 5, p1[0] + kept[0], toks[64..80]);
    try eagerSteps(m, 6, p1[1], toks[48..][0..kept[1]]);
    try eagerSteps(m, 6, p1[1] + kept[1], toks[80..96]);
    var keep_bad: usize = 0;
    for ([_][2]usize{ .{ 7, 5 }, .{ 8, 6 } }) |pair| {
        try ringOf(m, pair[0], ring_a);
        try ringOf(m, pair[1], ring_b);
        keep_bad += ringDiff(m, ring_a, ring_b);
    }
    if (keep_bad == 0 and m.captured() == graphs) {
        check.pass("ripple-forward taps 5: a verify round keeping 5 and 11 of 16 rows, then the next round at the rejected positions == one-row steps of the kept rows", .{});
    } else {
        std.debug.print("FAIL ripple-forward taps 5: {d} ring rows differ; graphs {d}, then {d}\n", .{ keep_bad, graphs, m.captured() });
        fails += 1;
    }
    return fails;
}

pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    const c = config();
    try wts.gate(&c);
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var w = try build(gpu, c, 0x7269_7070);
    defer w.deinit();
    var k = try fwd.Kernels.load(gpu.ctx);
    defer k.deinit();
    // slots 0-4 the references', 5-9 the rounds'
    var m = try fwd.Model.init(a, gpu.d, s, &w, &k, c.context, 2 * specs.len, round_rows);
    defer m.deinit();
    var x = try Streams.init(a, c.vocab, 0x6677);
    defer x.deinit();
    const logits = try a.alloc(u8, c.vocab * 2);
    defer a.free(logits);
    const rows = try a.alloc(u8, round_rows * c.vocab * 2);
    defer a.free(rows);
    const by_graph = try a.alloc(u8, round_rows * c.vocab * 2);
    defer a.free(by_graph);
    const by_fixed = try a.alloc(u8, round_rows * c.vocab * 2);
    defer a.free(by_fixed);
    var fails: usize = 0;

    // the references and B1's checks: eager one-row steps and rounds (graphs only where a check names them)
    m.graphs = false;
    for (0..specs.len) |j| fails += try reference(&m, &x, j, logits);

    // 1. a prompt chunk in another slot: the same bits
    const base = specs.len;
    var prompts: usize = 0;
    for (specs, 0..) |sp, j| {
        _ = try prefill(&m, base + j, x.ids[j][0..sp.prompt]);
        try m.download(m.b.logits, logits);
        if (!same("prompt of stream {s} in slot {d}", .{ sp.name, base + j }, logits, x.prompt[j])) {
            std.debug.print("FAIL ripple-forward prompt of stream {s}: slot {d} != slot {d}\n", .{ sp.name, base + j, j });
            prompts += 1;
        }
    }
    if (prompts == 0) check.pass("ripple-forward prompts: {d} streams' last rows the same in their test slots", .{specs.len});
    fails += prompts;

    // 2. rounds of several streams, verify windows and mixes
    var done: [specs.len]usize = @splat(0);
    fails += try runRounds(&m, &x, base, &done, logits, rows, by_graph, by_fixed);

    // 3. a slot reused after reset: B's (a wrapped ring, 7 pages) takes D, all its decode rows one verify window
    const slot = base + 1;
    try m.reset(slot, specs[1].prompt + decodeTokens(1));
    const d = 3;
    const n = specs[d].prompt;
    const t = decodeTokens(d);
    _ = try prefill(&m, slot, x.ids[d][0..n]);
    try m.download(m.b.logits, logits);
    var bad: usize = @intFromBool(!same("reused slot's prompt", .{}, logits, x.prompt[d]));
    // the window from a graph first, then B1's eager window into `rows`
    const window_by_graph = try graphVsEager(&m, "reused slot window", &.{.{ .slot = slot, .p0 = n, .tokens = t }}, x.ids[d][n..][0..t], by_graph, rows);
    for (0..t) |r| {
        if (!same("reused slot row {d}", .{r}, rows[r * c.vocab * 2 ..][0 .. c.vocab * 2], x.want[d][r])) bad += 1;
    }
    if (bad == 0) {
        check.pass("ripple-forward reused slot: stream D after B's reset, prompt and a {d}-row window == one-row steps", .{t});
    } else {
        std.debug.print("FAIL ripple-forward reused slot: {d} rows differ\n", .{bad});
        fails += 1;
    }
    if (window_by_graph) {
        check.pass("ripple-forward graph, reused slot: a {d}-row window replayed from a graph == the eager window", .{t});
    } else {
        std.debug.print("FAIL ripple-forward graph, reused slot: a {d}-row window from a graph != the eager window\n", .{t});
        fails += 1;
    }

    // 4. the one-row graph (the CLI's decode) by `forward`: captured once, replayed across the wrap and a page edge
    const reused = base;
    try m.reset(reused, specs[0].prompt + decodeTokens(0));
    var seq = try prefill(&m, reused, x.ids[1][0..specs[1].prompt]);
    try m.download(m.b.logits, logits);
    var steps: usize = @intFromBool(!same("B's prompt in slot {d}", .{reused}, logits, x.prompt[1]));
    m.graphs = true;
    const first = seq.len;
    for (x.want[1], first..) |want, p| {
        try m.forward(&seq, x.ids[1][p..][0..1], .decode);
        try m.head(0);
        try m.download(m.b.logits, logits);
        if (!same("one-row graph step at {d}", .{p}, logits, want)) steps += 1;
    }
    m.graphs = false;
    if (steps == 0) {
        check.pass("ripple-forward one-row graph: captured in slot {d} at {d}, replayed to {d} (the ring's wrap, a page edge) == one-row eager steps", .{ reused, first, seq.len - 1 });
    } else {
        std.debug.print("FAIL ripple-forward one-row graph: {d} of {d} steps differ from one-row eager steps\n", .{ steps, x.want[1].len });
        fails += 1;
    }

    // 5. graphs captured above, replayed at other positions and slots
    var toks: [round_rows]u32 = undefined;
    var prng = std.Random.DefaultPrng.init(0x6772_6170);
    for (&toks) |*tk| tk.* = prng.random().uintLessThan(u32, @intCast(c.vocab));
    for (elsewhere) |r| {
        if (try replay(&m, r, &toks, by_graph, rows)) {
            check.pass("ripple-forward graph elsewhere: {s} == eager", .{r.what});
        } else {
            std.debug.print("FAIL ripple-forward graph elsewhere: {s} != eager\n", .{r.what});
            fails += 1;
        }
    }

    // 6. graphs of several row counts coexist: replayed in another order, none captured again
    const graphs = m.captured();
    var order: usize = 0;
    for (any_order) |r| order += @intFromBool(!try replay(&m, r, &toks, by_graph, rows));
    if (order == 0 and m.captured() == graphs) {
        check.pass("ripple-forward graphs coexist: {d} row counts' graphs, replayed as 48, 1, 17, 4, 1 and 8 rows == eager, none captured again", .{graphs});
    } else {
        std.debug.print("FAIL ripple-forward graphs coexist: {d} of {d} replays differ, {d} graphs before and {d} after\n", .{ order, any_order.len, graphs, m.captured() });
        fails += 1;
    }

    // 7. a drafter's taps on the same model (last: setTaps drops the graphs above)
    fails += try taps(&m, &w, &x);
    try check.expect(fails == 0, "ripple-forward: {d} checks differ", .{fails});
}
