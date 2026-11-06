//! Ripple's forward rounds == one-row steps, bit for bit (ripple-forward): streams, verify windows, graphs, reuse.
const std = @import("std");
const cuda = @import("cuda");
const rc = @import("ricursive_cuda");
const check = @import("check.zig");

const fwd = rc.model;
const wts = rc.weights;
const Config = @FieldType(wts.Weights, "config");
const Weight = cuda.qmmf.Weight;

/// The release's attention and widths, so every lane matmul takes the release's plan (tile, K slices, cluster).
pub fn config() Config {
    var c: Config = .{ .hidden = 5120, .intermediate = 12288, .layers = 10, .heads = 80, .kv_heads = 8, .vocab = 16384, .eps = 1e-6, .theta = 10000, .window = 1024, .embed_scale = 71.55417527999327, .clamp = 20, .context = 8192, .fp4_mlp = true };
    for (0..c.layers) |i| c.global[i] = i % 5 == 4;
    return c;
}

/// bf16 bits of x, rounded to nearest even.
fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

const Build = struct {
    w: *wts.Weights,
    d: *const cuda.Driver,
    r: std.Random,

    fn put(b: Build, bytes: []const u8) !u64 {
        var buf = try cuda.DeviceBuffer.fromHost(b.d, bytes);
        errdefer buf.free();
        try b.w.buffers.append(b.w.gpa, buf);
        b.w.bytes += bytes.len;
        return buf.ptr;
    }

    /// `n` bf16 values of magnitude in [2^e, 2^(e+1)), random signs and mantissas (fast: the release's sizes).
    fn values(b: Build, n: usize, e: i32) ![]u16 {
        const v = try b.w.gpa.alloc(u16, n);
        const exp: u16 = @intCast((e + 127) << 7);
        var bits: u64 = 0;
        for (v, 0..) |*x, i| {
            if (i % 4 == 0) bits = b.r.int(u64);
            x.* = exp | @as(u16, @truncate(bits & 0x807f));
            bits >>= 16;
        }
        return v;
    }

    /// A norm's or the embedding's bf16 values, normal with deviation `sd` around `mean`.
    fn small(b: Build, n: usize, mean: f32, sd: f32) !u64 {
        const v = try b.w.gpa.alloc(u16, n);
        defer b.w.gpa.free(v);
        for (v) |*x| x.* = bf16(mean + sd * b.r.floatNorm(f32));
        return b.put(std.mem.sliceAsBytes(v));
    }

    /// A bf16 [n, k] projection packed for the lane and prompt matmuls, its values about 1 / sqrt(k).
    fn dense(b: Build, n: usize, k: usize) !Weight {
        const src = try b.values(n * k, -@as(i32, std.math.log2_int(usize, k) / 2) - 1);
        defer b.w.gpa.free(src);
        const out = try b.w.gpa.alloc(u16, cuda.modelopt.padded(n) * k);
        defer b.w.gpa.free(out);
        cuda.qmmf.packBf16(out, src, n, k);
        return cuda.modelopt.bf16Weight(try b.put(std.mem.sliceAsBytes(out)), n, k);
    }

    /// An NVFP4 [n, k] projection: random nibbles, E4M3 scales of 1 to 3, a global of 1/1024 (about 1 / sqrt(k)).
    fn fp4(b: Build, n: usize, k: usize) !Weight {
        const gpa = b.w.gpa;
        const codes = try gpa.alloc(u8, n * k / 2);
        defer gpa.free(codes);
        b.r.bytes(codes);
        const scales = try gpa.alloc(u8, n * k / 16);
        defer gpa.free(scales);
        const e4m3 = [_]u8{ 0x38, 0x3c, 0x40, 0x44 };
        b.r.bytes(scales);
        for (scales) |*x| x.* = e4m3[x.* & 3];
        const words = try gpa.alloc(u32, cuda.nvfp4.padded(n) / 64 * (k / 64) * 512);
        defer gpa.free(words);
        const tiles = try gpa.alloc(u8, cuda.nvfp4.padded(n) * (k / 16));
        defer gpa.free(tiles);
        cuda.nvfp4.packWords(words, codes, n, k);
        cuda.nvfp4.packScales(tiles, scales, n, k);
        return cuda.nvfp4.weight(try b.put(std.mem.sliceAsBytes(words)), try b.put(tiles), 1.0 / 1024.0, n, k);
    }
};

/// Random weights in the loader's layout (cuda_weights.load): q, k and g one projection, NVFP4 MLPs.
pub fn build(gpu: check.Gpu, c: Config, seed: u64) !wts.Weights {
    var w: wts.Weights = .{ .gpa = gpu.gpa, .config = c };
    errdefer w.deinit();
    var prng = std.Random.DefaultPrng.init(seed);
    const b: Build = .{ .w = &w, .d = gpu.d, .r = prng.random() };
    const h = c.hidden;
    const q = c.queryWidth();
    const kv = c.kvWidth();
    w.embed = try b.small(c.vocab * h, 0, 0.02);
    w.head = try b.dense(c.vocab, h);
    w.norm = try b.small(h, 1, 0.1);
    w.layers = try gpu.gpa.alloc(wts.Layer, c.layers);
    for (w.layers) |*l| l.* = .{
        .qkg = try b.dense(2 * q + kv, h),
        .v = try b.dense(kv, h),
        .o = try b.dense(h, q),
        .gate = try b.fp4(c.intermediate, h),
        .up = try b.fp4(c.intermediate, h),
        .down = try b.fp4(h, c.intermediate),
        .input_norm = try b.small(h, 1, 0.1),
        .post_attn = try b.small(h, 1, 0.1),
        .pre_ffn = try b.small(h, 1, 0.1),
        .post_ffn = try b.small(h, 1, 0.1),
        .q_norm = try b.small(128, 1, 0.1),
        .k_norm = try b.small(128, 1, 0.1),
        .blend = wts.blendOf(bf16(b.r.float(f32) * 2 - 1)),
    };
    return w;
}

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
    try check.expect(fails == 0, "ripple-forward: {d} checks differ", .{fails});
}
