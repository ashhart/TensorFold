//! TF_QWEN_PROMPT_SPLITK's passes against hpattn8(_whole)_kernel<256, 8, 2, 2> on the same rows: every byte equal.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const qp = @import("contract_qwen_pages.zig");

const hd = qp.hd;
const q_heads = qp.q_heads;
const kv_heads = qp.kv_heads;
const group = q_heads / kv_heads;
const f8: qp.Format = .fp8;
const layer = 1; // the second of the fixture's two attention layers, as qwen-pages runs
const tile_shared = 98304; // cuda_prompt.zig's smem
// cuda_prompt_split.zig's launch: threads, shared bytes of the score and chain passes, segments a key head at most
const threads = 32 * group;
const scores_shared = (2 * 32 * 32 + group * 16 * 32) * 16;
const chain_fragments = 2;
const chain_shared = 4 * 64 * chain_fragments * 16;
const most_segments = 88;

const sym = struct {
    const geo = "PK13__nv_bfloat16PKhS5_PKfS7_iiiiifiPKixiii";
    const tile = [2][:0]const u8{ "_ZN20tf_prefill_attention20hpattn8_whole_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifi", "_ZN20tf_prefill_attention14hpattn8_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifiPKixi" };
    const maxima = [2][:0]const u8{ "_ZN16tf_prefill_split13scores_kernelILi0ELb0EEEv" ++ geo ++ "iiPfS7_PjSA_", "_ZN16tf_prefill_split13scores_kernelILi0ELb1EEEv" ++ geo ++ "iiPfS7_PjSA_" };
    const weights = [2][:0]const u8{ "_ZN16tf_prefill_split13scores_kernelILi1ELb0EEEv" ++ geo ++ "iiPfS7_PjSA_", "_ZN16tf_prefill_split13scores_kernelILi1ELb1EEEv" ++ geo ++ "iiPfS7_PjSA_" };
    const chain = [2][:0]const u8{ "_ZN16tf_prefill_split12chain_kernelILb0EEEv" ++ geo ++ "iPKjS7_S7_PS1_", "_ZN16tf_prefill_split12chain_kernelILb1EEEv" ++ geo ++ "iPKjS7_S7_PS1_" };
    const scan = "_ZN16tf_prefill_split11scan_kernelEPKfPfS2_iii";
};
/// Each kernel twice: over a whole window (0) and over pages (1).
const Fns = struct { tile: [2]cuda.Function, maxima: [2]cuda.Function, weights: [2]cuda.Function, chain: [2]cuda.Function, scan: cuda.Function };

const Mode = enum { random, nonfinite };
/// Cached positions, positions a page, and the chunk sizes run over that cache.
const Case = struct { len: usize, ptok: usize, rows: []const usize, mode: Mode = .random };
const cases = [_]Case{
    .{ .len = 0, .ptok = 16, .rows = &.{ 1, 2, 16, 17, 64 } },
    .{ .len = 62, .ptok = 48, .rows = &.{ 1, 2, 3, 7 } },
    .{ .len = 63, .ptok = 16, .rows = &.{ 1, 15, 16 } },
    .{ .len = 127, .ptok = 128, .rows = &.{ 1, 31, 33 } },
    .{ .len = 4095, .ptok = 256, .rows = &.{ 1, 32, 48 } },
    .{ .len = 4096, .ptok = 48, .rows = &.{ 5, 63, 64 } },
    .{ .len = 5568, .ptok = 512, .rows = &.{ 1, 33, 64 } },
    .{ .len = 5631, .ptok = 256, .rows = &.{ 1, 2, 7 } },
    .{ .len = 33000, .ptok = 256, .rows = &.{ 1, 7, 20 } },
    .{ .len = 140000, .ptok = 256, .rows = &.{ 1, 7, 64 } },
    .{ .len = 300, .ptok = 16, .rows = &.{ 5, 20 }, .mode = .nonfinite },
};

/// What a split run leaves out or changes: nothing, the last key segment, or the scan.
const Plant = enum { none, short, unscanned };

/// One case's device data: queries, the cache as one window and as shuffled pages, the split passes' scratch.
const Staged = struct {
    c: Case,
    cap: usize,
    q: u64,
    kv: [2]u64,
    pt: u64,
    stats: u64,
    pa: u64,
    most: usize,

    fn pageBytes(s: Staged) usize {
        return qp.blockBytes(f8, s.c.ptok, qp.attentions);
    }
};

fn tilesOf(len: usize, rows: usize) usize {
    return (len + rows - 1) / 64 + 1;
}

/// Infinities and NaN among layer 1's key and value scales and the first row's queries.
fn plantNonFinite(c: Case, cap: usize, cache: []u8, q: []u16) void {
    const inf = std.math.inf(f32);
    const at = [_]struct { s: usize, h: usize, pos: usize, v: f32 }{
        .{ .s = 0, .h = 0, .pos = c.len / 2, .v = inf },
        .{ .s = 0, .h = 1, .pos = 0, .v = std.math.nan(f32) },
        .{ .s = 0, .h = 2, .pos = c.len + 1, .v = -inf },
        .{ .s = 1, .h = 3, .pos = 1, .v = std.math.nan(f32) },
        .{ .s = 1, .h = 2, .pos = c.len - 1, .v = inf },
    };
    for (at) |p| @memcpy(cache[qp.spans(f8, cap, layer, p.s, p.h, p.pos)[1].off..][0..4], std.mem.asBytes(&p.v));
    q[5 * hd + 3] = 0x7fc0; // head 5: NaN
    q[11 * hd + 7] = 0x7f80; // head 11: +Inf
    q[17 * hd + 9] = 0xff80; // head 17: -Inf
}

fn stage(rig: *rig_.Rig, a: std.mem.Allocator, c: Case, seed: u64) !Staged {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const most = std.mem.max(usize, c.rows);
    const cap = (c.len + most + c.ptok - 1) / c.ptok * c.ptok;
    const cache = try a.alloc(u8, qp.blockBytes(f8, cap, qp.attentions));
    for (0..qp.attentions) |at| for (0..2) |s| for (0..kv_heads) |h| for (0..cap) |pos| qp.Fixture.fill(f8, rng, cache, qp.spans(f8, cap, at, s, h, pos));
    const q = try qp.randoms(a, rng, most * q_heads * hd);
    if (c.mode == .nonfinite) plantNonFinite(c, cap, cache, q);
    // the same cache in pages of ptok positions, shuffled among two spare pages of random bytes
    const n = cap / c.ptok;
    const ids = try a.alloc(u32, n + 2);
    for (ids, 0..) |*x, i| x.* = @intCast(i);
    rng.shuffle(u32, ids);
    const page = qp.blockBytes(f8, c.ptok, qp.attentions);
    const arena = try a.alloc(u8, (n + 2) * page);
    rng.bytes(arena);
    for (0..qp.attentions) |at| for (0..2) |s| for (0..kv_heads) |h| for (0..cap) |pos| {
        const from = qp.spans(f8, cap, at, s, h, pos);
        const to = qp.spans(f8, c.ptok, at, s, h, pos % c.ptok);
        for (from, to) |x, y| @memcpy(arena[ids[pos / c.ptok] * page + y.off ..][0..y.len], cache[x.off..][0..x.len]);
    };
    const tiles = tilesOf(c.len, most);
    return .{
        .c = c,
        .cap = cap,
        .q = try rig.input(std.mem.sliceAsBytes(q)),
        .kv = .{ try rig.input(cache), try rig.input(arena) },
        .pt = try rig.input(std.mem.sliceAsBytes(ids[0..n])),
        .stats = try rig.scratch(3 * q_heads * tiles * 16 * 4),
        .pa = try rig.scratch(q_heads * tiles * 2048),
        .most = most,
    };
}

/// The arguments both kernels open with: queries, layer 1's keys, values and scales (a window's, or a page's).
fn cacheArgs(a: *cuda.Args, st: Staged, paged: bool) void {
    const n: u64 = if (paged) st.c.ptok else st.cap;
    const keys = st.kv[@intFromBool(paged)] + qp.blockBytes(f8, @intCast(n), layer);
    const values = keys + kv_heads * n * hd;
    const kscale = values + kv_heads * n * hd;
    inline for (.{ st.q, keys, values, kscale, kscale + kv_heads * n * 4 }) |x| a.add(x);
}

fn chunkArgs(a: *cuda.Args, st: Staged, rows: usize) void {
    inline for (.{ st.c.len, rows, q_heads, kv_heads, group }) |v| a.add(@as(c_int, @intCast(v)));
    a.add(@as(f32, 1.0 / 16.0));
    a.add(@as(c_int, @intCast(st.cap)));
}

/// cuda_prompt.zig's launch of the tile kernel over `rows` rows into `out`.
fn tile(s: cuda.Stream, fns: Fns, st: Staged, paged: bool, rows: usize, out: u64) !void {
    var a: cuda.Args = .{};
    cacheArgs(&a, st, paged);
    a.add(out);
    chunkArgs(&a, st, rows);
    if (paged) {
        a.add(st.pt);
        a.add(@as(i64, @intCast(st.pageBytes())));
        a.add(@as(c_int, @intCast(st.c.ptok)));
    }
    const grid: cuda.launch.Dim3 = .{ .x = @intCast((rows + 63) / 64), .y = kv_heads * (group / 2) };
    try cuda.launch.launch(fns.tile[@intFromBool(paged)], .{ .grid = grid, .block = .{ .x = 256 }, .shared = tile_shared }, s, &a);
}

fn splitArgs(a: *cuda.Args, st: Staged, paged: bool, rows: usize, r0: usize) void {
    cacheArgs(a, st, paged);
    chunkArgs(a, st, rows);
    a.add(@as(u64, if (paged) st.pt else 0));
    a.add(@as(i64, if (paged) @intCast(st.pageBytes()) else 0));
    a.add(@as(c_int, if (paged) @intCast(st.c.ptok) else 1));
    a.add(@as(c_int, @intCast(r0)));
    a.add(@as(c_int, @intCast(rows - 1)));
}

/// cuda_prompt_split.zig's launches over `rows` rows into `out`, with `plant` left out.
fn split(rig: *rig_.Rig, s: cuda.Stream, fns: Fns, st: Staged, paged: bool, rows: usize, out: u64, plant: Plant) !void {
    const pg: usize = @intFromBool(paged);
    const tiles = tilesOf(st.c.len, rows);
    const tps = (tiles + @min(tiles, most_segments) - 1) / @min(tiles, most_segments);
    const nseg = (tiles + tps - 1) / tps;
    const stat = q_heads * tiles * 16 * 4;
    const n = st.stats;
    const factors = n + stat;
    const sums = factors + stat;
    try rig.zero(st.stats, 3 * stat);
    try rig.zero(st.pa, q_heads * tiles * 2048);
    const z: u32 = @intCast(if (plant == .short) nseg - 1 else nseg);
    const scores_cfg: cuda.Config = .{ .grid = .{ .x = 1, .y = kv_heads, .z = z }, .block = .{ .x = threads }, .shared = scores_shared };
    var r0: usize = 0;
    while (r0 < rows) : (r0 += 16) {
        inline for (.{ fns.maxima[pg], fns.weights[pg] }, 0..) |pass, i| {
            if (i == 1 and plant != .unscanned) {
                var sa: cuda.Args = .{};
                inline for (.{ n, n, factors }) |x| sa.add(x);
                inline for (.{ q_heads, tiles, tiles }) |v| sa.add(@as(c_int, @intCast(v)));
                try cuda.launch.launch(fns.scan, .{ .grid = .{ .x = (q_heads * 16 + 127) / 128 }, .block = .{ .x = 128 } }, s, &sa);
            }
            var a: cuda.Args = .{};
            splitArgs(&a, st, paged, rows, r0);
            inline for (.{ tps, tiles }) |v| a.add(@as(c_int, @intCast(v)));
            inline for (.{ n, n, st.pa, sums }) |x| a.add(x);
            try cuda.launch.launch(pass, scores_cfg, s, &a);
        }
        var c: cuda.Args = .{};
        splitArgs(&c, st, paged, rows, r0);
        c.add(@as(c_int, @intCast(tiles)));
        inline for (.{ st.pa, sums, factors, out }) |x| c.add(x);
        const chain_cfg: cuda.Config = .{ .grid = .{ .x = kv_heads, .y = hd / 8 / chain_fragments }, .block = .{ .x = threads }, .shared = chain_shared };
        try cuda.launch.launch(fns.chain[pg], chain_cfg, s, &c);
    }
}

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.prefill_attention);
    defer mod.unload();
    var fns: Fns = undefined;
    for (0..2) |i| {
        fns.tile[i] = try mod.function(sym.tile[i]);
        fns.maxima[i] = try mod.function(sym.maxima[i]);
        fns.weights[i] = try mod.function(sym.weights[i]);
        fns.chain[i] = try mod.function(sym.chain[i]);
        try fns.tile[i].allowDynamicShared(tile_shared);
        try fns.maxima[i].allowDynamicShared(scores_shared);
        try fns.weights[i].allowDynamicShared(scores_shared);
    }
    fns.scan = try mod.function(sym.scan);
    var red_short = false;
    var red_scan = false;
    for (cases, 0..) |c, ci| {
        var arena = std.heap.ArenaAllocator.init(gpu.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        var rig = rig_.Rig{ .gpu = gpu, .s = s };
        defer rig.deinit();
        const st = try stage(&rig, a, c, 0x5b17 + ci);
        const bytes = st.most * q_heads * hd * 2;
        const want = try rig.output(bytes);
        const got = try rig.output(bytes);
        const before = try a.dupe(u8, try rig.inputBytes());
        for (c.rows) |rows| {
            const n = rows * q_heads * hd * 2;
            const tiles = tilesOf(c.len, rows);
            var whole: []u8 = &.{};
            for ([_]bool{ false, true }) |paged| {
                try tile(s, fns, st, paged, rows, want);
                const old = try a.dupe(u8, try rig.bytes(want, n));
                for (0..5) |_| {
                    try split(&rig, s, fns, st, paged, rows, got, .none);
                    try check.sameBytes("qwen35 prompt split: rows against the tile kernel", try rig.bytes(got, n), old);
                }
                if (!paged) whole = old else try check.sameBytes("qwen35 prompt split: paged against whole", old, whole);
                if (c.mode != .random or tiles < 2) continue;
                // red controls: the last key segment never launched; the weights taken before the scan
                try split(&rig, s, fns, st, paged, rows, got, .short);
                const short = !std.mem.eql(u8, try rig.bytes(got, n), old);
                try check.expect(short, "qwen35 prompt split: a missing key segment still gave the rows (control missed) at {d} keys, {d} rows", .{ c.len, rows });
                red_short = red_short or short;
                try split(&rig, s, fns, st, paged, rows, got, .unscanned);
                const unscanned = !std.mem.eql(u8, try rig.bytes(got, n), old);
                try check.expect(unscanned, "qwen35 prompt split: a skipped scan still gave the rows (control missed) at {d} keys, {d} rows", .{ c.len, rows });
                red_scan = red_scan or unscanned;
            }
        }
        try check.expect(try rig.intact() and !rig.overlap() and std.mem.eql(u8, before, try rig.inputBytes()), "qwen35 prompt split: a launch changed an input or guard", .{});
        std.debug.print("PASS qwen35 prompt split {t}: {d} keys, rows {any}, {d}-position pages: bytes of hpattn8 whole and paged, five repeats, guards\n", .{ c.mode, c.len, c.rows, c.ptok });
    }
    try check.expect(red_short and red_scan, "qwen35 prompt split: a control never fired", .{});
    std.debug.print("PASS qwen35 prompt split: controls RED as planted (missing key segment, skipped scan)\n", .{});
}
