//! The qwen35 caches (bf16, kv8) through a page table: rope, attention (every kernel), keep: paged == contiguous.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const numerical = @import("contract_qwen_pages_numerical.zig");

pub const hd = 256;
const max_chunks = 64;
const attentions = 2; // the fixture's attention layers: the kernels run layer 1, so a page's layer offset counts
pub const q_heads = 24; // the 27B's heads
pub const kv_heads = 4;
const group = q_heads / kv_heads;
const attn_tiles = 4 * 32 * hd * 2; // cuda_forward.zig's dynamic shared bytes for the bf16 tensor-core tiles
const prompt_shared = 98304; // cuda_prompt.zig's smem
/// cuda_prompt.zig's symbols: hpattn_kernel<256, 8, 2, 2> and hpattn8_kernel<256, 8, 2, 2>
const prompt_symbols = .{
    .bf16 = "_ZN20tf_prefill_attention13hpattn_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifiPKixi",
    .fp8 = "_ZN20tf_prefill_attention14hpattn8_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifiPKixi",
};
const e4m3 = core.quant.fp8.e4m3;

// qwen35.cuh's Geo, Seg, Ancestor and Path (src/families/qwen3_5/cuda_cache.zig mirrors them for the engine)
const Geo = extern struct { hid: c_int, nk: c_int, nv: c_int, qh: c_int, kvh: c_int, nl: c_int };
const Seg = extern struct { rec: u64 = 0, conv: u64 = 0, kv: u64, tap: u64 = 0, log: u64 = 0, before: u64 = 0, cap: c_int, len: c_int, row0: c_int, rows: c_int, tap_rows: c_int = 0, taps: c_int = 0, delta: c_int = 0, pt: u64 = 0, ps: u64 = 0, ptok: c_int = 0 };
const Ancestor = extern struct { bits: u64, depth: u32, chain: u32 };
const Path = extern struct { n: c_int, rows: [64]c_int };

/// bf16 rows (qwen35.cu), or e4m3 rows with an fp32 scale each (qwen35_kv8.cu, --kv-cache fp8).
pub const Format = enum { bf16, fp8 };

/// Two streams' cached positions, the window rows each adds, positions a page.
pub const Case = struct { lens: [2]usize, rows: usize, ptok: usize, mode: numerical.Mode = .random };
const cases = [_]Case{
    .{ .lens = .{ 0, 3 }, .rows = 1, .ptok = 16 },
    .{ .lens = .{ 127, 128 }, .rows = 4, .ptok = 16 },
    .{ .lens = .{ 129, 1000 }, .rows = 8, .ptok = 128 },
    .{ .lens = .{ 700, 33 }, .rows = 2, .ptok = 48 }, // a page no chunk width divides
    .{ .lens = .{ 4095, 5000 }, .rows = 16, .ptok = 256 },
    .{ .lens = .{ 9000, 300 }, .rows = 3, .ptok = 512 },
    .{ .lens = .{ 31, 128 }, .rows = 128, .ptok = 48 },
    .{ .lens = .{ 32767, 32768 }, .rows = 1, .ptok = 256 },
    .{ .lens = .{ 4096, 8192 }, .rows = 1, .ptok = 48, .mode = .stress },
    .{ .lens = .{ 8, 16 }, .rows = 5, .ptok = 16, .mode = .nonfinite },
};

const Span = struct { off: usize, len: usize };

/// Bytes of a layer's (or a whole block's, `layers`) rows, `n` positions a head.
fn blockBytes(f: Format, n: usize, layers: usize) usize {
    return layers * 2 * kv_heads * n * @as(usize, if (f == .fp8) hd + 4 else hd * 2);
}

/// Layer a's key (s 0) or value (s 1) at pos, head h, in a block of n positions a head: its row, and fp8's scale.
fn spans(f: Format, n: usize, a: usize, s: usize, h: usize, pos: usize) [2]Span {
    const blk = s * kv_heads + h;
    const layer = blockBytes(f, n, a);
    return switch (f) {
        .bf16 => .{ .{ .off = layer + (blk * n + pos) * hd * 2, .len = hd * 2 }, .{ .off = 0, .len = 0 } },
        .fp8 => .{ .{ .off = layer + (blk * n + pos) * hd, .len = hd }, .{ .off = layer + 2 * kv_heads * n * hd + (blk * n + pos) * 4, .len = 4 } },
    };
}

/// One case's host data: the logical caches (cap rows a head), the window's projections, the page tables.
pub const Fixture = struct {
    c: Case,
    format: Format,
    cap: usize,
    kv: [2][]u8,
    q: []u16, // [rows][q_heads][hd]
    k: []u16, // [rows][kv_heads][hd]
    v: []u16,
    ids: [2][]u32, // each stream's pages in the shared arena, shuffled across both
    pages: usize, // the arena's pages (two spare)

    fn streamBytes(f: Fixture) usize {
        return blockBytes(f.format, f.cap, attentions);
    }

    fn pageBytes(f: Fixture) usize {
        return blockBytes(f.format, f.c.ptok, attentions);
    }

    pub fn logical(f: Fixture, a: usize, s: usize, h: usize, pos: usize) [2]Span {
        return spans(f.format, f.cap, a, s, h, pos);
    }

    /// Where logical (a, s, h, pos) of stream `st` sits in the arena under `ids`.
    fn paged(f: Fixture, ids: [2][]const u32, st: usize, a: usize, s: usize, h: usize, pos: usize) [2]Span {
        var r = spans(f.format, f.c.ptok, a, s, h, pos % f.c.ptok);
        for (&r) |*x| x.off += ids[st][pos / f.c.ptok] * f.pageBytes();
        return r;
    }

    fn make(a: std.mem.Allocator, c: Case, format: Format, seed: u64) !Fixture {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const most = @max(c.lens[0], c.lens[1]) + c.rows;
        const cap = (most + c.ptok - 1) / c.ptok * c.ptok;
        var f: Fixture = .{ .c = c, .format = format, .cap = cap, .kv = undefined, .q = undefined, .k = undefined, .v = undefined, .ids = undefined, .pages = 2 * (cap / c.ptok) + 2 };
        for (&f.kv) |*x| {
            x.* = try a.alloc(u8, f.streamBytes());
            for (0..attentions) |at| for (0..2) |s| for (0..kv_heads) |h| for (0..cap) |pos| fill(f, rng, x.*, f.logical(at, s, h, pos));
        }
        f.q = try randoms(a, rng, c.rows * 2 * q_heads * hd);
        f.k = try randoms(a, rng, c.rows * 2 * kv_heads * hd);
        f.v = try randoms(a, rng, c.rows * 2 * kv_heads * hd);
        const order = try a.alloc(u32, f.pages);
        for (order, 0..) |*o, i| o.* = @intCast(i);
        rng.shuffle(u32, order);
        const n = cap / c.ptok;
        f.ids = .{ order[0..n], order[n .. 2 * n] };
        return f;
    }

    /// A row's bytes: bf16 in [-1, 1); e4m3 codes but NaN's, a scale in [1/1792, 1/448) (values within 1).
    fn fill(f: Fixture, rng: std.Random, out: []u8, sp: [2]Span) void {
        switch (f.format) {
            .bf16 => for (std.mem.bytesAsSlice(u16, out[sp[0].off..][0..sp[0].len])) |*x| {
                x.* = @truncate(@as(u32, @bitCast(rng.float(f32) * 2 - 1)) >> 16);
            },
            .fp8 => {
                for (out[sp[0].off..][0..sp[0].len]) |*b| {
                    b.* = rng.int(u8);
                    if (b.* & 0x7f == 0x7f) b.* ^= 1;
                }
                const sc: f32 = (1 + 3 * rng.float(f32)) / 1792;
                @memcpy(out[sp[1].off..][0..4], std.mem.asBytes(&sc));
            },
        }
    }

    /// The arena holding both streams' logical caches at their pages under `ids`; spare pages random.
    fn arena(f: Fixture, a: std.mem.Allocator, rng: std.Random, ids: [2][]const u32) ![]u8 {
        const out = try a.alloc(u8, f.pages * f.pageBytes());
        rng.bytes(out);
        for (0..2) |st| for (0..attentions) |at| for (0..2) |s| for (0..kv_heads) |h| for (0..f.cap) |pos| {
            const from = f.logical(at, s, h, pos);
            const to = f.paged(ids, st, at, s, h, pos);
            for (from, to) |x, y| @memcpy(out[y.off..][0..y.len], f.kv[st][x.off..][0..x.len]);
        };
        return out;
    }

    /// The arena's pages read back as stream `st`'s logical cache.
    fn unpage(f: Fixture, a: std.mem.Allocator, arena_: []const u8, ids: [2][]const u32, st: usize) ![]u8 {
        const out = try a.alloc(u8, f.streamBytes());
        for (0..attentions) |at| for (0..2) |s| for (0..kv_heads) |h| for (0..f.cap) |pos| {
            const to = f.logical(at, s, h, pos);
            const from = f.paged(ids, st, at, s, h, pos);
            for (from, to) |x, y| @memcpy(out[y.off..][0..y.len], arena_[x.off..][0..x.len]);
        };
        return out;
    }

    /// Logical value d of (a, s, h, pos) in stream cache `cache`.
    pub fn value(f: Fixture, cache: []const u8, a: usize, s: usize, h: usize, pos: usize, d: usize) f64 {
        const sp = f.logical(a, s, h, pos);
        return switch (f.format) {
            .bf16 => bf(std.mem.readInt(u16, cache[sp[0].off + 2 * d ..][0..2], .little)),
            .fp8 => e4m3(cache[sp[0].off + d]) * @as(f64, @as(f32, @bitCast(std.mem.readInt(u32, cache[sp[1].off..][0..4], .little)))),
        };
    }
};

/// bf16 in [-1, 1): finite, so a sum's bytes are its order's alone.
fn randoms(a: std.mem.Allocator, rng: std.Random, n: usize) ![]u16 {
    const out = try a.alloc(u16, n);
    for (out) |*x| x.* = @truncate(@as(u32, @bitCast(rng.float(f32) * 2 - 1)) >> 16);
    return out;
}

fn bf(x: u16) f64 {
    return @as(f32, @bitCast(@as(u32, x) << 16));
}

/// One layout's results: the rotated queries, the attention rows (the prompt tiles' too, with `Tc`), the caches.
pub const Result = struct { q: []u8, out: []u8, prompt: []u8, after_rope: [2][]u8, after_keep: [2][]u8 };

const Fns = struct { rope: cuda.Function, partial: cuda.Function, merge: cuda.Function, compact: cuda.Function };
/// The engine's attention kernels: short streams' rows in `partial`, long streams' in `rows`, a chunk's in `prompt`.
const Tc = struct { partial: cuda.Function, rows: cuda.Function, prompt: cuda.Function, shared: u32 };

/// rope, attention (`fns.partial`, or `tc`'s kernels), keep on layer 1 of both streams: contiguous, or `tables`' pages.
fn launch(gpu: check.Gpu, s: cuda.Stream, fns: Fns, f: Fixture, tables: ?[2][]const u32, given: ?[2][]const u32, a: std.mem.Allocator, seed: u64, tc: ?Tc) !Result {
    var rig = rig_.Rig{ .gpu = gpu, .s = s };
    defer rig.deinit();
    var prng = std.Random.DefaultPrng.init(seed);
    const m = 2 * f.c.rows;
    const geo: Geo = .{ .hid = 0, .nk = 0, .nv = 0, .qh = q_heads, .kvh = kv_heads, .nl = 0 };
    var segs: [2]Seg = undefined;
    var kv: [2]u64 = undefined; // a stream's buffer, or the arena twice
    if (tables) |t| {
        const host = try f.arena(a, prng.random(), t);
        const ptr = try rig.scratch(host.len);
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = ptr, .len = host.len }).upload(0, host);
        kv = .{ ptr, ptr };
    } else for (0..2) |st| {
        kv[st] = try rig.scratch(f.streamBytes());
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = kv[st], .len = f.streamBytes() }).upload(0, f.kv[st]);
    }
    for (0..2) |st| {
        segs[st] = .{ .kv = kv[st], .cap = @intCast(f.cap), .len = @intCast(f.c.lens[st]), .row0 = @intCast(st * f.c.rows), .rows = @intCast(f.c.rows) };
        if (given orelse tables) |t| {
            segs[st].pt = try rig.input(std.mem.sliceAsBytes(t[st]));
            segs[st].ps = if (f.format == .fp8) f.pageBytes() else f.pageBytes() / 2;
            segs[st].ptok = @intCast(f.c.ptok);
        }
    }
    const rowseg = try a.alloc(i32, m);
    const rowanc = try a.alloc(Ancestor, m);
    for (0..m) |r| {
        rowseg[r] = @intCast(r / f.c.rows);
        rowanc[r] = .{ .bits = 0, .depth = @intCast(r % f.c.rows), .chain = 1 };
    }
    const segs_d = try rig.input(std.mem.sliceAsBytes(&segs));
    const rowseg_d = try rig.input(std.mem.sliceAsBytes(rowseg));
    const rowanc_d = try rig.input(std.mem.sliceAsBytes(rowanc));
    const qn = try rig.scratch(f.q.len * 2); // rotated in place
    try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = qn, .len = f.q.len * 2 }).upload(0, std.mem.sliceAsBytes(f.q));
    const kn = try rig.input(std.mem.sliceAsBytes(f.k));
    const vr = try rig.input(std.mem.sliceAsBytes(f.v));
    const slots = m * q_heads * max_chunks;
    const pm = try rig.scratch(slots * 4);
    const pl = try rig.scratch(slots * 4);
    const po = try rig.scratch(slots * hd * 4);
    const out = try rig.output(m * q_heads * hd * 2);
    const before = try a.dupe(u8, try rig.inputBytes());
    const layer: c_int = 1;
    const none: u64 = 0;
    var args: cuda.Args = .{};
    inline for (.{ geo, qn, kn, vr, segs_d, rowseg_d, rowanc_d, layer, none, none }) |x| args.add(x);
    try cuda.launch.launch(fns.rope, .{ .grid = .{ .x = @intCast(m), .y = q_heads + kv_heads }, .block = .{ .x = hd } }, s, &args);
    args = .{};
    inline for (.{ geo, qn, segs_d, rowseg_d, rowanc_d, layer, none, @as(f32, 1.0 / 16.0), pm, pl, po }) |x| args.add(x);
    if (tc) |t| {
        // cuda_forward.zig's decode launches: every row through partial (long streams' rows return), then rows
        try cuda.launch.launch(t.partial, .{ .grid = .{ .x = @intCast(m), .y = max_chunks, .z = kv_heads }, .block = .{ .x = 128 }, .shared = t.shared }, s, &args);
        var rows_args: cuda.Args = .{};
        inline for (.{ geo, qn, segs_d, rowseg_d, rowanc_d, layer, none, @as(f32, 1.0 / 16.0), @as(c_int, @intCast(m)), pm, pl, po }) |x| rows_args.add(x);
        const groups = (m + 64 / group - 1) / (64 / group);
        try cuda.launch.launch(t.rows, .{ .grid = .{ .x = @intCast(groups), .y = max_chunks, .z = kv_heads }, .block = .{ .x = 256 }, .shared = t.shared }, s, &rows_args);
    } else try cuda.launch.launch(fns.partial, .{ .grid = .{ .x = max_chunks, .y = kv_heads, .z = @intCast(m) }, .block = .{ .x = 32 * group } }, s, &args);
    args = .{};
    inline for (.{ geo, pm, pl, po, segs_d, rowseg_d, rowanc_d, none, out }) |x| args.add(x);
    try cuda.launch.launch(fns.merge, .{ .grid = .{ .x = q_heads, .y = @intCast(m) }, .block = .{ .x = 32 } }, s, &args);
    // each stream's rows as one prompt chunk over its own cache (cuda_prompt.zig's launch): a slab's or a page's block
    const pout = try rig.output(m * q_heads * hd * 2);
    if (tc) |t| for (0..2) |st| {
        const n: u64 = if (tables != null) f.c.ptok else f.cap;
        const keys = kv[st] + blockBytes(f.format, @intCast(n), 1);
        var pa: cuda.Args = .{};
        pa.add(qn + st * f.c.rows * q_heads * hd * 2);
        if (f.format == .fp8) {
            const values = keys + kv_heads * n * hd;
            const kscale = values + kv_heads * n * hd;
            inline for (.{ keys, values, kscale, kscale + kv_heads * n * 4 }) |x| pa.add(x);
        } else inline for (.{ keys, keys + kv_heads * n * hd * 2 }) |x| pa.add(x);
        pa.add(pout + st * f.c.rows * q_heads * hd * 2);
        inline for (.{ f.c.lens[st], f.c.rows, q_heads, kv_heads, group }) |v| pa.add(@as(c_int, @intCast(v)));
        pa.add(@as(f32, 1.0 / 16.0));
        pa.add(@as(c_int, @intCast(f.cap)));
        pa.add(segs[st].pt);
        pa.add(@as(i64, @intCast(segs[st].ps)));
        pa.add(@as(c_int, if (segs[st].pt != 0) @intCast(f.c.ptok) else 1));
        try cuda.launch.launch(t.prompt, .{ .grid = .{ .x = @intCast((f.c.rows + 63) / 64), .y = kv_heads * (group / 2) }, .block = .{ .x = 256 }, .shared = prompt_shared }, s, &pa);
    };
    var r: Result = undefined;
    r.q = try a.dupe(u8, try rig.bytes(qn, f.q.len * 2));
    r.out = try a.dupe(u8, try rig.bytes(out, m * q_heads * hd * 2));
    r.prompt = if (tc != null) try a.dupe(u8, try rig.bytes(pout, m * q_heads * hd * 2)) else &.{};
    try readCaches(&rig, a, f, tables, kv, &r.after_rope);
    // keep: stream 0's window rows 0, 2 and the last (a tree's path), moved down after its cache
    if (f.c.rows >= 3) {
        var path: Path = .{ .n = 3, .rows = @splat(0) };
        path.rows[1] = 2;
        path.rows[2] = @intCast(f.c.rows - 1);
        args = .{};
        inline for (.{ geo, segs[0], path }) |x| args.add(x);
        try cuda.launch.launch(fns.compact, .{ .grid = .{ .x = attentions, .y = kv_heads }, .block = .{ .x = hd } }, s, &args);
    }
    try readCaches(&rig, a, f, tables, kv, &r.after_keep);
    try check.expect(try rig.intact() and !rig.overlap() and std.mem.eql(u8, before, try rig.inputBytes()), "qwen35 pages: a launch changed an input or guard", .{});
    return r;
}

fn readCaches(rig: *rig_.Rig, a: std.mem.Allocator, f: Fixture, tables: ?[2][]const u32, kv: [2]u64, out: *[2][]u8) !void {
    if (tables) |t| {
        const all = try a.dupe(u8, try rig.bytes(kv[0], f.pages * f.pageBytes()));
        for (0..2) |st| out[st] = try f.unpage(a, all, t, st);
    } else for (0..2) |st| out[st] = try a.dupe(u8, try rig.bytes(kv[st], f.streamBytes()));
}

/// Each attention row against float64 softmax(q k / 16) v: a bound (fp32 and fast exp), bf16 rounding of its size.
fn agrees(a: std.mem.Allocator, f: Fixture, r: Result) !bool {
    const q: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, r.q));
    const out: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, r.out));
    const keys_most = @max(f.c.lens[0], f.c.lens[1]) + f.c.rows;
    const p = try a.alloc(f64, keys_most);
    defer a.free(p);
    const vals = try a.alloc(f64, keys_most * hd);
    defer a.free(vals);
    for (0..2 * f.c.rows) |row| {
        const st = row / f.c.rows;
        const keys = f.c.lens[st] + row % f.c.rows + 1; // the chain's rows are positions after the cache
        const cache = r.after_rope[st];
        for (0..q_heads) |qh| {
            const h = qh / (q_heads / kv_heads);
            const qv = q[(row * q_heads + qh) * hd ..][0..hd];
            var top: f64 = -std.math.inf(f64);
            for (0..keys) |t| {
                var dot: f64 = 0;
                for (0..hd) |d| dot += bf(qv[d]) * f.value(cache, 1, 0, h, t, d);
                p[t] = dot / 16.0;
                top = @max(top, p[t]);
                for (0..hd) |d| vals[t * hd + d] = f.value(cache, 1, 1, h, t, d);
            }
            var sum: f64 = 0;
            for (p[0..keys]) |*x| {
                x.* = @exp(x.* - top);
                sum += x.*;
            }
            for (0..hd) |d| {
                var o: f64 = 0;
                var mag: f64 = 0;
                for (0..keys) |t| {
                    o += p[t] * vals[t * hd + d];
                    mag += p[t] * @abs(vals[t * hd + d]);
                }
                o /= sum;
                mag /= sum;
                const got = bf(out[(row * q_heads + qh) * hd + d]);
                if (!(@abs(got - o) <= std.math.ldexp(@as(f64, 1), -8) * (@abs(o) + mag))) return false;
            }
        }
    }
    return true;
}

fn checked(a: std.mem.Allocator, f: Fixture, r: Result) !bool {
    return if (f.c.mode == .random) agrees(a, f, r) else numerical.agrees(a, f, r, f.c.mode);
}

fn eqCaches(x: [2][]u8, y: [2][]u8) bool {
    return std.mem.eql(u8, x[0], y[0]) and std.mem.eql(u8, x[1], y[1]);
}

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.qwen35);
    defer mod.unload();
    var keep = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_keep);
    defer keep.unload();
    var kv8 = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_kv8);
    defer kv8.unload();
    var prompt = try cuda.Module.load(gpu.d, cuda.kernels.prefill_attention);
    defer prompt.unload();
    const merge = try mod.function("tf_q35_attn_merge");
    for (std.enums.values(Format)) |format| {
        const fns: Fns = switch (format) {
            .bf16 => .{ .rope = try mod.function("tf_q35_rope"), .partial = try mod.function("tf_q35_attn_partial_paged"), .merge = merge, .compact = try keep.function("tf_q35k_compact") },
            .fp8 => .{ .rope = try kv8.function("tf_q35_rope8"), .partial = try kv8.function("tf_q35_attn_partial8_paged"), .merge = merge, .compact = try kv8.function("tf_q35k_compact8") },
        };
        var flat_fns = fns;
        flat_fns.partial = try if (format == .fp8) kv8.function("tf_q35_attn_partial8_contiguous") else mod.function("tf_q35_attn_partial_contiguous");
        const tc: Tc = switch (format) {
            .bf16 => .{ .partial = try mod.function("tf_q35_attn_partial"), .rows = try mod.function("tf_q35_attn_rows"), .prompt = try prompt.function(prompt_symbols.bf16), .shared = attn_tiles },
            .fp8 => .{ .partial = try kv8.function("tf_q35_attn_partial8"), .rows = try kv8.function("tf_q35_attn_rows8"), .prompt = try prompt.function(prompt_symbols.fp8), .shared = 0 },
        };
        if (format == .bf16) {
            try tc.partial.allowDynamicShared(attn_tiles);
            try tc.rows.allowDynamicShared(attn_tiles);
        }
        try tc.prompt.allowDynamicShared(prompt_shared);
        var red_tc = false;
        var red_table = false;
        var red_bound = false;
        for (cases, 0..) |c, i| {
            var arena = std.heap.ArenaAllocator.init(gpu.gpa);
            defer arena.deinit();
            const a = arena.allocator();
            var f = try Fixture.make(a, c, format, 0x9a6e + i);
            try numerical.fill(&f, c.mode);
            const flat = try launch(gpu, s, flat_fns, f, null, null, a, 1, null);
            const old = try launch(gpu, s, fns, f, null, null, a, 1, null);
            try check.sameBytes("qwen35 pages: contiguous specialization against general entry", flat.out, old.out);
            try check.expect(try checked(a, f, flat), "qwen35 pages {t}: contiguous attention off float64 at lens {any}, rows {d}", .{ format, c.lens, c.rows });
            for (0..5) |rep| {
                var repeated = std.heap.ArenaAllocator.init(gpu.gpa);
                defer repeated.deinit();
                const pg = try launch(gpu, s, fns, f, .{ f.ids[0], f.ids[1] }, null, repeated.allocator(), 2 + rep, null);
                try check.sameBytes("qwen35 pages: rotated queries", pg.q, flat.q);
                try check.sameBytes("qwen35 pages: attention rows, paged against contiguous", pg.out, flat.out);
                try check.expect(eqCaches(pg.after_rope, flat.after_rope), "qwen35 pages {t}: rope's stores differ at lens {any}, page {d}", .{ format, c.lens, c.ptok });
                try check.expect(eqCaches(pg.after_keep, flat.after_keep), "qwen35 pages {t}: keep's moves differ at lens {any}, page {d}", .{ format, c.lens, c.ptok });
            }
            // red controls: two of stream 1's pages swapped in the table only (the pages stay); a planted row
            const n1 = (c.lens[1] + c.rows + c.ptok - 1) / c.ptok;
            if (n1 >= 2) {
                const swapped = try a.dupe(u32, f.ids[1]);
                std.mem.swap(u32, &swapped[0], &swapped[n1 - 1]);
                const bad = try launch(gpu, s, fns, f, .{ f.ids[0], f.ids[1] }, .{ f.ids[0], swapped }, a, 9, null);
                const moved = !std.mem.eql(u8, bad.out, flat.out);
                red_table = red_table or moved;
                try check.expect(moved, "qwen35 pages {t}: a swapped page table still gave contiguous bytes (control missed) at lens {any}", .{ format, c.lens });
            }
            // the engine's kernels: paged == contiguous; decode rows off float64 (the prompt tiles' bound: the stack's)
            const flat_tc = try launch(gpu, s, flat_fns, f, null, null, a, 1, tc);
            if (c.mode == .random) try check.expect(try agrees(a, f, flat_tc), "qwen35 pages {t}: tensor-core attention off float64 at lens {any}, rows {d}", .{ format, c.lens, c.rows });
            for (0..2) |rep| {
                var repeated = std.heap.ArenaAllocator.init(gpu.gpa);
                defer repeated.deinit();
                const pg = try launch(gpu, s, fns, f, .{ f.ids[0], f.ids[1] }, null, repeated.allocator(), 12 + rep, tc);
                try check.sameBytes("qwen35 pages: tensor-core rows, paged against contiguous", pg.out, flat_tc.out);
                try check.sameBytes("qwen35 pages: prompt rows, paged against contiguous", pg.prompt, flat_tc.prompt);
            }
            if (n1 >= 2) {
                const swapped = try a.dupe(u32, f.ids[1]);
                std.mem.swap(u32, &swapped[0], &swapped[n1 - 1]);
                const bad = try launch(gpu, s, fns, f, .{ f.ids[0], f.ids[1] }, .{ f.ids[0], swapped }, a, 9, tc);
                const moved = !std.mem.eql(u8, bad.out, flat_tc.out) and !std.mem.eql(u8, bad.prompt, flat_tc.prompt);
                red_tc = red_tc or moved;
                try check.expect(moved, "qwen35 pages {t}: a swapped page table still gave the tensor-core or prompt tiles contiguous bytes (control missed) at lens {any}", .{ format, c.lens });
            }
            const o = try a.dupe(u8, flat.out);
            std.mem.writeInt(u16, o[2 * (q_heads * hd + 5) ..][0..2], 0x4040, .little); // 3.0 where |o| <= 1
            var bad_r = flat;
            bad_r.out = o;
            const caught = !(try checked(a, f, bad_r));
            red_bound = red_bound or caught;
            try check.expect(caught, "qwen35 pages {t}: a planted attention row passed float64 (control missed)", .{format});
            std.debug.print("PASS qwen35 pages {t}: lens {any}, {d} rows a stream, {d}-position pages: paged == contiguous (queries, rows, stores, keep; tensor-core and prompt rows), repeats, float64 bound, guards\n", .{ format, c.lens, c.rows, c.ptok });
        }
        try check.expect(red_table and red_bound and red_tc, "qwen35 pages {t}: a control never fired", .{format});
        std.debug.print("PASS qwen35 pages {t}: controls RED as planted (swapped table, planted row)\n", .{format});
    }
}
