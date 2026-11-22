//! One-rank, text-only Inkling 2 Mini: the complete 48-visit CUDA forward.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const loop = @import("loop.zig");
const weights_ = @import("loop_weights.zig");
const ops = @import("ops.zig");
const attention = @import("kernels.zig");
const modelopt = @import("modelopt.zig");
const loop_math = @import("loop_math_cuda.zig");
const bf12 = @import("bf12.zig");

pub const prompt_rows: usize = 2048;
pub const window_rows: usize = 16;
/// Stream slots a model holds (--parallel), and the rows one batched forward takes across them.
pub const max_segments: usize = 16;
pub const max_batch_rows: usize = 128;
comptime {
    std.debug.assert(window_rows == weights_.window_rows);
}

pub const Kernels = struct {
    glue: ops.Kernels,
    attn: attention.Kernels,
    moe: modelopt.Kernels,
    loop: loop_math.Kernels,
    /// TF_INKLING2_BF12: the packed dense rows' matmul
    b12: ?bf12.Kernels = null,
    sample_mod: cuda.Module,
    draws: cuda.sample.Draws,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        const d = ctx.d;
        var glue = try ops.Kernels.load(d);
        errdefer glue.unload();
        var attn = try attention.Kernels.load(d);
        errdefer attn.unload();
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        var moe = try modelopt.Kernels.load(d, sms);
        errdefer moe.unload();
        var lm = try loop_math.Kernels.load(d);
        errdefer lm.unload();
        if (modelopt.ommaEnabled() or modelopt.qmmaEnabled()) try moe.loadOmma(d, sms);
        var b12: ?bf12.Kernels = if (bf12.enabled()) try bf12.Kernels.load(d) else null;
        errdefer if (b12) |*b| b.unload();
        var sample_mod = try cuda.Module.load(d, cuda.kernels.sample);
        errdefer sample_mod.unload();
        return .{
            .glue = glue,
            .attn = attn,
            .moe = moe,
            .loop = lm,
            .b12 = b12,
            .sample_mod = sample_mod,
            .draws = try cuda.sample.Draws.resolve(sample_mod),
        };
    }

    pub fn deinit(k: *Kernels) void {
        k.sample_mod.unload();
        if (k.b12) |*b| b.unload();
        k.loop.unload();
        k.moe.unload();
        k.attn.unload();
        k.glue.unload();
        k.* = undefined;
    }
};

const Scratch = struct {
    ids: u64,
    pos: u64,
    seg_of: u64,
    segs: u64,
    seg_of_b: u64, // a batched call's row -> segment map: int [max_batch_rows] (seg_of stays all zeros)
    segs_b: u64, // a batched call's attention segments: Seg [max_segments]
    conv_segs: u64, // a batched call's conv segments: int4 [max_segments] (row0, rows, slot, 0)
    tiles: u64, // a wide or batched call's tiles: max_tile_groups slices of 128, staged once a forward (stageTiles)
    tiles1: u64, // decode (one row) tiles, built once: a table slot a distinct heads-per-key-head group
    blocks: [ops.max_blocks]u64 = @splat(0),
    // each completed block's RMS normalization (loop_math.prenorm), FP32 for ar() and for ops.mix, kept for window_rows rows
    normed_ar: [ops.max_blocks]u64 = @splat(0),
    normed_mix: [ops.max_blocks]u64 = @splat(0),
    partial_a: u64,
    partial_b: u64,
    prelude: u64,
    bank0: u64,
    bank1: u64,
    raw: u64, // three FP32 AR/gate rows per token
    messages: u64, // four FP32 message rows per token
    normalized: u64, // three BF16 rows per token
    gate_norm_hm: u64, // BF16 [rows, 2*h]: M1 fast path uses window_rows rows, M8a TC path prompt_rows
    gate_zeta: u64, // BF16 [rows, 64]: same sizing as gate_norm_hm (256 KB at prompt_rows)
    // gate3 (narrow rows, and a batched call's rows: max_batch_rows of them)
    gate3_norm_hm: u64, // gate3: BF16 [3][rows][2*h] normalized pairs, one role after another
    gate3_zeta: u64, // gate3: BF16 [3][rows][64] shared rank zetas
    gate3_raw: u64, // gate3: FP32 [3][rows][h] gated rows (each role its own: none overwrites another's state)
    gate3_bf16: u64, // gate3: BF16 [3][rows][h] normed rows (the projectQkv inputs)
    // the wide tensor-core gate's normalized halves (loop_math.gateTc3): ar's states BF16 [3][prompt_rows][h], the
    // messages' BF16 [prompt_rows][4][h], role zetas BF16 [3][prompt_rows][64], role 0's gated rows FP32 [prompt_rows][h]
    tc_hn: u64,
    tc_mn: u64,
    tc_zeta: u64,
    tc_raw0: u64,
    t: u64,
    q_raw: u64,
    rel_raw: u64,
    sink: u64,
    gate: u64,
    k_raw: u64,
    v_raw: u64,
    k_conv: u64,
    v_conv: u64,
    k_dynamic: u64,
    v_dynamic: u64,
    q_prepared: u64,
    rel_prepared: u64,
    part_o: u64,
    part_ml: u64,
    attn_heads: u64,
    attn_out: u64,
    qrg_raw: u64, // fused wq|wr|gate matmul output: BF16 [prompt_rows][2*q+rel], read in place by prep and attend
    route_scores: u64,
    route_pick: u64,
    route_weights: u64,
    moe_out: u64,
    moe_scratch: u64,
    hidden: u64,
    logits: u64,
    backup: u64,
    hist: u64,
    rewind_sites: u64,
    b12_scratch: u64 = 0, // bf16 rows a wide call unpacks a packed matrix into
    b12_part: u64 = 0, // split-K partials of the matmul
};

const max_tile_groups = 4;
const max_tile_rows = 16;
/// Rows gate3's scratch holds: a narrow call's, or a batched call's (exact for any rows: the scalar gate's bits).
const gate3_rows: usize = @max(window_rows, max_batch_rows);

/// The tiles of `rows` rows (`group` query heads a key head) in `tiles1`: they never depend on the rows' position.
const TileSlot = struct { group: u32, rows: u32, ptr: u64, n: u32 };

/// What a captured graph baked in: its kernels' modules, its stream, its taps, and its width/record shape. A batched
/// call's also its stream count (the conv grid) and each tile group's tile count (the attention grids): everything
/// else it reads (ids, positions, segments, conv slots, tiles) sits in device buffers uploaded before each launch.
const GraphKey = struct {
    kernels: ?*const Kernels = null,
    stream: usize = 0,
    taps: u64 = 0,
    rows: u32 = 0,
    record: bool = false,
    slot: u32 = 0,
    streams: u32 = 0,
    tiles: [max_tile_groups]u32 = @splat(0),
};

const GraphEntry = struct { key: GraphKey, exec: cuda.graph.Exec, used: u64 };

/// TF_INKLING2_GRAPH (default 1): decode replays one captured graph a token.
fn graphEnabled() bool {
    return @import("sys").environment.read(bool, "TF_INKLING2_GRAPH", true) catch true;
}

/// A call's tiles for each of `groups` (distinct), group i's in slice i*128 of `all` (and of the device buffer at
/// `base`): `slots[i]` says where and how many. Returns the tiles to upload (through the last group's).
fn callTiles(segs: []const attention.Seg, groups: []const u32, all: *[max_tile_groups * 128]attention.Tile, base: u64, slots: *[max_tile_groups]TileSlot) !usize {
    var used: usize = 0;
    for (groups, 0..) |group, i| {
        const n = try attention.tiles(segs, group, all[i * 128 ..][0..128]);
        slots[i] = .{ .group = group, .rows = 0, .ptr = base + i * 128 * @sizeOf(attention.Tile), .n = @intCast(n) };
        used = i * 128 + n;
    }
    return used;
}

/// A batched forward's host side: rows in segment order, each segment's attention and conv entries.
const Batch = struct {
    rows: usize,
    last_pos: u32,
    ids: [max_batch_rows]i32,
    pos: [max_batch_rows]i32,
    seg_of: [max_batch_rows]i32,
    segs: [max_segments]attention.Seg,
    conv: [max_segments][4]i32, // ops.cu's int4 conv segment: row0, rows, state slot, 0
};

/// Lay out `segs` (distinct slots below `slots`, each 1..window_rows rows inside `context`; at most max_batch_rows
/// rows in all) as one batched call. A stream's caches sit `slot * slot_bytes` past slot 0's: `off` counts bf16s.
fn batch(segs: []const Model.BatchedSeg, slots: usize, context: usize, slot_bytes: usize, record: bool, out: *Batch) !void {
    if (segs.len == 0 or segs.len > max_segments or slot_bytes % 4 != 0) return error.BadLoopRows;
    var used: [max_segments]bool = @splat(false);
    var rows: usize = 0;
    out.last_pos = 0;
    for (segs, 0..) |sg, i| {
        const n = sg.tokens.len;
        if (sg.slot >= slots or used[sg.slot]) return error.BadLoopRows;
        used[sg.slot] = true;
        // a recorded stream's hist region holds window_rows rows (rewind keeps a prefix of them)
        if (n == 0 or (record and n > window_rows) or sg.p0 + n > context) return error.BadLoopRows;
        if (rows + n > max_batch_rows) return error.BatchTooWide;
        for (sg.tokens, 0..) |token, j| {
            out.ids[rows + j] = @intCast(token);
            out.pos[rows + j] = @intCast(sg.p0 + j);
            out.seg_of[rows + j] = @intCast(i);
        }
        out.segs[i] = .{ .row0 = @intCast(rows), .n = @intCast(n), .p0 = @intCast(sg.p0), .off = @intCast(@as(usize, sg.slot) * slot_bytes / 2) };
        out.conv[i] = .{ @intCast(rows), @intCast(n), @intCast(sg.slot), 0 };
        out.last_pos = @max(out.last_pos, @as(u32, @intCast(sg.p0 + n - 1)));
        rows += n;
    }
    out.rows = rows;
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: *const weights_.Weights,
    context: usize,
    kv_int8: bool = false,
    layout: loop.CacheLayout,
    cache: cuda.DeviceBuffer,
    /// Stream slots 0..segments-1: the cache holds one layout.bytes region a slot; m.slot selects the
    /// calling stream's region for every state address (slot 0 is the legacy single-stream path).
    segments: usize = 1,
    slot: u32 = 0,
    s: Scratch = undefined,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    tile_table: [max_tile_rows * max_tile_groups]TileSlot = undefined,
    n_tile_groups: usize = 0,
    /// A wide or batched call's tiles (in m.s.tiles), one slot a distinct group, staged before its body runs.
    call_tiles: [max_tile_groups]TileSlot = undefined,
    n_call_tiles: usize = 0,
    /// TF_INKLING2_GRAPH: small LRU cache of captured decode/verify graphs, keyed by (kernels, stream,
    /// chunk buckets, taps, rows, record) so rows=1/record=false and rows=2..8/record=true stay cached together.
    graphs: [64]?GraphEntry = @splat(null),
    /// Batched (S > 1) forwards' graphs, their own LRU so a round's shapes never evict the lanes' decode graphs.
    batch_graphs: [32]?GraphEntry = @splat(null),
    graph_tick: u64 = 0,
    graph_refused: bool = false,
    /// Set for the length of a batched forward (forwardBatched over several streams), null otherwise.
    multi: ?Multi = null,
    /// TF_INKLING2_MULTI_EXACT: keep multi-stream forwards on the 64-row decode split-K pinning.
    multi_exact: bool = false,

    /// A batched forward's streams: their attention segments (caches `off` elements from slot 0's), in row order.
    const Multi = struct { n: u32, last_pos: u32, segs: [max_segments]attention.Seg };

    fn alloc(m: *Model, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(m.d, @max(bytes, 256));
        errdefer b.free();
        try m.buffers.append(m.gpa, b);
        return b.ptr;
    }

    fn upload(m: *const Model, s: cuda.Stream, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(m.d, dst, bytes, s);
    }

    fn copy(m: *const Model, s: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
        try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, s.handle), "cuMemcpyDtoDAsync");
    }

    fn zeros(m: *const Model, s: cuda.Stream, ptr: u64, bytes: usize) !void {
        try m.d.check(m.d.api.cuMemsetD8Async(ptr, 0, bytes, s.handle), "cuMemsetD8Async");
    }

    fn layerGeometry(m: *const Model, layer: u32) attention.Layer {
        return if (config.isGlobal(layer))
            attention.Layer.global(@intCast(m.context))
        else
            attention.Layer.sliding(m.w.c.text.window, @intCast(@min(prompt_rows, m.context)));
    }

    fn cacheOf(m: *const Model, visit: u32, slot: u32) attention.Cache {
        std.debug.assert(slot < m.segments);
        const vs = m.layout.at(visit);
        const base = m.cache.ptr + @as(usize, slot) * m.layout.bytes + vs.kv_offset;
        if (vs.kv_int8) {
            return .{
                .k = base,
                .v = base + vs.k_codes_bytes,
                .ks = base + 2 * vs.k_codes_bytes,
                .vs = base + 2 * vs.k_codes_bytes + vs.k_scales_bytes,
            };
        }
        return .{ .k = base, .v = base + vs.kv_bytes / 2 };
    }

    fn convOf(m: *const Model, visit: u32, site: usize, slot: u32) u64 {
        std.debug.assert(slot < m.segments);
        const c = m.w.c.text;
        const kv: usize = @as(usize, c.local.kv) * c.local.dim;
        const h: usize = c.hidden;
        const widths = [4]usize{ kv, kv, h, h };
        var at = m.cache.ptr + @as(usize, slot) * m.layout.bytes + m.layout.at(visit).conv_offset;
        for (0..site) |i| at += 3 * widths[i] * 4;
        return at;
    }

    /// Device bytes of one stream's convolution-rollback state: the backup copy plus the staged histories.
    fn backupBytes(m: *const Model) usize {
        return m.layout.bytes - m.layout.at(0).conv_offset;
    }

    fn histStride(m: *const Model) usize {
        const c = m.w.c.text;
        const kv: usize = @as(usize, c.local.kv) * c.local.dim;
        const h: usize = c.hidden;
        return @as(usize, m.w.p.n_visits) * window_rows * (2 * kv + 2 * h) * 4;
    }

    fn histOf(m: *const Model, visit: u32, site: usize, slot: u32) u64 {
        std.debug.assert(slot < m.segments);
        const c = m.w.c.text;
        const kv: usize = @as(usize, c.local.kv) * c.local.dim;
        const h: usize = c.hidden;
        const widths = [4]usize{ kv, kv, h, h };
        var at = m.s.hist + @as(usize, slot) * m.histStride() + visit * window_rows * (2 * kv + 2 * h) * 4;
        for (0..site) |i| at += window_rows * widths[i] * 4;
        return at;
    }

    fn backupOf(m: *const Model, visit: u32, site: usize, slot: u32) usize {
        return m.convOf(visit, site, slot) - (m.cache.ptr + @as(usize, slot) * m.layout.bytes + m.layout.at(0).conv_offset);
    }

    /// One conv site's launch over this call's rows: the calling stream's state and hist (m.slot), or, in a batched
    /// call, every segment on its own slot's state and hist region (ops.cu's conv segments).
    fn conv(m: *const Model, k: *const Kernels, s: cuda.Stream, args: ops.ConvArgs, cache_slot: u32, site: usize, record: bool) !void {
        var a = args;
        if (m.multi) |mu| {
            a.state = m.convOf(cache_slot, site, 0);
            a.hist = if (record) m.histOf(cache_slot, site, 0) else 0;
            a.segs = m.s.conv_segs;
            a.state_stride = @intCast(m.layout.bytes / 4);
            a.hist_stride = @intCast(m.histStride() / 4);
            return ops.conv(&k.glue, s, a, mu.n);
        }
        a.state = m.convOf(cache_slot, site, m.slot);
        a.hist = if (record) m.histOf(cache_slot, site, m.slot) else 0;
        return ops.conv(&k.glue, s, a, 1);
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const weights_.Weights, context: usize, segments: usize) !Model {
        return initMode(gpa, d, s, w, context, segments, false);
    }

    pub fn initMode(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const weights_.Weights, context: usize, segments: usize, kv_int8: bool) !Model {
        std.debug.assert(segments >= 1 and segments <= max_segments);
        const layout = try loop.CacheLayout.initMode(&w.c.text, &w.p, @intCast(context), kv_int8);
        var m: Model = .{
            .gpa = gpa,
            .d = d,
            .w = w,
            .context = context,
            .kv_int8 = kv_int8,
            .layout = layout,
            .segments = segments,
            // one region a stream slot; segments=1 allocates exactly the legacy single-stream size
            .cache = try cuda.DeviceBuffer.alloc(d, segments * layout.bytes),
        };
        m.multi_exact = @import("sys").environment.read(bool, "TF_INKLING2_MULTI_EXACT", false) catch false;
        errdefer m.cache.free();
        errdefer m.deinitBuffers();
        const c = w.c.text;
        const h: usize = c.hidden;
        const q: usize = @as(usize, c.global.q) * c.global.dim;
        const kv: usize = @as(usize, c.local.kv) * c.local.dim;
        const rows = prompt_rows;
        const local = attention.Layer.sliding(c.window, @intCast(@min(prompt_rows, context)));
        const global = attention.Layer.global(@intCast(context));
        const local_slots = attention.slotsFor(local, context);
        const global_slots = attention.slotsFor(global, context);
        const part_slots = @max(rows * c.local.q * local_slots, rows * c.global.q * global_slots);
        var max_qrg: usize = 0;
        for (m.w.p.layers[0..m.w.p.n]) |plan_layer| {
            const lq: usize = plan_layer.heads.q * plan_layer.heads.dim;
            max_qrg = @max(max_qrg, 2 * lq + plan_layer.heads.q * c.d_rel);
        }
        m.s = .{
            .ids = try m.alloc(rows * 4),
            .pos = try m.alloc(rows * 4),
            .seg_of = try m.alloc(rows * 4),
            .segs = try m.alloc(@sizeOf(attention.Seg)),
            .seg_of_b = try m.alloc(max_batch_rows * 4),
            .segs_b = try m.alloc(max_segments * @sizeOf(attention.Seg)),
            .conv_segs = try m.alloc(max_segments * 16),
            .tiles = try m.alloc(max_tile_groups * 128 * @sizeOf(attention.Tile)),
            .tiles1 = try m.alloc(max_tile_rows * max_tile_groups * 128 * @sizeOf(attention.Tile)),
            .partial_a = try m.alloc(rows * h * 2),
            .partial_b = try m.alloc(rows * h * 2),
            .prelude = try m.alloc(rows * h * 2),
            .bank0 = try m.alloc(rows * h * 2),
            .bank1 = try m.alloc(rows * h * 2),
            .raw = try m.alloc(3 * rows * h * 4),
            .messages = try m.alloc(4 * rows * h * 4),
            .normalized = try m.alloc(3 * rows * h * 2),
            .gate_norm_hm = try m.alloc(prompt_rows * 2 * h * 2),
            .gate_zeta = try m.alloc(prompt_rows * 64 * 2),
            .gate3_norm_hm = try m.alloc(3 * gate3_rows * 2 * h * 2),
            .gate3_zeta = try m.alloc(3 * gate3_rows * 64 * 2),
            .gate3_raw = try m.alloc(3 * gate3_rows * h * 4),
            .gate3_bf16 = try m.alloc(3 * gate3_rows * h * 2),
            .tc_hn = try m.alloc(3 * rows * h * 2),
            .tc_mn = try m.alloc(4 * rows * h * 2),
            .tc_zeta = try m.alloc(3 * rows * 64 * 2),
            .tc_raw0 = try m.alloc(rows * h * 4),
            .t = try m.alloc(rows * h * 4),
            .q_raw = try m.alloc(rows * q * 2),
            .rel_raw = try m.alloc(rows * c.global.q * c.d_rel * 2),
            .sink = try m.alloc(rows * c.global.q * 2),
            .gate = try m.alloc(rows * q * 2),
            .k_raw = try m.alloc(rows * kv * 2),
            .v_raw = try m.alloc(rows * kv * 2),
            .k_conv = try m.alloc(rows * kv * 2),
            .v_conv = try m.alloc(rows * kv * 2),
            .k_dynamic = try m.alloc(rows * c.global.kv * config.conv_taps * 4),
            .v_dynamic = try m.alloc(rows * c.global.kv * config.conv_taps * 4),
            .q_prepared = try m.alloc(rows * q * 2),
            .rel_prepared = try m.alloc(rows * c.global.q * c.d_rel * 2),
            .part_o = try m.alloc(part_slots * attention.head_dim * 4),
            .part_ml = try m.alloc(part_slots * 8),
            .attn_heads = try m.alloc(rows * q * 2),
            .attn_out = try m.alloc(rows * h * 2),
            .qrg_raw = try m.alloc(prompt_rows * max_qrg * 2),
            .route_scores = try m.alloc(rows * c.experts * 4),
            .route_pick = try m.alloc(rows * c.experts * 4),
            .route_weights = try m.alloc(rows * c.experts * 4),
            .moe_out = try m.alloc(rows * h * 2),
            .moe_scratch = try m.alloc(try modelopt.scratchBytes(rows, c.topk, h, c.expert_inter)),
            .hidden = try m.alloc(rows * h * 2),
            .logits = try m.alloc(max_batch_rows * c.unpadded_vocab * 4),
            .backup = try m.alloc(m.segments * m.backupBytes()),
            .hist = try m.alloc(m.segments * m.histStride()),
            .rewind_sites = try m.alloc(m.segments * w.p.n_visits * 4 * @sizeOf(ops.RewindSite)),
        };
        if (w.bf12_scratch_bytes > 0) m.s.b12_scratch = try m.alloc(w.bf12_scratch_bytes);
        if (w.bf12_part_floats > 0) m.s.b12_part = try m.alloc(w.bf12_part_floats * 4);
        for (0..ops.max_blocks) |i| m.s.blocks[i] = try m.alloc(rows * h * 2);
        for (0..ops.max_blocks) |i| {
            m.s.normed_ar[i] = try m.alloc(window_rows * h * 4);
            m.s.normed_mix[i] = try m.alloc(window_rows * h * 4);
        }
        try m.buildDecodeTiles(s);
        // every row belongs to the one stream: segment zero, for any call, so it is written once
        try m.zeros(s, m.s.seg_of, rows * 4);
        const sites = try gpa.alloc(ops.RewindSite, m.segments * w.p.n_visits * 4);
        defer gpa.free(sites);
        for (0..m.segments) |slot| {
            const base_idx = slot * w.p.n_visits * 4;
            const slot_hist = m.s.hist + slot * m.histStride();
            for (0..w.p.n_visits) |v| for (0..4) |site| {
                const kv_site = site < 2;
                sites[base_idx + v * 4 + site] = .{
                    .state = m.convOf(@intCast(v), site, @intCast(slot)),
                    .hist = @intCast(m.histOf(@intCast(v), site, @intCast(slot)) - slot_hist),
                    .backup = @intCast(m.backupOf(@intCast(v), site, @intCast(slot))),
                    .c = @intCast(if (kv_site) kv else h),
                };
            };
        }
        try m.upload(s, m.s.rewind_sites, std.mem.sliceAsBytes(sites));
        try m.reset(s);
        try s.synchronize();
        return m;
    }

    /// Rows 1..max_tile_rows tiles for each distinct group of the plan's layers, uploaded once: captured graphs replay them.
    fn buildDecodeTiles(m: *Model, s: cuda.Stream) !void {
        const all = try m.gpa.alloc(attention.Tile, max_tile_rows * max_tile_groups * 128);
        defer m.gpa.free(all);
        var used: usize = 0;
        for (1..max_tile_rows + 1) |r| {
            const seg: attention.Seg = .{ .row0 = 0, .n = @intCast(r), .p0 = 0, .off = 0 };
            for (m.w.p.layers[0..m.w.p.n]) |layer| {
                const group = layer.heads.q / layer.heads.kv;
                if (m.tileSlot(@intCast(r), group) != null) continue;
                if (m.n_tile_groups == max_tile_rows * max_tile_groups) return error.TooManyTileGroups;
                const n = try attention.tiles(&.{seg}, group, all[used..][0..128]);
                m.tile_table[m.n_tile_groups] = .{ .group = group, .rows = @intCast(r), .ptr = m.s.tiles1 + used * @sizeOf(attention.Tile), .n = @intCast(n) };
                m.n_tile_groups += 1;
                used += n;
            }
        }
        if (used > 0) try m.upload(s, m.s.tiles1, std.mem.sliceAsBytes(all[0..used])); // staged: `all` may go
    }

    fn tileSlot(m: *const Model, rows: u32, group: u32) ?TileSlot {
        for (m.tile_table[0..m.n_tile_groups]) |t| if (t.rows == rows and t.group == group) return t;
        return null;
    }

    /// A wide or batched call's tiles for every distinct group of the plan's layers, in one upload before its body:
    /// the body's visits then make no host copies (and a batched body can be captured).
    fn stageTiles(m: *Model, s: cuda.Stream, segs: []const attention.Seg) !void {
        var groups: [max_tile_groups]u32 = undefined;
        var n_groups: usize = 0;
        for (m.w.p.layers[0..m.w.p.n]) |layer| {
            const group = layer.heads.q / layer.heads.kv;
            if (std.mem.indexOfScalar(u32, groups[0..n_groups], group) != null) continue;
            if (n_groups == max_tile_groups) return error.TooManyTileGroups;
            groups[n_groups] = group;
            n_groups += 1;
        }
        var all: [max_tile_groups * 128]attention.Tile = undefined;
        const used = try callTiles(segs, groups[0..n_groups], &all, m.s.tiles, &m.call_tiles);
        m.n_call_tiles = n_groups;
        try m.upload(s, m.s.tiles, std.mem.sliceAsBytes(all[0..used])); // staged: `all` may go
    }

    fn callTile(m: *const Model, group: u32) ?TileSlot {
        for (m.call_tiles[0..m.n_call_tiles]) |t| if (t.group == group) return t;
        return null;
    }

    fn deinitBuffers(m: *Model) void {
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
    }

    pub fn deinit(m: *Model) void {
        m.clearGraphs();
        m.deinitBuffers();
        m.cache.free();
        m.* = undefined;
    }

    pub fn clearGraphs(m: *Model) void {
        for ([_][]?GraphEntry{ &m.graphs, &m.batch_graphs }) |table| for (table) |*e| if (e.*) |*entry| {
            entry.exec.deinit();
            e.* = null;
        };
    }

    pub fn reset(m: *const Model, s: cuda.Stream) !void {
        for (0..m.segments) |slot| try m.resetStream(s, @intCast(slot));
    }

    /// Zero one stream's convolution state; the KV rings keep their committed rows until overwritten.
    pub fn resetStream(m: *const Model, s: cuda.Stream, slot: u32) !void {
        std.debug.assert(slot < m.segments);
        const base = m.cache.ptr + @as(usize, slot) * m.layout.bytes;
        try m.zeros(s, base + m.layout.at(0).conv_offset, m.backupBytes());
    }

    pub fn rewind(m: *const Model, k: *const Kernels, s: cuda.Stream, kept: usize) !void {
        if (kept > window_rows) return error.WindowTooWide;
        const sites_ptr = m.s.rewind_sites + @as(usize, m.slot) * m.w.p.n_visits * 4 * @sizeOf(ops.RewindSite);
        try ops.rewind(&k.glue, s, sites_ptr, m.w.p.n_visits * 4, m.w.c.text.hidden, m.s.backup + @as(usize, m.slot) * m.backupBytes(), m.s.hist + @as(usize, m.slot) * m.histStride(), 0, 0, kept);
    }

    /// Narrow calls read their completed blocks as `prenorm` left them, FP32 and already normalized. Wide prefill
    /// calls keep the BF16 blocks (half the DRAM bytes): ar_reg and mix_reg load each row once into registers.
    fn normed(rows: usize, blocks: []const u64) bool {
        return rows <= window_rows and blocks.len > 0;
    }

    fn arBlocks(m: *const Model, blocks: []const u64, partial: u64, query: u64, out: u64, nq: usize, partial_bf16: bool, rows: usize) loop_math.ArArgs {
        var a: loop_math.ArArgs = .{
            .partial = partial,
            .query = query,
            .out = out,
            .n_blocks = @intCast(blocks.len),
            .partial_bf16 = @intFromBool(partial_bf16),
            .nq = @intCast(nq),
            .rows = @intCast(rows),
            .c = @intCast(m.w.c.text.hidden),
            .eps = m.w.c.text.eps,
            .blocks_normed = @intFromBool(normed(rows, blocks)),
        };
        for (blocks, 0..) |b, i| a.blocks[i] = if (normed(rows, blocks)) m.s.normed_ar[i] else b;
        return a;
    }

    fn mixArgs(m: *const Model, blocks: []const u64, partial: u64, query: u64, norms: [3]u64, out: u64, nq: usize, partial_bf16: bool, rows: usize) ops.MixArgs {
        var a: ops.MixArgs = .{
            .partial = partial,
            .query = query,
            .out = out,
            .n_blocks = @intCast(blocks.len),
            .partial_bf16 = @intFromBool(partial_bf16),
            .nq = @intCast(nq),
            .rows = @intCast(rows),
            .c = @intCast(m.w.c.text.hidden),
            .eps = m.w.c.text.eps,
            .blocks_normed = @intFromBool(normed(rows, blocks)),
        };
        for (blocks, 0..) |b, i| a.blocks[i] = if (normed(rows, blocks)) m.s.normed_mix[i] else b;
        a.norm = norms;
        return a;
    }

    /// Controller gate: narrow calls use the bit-exact 3-stage fast path; wide prefill calls that fit
    /// (h % 64 == 0, aligned rows) use the tensor-core path (same numerics within < 5e-4); anything else falls back.
    fn gateCtl(m: *const Model, lk: *const loop_math.Kernels, s: cuda.Stream, h_in: u64, m_in: u64, ctl: weights_.Controller, out: u64, rows: usize, h: usize, m_stride: usize, eps: f32) !void {
        // batched calls keep the narrow path's exact gate at any width: each stream gets its own window's bits
        if (rows <= window_rows or m.multi != null) return loop_math.gateFast(lk, s, h_in, m_in, ctl.down, ctl.forget, ctl.gain, out, m.s.gate_norm_hm, m.s.gate_zeta, rows, h, m_stride, eps);
        if (loop_math.gateWideTcFits(h_in, m_in, out, h, m_stride)) return loop_math.gateWideTc(lk, s, h_in, m_in, ctl.down, ctl.forget, ctl.gain, out, m.s.gate_norm_hm, m.s.gate_zeta, rows, h, m_stride, eps);
        return loop_math.gateStrided(lk, s, h_in, m_in, ctl.down, ctl.forget, ctl.gain, out, rows, h, m_stride, eps);
    }

    /// Whether a call's three attention-role gates take loop_math.gate3: narrow rows, and a batched call's rows (the
    /// exact scalar gate a stream's own window takes, so S > 1 keeps every stream's bits).
    fn gate3Rows(m: *const Model, rows: usize) bool {
        return rows <= window_rows or (m.multi != null and rows <= gate3_rows);
    }

    /// Whether a call's controller gates take loop_math.gateTc3 (one stream's prompt rows): their norms formed by ar and
    /// messagesNorm, the attention roles in one launch a stage. The bits of gateCtl's wide tensor-core path.
    fn tcWide(m: *const Model, rows: usize, h: usize) bool {
        return rows > window_rows and m.multi == null and loop_math.arWide(rows, h) and loop_math.gateTc3Fits(h);
    }

    /// out[row, j] = x[row] . w[j] over `c`: ops.rowdot, or, for a matrix packed BF12 (TF_INKLING2_BF12), the tensor-core
    /// matmul on its packed bytes up to window_rows rows and the exact bf16 rows unpacked into scratch beyond them.
    pub fn dense(m: *const Model, k: *const Kernels, s: cuda.Stream, x: u64, w: u64, p12: ?bf12.Mat, out: u64, rows: usize, j: usize, c: usize, out_bf16: bool, sigmoid: bool) !void {
        const mat = p12 orelse return ops.rowdot(&k.glue, s, x, w, out, rows, j, c, out_bf16, sigmoid);
        const kb = k.b12 orelse return error.Bf12KernelsMissing;
        if (sigmoid) return error.Bf12Sigmoid; // only the rows without an activation are packed
        // Unified rows-based scaling: rows >= 64 run sk=1 whole-rows (no split-K partials) unless
        // TF_INKLING2_MULTI_EXACT pins a batched forward to the 64-row decode chunks.
        const pin_narrow = m.multi != null and m.multi_exact;
        const sk: usize = if (rows >= 64 and mat.n >= 64 and !pin_narrow) 1 else mat.plan.sk;
        const chunk: usize = if (sk > 1 or (pin_narrow and rows > 64)) 64 else rows;
        const out_elem: usize = if (out_bf16) 2 else 4;
        var r0: usize = 0;
        while (r0 < rows) {
            const take = @min(rows - r0, chunk);
            try kb.matmul(s, x + r0 * c * 2, c, &mat, out + r0 * j * out_elem, !out_bf16, m.s.b12_part, take);
            r0 += take;
        }
    }

    /// dense into FP32 rows holding BF16 values (each sum rounded to bf16 and widened): the BF12 matmul rounds in its
    /// store, ops.rowdot rows take loop_math.roundBf16 after. The same bits either way.
    fn denseRounded(m: *const Model, k: *const Kernels, s: cuda.Stream, x: u64, w: u64, p12: ?bf12.Mat, out: u64, rows: usize, j: usize, c: usize) !void {
        const mat = p12 orelse {
            try ops.rowdot(&k.glue, s, x, w, out, rows, j, c, false, false);
            return loop_math.roundBf16(&k.loop, s, out, out, rows * j);
        };
        const kb = k.b12 orelse return error.Bf12KernelsMissing;
        // dense's row chunks and split-K choices, so the sums are dense's
        const pin_narrow = m.multi != null and m.multi_exact;
        const sk: usize = if (rows >= 64 and mat.n >= 64 and !pin_narrow) 1 else mat.plan.sk;
        const chunk: usize = if (sk > 1 or (pin_narrow and rows > 64)) 64 else rows;
        var r0: usize = 0;
        while (r0 < rows) {
            const take = @min(rows - r0, chunk);
            try kb.matmulStore(s, x + r0 * c * 2, c, &mat, out + r0 * j * 4, .f32_bf16, m.s.b12_part, take);
            r0 += take;
        }
    }

    /// One prompt chunk or one decode window. `record` retains every convolution input for exact prefix rollback.
    /// A narrow call (rows<=8, heads==rows, head0==0, record either way) replays a captured graph of the whole
    /// forward (TF_INKLING2_GRAPH).
    pub fn forward(m: *Model, k: *const Kernels, s: cuda.Stream, tokens: []const u32, p0: usize, head0: usize, heads: usize, record: bool, taps: u64) !void {
        const rows = tokens.len;
        if (rows == 0 or rows > prompt_rows or p0 + rows > m.context or (!record and heads > window_rows) or head0 + heads > rows)
            return error.BadLoopRows;
        if (record and rows > window_rows) return error.WindowTooWide;
        var ids: [prompt_rows]i32 = undefined;
        var pos: [prompt_rows]i32 = undefined;
        for (tokens, 0..) |token, i| {
            ids[i] = @intCast(token);
            pos[i] = @intCast(p0 + i);
        }
        // everything the forward reads from the host goes up before it, so a captured one replays on the next call's values
        try m.upload(s, m.s.ids, std.mem.sliceAsBytes(ids[0..rows]));
        try m.upload(s, m.s.pos, std.mem.sliceAsBytes(pos[0..rows]));
        const seg: attention.Seg = .{ .row0 = 0, .n = @intCast(rows), .p0 = @intCast(p0), .off = 0 };
        try m.upload(s, m.s.segs, std.mem.asBytes(&seg));
        // wide rows: every visit's tiles in one upload (narrow rows take the init-built ones)
        if (rows > max_tile_rows) try m.stageTiles(s, &.{seg});
        if (rows <= max_tile_rows and head0 == 0 and heads == rows and !m.graph_refused and graphEnabled())
            return m.decodeGraph(k, s, p0, taps, rows, record) catch |err| switch (err) {
                // a refused capture ran nothing: the same forward runs eagerly, and the graph is not tried again
                error.GraphNotPermitted => {
                    m.graph_refused = true;
                    std.log.warn("inkling2: decode graph capture refused; decoding launch by launch", .{});
                    return m.body(k, s, rows, p0, head0, heads, record, taps);
                },
                else => err,
            };
        try m.body(k, s, rows, p0, head0, heads, record, taps);
    }

    /// Several streams' rows in one forward, each stream on its own slot (caches, conv states, hist region), so the
    /// weights are read once for all of them. A single segment is exactly forward(): slot 0's graphs and bits.
    /// Several segments give each stream the numbers of its own single-stream window: every row-count-dependent
    /// choice is pinned to the narrow path's (exact gate3, shape-plan BF12 in 64-row chunks). They replay a graph
    /// captured once per launch shape (TF_INKLING2_GRAPH): rows, streams and tile counts; the streams' slots,
    /// positions and tiles are device data uploaded before each replay.
    pub const BatchedSeg = struct { tokens: []const u32, p0: usize, slot: u32 };
    pub fn forwardBatched(m: *Model, k: *const Kernels, s: cuda.Stream, segs: []const BatchedSeg, heads: usize, record: bool, taps: u64) !void {
        if (segs.len == 0) return error.BadLoopRows;
        if (segs.len == 1) {
            const sg = segs[0];
            const keep = m.slot;
            m.slot = sg.slot;
            defer m.slot = keep;
            return m.forward(k, s, sg.tokens, sg.p0, 0, heads, record, taps);
        }
        var b: Batch = undefined;
        try batch(segs, m.segments, m.context, m.layout.bytes, record, &b);
        if (heads != b.rows) return error.BadLoopRows;
        try m.upload(s, m.s.ids, std.mem.sliceAsBytes(b.ids[0..b.rows]));
        try m.upload(s, m.s.pos, std.mem.sliceAsBytes(b.pos[0..b.rows]));
        try m.upload(s, m.s.seg_of_b, std.mem.sliceAsBytes(b.seg_of[0..b.rows]));
        try m.upload(s, m.s.segs_b, std.mem.sliceAsBytes(b.segs[0..segs.len]));
        try m.upload(s, m.s.conv_segs, std.mem.sliceAsBytes(b.conv[0..segs.len]));
        try m.stageTiles(s, b.segs[0..segs.len]);
        m.multi = .{ .n = @intCast(segs.len), .last_pos = b.last_pos, .segs = b.segs };
        defer m.multi = null;
        // eager, before the body (or its replay): each stream's backup from its own slot
        if (record) for (segs) |sg| try m.copy(s, m.s.backup + @as(usize, sg.slot) * m.backupBytes(), m.cache.ptr + @as(usize, sg.slot) * m.layout.bytes + m.layout.at(0).conv_offset, m.backupBytes());
        if (!m.graph_refused and graphEnabled()) {
            // last_pos only feeds attend's partials check, which batch() already guarantees (p0 + n <= context)
            var key: GraphKey = .{ .kernels = k, .stream = @intFromPtr(s.handle), .taps = taps, .rows = @intCast(b.rows), .record = record, .streams = @intCast(segs.len) };
            for (m.call_tiles[0..m.n_call_tiles], 0..) |t, i| key.tiles[i] = t.n;
            return m.replay(&m.batch_graphs, key, k, s, b.rows, 0, record, taps) catch |err| switch (err) {
                error.GraphNotPermitted => {
                    m.graph_refused = true;
                    std.log.warn("inkling2: batched graph capture refused; batched forwards run launch by launch", .{});
                    return m.body(k, s, b.rows, 0, 0, b.rows, record, taps);
                },
                else => err,
            };
        }
        try m.body(k, s, b.rows, 0, 0, b.rows, record, taps);
    }

    /// A narrow forward (rows<=max_tile_rows, record either way) captured once per (kernels, stream, taps,
    /// rows, record, slot) and replayed: record=true only touches fixed device pointers (backup, hist, tile_table).
    fn decodeGraph(m: *Model, k: *const Kernels, s: cuda.Stream, p0: usize, taps: u64, rows: usize, record: bool) !void {
        const key: GraphKey = .{
            .kernels = k,
            .stream = @intFromPtr(s.handle),
            .taps = taps,
            .rows = @intCast(rows),
            .record = record,
            .slot = m.slot,
        };
        return m.replay(&m.graphs, key, k, s, rows, p0, record, taps);
    }

    /// Replay `key`'s graph from `table` (an LRU), capturing body(rows, p0, record, taps) on a miss.
    fn replay(m: *Model, table: []?GraphEntry, key: GraphKey, k: *const Kernels, s: cuda.Stream, rows: usize, p0: usize, record: bool, taps: u64) !void {
        m.graph_tick += 1;
        for (table) |*e| if (e.*) |*entry| if (std.meta.eql(entry.key, key)) {
            entry.used = m.graph_tick;
            return entry.exec.launchOn(s);
        };
        // miss: take a free slot, else evict the least recently used entry
        var slot: ?*?GraphEntry = null;
        for (table) |*e| if (e.* == null) {
            slot = e;
            break;
        };
        const target = slot orelse blk: {
            var victim: *?GraphEntry = &table[0];
            for (table) |*e| if (e.*) |*entry| if (entry.used < victim.*.?.used) {
                victim = e;
            };
            victim.*.?.exec.deinit();
            victim.* = null;
            break :blk victim;
        };
        try cuda.graph.beginCapture(s, .thread_local);
        m.body(k, s, rows, p0, 0, rows, record, taps) catch |err| {
            if (cuda.graph.endCapture(s)) |g| {
                var discarded = g;
                discarded.deinit();
            } else |_| {}
            return err;
        };
        var graph = try cuda.graph.endCapture(s);
        defer graph.deinit();
        var exec = try graph.instantiate();
        var stored = false;
        errdefer if (!stored) exec.deinit();
        try exec.upload(s);
        target.* = .{ .key = key, .exec = exec, .used = m.graph_tick };
        stored = true;
        if (key.streams > 1) {
            std.log.info("inkling2: batched graph captured (rows={d} streams={d} record={})", .{ rows, key.streams, record });
        } else {
            std.log.info("inkling2: narrow graph captured (rows={d} record={})", .{ rows, record });
        }
        try target.*.?.exec.launchOn(s);
    }

    /// The kernels of one forward, launches only: no host copies, nothing that reads the stream's results.
    fn body(m: *Model, k: *const Kernels, s: cuda.Stream, rows: usize, p0: usize, head0: usize, heads: usize, record: bool, taps: u64) !void {
        const c = m.w.c.text;
        const h: usize = c.hidden;
        // a batched call copied each of its streams' backups before it began
        if (record and m.multi == null) try m.copy(s, m.s.backup + @as(usize, m.slot) * m.backupBytes(), m.cache.ptr + @as(usize, m.slot) * m.layout.bytes + m.layout.at(0).conv_offset, m.backupBytes());

        try ops.embed(&k.glue, s, m.s.ids, rows, m.w.embed, m.w.embed_norm, m.s.partial_a, h, c.eps);
        var partial = m.s.partial_a;
        var next = m.s.partial_b;
        var n_blocks: usize = 0;
        const lp = c.loop.?;

        for (m.w.p.visits[0..m.w.p.n_visits]) |visit| {
            const layer = m.w.p.at(visit.layer);
            const lw = &m.w.layers[visit.layer];
            if (taps != 0) {
                const ch: ?usize = switch (visit.cache_slot) {
                    5 => 0,
                    23 => 1,
                    41 => 2,
                    else => null,
                };
                if (ch) |channel| try loop_math.captureTap(&k.loop, s, partial, taps, rows, h, channel);
            }
            if (visit.pass >= 2 and visit.layer == lp.core_start) {
                try m.copy(s, partial, m.s.prelude, rows * h * 2);
                n_blocks = 1; // block zero is the embedding; later passes never inherit completed blocks
            }
            const messages = visit.hasMessages();
            const tc = messages and m.tcWide(rows, h);
            if (messages) {
                const core_j = visit.layer - lp.core_start;
                const qm = m.w.cross_qm + 2 * core_j * h * 4;
                const qkv = m.w.cross_kv + 2 * core_j * h * 4;
                if (tc) {
                    try loop_math.messagesNorm(&k.loop, s, m.s.bank0, m.s.bank1, qm, qkv, m.s.messages, m.s.tc_mn, rows, h, c.eps, c.eps);
                } else {
                    try loop_math.messages(&k.loop, s, m.s.bank0, m.s.bank1, qm, qkv, m.s.messages, rows, 1, h, c.eps);
                }
            }
            const nq: usize = if (layer.kind == .global) 3 else 1;
            const query = m.w.attn_queries[visit.layer];
            const blocks = m.s.blocks[0..n_blocks];
            const raw_stride = rows * h * 4;
            const norm_stride = rows * h * 2;
            if (messages) {
                const ar_args = m.arBlocks(blocks, partial, query, m.s.raw, if (nq == 3) 3 else 1, true, rows);
                // One AR result shared by all three roles (identical inputs) unless global; roles read m.s.raw.
                if (tc) try loop_math.arHn(&k.loop, s, ar_args, m.s.tc_hn, c.eps) else try loop_math.ar(&k.loop, s, ar_args);
                if (tc) {
                    // The three roles in one launch a stage. Role 0's gated rows take tc_raw0: the roles may share raw's
                    // first rows as their state, so no role writes another's.
                    var g: loop_math.GateTcArgs = .{ .zeta = m.s.tc_zeta, .rows = @intCast(rows), .c = @intCast(h), .mn_ld = @intCast(4 * h), .m_stride = @intCast(4 * h), .eps = c.eps };
                    for (0..3) |role| {
                        const q_of = if (nq == 3) role else 0;
                        const ctl = lw.controllers[role];
                        g.hn[role] = m.s.tc_hn + q_of * rows * h * 2;
                        g.mn[role] = m.s.tc_mn + role * h * 2;
                        g.down[role] = ctl.down;
                        g.forget[role] = ctl.forget;
                        g.gain[role] = ctl.gain;
                        g.h[role] = m.s.raw + q_of * raw_stride;
                        g.m[role] = m.s.messages + role * h * 4;
                        g.out[role] = if (role == 0) m.s.tc_raw0 else m.s.raw + role * raw_stride;
                        g.norm[role] = lw.attn.norms[role];
                        g.normed[role] = m.s.normalized + role * norm_stride;
                    }
                    try loop_math.gateTc3(&k.loop, s, g, 3);
                } else if (m.gate3Rows(rows)) {
                    // Batched 3-role gate + norm in one launch each: bit-identical to three gateCtl/mix rounds (narrow
                    // rows; a batched call's rows too: gateFast's per-role kernels with grid z = role, any row count).
                    var h_in: [3]u64 = undefined;
                    var m_in: [3]u64 = undefined;
                    var down: [3]u64 = undefined;
                    var forget: [3]u64 = undefined;
                    var gain: [3]u64 = undefined;
                    var norms: [3]u64 = undefined;
                    for (0..3) |role| {
                        h_in[role] = if (nq == 3) m.s.raw + role * raw_stride else m.s.raw;
                        m_in[role] = m.s.messages + role * h * 4;
                        down[role] = lw.controllers[role].down;
                        forget[role] = lw.controllers[role].forget;
                        gain[role] = lw.controllers[role].gain;
                        norms[role] = lw.attn.norms[role];
                    }
                    const g: loop_math.Gate3Args = .{
                        .h = h_in,
                        .m = m_in,
                        .down = down,
                        .forget = forget,
                        .gain = gain,
                        .norm = norms,
                        .norm_hm = m.s.gate3_norm_hm,
                        .zeta = m.s.gate3_zeta,
                        .out_raw = m.s.gate3_raw,
                        .out_bf16 = m.s.gate3_bf16,
                        .rows = @intCast(rows),
                        .c = @intCast(h),
                        .m_stride = @intCast(h * 4),
                        .eps = c.eps,
                    };
                    try loop_math.gate3(&k.loop, s, g);
                } else {
                    // Reverse order: role 0's out aliases the shared m.s.raw input, so roles 2 and 1
                    // must gate (and norm) before role 0 clobbers it. Global reads per-role AR rows.
                    for ([_]usize{ 2, 1, 0 }) |role| {
                        const ctl = lw.controllers[role];
                        const h_in = if (nq == 3) m.s.raw + role * raw_stride else m.s.raw;
                        try m.gateCtl(&k.loop, s, h_in, m.s.messages + role * h * 4, ctl, m.s.raw + role * raw_stride, rows, h, h * 4, c.eps);
                        if (h <= 2560) {
                            try loop_math.rmsnorm1(&k.loop, s, m.s.raw + role * raw_stride, lw.attn.norms[role], m.s.normalized + role * norm_stride, rows, h, c.eps);
                        } else {
                            const norms = [3]u64{ lw.attn.norms[role], lw.attn.norms[role], lw.attn.norms[role] };
                            try ops.mix(&k.glue, s, m.mixArgs(&.{}, m.s.raw + role * raw_stride, 0, norms, m.s.normalized + role * norm_stride, 1, false, rows));
                        }
                    }
                }
            } else {
                try ops.mix(&k.glue, s, m.mixArgs(blocks, partial, query, lw.attn.norms, m.s.normalized, nq, true, rows));
            }
            var u_q = m.s.normalized;
            var u_k = if (layer.kind == .global or messages) m.s.normalized + norm_stride else u_q;
            var u_v = if (layer.kind == .global or messages) m.s.normalized + 2 * norm_stride else u_q;
            if (messages and !tc and m.gate3Rows(rows)) {
                u_q = m.s.gate3_bf16;
                u_k = m.s.gate3_bf16 + rows * h * 2;
                u_v = m.s.gate3_bf16 + 2 * rows * h * 2;
            }
            const q_width: usize = @as(usize, layer.heads.q) * layer.heads.dim;
            const kv_width: usize = @as(usize, layer.heads.kv) * layer.heads.dim;
            // where prep finds raw q and rel and attend the gate, and their row strides (elements)
            var qrg_rows: struct { q: u64, rel: u64, gate: u64, ld: u32 = 0 } = .{ .q = m.s.q_raw, .rel = m.s.rel_raw, .gate = m.s.gate };
            if (lw.attn.wq_wr_gate_b12) |qrg| {
                const rel_width: usize = layer.heads.q * c.d_rel;
                const kb = k.b12 orelse return error.Bf12KernelsMissing;
                // Unified rows-based scaling: sk=1 runs one call over all rows, then one split. Pinned
                // batched calls (TF_INKLING2_MULTI_EXACT) keep 64-row chunks; narrow shape plans too.
                const pin_narrow = m.multi != null and m.multi_exact;
                const qrg_chunk: usize = if (pin_narrow) 64 else if (qrg.plan.sk > 1 and rows < 64) 64 else rows;
                var q0: usize = 0;
                while (q0 < rows) : (q0 += qrg_chunk) {
                    const take = @min(rows - q0, qrg_chunk);
                    try kb.matmul(s, u_q + q0 * h * 2, h, &qrg, m.s.qrg_raw + q0 * qrg.n * 2, false, m.s.b12_part, take);
                }
                // no split: prep and attend read q | rel | gate in place, a row qrg.n elements apart
                if (qrg.n != 2 * q_width + rel_width) return error.Bf12FusedQrgWidth;
                qrg_rows = .{ .q = m.s.qrg_raw, .rel = m.s.qrg_raw + q_width * 2, .gate = m.s.qrg_raw + (q_width + rel_width) * 2, .ld = @intCast(qrg.n) };
                if (lw.attn.wk_b12 != null) {
                    try m.dense(k, s, u_k, lw.attn.wk, lw.attn.wk_b12, m.s.k_raw, rows, kv_width, h, true, false);
                } else {
                    try ops.rowdot(&k.glue, s, u_k, lw.attn.wk, m.s.k_raw, rows, kv_width, h, true, false);
                }
                if (lw.attn.wv_b12 != null) {
                    try m.dense(k, s, u_v, lw.attn.wv, lw.attn.wv_b12, m.s.v_raw, rows, kv_width, h, true, false);
                } else {
                    try ops.rowdot(&k.glue, s, u_v, lw.attn.wv, m.s.v_raw, rows, kv_width, h, true, false);
                }
            } else if (lw.attn.wq_b12 != null or lw.attn.wk_b12 != null or lw.attn.wv_b12 != null) {
                try m.dense(k, s, u_q, lw.attn.wq, lw.attn.wq_b12, m.s.q_raw, rows, q_width, h, true, false);
                try m.dense(k, s, u_k, lw.attn.wk, lw.attn.wk_b12, m.s.k_raw, rows, kv_width, h, true, false);
                try m.dense(k, s, u_v, lw.attn.wv, lw.attn.wv_b12, m.s.v_raw, rows, kv_width, h, true, false);
            } else {
                try loop_math.projectQkv(&k.loop, s, u_q, u_k, u_v, lw.attn.wq, lw.attn.wk, lw.attn.wv, m.s.q_raw, m.s.k_raw, m.s.v_raw, rows, h, q_width, kv_width);
            }
            if (lw.attn.wq_wr_gate_b12 == null) {
                try m.dense(k, s, u_q, lw.attn.wr, lw.attn.wr_b12, m.s.rel_raw, rows, layer.heads.q * c.d_rel, h, true, false);
                try m.dense(k, s, u_q, lw.attn.gate, lw.attn.gate_b12, m.s.gate, rows, q_width, h, true, false);
            }
            try ops.rowdot(&k.glue, s, u_q, lw.attn.sink, m.s.sink, rows, layer.heads.q, h, true, false);

            const geometry = m.layerGeometry(visit.layer);
            // a batched call addresses slot 0's caches and states, each segment `off` (attention) or its slot (conv) past
            const call_slot: u32 = if (m.multi != null) 0 else m.slot;
            const cache = m.cacheOf(visit.cache_slot, call_slot);
            const dynamic = layer.kind == .global;
            for (0..2) |kind| {
                const input = if (kind == 0) u_k else u_v;
                const dyn_out = if (kind == 0) m.s.k_dynamic else m.s.v_dynamic;
                if (dynamic) {
                    try m.denseRounded(k, s, input, if (kind == 0) lw.attn.k_dynamic else lw.attn.v_dynamic, if (kind == 0) lw.attn.k_dynamic_b12 else lw.attn.v_dynamic_b12, dyn_out, rows, layer.heads.kv * config.conv_taps, h);
                }
                try m.conv(k, s, .{
                    .x = if (kind == 0) m.s.k_raw else m.s.v_raw,
                    .state = 0,
                    .w = if (kind == 0) lw.attn.k_conv else lw.attn.v_conv,
                    .dyn = if (dynamic) dyn_out else 0,
                    .out = if (kind == 0) m.s.k_conv else m.s.v_conv,
                    .x_bf16 = 1,
                    .heads = @intCast(layer.heads.kv),
                    .c = @intCast(kv_width),
                    .rows = @intCast(rows),
                    .mode = 0,
                }, visit.cache_slot, kind, record);
            }
            const group = layer.heads.q / layer.heads.kv;
            // tiles staged before the body: a wide or batched call's in m.s.tiles (stageTiles), narrow rows' at init
            const tile = (if (m.multi != null or rows > max_tile_rows) m.callTile(group) else m.tileSlot(@intCast(rows), group)) orelse
                return error.Inkling2TilesMissing;
            const stride = attention.slotsFor(geometry, m.context);
            const part_capacity = prompt_rows * layer.heads.q * stride;
            const r: attention.Rows = .{
                .rows = @intCast(rows),
                .heads = layer.heads.q,
                .kv_heads = layer.heads.kv,
                .pos = m.s.pos,
                .seg_of = if (m.multi != null) m.s.seg_of_b else m.s.seg_of,
                .segs = if (m.multi != null) m.s.segs_b else m.s.segs,
                .tiles = tile.ptr,
                .n_tiles = tile.n,
                .last_pos = if (m.multi) |mu| mu.last_pos else @intCast(p0 + rows - 1),
                .streams = if (m.multi) |mu| mu.n else 1,
            };
            const prep: attention.Prep = .{ .qk_clip = c.qk_clip, .rel_clip = c.rel_clip, .floor = @floatFromInt(c.log_floor), .alpha = c.log_alpha, .eps = c.eps };
            const q_ld: u32 = if (qrg_rows.ld != 0) qrg_rows.ld else @intCast(q_width);
            const rel_ld: u32 = if (qrg_rows.ld != 0) qrg_rows.ld else @intCast(layer.heads.q * c.d_rel);
            try attention.prep(&k.attn, s, r, geometry, prep, .{ qrg_rows.q, m.s.k_conv, m.s.v_conv, qrg_rows.rel }, .{ q_ld, rel_ld }, .{ lw.attn.q_scale, lw.attn.rel_scale }, m.s.q_prepared, m.s.rel_prepared, cache);
            // every attention kernel forms its biases from the prepared rel rows: no [rows*heads][1024] table
            try attention.attend(&k.attn, s, r, geometry, stride, part_capacity, m.s.q_prepared, cache, m.s.rel_prepared, lw.attn.rel_proj_t, m.s.part_o, m.s.part_ml, m.s.sink, qrg_rows.gate, q_ld, m.s.attn_heads);
            try m.dense(k, s, m.s.attn_heads, lw.attn.wo, lw.attn.wo_b12, m.s.attn_out, rows, h, q_width, true, false);

            if (layer.boundary) {
                try m.copy(s, m.s.blocks[n_blocks], partial, rows * h * 2);
                // the block is complete: narrow calls take its RMS norm once here, wide prefill reads the BF16 block
                if (rows <= window_rows)
                    try loop_math.prenorm(&k.loop, s, m.s.blocks[n_blocks], m.s.normed_ar[n_blocks], m.s.normed_mix[n_blocks], rows, h, c.eps);
                n_blocks += 1;
            }
            try m.conv(k, s, .{
                .x = m.s.attn_out,
                .state = 0,
                .w = lw.attn_conv,
                .base = if (layer.boundary) 0 else partial,
                .resid = m.s.t,
                .x_bf16 = 1,
                .c = @intCast(h),
                .rows = @intCast(rows),
                .mode = 1,
            }, visit.cache_slot, 2, record);

            const mlp_query = m.w.mlp_queries[visit.layer];
            const mlp_blocks = m.s.blocks[0..n_blocks];
            if (tc) {
                // the MLP role alone: gateTc3's one role, in place (each element read, then written, by one thread)
                try loop_math.arHn(&k.loop, s, m.arBlocks(mlp_blocks, m.s.t, mlp_query, m.s.raw, 1, false, rows), m.s.tc_hn, c.eps);
                const ctl = lw.controllers[3];
                var g: loop_math.GateTcArgs = .{ .zeta = m.s.tc_zeta, .rows = @intCast(rows), .c = @intCast(h), .mn_ld = @intCast(4 * h), .m_stride = @intCast(4 * h), .eps = c.eps };
                g.hn[0] = m.s.tc_hn;
                g.mn[0] = m.s.tc_mn + 3 * h * 2;
                g.down[0] = ctl.down;
                g.forget[0] = ctl.forget;
                g.gain[0] = ctl.gain;
                g.h[0] = m.s.raw;
                g.m[0] = m.s.messages + 3 * h * 4;
                g.out[0] = m.s.raw;
                g.norm[0] = lw.mlp_norm;
                g.normed[0] = m.s.normalized;
                try loop_math.gateTc3(&k.loop, s, g, 1);
            } else if (messages) {
                try loop_math.ar(&k.loop, s, m.arBlocks(mlp_blocks, m.s.t, mlp_query, m.s.raw, 1, false, rows));
                const ctl = lw.controllers[3];
                try m.gateCtl(&k.loop, s, m.s.raw, m.s.messages + 3 * h * 4, ctl, m.s.raw, rows, h, h * 4, c.eps);
                // Exact for any rows (h <= 2560): the staged mix's query==0 path in one register pass.
                if (h <= 2560) {
                    try loop_math.rmsnorm1(&k.loop, s, m.s.raw, lw.mlp_norm, m.s.normalized, rows, h, c.eps);
                } else {
                    try ops.mix(&k.glue, s, m.mixArgs(&.{}, m.s.raw, 0, .{ lw.mlp_norm, lw.mlp_norm, lw.mlp_norm }, m.s.normalized, 1, false, rows));
                }
            } else {
                try ops.mix(&k.glue, s, m.mixArgs(mlp_blocks, m.s.t, mlp_query, .{ lw.mlp_norm, lw.mlp_norm, lw.mlp_norm }, m.s.normalized, 1, false, rows));
            }
            try ops.rowdot(&k.glue, s, m.s.normalized, lw.router, m.s.route_scores, rows, c.experts, h, false, true);
            try ops.topk(&k.glue, s, m.s.route_scores, lw.router_bias, m.s.route_pick, m.s.route_weights, rows, c.experts, c.topk, c.route_scale);
            try modelopt.run(&k.moe, s, lw.experts, .{ .gate = c.gate_cap, .up = c.up_cap }, m.s.normalized, h, m.s.route_pick, m.s.route_weights, rows, c.topk, m.s.moe_out, m.s.moe_scratch);
            try m.conv(k, s, .{
                .x = m.s.moe_out,
                .state = 0,
                .w = lw.mlp_conv,
                .resid = m.s.t,
                .out = next,
                .x_bf16 = 1,
                .c = @intCast(h),
                .rows = @intCast(rows),
                .mode = 2,
            }, visit.cache_slot, 3, record);
            std.mem.swap(u64, &partial, &next);

            if (visit.layer + 1 == lp.core_start) try m.copy(s, m.s.prelude, partial, rows * h * 2);
            if (visit.pass > 0 and visit.pass < lp.passes and visit.layer + 1 == lp.core_end) {
                try m.copy(s, m.s.bank0, m.s.blocks[n_blocks - 1], rows * h * 2);
                try m.copy(s, m.s.bank1, partial, rows * h * 2);
            }
        }

        if (heads == 0 and taps == 0) return;
        try ops.mix(&k.glue, s, m.mixArgs(m.s.blocks[0..n_blocks], partial, m.w.output_query, .{ m.w.norm, m.w.norm, m.w.norm }, m.s.hidden, 1, true, rows));
        if (taps != 0) try loop_math.captureTap(&k.loop, s, m.s.hidden, taps, rows, h, 3);
        if (heads == 0) return;
        try m.denseRounded(k, s, m.s.hidden + head0 * h * 2, m.w.unembed, m.w.unembed_b12, m.s.logits, heads, c.unpadded_vocab, h);
        try ops.softcap(&k.glue, s, m.s.logits, heads * c.unpadded_vocab, c.logit_cap);
    }

    pub const Saved = struct {
        at: u32,
        visits: u32,
        passes: u32,
        hidden: u32,
        buf: cuda.DeviceBuffer,
    };

    /// Compact bytes for all valid K/V rows and every convolution state of all 48 visits.
    pub fn snapshotBytes(m: *const Model, at: usize) u64 {
        var bytes = m.layout.bytes - m.layout.at(0).conv_offset;
        for (0..m.w.p.n_visits) |v| {
            const visit = m.w.p.visits[v];
            const layer = m.w.p.at(visit.layer);
            const global = layer.kind == .global;
            const count = if (global) at else @min(at, m.w.c.text.window - 1);
            if (m.layout.at(visit.cache_slot).kv_int8) {
                bytes += 2 * count * @as(u64, layer.heads.kv) * (@as(u64, layer.heads.dim) + 4);
            } else {
                const row = @as(u64, layer.heads.kv) * layer.heads.dim * 2;
                bytes += 2 * count * row;
            }
        }
        return bytes;
    }

    fn copyState(m: *const Model, s: cuda.Stream, saved: *const Saved, out: bool) !void {
        const at: usize = saved.at;
        var off: usize = 0;
        for (0..m.w.p.n_visits) |v| {
            const visit = m.w.p.visits[v];
            const layer = m.w.p.at(visit.layer);
            const geometry = m.layerGeometry(visit.layer);
            const global = layer.kind == .global;
            const count = if (global) at else @min(at, m.w.c.text.window - 1);
            const first = at - count;
            const cache = m.cacheOf(visit.cache_slot, m.slot);
            if (m.layout.at(visit.cache_slot).kv_int8) {
                const rb_c = @as(usize, layer.heads.dim);
                const rb_s: usize = 4;
                for ([2]u64{ cache.k, cache.v }) |base| {
                    for (0..layer.heads.kv) |h| {
                        const dev = base + (h * geometry.ring + first) * rb_c;
                        const src = if (out) dev else saved.buf.ptr + off;
                        const dst = if (out) saved.buf.ptr + off else dev;
                        try m.copy(s, dst, src, count * rb_c);
                        off += count * rb_c;
                    }
                }
                for ([2]u64{ cache.ks, cache.vs }) |base| {
                    for (0..layer.heads.kv) |h| {
                        const dev = base + (h * geometry.ring + first) * rb_s;
                        const src = if (out) dev else saved.buf.ptr + off;
                        const dst = if (out) saved.buf.ptr + off else dev;
                        try m.copy(s, dst, src, count * rb_s);
                        off += count * rb_s;
                    }
                }
            } else {
                const rb = @as(usize, layer.heads.kv) * layer.heads.dim * 2;
                for ([2]u64{ cache.k, cache.v }) |base| {
                    var pos = first;
                    while (pos < at) {
                        const row = pos % geometry.ring;
                        const n = @min(at - pos, geometry.ring - row);
                        const dev = base + row * rb;
                        const src = if (out) dev else saved.buf.ptr + off;
                        const dst = if (out) saved.buf.ptr + off else dev;
                        try m.copy(s, dst, src, n * rb);
                        off += n * rb;
                        pos += n;
                    }
                }
            }
        }
        const states = m.cache.ptr + @as(usize, m.slot) * m.layout.bytes + m.layout.at(0).conv_offset;
        const bytes = m.backupBytes();
        try m.copy(s, if (out) saved.buf.ptr + off else states, if (out) states else saved.buf.ptr + off, bytes);
    }

    pub fn save(m: *const Model, s: cuda.Stream, at: u32) !*Saved {
        if (at == 0 or at > m.context) return error.NotAtMark;
        const gpa = m.gpa;
        const saved = try gpa.create(Saved);
        errdefer gpa.destroy(saved);
        const lp = m.w.c.text.loop.?;
        saved.* = .{ .at = at, .visits = m.w.p.n_visits, .passes = lp.passes, .hidden = m.w.c.text.hidden, .buf = try cuda.DeviceBuffer.alloc(m.d, @intCast(@max(snapshotBytes(m, at), 256))) };
        errdefer saved.buf.free();
        try m.copyState(s, saved, true);
        return saved;
    }

    pub fn restore(m: *const Model, s: cuda.Stream, saved: *const Saved) !void {
        const lp = m.w.c.text.loop.?;
        if (saved.visits != m.w.p.n_visits or saved.passes != lp.passes or saved.hidden != m.w.c.text.hidden)
            return error.StateOfAnotherModel;
        if (saved.at == 0 or saved.at > m.context or saved.buf.len < snapshotBytes(m, saved.at)) return error.BadLoopSnapshot;
        try m.copyState(s, saved, false);
    }

    pub fn drop(m: *const Model, s: cuda.Stream, saved: *Saved) void {
        s.synchronize() catch {};
        saved.buf.free();
        m.gpa.destroy(saved);
    }
};

test "pass snapshots and rollback layout cover every visit" {
    const config_test = @import("config_test.zig");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var why: config.Why = .{};
    const c = try config.parse(arena.allocator(), config_test.mini, &why);
    var precisions: [24]@import("plan.zig").Precision = @splat(.nvfp4);
    const p = try @import("plan.zig").build(&c.text, &precisions);
    const layout = try loop.CacheLayout.init(&c.text, &p, 4096);
    try std.testing.expectEqual(@as(u32, 48), layout.n);
    try std.testing.expect(layout.at(17).kv_offset != layout.at(29).kv_offset);
    try std.testing.expectEqual(@as(u64, layout.at(46).conv_offset + layout.at(46).conv_bytes), layout.at(47).conv_offset);
}

test "a batched call lays each stream's rows after the last, on its own slot's caches, states and hist region" {
    const t = std.testing;
    const a = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18 };
    const b = [_]u32{21};
    const c = [_]u32{ 31, 32, 33 };
    const segs = [_]Model.BatchedSeg{ .{ .tokens = &a, .p0 = 100, .slot = 3 }, .{ .tokens = &b, .p0 = 7, .slot = 0 }, .{ .tokens = &c, .p0 = 4000, .slot = 15 } };
    var out: Batch = undefined;
    try batch(&segs, 16, 4096, 1 << 20, true, &out);
    try t.expectEqual(@as(usize, 12), out.rows);
    try t.expectEqual(@as(u32, 4002), out.last_pos);
    try t.expectEqualSlices(i32, &.{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2 }, out.seg_of[0..12]);
    try t.expectEqualSlices(i32, &.{ 100, 101, 102, 103, 104, 105, 106, 107, 7, 4000, 4001, 4002 }, out.pos[0..12]);
    try t.expectEqualSlices(i32, &.{ 11, 12, 13, 14, 15, 16, 17, 18, 21, 31, 32, 33 }, out.ids[0..12]);
    try t.expectEqual(attention.Seg{ .row0 = 0, .n = 8, .p0 = 100, .off = 3 << 19 }, out.segs[0]);
    try t.expectEqual(attention.Seg{ .row0 = 8, .n = 1, .p0 = 7, .off = 0 }, out.segs[1]);
    try t.expectEqual(attention.Seg{ .row0 = 9, .n = 3, .p0 = 4000, .off = 15 << 19 }, out.segs[2]);
    try t.expectEqual([4]i32{ 0, 8, 3, 0 }, out.conv[0]);
    try t.expectEqual([4]i32{ 8, 1, 0, 0 }, out.conv[1]);
    try t.expectEqual([4]i32{ 9, 3, 15, 0 }, out.conv[2]);
    // a tile never spans two streams: one tile a narrow window at group 6
    var tl: [128]attention.Tile = undefined;
    try t.expectEqual(@as(usize, 3), try attention.tiles(out.segs[0..3], 6, &tl));
}

test "a batched call refuses shared slots, slots past the model's, wide recorded windows and rows past max_batch_rows" {
    const t = std.testing;
    var out: Batch = undefined;
    const w8: [8]u32 = @splat(0);
    const w17: [17]u32 = @splat(0);
    const twice = [_]Model.BatchedSeg{ .{ .tokens = &w8, .p0 = 0, .slot = 1 }, .{ .tokens = &w8, .p0 = 0, .slot = 1 } };
    try t.expectError(error.BadLoopRows, batch(&twice, 16, 4096, 1024, true, &out));
    const past = [_]Model.BatchedSeg{ .{ .tokens = &w8, .p0 = 0, .slot = 0 }, .{ .tokens = &w8, .p0 = 0, .slot = 5 } };
    try t.expectError(error.BadLoopRows, batch(&past, 5, 4096, 1024, true, &out));
    const wide = [_]Model.BatchedSeg{ .{ .tokens = &w17, .p0 = 0, .slot = 0 }, .{ .tokens = &w8, .p0 = 0, .slot = 1 } };
    try t.expectError(error.BadLoopRows, batch(&wide, 16, 4096, 1024, true, &out));
    try batch(&wide, 16, 4096, 1024, false, &out); // unrecorded rows keep no hist
    const end = [_]Model.BatchedSeg{ .{ .tokens = &w8, .p0 = 4089, .slot = 0 }, .{ .tokens = &w8, .p0 = 0, .slot = 1 } };
    try t.expectError(error.BadLoopRows, batch(&end, 16, 4096, 1024, true, &out));
    // 16 streams of 8 rows: exactly max_batch_rows; a 9th row anywhere is too wide
    var full: [max_segments]Model.BatchedSeg = undefined;
    for (&full, 0..) |*sg, i| sg.* = .{ .tokens = &w8, .p0 = 0, .slot = @intCast(i) };
    try batch(&full, 16, 4096, 1024, true, &out);
    try t.expectEqual(max_batch_rows, out.rows);
    const w9: [9]u32 = @splat(0);
    full[15].tokens = &w9;
    try t.expectError(error.BatchTooWide, batch(&full, 16, 4096, 1024, true, &out));
    try t.expectError(error.BadLoopRows, batch(&.{}, 16, 4096, 1024, true, &out));
}

test "a batched call holds every stream's eight-row window, and narrow windows take the init-built tiles" {
    try std.testing.expect(max_batch_rows >= max_segments * 8); // DSpark depth 7: 8-row windows from every stream
    try std.testing.expect(max_tile_rows == window_rows);
    try std.testing.expect(gate3_rows >= max_batch_rows); // every batched call's rows take gate3
}

test "a call's tiles are staged once for every group: each group's slice holds what attention.tiles would" {
    const t = std.testing;
    const segs = [_]attention.Seg{ .{ .row0 = 0, .n = 8, .p0 = 100, .off = 0 }, .{ .row0 = 8, .n = 16, .p0 = 7, .off = 1 << 19 }, .{ .row0 = 24, .n = 3, .p0 = 9, .off = 2 << 19 } };
    var all: [max_tile_groups * 128]attention.Tile = undefined;
    var slots: [max_tile_groups]TileSlot = undefined;
    const base: u64 = 0x10000;
    const used = try callTiles(&segs, &.{ 3, 6 }, &all, base, &slots);
    var want: [128]attention.Tile = undefined;
    for ([_]u32{ 3, 6 }, 0..) |group, i| {
        const n = try attention.tiles(&segs, group, &want);
        try t.expectEqual(group, slots[i].group);
        try t.expectEqual(@as(u32, @intCast(n)), slots[i].n);
        try t.expectEqual(base + i * 128 * @sizeOf(attention.Tile), slots[i].ptr);
        try t.expectEqualSlices(attention.Tile, want[0..n], all[i * 128 ..][0..n]);
    }
    try t.expectEqual(128 + slots[1].n, used);
    // 16 rows at group 6 span two tiles; the others one each
    try t.expectEqual(@as(u32, 4), slots[1].n);
    // a prompt-width call: 2048 rows at group 3 fit one slice
    const wide = [_]attention.Seg{.{ .row0 = 0, .n = 2048, .p0 = 0, .off = 0 }};
    _ = try callTiles(&wide, &.{3}, &all, base, &slots);
    try t.expectEqual(@as(u32, 96), slots[0].n);
}

test "batched graph keys tell launch shapes apart and ignore the streams' slots and positions" {
    const t = std.testing;
    const a: GraphKey = .{ .rows = 24, .streams = 3, .tiles = .{ 3, 0, 0, 0 } };
    var b = a;
    try t.expect(std.meta.eql(a, b));
    b.tiles[0] = 4;
    try t.expect(!std.meta.eql(a, b));
    b = a;
    b.streams = 2;
    try t.expect(!std.meta.eql(a, b));
    b = a;
    b.record = true;
    try t.expect(!std.meta.eql(a, b));
}
