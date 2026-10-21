//! KV memory for 200k-token contexts on 16 GiB cards (K10, docs/plans/exl3-kv-memory.md): the per-sequence and
//! per-card byte arithmetic of each lever, and the int8-KV attention kernels (flashnext_exl3_kv.cu).
//!
//! The kernels are upstream's flashnext_attn.cu prep / chunks / merge with the cache stored as int8 codes and one fp16
//! scale a 32-group (exl3_cache.zig's policy: an optional normalized 32-point Hadamard per group). Their pointers may be
//! device memory or mapped pinned host memory (`HostBuffer.allocMapped(..).device()`): the chunk kernel reads only the
//! keys the sparse indexer listed (<= 2,051 a row), so a host-resident cache costs PCIe reads, not device bytes.
//! Not wired into the pass yet: the hook is one branch in cuda_batch.zig attnRows (see the plan).

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const gdn = @import("cuda_gdn.zig");
const kvpage = @import("cuda_exl3_kvpage.zig");

pub const Kv = enum { bf16, int8, int4 };

const group: usize = 32;
const head_dim = wt.head_dim;
const kv_heads = wt.kv_heads;
const index_row: usize = wt.index_dim * 2;

/// One token's K and V in one attention layer for `heads` KV heads (codes plus a 2-byte scale a 32-group).
pub fn kvRowBytes(kv: Kv, heads: usize) usize {
    const per_head: usize = switch (kv) {
        .bf16 => head_dim * 2,
        .int8 => head_dim + head_dim / group * 2,
        .int4 => head_dim / 2 + head_dim / group * 2,
    };
    return 2 * heads * per_head;
}

/// The levers that change a sequence's device bytes.
pub const Policy = struct {
    kv: Kv = .bf16,
    /// KV heads a rank keeps (2: replicated as upstream; 1: the attention head split across two ranks).
    heads: usize = kv_heads,
    /// K/V in mapped pinned host memory, read in place by the chunk kernel (only selected keys cross PCIe).
    kv_host: bool = false,
    /// The raw indexer keys [tokens][128] in mapped host memory (read only when a block of 4 completes and pools).
    ikc_host: bool = false,
};

/// Bytes a sequence of `cap` tokens holds, split by where they live.
pub const SeqBytes = struct {
    gdn: u64,
    kv: u64,
    ikc: u64,
    pooled: u64,
    tail: u64,
    host: u64,

    pub fn device(s: SeqBytes) u64 {
        return s.gdn + s.kv + s.ikc + s.pooled + s.tail;
    }
};

pub fn pooledBytes(cap: usize) u64 {
    return (cap + wt.index_ratio - 1) / wt.index_ratio * index_row;
}

/// cuda_prompt.zig seqBytes split by lever (the default policy gives upstream's number: a host test checks it).
pub fn seqBytes(cap: usize, p: Policy) SeqBytes {
    const attn: u64 = wt.attn_n;
    const kv: u64 = attn * cap * kvRowBytes(p.kv, p.heads);
    const ikc: u64 = attn * cap * index_row;
    var s: SeqBytes = .{
        .gdn = wt.linear_n * ((gdn.taps - 1) * gdn.conv_dim * 2 + wt.state_n * 4),
        .kv = if (p.kv_host) 0 else kv,
        .ikc = if (p.ikc_host) 0 else ikc,
        .pooled = attn * pooledBytes(cap),
        .tail = wt.ple_tail_n * wt.wide * 2,
        .host = 0,
    };
    if (p.kv_host) s.host += kv;
    if (p.ikc_host) s.host += ikc;
    return s;
}

/// The MTP head's one attention layer of its draft cache (the part of cuda_mtp_draft.zig Draft.bytes that grows with
/// `cap`) under the same policy; the rest of Draft.bytes is fixed (streams, logits, saved states).
pub fn draftAttnBytes(cap: usize, p: Policy) u64 {
    return (if (p.kv_host) 0 else @as(u64, cap) * kvRowBytes(p.kv, p.heads)) + (if (p.ikc_host) 0 else @as(u64, cap) * index_row) + pooledBytes(cap);
}

pub const gib: f64 = 1024.0 * 1024.0 * 1024.0;

// -- the int8 kernels -----------------------------------------------------------------------------------------------

const tile_heads: u32 = 16;
const tile_keys: u32 = 32;

/// Dynamic shared bytes of fn_kv8_chunks with `nb` raw buffers (see the .cu: q rows, one bf16 tile pair, p, raw codes).
pub fn chunkSmem(nb: u32) u32 {
    const row: u32 = @intCast((head_dim + 8) * 2);
    return tile_heads * row + 2 * tile_keys * row + tile_heads * (tile_keys + 8) * 2 + nb * 2 * tile_keys * @as(u32, @intCast(head_dim)) + nb * 2 * tile_keys * @as(u32, @intCast(head_dim / group)) * 2;
}

/// Rows up to which the chunk kernel keeps the next tile's codes in flight (upstream's bf16 kernel: 16). The int8
/// tile's smem is one bf16 pair plus NB raw buffers, so NB 2 costs 17 KiB more, not a second bf16 pair.
pub var ahead_rows: usize = 16;

pub const Kv8 = struct {
    module: cuda.Module,
    prep: cuda.Function,
    chunks: cuda.Function,
    chunks_ahead: cuda.Function,
    merge: cuda.Function,
    pack: cuda.Function,
    gather: cuda.Function,
    roundtrip: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kv8 {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kv8 = undefined;
        k.module = try cuda.Module.load(d, cuda.kernels.flashnext_exl3_kv);
        errdefer k.module.unload();
        k.prep = try k.module.function("fn_kv8_prep");
        k.chunks = try k.module.function("fn_kv8_chunks");
        k.chunks_ahead = try k.module.function("fn_kv8_chunks_ahead");
        k.merge = try k.module.function("fn_kv8_merge");
        k.pack = try k.module.function("fn_kv8_pack");
        k.gather = try k.module.function("fn_kv_gather");
        k.roundtrip = try k.module.function("fn_kv_roundtrip");
        try k.chunks.allowDynamicShared(chunkSmem(1));
        try k.chunks_ahead.allowDynamicShared(chunkSmem(2));
        return k;
    }

    pub fn deinit(k: *Kv8) void {
        k.module.unload();
    }

    fn go(f: cuda.Function, s: cuda.Stream, grid: [3]usize, block: u32, shared: u32, a: *cuda.Args) !void {
        if (grid[0] == 0 or grid[1] == 0 or grid[2] == 0) return;
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = block }, .shared = shared }, s, a);
    }

    fn int(v: usize) c_int {
        return @intCast(v);
    }

    /// A layer's int8 cache: codes [cap][heads][256] and scales [cap][heads][8] for K and for V.
    pub const Cache = struct { k8: u64, ks: u64, v8: u64, vs: u64 };

    pub fn cacheBytes(cap: usize, heads: usize) struct { codes: usize, scales: usize } {
        return .{ .codes = cap * heads * head_dim, .scales = cap * heads * (head_dim / group) * 2 };
    }

    /// fn_attn_prep with the K/V rows written as codes and scales; `nq`/`nkv` the rank's heads (proj row `pw` wide).
    pub fn prepRows(k: *const Kv8, s: cuda.Stream, proj: u64, pos: u64, q_scale: u64, k_scale: u64, i_scale: u64, inv: u64, q: u64, c: Cache, iq: u64, ikc: u64, eps: f32, rows: usize, pw: usize, nq: usize, nkv: usize, rot: bool) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ proj, pos, q_scale, k_scale, i_scale, inv, q, c.k8, c.ks, c.v8, c.vs, iq, ikc }) |v| a.add(v);
        a.add(eps);
        for ([_]usize{ pw, nq, nkv, wt.index_heads, wt.index_dim, wt.half, @intFromBool(rot) }) |v| a.add(int(v));
        try go(k.prep, s, .{ rows, nq + nkv + wt.index_heads + 1, 1 }, @intCast(head_dim), 0, &a);
    }

    /// Upstream's attnChunks over an int8 cache (same grid, same 512-key chunks, `nq` query heads over `nkv` KV heads).
    pub fn attnChunks(k: *const Kv8, s: cuda.Stream, q: u64, c: Cache, pos: u64, po: u64, pm: u64, pl: u64, ids: u64, nk: u64, sparse: u64, rows: usize, nq: usize, nkv: usize, chunks: usize, qsa: bool) !void {
        const g = nq / nkv;
        if (g > tile_heads) return error.RowShape;
        var a: cuda.Args = .{};
        for ([_]u64{ q, c.k8, c.ks, c.v8, c.vs, pos, po, pm, pl, ids, nk, sparse }) |v| a.add(v);
        for ([_]usize{ nq, nkv, g, 512, if (qsa) wt.nch else wt.dense_nch }) |v| a.add(int(v));
        a.add(@as(f32, 0.0625));
        for ([_]usize{ wt.index_idw, @intFromBool(qsa) }) |v| a.add(int(v));
        const ahead = rows <= ahead_rows;
        try go(if (ahead) k.chunks_ahead else k.chunks, s, .{ rows, nkv, chunks }, 128, chunkSmem(if (ahead) 2 else 1), &a);
    }

    pub fn attnMerge(k: *const Kv8, s: cuda.Stream, po: u64, pm: u64, pl: u64, pos: u64, out: u64, nk: u64, sparse: u64, rows: usize, nq: usize, nkv: usize, qsa: bool, rot: bool) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ po, pm, pl, pos, out, nk, sparse }) |v| a.add(v);
        for ([_]usize{ nq, nkv, nq / nkv, 512, if (qsa) wt.nch else wt.dense_nch, @intFromBool(qsa), @intFromBool(rot) }) |v| a.add(int(v));
        try go(k.merge, s, .{ rows, nq, 1 }, @intCast(head_dim), 0, &a);
    }

    /// `rows` rows of 256 bf16 (a (pos, head) row each) into codes and scales.
    pub fn packRows(k: *const Kv8, s: cuda.Stream, x: u64, codes: u64, scales: u64, rows: usize, rot: bool) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(codes);
        a.add(scales);
        a.add(@as(i64, @intCast(rows)));
        a.add(int(@intFromBool(rot)));
        try go(k.pack, s, .{ rows, 1, 1 }, @intCast(head_dim), 0, &a);
    }

    /// Test support: `rows` bf16 rows of 256 replaced in place by an int8 (bits 8) or int4 (bits 4) cache's values.
    pub fn roundTrip(k: *const Kv8, s: cuda.Stream, x: u64, rows: usize, rot: bool, bits: u32) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(rows)));
        a.add(int(@intFromBool(rot)));
        a.add(@as(c_int, @intCast(bits)));
        try go(k.roundtrip, s, .{ rows, 1, 1 }, @intCast(head_dim), 0, &a);
    }

    /// `n` rows of `row_bytes` from `src` at `ids` into `dst` (a warp a row, 8 rows a block).
    pub fn gatherRows(k: *const Kv8, s: cuda.Stream, src: u64, dst: u64, ids: u64, n: usize, row_bytes: usize) !void {
        if (row_bytes % 16 != 0) return error.RowShape;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(ids);
        a.add(int(n));
        a.add(int(row_bytes / 16));
        try go(k.gather, s, .{ (n + 7) / 8, 1, 1 }, 256, 0, &a);
    }
};

// -- the pass's attention block over an int8 cache ------------------------------------------------------------------

/// cuda_batch.zig Pass.attnBlock with the layer's K/V in `c` (int8 codes + scales) instead of seq.kc/vc: the same
/// projection, pooling, block scores, selection, gate and output projection; prep, chunks and merge are ours. The hook
/// in the pass is one branch at the top of attnBlock (`if (self.seq.kv8) |c| return exl3_kv.attnBlock(...)`, plan).
/// One sequence's rows (a prompt chunk or one stream's decode rows); a decode round over several lanes is refused here.
pub fn attnBlock(p: anytype, k8: *const Kv8, lw: anytype, ai: usize, c: Kv8.Cache, rot: bool) !void {
    const s = p.s;
    const n = p.rows;
    if (p.lanes.len > 0) return error.LanesUnsupported;
    const pa = s.proj;
    const att_rows: usize = 256;
    const ratio = wt.index_ratio;
    const top = wt.index_top;
    const idw = wt.index_idw;
    const proj_n = wt.proj_n;
    try p.lane(s.mixed, s.xsm, lw.proj, pa, s.dummy, n, 1);
    const ikc = p.seq.ikc[ai].ptr;
    const pooled = p.seq.pooled[ai].ptr;
    try k8.prepRows(p.stream, pa, s.pos, lw.q_scale.ptr, lw.k_scale.ptr, lw.i_scale.ptr, s.inv, s.q, c, s.iq, ikc, 1e-6, n, proj_n, wt.q_heads, kv_heads, rot);
    try p.tri.qsaPool(ikc, pooled, s.pos, lw.ik_scale.ptr, s.inv, 1e-6, n);
    const nb = (cdiv(p.seq.cap, ratio) + 15) / 16 * 16;
    var r0: usize = 0;
    var b: usize = 0;
    while (r0 < n) : ({
        r0 += att_rows;
        b += 1;
    }) {
        const m = @min(att_rows, n - r0);
        const ends = p.pos0 + r0 + m;
        if (ends > p.seq.cap) return error.UnexpectedTensor;
        const pos = s.pos + b * 16;
        const iq = s.iq + r0 * wt.index_heads * wt.index_dim * 2;
        if (ends / ratio > top) try p.tri.qsaScores(iq, pooled, pos, p.seq.scores, nb, m, @min(nb, cdiv(ends, ratio)));
        var a: cuda.Args = .{};
        a.add(p.seq.scores);
        a.add(pos);
        a.add(s.qids);
        a.add(s.qnk);
        a.add(s.qspr);
        for ([_]usize{ nb, idw, top, ratio }) |v| a.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(p.k.select, .{ .grid = .{ .x = @intCast(m) }, .block = .{ .x = 1024 } }, p.stream, &a);
        const chunks = cdiv(@min(ends, wt.dense_keys), 512);
        const q = s.q + r0 * wt.q_heads * head_dim * 2;
        try k8.attnChunks(p.stream, q, c, pos, s.po, s.pm, s.pl, s.qids, s.qnk, s.qspr, m, wt.q_heads, kv_heads, chunks, true);
        try k8.attnMerge(p.stream, s.po, s.pm, s.pl, pos, s.ao + r0 * wt.out_k * 2, s.qnk, s.qspr, m, wt.q_heads, kv_heads, true, rot);
    }
    try p.tri.attnGate(s.ao, pa, s.gated, s.gxs_attn, n, proj_n, wt.q_heads, head_dim);
    try p.lane(s.gated, s.gxs_attn, lw.out, s.dummy, s.part, n, wt.out_sk);
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

// -- the int8 cache in the family's engine (plan §8): `of(engine)`, Seq/SeqState.kv8, Pass.attnRows -------------

/// What an engine's new sequences hold (set once at load by `setup`, before any sequence; kept here by engine address
/// so prompt.Engine carries no field: upstream's tests build engines field by field).
pub const EngineKv = struct {
    /// The int8 kernels, loaded: every new sequence keeps int8 K/V (kc/vc hold the codes, SeqKv8 the scales).
    k8: ?*Kv8 = null,
    /// Each new sequence's raw indexer keys [cap][128] in mapped pinned host memory (read only when a block pools).
    ikc_host: bool = false,
    /// The 32-point Hadamard on each group (K10: off by default; plain int8 sat at int12's noise floor).
    rot: bool = false,
    /// KV heads a sequence holds: 1 when the attention heads are split across two ranks (cuda_exl3_heads.zig).
    heads: usize = kv_heads,

    pub fn policy(k: EngineKv) Policy {
        return .{ .kv = if (k.k8 != null) .int8 else .bf16, .ikc_host = k.ikc_host, .heads = k.heads };
    }

    /// Whether new sequences differ from upstream's newSeq.
    pub fn placed(k: EngineKv) bool {
        return k.k8 != null or k.ikc_host or k.heads != kv_heads;
    }

    /// A snapshot's rows of one layer a token: K (and V) bytes, and their int8 scales (0: bf16).
    pub fn rows(k: EngineKv) Rows {
        return snapRows(k.k8 != null, k.heads);
    }
};

pub const Rows = struct { kv: usize, scales: usize };

/// A sequence's int8 cache beside its codes in kc/vc: each attention layer's K and V scales [cap][heads][8] fp16.
pub const SeqKv8 = struct {
    k: *const Kv8,
    ks: []cuda.DeviceBuffer,
    vs: []cuda.DeviceBuffer,
    rot: bool = false,

    pub fn cache(c: SeqKv8, kc: []cuda.DeviceBuffer, vc: []cuda.DeviceBuffer, ai: usize) Kv8.Cache {
        return .{ .k8 = kc[ai].ptr, .ks = c.ks[ai].ptr, .v8 = vc[ai].ptr, .vs = c.vs[ai].ptr };
    }
};

/// TF_FLASHNEXT_KV=int8 (or int8r: rotated groups), TF_FLASHNEXT_IKC_HOST=1|0 (default: on with int8).
pub fn fromEnv(kv8_flag: bool) struct { int8: bool, rot: bool, ikc_host: bool } {
    var int8 = kv8_flag;
    var rot = false;
    if (std.c.getenv("TF_FLASHNEXT_KV")) |v| {
        const s = std.mem.span(v);
        if (std.mem.eql(u8, s, "int8") or std.mem.eql(u8, s, "int8r")) int8 = true;
        if (std.mem.eql(u8, s, "int8r")) rot = true;
        if (std.mem.eql(u8, s, "bf16")) int8 = false;
    }
    var ikc_host = int8;
    if (std.c.getenv("TF_FLASHNEXT_IKC_HOST")) |v| ikc_host = std.mem.eql(u8, std.mem.span(v), "1");
    return .{ .int8 = int8, .rot = rot, .ikc_host = ikc_host };
}

/// Loads the int8 kernels into `e` (an engine with no sequence yet) and sets its sequences' placement.
pub fn enable(e: anytype, int8: bool, rot: bool, ikc_host: bool) !void {
    const kv = try entryOf(e);
    if (int8 and kv.k8 == null) {
        const k = try e.gpa.create(Kv8);
        errdefer e.gpa.destroy(k);
        k.* = try Kv8.load(e.driver);
        kv.k8 = k;
    }
    kv.rot = rot;
    kv.ikc_host = ikc_host;
    std.log.info("flashnext: KV caches {s}{s}, indexer keys in {s} memory", .{ if (kv.k8 != null) "int8 (32-groups, fp16 scales)" else "bf16", if (rot and kv.k8 != null) " rotated" else "", if (ikc_host) "mapped pinned host" else "device" });
}

/// Widest logits pass the server runs (a decode round of 32 streams, an MTP window): `head_local`'s rows.
pub const head_local_serve_rows: usize = 64;
/// K10 lever 3: the pass's chunk for an EXL3 checkpoint unless TF_FLASHNEXT_CHUNK says (+3% prompt time a chunk at
/// 1024 vs 1536, 0.45 GiB less scratch).
pub const exl3_chunk: usize = 1024;

/// The server's load hook (cuda_lanes_load.zig, right after the engine loads, before any scratch or sequence): the
/// KV placement from `--kv-cache fp8` (int8 here: our cache is int8 codes, not e4m3) and TF_FLASHNEXT_KV /
/// TF_FLASHNEXT_IKC_HOST; for an EXL3 checkpoint also the chunk (1024) and `head_local` sized for logits passes only.
pub fn setup(e: anytype, kv8_flag: bool, exl3: bool) !void {
    const f = fromEnv(kv8_flag);
    const heads = @import("cuda_exl3_heads.zig").kvHeads(e.layers[0..e.n_layers]);
    if (kvpage.configFromEnv() != null and (f.int8 or heads != kv_heads)) return error.KvPagePlacement;
    if (heads != kv_heads) (try entryOf(e)).heads = heads;
    if (f.int8 or f.ikc_host) try enable(e, f.int8, f.rot, f.ikc_host);
    if (heads != kv_heads) std.log.info("flashnext: attention heads split across the ranks: {d} query heads and {d} KV head a rank", .{ wt.q_heads / 2, heads });
    if (!exl3) return;
    if (e.batch_s != null) return error.ScratchAlreadyReserved;
    const decoder_exl3 = e.n_layers != 0 and for (e.layers[0..e.n_layers]) |layer| {
        if (layer.exl3 == null) break false;
    } else true;
    const draft_exl3 = @import("cuda_exl3_load.zig").hasExperts(&e.store, "language_model.mtp.layers.0.mlp");
    e.scratch_format = if (decoder_exl3 and draft_exl3) .exl3 else .classic;
    if (std.c.getenv("TF_FLASHNEXT_CHUNK") == null) e.chunk = exl3_chunk;
    @import("cuda_batch_state.zig").head_local_rows = if (e.head_q.n == wt.vocab) 0 else head_local_serve_rows;
}

/// One log line: a stream's device bytes at the window by piece, and the host bytes it maps.
pub fn logStream(e: anytype, window: usize, stream_bytes: usize) !void {
    const kv = of(e);
    const layout = try sequenceLayout(kv, window, e.chunk, kvpage.configFromEnv());
    const b = layout.bytes;
    if (layout.paging) |cfg| std.log.info("flashnext: BF16 K/V RAM paging enabled: window {d}, slots {d}, rows {d}", .{ cfg.window, cfg.slots, cfg.rows });
    std.log.info("flashnext: a stream at a {d}-token window holds {d:.3} GiB on the device (K/V {d:.3} {s}, indexer keys {d:.3}, pooled {d:.3}, DeltaNet {d:.3}, draft and the rest {d:.3}) and {d:.3} GiB of mapped host memory; chunk {d} rows, scratch {d:.3} GiB", .{
        window,
        @as(f64, @floatFromInt(stream_bytes)) / gib,
        @as(f64, @floatFromInt(b.kv)) / gib,
        if (kv.k8 != null) "int8" else "bf16",
        @as(f64, @floatFromInt(b.ikc)) / gib,
        @as(f64, @floatFromInt(b.pooled)) / gib,
        @as(f64, @floatFromInt(b.gdn)) / gib,
        @as(f64, @floatFromInt(stream_bytes -| b.device())) / gib,
        @as(f64, @floatFromInt(b.host)) / gib,
        e.chunk,
        @as(f64, @floatFromInt(@import("cuda_batch_state.zig").Scratch.bytesForFormat(e.chunk, e.scratch_format))) / gib,
    });
}

/// Whether kept prompt states go to pinned host memory (cuda_snap.on_host, set here): TF_FLASHNEXT_SNAP_HOST=1|0,
/// by default for an EXL3 checkpoint served past 32,768 tokens (its streams leave the card no room for 100+ MiB states
/// that grow by 27 KiB a token).
pub fn snapHost(context: usize, exl3: bool) bool {
    var on = exl3 and context > 32768;
    if (std.c.getenv("TF_FLASHNEXT_SNAP_HOST")) |v| on = std.mem.eql(u8, std.mem.span(v), "1");
    if (kvpage.configFromEnv() != null) on = true;
    @import("cuda_snap.zig").on_host = on;
    return on;
}

/// cuda_lanes.zig rankRoom: the ranks agree how many STREAMS fit (each its own stream_bytes: rank 1 holds no MTP head
/// cache, so its streams are smaller), returned as this rank's bytes for that many streams. The native host then
/// admits the same count on both ranks. Fixed point (1/65536 of a stream) through Session.room's min.
pub fn rankRoom(r: anytype, local: u64, stream_bytes: usize) !u64 {
    if (stream_bytes == 0) return r.room(local);
    const sb: u64 = stream_bytes;
    const q = (local / sb) << 16 | ((local % sb) << 16) / sb;
    const m = try r.room(q);
    return (m >> 16) * sb + ((m & 0xffff) * sb >> 16);
}

/// The engine's deinit.
pub fn release(gpa: std.mem.Allocator, k: *EngineKv) void {
    if (k.k8) |x| {
        x.deinit();
        gpa.destroy(x);
    }
    k.* = .{};
}

pub const SeqLayout = struct {
    cache_rows: usize,
    row: Rows,
    paging: ?kvpage.Config,
    bytes: SeqBytes,
};

pub fn sequenceLayout(placement: EngineKv, cap: usize, chunk: usize, paging: ?kvpage.Config) !SeqLayout {
    if (cap == 0) return error.UnexpectedTensor;
    const page_bytes = if (paging) |requested| try kvpage.deviceBytesFor(cap, requested.forChunk(chunk)) else 0;
    var layout: SeqLayout = .{ .cache_rows = cap, .row = placement.rows(), .paging = null, .bytes = seqBytes(cap, placement.policy()) };
    if (paging) |requested| {
        if (placement.k8 != null or placement.heads != kv_heads) return error.KvPagePlacement;
        const cfg = requested.forChunk(chunk);
        try cfg.validate();
        layout.cache_rows = 1;
        layout.paging = cfg;
        layout.bytes.host = try std.math.add(u64, layout.bytes.host, layout.bytes.kv);
        layout.bytes.kv = try std.math.add(u64, page_bytes, wt.attn_n * 2 * layout.row.kv);
    }
    return layout;
}

/// Device bytes of the sequence caches under the same placement and paging shape used by newSeq.
pub fn engineSeqBytes(k: EngineKv, cap: usize, chunk: usize) !usize {
    return @intCast((try sequenceLayout(k, cap, chunk, kvpage.configFromEnv())).bytes.device());
}

/// The snapshot rows of one layer of a sequence of `heads` KV heads: codes or bf16 K (and V) a token, and scales a
/// token (0: bf16).
pub fn snapRows(int8: bool, heads: usize) Rows {
    return if (int8) .{ .kv = heads * head_dim, .scales = heads * (head_dim / group) * 2 } else .{ .kv = heads * head_dim * 2, .scales = 0 };
}

const Entry = struct { e: ?*const anyopaque = null, kv: EngineKv = .{} };
var table: [8]Entry = @splat(.{});

/// The placement `setup` gave engine `e` (upstream's when none: bf16 on the device, two KV heads).
pub fn of(e: *const anyopaque) EngineKv {
    for (&table) |*t| if (t.e == e) return t.kv;
    return .{};
}

fn entryOf(e: *const anyopaque) !*EngineKv {
    for (&table) |*t| if (t.e == e) return &t.kv;
    for (&table) |*t| if (t.e == null) {
        t.* = .{ .e = e };
        return &t.kv;
    };
    return error.OutOfMemory;
}

/// The engine's deinit: its placement and int8 kernels.
pub fn forget(gpa: std.mem.Allocator, e: *const anyopaque) void {
    for (&table) |*t| if (t.e == e) {
        release(gpa, &t.kv);
        t.* = .{};
    };
}

/// prompt.Engine.newSeq under a non-default placement: the same buffers, K/V as codes (+ scales) and/or the indexer
/// keys in one mapped host buffer the 12 layers' DeviceBuffers borrow.
pub fn newSeq(e: anytype, comptime Seq: type, cap: usize) !Seq {
    const gpa = e.gpa;
    const d = e.driver;
    const st = e.stream.*;
    const attn = wt.attn_n;
    const kv = of(e);
    const layout = try sequenceLayout(kv, cap, e.chunk, e.page_config);
    const paged = if (layout.paging) |cfg| if (cfg.shared_lanes != 0) try kvpage.Paged.initShared(gpa, d, cap, cfg, e.shared_arena orelse return error.SharedKvArenaRequired) else try kvpage.Paged.init(gpa, d, cap, cfg) else null;
    errdefer if (paged) |pg| pg.destroy();
    const logits = try gpa.alloc(u8, wt.vocab * 2);
    errdefer gpa.free(logits);
    var held: Held = .{ .gpa = gpa };
    errdefer held.free();
    const conv = try held.bufs(d, st, wt.linear_n, (gdn.taps - 1) * gdn.conv_dim * 2);
    const state = try held.bufs(d, st, wt.linear_n, wt.state_n * 4);
    const int8 = kv.k8 != null;
    if (int8 and kv.heads != kv_heads) return error.SplitAttentionUnsupported;
    const row = layout.row;
    const kc = try held.bufs(d, st, attn, layout.cache_rows * row.kv);
    const vc = try held.bufs(d, st, attn, layout.cache_rows * row.kv);
    var ks: []cuda.DeviceBuffer = &.{};
    var vs: []cuda.DeviceBuffer = &.{};
    if (int8) {
        ks = try held.bufs(d, st, attn, cap * row.scales);
        vs = try held.bufs(d, st, attn, cap * row.scales);
    }
    var host: ?cuda.HostBuffer = null;
    errdefer if (host) |*h| h.free();
    const ikc_len = cap * index_row;
    const ikc = if (kv.ikc_host) blk: {
        const stride = std.mem.alignForward(usize, ikc_len, 256);
        host = try cuda.HostBuffer.allocMapped(d, stride * attn);
        @memset(host.?.bytes, 0);
        const base = try host.?.device();
        const out = try gpa.alloc(cuda.DeviceBuffer, attn);
        for (out, 0..) |*b, i| b.* = .{ .d = d, .ptr = base + i * stride, .len = ikc_len, .borrowed = true };
        break :blk out;
    } else try held.bufs(d, st, attn, ikc_len);
    errdefer if (host != null) gpa.free(ikc);
    const pooled = try held.bufs(d, st, attn, @intCast(pooledBytes(cap)));
    var tail = try cuda.DeviceBuffer.alloc(d, wt.ple_tail_n * wt.wide * 2);
    errdefer tail.free();
    try tail.fill8(0, st.handle);
    try st.synchronize();
    return .{
        .conv = conv,
        .state = state,
        .kc = kc,
        .vc = vc,
        .ikc = ikc,
        .pooled = pooled,
        .tail = tail,
        .history = .{ e.eos, e.eos },
        .fed = 0,
        .logits = logits,
        .cap = cap,
        .kv8 = if (kv.k8) |k| .{ .k = k, .ks = ks, .vs = vs, .rot = kv.rot } else null,
        .ikc_host = host,
        .paged = paged,
    };
}

/// prompt.Seq.deinit's part for what newSeq adds (the scales and the indexer keys' host buffer).
pub fn deinitSeq(gpa: std.mem.Allocator, kv8: *?SeqKv8, host: *?cuda.HostBuffer) void {
    if (kv8.*) |c| {
        for (c.ks) |*b| b.free();
        for (c.vs) |*b| b.free();
        gpa.free(c.ks);
        gpa.free(c.vs);
        kv8.* = null;
    }
    if (host.*) |*h| {
        h.free();
        host.* = null;
    }
}

/// Zeroed device buffers made so far, freed together on a failure.
const Held = struct {
    gpa: std.mem.Allocator,
    lists: [8][]cuda.DeviceBuffer = undefined,
    n: usize = 0,

    fn bufs(h: *Held, d: *const cuda.Driver, st: cuda.Stream, count: usize, len: usize) ![]cuda.DeviceBuffer {
        const out = try h.gpa.alloc(cuda.DeviceBuffer, count);
        var made: usize = 0;
        errdefer {
            for (out[0..made]) |*b| b.free();
            h.gpa.free(out);
        }
        while (made < count) : (made += 1) {
            out[made] = try cuda.DeviceBuffer.alloc(d, len);
            out[made].fill8(0, st.handle) catch |err| {
                out[made].free();
                return err;
            };
        }
        h.lists[h.n] = out;
        h.n += 1;
        return out;
    }

    fn free(h: *Held) void {
        for (h.lists[0..h.n]) |l| {
            for (l) |*b| b.free();
            h.gpa.free(l);
        }
        h.n = 0;
    }
};

/// cuda_batch.zig Pass.attnRows over an int8 cache (the hook: `if (self.seq.kv8) |c| return ...`): `n` rows of
/// p.seq from p.pos0, at `at` (positions, queries, indexer queries, outputs). The same projection rows, pooling,
/// block scores and selection as upstream's; prep, chunks and merge ours. A decode round calls it once a lane with that
/// lane's seq, as upstream's attnBlock does, so multi-lane rounds take int8 caches too.
pub fn attnRows(p: anytype, lw: anytype, ai: usize, pa: u64, n: usize, at: anytype, c: SeqKv8) !void {
    const s = p.s;
    if (p.seq.media != null) return error.Kv8MediaUnsupported;
    const k8 = c.k;
    const kv = c.cache(p.seq.kc, p.seq.vc, ai);
    const ikc = p.seq.ikc[ai].ptr;
    const pooled = p.seq.pooled[ai].ptr;
    const ratio = wt.index_ratio;
    const top = wt.index_top;
    const att_rows: usize = 256;
    try k8.prepRows(p.stream, pa, at.pos, lw.q_scale.ptr, lw.k_scale.ptr, lw.i_scale.ptr, s.inv, at.q, kv, at.iq, ikc, 1e-6, n, wt.proj_n, wt.q_heads, kv_heads, c.rot);
    try p.tri.qsaPool(ikc, pooled, at.pos, lw.ik_scale.ptr, s.inv, 1e-6, n);
    const nb = (cdiv(p.seq.cap, ratio) + 15) / 16 * 16;
    var r0: usize = 0;
    var b: usize = 0;
    while (r0 < n) : ({
        r0 += att_rows;
        b += 1;
    }) {
        const m = @min(att_rows, n - r0);
        const ends = p.pos0 + r0 + m;
        if (ends > p.seq.cap) return error.UnexpectedTensor;
        const pos = at.pos + b * 16;
        const iq = at.iq + r0 * wt.index_heads * wt.index_dim * 2;
        if (ends / ratio > top) try p.tri.qsaScores(iq, pooled, pos, p.seq.scores, nb, m, @min(nb, cdiv(ends, ratio)));
        var a: cuda.Args = .{};
        a.add(p.seq.scores);
        a.add(pos);
        a.add(s.qids);
        a.add(s.qnk);
        a.add(s.qspr);
        for ([_]usize{ nb, wt.index_idw, top, ratio }) |v| a.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(p.k.select, .{ .grid = .{ .x = @intCast(m) }, .block = .{ .x = 1024 } }, p.stream, &a);
        const chunks = cdiv(@min(ends, wt.dense_keys), 512);
        try k8.attnChunks(p.stream, at.q + r0 * wt.q_heads * head_dim * 2, kv, pos, s.po, s.pm, s.pl, s.qids, s.qnk, s.qspr, m, wt.q_heads, kv_heads, chunks, true);
        try k8.attnMerge(p.stream, s.po, s.pm, s.pl, pos, at.ao + r0 * wt.out_k * 2, s.qnk, s.qspr, m, wt.q_heads, kv_heads, true, c.rot);
    }
}

// -- host tests -------------------------------------------------------------------------------------------------------

test "exl3: bf16 paging and host indexer placement share allocation and budget geometry" {
    const cap: usize = 200_016;
    const cfg: kvpage.Config = .{};
    for ([_]bool{ false, true }) |host_index| {
        const placement: EngineKv = .{ .ikc_host = host_index };
        const resident = try sequenceLayout(placement, cap, 1024, null);
        const paged = try sequenceLayout(placement, cap, 4096, cfg);
        try std.testing.expectEqual(cap, resident.cache_rows);
        try std.testing.expectEqual(@as(usize, 1), paged.cache_rows);
        try std.testing.expectEqual(@as(usize, 4096), paged.paging.?.rows);
        try std.testing.expectEqual(snapRows(false, kv_heads), paged.row);
        const placeholders = wt.attn_n * 2 * paged.row.kv;
        try std.testing.expectEqual(try kvpage.deviceBytesFor(cap, cfg.forChunk(4096)) + placeholders, paged.bytes.kv);
        try std.testing.expectEqual(resident.bytes.host + resident.bytes.kv, paged.bytes.host);
        try std.testing.expectEqual(resident.bytes.device() - resident.bytes.kv + paged.bytes.kv, paged.bytes.device());
        try std.testing.expectEqual(resident.bytes.ikc, paged.bytes.ikc);
        try std.testing.expectEqual(@as(u64, if (host_index) 0 else wt.attn_n * cap * index_row), paged.bytes.ikc);
        const small = try sequenceLayout(placement, cap, 1024, cfg);
        try std.testing.expectEqual(@as(u64, 2 * wt.attn_n * (4096 - cfg.rows) * kvpage.row_bytes), paged.bytes.device() - small.bytes.device());
    }
}

test "exl3: paging refuses incompatible cache layouts before allocating" {
    const cfg: kvpage.Config = .{};
    var k8: Kv8 = undefined;
    try std.testing.expectError(error.KvPagePlacement, sequenceLayout(.{ .k8 = &k8 }, 4096, 1024, cfg));
    try std.testing.expectError(error.KvPagePlacement, sequenceLayout(.{ .heads = 1 }, 4096, 1024, cfg));
    try std.testing.expectError(error.KvPageConfig, sequenceLayout(.{}, 4096, 1024, .{ .slots = 0 }));
    try std.testing.expectError(error.KvPageConfig, sequenceLayout(.{}, 4096, 1024, .{ .window = wt.dense_keys - 1 }));
}

test "exl3: newSeq with host indexer keys enters the paging allocator" {
    if (cuda.kernels.available) return error.SkipZigTest;
    const old_force = kvpage.force;
    const old_off = kvpage.force_off;
    defer {
        kvpage.force = old_force;
        kvpage.force_off = old_off;
    }
    kvpage.force = .{};
    kvpage.force_off = false;
    var driver: cuda.Driver = undefined;
    var stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var engine: @import("cuda_prompt.zig").Engine = undefined;
    engine.gpa = std.testing.allocator;
    engine.driver = &driver;
    engine.stream = &stream;
    engine.chunk = 4096;
    engine.round_graph = .{};
    engine.page_config = kvpage.force;
    (try entryOf(&engine)).ikc_host = true;
    defer forget(engine.gpa, &engine);
    try std.testing.expectError(error.BuiltWithoutKernels, engine.newSeq(8192));
}

test "exl3: kv the default policy is upstream's seqBytes, split" {
    const prompt = @import("cuda_prompt.zig");
    for ([_]usize{ 1, 5, 4096, 200_016, 262_144 }) |cap| {
        const s = seqBytes(cap, .{});
        try std.testing.expectEqual(@as(u64, prompt.seqBytes(cap)), s.device());
        try std.testing.expectEqual(@as(u64, 0), s.host);
    }
}

test "exl3: kv int8 is 34 bytes a 32-group, int4 18; a head split halves K/V, not the indexer" {
    try std.testing.expectEqual(@as(usize, 2048), kvRowBytes(.bf16, 2));
    try std.testing.expectEqual(@as(usize, 1088), kvRowBytes(.int8, 2));
    try std.testing.expectEqual(@as(usize, 576), kvRowBytes(.int4, 2));
    try std.testing.expectEqual(@as(usize, 544), kvRowBytes(.int8, 1));
    const cap = 200_000;
    const a = seqBytes(cap, .{ .kv = .int8 });
    const b = seqBytes(cap, .{ .kv = .int8, .heads = 1 });
    try std.testing.expectEqual(a.kv, 2 * b.kv);
    try std.testing.expectEqual(a.ikc, b.ikc);
    const h = seqBytes(cap, .{ .kv = .int8, .kv_host = true, .ikc_host = true });
    try std.testing.expectEqual(a.kv + a.ikc, h.host);
    try std.testing.expectEqual(a.device() - a.kv - a.ikc, h.device());
}

test "exl3: kv chunk smem fits the 99 KiB opt-in on sm_120" {
    try std.testing.expect(chunkSmem(2) <= 99 * 1024);
    try std.testing.expectEqual(@as(u32, 60928), chunkSmem(1));
    try std.testing.expectEqual(@as(u32, 78336), chunkSmem(2));
}
