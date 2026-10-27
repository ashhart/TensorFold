//! KV memory for 200k-token contexts on 16 GiB cards (K10, docs/plans/exl3-kv-memory.md): the per-sequence and

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const gdn = @import("../../cuda_gdn.zig");
const kvpage = @import("cuda_exl3_kvpage.zig");
const int8k = @import("cuda_exl3_kv8.zig");

const group: usize = 32;
const head_dim = wt.head_dim;
const kv_heads = wt.kv_heads;
const index_row: usize = wt.index_dim * 2;

const cdiv = int8k.cdiv;

pub const Kv = enum { bf16, int8, int4 };

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
pub fn draftAttnBytes(cap: usize, p: Policy) u64 {
    return (if (p.kv_host) 0 else @as(u64, cap) * kvRowBytes(p.kv, p.heads)) + (if (p.ikc_host) 0 else @as(u64, cap) * index_row) + pooledBytes(cap);
}

pub const gib: f64 = 1024.0 * 1024.0 * 1024.0;

/// Rows up to which the chunk kernel keeps the next tile's codes in flight (upstream's bf16 kernel: 16).
pub var ahead_rows: usize = 16;

// -- the int8 cache in the family's engine (plan §8): `of(engine)`, Seq/SeqState.kv8, Pass.attnRows -------------

/// What an engine's new sequences hold (set once at load by `setup`, before any sequence; kept here by engine address
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
pub const exl3_chunk: usize = 1024;

/// The server's load hook (cuda_lanes_load.zig, right after the engine loads, before any scratch or sequence).
pub fn setup(e: anytype, kv8_flag: bool, exl3: bool) !void {
    const f = fromEnv(kv8_flag);
    const heads = @import("../model/cuda_exl3_heads.zig").kvHeads(e.layers[0..e.n_layers]);
    if (kvpage.configFromEnv() != null and (f.int8 or heads != kv_heads)) return error.KvPagePlacement;
    if (heads != kv_heads) (try entryOf(e)).heads = heads;
    if (f.int8 or f.ikc_host) try enable(e, f.int8, f.rot, f.ikc_host);
    if (heads != kv_heads) std.log.info("flashnext: attention heads split across the ranks: {d} query heads and {d} KV head a rank", .{ wt.q_heads / 2, heads });
    if (!exl3) return;
    if (e.batch_s != null) return error.ScratchAlreadyReserved;
    const decoder_exl3 = e.n_layers != 0 and for (e.layers[0..e.n_layers]) |layer| {
        if (layer.exl3 == null) break false;
    } else true;
    const draft_exl3 = @import("../pack/cuda_exl3_load.zig").hasExperts(&e.store, "language_model.mtp.layers.0.mlp");
    e.scratch_format = if (decoder_exl3 and draft_exl3) .exl3 else .classic;
    if (std.c.getenv("TF_FLASHNEXT_CHUNK") == null) e.chunk = exl3_chunk;
    @import("../../cuda_batch_state.zig").head_local_rows = if (e.head_q.n == wt.vocab) 0 else head_local_serve_rows;
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
        @as(f64, @floatFromInt(@import("../../cuda_batch_state.zig").Scratch.bytesForFormat(e.chunk, e.scratch_format))) / gib,
    });
}

/// Whether kept prompt states go to pinned host memory (cuda_snap.on_host, set here):
pub fn snapHost(context: usize, exl3: bool) bool {
    var on = exl3 and context > 32768;
    if (std.c.getenv("TF_FLASHNEXT_SNAP_HOST")) |v| on = std.mem.eql(u8, std.mem.span(v), "1");
    if (kvpage.configFromEnv() != null) on = true;
    @import("../../cuda_snap.zig").on_host = on;
    return on;
}

/// cuda_lanes.zig rankRoom: the ranks agree how many STREAMS fit (each its own stream_bytes: rank 1 holds no MTP head
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

pub fn entryOf(e: *const anyopaque) !*EngineKv {
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

/// cuda_batch.zig Pass.attnRows over an int8 cache (the hook:
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
    try p.tri.qsaPool(ikc, pooled, at.pos, lw.ik_scale.ptr, s.inv, 1e-6, n, .{});
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
        if (ends / ratio > top) try p.tri.qsaScores(iq, pooled, pos, p.seq.scores, nb, m, @min(nb, cdiv(ends, ratio)), .{});
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

pub const Kv8 = int8k.Kv8;
pub const chunkSmem = int8k.chunkSmem;
pub const attnBlock = int8k.attnBlock;
