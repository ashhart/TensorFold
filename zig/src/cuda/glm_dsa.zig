//! GLM-5.3-Flash's DSA block on one rank (forward.py dsa_block, latent path, long context, fp8 latent cache).
const std = @import("std");
const driver = @import("driver.zig");
const memory = @import("memory.zig");
const stream_ = @import("stream.zig");
const exl3 = @import("exl3.zig");
const glue2_ = @import("glm_glue2.zig");
const bmm_ = @import("glm_bmm.zig");
const latent_ = @import("glm_latent.zig");
const sparse_ = @import("glm_sparse.zig");
const Stream = stream_.Stream;
const B16 = bmm_.B16;

pub const hidden = 4096;
pub const q_lora = 1536;
pub const kv_lora = latent_.lw;
pub const heads = latent_.heads; // the rank's attention heads
pub const qk_dim = latent_.d;
pub const index_dim = sparse_.d;
pub const ikr_w = index_dim + sparse_.heads; // ikr's row: the index key, then the 32 head weights
pub const chunk = 512; // keys a dense chunk (attention.CHUNK)
pub const dense_limit = sparse_.width; // the largest context dense attention covers (2,051)
pub const prompt_att_rows = 512; // rows one dense call takes in a prompt chunk (forward.PROMPT_ATT_ROWS)
pub const select_rows = 512; // rows scored at once (sparse.SELECT_ROWS)
const sparse_nch = (sparse_.width + chunk - 1) / chunk;

/// The kernel families the block launches; the caller owns them (a layer's Kernels holds the values).
pub const Kernels = struct {
    glue2: *const glue2_.Glue2,
    bmm: *const bmm_.Bmm,
    dense: *exl3.Dense,
    latent: *latent_.Latent,
    sparse: *sparse_.Sparse,
};

/// The model's constants the block reads (weights.Config; the indexer's LayerNorm eps is sparse.py's 1e-6).
pub const Config = struct { eps: f32, index_eps: f32, scale: f32, index_scale: f32, wscale: f32 };

/// The replicated indexer: wk | weights_proj [160, D], wq_b [4096, 1536], the gate [128, D], LayerNorm, pool ape.
pub const IndexW = struct { kw: B16, qb: B16, gate: u64, ln_w: u64, ln_b: u64, ape: u64 };

/// The rank's DSA weights: q_a/kv_a/q_b and o_proj's rows (EXL3), the bf16 norms, kv_b as AbsorbW [H, 256, 512].
pub const DsaW = struct {
    q_a: exl3.Weight,
    kv_a: exl3.Weight,
    q_b: exl3.Weight,
    o: exl3.Weight,
    q_norm: u64,
    kv_norm: u64,
    wk: u64,
    wv: u64,
    index: IndexW,
};

/// This layer's committed caches: fp8 latents [cap, 512] with f32 scales [cap], ik/ig [cap, 128], pk [cap/4 + 2, 128].
pub const DsaState = struct { lc: u64, scales: u64, ik: u64, ig: u64, pk: u64, pos_dev: u64, cap: usize };

const Lay = struct {
    at: u64,
    fn take(l: *Lay, bytes: usize) u64 {
        const r = l.at;
        l.at += (bytes + 255) / 256 * 256;
        return r;
    }
};

/// Buffers for windows of up to `rows` rows (forward.Buffers' DSA slice); torch.empty's: nothing initialized.
pub const Scratch = struct {
    buf: memory.DeviceBuffer,
    rows: usize,
    part_rows: usize, // rows a dense call takes: the window's in decode, 512 in a prompt chunk
    nch: usize, // the dense pass's chunk cap (LatentScratch.nch)
    np_max: usize, // pools a selection may score (the caches' cap / 4)
    dqa: u64,
    dkv: u64,
    sums: u64, // the rmsnorms' dead group sums
    qr: u64,
    lat: u64,
    q: u64,
    ikr: u64,
    igr: u64,
    mpart: u64, // split-K partials of the bf16 matmuls and the router (one at a time)
    qa: u64,
    ol: u64,
    vn: u64,
    qi: u64,
    po: u64,
    pm: u64,
    pl: u64,
    scores: u64,
    pools: u64,
    tokens: u64,
    counts: u64,
    posv: u64, // a prompt chunk's dense blocks' positions (pos + r0), device i32 each

    fn place(l: *Lay, w: *const DsaW, rows: usize, prefill: bool, cap: usize) !Scratch {
        if (rows == 0 or cap < 4 * select_rows) return error.GlmDsaScratch;
        var sc: Scratch = undefined;
        sc.rows = rows;
        sc.part_rows = if (prefill) @min(rows, prompt_att_rows) else rows;
        sc.nch = (@min(cap, 2560) + rows + chunk - 1) / chunk;
        sc.np_max = cap / 4;
        const qn: usize = heads * qk_dim;
        const parts = @max(sc.nch * sc.part_rows, sparse_nch * rows) * heads;
        var mp: usize = @as(usize, bmm_.router_ks) * rows * index_dim * 4;
        for ([_]B16{ w.index.kw, w.index.qb }) |b| mp = @max(mp, bmm_.Bmm.partBytes(b, rows));
        sc.dqa = l.take(rows * q_lora * 2);
        sc.dkv = l.take(rows * kv_lora * 2);
        sc.sums = l.take(rows * q_lora / 64 * 4);
        sc.qr = l.take(rows * q_lora * 2);
        sc.lat = l.take(rows * kv_lora * 2);
        sc.q = l.take(rows * qn * 2);
        sc.ikr = l.take(rows * ikr_w * 2);
        sc.igr = l.take(rows * index_dim * 4);
        sc.mpart = l.take(mp);
        sc.qa = l.take(rows * heads * kv_lora * 2);
        sc.ol = l.take(rows * heads * kv_lora * 2);
        sc.vn = l.take(rows * qn * 2);
        sc.qi = l.take(rows * sparse_.heads * index_dim * 2);
        sc.po = l.take(parts * kv_lora * 4);
        sc.pm = l.take(parts * 4);
        sc.pl = l.take(parts * 4);
        sc.scores = l.take(@min(rows, select_rows) * sc.np_max * 4);
        sc.pools = l.take(rows * sparse_.topk_pools * 8);
        sc.tokens = l.take(rows * sparse_.width * 4);
        sc.counts = l.take(rows * 4);
        sc.posv = l.take((rows + prompt_att_rows - 1) / prompt_att_rows * 4);
        return sc;
    }

    /// `rows`: the buffer's (Buffers.rows: they cap the dense chunks, as LatentScratch.nch does); `cap`: the caches'.
    pub fn init(d: *const driver.Driver, w: *const DsaW, rows: usize, prefill: bool, cap: usize) !Scratch {
        var probe: Lay = .{ .at = 0 };
        _ = try place(&probe, w, rows, prefill, cap);
        var buf = try memory.DeviceBuffer.alloc(d, probe.at);
        errdefer buf.free();
        var l: Lay = .{ .at = buf.ptr };
        var sc = try place(&l, w, rows, prefill, cap);
        sc.buf = buf;
        return sc;
    }

    pub fn deinit(sc: *Scratch) void {
        sc.buf.free();
    }
};

/// sparse.pool_bucket: the visible pools rounded up to a power of two, at least 1,024, at most the caches'.
pub fn poolBucket(pos: usize, rows: usize, np_max: usize) usize {
    const visible = (pos + rows) / sparse_.pool + 1;
    return @min(np_max, @max(1024, std.math.ceilPowerOfTwoAssert(usize, visible)));
}

/// dsa_front: q_a and kv_a on normed, their rmsnorms (qr, lat), then q_b into q [rows, 32, 256].
fn front(k: Kernels, s: *Stream, c: Config, w: *const DsaW, sc: *const Scratch, normed: u64, rows: usize) !void {
    try k.dense.prompt(s.*, w.q_a, normed, .bf16, rows, sc.dqa, .bf16);
    try k.dense.prompt(s.*, w.kv_a, normed, .bf16, rows, sc.dkv, .bf16);
    try k.glue2.rmsnorm2(s.*, sc.dqa, q_lora, w.q_norm, sc.qr, q_lora, sc.sums, c.eps, rows, q_lora);
    try k.glue2.rmsnorm2(s.*, sc.dkv, kv_lora, w.kv_norm, sc.lat, kv_lora, sc.sums, c.eps, rows, kv_lora);
    try k.dense.prompt(s.*, w.q_b, sc.qr, .bf16, rows, sc.q, .bf16);
}

/// forward.dense_attention: part_rows-row calls, each at its first row's position, over the capped chunk count.
fn denseAttn(k: Kernels, s: *Stream, c: Config, st: DsaState, sc: *const Scratch, rows: usize, pos: usize, all: bool) !void {
    const nch: u32 = @intCast(if (all) sc.nch else @min((pos + rows + chunk - 1) / chunk, sc.nch));
    const row_qa: u64 = heads * kv_lora * 2;
    var r0: usize = 0;
    while (r0 < rows) : (r0 += sc.part_rows) {
        const n: u32 = @intCast(@min(sc.part_rows, rows - r0));
        var at = st.pos_dev;
        if (r0 > 0) { // pos_dev + r0: a device word written in stream order
            at = sc.posv + r0 / sc.part_rows * 4;
            const slot: memory.DeviceBuffer = .{ .d = sc.buf.d, .ptr = at, .len = 4, .borrowed = true };
            try slot.fill32(@intCast(pos + r0), s.handle);
        }
        try k.latent.attnChunks(s, sc.qa + r0 * row_qa, st.lc, st.scales, at, 0, 0, sc.po, sc.pm, sc.pl, n, 0, nch, true, false, c.scale);
        try k.latent.merge(s, sc.po, sc.pm, sc.pl, sc.ol + r0 * row_qa, 0, n, nch, false);
    }
}

/// select_tokens and sparse_attention: rows past the dense limit attend their top-512 pools' tokens.
fn sparseAttn(k: Kernels, s: *Stream, c: Config, w: *const DsaW, st: DsaState, sc: *const Scratch, rows: usize, pos: usize, capture: bool) !void {
    const posp: u64 = if (capture) st.pos_dev else 0; // a captured step is one row: its position is pos_dev's
    try k.bmm.matmul(s.*, w.index.qb, sc.qr, q_lora, rows, sc.qi, false, sc.mpart);
    const np = poolBucket(pos, rows, sc.np_max);
    if (np < sparse_.topk_pools) return error.GlmDsaCapacity; // sparse.top_pools' torch fallback is not ported
    const qi_row: u64 = sparse_.heads * index_dim * 2;
    var a: usize = 0;
    while (a < rows) : (a += select_rows) {
        const n: u32 = @intCast(@min(select_rows, rows - a));
        const at: i32 = @intCast(pos + a);
        try k.sparse.scores(s, sc.qi + a * qi_row, sparse_.heads * index_dim, sc.ikr + (a * ikr_w + index_dim) * 2, ikr_w, st.pk, sc.scores, at, posp, n, @intCast(np), c.index_scale, c.wscale);
        try k.sparse.selectRows(s, sc.scores, @intCast(np), sc.pools + a * sparse_.topk_pools * 8, at, posp, sparse_.topk_pools, n, true);
    }
    const r32: u32 = @intCast(rows);
    try k.sparse.tokensCounts(s, sc.pools, sc.tokens, sc.counts, @intCast(pos), posp, r32);
    try k.latent.attnChunks(s, sc.qa, st.lc, st.scales, 0, sc.tokens, sc.counts, sc.po, sc.pm, sc.pl, r32, sparse_.width, sparse_nch, true, true, c.scale);
    try k.latent.merge(s, sc.po, sc.pm, sc.pl, sc.ol, sc.counts, r32, sparse_nch, true);
}

/// dsa_block at pos (= st.pos_dev) into st and part; `capture`: a graph's one-row step (pool bucket fixed by pos).
pub fn dsaBlock(k: Kernels, s: *Stream, c: Config, w: *const DsaW, st: DsaState, sc: *const Scratch, normed: u64, part: u64, rows: usize, pos: usize, capture: bool) !void {
    if (rows == 0 or rows > sc.rows) return error.GlmDsaRows;
    if (pos + rows > st.cap or st.cap / 4 != sc.np_max) return error.GlmDsaCapacity;
    const r32: u32 = @intCast(rows);
    const p32: i32 = @intCast(pos);
    try front(k, s, c, w, sc, normed, rows);
    try k.latent.latWrite(s, sc.lat, kv_lora, st.lc, st.pos_dev, st.scales, r32, true);
    const all_sparse = pos >= dense_limit;
    const sparse_rows = pos + rows - 1 >= dense_limit;
    if (capture and (rows != 1 or sparse_rows != all_sparse)) return error.GlmDsaCapture;
    // the indexer runs on every call: it fills the caches while the window is still dense
    const ix = w.index;
    try k.bmm.matmul(s.*, ix.kw, normed, hidden, rows, sc.ikr, false, sc.mpart);
    try k.bmm.router(s.*, normed, hidden, ix.gate, rows, index_dim, hidden, sc.igr, sc.mpart);
    try k.sparse.indexWrite(s, sc.ikr, ikr_w, sc.igr, ix.ln_w, ix.ln_b, st.ik, st.ig, p32, st.pos_dev, c.index_eps, r32);
    try k.sparse.poolKeys(s, st.ik, st.ig, ix.ape, st.pk, p32, st.pos_dev, r32);
    try k.latent.absorbQ(s, sc.q, w.wk, sc.qa, r32);
    // rows within the limit keep the dense result; the sparse merge overwrites only rows with tokens
    if (!all_sparse) try denseAttn(k, s, c, st, sc, rows, pos, capture);
    if (sparse_rows) try sparseAttn(k, s, c, w, st, sc, rows, pos, capture);
    try k.latent.expandV(s, sc.ol, w.wv, sc.vn, r32);
    try k.dense.prompt(s.*, w.o, sc.vn, .bf16, rows, part, .f32);
}

test "pool buckets follow sparse.pool_bucket" {
    try std.testing.expectEqual(@as(usize, 1024), poolBucket(100, 1, 2560));
    try std.testing.expectEqual(@as(usize, 2048), poolBucket(8000, 8, 2560));
    try std.testing.expectEqual(@as(usize, 2560), poolBucket(9000, 8, 2560));
    try std.testing.expectEqual(@as(usize, 1024), poolBucket(4092, 3, 4096));
    try std.testing.expectEqual(@as(usize, 2048), poolBucket(4092, 4, 4096));
}
