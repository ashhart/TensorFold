//! GLM-5.3-Flash's DSA block on CUDA, a rank's latent-path share (the Python family's dsa_block) to an fp32 partial.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const b16 = @import("cuda_b16.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");

/// A prompt chunk's rows one dense latent attention call takes (geometry.MLA_PROMPT_ATT_ROWS).
pub const prompt_att_rows = 512;

/// The indexer's weights, replicated: [wk | weights_proj] [d + heads][dims], gate, key LayerNorm, pool bias, wq_b.
pub const Index = struct { kw: u64, gate: u64, ln_w: u64, ln_b: u64, ape: u64, qb: u64, heads: usize, d: usize };

/// One DSA layer on a rank of `heads` heads: proj, norms, q_b, kv_b per head as wk/wv, o, indexer; shapes from config.
pub const Layer = struct {
    proj: u64,
    q_norm: u64,
    kv_norm: u64,
    q_b: u64,
    wk: u64,
    wv: u64,
    o: u64,
    index: Index,
    heads: usize,
    qk: usize,
    dv: usize,
    lw: usize,
    q_lora: usize,
    dims: usize,
    eps: f32,
    dense_limit: usize, // the largest context whose top-k selection keeps every visible key (Config.keyWidth)
};

/// The caches: latent bf16 [capacity][lw], index keys/gates bf16 [capacity][128], pooled keys bf16 [pk_rows][128].
pub const Caches = struct { lc: u64, ik: u64, ig: u64, pk: u64, pk_rows: usize };

/// The block's scratch for up to `rows` rows.
pub const Scratch = struct {
    dp: u64, // bf16 [rows][q_lora + lw]
    qr: u64, // bf16 [rows][q_lora]
    xs_qr: u64, // fp32 [rows][q_lora / 64]
    lat: u64, // bf16 [rows][lw]
    xs_lat: u64, // fp32 [rows][lw / 64]
    q: u64, // bf16 [rows][heads qk]
    ikr: u64, // bf16 [rows][160]
    igr: u64, // fp32 [rows][128]
    router_part: u64, // fp32 [router_ks][rows][128]
    qa: u64, // bf16 [rows][heads][lw]
    ol: u64, // bf16 [rows][heads][lw]
    vn: u64, // bf16 [rows][heads dv]
    qi: u64, // bf16 [rows][index heads d]
    partials: latent.Partials,
    dummy: u64, // an i32
    select: sparse.Scratch,
    tokens: u64, // i32 [rows][sparse.width]
    counts: u64, // i32 [rows]
    mm_part: u64, // fp32 split-K partials
    at: u64, // an i32: a prompt block's first position (pos + its first row)
};

/// The kernels: the captured Triton set and glm_ops (the token lists and position offsets).
pub const Kernels = struct { set: glue.Set, ops: cuda.Module };

/// forward.dsa_block (latent path) of `rows` rows at pos to an fp32 partial; `nch`: 512-key chunks, `part_rows`: block.
pub fn block(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, x: u64, x_stride: usize, rows: usize, pos: u64, host_pos: usize, nch: usize, part_rows: usize, out: u64) !void {
    try body(k, stream, l, c, s, x, x_stride, rows, pos, host_pos, nch, part_rows);
    try fill(k.set, stream, l, s, 0, rows, out);
}

/// forward.out_proj's fill for the DSA block: o_proj of rows lo..hi (s.vn) into out [hi - lo][dims] fp32.
pub fn fill(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, lo: usize, hi: usize, out: u64) !void {
    const w = l.heads * l.dv;
    try b16.matmul(set, stream, s.vn + lo * w * 2, w, l.o, out, s.mm_part, hi - lo, l.dims, w, true);
}

/// The block up to its out_proj (s.vn holds the expanded rows).
pub fn body(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, x: u64, x_stride: usize, rows: usize, pos: u64, host_pos: usize, nch: usize, part_rows: usize) !void {
    try prepare(k.set, stream, l, s, x, x_stride, rows, sparseRows(l, host_pos, rows));
    try attend(k, stream, l, c, s, 0, rows, pos, host_pos, nch, part_rows);
    try latent.expandV(k.set, stream, s.ol, l.wv, s.vn, rows, l.heads, l.dv, l.lw);
}

/// Whether a sequence at `host_pos` has rows past the dense limit among its next `rows` (the sparse pass).
pub fn sparseRows(l: Layer, host_pos: usize, rows: usize) bool {
    return host_pos + rows - 1 >= l.dense_limit;
}

/// The block's row-by-row part for a shared window: projections, norms, q_b, indexer, absorbed query, `qi`'s query.
pub fn prepare(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, qi: bool) !void {
    const width = l.q_lora + l.lw;
    try b16.matmul(set, stream, x, x_stride, l.proj, s.dp, s.mm_part, rows, width, l.dims, false);
    try glue.rmsnorm(set, stream, s.dp, width, l.q_norm, s.qr, l.q_lora, s.xs_qr, l.eps, rows, l.q_lora);
    try glue.rmsnorm(set, stream, s.dp + l.q_lora * 2, width, l.kv_norm, s.lat, l.lw, s.xs_lat, l.eps, rows, l.lw);
    try b16.matmul(set, stream, s.qr, l.q_lora, l.q_b, s.q, s.mm_part, rows, l.heads * l.qk, l.q_lora, false);
    const ix = l.index;
    try b16.matmul(set, stream, x, x_stride, ix.kw, s.ikr, s.mm_part, rows, ix.d + ix.heads, l.dims, false);
    try glue.router(set, stream, x, x_stride, ix.gate, s.igr, s.router_part, rows, l.dims, ix.d);
    try latent.absorbQ(set, stream, s.q, l.wk, s.qa, rows, l.heads, l.qk, l.lw);
    if (qi) try b16.matmul(set, stream, s.qr, l.q_lora, ix.qb, s.qi, s.mm_part, rows, ix.heads * ix.d, l.q_lora, false);
}

/// One sequence's rows r0..r0 + rows at pos: latent and index keys written, dense then sparse attention into s.ol.
pub fn attend(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, r0: usize, rows: usize, pos: u64, host_pos: usize, nch: usize, part_rows: usize) !void {
    const set = k.set;
    const ix = l.index;
    const kw_rows = ix.d + ix.heads;
    var v = s; // the scratch from row r0
    v.lat += r0 * l.lw * 2;
    v.ikr += r0 * kw_rows * 2;
    v.igr += r0 * ix.d * 4;
    v.qa += r0 * l.heads * l.lw * 2;
    v.ol += r0 * l.heads * l.lw * 2;
    v.qi += r0 * ix.heads * ix.d * 2;
    try latent.latentWrite(set, stream, v.lat, l.lw, c.lc, pos, rows, l.lw);
    try sparse.indexUpdate(set, stream, v.ikr, kw_rows, v.igr, ix.ln_w, ix.ln_b, ix.ape, c.ik, c.ig, c.pk, pos, rows);
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(l.qk), -0.5));
    if (host_pos < l.dense_limit) try denseAttention(k, stream, l, c, v, pos, rows, nch, part_rows, scale);
    if (sparseRows(l, host_pos, rows)) {
        try sparse.selectTokens(set, k.ops, stream, v.qi, v.ikr + ix.d * 2, kw_rows, c.pk, pos, host_pos, rows, ix.heads, ix.d, c.pk_rows - 2, v.select, v.tokens, v.counts);
        try latent.sparseAttention(set, stream, v.qa, c.lc, v.tokens, v.counts, sparse.width, v.partials, v.ol, rows, l.heads, l.lw, scale);
    }
}

/// forward.dense_attention: latent.attention in `part_rows`-row blocks, each at its first row's position.
fn denseAttention(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, pos: u64, rows: usize, nch: usize, part_rows: usize, scale: f32) !void {
    const hb = latent.headBlock(rows);
    var r0: usize = 0;
    while (r0 < rows) : (r0 += part_rows) {
        const n = @min(part_rows, rows - r0);
        var at = pos;
        if (r0 > 0) {
            var a: cuda.Args = .{};
            a.add(pos);
            a.add(@as(c_int, @intCast(r0)));
            a.add(s.at);
            try cuda.launch.launch(try k.ops.function("tf_glm_add_i32"), .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, stream, &a);
            at = s.at;
        }
        const off = r0 * l.heads * l.lw * 2;
        try latent.attention(k.set, stream, s.qa + off, c.lc, at, s.partials, s.ol + off, s.dummy, n, l.heads, l.lw, nch, scale, hb);
    }
}
