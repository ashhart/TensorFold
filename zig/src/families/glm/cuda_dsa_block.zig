//! GLM-5.3-Flash's DSA block on CUDA, a rank's latent-path share (the Python family's dsa_block) to an fp32 partial.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const projection = @import("cuda_proj.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");
const prof = @import("cuda_prof.zig");
const kv_host = @import("cuda_kv_host.zig");

/// A prompt chunk's rows one dense latent attention call takes (geometry.MLA_PROMPT_ATT_ROWS).
pub const prompt_att_rows = 512;

/// The indexer's weights, replicated: [wk | weights_proj] [d + heads][dims], wq_b (as stored), gate, LayerNorm, bias.
pub const Index = struct { kw: projection.Proj, gate: u64, ln_w: u64, ln_b: u64, ape: u64, qb: projection.Proj, heads: usize, d: usize };

/// One DSA layer on a rank of `heads` heads: proj, norms, q_b, kv_b per head as wk/wv, o, indexer; shapes from config.
pub const Layer = struct {
    proj: projection.Proj, // [q_a | kv_a], q_b and o as the checkpoint stores them: bf16, block FP8 or the MTP's MXFP8
    q_norm: u64,
    kv_norm: u64,
    q_b: projection.Proj,
    wk: u64,
    wv: u64,
    o: projection.Proj,
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
pub const Caches = struct {
    lc: u64,
    ls: u64 = 0,
    ik: u64,
    ig: u64,
    pk: u64,
    pk_rows: usize,
    pt: u64 = 0, // on pages: the stream's table (i32 page ids), lc and pk then this cache's place in page 0; 0: rows
    ps: u64 = 0, // on pages: bytes from one page to the next
    pks: u64 = 0, // TF_GLM_KV_HOST: the pooled keys' own page stride (their pages on the device); 0: ps
    pkt: u64 = 0, // TF_GLM_KV_SPLIT: the pooled keys' own table (pt is then the latent rows' alone); 0: pt
};

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
pub const Kernels = struct { set: glue.Set, ops: cuda.Module, prof: ?*prof.Prof = null, split: ?sparse.Split = null };

/// forward.dsa_block (latent path) of `rows` rows at pos to an fp32 partial; `nch`: 512-key chunks, `part_rows`: block.
pub fn block(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, x: u64, x_stride: usize, rows: usize, pos: u64, host_pos: usize, nch: usize, part_rows: usize, out: u64) !void {
    try body(k, stream, l, c, s, x, x_stride, rows, pos, host_pos, nch, part_rows);
    try fill(k.set, stream, l, s, 0, rows, out);
}

/// forward.out_proj's fill for the DSA block: o_proj of rows lo..hi (s.vn) into out [hi - lo][dims] fp32.
pub fn fill(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, lo: usize, hi: usize, out: u64) !void {
    const w = l.heads * l.dv;
    try projection.matmul(set, stream, l.o, s.vn + lo * w * 2, w, out, s.mm_part, hi - lo, l.dims, w, true);
}

/// The block up to its out_proj (s.vn holds the expanded rows).
pub fn body(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, x: u64, x_stride: usize, rows: usize, pos: u64, host_pos: usize, nch: usize, part_rows: usize) !void {
    try prepareProfile(k.set, stream, l, s, x, x_stride, rows, sparseRows(l, host_pos, rows), k.prof);
    try attend(k, stream, l, c, s, 0, rows, pos, host_pos, nch, part_rows);
    const stamp = prof.span(k.prof, stream, .projections);
    defer stamp.end();
    try latent.expandV(k.set, stream, s.ol, l.wv, s.vn, rows, l.heads, l.dv, l.lw);
}

/// Whether a sequence at `host_pos` has rows past the dense limit among its next `rows` (the sparse pass).
pub fn sparseRows(l: Layer, host_pos: usize, rows: usize) bool {
    return host_pos + rows - 1 >= l.dense_limit;
}

/// The block's row-by-row part for a shared window: projections, norms, q_b, indexer, absorbed query, `qi`'s query.
pub fn prepare(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, qi: bool) !void {
    return prepareProfile(set, stream, l, s, x, x_stride, rows, qi, null);
}

pub fn prepareProfile(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, qi: bool, profile: ?*prof.Prof) !void {
    var stamp = prof.span(profile, stream, .replicated_qakva);
    const width = l.q_lora + l.lw;
    try projection.matmul(set, stream, l.proj, x, x_stride, s.dp, s.mm_part, rows, width, l.dims, false);
    try glue.rmsnorm(set, stream, s.dp, width, l.q_norm, s.qr, l.q_lora, s.xs_qr, l.eps, rows, l.q_lora);
    try glue.rmsnorm(set, stream, s.dp + l.q_lora * 2, width, l.kv_norm, s.lat, l.lw, s.xs_lat, l.eps, rows, l.lw);
    stamp.end();
    stamp = prof.span(profile, stream, .projections);
    defer stamp.end();
    try projection.matmul(set, stream, l.q_b, s.qr, l.q_lora, s.q, s.mm_part, rows, l.heads * l.qk, l.q_lora, false);
    const ix = l.index;
    try projection.matmul(set, stream, ix.kw, x, x_stride, s.ikr, s.mm_part, rows, ix.d + ix.heads, l.dims, false);
    try glue.router(set, stream, x, x_stride, ix.gate, s.igr, s.router_part, rows, l.dims, ix.d);
    try latent.absorbQ(set, stream, s.q, l.wk, s.qa, rows, l.heads, l.qk, l.lw);
    if (qi) try projection.matmul(set, stream, ix.qb, s.qr, l.q_lora, s.qi, s.mm_part, rows, ix.heads * ix.d, l.q_lora, false);
}

/// One sequence's rows r0..r0 + rows at pos: latent and index keys written, dense then sparse attention into s.ol.
pub fn attend(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, r0: usize, rows: usize, pos: u64, host_pos: usize, nch: usize, part_rows: usize) !void {
    const set = k.set;
    const ix = l.index;
    const kw_rows = ix.d + ix.heads;
    const v = rowScratch(l, s, r0);
    try writeKeys(k, stream, l, c, v, rows, pos);
    const t, const tk = try tables(l, c);
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(l.qk), -0.5));
    if (host_pos < l.dense_limit) {
        const stamp = prof.span(k.prof, stream, .mla);
        var dc = c;
        if (c.pks != 0) if (kv_host.active) |*g| { // host rows: the visible ones, in order, into the slots
            dc = .{ .lc = try g.visible(k.ops, stream, c.lc, c.pt, t.page_rows, pos, rows, l.dense_limit + rows), .ik = c.ik, .ig = c.ig, .pk = c.pk, .pk_rows = c.pk_rows };
        };
        try denseAttention(k, stream, l, dc, v, pos, rows, nch, part_rows, scale);
        stamp.end();
    }
    if (sparseRows(l, host_pos, rows)) {
        try sparse.selectTokensProfile(set, k.ops, stream, v.qi, v.ikr + ix.d * 2, kw_rows, c.pk, tk, t, pos, host_pos, rows, ix.heads, ix.d, c.pk_rows - 2, v.select, v.tokens, v.counts, k.split, k.prof);
        const stamp = prof.span(k.prof, stream, .mla);
        defer stamp.end();
        if (c.pks != 0) if (kv_host.active) |*g| { // host rows: gather the picks into device slots, attend there
            const slots = try g.gather(k.ops, stream, ix.gate, c.lc, c.pt, t.page_rows, pos, v.tokens, v.counts, rows, sparse.width);
            return latent.sparseAttentionFormat(set, stream, v.qa, slots, 0, .{}, v.tokens, v.counts, sparse.width, v.partials, v.ol, rows, l.heads, l.lw, scale);
        };
        try latent.sparseAttentionFormat(set, stream, v.qa, c.lc, c.ls, t, v.tokens, v.counts, sparse.width, v.partials, v.ol, rows, l.heads, l.lw, scale);
    }
}

/// The scratch from row r0 (the token lists and partials stay at the front).
fn rowScratch(l: Layer, s: Scratch, r0: usize) Scratch {
    const ix = l.index;
    var v = s;
    v.lat += r0 * l.lw * 2;
    v.ikr += r0 * (ix.d + ix.heads) * 2;
    v.igr += r0 * ix.d * 4;
    v.qa += r0 * l.heads * l.lw * 2;
    v.ol += r0 * l.heads * l.lw * 2;
    v.qi += r0 * ix.heads * ix.d * 2;
    return v;
}

/// TF_GLM_KV_SPLIT, a rank without the latent rows: attend's writes (its latent rows to the trash), its token lists.
pub fn attendNone(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, r0: usize, rows: usize, pos: u64, host_pos: usize) !void {
    const ix = l.index;
    const v = rowScratch(l, s, r0);
    try writeKeys(k, stream, l, c, v, rows, pos);
    if (!sparseRows(l, host_pos, rows)) return;
    const t, const tk = try tables(l, c); // every rank picks (a split selection gathers every rank's rows)
    try sparse.selectTokensProfile(k.set, k.ops, stream, v.qi, v.ikr + ix.d * 2, ix.d + ix.heads, c.pk, tk, t, pos, host_pos, rows, ix.heads, ix.d, c.pk_rows - 2, v.select, v.tokens, v.counts, k.split, k.prof);
}

/// TF_GLM_KV_SPLIT, the owner: attend's attention alone for another rank's `heads` heads, queries `qa` into `ol`.
pub fn attendPeer(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, rows: usize, pos: u64, host_pos: usize, nch: usize, part_rows: usize, qa: u64, ol: u64, heads: usize) !void {
    const t, _ = try tables(l, c);
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(l.qk), -0.5));
    var p = l;
    p.heads = heads;
    var v = s; // its token lists, made by this rank's own pass of the same rows, are the scratch's first rows
    v.qa = qa;
    v.ol = ol;
    // the partials hold this rank's heads: a wider share (TP=3's 22 against 21) takes fewer rows a call
    const fit = if (heads <= l.heads) rows else @max(1, rows * l.heads / heads);
    if (host_pos < l.dense_limit) {
        const stamp = prof.span(k.prof, stream, .mla);
        defer stamp.end();
        try denseAttention(k, stream, p, c, v, pos, rows, nch, if (heads <= l.heads) part_rows else @max(1, part_rows * l.heads / heads), scale);
    }
    if (!sparseRows(l, host_pos, rows)) return;
    const stamp = prof.span(k.prof, stream, .mla);
    defer stamp.end();
    var r0: usize = 0;
    while (r0 < rows) : (r0 += fit) {
        const n = @min(fit, rows - r0);
        const q = r0 * heads * l.lw * 2;
        try latent.sparseAttentionBlock(k.set, stream, qa + q, c.lc, c.ls, t, v.tokens + r0 * sparse.width * 4, v.counts + r0 * 4, sparse.width, v.partials, ol + q, n, heads, l.lw, scale, latent.headBlock(rows));
    }
}

/// TF_GLM_KV_SPLIT, the owner: attendLastRow's attention for another rank's `heads` heads, after its own pass.
pub fn lastRowPeer(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, rows: usize, pos: u64, host_pos: usize, nch: usize, qa: u64, ol: u64, heads: usize) !void {
    const t, _ = try tables(l, c);
    const at = if (rows > 1) s.at else pos; // the last row's position, as the own pass put it
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(l.qk), -0.5));
    const stamp = prof.span(k.prof, stream, .mla);
    defer stamp.end();
    if (host_pos < l.dense_limit) try latent.attentionFormat(k.set, stream, qa, c.lc, c.ls, .{ .pt = c.pt, .ps = c.ps }, at, s.partials, ol, s.dummy, 1, heads, l.lw, nch, scale, latent.headBlock(rows));
    if (sparseRows(l, host_pos, rows)) try latent.sparseAttentionFormat(k.set, stream, qa, c.lc, c.ls, t, s.tokens, s.counts, sparse.width, s.partials, ol, 1, heads, l.lw, scale);
}

/// A prompt MTP chunk's block: every row's caches, then the last row's attention and o_proj partial into `out` row 0.
pub fn blockLastRow(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, x: u64, x_stride: usize, rows: usize, pos: u64, host_pos: usize, nch: usize, out: u64) !void {
    if (rows == 0) return error.BadWindow;
    const t, const tk = try tables(l, c);
    const ix = l.index;
    const kw_rows = ix.d + ix.heads;
    try prepareProfile(k.set, stream, l, s, x, x_stride, rows, sparseRows(l, host_pos, rows), k.prof); // every row keeps its query, its latent, its keys
    var stamp = prof.span(k.prof, stream, .cache_write);
    try latent.latentWriteFormat(k.set, stream, s.lat, l.lw, c.lc, c.ls, t, pos, rows, l.lw); // all rows, at pos, before any of them attends
    stamp.end();
    stamp = prof.span(k.prof, stream, .index_update);
    try sparse.indexUpdate(k.set, stream, s.ikr, kw_rows, s.igr, ix.ln_w, ix.ln_b, ix.ape, c.ik, c.ig, c.pk, tk, pos, rows);
    stamp.end();
    try attendLastRow(k, stream, l, c, s, rows, pos, host_pos, nch);
    const pstamp = prof.span(k.prof, stream, .projections);
    defer pstamp.end();
    const off = (rows - 1) * l.heads * l.lw * 2;
    try latent.expandV(k.set, stream, s.ol + off, l.wv, s.vn, 1, l.heads, l.dv, l.lw); // its o, into vn's row 0
    try fill(k.set, stream, l, s, 0, 1, out);
}

/// A chunk's last row attending over caches already written: its own query, causal limit and token list.
pub fn attendLastRow(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, rows: usize, pos: u64, host_pos: usize, nch: usize) !void {
    const t, const tk = try tables(l, c);
    const ix = l.index;
    const r0 = rows - 1;
    var v = s; // the scratch from the last row (its token lists and partials stay at the front)
    v.qa += r0 * l.heads * l.lw * 2;
    v.ol += r0 * l.heads * l.lw * 2;
    v.qi += r0 * ix.heads * ix.d * 2;
    var at = pos; // the last row's own position on the device: pos + r0, so its causal limit is unchanged
    if (r0 > 0) {
        var a: cuda.Args = .{};
        a.add(pos);
        a.add(@as(c_int, @intCast(r0)));
        a.add(v.at);
        try cuda.launch.launch(try k.ops.function("tf_glm_add_i32"), .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, stream, &a);
        at = v.at;
    }
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(l.qk), -0.5));
    if (host_pos < l.dense_limit) { // the same keys today's whole-chunk call read for this row
        var stamp = prof.span(k.prof, stream, .mla);
        try latent.attentionFormat(k.set, stream, v.qa, c.lc, c.ls, .{ .pt = c.pt, .ps = c.ps }, at, v.partials, v.ol, v.dummy, 1, l.heads, l.lw, nch, scale, latent.headBlock(rows));
        stamp.end();
    }
    if (sparseRows(l, host_pos, rows)) {
        // one row at host_pos + r0: the same pool bucket, score columns and token list its block computed for it
        try sparse.selectTokensProfile(k.set, k.ops, stream, v.qi, s.ikr + r0 * (ix.d + ix.heads) * 2 + ix.d * 2, ix.d + ix.heads, c.pk, tk, t, at, host_pos + r0, 1, ix.heads, ix.d, c.pk_rows - 2, v.select, v.tokens, v.counts, null, k.prof);
        var stamp = prof.span(k.prof, stream, .mla);
        defer stamp.end();
        try latent.sparseAttentionFormat(k.set, stream, v.qa, c.lc, c.ls, t, v.tokens, v.counts, sparse.width, v.partials, v.ol, 1, l.heads, l.lw, scale);
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
        try latent.attentionFormat(k.set, stream, s.qa + off, c.lc, c.ls, .{ .pt = c.pt, .ps = c.ps }, at, s.partials, s.ol + off, s.dummy, n, l.heads, l.lw, nch, scale, hb);
    }
}

/// A cache's tables: latent rows `t`, pooled keys `tk` (TF_GLM_KV_HOST: own pages; TF_GLM_KV_SPLIT: own table too).
fn tables(l: Layer, c: Caches) !struct { latent.Table, latent.Table } {
    if (c.ps % (l.lw * 2) != 0) return error.LatentShape; // pages whole latent rows apart (the token lists' unit)
    if (c.pkt != 0 and c.pks == 0) return error.LatentShape; // a region of pooled keys has its own stride
    const t: latent.Table = .{ .pt = c.pt, .ps = c.ps, .page_rows = c.ps / (l.lw * 2) };
    return .{ t, .{ .pt = if (c.pkt != 0) c.pkt else c.pt, .ps = if (c.pks != 0) c.pks else c.ps, .page_rows = t.page_rows } };
}

pub fn writeKeys(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, rows: usize, pos: u64) !void {
    const set = k.set;
    const ix = l.index;
    const kw_rows = ix.d + ix.heads;
    const t, const tk = try tables(l, c);
    var stamp = prof.span(k.prof, stream, .cache_write);
    try latent.latentWriteFormat(set, stream, s.lat, l.lw, c.lc, c.ls, t, pos, rows, l.lw);
    stamp.end();
    stamp = prof.span(k.prof, stream, .index_update);
    try sparse.indexUpdate(set, stream, s.ikr, kw_rows, s.igr, ix.ln_w, ix.ln_b, ix.ape, c.ik, c.ig, c.pk, tk, pos, rows);
    stamp.end();
}

/// The existing row projections and cache writes, without attention or output projections.
pub fn commit(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, x: u64, stride: usize, rows: usize, pos: u64, host_pos: usize) !void {
    try prepareProfile(k.set, stream, l, s, x, stride, rows, sparseRows(l, host_pos, rows), k.prof);
    try writeKeys(k, stream, l, c, s, rows, pos);
}
