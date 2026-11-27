//! GLM-5.3-Flash's DSA block on CUDA, a rank's latent-path share (the Python family's dsa_block) to an fp32 partial.

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const cuda_pdl = @import("cuda_pdl.zig");
const glue = @import("cuda_glue.zig");
const projection = @import("cuda_proj.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");
const prof = @import("cuda_prof.zig");
const kv_host = @import("cuda_kv_host.zig");
const shard = @import("cuda_shard.zig");

/// A prompt chunk's rows one dense latent attention call takes (geometry.MLA_PROMPT_ATT_ROWS).
pub const prompt_att_rows = 512;

/// The indexer's weights, replicated: [wk | weights_proj] [d + heads][dims], wq_b (as stored), gate, LayerNorm, bias.
pub const Index = struct {
    kw: projection.Proj,
    gate: u64, // GLM-5.3-Flash's pool gate (0: pools of 1, keys as written)
    ln_w: u64,
    ln_b: u64,
    ape: u64,
    qb: projection.Proj,
    qb_columns: ?shard.Span = null,
    heads: usize,
    d: usize,
    geom: sparse.Geometry,
    own: bool = true, // false: a shared layer (full GLM-5.3) attends the token lists its indexed layer left in Scratch
    apart: bool = false, // full GLM-5.3: each sequence's token lists at its own rows, for the shared layers after it
};

/// One DSA layer on a rank of `heads` heads: proj, norms, q_b, kv_b per head as wk/wv, o, indexer; shapes from config.
pub const Layer = struct {
    proj: projection.Proj, // [q_a | kv_a], q_b and o as the checkpoint stores them: bf16, block FP8 or the MTP's MXFP8
    proj_columns: ?shard.Span = null,
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
    rope: usize = 0, // qk_rope: roped dims past qk in a q head and past lw in a latent row (none on GLM-5.3-Flash)
    rope_inv: u64 = 0, // their fp32 frequencies [rope / 2] on the device

    /// A latent cache row and an absorbed q row: the latent, then the roped key columns (512 or 576).
    pub fn row(l: Layer) usize {
        return l.lw + l.rope;
    }

    /// A q head's columns from q_b: qk latent-path dims, then the roped ones.
    pub fn qHead(l: Layer) usize {
        return l.qk + l.rope;
    }

    /// The softmax scale: 1 / sqrt(qk + rope), 1 / 16 on both.
    pub fn scale(l: Layer) f32 {
        return @floatCast(std.math.pow(f64, @floatFromInt(l.qHead()), -0.5));
    }
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
};

/// The block's scratch for up to `rows` rows.
pub const Scratch = struct {
    dp: u64, // bf16 [rows][q_lora + row]
    qr: u64, // bf16 [rows][q_lora]
    xs_qr: u64, // fp32 [rows][q_lora / 64]
    lat: u64, // bf16 [rows][row]: the normed latent, then the roped key
    xs_lat: u64, // fp32 [rows][lw / 64]
    q: u64, // bf16 [rows][heads qHead]
    ikr: u64, // bf16 [rows][160]
    igr: u64, // fp32 [rows][128]
    router_part: u64, // fp32 [router_ks][rows][128]
    qa: u64, // bf16 [rows][heads][row]: the absorbed q, then the roped q
    ol: u64, // bf16 [rows][heads][lw]
    vn: u64, // bf16 [rows][heads dv]
    qi: u64, // bf16 [rows][index heads d]
    partials: latent.Partials,
    dummy: u64, // an i32
    select: sparse.Scratch,
    tokens: u64, // i32 [rows][index width]
    counts: u64, // i32 [rows]
    mm_part: u64, // fp32 split-K partials
    at: u64, // an i32: a prompt block's first position (pos + its first row)
};

/// One projection's column exchange, on the same stream as its matmul.
pub const Columns = struct {
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    collective: ?tp.Collective,
    rank: usize,
    world: usize,
};

/// The kernels: the captured Triton set and glm_ops (the token lists and position offsets).
pub const Kernels = struct {
    set: glue.Set,
    ops: cuda.Module,
    prof: ?*prof.Prof = null,
    split: ?sparse.Split = null,
    columns: ?Columns = null,
};

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
    try prepareMode(k.set, stream, l, s, x, x_stride, rows, sparseRows(l, host_pos, rows), k.prof, false, k.columns);
    try attend(k, stream, l, c, s, 0, rows, pos, host_pos, nch, part_rows);
    const stamp = prof.span(k.prof, stream, .projections);
    defer stamp.end();
    try latent.expandV(k.set, stream, s.ol, l.wv, s.vn, rows, l.heads, l.dv, l.lw);
}

/// Whether a sequence at `host_pos` has rows past the dense limit among its next `rows` (the sparse pass).
pub fn sparseRows(l: Layer, host_pos: usize, rows: usize) bool {
    return host_pos + rows - 1 >= l.dense_limit;
}

/// The default-off shared-window preparation, source-compatible with the old set-only call.
pub fn prepare(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, qi: bool) !void {
    return prepareProfile(set, stream, l, s, x, x_stride, rows, qi, null);
}

pub fn prepareProfile(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, qi: bool, profile: ?*prof.Prof) !void {
    return prepareMode(set, stream, l, s, x, x_stride, rows, qi, profile, false, null);
}

/// The same preparation with the rank's optional output-column exchange.
pub fn prepareKernels(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, qi: bool) !void {
    return prepareMode(k.set, stream, l, s, x, x_stride, rows, qi, k.prof, false, k.columns);
}

const ActiveColumns = struct { span: shard.Span, exchange: Columns };

fn activeColumns(metadata: ?shard.Span, exchange: ?Columns, width: usize) !?ActiveColumns {
    const span = metadata orelse return null;
    const x = exchange orelse return error.MissingDsaColumnExchange;
    const want = shard.Share.outputColumns(width, x.rank, x.world) catch return error.BadDsaColumnShard;
    if (span.lo != want.lo or span.hi != want.hi) return error.BadDsaColumnShard;
    return .{ .span = span, .exchange = x };
}

fn gatherColumns(x: Columns, stream: cuda.Stream, src: u64, dst: u64, count: usize) !void {
    if (x.collective) |coll| return coll.allGather(src, dst, count, .bf16, stream.handle) catch |e| return tp.group.underlying(coll, e);
    try x.lib.check(x.lib.api.ncclAllGather(src, dst, count, .bf16, x.comm, stream.handle), "ncclAllGather");
}

/// Rank-major [world][rows][local] bf16 blocks into full row-major [rows][full].
pub fn unpackColumns(set: glue.Set, stream: cuda.Stream, stage: u64, out: u64, local: usize, full: usize, rows: usize, world: usize) !void {
    if (world == 0 or full % world != 0 or full / world != local) return error.BadDsaColumnShard;
    const f = set.copy_rows orelse return error.FormatNotLoaded;
    for (0..world) |rank| {
        var a: cuda.Args = .{};
        a.add(stage + rank * rows * local * 2);
        a.add(@as(c_longlong, @intCast(local * 2)));
        a.add(out + rank * local * 2);
        a.add(@as(c_longlong, @intCast(full * 2)));
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(local * 2)));
        try cuda_pdl.launch(f, .{ .grid = .{ .x = @intCast((rows * local * 2 + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
    }
}

fn columnMatmul(set: glue.Set, stream: cuda.Stream, w: projection.Proj, x: u64, x_stride: usize, out: u64, stage: u64, part: u64, rows: usize, full: usize, k: usize, columns: ActiveColumns) !void {
    const local = columns.span.len();
    const dst = if (rows == 1) out + columns.span.lo * 2 else out;
    try projection.matmul(set, stream, w, x, x_stride, dst, part, rows, local, k, false);
    try gatherColumns(columns.exchange, stream, dst, if (rows == 1) out else stage, rows * local);
    if (rows > 1) try unpackColumns(set, stream, stage, out, local, full, rows, columns.exchange.world);
}

/// Keys-only commits keep KV and index keys; compact KV uses the same tiled K reduction.
fn prepareMode(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, qi: bool, profile: ?*prof.Prof, keys_only: bool, exchange: ?Columns) !void {
    const width = l.q_lora + l.row();
    const ix = l.index;
    const pc = try activeColumns(l.proj_columns, exchange, width);
    const qc = if (!keys_only and qi and ix.own) try activeColumns(ix.qb_columns, exchange, ix.heads * ix.d) else null;
    if (pc != null and l.proj != .bf16) return error.BadDsaColumnShard;
    if (qc != null and ix.qb != .bf16) return error.BadDsaColumnShard;
    if (rows > 1 and (pc != null or qc != null) and set.copy_rows == null) return error.FormatNotLoaded;
    if (rows > 1 and qc != null and (l.rope != 0 or ix.apart or ix.geom.rope != 0)) return error.BadDsaColumnShard;
    var stamp = prof.span(profile, stream, .replicated_qakva);
    const compact = keys_only and compactKv(set, l);
    if (compact) {
        try projection.matmul(set, stream, .{ .bf16 = l.proj.bf16 + l.q_lora * l.dims * 2 }, x, x_stride, s.dp, s.mm_part, rows, l.row(), l.dims, false);
    } else if (pc) |columns| {
        try columnMatmul(set, stream, l.proj, x, x_stride, s.dp, s.qi, s.mm_part, rows, width, l.dims, columns);
    } else {
        try projection.matmul(set, stream, l.proj, x, x_stride, s.dp, s.mm_part, rows, width, l.dims, false);
    }
    if (!keys_only) try glue.rmsnorm(set, stream, s.dp, width, l.q_norm, s.qr, l.q_lora, s.xs_qr, l.eps, rows, l.q_lora);
    const kv_offset: u64 = if (compact) 0 else l.q_lora * 2;
    const kv_stride = if (compact) l.row() else width;
    try glue.rmsnorm(set, stream, s.dp + kv_offset, kv_stride, l.kv_norm, s.lat, l.row(), s.xs_lat, l.eps, rows, l.lw);
    stamp.end();
    stamp = prof.span(profile, stream, .projections);
    defer stamp.end();
    if (!keys_only) try projection.matmul(set, stream, l.q_b, s.qr, l.q_lora, s.q, s.mm_part, rows, l.heads * l.qHead(), l.q_lora, false);
    if (ix.own) try projection.matmul(set, stream, ix.kw, x, x_stride, s.ikr, s.mm_part, rows, ix.d + ix.heads, l.dims, false);
    if (ix.own and ix.gate != 0) try glue.router(set, stream, x, x_stride, ix.gate, s.igr, s.router_part, rows, l.dims, ix.d);
    if (!keys_only) {
        try latent.absorbQ(set, stream, s.q, l.wk, s.qa, rows, l.heads, l.qk, l.rope, l.lw);
        if (qc) |columns| {
            try columnMatmul(set, stream, ix.qb, s.qr, l.q_lora, s.qi, s.q, s.mm_part, rows, ix.heads * ix.d, l.q_lora, columns);
        } else if (!keys_only and qi and ix.own) {
            try projection.matmul(set, stream, ix.qb, s.qr, l.q_lora, s.qi, s.mm_part, rows, ix.heads * ix.d, l.q_lora, false);
        }
    }
}

fn compactKv(set: glue.Set, l: Layer) bool {
    if (l.proj_columns != null or !set.prompt or set.packs != null or l.rope != 0 or l.q_lora % 8 != 0 or l.proj != .bf16) return false;
    return if (set.mm) |mk| !mk.prompt_fixed_k else false;
}

/// One sequence's rows r0..r0 + rows at pos: latent and index keys written, dense then sparse attention into s.ol.
pub fn attend(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, r0: usize, rows: usize, pos: u64, host_pos: usize, nch: usize, part_rows: usize) !void {
    const set = k.set;
    const ix = l.index;
    const kw_rows = ix.d + ix.heads;
    var v = s; // the scratch from row r0
    v.dp += r0 * (l.q_lora + l.row()) * 2;
    v.lat += r0 * l.row() * 2;
    v.ikr += r0 * kw_rows * 2;
    v.igr += r0 * ix.d * 4;
    v.q += r0 * l.heads * l.qHead() * 2;
    v.qa += r0 * l.heads * l.row() * 2;
    v.ol += r0 * l.heads * l.lw * 2;
    v.qi += r0 * ix.heads * ix.d * 2;
    if (ix.apart) { // each sequence's token lists at its own rows: a shared layer reads them after every one has picked
        v.tokens += r0 * ix.geom.width() * 4;
        v.counts += r0 * 4;
    }
    try writeKeys(k, stream, l, c, v, rows, pos);
    try ropeQueries(set, stream, l, v, rows, pos);
    const t, const tk = try tables(l, c);
    const scale = l.scale();
    if (host_pos < l.dense_limit) {
        const dense_rows = @min(rows, l.dense_limit - host_pos);
        const stamp = prof.span(k.prof, stream, .mla);
        var dc = c;
        if (c.pks != 0) if (kv_host.active) |*g| { // host rows: the visible ones, in order, into the slots
            dc = .{ .lc = try g.visible(k.ops, stream, c.lc, c.pt, t.page_rows, pos, rows, l.dense_limit + rows), .ik = c.ik, .ig = c.ig, .pk = c.pk, .pk_rows = c.pk_rows };
        };
        try denseAttention(k, stream, l, dc, v, pos, dense_rows, nch, part_rows, scale, latent.headBlockFor(rows, l.row()));
        stamp.end();
    }
    if (sparseRows(l, host_pos, rows)) {
        // a shared layer's rows attend the lists its indexed layer chose for these rows (left in v.tokens, v.counts)
        if (ix.own) try sparse.selectTokensProfile(set, k.ops, stream, v.qi, v.ikr + ix.d * 2, kw_rows, c.pk, tk, pos, host_pos, rows, ix.heads, ix.d, ix.geom, c.pk_rows - 2, v.select, v.tokens, v.counts, k.split, k.prof);
        const stamp = prof.span(k.prof, stream, .mla);
        defer stamp.end();
        if (c.pks != 0) if (kv_host.active) |*g| { // host rows: gather the picks into device slots, attend there
            const slots = try g.gather(k.ops, stream, ix.gate, c.lc, c.pt, t.page_rows, pos, v.tokens, v.counts, rows, ix.geom.width());
            return latent.sparseAttentionFormat(set, stream, v.qa, slots, 0, .{}, v.tokens, v.counts, ix.geom.width(), v.partials, v.ol, rows, l.heads, l.row(), scale);
        };
        try latent.sparseAttentionFormat(set, stream, v.qa, c.lc, c.ls, t, v.tokens, v.counts, ix.geom.width(), v.partials, v.ol, rows, l.heads, l.row(), scale);
    }
}

/// A prompt MTP chunk's block: every row's caches, then the last row's attention and o_proj partial into `out` row 0.
pub fn blockLastRow(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, x: u64, x_stride: usize, rows: usize, pos: u64, host_pos: usize, nch: usize, out: u64) !void {
    if (rows == 0) return error.BadWindow;
    const t, const tk = try tables(l, c);
    const ix = l.index;
    const kw_rows = ix.d + ix.heads;
    try prepareMode(k.set, stream, l, s, x, x_stride, rows, sparseRows(l, host_pos, rows), k.prof, false, k.columns); // every row keeps its query, its latent, its keys
    var stamp = prof.span(k.prof, stream, .cache_write);
    try writeLatent(k.set, stream, l, c, s, t, pos, rows); // all rows, at pos, before any of them attends
    stamp.end();
    stamp = prof.span(k.prof, stream, .index_update);
    try sparse.indexUpdate(k.set, stream, s.ikr, kw_rows, s.igr, ix.ln_w, ix.ln_b, ix.ape, c.ik, c.ig, c.pk, tk, pos, rows, ix.geom);
    stamp.end();
    try attendLastRow(k, stream, l, c, s, rows, pos, host_pos, nch, t, tk);
    const pstamp = prof.span(k.prof, stream, .projections);
    defer pstamp.end();
    const off = (rows - 1) * l.heads * l.lw * 2;
    try latent.expandV(k.set, stream, s.ol + off, l.wv, s.vn, 1, l.heads, l.dv, l.lw); // its o, into vn's row 0
    try fill(k.set, stream, l, s, 0, 1, out);
}

/// A chunk's last row attending over caches already written: its own query, causal limit and token list.
fn attendLastRow(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, rows: usize, pos: u64, host_pos: usize, nch: usize, t: latent.Table, tk: latent.Table) !void {
    const ix = l.index;
    const r0 = rows - 1;
    var v = s; // the scratch from the last row (its token lists and partials stay at the front)
    v.q += r0 * l.heads * l.qHead() * 2;
    v.qa += r0 * l.heads * l.row() * 2;
    v.ol += r0 * l.heads * l.lw * 2;
    v.qi += r0 * ix.heads * ix.d * 2;
    var at = pos; // the last row's own position on the device: pos + r0, so its causal limit is unchanged
    if (r0 > 0) {
        var a: cuda.Args = .{};
        a.add(pos);
        a.add(@as(c_int, @intCast(r0)));
        a.add(v.at);
        try cuda_pdl.launch(try k.ops.function("tf_glm_add_i32"), .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, stream, &a);
        at = v.at;
    }
    try ropeQueries(k.set, stream, l, v, 1, at);
    const scale = l.scale();
    if (host_pos < l.dense_limit) { // the same keys today's whole-chunk call read for this row
        var stamp = prof.span(k.prof, stream, .mla);
        try latent.attentionFormat(k.set, stream, v.qa, c.lc, c.ls, .{ .pt = c.pt, .ps = c.ps }, at, v.partials, v.ol, v.dummy, 1, l.heads, l.row(), nch, scale, latent.headBlockFor(rows, l.row()));
        stamp.end();
    }
    if (sparseRows(l, host_pos, rows)) {
        // one row at host_pos + r0: the same pool bucket, score columns and token list its block computed for it
        try sparse.selectTokensProfile(k.set, k.ops, stream, v.qi, s.ikr + r0 * (ix.d + ix.heads) * 2 + ix.d * 2, ix.d + ix.heads, c.pk, tk, at, host_pos + r0, 1, ix.heads, ix.d, ix.geom, c.pk_rows - 2, v.select, v.tokens, v.counts, null, k.prof);
        var stamp = prof.span(k.prof, stream, .mla);
        defer stamp.end();
        try latent.sparseAttentionFormat(k.set, stream, v.qa, c.lc, c.ls, t, v.tokens, v.counts, ix.geom.width(), v.partials, v.ol, 1, l.heads, l.row(), scale);
    }
}

/// forward.dense_attention: latent.attention in `part_rows`-row blocks, each at its first row's position.
fn denseAttention(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, pos: u64, rows: usize, nch: usize, part_rows: usize, scale: f32, hb: usize) !void {
    var r0: usize = 0;
    while (r0 < rows) : (r0 += part_rows) {
        const n = @min(part_rows, rows - r0);
        var at = pos;
        if (r0 > 0) {
            var a: cuda.Args = .{};
            a.add(pos);
            a.add(@as(c_int, @intCast(r0)));
            a.add(s.at);
            try cuda_pdl.launch(try k.ops.function("tf_glm_add_i32"), .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, stream, &a);
            at = s.at;
        }
        const qa = s.qa + r0 * l.heads * l.row() * 2;
        try latent.attentionFormat(k.set, stream, qa, c.lc, c.ls, .{ .pt = c.pt, .ps = c.ps }, at, s.partials, s.ol + r0 * l.heads * l.lw * 2, s.dummy, n, l.heads, l.row(), nch, scale, hb);
    }
}

/// A cache's tables: latent rows `t`, and pooled keys `tk` (on their own device pages under TF_GLM_KV_HOST).
fn tables(l: Layer, c: Caches) !struct { latent.Table, latent.Table } {
    const rb = l.row() * 2;
    if (c.ps % rb != 0) return error.LatentShape; // pages whole latent rows apart (the token lists' unit)
    const t: latent.Table = .{ .pt = c.pt, .ps = c.ps, .page_rows = c.ps / rb };
    return .{ t, .{ .pt = c.pt, .ps = if (c.pks != 0) c.pks else c.ps, .page_rows = t.page_rows } };
}

fn writeKeys(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, rows: usize, pos: u64) !void {
    const set = k.set;
    const ix = l.index;
    const kw_rows = ix.d + ix.heads;
    const t, const tk = try tables(l, c);
    var stamp = prof.span(k.prof, stream, .cache_write);
    try writeLatent(set, stream, l, c, s, t, pos, rows);
    stamp.end();
    if (!ix.own) return; // a shared layer keeps no index keys
    stamp = prof.span(k.prof, stream, .index_update);
    try sparse.indexUpdate(set, stream, s.ikr, kw_rows, s.igr, ix.ln_w, ix.ln_b, ix.ape, c.ik, c.ig, c.pk, tk, pos, rows, ix.geom);
    stamp.end();
}

/// The latent rows into the cache at pos; a roped row's key columns roped first (k_pe from dp, at pos + r).
fn writeLatent(set: glue.Set, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, t: latent.Table, pos: u64, rows: usize) !void {
    if (l.rope != 0) {
        const kpe: latent.RopeRows = .{ .x = s.dp + (l.q_lora + l.lw) * 2, .x_row = l.q_lora + l.row(), .x_head = 0, .out = s.lat + l.lw * 2, .o_row = l.row(), .o_head = 0 };
        try latent.rope(set, stream, kpe, rows, 1, pos, l.rope_inv);
    }
    try latent.latentWriteFormat(set, stream, s.lat, l.row(), c.lc, c.ls, t, pos, rows, l.row());
}

/// A roped row's queries: each head's rope dims of q (past qk) roped at pos + r into qa past the latent.
fn ropeQueries(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, rows: usize, pos: u64) !void {
    if (l.rope == 0) return;
    const q: latent.RopeRows = .{ .x = s.q + l.qk * 2, .x_row = l.heads * l.qHead(), .x_head = l.qHead(), .out = s.qa + l.lw * 2, .o_row = l.heads * l.row(), .o_head = l.row() };
    try latent.rope(set, stream, q, rows, l.heads, pos, l.rope_inv);
}

/// Latent/index cache writes only: query projections and attention are unread on this path.
pub fn commit(k: Kernels, stream: cuda.Stream, l: Layer, c: Caches, s: Scratch, x: u64, stride: usize, rows: usize, pos: u64, host_pos: usize) !void {
    try prepareMode(k.set, stream, l, s, x, stride, rows, sparseRows(l, host_pos, rows), k.prof, true, k.columns);
    try writeKeys(k, stream, l, c, s, rows, pos);
}

test "GLM DSA column transport: missing exchange and malformed span or world refuse before any launch" {
    const l: Layer = .{
        .proj = .{ .bf16 = 0 },
        .proj_columns = .{ .lo = 0, .hi = 1024 },
        .q_norm = 0,
        .kv_norm = 0,
        .q_b = .{ .bf16 = 0 },
        .wk = 0,
        .wv = 0,
        .o = .{ .bf16 = 0 },
        .index = .{ .kw = .{ .bf16 = 0 }, .gate = 0, .ln_w = 0, .ln_b = 0, .ape = 0, .qb = .{ .bf16 = 0 }, .qb_columns = .{ .lo = 0, .hi = 2048 }, .heads = 64, .d = 64, .geom = .{} },
        .heads = 32,
        .qk = 512,
        .dv = 512,
        .lw = 512,
        .q_lora = 1536,
        .dims = 4096,
        .eps = 1e-6,
        .dense_limit = 0,
    };
    const set: glue.Set = .{ .glue = undefined };
    const stream: cuda.Stream = undefined;
    const s: Scratch = undefined;
    try std.testing.expectError(error.MissingDsaColumnExchange, prepare(set, stream, l, s, 0, 4096, 1, true));
    var k: Kernels = .{ .set = set, .ops = undefined, .columns = .{ .lib = undefined, .comm = null, .collective = null, .rank = 0, .world = 2 } };
    var bad = l;
    bad.proj_columns = .{ .lo = 64, .hi = 1088 };
    try std.testing.expectError(error.BadDsaColumnShard, prepareKernels(k, stream, bad, s, 0, 4096, 1, true));
    k.columns.?.world = 3;
    try std.testing.expectError(error.BadDsaColumnShard, prepareKernels(k, stream, l, s, 0, 4096, 1, true));
}
