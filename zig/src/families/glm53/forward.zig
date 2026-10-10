//! Full GLM-5.3 forward over a block of 1..max_rows consecutive rows (decode is one row; a prompt runs in blocks) on
//! one serial encoder: this process's share of heads, intermediate rows and vocab, its partials summed in slot order
//! (slot i = canonical slice i). Every row's arithmetic is the same at any block width, so a prompt read in blocks
//! leaves the bits a row-by-row read leaves. One Mac computes every slice; N Macs each compute their own and swap
//! partials through the fabric window inside the same command buffer.
const std = @import("std");
const mtl = @import("metal");
const m = @import("model.zig");
const Exchange = @import("exchange.zig").Exchange;
const Lw = @import("lw_gpu.zig").Lw;
const Ref = m.Ref;

const source = @embedFile("kernels.metal");

pub const QArgs = extern struct { K: i32, N: i32, ldw: i32, lds: i32, zw: i32, zs: i32, xr: i32, zx: i32, yr: i32, zy: i32, ids: i32, xids: i32, yids: i32, items: i32, segs: i32 };
const GArgs = extern struct { K: i32, N: i32, ldw: i32, xr: i32, yr: i32 };
const NArgs = extern struct { D: i32, eps: f32, round_bf16: i32, ld: i32 };
const SArgs = extern struct { D: i32, S: i32, stride: i32, eps: f32, norm: i32 };
const RArgs = extern struct { rows: i32, stride: i32, off: i32, dims: i32, pos: i32, log2base: f32, hpr: i32 };
const CArgs = extern struct { pos: i32, eps: f32, log2base: f32 };
pub const IArgs2 = extern struct { p0: i32, wscale: f32, cap: i32, keys: i32, span: i32, rows: i32 = 0 };
pub const TkArgs = extern struct { p0: i32, top: i32, cap: i32, chunk: i32, shift: i32, fixed: i32 = 0, lo: i32 = 0, rstride: i32 = 0 };
const TArgs = extern struct { p0: i32, top: i32, cap: i32 };
const AArgs = extern struct { p0: i32, sparse: i32, scale: f32, heads: i32, keys: i32, qp_stride: i32 };
const MArgs = extern struct { experts: i32, top: i32, scale: f32, direct: i32 };
const CbArgs = extern struct { D: i32, top: i32 };
const LwArgs = extern struct { K: i32, xr: i32, c0: i32, R: i32, rows: i32 };
/// The Living Weights site: layer 77 (the last trunk layer), its shared expert's down projection.
pub const lw_layer: usize = m.LAYERS - 1;

/// Launch classes, for a profile that times each alone.
pub const Op = enum { q_a, kv_a, q_norm, kv_store, q_b, rope_q, embed_q, idx_k, idx_q, idx_scores, idx_topk, attn, unembed, o_proj, sum, router, gateup, gateup_sh, down, down_sh, combine };

pub const Pipes = struct {
    qmv_b8_p1_m1: mtl.Pipeline,
    qmv_b8_p2_m1: mtl.Pipeline,
    qmv_b6_p1_m1: mtl.Pipeline,
    qmv_b6_p2_m1: mtl.Pipeline,
    gateup_b6_p2_m1: mtl.Pipeline,
    gateup_b8_p2_m1: mtl.Pipeline,
    qmv_sk_b8: mtl.Pipeline,
    qmv_sk_b6: mtl.Pipeline,
    gateup_sk_b8: mtl.Pipeline,
    gateup_sk_b6: mtl.Pipeline,
    gemv1_bf16: mtl.Pipeline,
    qmm_b6: mtl.Pipeline,
    qmm_b8: mtl.Pipeline,
    qmm_b16: mtl.Pipeline,
    qmm_gateup_b6: mtl.Pipeline,
    qmm_gateup_b8: mtl.Pipeline,
    qmm2_b6_m32: mtl.Pipeline,
    qmm2_b8_m32: mtl.Pipeline,
    qmm2_gateup_b6_m32: mtl.Pipeline,
    qmm2_gateup_b8_m32: mtl.Pipeline,
    gemv_bf16: mtl.Pipeline,
    gemv_rows_bf16: mtl.Pipeline,
    gemv1_rows_bf16: mtl.Pipeline,
    rms: mtl.Pipeline,
    sum_res_rms: mtl.Pipeline,
    wsum_res_rms: mtl.Pipeline,
    embed: mtl.Pipeline,
    rope: mtl.Pipeline,
    kv_store: mtl.Pipeline,
    idx_k_store: mtl.Pipeline,
    idx_scores2: mtl.Pipeline,
    idx_scores3: mtl.Pipeline,
    topk: mtl.Pipeline,
    tk_init: mtl.Pipeline,
    tk_hist: mtl.Pipeline,
    tk_digit: mtl.Pipeline,
    tk_count: mtl.Pipeline,
    tk_scan: mtl.Pipeline,
    tk_write: mtl.Pipeline,
    attn_block: mtl.Pipeline,
    attn_join: mtl.Pipeline,
    attn_block4h: mtl.Pipeline,
    attn_joinb: mtl.Pipeline,
    tk_wpairs: mtl.Pipeline,
    tk_allpairs: mtl.Pipeline,
    kconcat: mtl.Pipeline,
    kmerge: mtl.Pipeline,
    route: mtl.Pipeline,
    sort_picks: mtl.Pipeline,
    moe_combine: mtl.Pipeline,
    argmax: mtl.Pipeline,
    pick: mtl.Pipeline,
    post: mtl.Pipeline,
    wait: mtl.Pipeline,
    copy: mtl.Pipeline,
    copy_u32: mtl.Pipeline,
    copy_rows: mtl.Pipeline,
    argmax_rows: mtl.Pipeline,
    pick_rows: mtl.Pipeline,
    mtp_cat: mtl.Pipeline,
    touch: mtl.Pipeline,
    lw_in: mtl.Pipeline,
    lw_out: mtl.Pipeline,

    fn init(device: mtl.Device, kv16: bool) !Pipes {
        const src = if (kv16) "#define KVT half\n" ++ source else source;
        const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipes = undefined;
        inline for (@typeInfo(Pipes).@"struct".field_names) |name| @field(p, name) = try mtl.Pipeline.init(device, lib, "g53_" ++ name, false);
        return p;
    }
};

/// Where this process stands: its weights' share, the canonical slices it computes, and (N Macs) its exchange.
pub const Plan = struct {
    slices: []const m.Share, // the slices this process sums into slots (one Mac: all of them; N Macs: its own)
    slot_of_first: usize, // the slot index of slices[0]
    nslots: usize, // every slot the sum reads (= canonical slice count)
};

/// One quantized matvec launch: rows [n0, n0 + n) of the stored matrix, input columns [k0, k0 + K), over `items`
/// inputs (rows or picks) and `z` stacked matrices (heads).
const QL = struct {
    n0: usize = 0,
    n: usize,
    k0: usize = 0,
    K: usize,
    items: usize = 1,
    z: usize = 1,
    xr: usize = 0,
    zx: usize = 0,
    yr: usize = 0,
    zy: usize = 0,
    ids: ?Ref = null,
    xids: ?Ref = null,
    yids: ?Ref = null,
    seg: ?Ref = null, // batches are SEG's segments (picks grouped by expert)
    stacked: bool = false,
};

/// Blocks of at most this many rows take top-k over many threadgroups (decode is one row).
pub const tk_rows_max: usize = 8;
/// ... and at least this many keys: below it the old one-threadgroup top-k is faster (11 launches vs 1; measured 0.06 vs 0.02 ms
/// at 2K keys, 0.043 vs 0.08 ms at 64K, 0.051 vs 0.48 ms at 393K).
pub const tk_min_keys: usize = 32768;

pub const TkPipes = struct { init: mtl.Pipeline, hist: mtl.Pipeline, digit: mtl.Pipeline, count: mtl.Pipeline, scan: mtl.Pipeline, write: mtl.Pipeline };
/// KSPLIT: the few-launch select over a fixed range per row, its picks as (score, lo + i) pairs into `cand` rows of `rstride` words.
pub const TkPairs = struct { fixed: u32, lo: u32, rstride: u32, wpairs: mtl.Pipeline, allpairs: mtl.Pipeline, cand: Ref };
pub const TkBufs = struct { scores: Ref, idx: Ref, state: Ref, hist: Ref, cnt: Ref, base: Ref };

/// Keys a top-k chunk takes: at most 1,024 chunks a row, at least 1,024 keys a chunk, a multiple of 256.
pub fn tkChunk(n: usize) usize {
    const c = @max(1024, (n + 1023) / 1024);
    return (c + 255) / 256 * 256;
}

/// The top-k keys of `rows` (<= tk_rows_max) rows at positions p0.. over many threadgroups: the same picks as g53_topk.
pub fn encodeTopkFew(enc: mtl.ComputeEncoder, p: TkPipes, b: TkBufs, p0: u32, rows: usize, last_n: usize, cap: u32, conc: bool) void {
    encodeTopkFewX(enc, p, b, p0, rows, last_n, cap, conc, null);
}

pub fn encodeTopkFewX(enc: mtl.ComputeEncoder, p: TkPipes, b: TkBufs, p0: u32, rows: usize, last_n: usize, cap: u32, conc: bool, pairs: ?TkPairs) void {
    const chunk = tkChunk(last_n);
    const groups = (last_n + chunk - 1) / chunk;
    var ta: TkArgs = .{ .p0 = @intCast(p0), .top = m.KEYS, .cap = @intCast(cap), .chunk = @intCast(chunk), .shift = 0 };
    if (pairs) |pp| {
        ta.fixed = @intCast(pp.fixed);
        ta.lo = @intCast(pp.lo);
        ta.rstride = @intCast(pp.rstride);
    }
    const S = mtl.Size.of;
    enc.setPipeline(p.init);
    enc.setBuffer(b.state.buf, b.state.off, 0);
    enc.setBuffer(b.hist.buf, b.hist.off, 1);
    enc.setValue(ta, 2);
    enc.dispatchGroups(S(rows, 1, 1), S(256, 1, 1));
    if (conc) enc.barrier();
    var shift: i32 = 24;
    while (shift >= 0) : (shift -= 8) {
        ta.shift = shift;
        enc.setPipeline(p.hist);
        enc.setBuffer(b.scores.buf, b.scores.off, 0);
        enc.setBuffer(b.hist.buf, b.hist.off, 1);
        enc.setBuffer(b.state.buf, b.state.off, 2);
        enc.setValue(ta, 3);
        enc.dispatchGroups(S(groups, rows, 1), S(256, 1, 1));
        if (conc) enc.barrier();
        enc.setPipeline(p.digit);
        enc.setBuffer(b.state.buf, b.state.off, 0);
        enc.setBuffer(b.hist.buf, b.hist.off, 1);
        enc.setValue(ta, 2);
        enc.dispatchGroups(S(rows, 1, 1), S(256, 1, 1));
        if (conc) enc.barrier();
    }
    enc.setPipeline(p.count);
    enc.setBuffer(b.scores.buf, b.scores.off, 0);
    enc.setBuffer(b.state.buf, b.state.off, 1);
    enc.setBuffer(b.cnt.buf, b.cnt.off, 2);
    enc.setValue(ta, 3);
    enc.dispatchGroups(S(groups, rows, 1), S(256, 1, 1));
    if (conc) enc.barrier();
    enc.setPipeline(p.scan);
    enc.setBuffer(b.state.buf, b.state.off, 0);
    enc.setBuffer(b.cnt.buf, b.cnt.off, 1);
    enc.setBuffer(b.base.buf, b.base.off, 2);
    enc.setValue(ta, 3);
    enc.dispatchGroups(S(rows, 1, 1), S(1024, 1, 1));
    if (conc) enc.barrier();
    enc.setPipeline(if (pairs) |pp| pp.wpairs else p.write);
    enc.setBuffer(b.scores.buf, b.scores.off, 0);
    enc.setBuffer(b.state.buf, b.state.off, 1);
    enc.setBuffer(b.base.buf, b.base.off, 2);
    if (pairs) |pp| enc.setBuffer(pp.cand.buf, pp.cand.off, 3) else enc.setBuffer(b.idx.buf, b.idx.off, 3);
    enc.setValue(ta, 4);
    enc.dispatchGroups(S(groups, rows, 1), S(256, 1, 1));
    if (pairs) |pp| { // rows whose range has <= top keys: every key a candidate (the launches above skip those rows)
        enc.setPipeline(pp.allpairs);
        enc.setBuffer(b.scores.buf, b.scores.off, 0);
        enc.setBuffer(pp.cand.buf, pp.cand.off, 1);
        enc.setValue(ta, 2);
        enc.dispatchGroups(S(rows, 1, 1), S(256, 1, 1));
    }
}

/// The most rows a verify block takes (the pending token and up to verify_max - 1 drafts).
pub const verify_max: usize = 8;
/// Own-row DSA front: blocks of at least this many rows (every rank then takes >= 16, the block's own kernels).
pub const own_min_rows: usize = 64;

pub const Engine = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    p: Pipes,
    w: *const m.Weights,
    plan: Plan,
    xc: ?*Exchange,
    cap: u32,
    max_rows: u32,
    // activations, max_rows rows each
    h: Ref,
    xn: Ref,
    qa: Ref,
    qr: Ref,
    kv: Ref,
    q: Ref,
    ql: Ref,
    att: Ref,
    vals: Ref,
    iq: Ref,
    ik: Ref,
    iw: Ref,
    scores: Ref,
    idx: Ref,
    po: Ref,
    pm: Ref,
    pl: Ref,
    logits_r: Ref,
    ids: Ref,
    wts: Ref,
    se: Ref, // picks grouped by expert: expert, row, pick slot
    sr: Ref,
    ss: Ref,
    seg: Ref, // segments of <= 8 same-expert picks
    act: Ref, // [rows * top][ir] routed, then [rows][ir] shared
    y: Ref, // [rows * top][D] routed
    ys: Ref, // [rows][D] shared
    slots: Ref, // one Mac: [nslots][max_rows][D] fp32 partials
    logits: Ref,
    args: Ref, // one Mac: the argmax pair
    tok: Ref, // u32 [max_rows]: the block's tokens; the pick lands in tok[0]
    dump: ?Ref = null,
    idx_dump: ?Ref = null,
    kvc: [m.LAYERS + 1]?Ref = @splat(null), // [m.MTP]: the draft head's caches
    ic: [m.LAYERS + 1]?Ref = @splat(null),
    // MTP: verify blocks (rows through the row path, a pick per row) and the draft head
    verify: bool = false, // this block is a verify block: row path (no tiles), the head over every row into `picks`
    last_rows: usize = 0, // rows in the last prefill block (its last row = the prompt's last token)
    vlogits: Ref = undefined, // [verify_max][vocab share]
    vargs: Ref = undefined, // one Mac: [verify_max] (value, index) pairs
    picks: Ref = undefined, // u32 [verify_max]: the trunk's pick at each verify row
    dtok: Ref = undefined, // u32 [verify_max]: drafts
    mtok: Ref = undefined, // u32 [max_rows]: the head's next tokens
    m_emb: Ref = undefined, // [max_rows][D]
    m_hin: Ref = undefined, // [1][D]: a chain step's h (the head's previous raw output)
    m_cat: Ref = undefined, // [max_rows][2D]
    bufs: std.ArrayList(mtl.Buffer) = .empty,
    seq: u32 = 0, // exchanges encoded so far
    only: ?Op = null, // profile: encode only this class
    kv_bytes: usize = 4, // KV cache element bytes (2: fp16 latent and indexer keys)
    grows: bool = false, // G53_GROWS=1: bf16 gemv rows (verify head, eh_proj, router, indexer) share each weight read (same bits)
    tiles: bool = false, // this block's matmuls on simdgroup-matrix tiles (prompt blocks)
    tk_state: Ref = undefined,
    tk_hist: Ref = undefined,
    tk_cnt: Ref = undefined,
    tk_base: Ref = undefined,
    idx_v3: bool = false, // G53_IDX=3: indexer scores two rows a threadgroup (same bits)
    fwait: bool = false, // G53_FWAIT=1: the exchange's wait inside the sum launch (one launch fewer an exchange, same bits)
    own_front: bool = false, // G53_OWN_FRONT=1: prompt blocks (>= 64 rows) run the indexer's query side, scores and top-k
    // on this rank's quarter of the rows; the picks are all-gathered (one Mac with N slices: N virtual ranks, same bits)
    ksplit: bool = false, // G53_KSPLIT=1: decode/verify rows at >= ksplit_min keys score a quarter of the keys a rank, candidates
    // swapped and merged (g53_topk's picks exactly); one Mac: N virtual ranks in turn
    ksplit_min: u32 = 65536,
    kcs: Ref = undefined, // [verify_max][4 * KEYS] candidate scores
    kci: Ref = undefined, // [verify_max][4 * KEYS] candidate indices
    kmn: Ref = undefined, // [verify_max] candidate counts
    attn4h: bool = false, // G53_ATTN4H=1: attention 4 heads a threadgroup (13.4 KB of threadgroup memory, two a core; same bits)
    joinb: bool = false, // G53_JOINB=1: blocks of <= 16 rows: the attention join issues its loads 8 blocks at a time (same adds, same order)
    // G53_LSPLIT=1 (N Macs): layer i's latent cache and attention (all heads) on rank i % ranks only; the
    // owner's every-head values reach the others in one exchange, each rank's o_proj reads its heads' columns from them
    ls: bool = false,
    // memory trims (exact: the same kernels on the same rows, fewer rows a launch): prompt blocks run the
    // replicated indexer's scores + top-k (score_rows) and the attention (attn_rows) in row chunks, so the cap-sized score
    // buffer and the attention partials hold that many rows, not max_rows. G53_SCORE_ROWS / G53_ATTN_ROWS (0 = max_rows);
    // default 64 each under G53_LSPLIT (-3.1 GB a rank at 2M), else max_rows (unchanged).
    score_rows: usize = 0,
    attn_rows: usize = 0,
    ls_rank: usize = 0,
    // G53_LSPLIT_FROM=X (hybrid): positions < X stay on EVERY rank (every layer), so a block whose keys all lie
    // below X runs the plain engine exactly (no extra exchange, today's speed); a block reaching past X runs the layer
    // split. Top-k picks ascend by position, the owner holds every position of its layers: exact in both regimes.
    ls_from: u32 = 0,
    vals_own: Ref = undefined, // G53_LSPLIT non-owner: this rank's heads' values [rows][16][256] from the layer's owner
    ls_ranks: usize = 1,
    conc: bool = false, // G53_CONC=1: concurrent encoder, barriers only at data dependencies (same kernels, same bits)
    lw: ?*Lw = null, // Living Weights: the committed change at layer 77's shared expert (null: the stock forward)
    // instrumentation: per-round CPU encode, GPU, and wall (commit->wait) seconds, summed per job
    st_enc: f64 = 0,
    st_gpu: f64 = 0,
    st_wall: f64 = 0,
    st_rounds: u64 = 0,
    pf_enc: f64 = 0,
    pf_gpu: f64 = 0,
    pf_wall: f64 = 0,
    pf_blocks: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, w: *const m.Weights, plan: Plan, xc: ?*Exchange, cap: u32, max_rows: u32, kv16: bool) !*Engine {
        const e = try gpa.create(Engine);
        const queue = try device.queue();
        if (w.resident) |set| queue.addResidencySet(set);
        e.* = .{ .idx_v3 = if (std.c.getenv("G53_IDX")) |v| v[0] == '3' else false, .conc = if (std.c.getenv("G53_CONC")) |v| v[0] == '1' else false, .own_front = if (std.c.getenv("G53_OWN_FRONT")) |v| v[0] == '1' else false, .ksplit = if (std.c.getenv("G53_KSPLIT")) |v| v[0] == '1' else false, .ksplit_min = if (std.c.getenv("G53_KSPLIT_MIN")) |v| (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 65536) else 65536, .attn4h = if (std.c.getenv("G53_ATTN4H")) |v| v[0] == '1' else false, .joinb = if (std.c.getenv("G53_JOINB")) |v| v[0] == '1' else false, .fwait = if (std.c.getenv("G53_FWAIT")) |v| v[0] == '1' else false, .grows = if (std.c.getenv("G53_GROWS")) |v| v[0] == '1' else false, .gpa = gpa, .device = device, .queue = queue, .p = try Pipes.init(device, kv16), .kv_bytes = if (kv16) 2 else 4, .w = w, .plan = plan, .xc = xc, .cap = cap, .max_rows = max_rows, .h = undefined, .xn = undefined, .qa = undefined, .qr = undefined, .kv = undefined, .q = undefined, .ql = undefined, .att = undefined, .vals = undefined, .iq = undefined, .ik = undefined, .iw = undefined, .scores = undefined, .idx = undefined, .po = undefined, .pm = undefined, .pl = undefined, .logits_r = undefined, .ids = undefined, .wts = undefined, .se = undefined, .sr = undefined, .ss = undefined, .seg = undefined, .act = undefined, .y = undefined, .ys = undefined, .slots = undefined, .logits = undefined, .args = undefined, .tok = undefined };
        if (xc) |x| x.conc = e.conc;
        const R: usize = max_rows;
        const nh0 = w.share.heads[1] - w.share.heads[0];
        if (m.lsplitOn() and xc != null and nh0 < m.H) {
            e.ls = true;
            e.ls_ranks = m.H / nh0;
            e.ls_rank = w.share.heads[0] / nh0;
            if (std.c.getenv("G53_LSPLIT_FROM")) |v| e.ls_from = @min(cap, std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0);
            std.debug.print("glm53: LSPLIT rank {d}/{d}: owns {d} of {d} layers' latent cache + attention; positions < {d} on every rank\n", .{ e.ls_rank, e.ls_ranks, blk: {
                var c: usize = 0;
                for (0..m.LAYERS + 1) |li| c += @intFromBool(m.lsOwnerRank(li, e.ls_ranks) == e.ls_rank);
                break :blk c;
            }, m.LAYERS + 1, e.ls_from });
        }
        const nh = if (e.ls) m.H else nh0; // the attention buffers: every head on a layer-split owner
        const ir = @max(w.share.moe[1] - w.share.moe[0], w.share.dense[1] - w.share.dense[0]);
        const vr = w.share.vocab[1] - w.share.vocab[0];
        e.h = try e.buf(R * m.D * 4);
        e.xn = try e.buf(R * m.D * 4);
        e.qa = try e.buf(R * m.QL * 4);
        e.qr = try e.buf(R * m.QL * 4);
        e.kv = try e.buf(R * m.CROW * 4);
        e.q = try e.buf(R * nh * m.QH * 4);
        e.ql = try e.buf(R * nh * m.KVL * 4);
        e.att = try e.buf(R * nh * m.KVL * 4);
        e.vals = try e.buf(R * nh * m.VD * 4);
        if (e.ls) e.vals_own = try e.buf(R * nh0 * m.VD * 4);
        e.iq = try e.buf(R * m.IH * m.ID * 4);
        e.ik = try e.buf(R * m.ID * 4);
        e.iw = try e.buf(R * m.IH * 4);
        const envRows = struct {
            fn f(n: [*:0]const u8, dflt: usize, most: usize) usize {
                const v = if (std.c.getenv(n)) |z| (std.fmt.parseInt(usize, std.mem.span(z), 10) catch 0) else dflt;
                return if (v == 0 or v > most) most else @max(v, tk_rows_max);
            }
        }.f;
        e.score_rows = envRows("G53_SCORE_ROWS", if (e.ls) 64 else 0, R);
        e.attn_rows = envRows("G53_ATTN_ROWS", if (e.ls) 64 else 0, R);
        if (e.score_rows < R or e.attn_rows < R) std.debug.print("glm53: row chunks: indexer scores {d} rows, attention {d} rows a launch\n", .{ e.score_rows, e.attn_rows });
        e.scores = try e.buf(e.score_rows * @as(usize, cap) * 4);
        e.idx = try e.buf(R * m.KEYS * 4);
        e.tk_state = try e.buf(tk_rows_max * 4 * 4);
        e.tk_hist = try e.buf(tk_rows_max * 256 * 4);
        e.tk_cnt = try e.buf(tk_rows_max * 1024 * 2 * 4);
        e.tk_base = try e.buf(tk_rows_max * 1024 * 2 * 4);
        e.po = try e.buf(e.attn_rows * 64 * nh * m.KVL * 4 + 4096);
        e.pm = try e.buf(e.attn_rows * 64 * nh * 4 + 256);
        e.pl = try e.buf(e.attn_rows * 64 * nh * 4 + 256);
        e.logits_r = try e.buf(R * m.E * 4);
        e.ids = try e.buf(R * m.TOP * 4);
        e.wts = try e.buf(R * m.TOP * 4);
        e.se = try e.buf(R * m.TOP * 4);
        e.sr = try e.buf(R * m.TOP * 4);
        e.ss = try e.buf(R * m.TOP * 4);
        e.seg = try e.buf((1 + 2 * (R * m.TOP + m.E)) * 4);
        e.act = try e.buf(R * (m.TOP + 1) * ir * 4);
        e.y = try e.buf(R * m.TOP * m.D * 4);
        e.ys = try e.buf(R * m.D * 4);
        e.slots = try e.buf(plan.nslots * R * m.D * 4);
        e.logits = try e.buf(vr * 4);
        e.args = try e.buf(64);
        e.tok = try e.buf(R * 4 + 64);
        for (w.first..w.last) |i| {
            if (e.kvRows(i) > 0) e.kvc[i] = try e.buf(@as(usize, e.kvRows(i)) * m.CROW * e.kv_bytes);
            if (m.fullIndexer(i)) e.ic[i] = try e.buf(@as(usize, cap) * m.ID * e.kv_bytes);
        }
        e.kcs = try e.buf(verify_max * 4 * m.KEYS * 4);
        e.kci = try e.buf(verify_max * 4 * m.KEYS * 4);
        e.kmn = try e.buf(verify_max * 4 + 64);
        e.vlogits = try e.buf(verify_max * vr * 4);
        e.vargs = try e.buf(verify_max * 16 + 64);
        e.picks = try e.buf(verify_max * 4 + 64);
        e.dtok = try e.buf(verify_max * 4 + 64);
        if (w.mtp != null) {
            if (e.kvRows(m.MTP) > 0) e.kvc[m.MTP] = try e.buf(@as(usize, e.kvRows(m.MTP)) * m.CROW * e.kv_bytes);
            e.ic[m.MTP] = try e.buf(@as(usize, cap) * m.ID * e.kv_bytes);
            e.mtok = try e.buf(R * 4 + 64);
            e.m_emb = try e.buf(R * m.D * 4);
            e.m_hin = try e.buf(m.D * 4);
            e.m_cat = try e.buf(R * 2 * m.D * 4);
        }
        if (std.c.getenv("G53_KV_RESIDENT")) |v| if (v[0] == '1') if (w.resident) |set| { // KV + activations wired for the queue too
            for (e.bufs.items) |b| set.add(b);
            set.commit();
            set.requestResidency();
            std.debug.print("glm53: {d} engine buffers added to the residency set\n", .{e.bufs.items.len});
        };
        return e;
    }

    fn buf(e: *Engine, len: usize) !Ref {
        const b = try m.hostBuffer(e.device, len);
        try e.bufs.append(e.gpa, b);
        @memset(b.contents()[0..len], 0);
        return .{ .buf = b };
    }

    /// The encoder kind for step command buffers (profiles stay serial).
    pub fn dtype(e: *const Engine) mtl.DispatchType {
        return if (e.conc) .concurrent else .serial;
    }

    /// Concurrent encoder: later launches see every write before this point. Serial: a no-op (every launch already waits).
    pub fn bar(e: *const Engine, enc: mtl.ComputeEncoder) void {
        if (e.conc) enc.barrier();
    }

    pub fn addBuf(e: *Engine, len: usize) !Ref {
        return e.buf(len);
    }

    /// A dump area for `floats` values (hidden states captured during steps).
    pub fn addDump(e: *Engine, floats: usize) !void {
        e.dump = try e.buf(floats * 4);
    }

    // ------------------------------------------------------------------------------------------------ dispatch
    fn bind(enc: mtl.ComputeEncoder, first: usize, refs: anytype) void {
        inline for (refs, 0..) |r, i| enc.setBuffer(r.buf, r.off, first + i);
    }

    fn sz(a: usize, b: usize, c: usize) mtl.Size {
        return mtl.Size.of(a, b, c);
    }

    fn qargs(q: m.Q, l: QL) QArgs {
        return .{ .K = @intCast(l.K), .N = @intCast(l.n), .ldw = @intCast(q.ldw()), .lds = @intCast(q.lds()), .zw = if (l.stacked) @intCast(q.n * q.ldw()) else 0, .zs = if (l.stacked) @intCast(q.n * q.lds()) else 0, .xr = @intCast(l.xr), .zx = @intCast(l.zx), .yr = @intCast(l.yr), .zy = @intCast(l.zy), .ids = @intFromBool(l.ids != null), .xids = @intFromBool(l.xids != null), .yids = @intFromBool(l.yids != null), .items = @intCast(l.items), .segs = @intFromBool(l.seg != null) };
    }

    fn qpipe(e: *Engine, bits: u8, p2: bool) mtl.Pipeline {
        const p = &e.p;
        return if (bits == 8) (if (p2) p.qmv_b8_p2_m1 else p.qmv_b8_p1_m1) else (if (p2) p.qmv_b6_p2_m1 else p.qmv_b6_p1_m1);
    }

    fn gpipe(e: *Engine, bits: u8) mtl.Pipeline {
        const p = &e.p;
        return if (bits == 8) p.gateup_b8_p2_m1 else p.gateup_b6_p2_m1;
    }

    /// Short outputs over a long K, one input a threadgroup row: split K over 8 simdgroups (more threadgroups in flight).
    fn splitK(l: QL) bool {
        return l.ids == null and l.seg == null and l.z == 1 and l.n <= 2048 and l.K % 2048 == 0 and l.K >= 4096;
    }

    /// Tile launches: segments of <= 32 picks (an upper bound) by 64 outputs (qmm2), or 32 inputs by 32 outputs (qmm).
    fn tileGrid(l: QL, n: usize) mtl.Size {
        if (l.seg != null) return sz((l.items + 31) / 32 + @min(l.items, m.E), n / 64, l.z);
        return sz((l.items + 31) / 32, n / 32, l.z);
    }

    fn qmv(e: *Engine, enc: mtl.ComputeEncoder, q: m.Q, x: Ref, y: Ref, l: QL) void {
        if (e.tiles) {
            const segs = l.seg != null;
            enc.setPipeline(if (q.bits == 8) (if (segs) e.p.qmm2_b8_m32 else e.p.qmm_b8) else (if (segs) e.p.qmm2_b6_m32 else e.p.qmm_b6));
            const ldw: usize = q.ldw();
            const lds: usize = q.lds();
            enc.setBuffer(q.w.buf, q.w.off + l.n0 * ldw + l.k0 * q.bits / 8, 0);
            enc.setBuffer(q.s.buf, q.s.off + (l.n0 * lds + l.k0 / 64) * 2, 1);
            enc.setBuffer(q.b.buf, q.b.off + (l.n0 * lds + l.k0 / 64) * 2, 2);
            bind(enc, 3, .{ x, y });
            enc.setValue(qargs(q, l), 5);
            bind(enc, 6, .{ l.ids orelse x, l.xids orelse x, l.yids orelse x, l.seg orelse x });
            enc.dispatchGroups(tileGrid(l, l.n), sz(128, 1, 1));
            return;
        }
        if (splitK(l)) {
            enc.setPipeline(if (q.bits == 8) e.p.qmv_sk_b8 else e.p.qmv_sk_b6);
            const ldw: usize = q.ldw();
            const lds: usize = q.lds();
            enc.setBuffer(q.w.buf, q.w.off + l.n0 * ldw + l.k0 * q.bits / 8, 0);
            enc.setBuffer(q.s.buf, q.s.off + (l.n0 * lds + l.k0 / 64) * 2, 1);
            enc.setBuffer(q.b.buf, q.b.off + (l.n0 * lds + l.k0 / 64) * 2, 2);
            bind(enc, 3, .{ x, y });
            enc.setValue(qargs(q, l), 5);
            bind(enc, 6, .{ l.ids orelse x, l.xids orelse x, l.yids orelse x, l.seg orelse x });
            enc.dispatchGroups(sz(l.items, l.n / 4, 1), sz(256, 1, 1));
            return;
        }
        const p2 = l.K % 256 == 0;
        enc.setPipeline(e.qpipe(q.bits, p2));
        const ldw: usize = q.ldw();
        const lds: usize = q.lds();
        enc.setBuffer(q.w.buf, q.w.off + l.n0 * ldw + l.k0 * q.bits / 8, 0);
        enc.setBuffer(q.s.buf, q.s.off + (l.n0 * lds + l.k0 / 64) * 2, 1);
        enc.setBuffer(q.b.buf, q.b.off + (l.n0 * lds + l.k0 / 64) * 2, 2);
        bind(enc, 3, .{ x, y });
        enc.setValue(qargs(q, l), 5);
        bind(enc, 6, .{ l.ids orelse x, l.xids orelse x, l.yids orelse x, l.seg orelse x });
        enc.dispatchGroups(sz(l.items, l.n / 8, l.z), sz(32, 2, 1)); // an item (row or pick) a threadgroup
    }

    /// silu(gate x) * (up x) for every item, the matrices' whole rows.
    fn gateup(e: *Engine, enc: mtl.ComputeEncoder, g: m.Q, u: m.Q, x: Ref, y: Ref, l: QL) void {
        if (e.tiles) {
            const segs = l.seg != null;
            enc.setPipeline(if (g.bits == 8) (if (segs) e.p.qmm2_gateup_b8_m32 else e.p.qmm_gateup_b8) else (if (segs) e.p.qmm2_gateup_b6_m32 else e.p.qmm_gateup_b6));
            bind(enc, 0, .{ g.w, g.s, g.b, u.w, u.s, u.b, x, y });
            var a = qargs(g, l);
            a.zw = @intCast(g.n * g.ldw());
            a.zs = @intCast(g.n * g.lds());
            enc.setValue(a, 8);
            bind(enc, 9, .{ l.ids orelse x, l.xids orelse x, l.yids orelse x, l.seg orelse x });
            enc.dispatchGroups(tileGrid(l, g.n), sz(128, 1, 1));
            return;
        }
        if (splitK(l)) {
            enc.setPipeline(if (g.bits == 8) e.p.gateup_sk_b8 else e.p.gateup_sk_b6);
            bind(enc, 0, .{ g.w, g.s, g.b, u.w, u.s, u.b, x, y });
            enc.setValue(qargs(g, l), 8);
            bind(enc, 9, .{ l.ids orelse x, l.xids orelse x, l.yids orelse x, l.seg orelse x });
            enc.dispatchGroups(sz(l.items, g.n / 4, 1), sz(256, 1, 1));
            return;
        }
        enc.setPipeline(e.gpipe(g.bits));
        bind(enc, 0, .{ g.w, g.s, g.b, u.w, u.s, u.b, x, y });
        var a = qargs(g, l);
        a.zw = @intCast(g.n * g.ldw());
        a.zs = @intCast(g.n * g.lds());
        enc.setValue(a, 8);
        bind(enc, 9, .{ l.ids orelse x, l.xids orelse x, l.yids orelse x, l.seg orelse x });
        enc.dispatchGroups(sz(l.items, g.n / 8, 1), sz(32, 2, 1));
    }

    fn gemv(e: *Engine, enc: mtl.ComputeEncoder, wt: Ref, x: Ref, y: Ref, n: usize, k: usize, rows: usize) void {
        if (e.tiles and rows > 1 and n >= 256 and n % 32 == 0) {
            enc.setPipeline(e.p.qmm_b16);
            bind(enc, 0, .{ wt, wt, wt, x, y });
            const l: QL = .{ .n = n, .K = k, .items = rows, .xr = k, .yr = n };
            enc.setValue(QArgs{ .K = @intCast(k), .N = @intCast(n), .ldw = @intCast(k * 2), .lds = 0, .zw = 0, .zs = 0, .xr = @intCast(k), .zx = 0, .yr = @intCast(n), .zy = 0, .ids = 0, .xids = 0, .yids = 0, .items = @intCast(rows), .segs = 0 }, 5);
            bind(enc, 6, .{ x, x, x, x });
            enc.dispatchGroups(tileGrid(l, n), sz(128, 1, 1));
            return;
        }
        const one = n <= 1024; // short outputs: a row a threadgroup
        if (e.grows and rows > 1) {
            enc.setPipeline(if (one) e.p.gemv1_rows_bf16 else e.p.gemv_rows_bf16);
            bind(enc, 0, .{ wt, x, y });
            enc.setValue(GArgs{ .K = @intCast(k), .N = @intCast(n), .ldw = @intCast(k), .xr = @intCast(k), .yr = @intCast(n) }, 3);
            enc.setValue(@as(i32, @intCast(rows)), 4);
            enc.dispatchGroups(sz(if (one) n else (n + 3) / 4, (rows + 7) / 8, 1), sz(256, 1, 1));
            return;
        }
        enc.setPipeline(if (one) e.p.gemv1_bf16 else e.p.gemv_bf16);
        bind(enc, 0, .{ wt, x, y });
        enc.setValue(GArgs{ .K = @intCast(k), .N = @intCast(n), .ldw = @intCast(k), .xr = @intCast(k), .yr = @intCast(n) }, 3);
        enc.dispatchGroups(sz(if (one) n else (n + 3) / 4, rows, 1), sz(256, 1, 1));
    }

    fn rms(e: *Engine, enc: mtl.ComputeEncoder, x: Ref, wt: Ref, out: Ref, d: usize, eps: f32, round: bool, rows: usize) void {
        enc.setPipeline(e.p.rms);
        bind(enc, 0, .{ x, wt, out });
        enc.setValue(NArgs{ .D = @intCast(d), .eps = eps, .round_bf16 = @intFromBool(round), .ld = @intCast(d) }, 3);
        enc.dispatchGroups(sz(rows, 1, 1), sz(1024, 1, 1));
    }

    /// Rope on `rows` rows `stride` apart (`hpr` a token), the block's first token at pos.
    fn rope(e: *Engine, enc: mtl.ComputeEncoder, x: Ref, rows: usize, stride: usize, off: usize, pos: u32, hpr: usize) void {
        enc.setPipeline(e.p.rope);
        bind(enc, 0, .{x});
        enc.setValue(RArgs{ .rows = @intCast(rows), .stride = @intCast(stride), .off = @intCast(off), .dims = m.ROPE, .pos = @intCast(pos), .log2base = m.LOG2_THETA, .hpr = @intCast(hpr) }, 1);
        enc.dispatchThreads(sz(m.ROPE / 2, rows, 1), sz(m.ROPE / 2, 1, 1));
    }

    fn copy(e: *Engine, enc: mtl.ComputeEncoder, src: Ref, dst: Ref, n: usize) void {
        enc.setPipeline(e.p.copy);
        bind(enc, 0, .{ src, dst });
        enc.setValue(@as(i32, @intCast(n)), 2);
        enc.dispatchThreads(sz(n, 1, 1), sz(256, 1, 1));
    }

    /// The partial slots' base (the window on N Macs) and the floats between slots.
    fn slotBase(e: *Engine) struct { base: Ref, stride: usize } {
        if (e.xc) |x| return .{ .base = x.slot(e.seq), .stride = x.slotFloats() };
        return .{ .base = e.slots, .stride = @as(usize, e.max_rows) * m.D };
    }

    /// Slot `i` (index into the plan's slices) for this exchange.
    fn slotFor(e: *Engine, i: usize) Ref {
        const s = e.slotBase();
        return s.base.at((e.plan.slot_of_first + i) * s.stride * 4);
    }

    /// Partials are written: swap them (N Macs), then h += their sum and xn = rms(h) * norm (or only the add).
    fn reduce(e: *Engine, enc: mtl.ComputeEncoder, norm: ?Ref, rows: usize) void {
        const s = e.slotBase();
        if (e.xc) |x| if (e.fwait) {
            x.encodeCopies(enc, e.p.copy, e.seq, rows * m.D * 4);
            x.encodePost(enc, e.p.post, e.seq, rows * m.D * 4);
            enc.setPipeline(e.p.wsum_res_rms);
            bind(enc, 0, .{ s.base, e.h, norm orelse e.w.norm, e.xn });
            enc.setValue(SArgs{ .D = m.D, .S = @intCast(e.plan.nslots), .stride = @intCast(s.stride), .eps = 1e-5, .norm = @intFromBool(norm != null) }, 4);
            x.waitBind(enc, e.seq, 5);
            enc.dispatchGroups(sz(rows, 1, 1), sz(1024, 1, 1));
            e.bar(enc);
            e.seq += 1;
            return;
        };
        if (e.xc) |x| {
            x.encodeCopies(enc, e.p.copy, e.seq, rows * m.D * 4);
            x.encode(enc, e.p.post, e.p.wait, e.seq, rows * m.D * 4); // concurrent: barriers before the post, after the wait
        } else e.bar(enc);
        enc.setPipeline(e.p.sum_res_rms);
        bind(enc, 0, .{ s.base, e.h, norm orelse e.w.norm, e.xn });
        enc.setValue(SArgs{ .D = m.D, .S = @intCast(e.plan.nslots), .stride = @intCast(s.stride), .eps = 1e-5, .norm = @intFromBool(norm != null) }, 4);
        enc.dispatchGroups(sz(rows, 1, 1), sz(1024, 1, 1));
        e.bar(enc);
        e.seq += 1;
    }

    /// G53_LSPLIT: does this rank hold layer i's latent cache (and compute its attention)? Off: every rank, every layer.
    pub fn owns(e: *const Engine, i: usize) bool {
        return !e.ls or m.lsOwnerRank(i, e.ls_ranks) == e.ls_rank;
    }

    /// Latent rows this rank holds for layer i: every position (owner, or no split), positions < ls_from (others).
    pub fn kvRows(e: *const Engine, i: usize) u32 {
        return if (e.owns(i)) e.cap else e.ls_from;
    }

    /// Rows a layer-split values exchange carries: 128 rows x 16 heads x 256 floats = 2 MiB a peer (well under the 8 MiB
    /// receive ring that once dropped a 6.3 MB partial), and never more than a slot holds.
    pub const ls_chunk_rows: usize = 128;

    fn copyRows(e: *Engine, enc: mtl.ComputeEncoder, src: Ref, dst: Ref, len: usize, sld: usize, dld: usize, rows: usize) void {
        enc.setPipeline(e.p.copy_rows);
        bind(enc, 0, .{ src, dst });
        enc.setValue([4]i32{ @intCast(len), @intCast(sld), @intCast(dld), @intCast(rows) }, 2);
        enc.dispatchThreads(sz(len, rows, 1), sz(256, 1, 1));
    }

    /// G53_LSPLIT: layer i's owner has every head's values in e.vals [rows][64][256]. Each peer gets ONLY its own heads'
    /// values (16 KB a row, from a send region of its own; 1/4 of an every-head broadcast) into e.vals_own [rows][16][256],
    /// in chunks of ls_chunk_rows rows; the others post 64 bytes of nothing for the same exchange numbers. The values
    /// are the owner's bits; the copies change none.
    fn lsSwap(e: *Engine, enc: mtl.ComputeEncoder, i: usize, rows: usize) void {
        const x = e.xc.?;
        const owner = m.lsOwnerRank(i, e.ls_ranks);
        const own = owner == e.ls_rank;
        const nhs = m.H / e.ls_ranks;
        const per: usize = nhs * m.VD; // floats a row a peer
        const all: usize = m.H * m.VD;
        const fit = @max(1, (x.slotFloats() - 64) / per);
        const chunk = @min(ls_chunk_rows, fit);
        var r0: usize = 0;
        while (r0 < rows) : (r0 += chunk) {
            const n = @min(chunk, rows - r0);
            const s = e.slotBase();
            if (own) {
                for (0..x.ranks) |p| if (p != x.rank) e.copyRows(enc, e.vals.at((r0 * all + p * per) * 4), x.sendRef(e.seq, p), per, all, per, n);
                e.bar(enc);
                x.encodePeers(enc, e.p.post, e.p.wait, e.seq, n * per * 4);
            } else x.encode(enc, e.p.post, e.p.wait, e.seq, 64);
            e.bar(enc);
            if (!own) {
                e.copy(enc, s.base.at(owner * s.stride * 4), e.vals_own.at(r0 * per * 4), n * per);
                e.bar(enc);
            }
            e.seq += 1;
        }
    }

    fn on(e: *const Engine, op: Op) bool {
        return e.only == null or e.only.? == op;
    }

    /// Keys a g53_idx_scores2 threadgroup scores: about 2,048 threadgroups in flight or more, 64..1,024 keys each.
    pub fn idxSpan(n: usize, rows: usize) usize {
        const want = (n * rows + 2047) / 2048;
        const s = std.math.clamp(want, 64, 1024);
        return (s + 63) / 64 * 64;
    }

    /// The indexer's query side, scores and top-k for rows [r0, r0 + n) of a block at p0 (their picks into `dst`, u32
    /// [n][KEYS]): each kernel is the whole block's on the same row (per-row arithmetic does not depend on the rows beside it).
    fn idxRows(e: *Engine, enc: mtl.ComputeEncoder, i: usize, p0: u32, r0: usize, n: usize, last_n: u32, dst: Ref) void {
        const a = &e.w.layers[i].attn;
        const q0: u32 = p0 + @as(u32, @intCast(r0));
        if (e.on(.idx_q)) {
            e.gemv(enc, a.wq_b.?, e.qr.at(r0 * m.QL * 4), e.iq, m.IH * m.ID, m.QL, n);
            e.gemv(enc, a.wproj.?, e.xn.at(r0 * m.D * 4), e.iw, m.IH, m.D, n);
            e.bar(enc);
            e.rope(enc, e.iq, n * m.IH, m.ID, 0, q0, m.IH);
            e.bar(enc);
        }
        const own_last = q0 + @as(u32, @intCast(n)); // the last own row's key count
        _ = last_n;
        if (e.on(.idx_scores)) {
            const wscale = 1.0 / (@sqrt(@as(f32, m.IH)) * @sqrt(@as(f32, m.ID)));
            const span = idxSpan(own_last, n);
            const two = e.idx_v3 and n > 1;
            enc.setPipeline(if (two) e.p.idx_scores3 else e.p.idx_scores2);
            bind(enc, 0, .{ e.iq, e.iw, e.ic[i].?, e.scores });
            enc.setValue(IArgs2{ .p0 = @intCast(q0), .wscale = wscale, .cap = @intCast(e.cap), .keys = m.KEYS, .span = @intCast(span), .rows = @intCast(n) }, 4);
            enc.dispatchGroups(sz(if (two) (n + 1) / 2 else n, (own_last + span - 1) / span, 1), sz(256, 1, 1));
            e.bar(enc);
        }
        if (e.on(.idx_topk)) {
            enc.setPipeline(e.p.topk); // n >= own_min_rows / ranks > tk_rows_max: the one-threadgroup-a-row top-k, as the whole block
            bind(enc, 0, .{ e.scores, dst });
            enc.setValue(TArgs{ .p0 = @intCast(q0), .top = m.KEYS, .cap = @intCast(e.cap) }, 2);
            enc.dispatchGroups(sz(n, 1, 1), sz(1024, 1, 1));
            e.bar(enc);
        }
    }

    /// Own-row DSA front (G53_OWN_FRONT): this rank's rows' picks into its exchange slot, all-gathered, unpacked into
    /// e.idx; one Mac runs every virtual rank's rows in turn straight into e.idx. Idea: MiaAI-Lab's prompt sequence-parallel
    /// select and drowzeys' TF_GLM53_SP_SELECT; ours shares only the picks (every rank already holds
    /// every row's xn and qr after the slot-ordered sum).
    fn ownFront(e: *Engine, enc: mtl.ComputeEncoder, i: usize, p0: u32, R: usize, last_n: u32, vranks: usize) void {
        if (e.xc) |x| {
            const own = m.part(R, vranks, x.rank);
            const s = e.slotBase();
            // in chunks of score_rows (the score buffer holds that many rows; the same per-row kernels)
            var r0: usize = own[0];
            while (r0 < own[1]) : (r0 += e.score_rows) {
                const n = @min(e.score_rows, own[1] - r0);
                e.idxRows(enc, i, p0, r0, n, last_n, s.base.at((@as(usize, x.rank) * s.stride + (r0 - own[0]) * m.KEYS) * 4));
            }
            x.encodeCopies(enc, e.p.copy, e.seq, (own[1] - own[0]) * m.KEYS * 4);
            x.encode(enc, e.p.post, e.p.wait, e.seq, (own[1] - own[0]) * m.KEYS * 4);
            e.bar(enc);
            for (0..vranks) |j| {
                const pj = m.part(R, vranks, j);
                e.copyU32(enc, s.base.at(j * s.stride * 4), e.idx.at(pj[0] * m.KEYS * 4), (pj[1] - pj[0]) * m.KEYS);
            }
            e.bar(enc);
            e.seq += 1;
        } else for (0..vranks) |j| {
            const pj = m.part(R, vranks, j);
            var r0: usize = pj[0];
            while (r0 < pj[1]) : (r0 += e.score_rows) {
                const n = @min(e.score_rows, pj[1] - r0);
                e.idxRows(enc, i, p0, r0, n, last_n, e.idx.at(r0 * m.KEYS * 4));
            }
        }
    }

    /// Key-split indexer (G53_KSPLIT) for a block of <= tk_rows_max rows: this rank scores keys [lo, hi) (its part of the
    /// block's last row's keys), keeps that range's top KEYS as (score, index) candidates in its exchange slot, the ranks
    /// swap them, and every rank merges the union into e.idx (g53_topk's picks and order; bitwise gate: ksplitcheck.swift).
    /// One Mac: every virtual rank's range in turn into the one-Mac slots, then the same merge.
    fn keySplit(e: *Engine, enc: mtl.ComputeEncoder, i: usize, p0: u32, R: usize, last_n: u32, vranks: usize) void {
        const rstride: usize = 4 + 2 * m.KEYS;
        const wscale = 1.0 / (@sqrt(@as(f32, m.IH)) * @sqrt(@as(f32, m.ID)));
        const s = e.slotBase();
        const me: usize = if (e.xc) |x| x.rank else 0;
        const first: usize = if (e.xc != null) me else 0;
        const count: usize = if (e.xc != null) 1 else vranks;
        for (first..first + count) |j| {
            const rg = m.part(last_n, vranks, j);
            const lo: u32 = @intCast(rg[0]);
            const hi: u32 = @intCast(rg[1]);
            const last = j + 1 == vranks;
            // scores: row r scores keys [lo, lo + n_r') with n_r' = (last ? p0 + r + 1 : hi + r) - lo (extra keys past hi are
            // scored but never taken: the select below takes each row's first hi - lo keys only)
            const pq: u32 = if (last) p0 - lo else hi - lo - 1;
            const most: u32 = pq + @as(u32, @intCast(R));
            const span = idxSpan(most, R);
            enc.setPipeline(e.p.idx_scores2);
            bind(enc, 0, .{ e.iq, e.iw, e.ic[i].?.at(@as(usize, lo) * m.ID * e.kv_bytes), e.scores });
            enc.setValue(IArgs2{ .p0 = @intCast(pq), .wscale = wscale, .cap = @intCast(e.cap), .keys = 0, .span = @intCast(span), .rows = @intCast(R) }, 4);
            enc.dispatchGroups(sz(R, (most + span - 1) / span, 1), sz(256, 1, 1));
            e.bar(enc);
            // the range's top KEYS per row on the many-threadgroup select (review N3: one threadgroup over 16K-150K keys is
            // the slow regime), picks as (score, global index) pairs: last rank row r has p0 - lo + r + 1 keys, the
            // others exactly hi - lo (the extra keys scored past hi are never looked at)
            const pairs: TkPairs = .{ .fixed = if (last) 0 else hi - lo, .lo = lo, .rstride = @intCast(rstride), .wpairs = e.p.tk_wpairs, .allpairs = e.p.tk_allpairs, .cand = s.base.at(j * s.stride * 4) };
            encodeTopkFewX(enc, .{ .init = e.p.tk_init, .hist = e.p.tk_hist, .digit = e.p.tk_digit, .count = e.p.tk_count, .scan = e.p.tk_scan, .write = e.p.tk_write }, .{ .scores = e.scores, .idx = e.idx, .state = e.tk_state, .hist = e.tk_hist, .cnt = e.tk_cnt, .base = e.tk_base }, if (last) p0 - lo else 0, R, if (last) most else hi - lo, e.cap, e.conc, pairs);
            e.bar(enc);
        }
        if (e.xc) |x| {
            x.encodeCopies(enc, e.p.copy, e.seq, R * rstride * 4);
            x.encode(enc, e.p.post, e.p.wait, e.seq, R * rstride * 4);
        }
        e.bar(enc);
        enc.setPipeline(e.p.kconcat);
        bind(enc, 0, .{ s.base, e.kcs, e.kci, e.kmn });
        enc.setValue([4]i32{ @intCast(vranks), @intCast(s.stride), @intCast(rstride), m.KEYS }, 4);
        enc.dispatchGroups(sz(R, 1, 1), sz(1024, 1, 1));
        e.bar(enc);
        enc.setPipeline(e.p.kmerge);
        bind(enc, 0, .{ e.kcs, e.kci, e.kmn, e.idx });
        enc.setValue(@as(i32, m.KEYS), 4);
        enc.dispatchGroups(sz(R, 1, 1), sz(1024, 1, 1));
        e.bar(enc);
        if (e.xc != null) e.seq += 1;
    }

    fn attention(e: *Engine, enc: mtl.ComputeEncoder, i: usize, p0: u32, rows: usize) void {
        const L = &e.w.layers[i];
        const a = &L.attn;
        const R = rows;
        const last_n = p0 + @as(u32, @intCast(R)); // the last row's key count
        // G53_LSPLIT: a block reaching past ls_from runs the layer split (layer i's owner: every head); below it, the plain engine
        const deep = e.ls and last_n > e.ls_from;
        const mine = e.owns(i);
        const own = !deep or mine; // this rank runs layer i's attention for this block
        const nh = if (deep) m.H else e.w.share.heads[1] - e.w.share.heads[0];
        const q_b = if (deep and mine) a.q_b_all.? else a.q_b;
        const embed_q = if (deep and mine) a.embed_q_all.? else a.embed_q;
        const unembed = if (deep and mine) a.unembed_all.? else a.unembed;
        const store: usize = if (mine) R else if (p0 < e.ls_from) @min(R, e.ls_from - p0) else 0; // rows this rank keeps
        const sparse = last_n > m.KEYS;
        const full = m.fullIndexer(i);
        const vranks: usize = if (e.xc) |x| x.ranks else e.plan.nslots;
        const split = full and sparse and e.own_front and R >= own_min_rows and R % (32 * vranks) == 0 and !e.verify and vranks > 1 and e.only == null; // 32-aligned shares: a row keeps its tile position
        const ks_pre = full and sparse and !split and e.ksplit and R <= tk_rows_max and last_n >= e.ksplit_min and vranks > 1 and vranks <= 4 and e.only == null;
        const chunked = full and sparse and !split and !ks_pre and R > e.score_rows; // scores + top-k in row chunks (idxRows)
        const repl_q = full and sparse and !split and !chunked; // the replicated indexer query side
        // stage 1 (concurrent encoder): every launch that reads xn
        if (e.on(.q_a)) e.qmv(enc, a.q_a, e.xn, e.qa, .{ .n = m.QL, .K = m.D, .items = R, .xr = m.D, .yr = m.QL });
        if (e.on(.kv_a)) e.qmv(enc, a.kv_a, e.xn, e.kv, .{ .n = m.CROW, .K = m.D, .items = R, .xr = m.D, .yr = m.CROW });
        if (full and e.on(.idx_k)) e.gemv(enc, a.wk.?, e.xn, e.ik, m.ID, m.D, R);
        if (repl_q and e.on(.idx_q)) e.gemv(enc, a.wproj.?, e.xn, e.iw, m.IH, m.D, R);
        e.bar(enc);
        // stage 2: norms and cache stores
        if (e.on(.q_norm)) e.rms(enc, e.qa, a.q_a_norm, e.qr, m.QL, 1e-6, false, R);
        if (store > 0 and e.on(.kv_store)) {
            enc.setPipeline(e.p.kv_store);
            bind(enc, 0, .{ e.kv, a.kv_a_norm, e.kvc[i].? });
            enc.setValue(CArgs{ .pos = @intCast(p0), .eps = 1e-6, .log2base = m.LOG2_THETA }, 3);
            enc.dispatchGroups(sz(store, 1, 1), sz(512, 1, 1));
        }
        if (full and e.on(.idx_k)) {
            enc.setPipeline(e.p.idx_k_store);
            bind(enc, 0, .{ e.ik, a.k_norm_w.?, a.k_norm_b.?, e.ic[i].? });
            enc.setValue(CArgs{ .pos = @intCast(p0), .eps = 1e-5, .log2base = m.LOG2_THETA }, 4);
            enc.dispatchGroups(sz(R, 1, 1), sz(m.ID, 1, 1));
        }
        e.bar(enc);
        // stage 3: what reads qr
        if (own and e.on(.q_b)) e.qmv(enc, q_b, e.qr, e.q, .{ .n = nh * m.QH, .K = m.QL, .items = R, .xr = m.QL, .yr = nh * m.QH });
        if (repl_q and e.on(.idx_q)) e.gemv(enc, a.wq_b.?, e.qr, e.iq, m.IH * m.ID, m.QL, R);
        e.bar(enc);
        // stage 4: rope on q's rope dims, embed_q on its nope dims (disjoint bytes), rope on the indexer queries
        if (own and e.on(.rope_q)) e.rope(enc, e.q, R * nh, m.QH, m.NOPE, p0, nh);
        if (own and e.on(.embed_q)) e.qmv(enc, embed_q, e.q, e.ql, .{ .n = m.KVL, .K = m.NOPE, .items = R, .z = nh, .xr = nh * m.QH, .zx = m.QH, .yr = nh * m.KVL, .zy = m.KVL, .stacked = true });
        if (repl_q and e.on(.idx_q)) e.rope(enc, e.iq, R * m.IH, m.ID, 0, p0, m.IH);
        e.bar(enc);
        const ks = full and sparse and !split and e.ksplit and R <= tk_rows_max and last_n >= e.ksplit_min and vranks > 1 and vranks <= 4 and e.only == null; // KCAND = 4 x KEYS
        if (ks) if (e.idx_dump) |d| if (R == 1) { // review N5: idx dumps see the merged picks too
            e.keySplit(enc, i, p0, R, last_n, vranks);
            enc.setPipeline(e.p.copy_u32);
            bind(enc, 0, .{ e.idx, d.at(i * m.KEYS * 4) });
            enc.setValue(@as(i32, m.KEYS), 2);
            enc.dispatchThreads(sz(m.KEYS, 1, 1), sz(256, 1, 1));
            e.bar(enc);
        };
        const ks_done = ks and e.idx_dump != null and R == 1;
        if (chunked) {
            var r0: usize = 0;
            while (r0 < R) : (r0 += e.score_rows) {
                const n = @min(e.score_rows, R - r0);
                e.idxRows(enc, i, p0, r0, n, last_n, e.idx.at(r0 * m.KEYS * 4));
            }
        } else if (split) e.ownFront(enc, i, p0, R, last_n, vranks) else if (ks) {
            if (!ks_done) e.keySplit(enc, i, p0, R, last_n, vranks);
        } else if (full and sparse) {
            if (e.on(.idx_scores)) {
                const wscale = 1.0 / (@sqrt(@as(f32, m.IH)) * @sqrt(@as(f32, m.ID)));
                const span = idxSpan(last_n, R);
                const two = e.idx_v3 and R > 1;
                enc.setPipeline(if (two) e.p.idx_scores3 else e.p.idx_scores2);
                bind(enc, 0, .{ e.iq, e.iw, e.ic[i].?, e.scores });
                enc.setValue(IArgs2{ .p0 = @intCast(p0), .wscale = wscale, .cap = @intCast(e.cap), .keys = m.KEYS, .span = @intCast(span), .rows = @intCast(R) }, 4);
                enc.dispatchGroups(sz(if (two) (R + 1) / 2 else R, (last_n + span - 1) / span, 1), sz(256, 1, 1)); // rows fastest: a block's rows share a span of keys
                e.bar(enc);
            }
            if (e.on(.idx_topk)) {
                if (R <= tk_rows_max and last_n >= tk_min_keys) {
                    encodeTopkFew(enc, .{ .init = e.p.tk_init, .hist = e.p.tk_hist, .digit = e.p.tk_digit, .count = e.p.tk_count, .scan = e.p.tk_scan, .write = e.p.tk_write }, .{ .scores = e.scores, .idx = e.idx, .state = e.tk_state, .hist = e.tk_hist, .cnt = e.tk_cnt, .base = e.tk_base }, p0, R, last_n, e.cap, e.conc);
                } else {
                    enc.setPipeline(e.p.topk);
                    bind(enc, 0, .{ e.scores, e.idx });
                    enc.setValue(TArgs{ .p0 = @intCast(p0), .top = m.KEYS, .cap = @intCast(e.cap) }, 2);
                    enc.dispatchGroups(sz(R, 1, 1), sz(1024, 1, 1));
                }
                e.bar(enc);
            }
            if (e.idx_dump) |d| if (R == 1) {
                enc.setPipeline(e.p.copy_u32);
                bind(enc, 0, .{ e.idx, d.at(i * m.KEYS * 4) });
                enc.setValue(@as(i32, m.KEYS), 2);
                enc.dispatchThreads(sz(m.KEYS, 1, 1), sz(256, 1, 1));
            };
        }
        if (own) {
            var r0: usize = 0;
            while (r0 < R) : (r0 += e.attn_rows) e.attnRows(enc, i, p0, r0, @min(e.attn_rows, R - r0), nh, R);
        }
        if (own and e.on(.unembed)) e.qmv(enc, unembed, e.att, e.vals, .{ .n = m.VD, .K = m.KVL, .items = R, .z = nh, .xr = nh * m.KVL, .zx = m.KVL, .yr = nh * m.VD, .zy = m.VD, .stacked = true });
        e.bar(enc);
        if (deep and e.only == null) e.lsSwap(enc, i, R); // every rank now has the owner's values for its heads
        // o_proj: each slice's columns (its heads) into its slot (layer split: e.vals holds every head, columns by head)
        for (e.plan.slices, 0..) |s, si| {
            const k0 = (s.heads[0] - e.w.share.heads[0]) * m.VD;
            const K = (s.heads[1] - s.heads[0]) * m.VD;
            // layer split: the owner reads its heads' columns of every head's values; the others their own values
            const xv = if (deep and mine) e.vals.at(s.heads[0] * m.VD * 4) else if (deep) e.vals_own.at(k0 * 4) else e.vals.at(k0 * 4);
            const xr = if (deep and !mine) (e.w.share.heads[1] - e.w.share.heads[0]) * m.VD else nh * m.VD;
            if (e.on(.o_proj)) e.qmv(enc, a.o_proj, xv, e.slotFor(si), .{ .n = m.D, .k0 = k0, .K = K, .items = R, .xr = xr, .yr = m.D });
        }
    }

    /// Attention for rows [r0, r0 + n) of a block at p0 (nh heads; e.idx's picks for those rows): every kernel per row,
    /// the same launches a whole block would take on those rows (row chunks keep the partials small).
    fn attnRows(e: *Engine, enc: mtl.ComputeEncoder, i: usize, p0: u32, r0: usize, n: usize, nh: usize, rv: usize) void {
        const pa: u32 = p0 + @as(u32, @intCast(r0));
        const ql = e.ql.at(r0 * nh * m.KVL * 4);
        const qp = e.q.at((r0 * nh * m.QH + m.NOPE) * 4);
        const idx = e.idx.at(r0 * m.KEYS * 4);
        const att = e.att.at(r0 * nh * m.KVL * 4);
        const blocks = (@min(pa + @as(u32, @intCast(n)), m.KEYS) + 31) / 32;
        const aa: AArgs = .{ .p0 = @intCast(pa), .sparse = 1, .scale = 1.0 / 16.0, .heads = @intCast(nh), .keys = m.KEYS, .qp_stride = m.QH };
        if (!e.on(.attn)) return;
        bind(enc, 0, .{ ql, qp, e.kvc[i].?, idx, e.po, e.pm, e.pl });
        enc.setValue(aa, 7);
        if (e.attn4h) {
            enc.setPipeline(e.p.attn_block4h);
            enc.dispatchGroups(sz(blocks, (nh + 3) / 4, n), sz(256, 1, 1));
        } else {
            enc.setPipeline(e.p.attn_block);
            enc.dispatchGroups(sz(blocks, (nh + 7) / 8, n), sz(256, 1, 1));
        }
        e.bar(enc);
        enc.setPipeline(if (e.joinb and rv <= 16) e.p.attn_joinb else e.p.attn_join);
        bind(enc, 0, .{ e.po, e.pm, e.pl, att });
        enc.setValue(aa, 4);
        enc.dispatchGroups(sz(nh, n, 1), sz(512, 1, 1));
        e.bar(enc);
    }

    fn mlp(e: *Engine, enc: mtl.ComputeEncoder, i: usize, rows: usize) void {
        const L = &e.w.layers[i];
        const R = rows;
        switch (L.mlp) {
            .dense => |*d| {
                const lo = e.w.share.dense[0];
                const ir = e.w.share.dense[1] - lo;
                if (e.on(.gateup)) e.gateup(enc, d.gate, d.up, e.xn, e.act, .{ .n = ir, .K = m.D, .items = R, .xr = m.D, .yr = ir });
                e.bar(enc);
                for (e.plan.slices, 0..) |s, si| {
                    const k0 = s.dense[0] - lo;
                    const K = s.dense[1] - s.dense[0];
                    if (e.on(.down)) e.qmv(enc, d.down, e.act.at(k0 * 4), e.slotFor(si), .{ .n = m.D, .k0 = k0, .K = K, .items = R, .xr = ir, .yr = m.D });
                }
            },
            .moe => |*x| {
                const lo = e.w.share.moe[0];
                const ir = e.w.share.moe[1] - lo;
                const P = R * m.TOP;
                const act_sh = e.act.at(P * ir * 4);
                const one = e.plan.slices.len == 1; // 4-way: one slice a rank -> the shared expert's down runs early
                // stage 1: the router and the shared expert's gate-up both read xn
                if (e.on(.router)) e.gemv(enc, x.router, e.xn, e.logits_r, m.E, m.D, R);
                if (e.on(.gateup_sh)) e.gateup(enc, x.sh_gate, x.sh_up, e.xn, act_sh, .{ .n = ir, .K = m.D, .items = R, .xr = m.D, .yr = ir });
                e.bar(enc);
                // stage 2: route (and, one slice a rank, the shared expert's down beside it)
                if (e.on(.router)) {
                    enc.setPipeline(e.p.route);
                    bind(enc, 0, .{ e.logits_r, x.bias, e.ids, e.wts });
                    const direct = R == 1;
                    enc.setValue(MArgs{ .experts = m.E, .top = m.TOP, .scale = 2.5, .direct = @intFromBool(direct) }, 4);
                    bind(enc, 5, .{ e.se, e.sr, e.ss, e.seg });
                    enc.dispatchGroups(sz(R, 1, 1), sz(m.E, 1, 1));
                }
                if (one) {
                    const s = e.plan.slices[0];
                    const k0 = s.moe[0] - lo;
                    const K = s.moe[1] - s.moe[0];
                    if (e.on(.down_sh)) e.qmv(enc, x.sh_down, act_sh.at(k0 * 4), e.ys, .{ .n = m.D, .k0 = k0, .K = K, .items = R, .xr = ir, .yr = m.D });
                }
                e.bar(enc);
                if (one and i == lw_layer) {
                    const s = e.plan.slices[0];
                    e.lwApply(enc, act_sh.at((s.moe[0] - lo) * 4), ir, s, R);
                }
                if (e.on(.router) and R != 1) {
                    enc.setPipeline(e.p.sort_picks);
                    bind(enc, 0, .{ e.ids, e.se, e.sr, e.ss });
                    enc.setValue([3]i32{ @intCast(P), m.TOP, if (e.tiles) 32 else 1 }, 4);
                    bind(enc, 5, .{e.seg});
                    enc.dispatchGroups(sz(1, 1, 1), sz(1024, 1, 1));
                    e.bar(enc);
                }
                if (e.on(.gateup)) e.gateup(enc, x.gate, x.up, e.xn, e.act, .{ .n = ir, .K = m.D, .items = P, .xr = m.D, .yr = ir, .ids = e.se, .xids = e.sr, .yids = e.ss, .seg = e.seg });
                e.bar(enc);
                for (e.plan.slices, 0..) |s, si| {
                    const k0 = s.moe[0] - lo;
                    const K = s.moe[1] - s.moe[0];
                    if (e.on(.down)) e.qmv(enc, x.down, e.act.at(k0 * 4), e.y, .{ .n = m.D, .k0 = k0, .K = K, .items = P, .xr = ir, .yr = m.D, .ids = e.se, .xids = e.ss, .yids = e.ss, .seg = e.seg, .stacked = true });
                    if (!one and e.on(.down_sh)) e.qmv(enc, x.sh_down, act_sh.at(k0 * 4), e.ys, .{ .n = m.D, .k0 = k0, .K = K, .items = R, .xr = ir, .yr = m.D });
                    if (!one and i == lw_layer) e.lwApply(enc, act_sh.at(k0 * 4), ir, s, R);
                    e.bar(enc);
                    if (e.on(.combine)) {
                        enc.setPipeline(e.p.moe_combine);
                        bind(enc, 0, .{ e.y, e.wts, e.slotFor(si) });
                        enc.setValue(CbArgs{ .D = m.D, .top = m.TOP }, 3);
                        bind(enc, 4, .{e.ys});
                        enc.dispatchThreads(sz(m.D, R, 1), sz(256, 1, 1));
                    }
                    e.bar(enc); // the next slice reuses y and ys
                }
            },
        }
    }

    /// Living Weights: slice `s`'s part of the committed change, 10 (x_s a_s^T) b, added into ys (its shared expert's output)
    /// before moe_combine sums ys into the slice's slot. x = the slice's site input (silu(gate) up, fp32, rows `xr` apart).
    /// Nothing is encoded without a change (no Lw, off, or no served ranks): the stock launches exactly.
    fn lwApply(e: *Engine, enc: mtl.ComputeEncoder, x: Ref, xr: usize, s: m.Share, rows: usize) void {
        const l = e.lw orelse return;
        if (!l.on or l.served == 0 or !e.on(.down_sh)) return;
        const a: LwArgs = .{ .K = @intCast(s.moe[1] - s.moe[0]), .xr = @intCast(xr), .c0 = @intCast(s.moe[0]), .R = @intCast(l.served), .rows = @intCast(rows) };
        e.bar(enc); // sh_down's ys written
        enc.setPipeline(e.p.lw_in);
        bind(enc, 0, .{ x, l.a, l.u });
        enc.setValue(a, 3);
        enc.dispatchThreads(sz(l.served, rows, 1), sz(@min(l.served, 256), 1, 1));
        e.bar(enc);
        enc.setPipeline(e.p.lw_out);
        bind(enc, 0, .{ l.u, l.b, e.ys });
        enc.setValue(a, 3);
        enc.dispatchThreads(sz(m.D, rows, 1), sz(256, 1, 1));
        e.bar(enc);
    }

    /// Encode a block of `rows` rows at positions p0.. (their tokens in `tok`); `dump_at`: append the block's last row
    /// of h after the embedding and every layer there. `head`: the last row's pick into tok[0].
    pub fn step(e: *Engine, enc: mtl.ComputeEncoder, p0: u32, rows: usize, dump_at: ?usize, head: bool) void {
        std.debug.assert(rows >= 1 and rows <= e.max_rows);
        e.tiles = rows > 1 and !e.verify;
        const last = (rows - 1) * m.D * 4;
        enc.setPipeline(e.p.embed);
        bind(enc, 0, .{ e.w.embed, e.tok, e.h });
        enc.setValue(@as(i32, m.D), 3);
        enc.dispatchThreads(sz(m.D, rows, 1), sz(256, 1, 1));
        e.bar(enc);
        var at = dump_at;
        if (at) |o| {
            e.copy(enc, e.h.at(last), e.dump.?.at(o * 4), m.D);
            at = o + m.D;
        }
        for (e.w.first..e.w.last) |i| {
            if (i == e.w.first) {
                e.rms(enc, e.h, e.w.layers[i].in_norm, e.xn, m.D, 1e-5, i == 0, rows);
                e.bar(enc);
            }
            e.attention(enc, i, p0, rows);
            e.reduce(enc, e.w.layers[i].post_norm, rows);
            e.mlp(enc, i, rows);
            e.reduce(enc, if (i + 1 < e.w.last) e.w.layers[i + 1].in_norm else e.w.norm, rows);
            if (at) |o| {
                e.copy(enc, e.h.at(last), e.dump.?.at(o * 4), m.D);
                at = o + m.D;
                e.bar(enc);
            }
        }
        if (!head) return;
        e.headPick(enc, rows);
    }

    /// The block's last row's pick into tok[0] (verify blocks: a pick per row into `picks`).
    fn headPick(e: *Engine, enc: mtl.ComputeEncoder, rows: usize) void {
        const last = (rows - 1) * m.D * 4;
        if (e.verify) return e.headRows(enc, e.xn, rows, e.picks);
        const vr = e.w.share.vocab[1] - e.w.share.vocab[0];
        e.gemv(enc, e.w.head, e.xn.at(last), e.logits, vr, m.D, 1);
        e.bar(enc);
        const argslot: Ref = if (e.xc) |x| x.argSlot(e.seq) else e.args;
        enc.setPipeline(e.p.argmax);
        bind(enc, 0, .{ e.logits, argslot });
        enc.setValue([2]i32{ @intCast(vr), @intCast(e.w.share.vocab[0]) }, 2);
        enc.dispatchGroups(sz(1, 1, 1), sz(1024, 1, 1));
        if (e.xc) |x| {
            x.encodeArgs(enc, e.p.post, e.p.wait, e.seq);
            enc.setPipeline(e.p.pick);
            bind(enc, 0, .{ x.argBase(e.seq), e.tok });
            enc.setValue([2]i32{ @intCast(x.ranks), @intCast(x.argFloats()) }, 2);
            enc.dispatchGroups(sz(1, 1, 1), sz(1, 1, 1));
            e.seq += 1;
        } else {
            e.bar(enc);
            enc.setPipeline(e.p.pick);
            bind(enc, 0, .{ e.args, e.tok });
            enc.setValue([2]i32{ 1, 2 }, 2);
            enc.dispatchGroups(sz(1, 1, 1), sz(1, 1, 1));
        }
    }

    /// One block in its own command buffer: encode, commit, serve the exchanges, wait. Returns the GPU seconds.
    pub fn runRows(e: *Engine, p0: u32, rows: usize, dump_at: ?usize, head: bool) !f64 {
        const first = e.seq;
        const cb = e.queue.commandBufferUnretained();
        const enc = cb.compute(e.dtype());
        e.step(enc, p0, rows, dump_at, head);
        enc.end();
        cb.commit();
        if (e.xc) |x| try x.serve(first, e.seq);
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("glm53: command buffer failed: {s}", .{msg});
            return error.GpuFailed;
        }
        if (e.xc) |x| if (x.gaveUp() != 0) return error.ExchangeGaveUp;
        return cb.gpuSeconds();
    }

    /// A prompt block, then the head absorbs its first `k` rows (next tokens already in mtok) in the same buffer.
    fn runRowsMtp(e: *Engine, p0: u32, rows: usize, head: bool, k: usize) !f64 {
        const t_start = mtl.clock.seconds();
        const first = e.seq;
        const cb = e.queue.commandBufferUnretained();
        const enc = cb.compute(e.dtype());
        e.step(enc, p0, rows, null, head);
        if (k > 0) e.mtpLayer(enc, e.xn, k, p0, true);
        enc.end();
        const t_enc = mtl.clock.seconds();
        cb.commit();
        if (e.xc) |x| try x.serve(first, e.seq);
        cb.wait();
        e.pf_enc += t_enc - t_start;
        e.pf_wall += mtl.clock.seconds() - t_enc;
        e.pf_gpu += cb.gpuSeconds();
        e.pf_blocks += 1;
        if (cb.failure()) |msg| {
            std.log.err("glm53: command buffer failed: {s}", .{msg});
            return error.GpuFailed;
        }
        if (e.xc) |x| if (x.gaveUp() != 0) return error.ExchangeGaveUp;
        return cb.gpuSeconds();
    }

    pub fn run(e: *Engine, pos: u32, dump_at: ?usize, head: bool) !f64 {
        return e.runRows(pos, 1, dump_at, head);
    }

    /// A prompt from position p0 in blocks of up to max_rows; the last block's pick is the first reply token.
    pub fn prefill(e: *Engine, p0: u32, ids: []const u32) !f64 {
        var gpu: f64 = 0;
        var at: usize = 0;
        while (at < ids.len) {
            const n = @min(ids.len - at, e.max_rows);
            @memcpy(e.tok.buf.slice(u32, n), ids[at..][0..n]);
            const lastb = at + n == ids.len;
            if (e.w.mtp != null) {
                // the head absorbs every row whose next token is known (all but the prompt's last row)
                const k = if (lastb) n - 1 else n;
                @memcpy(e.mtok.buf.slice(u32, k), ids[at + 1 ..][0..k]);
                gpu += try e.runRowsMtp(p0 + @as(u32, @intCast(at)), n, lastb, k);
            } else gpu += try e.runRows(p0 + @as(u32, @intCast(at)), n, null, lastb);
            e.last_rows = n;
            at += n;
        }
        return gpu;
    }

    /// Each launch class alone at layer `i` and position `pos`, `reps` times in one command buffer: GPU ms a time.
    pub fn profile(e: *Engine, i: usize, pos: u32, reps: usize, rows: usize) !void {
        const conc = e.conc;
        const own = e.own_front;
        const ks = e.ksplit;
        e.conc = false; // profiles time each class alone on a serial encoder, with no exchanges
        e.own_front = false;
        e.ksplit = false;
        defer {
            e.conc = conc;
            e.own_front = own;
            e.ksplit = ks;
        }
        var total: f64 = 0;
        e.tiles = rows > 1;
        inline for (@typeInfo(Op).@"enum".field_names) |name| {
            e.only = @field(Op, name);
            const cb = e.queue.commandBufferUnretained();
            const enc = cb.compute(.serial);
            for (0..reps) |_| {
                e.attention(enc, i, pos, rows);
                if (e.on(.sum)) {
                    enc.setPipeline(e.p.sum_res_rms);
                    const s = e.slotBase();
                    bind(enc, 0, .{ s.base, e.h, e.w.norm, e.xn });
                    enc.setValue(SArgs{ .D = m.D, .S = @intCast(e.plan.nslots), .stride = @intCast(s.stride), .eps = 1e-5, .norm = 1 }, 4);
                    enc.dispatchGroups(sz(rows, 1, 1), sz(1024, 1, 1));
                }
                e.mlp(enc, i, rows);
            }
            enc.end();
            cb.commit();
            cb.wait();
            const ms = cb.gpuSeconds() * 1e3 / @as(f64, @floatFromInt(reps));
            total += ms;
            std.debug.print("  {s:<10} {d:8.4} ms\n", .{ name, ms });
        }
        e.only = null;
        std.debug.print("  {s:<10} {d:8.4} ms (layer {d}, pos {d}, rows {d})\n", .{ "TOTAL", total, i, pos, rows });
    }

    /// Each launch class alone over every loaded layer once (cold weights, as a decode step reads them): ms a step.
    pub fn profileLayers(e: *Engine, pos: u32, rows: usize) !void {
        const conc = e.conc;
        const own = e.own_front;
        const ks = e.ksplit;
        e.conc = false; // profiles time each class alone on a serial encoder, with no exchanges
        e.own_front = false;
        e.ksplit = false;
        defer {
            e.conc = conc;
            e.own_front = own;
            e.ksplit = ks;
        }
        var total: f64 = 0;
        e.tiles = rows > 1;
        inline for (@typeInfo(Op).@"enum".field_names) |name| {
            e.only = @field(Op, name);
            const cb = e.queue.commandBufferUnretained();
            const enc = cb.compute(.serial);
            for (e.w.first..e.w.last) |i| {
                e.attention(enc, i, pos, rows);
                if (e.on(.sum)) for (0..2) |_| {
                    enc.setPipeline(e.p.sum_res_rms);
                    const s = e.slotBase();
                    bind(enc, 0, .{ s.base, e.h, e.w.norm, e.xn });
                    enc.setValue(SArgs{ .D = m.D, .S = @intCast(e.plan.nslots), .stride = @intCast(s.stride), .eps = 1e-5, .norm = 1 }, 4);
                    enc.dispatchGroups(sz(rows, 1, 1), sz(1024, 1, 1));
                };
                e.mlp(enc, i, rows);
            }
            enc.end();
            cb.commit();
            cb.wait();
            const ms = cb.gpuSeconds() * 1e3;
            total += ms;
            std.debug.print("  {s:<10} {d:8.3} ms\n", .{ name, ms });
        }
        e.only = null;
        std.debug.print("  {s:<10} {d:8.3} ms (layers {d}..{d}, pos {d}, rows {d})\n", .{ "TOTAL", total, e.w.first, e.w.last, pos, rows });
    }

    /// The head over `rows` final-norm rows at `x` (row path): each row's greedy pick into out[r] (u32), on every rank.
    /// Each row's arithmetic is the one-row head's (same gemv, argmax and pick per row).
    pub fn headRows(e: *Engine, enc: mtl.ComputeEncoder, x: Ref, rows: usize, out: Ref) void {
        std.debug.assert(rows <= verify_max);
        const vr = e.w.share.vocab[1] - e.w.share.vocab[0];
        const tiles = e.tiles;
        e.tiles = false;
        e.gemv(enc, e.w.head, x, e.vlogits, vr, m.D, rows);
        e.tiles = tiles;
        e.bar(enc);
        const argslot: Ref = if (e.xc) |xx| xx.argSlot(e.seq) else e.vargs;
        enc.setPipeline(e.p.argmax_rows);
        bind(enc, 0, .{ e.vlogits, argslot });
        enc.setValue([2]i32{ @intCast(vr), @intCast(e.w.share.vocab[0]) }, 2);
        enc.dispatchGroups(sz(rows, 1, 1), sz(1024, 1, 1));
        if (e.xc) |xx| {
            xx.encodeArgsN(enc, e.p.post, e.p.wait, e.seq, rows * 8);
            enc.setPipeline(e.p.pick_rows);
            bind(enc, 0, .{ xx.argBase(e.seq), out });
            enc.setValue([3]i32{ @intCast(xx.ranks), @intCast(xx.argFloats()), @intCast(rows) }, 2);
            enc.dispatchThreads(sz(rows, 1, 1), sz(verify_max, 1, 1));
            e.seq += 1;
        } else {
            e.bar(enc);
            enc.setPipeline(e.p.pick_rows);
            bind(enc, 0, .{ e.vargs, out });
            enc.setValue([3]i32{ 1, 0, @intCast(rows) }, 2);
            enc.dispatchThreads(sz(rows, 1, 1), sz(verify_max, 1, 1));
        }
        e.bar(enc);
    }

    fn copyU32(e: *Engine, enc: mtl.ComputeEncoder, src: Ref, dst: Ref, n: usize) void {
        enc.setPipeline(e.p.copy_u32);
        bind(enc, 0, .{ src, dst });
        enc.setValue(@as(i32, @intCast(n)), 2);
        enc.dispatchThreads(sz(n, 1, 1), sz(64, 1, 1));
    }

    /// The draft head over `rows` rows at positions p0..: h rows at `hn` (trunk final-norm rows, or a chain step's raw
    /// head output), next tokens in mtok. Leaves the head's raw output in h and shared_head.norm(it) in xn.
    pub fn mtpLayer(e: *Engine, enc: mtl.ComputeEncoder, hn: Ref, rows: usize, p0: u32, tiles: bool) void {
        const x = e.w.mtp.?;
        const L = &e.w.layers[m.MTP];
        e.tiles = tiles and rows > 1;
        enc.setPipeline(e.p.embed);
        bind(enc, 0, .{ e.w.embed, e.mtok, e.m_emb });
        enc.setValue(@as(i32, m.D), 3);
        enc.dispatchThreads(sz(m.D, rows, 1), sz(256, 1, 1));
        e.bar(enc);
        enc.setPipeline(e.p.mtp_cat);
        bind(enc, 0, .{ e.m_emb, hn, x.enorm, x.hnorm, e.m_cat });
        enc.setValue(NArgs{ .D = m.D, .eps = 1e-5, .round_bf16 = 0, .ld = m.D }, 5);
        enc.dispatchGroups(sz(rows, 1, 1), sz(1024, 1, 1));
        e.bar(enc);
        e.gemv(enc, x.eh_proj, e.m_cat, e.h, m.D, 2 * m.D, rows);
        e.bar(enc);
        e.rms(enc, e.h, L.in_norm, e.xn, m.D, 1e-5, false, rows);
        e.bar(enc);
        e.attention(enc, m.MTP, p0, rows);
        e.reduce(enc, L.post_norm, rows);
        e.mlp(enc, m.MTP, rows);
        e.reduce(enc, x.norm, rows);
    }

    /// One MTP round in one command buffer. First the head absorbs the previous block's accepted rows (xn rows
    /// r0..r0+n-1 at positions hp0.., next tokens `next`) and drafts k tokens (chained); then the verify block
    /// [t, drafts] at position p runs through the trunk with a pick per row. After it: picks[0..k], dtok[0..k).
    /// Without a head (k must be 0) it is the plain step with the row-path head.
    pub fn specRound(e: *Engine, r0: usize, next: []const u32, hp0: u32, t: u32, p: u32, k: usize, force: ?[]const u32) !f64 {
        return e.specRoundCopy(r0, next, hp0, t, p, k, force, true);
    }

    /// specRound with `chain` false: the head only absorbs (copy drafts in `force` replace its chain).
    pub fn specRoundCopy(e: *Engine, r0: usize, next: []const u32, hp0: u32, t: u32, p: u32, k: usize, force: ?[]const u32, chain: bool) !f64 {
        std.debug.assert(k + 1 <= verify_max);
        const t_start = mtl.clock.seconds();
        const first = e.seq;
        const cb = e.queue.commandBufferUnretained();
        const enc = cb.compute(e.dtype());
        if (e.w.mtp != null and next.len > 0) {
            const n = next.len;
            @memcpy(e.mtok.buf.slice(u32, n), next);
            e.mtpLayer(enc, e.xn.at(r0 * m.D * 4), n, hp0, false);
            if (k > 0 and chain) e.chainDrafts(enc, n - 1, hp0, k);
        } else std.debug.assert(k == 0 or force != null);
        e.tok.buf.slice(u32, 1)[0] = t;
        if (force) |f| { // tests: these drafts instead of the head's
            @memcpy(e.tok.buf.slice(u32, k + 1)[1..], f[0..k]);
        } else if (k > 0) e.copyU32(enc, e.dtok, e.tok.at(4), k);
        e.bar(enc);
        e.verify = true;
        e.step(enc, p, k + 1, null, true);
        e.verify = false;
        enc.end();
        const t_enc = mtl.clock.seconds();
        cb.commit();
        if (e.xc) |x| try x.serve(first, e.seq);
        cb.wait();
        const t_done = mtl.clock.seconds();
        if (cb.failure()) |msg| {
            std.log.err("glm53: command buffer failed: {s}", .{msg});
            return error.GpuFailed;
        }
        if (e.xc) |x| if (x.gaveUp() != 0) return error.ExchangeGaveUp;
        const g = cb.gpuSeconds();
        e.st_enc += t_enc - t_start;
        e.st_wall += t_done - t_enc;
        e.st_gpu += g;
        e.st_rounds += 1;
        return g;
    }

    /// The head's chain after an absorb of rows ending at xn row lr: k drafts into dtok.
    fn chainDrafts(e: *Engine, enc: mtl.ComputeEncoder, lr: usize, hp0: u32, k: usize) void {
        e.headRows(enc, e.xn.at(lr * m.D * 4), 1, e.dtok);
        for (1..k) |c| {
            e.copy(enc, e.h.at((if (c == 1) lr else 0) * m.D * 4), e.m_hin, m.D);
            e.copyU32(enc, e.dtok.at((c - 1) * 4), e.mtok, 1);
            e.bar(enc);
            e.mtpLayer(enc, e.m_hin, 1, hp0 + @as(u32, @intCast(lr + c)), false);
            e.headRows(enc, e.xn, 1, e.dtok.at(c * 4));
        }
    }

    /// Touch every 16 KB of the first n rows of each KV cache (trunk and head) in one command buffer (G53_TOUCH, after a
    /// snapshot load): the page cost of freshly written buffers is paid here instead of by the next job's first block.
    pub fn touchKv(e: *Engine, n: u32) void {
        const cb = e.queue.commandBufferUnretained();
        const enc = cb.compute(.serial);
        for (0..m.LAYERS + 1) |i| {
            const parts = [_]?Ref{ e.kvc[i], e.ic[i] };
            const widths = [_]usize{ m.CROW, m.ID };
            const held = [_]u32{ @min(n, e.kvRows(i)), n }; // a layer-split rank's other layers hold positions < ls_from only
            for (parts, widths, held) |pr, wd, hn| {
                const r = pr orelse continue;
                const bytes = @as(usize, hn) * wd * e.kv_bytes;
                const pages = (bytes + 16383) / 16384;
                if (pages == 0) continue;
                enc.setPipeline(e.p.touch);
                bind(enc, 0, .{ r, e.kmn }); // the sink: scratch no job relies on between jobs
                enc.setValue([2]u32{ @intCast(pages), 4096 }, 2);
                enc.dispatchThreads(sz(pages, 1, 1), sz(256, 1, 1));
            }
        }
        enc.end();
        cb.commit();
        cb.wait();
    }

    /// Only the head's absorb of rows (no drafts, no verify): the end of a reply, so its cache has every kept row.
    pub fn absorb(e: *Engine, r0: usize, next: []const u32, hp0: u32) !void {
        if (e.w.mtp == null or next.len == 0) return;
        const first = e.seq;
        const cb = e.queue.commandBufferUnretained();
        const enc = cb.compute(e.dtype());
        @memcpy(e.mtok.buf.slice(u32, next.len), next);
        e.mtpLayer(enc, e.xn.at(r0 * m.D * 4), next.len, hp0, false);
        enc.end();
        cb.commit();
        if (e.xc) |x| try x.serve(first, e.seq);
        cb.wait();
        if (e.xc) |x| if (x.gaveUp() != 0) return error.ExchangeGaveUp;
    }

    pub fn picksSlice(e: *Engine, n: usize) []const u32 {
        return e.picks.buf.slice(u32, n);
    }
    pub fn draftsSlice(e: *Engine, n: usize) []const u32 {
        return e.dtok.buf.slice(u32, n);
    }

    pub fn token(e: *Engine) u32 {
        return e.tok.buf.slice(u32, 1)[0];
    }

    pub fn setToken(e: *Engine, t: u32) void {
        e.tok.buf.slice(u32, 1)[0] = t;
    }
};
