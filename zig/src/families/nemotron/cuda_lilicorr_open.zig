//! The LiLiCorr open sequence: the Draft's lifecycle (create/drop, per-stream open and
//! release), the device scratch carve, streamBytesFor, and init() with its probe gates.

const std = @import("std");
const cuda = @import("cuda");
const head = @import("cuda_lilicorr_head.zig");
const lc = @import("cuda_lilicorr.zig");
const stage = @import("cuda_lilicorr_stage.zig");
const stack2 = @import("cuda_lilicorr_stack.zig");
const bgate = @import("cuda_lilicorr_bodygate.zig");

const lanes = @import("lanes");
const Stream = lanes.Stream;
const core = @import("core");

const deviceTokenStatesAB = stage.deviceTokenStatesAB;
const deviceLatticeAttentionAB = stage.deviceLatticeAttentionAB;
const deviceFactorEdgeAB = stage.deviceFactorEdgeAB;
const deviceLayerStackAB = stack2.deviceLayerStackAB;
const deviceBodyFixtureAB = bgate.deviceBodyFixtureAB;
const deviceFullChainAB = bgate.deviceFullChainAB;
const E = lc.E;
const Config = lc.Config;
const Weights = lc.Weights;
const run = lc.run;
const roundq = @import("cuda_lilicorr_round.zig");
const norm = roundq.norm;
const projDense = roundq.projDense;
const staging_rows = lc.staging_rows;
const words_at = lc.words_at;
const segments_at = lc.segments_at;
const anchor_at = lc.anchor_at;
const token_emb_at = lc.token_emb_at;
const pass_at = lc.pass_at;
const scores_bytes = lc.scores_bytes;
const max_streams = lc.max_streams;
const Segment = lc.Segment;
const Ctx = lc.Ctx;
const Draft = lc.Draft;

/// S5b: the device chain's scratch — one carve at open, pointers into it (byte offsets are
/// 256-aligned). The buffer-DAG comment lives in stageDeviceScratch.
pub const DeviceScratch = struct {
    buf: cuda.DeviceBuffer = undefined,
    staged: bool = false,
    pass_states: u64 = 0, // [hd] f32 — the pass row's projection
    anchor_state: u64 = 0, // [hd] f32 — the anchor's projection (pre-norm)
    anchor_normed: u64 = 0, // [hd] f32 — the anchor's f32 rms output
    anchor_bits: u64 = 0, // [hd] bf16 — anchor_out_head's input bits
    fm_up: u64 = 0, // [56, hd] f32 — the feature mlp's hidden
    fm_up_bits: u64 = 0, // [56, hd] bf16 — silu'd store
    fm: u64 = 0, // [56, hd] f32 — the feature mlp's output
    lattice: u64 = 0, // [56, hd] bf16 — the assembly + the layers' alternating add
    lattice2: u64 = 0, // [56, hd] bf16 — the layers' add output
    ain: u64 = 0, // [56, hd] bf16 — the rms outputs
    ain_f32: u64 = 0, // [56, hd] f32 — widened factor input
    qkv: u64 = 0, // [56, 3*hd] f32
    att: u64 = 0, // [56, hd] f32
    proj: u64 = 0, // [56, hd] f32
    up: u64 = 0, // [56, 2*hd] f32
    up_bf16: u64 = 0, // [56, 2*hd] bf16
    pre_bits: u64 = 0, // [56, hd] bf16 — the factor's output
    edges: u64 = 0, // [56, 2*factor] f32 + the [hd] f32 anchor tail
    anchor_tail_at: u64 = 0, // byte offset of the anchor tail inside edges
    start: u64 = 0, // [topk] f32
    pair: u64 = 0, // [(slots-1)*topk*topk] f32
    fused_w: u64 = 0, // [2*factor, hd] f32 — out_head ∥ in_head, staged at open
    fused_b: u64 = 0, // [2*factor] f32
    bias: u64 = 0, // [heads*56*56] f32 — buildBias, staged at open
    feat_bits: u64 = 0, // [56,5] bf16 — the host-computed features, uploaded per hold
};

/// The owner A4 reset: a context's FRESH-OPEN state — exactly what create() hands back.
/// Both open paths call it (the brand-new allocation here and every pool pop in openStream)
/// so a reuse path and a first open cannot drift: the K/V + tap ring bytes re-zero (prior
/// stream rows can never leak into the next stream's holds), and every per-stream scalar —
/// ring positions begin/end, the held drafts (held_n and the array bytes), the anchor gate —
/// returns to the create() defaults. span_base rides along (null = no prompt phase seen).
pub fn resetStream(ctx: *Ctx) !void {
    try ctx.ring.fill8(0, null);
    ctx.begin = 0;
    ctx.end = 0;
    ctx.held = @splat(0);
    ctx.held_n = 0;
    ctx.anchor_valid = false;
    ctx.span_base = null;
}

pub fn create(d: *Draft) !*Ctx {
    const ctx = try d.gpa.create(Ctx);
    errdefer d.gpa.destroy(ctx);
    ctx.* = .{ .ring = try cuda.DeviceBuffer.alloc(d.e.ctx.d, d.c.ringBytes()) };
    errdefer ctx.ring.free();
    try resetStream(ctx);
    return ctx;
}

pub fn drop(d: *Draft, ctx: *Ctx) void {
    ctx.ring.free();
    d.gpa.destroy(ctx);
}

pub fn openStream(d: *Draft, s: *Stream) !void {
    if (d.streams.contains(s)) return error.StreamOpen;
    const ctx = d.pool.pop() orelse try create(
        d,
    );
    // The owner A4 cure: a POOLED context re-opens through the same resetStream create()
    // uses — without this line the reuse path kept the prior stream's ring bytes, begin/end,
    // held drafts and anchor gate (the fresh-open state create() gives, lost).
    try resetStream(ctx);
    // The map put keeps ownership: on its failure the LIVE context returns to the pool, and
    // only a pool append that ITSELF fails (gpa exhausted) actually drops it. The old shape
    // destroyed the context first and then pooled the freed pointer — a use-after-free.
    errdefer d.pool.append(d.gpa, ctx) catch drop(d, ctx);
    try d.streams.put(d.gpa, s, ctx);
}

pub fn release(d: *Draft, s: *Stream) void {
    if (d.streams.fetchRemove(s)) |kv| d.pool.append(d.gpa, kv.value) catch drop(d, kv.value);
}

pub fn held(d: *Draft, streams: []const *Stream, out: []const []u32) !void {
    for (streams, out) |s, o| {
        const ctx = d.streams.get(s) orelse return error.UnknownStream;
        if (o.len > ctx.held_n) return error.TooFewDrafts;
        @memcpy(o, ctx.held[0..o.len]);
    }
}

pub fn taps(d: *Draft) []const u32 {
    return d.c.taps[0..d.c.n_taps];
}

/// Adopt the loader's body weights into the draft and run the staged open init: the
/// draft owns `w` from the store on (init() re-stores it), and a failure that arrives
/// before init() has registered its own ownership (open_stage still .none) hands the
/// weights straight back — the partial-failure free the old call-site lacked (owner A6).
pub fn attachWeights(d: *Draft, gpa: std.mem.Allocator, io: std.Io, e: *E, c: Config, dir: []const u8, w: Weights) !void {
    d.w = w; // init() re-stores it: the draft's field is authoritative from the store on
    errdefer if (d.open_stage == .none) d.w.deinit(gpa); // owner A6: init() had not taken ownership yet
    try init(d, gpa, io, e, c, w, dir);
}

/// Release every buffer and module the drafter holds; the caller frees the Draft itself.
/// Staged ownership per Draft.open_stage: every pooled context drops — the idle pool's
/// contexts are owned too, and the old teardown freed only the pool's list storage.
pub fn deinit(d: *Draft) void {
    if (d.open_stage == .none) return; // a failed open() before init(): nothing is owned yet
    // The body weights the loader staged: the device arena plus the host layers slice (owner A6).
    d.w.deinit(d.gpa);
    // Iterate through streams and the idle pool calling drop
    var it = d.streams.iterator();
    while (it.next()) |kv| {
        drop(d, kv.value_ptr.*);
    }
    while (d.pool.pop()) |pooled| drop(d, pooled); // drop every idle context, not just the list
    d.pool.deinit(d.gpa);
    d.streams.deinit(d.gpa);

    // Deinitialize head weights if loaded
    if (d.head_loaded) {
        // Clean up head arena
        if (d.head_arena) |*a| a.deinit();
        d.head_arena = null;
        d.head_loaded = false;
    }
    if (d.dhw) |*dw| dw.deinit(d.gpa);

    // Free device buffers
    d.scratch_buf.free();
    d.xs_buf.free();
    d.pm_buf.free();
    d.pl_buf.free();
    d.po_buf.free();
    d.ids_buf.free();
    d.segments_buf.free();
    d.cand_ids_buf.free();
    d.cand_emb_buf.free();
    d.anchor_buf.free();
    d.token_states_buf.free();
    d.draft_picks_buf.free();
    if (d.ds.staged) d.ds.buf.free();
    if (d.open_stage == .ring or d.open_stage == .ready) d.tap_ring.free(); // owned from .ring on: undefined below it

    // Unload modules
    d.mod.unload();
    d.mod_q.unload();
}

pub fn streamBytesFor(c: Config, hidden: u32, tap_rows: u32) usize {
    // Base layout from stage 1
    const base_layout = c.ringBytes() + tap_rows * c.n_taps * hidden * 2 + 0;

    // Add new scratch fields as per stage 2 requirements
    const h_size = tap_rows * c.hidden * 2; // h
    const h2_size = tap_rows * c.hidden * 2; // h2
    const x_size = tap_rows * c.hidden * 2; // x
    const xin_size = tap_rows * c.hidden * 2; // xin
    const dyn_size = tap_rows * c.dynamic() * 2; // dyn
    const qraw_size = tap_rows * c.qWidth() * 2; // qraw
    const q_size = tap_rows * c.qWidth() * 2; // q
    const kraw_size = tap_rows * c.kvWidth() * 2; // kraw
    const kb_size = tap_rows * c.kvWidth() * 2; // kb
    const vb_size = tap_rows * c.kvWidth() * 2; // vb
    const att_size = tap_rows * c.heads * c.head_dim * 2; // att
    const o_size = tap_rows * c.hidden * 2; // o
    const gate_size = tap_rows * c.intermediate * 2; // gate
    const up_size = tap_rows * c.intermediate * 2; // up
    const act_size = tap_rows * c.intermediate * 2; // act
    const logits_size = tap_rows * c.vocab * 2; // logits
    const words_size = tap_rows * (3 + 2 * 20) * 4; // words (lp_words = 3 + 2 * 20 per row)
    const parts_size = tap_rows * c.topk * 2; // parts
    const segments_size = max_streams * @sizeOf(Segment); // frame.Segment per stream

    // Staged buffer (ids up, segments up, words down, anchor down) - correct layout
    const staged_size = tap_rows * c.block * 4 + max_streams * @sizeOf(Segment) + tap_rows * (3 + 2 * 20) * 4 + 4096 * 2; // anchor = 4096 * 2 bytes

    return base_layout + h_size + h2_size + x_size + xin_size + dyn_size +
        qraw_size + q_size + kraw_size + kb_size + vb_size + att_size + o_size +
        gate_size + up_size + act_size + logits_size + words_size + parts_size +
        segments_size + staged_size;
}

pub fn init(d: *Draft, gpa: std.mem.Allocator, io: std.Io, e: *E, c: Config, weights: Weights, dir: []const u8) !void {
    // gpa.create(Draft) leaves every field undefined: the maps' .empty defaults never apply, so
    // the streams' first open and any teardown read garbage until these run.
    d.streams = .empty;
    d.pool = .empty;
    d.head_loaded = false;
    d.head_arena = null;
    d.e = e;
    d.c = c;
    d.w = weights;
    d.gpa = gpa;
    // Store the directory for head loading
    d.dir = dir;

    // Staged ownership: each acquisition below registers its own errdefer, so a mid-init
    // failure unwinds exactly what init() acquired so far (deinit() no-ops while open_stage
    // is .none). The .core mark at the end flips the failure lane over to deinit().

    // The correlator head loads at open: its weights ride the same checkpoint the body loaded
    d.head_arena = std.heap.ArenaAllocator.init(gpa);
    errdefer d.head_arena.?.deinit();
    const hw_gpa = d.head_arena.?.allocator();
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer core.checkpoint.cache.close(&ck);
    d.hw = try head.load(hw_gpa, &ck);
    d.dhw = try head.loadDevice(gpa, e.ctx, &d.hw); // S1: staged, never read — the env-gated round-trip inside proves the bytes
    errdefer if (d.dhw) |*dw| dw.deinit(gpa);
    d.head_loaded = true;

    d.mod = try cuda.Module.load(e.ctx.d, cuda.kernels.qwen35_draft);
    errdefer d.mod.unload();
    d.mod_q = try cuda.Module.load(e.ctx.d, cuda.kernels.qwen35);
    errdefer d.mod_q.unload();

    d.fns.gather_taps = try d.mod.function("tf_q35d_gather_taps");
    d.fns.gather_rows = try d.mod.function("tf_q35d_gather_rows");
    d.fns.norm_rope = try d.mod.function("tf_q35d_norm_rope");
    d.fns.ring_put = try d.mod.function("tf_q35d_ring_put");
    d.fns.sums = try d.mod_q.function("tf_q35_sums");
    d.fns.norm_rope_batch = try d.mod.function("tf_q35d_norm_rope_batch");
    d.fns.attn_partial = try d.mod.function("tf_q35d_attn_partial");
    d.fns.attn_merge = try d.mod.function("tf_q35d_attn_merge");
    d.fns.mlp_act = try d.mod_q.function("tf_q35_mlp_act");
    d.fns.mask_rows = try d.mod.function("tf_q35d_mask_rows");
    d.fns.dyn_conv_batch = try d.mod.function("tf_q35d_dyn_conv_batch");
    d.fns.token_proj = try d.mod.function("tf_gemv_bf16_rows");
    d.fns.lat_attn = try d.mod.function("tf_lilicorr_lat_attn");
    d.fns.factor = try d.mod.function("tf_lilicorr_factor_rows");
    d.fns.edge_linear = try d.mod.function("tf_gemv_bf16_rows");
    d.fns.l2norm = try d.mod.function("tf_lilicorr_l2norm");
    d.fns.dots = try d.mod.function("tf_lilicorr_dots");
    d.fns.rms = try d.mod.function("tf_lilicorr_rms");
    d.fns.lin_f32 = try d.mod.function("tf_gemv_f32_rows");
    d.fns.silu_bf16 = try d.mod.function("tf_lilicorr_silu_bf16");
    d.fns.add_bf16 = try d.mod.function("tf_lilicorr_add_bf16");
    d.fns.lattice = try d.mod.function("tf_lilicorr_lattice");
    d.fns.rms_f32 = try d.mod.function("tf_lilicorr_rms_f32");
    d.fns.widen = try d.mod.function("tf_lilicorr_widen");
    d.fns.narrow = try d.mod.function("tf_lilicorr_narrow");

    // Initialize scratch buffers
    // The staged layout is cumulative: words_at/anchor_at/token_emb_at/pass_at each build on the
    // previous, so pass_at already covers ids + segments + words + the anchor. This adds the two
    // v1 head read-backs (the widen reads lattice x model_hidden bf16 from each area) and margin.
    const staged_bytes = pass_at + head.lattice * head.model_hidden * 2 + 4096 * 2 + scores_bytes;
    d.staged = try cuda.HostBuffer.alloc(e.ctx.d, staged_bytes);
    errdefer d.staged.free();

    // Allocate device buffers
    var b = try cuda.DeviceBuffer.alloc(e.ctx.d, 56 * 4); // 56 * 4 bytes
    d.cand_ids = b.ptr;
    d.cand_ids_buf = b;
    errdefer d.cand_ids_buf.free();

    b = try cuda.DeviceBuffer.alloc(e.ctx.d, 56 * d.c.hidden * 2); // 56 * hidden * 2 bytes
    d.cand_emb = b.ptr;
    d.cand_emb_buf = b;
    errdefer d.cand_emb_buf.free();

    b = try cuda.DeviceBuffer.alloc(e.ctx.d, d.c.hidden * 2); // hidden * 2 bytes
    d.anchor = b.ptr;
    d.anchor_buf = b;
    errdefer d.anchor_buf.free();

    // Allocate draft_picks buffer: staging_rows * 4 bytes (u32 per row)
    b = try cuda.DeviceBuffer.alloc(e.ctx.d, staging_rows * 4);
    d.draft_picks = b.ptr;
    d.draft_picks_buf = b;
    errdefer d.draft_picks_buf.free();
    // Zero-fill the buffer
    try e.ctx.d.check(e.ctx.d.api.cuMemsetD8Async(d.draft_picks, 0, staging_rows * 4, e.stream.handle), "cuMemsetD8Async draft_picks zero");

    // Allocate token_states buffer: [56 x hd] f32
    b = try cuda.DeviceBuffer.alloc(e.ctx.d, 56 * head.hd * 4); // 56 rows, hd columns, 4 bytes each (f32)
    d.token_states = b.ptr;
    d.token_states_buf = b;
    errdefer d.token_states_buf.free();

    // S5b: stage the device-chain scratch ONCE at open (L11 buffer DAG) when the device path
    // or the fixture A/Bs are armed; the default host path pays nothing.
    if (std.c.getenv("TF_LILICORR_DEVICE") != null or std.c.getenv("TF_LILICORR_FIXTURE") != null or std.c.getenv("TF_BODY_FIXTURE") != null) {
        try stageDeviceScratch(d);
    }

    // S2 A/B (env-gated): TF_LILICORR_FIXTURE=<dir> runs the device token projection against
    // the fixture's golden token_states at open (gate 4e-3); a breach fails the open. Unset = skip.
    if (std.c.getenv("TF_LILICORR_FIXTURE")) |fix_dir| {
        try deviceTokenStatesAB(gpa, io, d, std.mem.span(fix_dir));
        try deviceLatticeAttentionAB(gpa, io, d, std.mem.span(fix_dir));
        try deviceFactorEdgeAB(gpa, io, d, std.mem.span(fix_dir));
        try deviceLayerStackAB(gpa, io, d, std.mem.span(fix_dir));
        try deviceFullChainAB(gpa, io, d, std.mem.span(fix_dir));
    }
    // 16b: the body fixture gate runs AFTER the body scratch carve below (its buffers live
    // there); see deviceBodyFixtureAB.

    d.taps_in = 0;
    d.th = 0;
    d.thn = 0;

    // The absorb/forward scratch plane: every stage's rows at the staging cap, one device block,
    // sections in declaration order (all sizes are 256-byte multiples, so the carve needs no padding).
    const R: usize = staging_rows;
    const w_hid: usize = R * c.hidden * 2;
    const w_kv: usize = R * c.kvWidth() * 2;
    const w_q: usize = R * c.qWidth() * 2;
    const sizes = [_]usize{
        R * c.n_taps * c.hidden * 2, // taps_in: the gather's rows (n_taps x hidden bf16 a row)
        w_hid, // th: the fc's rows
        w_hid, // thn: hidden_norm's rows
        w_kv, // kabs
        w_kv, // vabs
        w_kv, // kn
        w_hid, // h: the block's embed rows
        w_hid, // h2
        w_hid, // x
        w_hid, // xin
        R * c.dynamic() * 2, // dyn: the convs' dynamic width
        w_q, // qraw
        w_q, // q
        w_kv, // kraw
        w_kv, // kb
        w_kv, // vb
        R * c.heads * c.head_dim * 2, // att: the merge's rows (head_dim a head)
        w_hid, // o
        R * c.intermediate * 2, // gate
        R * c.intermediate * 2, // up
        R * c.intermediate * 2, // act
        R * c.vocab * 2, // logits: the head's rows
        R * (3 + 2 * 20) * 4, // words: topkRows' words out (lp_words u32 a row)
    };
    var total: usize = 0;
    for (sizes) |z| total += z;
    d.scratch_buf = try cuda.DeviceBuffer.alloc(e.ctx.d, total);
    errdefer d.scratch_buf.free();
    var at: u64 = d.scratch_buf.ptr;
    inline for (fields: {
        break :fields &.{ "taps_in", "th", "thn", "kabs", "vabs", "kn", "h", "h2", "x", "xin", "dyn", "qraw", "q", "kraw", "kb", "vb", "att", "o", "gate", "up", "act", "logits", "words" };
    }, sizes) |name, z| {
        @field(d, name) = at;
        at += z;
    }

    d.step_ms = 1.0;
    // the projections' 64-group sums scratch (f32): the biggest projection's width is the fc's
    d.xs_buf = try cuda.DeviceBuffer.alloc(e.ctx.d, staging_rows * (24576 / 64) * 4);
    d.xs = d.xs_buf.ptr;
    errdefer d.xs_buf.free();
    d.ids_buf = try cuda.DeviceBuffer.alloc(e.ctx.d, staging_rows * c.block * 4);
    d.ids = d.ids_buf.ptr;
    errdefer d.ids_buf.free();
    d.segments_buf = try cuda.DeviceBuffer.alloc(e.ctx.d, max_streams * @sizeOf(Segment));
    d.segments = d.segments_buf.ptr;
    errdefer d.segments_buf.free();
    // the attention partials: per (row, head, chunk) floats, the output HD floats more
    const partials = staging_rows * c.heads * c.chunks();
    d.pm_buf = try cuda.DeviceBuffer.alloc(e.ctx.d, partials * 4);
    d.pm = d.pm_buf.ptr;
    errdefer d.pm_buf.free();
    d.pl_buf = try cuda.DeviceBuffer.alloc(e.ctx.d, partials * 4);
    d.pl = d.pl_buf.ptr;
    errdefer d.pl_buf.free();
    d.po_buf = try cuda.DeviceBuffer.alloc(e.ctx.d, partials * 128 * 4);
    d.po = d.po_buf.ptr;
    errdefer d.po_buf.free();
    // c.ringRows() is the authoritative ring-row count: setTaps is called with exactly it below
    // (it cannot change the value), and init runs first — e.tap_rows is still 0 here.
    d.tap_rows = @intCast(c.ringRows());
    // 16b: the body fixture gate (its buffers = the carve above; TF_BODY_FIXTURE=<fixture dir>)
    if (std.c.getenv("TF_BODY_FIXTURE")) |fix_dir| {
        try deviceBodyFixtureAB(gpa, io, d, std.mem.span(fix_dir));
    }
    d.open_stage = .core; // every init() acquisition is owned: later failures unwind through deinit()
}

/// S5b: the device-chain scratch, carved ONCE at open (L11 buffer DAG — every buffer with its
/// size, in carve order; hold()-time launches only). Also stages the two open-time tensors the
/// chain reads but never writes: the fused edge head (out_head ∥ in_head) and the attention bias.
/// Caller: init(), gated on TF_LILICORR_DEVICE or TF_LILICORR_FIXTURE.
pub fn stageDeviceScratch(d: *Draft) !void {
    const rows = head.lattice;
    const hd = head.hd;
    const factor = head.factor;
    const topk = head.topk;
    const slots = head.slots;
    if (d.dhw == null) return error.DeviceWeightsMissing;

    // The carve (bytes, 256-aligned each): pass_states 4096 | anchor_state 4096 | anchor_normed
    // 4096 | anchor_bits 2048 | fm_up 229376 | fm_up_bits 114688 | fm 229376 | lattice 114688 |
    // lattice2 114688 | ain 114688 | ain_f32 229376 | qkv 688128 | att 229376 | proj 229376 |
    // up 458752 | up_bf16 229376 | pre_bits 114688 | edges 458752+4096 | start 32 | pair 1536 |
    // fused_w 8388608 | fused_b 8192 | bias 100352 | feat_bits 640.
    const sizes = [_]usize{
        hd * 4, // pass_states
        hd * 4, // anchor_state
        hd * 4, // anchor_normed
        hd * 2, // anchor_bits
        rows * hd * 4, // fm_up
        rows * hd * 2, // fm_up_bits
        rows * hd * 4, // fm
        rows * hd * 2, // lattice
        rows * hd * 2, // lattice2
        rows * hd * 2, // ain
        rows * hd * 4, // ain_f32
        rows * 3 * hd * 4, // qkv
        rows * hd * 4, // att
        rows * hd * 4, // proj
        rows * 2 * hd * 4, // up
        rows * 2 * hd * 2, // up_bf16
        rows * hd * 2, // pre_bits
        rows * 2 * factor * 4 + hd * 4, // edges + anchor tail
        topk * 4, // start
        (slots - 1) * topk * topk * 4, // pair
        2 * factor * hd * 4, // fused_w
        2 * factor * 4, // fused_b
        head.heads * rows * rows * 4, // bias
        rows * 5 * 2, // feat_bits
    };
    var total: usize = 0;
    for (sizes) |z| total += (z + 255) / 256 * 256;
    d.ds.buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, total);
    errdefer d.ds.buf.free();
    var at: u64 = 0;
    inline for (fields: {
        break :fields &.{ "pass_states", "anchor_state", "anchor_normed", "anchor_bits", "fm_up", "fm_up_bits", "fm", "lattice", "lattice2", "ain", "ain_f32", "qkv", "att", "proj", "up", "up_bf16", "pre_bits", "edges", "start", "pair", "fused_w", "fused_b", "bias", "feat_bits" };
    }, sizes) |name, z| {
        @field(d.ds, name) = d.ds.buf.ptr + at;
        at += (z + 255) / 256 * 256;
    }
    d.ds.anchor_tail_at = @intCast(rows * 2 * factor * 4);
    d.ds.staged = true;

    // The fused edge head + the attention bias: open-time tensors, uploaded once here.
    const fused_w = try d.gpa.alloc(f32, 2 * factor * hd);
    defer d.gpa.free(fused_w);
    const fused_b = try d.gpa.alloc(f32, 2 * factor);
    defer d.gpa.free(fused_b);
    @memcpy(fused_w[0 .. factor * hd], d.hw.out_head.w);
    @memcpy(fused_w[factor * hd ..], d.hw.in_head.w);
    @memcpy(fused_b[0..factor], d.hw.out_head.b);
    @memcpy(fused_b[factor..], d.hw.in_head.b);
    try d.ds.buf.upload(@intCast(d.ds.fused_w - d.ds.buf.ptr), std.mem.sliceAsBytes(fused_w));
    try d.ds.buf.upload(@intCast(d.ds.fused_b - d.ds.buf.ptr), std.mem.sliceAsBytes(fused_b));

    const bias = try head.buildBias(d.gpa, &d.hw);
    defer d.gpa.free(bias);
    try d.ds.buf.upload(@intCast(d.ds.bias - d.ds.buf.ptr), std.mem.sliceAsBytes(bias));
}
