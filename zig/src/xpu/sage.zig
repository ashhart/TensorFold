//! Sage-A1.5 (qwen3_5_moe, 35B-A3B) on Intel Xe2 over Level Zero, first form: MLX affine 4/64 weights, decode kernels
//! plus prompt chunks (norms, GEMMs and MoE over the chunk), one sequence. It reuses the 27B's device layer, loaders and
//! kernels, built at Sage's shape (shape.h, sage_glue and sage_decode), plus the MoE ops of sage_moe.cl.
const std = @import("std");
const ze = @import("ze");
const kb = @import("xpu_qwen27_kernels");
const cfgm = @import("sage_config");
const sl = @import("sage_load");
const dev = @import("dev.zig");
const loader = @import("load.zig");
const fwd = @import("forward.zig");
const q = @import("qwen27.zig"); // the profiler state (XPU_PROFILE) the Dev records into
const Dev = dev.Dev;
const Checkpoint = dev.Checkpoint;
const Proj = loader.Proj;
const off = dev.off;
const alloc = std.heap.c_allocator;
extern "c" fn malloc_trim(pad: usize) c_int;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;

pub const EPS: f32 = 1e-6;
pub const SPLIT = 32; // keys per attention split (decode.cl)
/// Decode attention slices at most (attn_decode_span): a slice holds a multiple of SPAN_STEP keys.
const SLICES = 256;
/// Short caches get short slices too, so that enough sub-groups share the keys.
const SPAN_STEP = 8;

/// Keys a slice of attn_decode_span holds for `keys` keys: at most SLICES slices.
fn span(keys: u32) u32 {
    return @max(1, (keys + SLICES * SPAN_STEP - 1) / (SLICES * SPAN_STEP)) * SPAN_STEP;
}
/// Rows of one forward: the decode kernels' limit (the chain table and the serial GDN).
pub const WINDOW = 16;
/// Rows of one prompt chunk: norms, GEMMs and MoE take the whole chunk, the mixers WINDOW rows at a time.
pub const CHUNK = 512;
/// Streams one runner holds (a stepMulti row each): K/V caches, GDN and convolution states of their own.
pub const MAX_SLOTS = 8;

/// A prompt piece's images (promptFeed): its rows' M-RoPE positions (t, h, w), and tower rows over placeholders.
pub const Feed = struct { rope: ?[]const [3]i32 = null, rows: []const Rows = &.{} };
/// Rows [at, at + n) of a piece from `src`: fp32 [n, hidden] on the device (vision.Tower.encode's output).
pub const Rows = struct { at: u32, n: u32, src: *anyopaque };
/// Query heads an attn_decode subgroup holds at Sage's shape (sage_config.sage_defines: Q27_QPG).
const QPG = 4;

/// A layer's routed experts: gate|up rows of an expert stacked (2 ff rows), and down, MLX affine 4/64 as stored.
pub const Experts = struct { gu_w: *anyopaque, gu_s: *anyopaque, gu_b: *anyopaque, dn_w: *anyopaque, dn_s: *anyopaque, dn_b: *anyopaque };

pub const Layer = struct {
    in_norm: *anyopaque,
    post_norm: *anyopaque,
    attn: bool,
    mixer_in: Proj, // GDN: qkv | z | a | b ; attention: (q | gate) per head | k | v
    mixer_out: Proj,
    router: *anyopaque, // fp16 [experts, D]
    shared_gate: *anyopaque, // fp16 [D]
    experts: Experts,
    shared: Experts, // one expert, the same layout
    // GDN
    conv_w: ?*anyopaque = null,
    alog: ?*anyopaque = null,
    dt: ?*anyopaque = null,
    gnorm: ?*anyopaque = null,
    rec: ?*anyopaque = null, // fp32 [GV, 128, 128]
    rec_save: ?*anyopaque = null, // a verified block's state after all its rows (Runner.verify); rec keeps the entry
    stash: ?*anyopaque = null, // the block's gdn_pre outputs and mixer rows, for Runner.commit's replay (stashAt)
    conv: [2]?*anyopaque = .{ null, null }, // fp16 [3, NCONV], ping-pong
    conv_cur: u1 = 0,
    // attention
    qn: ?*anyopaque = null,
    kn: ?*anyopaque = null,
    kc: ?*anyopaque = null, // bf16 [ctx, AKV, 256]
    vc: ?*anyopaque = null,
    // every slot's: the fields above are the current slot's (Runner.use)
    st: [MAX_SLOTS]SlotState = undefined,
};

/// One slot's state in one layer.
pub const SlotState = struct { kc: ?*anyopaque = null, vc: ?*anyopaque = null, rec: ?*anyopaque = null, rec_save: ?*anyopaque = null, stash: ?*anyopaque = null, conv: [2]?*anyopaque = .{ null, null }, conv_cur: u1 = 0 };

/// Byte offsets in a GDN layer's stash of WINDOW rows: q at 0, then k, v, g, beta and the mixer input rows p.
pub const StashAt = struct {
    k: usize,
    v: usize,
    g: usize,
    b: usize,
    p: usize,
    end: usize,
    pub fn of(z: Sizes) StashAt {
        const qk = @as(usize, WINDOW) * z.s.gdn_k_heads * 128 * 2;
        const v = @as(usize, WINDOW) * z.vd * 2;
        const g = @as(usize, WINDOW) * z.s.gdn_v_heads * 4;
        const p = @as(usize, WINDOW) * z.gdn_in * 2;
        return .{ .k = qk, .v = 2 * qk, .g = 2 * qk + v, .b = 2 * qk + v + g, .p = 2 * qk + v + 2 * g, .end = 2 * qk + v + 2 * g + p };
    }
};

pub const Model = struct {
    embed_w: *anyopaque,
    embed_s: *anyopaque,
    embed_b: *anyopaque,
    layers: []Layer,
    norm: *anyopaque,
    head: Proj,
    invf: *anyopaque,
    embed_h: ?*anyopaque = null, // fp16 [vocab, D]: a bf16 embedding (AutoRound), read by embed_dense
    head_h: ?*anyopaque = null, // fp16 [vocab, D]: a bf16 lm_head (AutoRound), read by dense_rows
};

/// The sizes derived from the shape, as the kernels compiled with sage_defines see them.
pub const Sizes = struct {
    s: cfgm.Shape,
    nconv: u32,
    vd: u32,
    od: u32,
    gdn_in: u32,
    att_in: u32,

    pub fn of(s: cfgm.Shape) Sizes {
        const nconv = 2 * s.gdn_k_heads * 128 + s.gdn_v_heads * 128;
        const vd = s.gdn_v_heads * 128;
        return .{ .s = s, .nconv = nconv, .vd = vd, .od = s.q_heads * 256, .gdn_in = (nconv + vd + 2 * s.gdn_v_heads + 255) / 256 * 256, .att_in = s.attIn() };
    }
};

pub const Acts = struct {
    x: *anyopaque, // fp32 [CHUNK, D]
    h: *anyopaque, // fp16 [CHUNK, widest GEMM input]
    ahi: *anyopaque,
    alo: *anyopaque,
    r: *anyopaque, // fp16 [CHUNK, D]
    p: *anyopaque, // fp16 [CHUNK, widest mixer input projection]
    q: *anyopaque,
    k: *anyopaque,
    v: *anyopaque,
    g: *anyopaque,
    beta: *anyopaque,
    y: *anyopaque,
    pm: *anyopaque,
    pl: *anyopaque,
    po: *anyopaque,
    pscr: *anyopaque,
    logits: *anyopaque, // fp16 [WINDOW, VOCAB]
    tok: *anyopaque,
    pv: *anyopaque,
    pi: *anyopaque,
    ids: *anyopaque,
    chain: *anyopaque,
    // MoE
    mt: *anyopaque, // u64 [layers, WINDOW, 8]: stepMulti's per-row state table (the 27B's acts.mt)
    mpos: *anyopaque, // i32 [WINDOW]: stepMulti's row positions
    rope: *anyopaque, // i32 [CHUNK, 3]: a forward's M-RoPE positions (t, h, w) when its slots hold images
    mlog: *anyopaque, // fp32 [CHUNK, experts + 1]
    mids: *anyopaque, // u32 [CHUNK, top_k]
    mwts: *anyopaque, // fp32 [CHUNK, top_k]
    msw: *anyopaque, // fp32 [CHUNK]: the shared expert's gate
    mact: *anyopaque, // fp16 [CHUNK (top_k + 1), ff]
    my: *anyopaque, // fp32 [CHUNK (top_k + 1), D]
    // grouped prompt MoE
    morder: *anyopaque, // u32 [CHUNK (top_k + 1)]: pair indices by expert
    mtiles: *anyopaque, // u32x4 [tileBound(CHUNK)]
    mnt: *anyopaque, // u32: the tile count
    mplan: *anyopaque, // u32: sage_group_plan's groups of a decode step of up to WINDOW rows
    mxs: *anyopaque, // fp32 [CHUNK, D / 64]: group sums of h
    mas: *anyopaque, // fp32 [CHUNK (top_k + 1), ff / 64]: group sums of act

    pub fn init(d: *Dev, z: Sizes, ctx: u32) !Acts {
        const s = z.s;
        const w: usize = WINDOW;
        const c: usize = CHUNK;
        const widest_in: usize = @max(s.hidden, @max(z.vd, z.od));
        const ns: usize = @max((ctx + SPLIT - 1) / SPLIT, SLICES);
        const pairs: usize = c * (s.top_k + 1);
        return .{
            .x = try d.alloc_(c * s.hidden * 4),
            .h = try d.alloc_(c * widest_in * 2),
            .ahi = try d.alloc_(c * (widest_in / 64) * 2),
            .alo = try d.alloc_(c * (widest_in / 64) * 2),
            .r = try d.alloc_(c * s.hidden * 2),
            .p = try d.alloc_(c * @max(z.gdn_in, z.att_in) * 2),
            .q = try d.alloc_(c * @max(s.q_heads * 256, s.gdn_k_heads * 128) * 2),
            .k = try d.alloc_(w * s.gdn_k_heads * 128 * 2),
            .v = try d.alloc_(w * z.vd * 2),
            .g = try d.alloc_(w * s.gdn_v_heads * 4),
            .beta = try d.alloc_(w * s.gdn_v_heads * 4),
            .y = try d.alloc_(c * @max(z.vd, z.od) * 2),
            .pm = try d.alloc_(w * s.q_heads * ns * 4),
            .pl = try d.alloc_(w * s.q_heads * ns * 4),
            .po = try d.alloc_(w * s.q_heads * ns * 256 * 4),
            .pscr = try d.alloc_(c * s.hidden * 4),
            .logits = try d.alloc_(w * s.vocab * 2),
            .tok = try d.alloc_(w * 4),
            .pv = try d.alloc_(w * 1024),
            .pi = try d.alloc_(w * 1024),
            .ids = try d.alloc_(c * 4), // i32 [CHUNK]: a forward's ids
            .chain = try d.upload(std.mem.sliceAsBytes(&fwd.chainTable())),
            .mt = try d.alloc_(@as(usize, s.layers) * w * 8 * 8),
            .mpos = try d.alloc_(w * 4),
            .rope = try d.alloc_(c * 3 * 4),
            .mlog = try d.alloc_(c * (s.experts + 1) * 4),
            .mids = try d.alloc_(c * s.top_k * 4),
            .mwts = try d.alloc_(c * s.top_k * 4),
            .msw = try d.alloc_(c * 4),
            .mact = try d.alloc_(pairs * @max(s.expert_ff, s.shared_ff) * 2),
            .my = try d.alloc_(pairs * s.hidden * 4),
            .morder = try d.alloc_(pairs * 4),
            .mtiles = try d.alloc_(tileBound(s, CHUNK) * 16),
            .mnt = try d.alloc_(4),
            .mplan = try d.alloc_((1 + (w * s.top_k + 1) * (2 + 16)) * 4),
            .mxs = try d.alloc_(c * (s.hidden / 64) * 4),
            .mas = try d.alloc_(pairs * (@max(s.expert_ff, s.shared_ff) / 64) * 4),
        };
    }
};

/// sage_moe.cl's kernels and the long-cache attention pair of decode.cl (the Dev's table is the 27B runner's, filled from
/// the Sage images).
pub const MoeKernels = struct { attn_span: ze.H, merge_span: ze.H, down8: ze.H, swiglu4: ze.H, down_combine: ze.H, gdn_one: ze.H, router: ze.H, route: ze.H, swiglu: ze.H, down: ze.H, combine: ze.H, group_plan: ze.H, group_swiglu: ze.H, group_down: ze.H, router_tile: ze.H, sums: ze.H, sort: ze.H, tile_swiglu: ze.H, tile_down: ze.H };

/// Sub-groups a work-group of the decode expert kernels (one sub-group a row).
const MOE_SG = 4;
/// Work-items on one sage_expert_swiglu row (its K split over them).
const SWIGLU_WG = 32;
/// Output rows a sub-group of sage_down_combine (sage_moe.cl CROWS).
const DC_ROWS = 2;
/// Rows of a grouped MoE tile (sage_moe.cl MC).
const MC = 16;
/// The most tiles `rows` rows can make: full tiles plus at most one partial tile an expert (the shared one too).
fn tileBound(s: cfgm.Shape, rows: u32) u32 {
    return rows * (s.top_k + 1) / MC + s.experts + 1;
}

// ---------------------------------------------------------------- loading

fn name(comptime f: []const u8, args: anytype) []const u8 {
    return std.fmt.allocPrint(alloc, f, args) catch @panic("out of memory");
}

fn halfTensor(d: *Dev, c: *const Checkpoint, p: []const u8) !*anyopaque {
    const h = try sl.halfRows(alloc, c, p);
    defer alloc.free(h);
    return d.upload(std.mem.sliceAsBytes(h));
}

/// Device copies of one expert table: sage_load packs it, this sink copies each piece into its buffer.
const TableSink = struct {
    d: *Dev,
    x: Experts,
    pub fn put(s: *TableSink, part: sl.Part, at: usize, b: []const u8) !void {
        const dst = switch (part) {
            .gu_w => s.x.gu_w,
            .gu_s => s.x.gu_s,
            .gu_b => s.x.gu_b,
            .dn_w => s.x.dn_w,
            .dn_s => s.x.dn_s,
            .dn_b => s.x.dn_b,
        };
        try s.d.copyIn(off(dst, at), b);
    }
};

fn experts(d: *Dev, c: *const Checkpoint, p: []const u8, count: u32, shared: bool, ff: u32, D: u32) !Experts {
    const z = sl.Sizes.of(ff, D);
    const n: usize = count;
    var s: TableSink = .{ .d = d, .x = .{
        .gu_w = try d.alloc_(n * z.gu_w),
        .gu_s = try d.alloc_(n * z.gu_g),
        .gu_b = try d.alloc_(n * z.gu_g),
        .dn_w = try d.alloc_(n * z.dn_w),
        .dn_s = try d.alloc_(n * z.dn_g),
        .dn_b = try d.alloc_(n * z.dn_g),
    } };
    try sl.packExperts(c, p, count, shared, ff, D, &s);
    return s.x;
}

/// MLX affine 4/64 only in this form; the scales of the experts must be BF16 (sage_moe.cl reads bf16). Checked on the
/// Sage MLX-4bit checkpoint by tools/xpu/sage_mlx_check.py: stacked switch_mlp experts, 8 bit router, conv1d [NCONV, 4, 1].
/// kv8: an int8 K/V cache (256 int8 a position and head, then the fp32 scales after `ctx` positions), else bf16.
pub fn load(d: *Dev, c: *const Checkpoint, z: Sizes, ctx: u32, slots: u32, kv8: bool) !Model {
    const s = z.s;
    if (c.has("in_proj_qkv.qweight") or c.has("in_proj_qkv.trellis")) return error.OnlyMlxInThisForm;
    const stage = try d.alloc_(800 << 20);
    var m: Model = undefined;
    m.embed_h = null;
    m.head_h = null;
    if (std.mem.eql(u8, (try c.get("embed_tokens.weight")).dtype, "BF16")) {
        m.embed_h = try halfTensor(d, c, "embed_tokens");
        m.embed_w = m.embed_h.?;
        m.embed_s = m.embed_h.?;
        m.embed_b = m.embed_h.?;
    } else {
        m.embed_w = try d.upload((try c.get("embed_tokens.weight")).bytes);
        m.embed_s = try loader.bf16Tensor(d, c, "embed_tokens.scales");
        m.embed_b = try loader.bf16Tensor(d, c, "embed_tokens.biases");
    }
    m.norm = try loader.bf16Tensor(d, c, "model.norm.weight");
    if (std.mem.eql(u8, (try c.get("lm_head.weight")).dtype, "BF16")) {
        if (s.vocab % 16 != 0) return error.UnexpectedShape;
        m.head_h = try halfTensor(d, c, "lm_head");
        m.head = .{ .w = m.head_h.?, .s = m.head_h.?, .bg = m.head_h.?, .n = s.vocab, .k = s.hidden, .wide = true };
    } else m.head = try loader.projection(d, c, "lm_head", &.{""}, true, false, stage);
    var invf: [32]f32 = undefined;
    const half_dims: f64 = @floatFromInt(s.rope_dims / 2);
    for (&invf, 0..) |*f, i| f.* = @floatCast(std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(i)) / half_dims));
    m.invf = try d.upload(std.mem.sliceAsBytes(invf[0 .. s.rope_dims / 2]));
    m.layers = try alloc.alloc(Layer, s.layers);
    const kv_bytes = @as(usize, ctx) * s.kv_heads * @as(usize, if (kv8) 256 + 4 else 256 * 2);
    for (m.layers, 0..) |*l, i| {
        const p = name("model.layers.{d}.", .{i});
        const attn = (i + 1) % 4 == 0;
        l.* = .{
            .in_norm = try loader.bf16Tensor(d, c, name("{s}input_layernorm.weight", .{p})),
            .post_norm = try loader.bf16Tensor(d, c, name("{s}post_attention_layernorm.weight", .{p})),
            .attn = attn,
            .mixer_in = undefined,
            .mixer_out = undefined,
            .router = try halfTensor(d, c, name("{s}mlp.gate", .{p})),
            .shared_gate = try halfTensor(d, c, name("{s}mlp.shared_expert_gate", .{p})),
            .experts = try experts(d, c, p, s.experts, false, s.expert_ff, s.hidden),
            .shared = try experts(d, c, p, 1, true, s.shared_ff, s.hidden),
        };
        if (attn) {
            l.mixer_in = try loader.projection(d, c, name("{s}self_attn.", .{p}), &.{ "q_proj", "k_proj", "v_proj" }, true, false, stage);
            l.mixer_out = try loader.projection(d, c, name("{s}self_attn.", .{p}), &.{"o_proj"}, false, false, stage);
            if (l.mixer_in.n != z.att_in) return error.UnexpectedShape;
            l.qn = try loader.bf16Tensor(d, c, name("{s}self_attn.q_norm.weight", .{p}));
            l.kn = try loader.bf16Tensor(d, c, name("{s}self_attn.k_norm.weight", .{p}));
            for (l.st[0..slots]) |*st| st.* = .{ .kc = try d.alloc_(kv_bytes), .vc = try d.alloc_(kv_bytes) };
            l.kc = l.st[0].kc;
            l.vc = l.st[0].vc;
        } else {
            l.mixer_in = try loader.projection(d, c, name("{s}linear_attn.", .{p}), &.{ "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b" }, true, false, stage);
            l.mixer_out = try loader.projection(d, c, name("{s}linear_attn.", .{p}), &.{"out_proj"}, false, false, stage);
            if (l.mixer_in.n != z.gdn_in) return error.UnexpectedShape;
            l.conv_w = try loader.bf16Tensor(d, c, name("{s}linear_attn.conv1d.weight", .{p}));
            l.alog = try loader.bf16Tensor(d, c, name("{s}linear_attn.A_log", .{p}));
            l.dt = try loader.bf16Tensor(d, c, name("{s}linear_attn.dt_bias", .{p}));
            l.gnorm = try loader.bf16Tensor(d, c, name("{s}linear_attn.norm.weight", .{p}));
            const rec_bytes = @as(usize, s.gdn_v_heads) * 128 * 128 * 4;
            for (l.st[0..slots]) |*st| {
                st.* = .{ .rec = try d.alloc_(rec_bytes), .rec_save = try d.alloc_(rec_bytes), .stash = try d.alloc_(StashAt.of(z).end) };
                for (&st.conv) |*cv| cv.* = try d.alloc_(3 * @as(usize, z.nconv) * 2);
            }
            l.rec = l.st[0].rec;
            l.rec_save = l.st[0].rec_save;
            l.stash = l.st[0].stash;
            l.conv = l.st[0].conv;
        }
        if (i % 8 == 7) std.debug.print("  layers loaded: {d}/{d}\n", .{ i + 1, s.layers });
    }
    try d.sync();
    d.g.free(stage);
    d.allocated -= 800 << 20;
    return m;
}

// ---------------------------------------------------------------- the MTP head

/// Sage's multi-token prediction head (mtp_num_hidden_layers 1), the drafter of Runner.specStep. A row at slot j reads
/// the target's normed final hidden of row j and the token after it, and drafts the token after that:
/// x = fc_e(norm_e(embed(token))) + fc_h(norm_h(hidden)), then one attention + MoE layer with its own K/V cache, then
/// norm and the target's lm_head. A chained draft reads the row before's normed output as its hidden.
pub const Mtp = struct {
    layer: Layer,
    fc_e: Proj, // fc [D, 2 D] split by input: the embedding half and the hidden half
    fc_h: Proj,
    norm_e: *anyopaque,
    norm_h: *anyopaque,
    norm: *anyopaque,
    hin: *anyopaque, // fp16 [CHUNK, D]: the rows' hidden inputs
    x2: *anyopaque, // fp32 [CHUNK, D]
    r2: *anyopaque, // fp16 [CHUNK, D]
    zeros: *anyopaque, // fp32 [CHUNK, D] of 0
    state: *anyopaque, // fp16 [MAX_SLOTS, D]: each slot's newest row's normed output
};

/// The MTP head from the W4A16 AutoRound checkpoint in `dir` (the MLX export drops it): its projections and experts
/// shown as MLX affine 4/64 with no requantization, fc quantized to 4/64 from bf16.
pub fn loadMtp(d: *Dev, dir: []const u8, z: Sizes, ctx: u32, slots: u32, kv8: bool) !*Mtp {
    const s = z.s;
    const shard = try mtpShard(dir);
    var c: Checkpoint = .{};
    defer c.deinit();
    try c.addShard(shard);
    _ = try showAsMlx(&c, .mtp);
    { // fc [D, 2 D] bf16: each half of its inputs as its own 4/64 projection
        const fc = try c.get("mtp.fc.weight");
        if (fc.rank != 2 or fc.shape[0] != s.hidden or fc.shape[1] != 2 * s.hidden) return error.UnexpectedShape;
        const D: usize = s.hidden;
        for ([_][]const u8{ "fc_e", "fc_h" }, 0..) |half, hi| {
            const rows = try alloc.alloc(u8, D * D * 2);
            defer alloc.free(rows);
            for (0..D) |i| @memcpy(rows[i * D * 2 ..][0 .. D * 2], fc.bytes[(i * 2 * D + hi * D) * 2 ..][0 .. D * 2]);
            const w = try alloc.alloc(u32, D * D / 8);
            const sc = try alloc.alloc(u16, D * D / 64);
            const bi = try alloc.alloc(u16, D * D / 64);
            try sl.quantizeRows(rows, D, D, w, sc, bi);
            try c.put(name("mtp.{s}.weight", .{half}), "U32", &.{ D, D / 8 }, std.mem.sliceAsBytes(w));
            try c.put(name("mtp.{s}.scales", .{half}), "BF16", &.{ D, D / 64 }, std.mem.sliceAsBytes(sc));
            try c.put(name("mtp.{s}.biases", .{half}), "BF16", &.{ D, D / 64 }, std.mem.sliceAsBytes(bi));
        }
    }
    const stage = try d.alloc_(64 << 20);
    defer {
        d.g.free(stage);
        d.allocated -= 64 << 20;
    }
    const p = "mtp.layers.0.";
    const mt = try alloc.create(Mtp);
    mt.layer = .{
        .in_norm = try loader.bf16Tensor(d, &c, p ++ "input_layernorm.weight"),
        .post_norm = try loader.bf16Tensor(d, &c, p ++ "post_attention_layernorm.weight"),
        .attn = true,
        .mixer_in = try loader.projection(d, &c, p ++ "self_attn.", &.{ "q_proj", "k_proj", "v_proj" }, true, false, stage),
        .mixer_out = try loader.projection(d, &c, p ++ "self_attn.", &.{"o_proj"}, false, false, stage),
        .router = try halfTensor(d, &c, p ++ "mlp.gate"),
        .shared_gate = try halfTensor(d, &c, p ++ "mlp.shared_expert_gate"),
        .experts = try experts(d, &c, p, s.experts, false, s.expert_ff, s.hidden),
        .shared = try experts(d, &c, p, 1, true, s.shared_ff, s.hidden),
        .qn = try loader.bf16Tensor(d, &c, p ++ "self_attn.q_norm.weight"),
        .kn = try loader.bf16Tensor(d, &c, p ++ "self_attn.k_norm.weight"),
    };
    if (mt.layer.mixer_in.n != z.att_in) return error.UnexpectedShape;
    const kv_bytes = @as(usize, ctx) * s.kv_heads * @as(usize, if (kv8) 256 + 4 else 256 * 2);
    for (mt.layer.st[0..slots]) |*st| st.* = .{ .kc = try d.alloc_(kv_bytes), .vc = try d.alloc_(kv_bytes) };
    mt.layer.kc = mt.layer.st[0].kc;
    mt.layer.vc = mt.layer.st[0].vc;
    mt.fc_e = try loader.projection(d, &c, "mtp.", &.{"fc_e"}, false, false, stage);
    mt.fc_h = try loader.projection(d, &c, "mtp.", &.{"fc_h"}, false, false, stage);
    mt.norm_e = try loader.bf16Tensor(d, &c, "mtp.pre_fc_norm_embedding.weight");
    mt.norm_h = try loader.bf16Tensor(d, &c, "mtp.pre_fc_norm_hidden.weight");
    mt.norm = try loader.bf16Tensor(d, &c, "mtp.norm.weight");
    const rows: usize = @as(usize, CHUNK) * s.hidden;
    mt.hin = try d.alloc_(rows * 2);
    mt.x2 = try d.alloc_(rows * 4);
    mt.r2 = try d.alloc_(rows * 2);
    mt.zeros = try d.alloc_(rows * 4);
    try d.fill(mt.zeros, @as(u32, 0), rows * 4);
    mt.state = try d.alloc_(@as(usize, MAX_SLOTS) * s.hidden * 2);
    try d.sync();
    std.debug.print("  MTP head loaded from {s}\n", .{shard});
    return mt;
}

/// The AutoRound shard that holds mtp.fc.weight, from the checkpoint's index.
fn mtpShard(dir: []const u8) ![:0]const u8 {
    const bytes = try readFile(name("{s}/model.safetensors.index.json", .{dir}));
    defer alloc.free(bytes);
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, bytes, .{}) catch return error.BadIndex;
    defer parsed.deinit();
    const map = if (parsed.value == .object) parsed.value.object.get("weight_map") else null;
    if (map == null or map.? != .object) return error.BadIndex;
    const f = map.?.object.get("mtp.fc.weight") orelse return error.NoMtpHead;
    if (f != .string) return error.BadIndex;
    return std.fmt.allocPrintSentinel(alloc, "{s}/{s}", .{ dir, f.string }, 0);
}

// ---------------------------------------------------------------- the runner

pub const Runner = struct {
    d: Dev,
    mk: MoeKernels,
    z: Sizes,
    model: Model,
    acts: Acts,
    ckpt: Checkpoint,
    ctx: u32,
    cap: i32 = 0, // > 0: the int8 K/V cache (XPU_KV_INT8), its positions
    pos: u32 = 0,
    last: i32 = -1, // the greedy token after the last fed row
    slots: u32 = 1,
    cur: u32 = 0, // the slot pos, last and the layers' current fields are
    spos: [MAX_SLOTS]u32 = @splat(0),
    slast: [MAX_SLOTS]i32 = @splat(-1),
    mt_host: []u64 = &.{},
    shift: [MAX_SLOTS]i32 = @splat(0), // M-RoPE: a row past a slot's images turns at its index plus this
    rope_host: [][3]i32 = &.{}, // a forward's rows' positions on their way to acts.rope
    mtp: ?*Mtp = null, // XPU_MTP: the drafter (an AutoRound checkpoint directory)
    sdraft: [MAX_SLOTS]i32 = @splat(-1), // each slot's first draft, from its newest absorbed row
    mpos: [MAX_SLOTS]u32 = @splat(0), // each slot's next MTP slot to absorb
    verifying: bool = false, // forward runs a block for verify: every row's token, states kept for commit
    block: [WINDOW]i32 = undefined, // the last verified block's greedy tokens, a row each
    block_rows: u32 = 0,

    /// `slots` streams (1 to MAX_SLOTS) of `ctx` positions each.
    pub fn init(dir: []const u8, ctx: u32, slots: u32) !*Runner {
        if (slots == 0 or slots > MAX_SLOTS) return error.BadSlots;
        const z = Sizes.of(try readShape(dir));
        var ckpt: Checkpoint = .{};
        errdefer ckpt.deinit();
        try addShards(&ckpt, dir);
        const gptq = try autoRound(&ckpt);
        const r = try alloc.create(Runner);
        errdefer alloc.destroy(r);
        r.* = .{ .d = .{ .g = try ze.Gpu.open(0xe223), .list = undefined, .k = undefined }, .mk = undefined, .z = z, .model = undefined, .acts = undefined, .ckpt = ckpt, .ctx = ctx, .slots = slots };
        const d = &r.d;
        errdefer d.g.close();
        d.list = try d.g.immediate(0, 0);
        try kernels(d, &r.mk);
        if (std.c.getenv("XPU_PROFILE") != null) {
            q.prof.events = try d.g.events(4096);
            inline for (@typeInfo(@TypeOf(d.k)).@"struct".field_names, 0..) |n, i| q.prof.names[i] = .{ .k = @field(d.k, n), .name = n };
            const base = @typeInfo(@TypeOf(d.k)).@"struct".field_names.len;
            inline for (@typeInfo(MoeKernels).@"struct".field_names, 0..) |n, i| q.prof.names[base + i] = .{ .k = @field(r.mk, n), .name = "moe_" ++ n };
        }
        if (std.c.getenv("XPU_KV_INT8") != null) r.cap = @intCast(ctx);
        r.model = try load(d, &r.ckpt, z, ctx, slots, r.cap > 0);
        if (std.c.getenv("XPU_MTP")) |md| r.mtp = try loadMtp(d, std.mem.span(md), z, ctx, slots, r.cap > 0);
        if (gptq) {
            // the converted host copies are on the device now; the conversion threads freed them into their own malloc
            // arenas, which keep the pages until a trim gives them back
            r.ckpt.dropPut();
            _ = malloc_trim(0);
        }
        r.mt_host = try alloc.alloc(u64, z.s.layers * WINDOW * 8);
        r.rope_host = try alloc.alloc([3]i32, CHUNK);
        q.prof.on = std.c.getenv("XPU_PROFILE") != null; // not during the load
        r.acts = try Acts.init(d, z, ctx);
        for (0..slots) |i| {
            try r.use(@intCast(i));
            try r.reset();
        }
        try r.use(0);
        return r;
    }

    pub fn deinit(r: *Runner) void {
        r.d.deinit();
        r.ckpt.deinit();
        alloc.free(r.model.layers);
        alloc.free(r.mt_host);
        alloc.free(r.rope_host);
        alloc.destroy(r);
    }

    /// Makes `slot` current: prompt, step, reset and logits act on it.
    pub fn use(r: *Runner, slot: u32) !void {
        if (slot >= r.slots) return error.BadSlot;
        r.spos[r.cur] = r.pos;
        r.slast[r.cur] = r.last;
        for (r.model.layers) |*l| {
            l.st[r.cur].conv_cur = l.conv_cur;
            const st = l.st[slot];
            if (l.attn) {
                l.kc = st.kc;
                l.vc = st.vc;
            } else {
                l.rec = st.rec;
                l.rec_save = st.rec_save;
                l.stash = st.stash;
                l.conv = st.conv;
                l.conv_cur = st.conv_cur;
            }
        }
        if (r.mtp) |mt| {
            mt.layer.kc = mt.layer.st[slot].kc;
            mt.layer.vc = mt.layer.st[slot].vc;
        }
        r.cur = slot;
        r.pos = r.spos[slot];
        r.last = r.slast[slot];
    }

    /// One decode token for each slot in `slots` (distinct, at most WINDOW), fed `toks`: one forward of a row a slot.
    /// Each slot's greedy next token goes to `out`; logits row i (rowLogits) is slot i's.
    pub fn stepMulti(r: *Runner, slots: []const u32, toks: []const i32, out: []i32) !void {
        const T: u32 = @intCast(slots.len);
        if (T == 0 or T > WINDOW or toks.len != T or out.len != T) return error.BadWindow;
        try r.use(r.cur); // store the current slot's fields
        const d = &r.d;
        const a = &r.acts;
        const m = &r.model;
        const z = r.z;
        const s = z.s;
        var pos: [WINDOW]i32 = undefined;
        var max_pos: u32 = 0;
        for (slots, 0..) |slot, t| {
            if (slot >= r.slots) return error.BadSlot;
            for (slots[0..t]) |o| if (o == slot) return error.SlotTwice;
            if (r.spos[slot] + 1 > r.ctx) return error.ContextFull;
            pos[t] = @intCast(r.spos[slot]);
            max_pos = @max(max_pos, r.spos[slot]);
        }
        for (m.layers, 0..) |*l, i| for (slots, 0..) |slot, t| {
            const e = r.mt_host[(i * WINDOW + t) * 8 ..][0..8];
            e.* = @splat(0);
            e[6] = t;
            e[7] = 1;
            const st = l.st[slot];
            if (l.attn) {
                e[0] = @intFromPtr(st.kc.?);
                e[1] = @intFromPtr(st.vc.?);
            } else {
                e[2] = @intFromPtr(st.conv[st.conv_cur].?);
                e[3] = @intFromPtr(st.conv[1 - st.conv_cur].?);
                e[4] = @intFromPtr(st.rec.?);
                e[5] = @intFromPtr(st.rec.?);
            }
        };
        const D: i32 = @intCast(s.hidden);
        const gs: i32 = 64;
        const t32: i32 = @intCast(T);
        const nul: ?*anyopaque = null;
        const zero: i32 = 0;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, a.mt, r.mt_host.ptr, r.mt_host.len * 8, null, 0, null));
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, a.mpos, &pos, T * 4, null, 0, null));
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, a.ids, toks.ptr, T * 4, null, 0, null));
        errdefer d.settle();
        var shifted = false;
        for (slots, 0..) |slot, t| {
            r.rope_host[t] = @splat(pos[t] + r.shift[slot]);
            shifted = shifted or r.shift[slot] != 0;
        }
        const rope = if (shifted) try r.ropeRows(T) else null;
        try r.embed(T);
        const sp: i32 = @intCast(span(max_pos + 1));
        const ns: u32 = (max_pos + 1 + @as(u32, @intCast(sp)) - 1) / @as(u32, @intCast(sp));
        for (m.layers, 0..) |*l, i| {
            const mt = off(a.mt, i * WINDOW * 8 * 8);
            try d.launch(d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, l.in_norm, a.h, a.ahi, a.alo, D, EPS, @as(i32, if (i == 0) 0 else 1), gs, nul, nul, zero, zero });
            try fwd.gemm(d, a.pscr, a.h, a.ahi, a.alo, l.mixer_in, a.p, T);
            if (l.attn) {
                try d.launch(d.k.attn_prep, .{ .x = T }, .{ a.p, l.qn.?, l.kn.?, m.invf, a.q, l.kc.?, l.vc.?, zero, EPS, nul, mt, a.mpos, zero, r.cap, rope });
                try d.launch(r.mk.attn_span, .{ .x = ns, .y = s.kv_heads * (s.q_heads / s.kv_heads / QPG), .z = T }, .{ a.q, l.kc.?, l.vc.?, a.pm, a.pl, a.po, zero, @as(i32, @intCast(ns)), sp, @as(f32, 1.0 / 16.0), mt, a.mpos, r.cap });
                try d.launch(r.mk.merge_span, .{ .x = s.q_heads, .y = T }, .{ a.pm, a.pl, a.po, a.y, zero, @as(i32, @intCast(ns)), sp, a.mpos });
                try d.launch(d.k.attn_gate, .{ .x = T }, .{ a.y, a.p, a.h, a.ahi, a.alo, gs, zero, nul, nul, zero });
            } else {
                try d.launch(d.k.gdn_pre, .{ .x = T, .y = z.nconv / 256 }, .{ a.p, l.conv[0].?, l.conv[1].?, l.conv_w.?, l.alog.?, l.dt.?, a.q, a.k, a.v, a.g, a.beta, t32, nul, mt });
                try d.launch(d.k.gdn_step, .{ .x = 16, .y = s.gdn_v_heads, .z = T }, .{ a.q, a.k, a.v, a.g, a.beta, l.rec.?, l.rec.?, a.y, t32, nul, nul, nul, mt });
                try d.launch(d.k.gated_norm, .{ .x = T }, .{ a.y, a.p, l.gnorm.?, a.h, a.ahi, a.alo, EPS, gs, zero, nul, nul, zero });
            }
            try fwd.gemm(d, a.pscr, a.h, a.ahi, a.alo, l.mixer_out, a.r, T);
            try d.launch(d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, l.post_norm, a.h, a.ahi, a.alo, D, EPS, @as(i32, 1), gs, nul, nul, zero, zero });
            try r.moe(l, T);
        }
        const hgs: i32 = @intCast(m.head.gs);
        try d.launch(d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, m.norm, a.h, a.ahi, a.alo, D, EPS, @as(i32, 1), hgs, nul, nul, zero, zero });
        try r.headRows(a.h, a.ahi, a.alo, T);
        try d.launch(d.k.argmax_part, .{ .x = 256, .y = T }, .{ a.logits, @as(i32, @intCast(s.vocab)), a.pv, a.pi });
        try d.launch(d.k.argmax, .{ .x = T }, .{ a.pv, a.pi, a.tok });
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, out.ptr, a.tok, T * 4, null, 0, null));
        try d.sync();
        for (m.layers) |*l| if (!l.attn) for (slots) |slot| {
            l.st[slot].conv_cur = 1 - l.st[slot].conv_cur;
        };
        for (slots, out) |slot, o| {
            r.spos[slot] += 1;
            r.slast[slot] = o;
        }
        // the current slot's fields again (its conv_cur and position may have moved)
        for (m.layers) |*l| if (!l.attn) {
            l.conv_cur = l.st[r.cur].conv_cur;
        };
        r.pos = r.spos[r.cur];
        r.last = r.slast[r.cur];
    }

    /// The fp16 logits of row `row` of the last stepMulti.
    pub fn rowLogits(r: *Runner, row: u32, out: []f16) !void {
        if (out.len != r.z.s.vocab) return error.BadLength;
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, out.ptr, off(r.acts.logits, @as(usize, row) * out.len * 2), out.len * 2, null, 0, null));
        try r.d.sync();
    }

    /// A fresh sequence: zero GDN and convolution states, position 0 (the K/V rows are overwritten as they are fed).
    pub fn reset(r: *Runner) !void {
        const s = r.z.s;
        for (r.model.layers) |*l| if (!l.attn) {
            try r.d.fill(l.rec.?, @as(u32, 0), @as(usize, s.gdn_v_heads) * 128 * 128 * 4);
            for (l.conv) |cv| try r.d.fill(cv.?, @as(u16, 0), 3 * @as(usize, r.z.nconv) * 2);
            l.conv_cur = 0;
        };
        r.pos = 0;
        r.last = -1;
        r.shift[r.cur] = 0;
        r.mpos[r.cur] = 0;
        r.sdraft[r.cur] = -1;
    }

    /// Feeds tokens in chunks of CHUNK rows (a tail of WINDOW rows or less as one decode forward); `last` afterwards
    /// is the greedy token after the final one.
    pub fn prompt(r: *Runner, ids: []const i32) !void {
        var at: usize = 0;
        while (at < ids.len) {
            const n = @min(CHUNK, ids.len - at);
            const tail = at + n == ids.len;
            const p0 = r.pos;
            r.last = if (n > WINDOW) try r.chunk(ids[at..][0..n], tail, .{}) else try r.forward(ids[at..][0..n], tail, .{});
            if (r.mtp != null) { // the MTP reads each row with the token after it: the next prompt token or the greedy one
                var follow: [CHUNK]i32 = undefined;
                @memcpy(follow[0 .. n - 1], ids[at + 1 ..][0 .. n - 1]);
                follow[n - 1] = if (tail) r.last else ids[at + n];
                try r.absorb(follow[0..n], p0);
            }
            at += n;
        }
    }

    /// One prompt piece of at most CHUNK rows with images: `f.rope` holds every row's M-RoPE positions and `f.rows`
    /// the tower rows that replace placeholder embeddings. `last` afterwards is the greedy token after the piece.
    pub fn promptFeed(r: *Runner, ids: []const i32, f: Feed) !void {
        if (ids.len > CHUNK or (f.rope != null and f.rope.?.len != ids.len)) return error.BadWindow;
        const p0 = r.pos;
        r.last = if (ids.len > WINDOW) try r.chunk(ids, true, f) else try r.forward(ids, true, f);
        if (r.mtp != null) { // as in prompt; a piece's last row reads the greedy token, the next piece's first id is not known here
            var follow: [CHUNK]i32 = undefined;
            @memcpy(follow[0 .. ids.len - 1], ids[1..]);
            follow[ids.len - 1] = r.last;
            try r.absorb(follow[0..ids.len], p0);
        }
    }

    /// The MTP over the last forward's rows (their normed final hidden in acts.h) at slots p0..: `follow` holds the
    /// token after each row. The slot's first draft becomes the last row's.
    fn absorb(r: *Runner, follow: []const i32, p0: u32) !void {
        const mt = r.mtp.?;
        try dev.copy(&r.d, mt.hin, r.acts.h, follow.len * r.z.s.hidden * 2);
        r.sdraft[r.cur] = try r.mtpForward(follow, p0);
        r.mpos[r.cur] = p0 + @as(u32, @intCast(follow.len));
    }

    /// The MTP layer over rows `ids` (with mtp.hin their hidden) at slots p0..; the last row's normed output goes to
    /// the slot's state, and its greedy draft is returned.
    fn mtpForward(r: *Runner, ids: []const i32, p0: u32) !i32 {
        const mt = r.mtp.?;
        const l = &mt.layer;
        const d = &r.d;
        const a = &r.acts;
        const m = &r.model;
        const s = r.z.s;
        const T: u32 = @intCast(ids.len);
        if (T == 0 or T > CHUNK) return error.BadWindow;
        if (p0 + T > r.ctx) return error.ContextFull;
        const D: i32 = @intCast(s.hidden);
        const gs: i32 = 64;
        const nul: ?*anyopaque = null;
        const zero: i32 = 0;
        const one: i32 = 1;
        const p32: i32 = @intCast(p0);
        const xb = T * s.hidden * 4;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, a.ids, ids.ptr, ids.len * 4, null, 0, null));
        errdefer d.settle();
        try d.launch(d.k.embed, .{ .x = T }, .{ a.ids, m.embed_w, m.embed_s, m.embed_b, a.x, D });
        try d.launch(d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, mt.norm_e, a.h, a.ahi, a.alo, D, EPS, zero, gs, nul, nul, zero, zero });
        try fwd.gemm(d, a.pscr, a.h, a.ahi, a.alo, mt.fc_e, mt.r2, T);
        try dev.copy(d, mt.x2, mt.zeros, xb);
        try d.launch(d.k.add_rmsnorm_dec, .{ .x = T }, .{ mt.x2, mt.hin, mt.norm_h, a.h, a.ahi, a.alo, D, EPS, one, gs, nul, nul, zero, zero });
        try fwd.gemm(d, a.pscr, a.h, a.ahi, a.alo, mt.fc_h, a.r, T);
        // the residual: fc_e(en) + fc_h(hn), in fp32 as the CUDA head adds them
        try dev.copy(d, a.x, mt.zeros, xb);
        try d.launch(d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, mt.r2, l.in_norm, a.h, a.ahi, a.alo, D, EPS, one, gs, nul, nul, zero, zero });
        try d.launch(d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, l.in_norm, a.h, a.ahi, a.alo, D, EPS, one, gs, nul, nul, zero, zero });
        try fwd.gemm(d, a.pscr, a.h, a.ahi, a.alo, l.mixer_in, a.p, T);
        try d.launch(d.k.attn_prep, .{ .x = T }, .{ a.p, l.qn.?, l.kn.?, m.invf, a.q, l.kc.?, l.vc.?, p32, EPS, nul, nul, nul, zero, r.cap, nul });
        if (T <= WINDOW) {
            const sp = span(p0 + T);
            const ns: u32 = (p0 + T + sp - 1) / sp;
            try d.launch(r.mk.attn_span, .{ .x = ns, .y = s.kv_heads * (s.q_heads / s.kv_heads / QPG), .z = T }, .{ a.q, l.kc.?, l.vc.?, a.pm, a.pl, a.po, p32, @as(i32, @intCast(ns)), @as(i32, @intCast(sp)), @as(f32, 1.0 / 16.0), nul, nul, r.cap });
            try d.launch(r.mk.merge_span, .{ .x = s.q_heads, .y = T }, .{ a.pm, a.pl, a.po, a.y, p32, @as(i32, @intCast(ns)), @as(i32, @intCast(sp)), nul });
        } else if (r.cap > 0)
            try d.launch(d.k.attn_prompt8, .{ .x = (T + 7) / 8, .y = s.kv_heads }, .{ a.q, l.kc.?, l.vc.?, a.y, p32, @as(i32, @intCast(T)), @as(f32, 1.0 / 16.0), r.cap })
        else
            try d.launch(d.k.attn_prompt, .{ .x = (T + 7) / 8, .y = s.kv_heads }, .{ a.q, l.kc.?, l.vc.?, a.y, p32, @as(i32, @intCast(T)), @as(f32, 1.0 / 16.0) });
        try d.launch(d.k.attn_gate, .{ .x = T }, .{ a.y, a.p, a.h, a.ahi, a.alo, gs, zero, nul, nul, zero });
        try fwd.gemm(d, a.pscr, a.h, a.ahi, a.alo, l.mixer_out, a.r, T);
        try d.launch(d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, l.post_norm, a.h, a.ahi, a.alo, D, EPS, one, gs, nul, nul, zero, zero });
        try r.moe(l, T);
        const hgs: i32 = @intCast(m.head.gs);
        try d.launch(d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, mt.norm, a.h, a.ahi, a.alo, D, EPS, one, hgs, nul, nul, zero, zero });
        const last = T - 1;
        const kg = s.hidden / m.head.gs;
        try dev.copy(d, off(mt.state, r.cur * s.hidden * 2), off(a.h, last * s.hidden * 2), s.hidden * 2);
        try fwd.gemm(d, a.pscr, off(a.h, last * s.hidden * 2), off(a.ahi, last * kg * 2), off(a.alo, last * kg * 2), m.head, a.logits, 1);
        try d.launch(d.k.argmax_part, .{ .x = 256, .y = 1 }, .{ a.logits, @as(i32, @intCast(s.vocab)), a.pv, a.pi });
        try d.launch(d.k.argmax, .{ .x = 1 }, .{ a.pv, a.pi, a.tok });
        var tok: [1]i32 = undefined;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &tok, a.tok, 4, null, 0, null));
        try d.sync();
        return tok[0];
    }

    /// One speculative step of the current slot (greedy, MTP on): up to `k` drafts from the MTP, one verify block,
    /// the kept rows committed and absorbed. The tokens it yields go to `out` (1 to k + 1 of them); r.last is the
    /// newest, not yet fed.
    pub fn specStep(r: *Runner, k: u32, out: []i32) !usize {
        const mt = r.mtp orelse return error.NoMtpHead;
        const s = r.z.s;
        if (k == 0 or k + 1 > WINDOW or out.len < k + 1) return error.BadDrafts;
        const n: u32 = @min(k, r.ctx - r.pos - 1);
        if (n == 0) {
            try r.step(r.last);
            out[0] = r.last;
            return 1;
        }
        var blk: [WINDOW]i32 = undefined;
        blk[0] = r.last;
        blk[1] = r.sdraft[r.cur];
        var j: u32 = 1;
        while (j < n) : (j += 1) { // chained drafts: the row before's output, at the next MTP slots
            try dev.copy(&r.d, mt.hin, off(mt.state, r.cur * s.hidden * 2), s.hidden * 2);
            blk[j + 1] = try r.mtpForward(blk[j .. j + 1], r.mpos[r.cur] + j - 1);
        }
        const p0 = r.pos;
        const g = try r.verify(blk[0 .. n + 1]);
        var a: usize = 0;
        while (a < n and blk[a + 1] == g[a]) a += 1;
        @memcpy(out[0 .. a + 1], g[0 .. a + 1]);
        try r.commit(@intCast(a + 1));
        try r.absorb(out[0 .. a + 1], p0);
        return a + 1;
    }

    /// The fp16 logits of the last forward's final row (`out.len` is the vocabulary size).
    pub fn logits(r: *Runner, out: []f16) !void {
        if (out.len != r.z.s.vocab) return error.BadLength;
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, out.ptr, r.acts.logits, out.len * 2, null, 0, null));
        try r.d.sync();
    }

    /// One serial decode token.
    pub fn step(r: *Runner, token: i32) !void {
        r.last = try r.forward(&.{token}, true, .{});
    }

    /// Runs a block at r.pos (2 to WINDOW rows: the pending token, then drafts) and returns each row's greedy token,
    /// with the same arithmetic a row as `step`. The position and states move only at `commit`.
    pub fn verify(r: *Runner, ids: []const i32) ![]const i32 {
        if (ids.len < 2 or ids.len > WINDOW) return error.BadBlock;
        const p = r.pos;
        r.verifying = true;
        defer r.verifying = false;
        _ = try r.forward(ids, true, .{});
        r.pos = p;
        r.block_rows = @intCast(ids.len);
        return r.block[0..ids.len];
    }

    /// Keeps the first `keep` rows of the last verified block. All kept: the block's final GDN states are the states.
    /// Else each GDN layer replays the kept rows from the entry state, and its convolution state is rebuilt. The K/V
    /// rows past the kept ones are written again by later forwards.
    pub fn commit(r: *Runner, keep: u32) !void {
        if (keep == 0 or keep > r.block_rows) return error.BadCommit;
        const d = &r.d;
        const z = r.z;
        const s = z.s;
        const at = StashAt.of(z);
        const nul: ?*anyopaque = null;
        const k32: i32 = @intCast(keep);
        for (r.model.layers) |*l| if (!l.attn) {
            if (keep == r.block_rows) {
                std.mem.swap(?*anyopaque, &l.rec, &l.rec_save);
                l.st[r.cur].rec = l.rec;
                l.st[r.cur].rec_save = l.rec_save;
                continue;
            }
            const sb = l.stash.?;
            try d.launch(d.k.gdn_step, .{ .x = 16, .y = s.gdn_v_heads, .z = 1 }, .{ sb, off(sb, at.k), off(sb, at.v), off(sb, at.g), off(sb, at.b), l.rec.?, l.rec.?, r.acts.y, k32, nul, nul, nul, nul });
            // the block's forward flipped conv_cur: conv[cur] holds the state after all rows, conv[1 - cur] the entry
            try d.launch(d.k.conv_commit, .{ .x = z.nconv / 256 }, .{ l.conv[1 - l.conv_cur].?, off(sb, at.p), l.conv[l.conv_cur].?, k32, nul });
        };
        r.pos += keep;
        r.last = r.block[keep - 1];
        r.block_rows = 0;
    }

    /// One forward over at most WINDOW rows at r.pos; the last row's greedy token when `want`, else -1.
    fn forward(r: *Runner, ids: []const i32, want: bool, f: Feed) !i32 {
        if (ids.len == 0 or ids.len > WINDOW) return error.BadWindow;
        if (r.pos + ids.len > r.ctx) return error.ContextFull;
        const d = &r.d;
        const a = &r.acts;
        const m = &r.model;
        const z = r.z;
        const s = z.s;
        const T: u32 = @intCast(ids.len);
        const t32: i32 = @intCast(T);
        const p0: i32 = @intCast(r.pos);
        const D: i32 = @intCast(s.hidden);
        const gs: i32 = 64;
        const nul: ?*anyopaque = null;
        const zero: i32 = 0;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, a.ids, ids.ptr, ids.len * 4, null, 0, null));
        errdefer d.settle();
        const rope = try r.feedRope(f, T);
        try r.embed(T);
        try r.inject(f);
        for (m.layers, 0..) |*l, i| {
            try d.launch(d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, l.in_norm, a.h, a.ahi, a.alo, D, EPS, @as(i32, if (i == 0) 0 else 1), gs, nul, nul, zero, zero });
            try fwd.gemm(d, a.pscr, a.h, a.ahi, a.alo, l.mixer_in, a.p, T);
            if (l.attn) {
                const sp = span(r.pos + T);
                const ns: u32 = (r.pos + T + sp - 1) / sp;
                try d.launch(d.k.attn_prep, .{ .x = T }, .{ a.p, l.qn.?, l.kn.?, m.invf, a.q, l.kc.?, l.vc.?, p0, EPS, nul, nul, nul, zero, r.cap, rope });
                try d.launch(r.mk.attn_span, .{ .x = ns, .y = s.kv_heads * (s.q_heads / s.kv_heads / QPG), .z = T }, .{ a.q, l.kc.?, l.vc.?, a.pm, a.pl, a.po, p0, @as(i32, @intCast(ns)), @as(i32, @intCast(sp)), @as(f32, 1.0 / 16.0), nul, nul, r.cap });
                try d.launch(r.mk.merge_span, .{ .x = s.q_heads, .y = T }, .{ a.pm, a.pl, a.po, a.y, p0, @as(i32, @intCast(ns)), @as(i32, @intCast(sp)), nul });
                try d.launch(d.k.attn_gate, .{ .x = T }, .{ a.y, a.p, a.h, a.ahi, a.alo, gs, zero, nul, nul, zero });
            } else {
                const cin = l.conv[l.conv_cur].?;
                const cout = l.conv[1 - l.conv_cur].?;
                l.conv_cur = 1 - l.conv_cur;
                if (T == 1 and !r.verifying) // one row: the three GDN kernels in one
                    try d.launch(r.mk.gdn_one, .{ .x = s.gdn_v_heads }, .{ a.p, cin, cout, l.conv_w.?, l.alog.?, l.dt.?, l.rec.?, l.gnorm.?, a.h, a.ahi, a.alo, EPS, gs })
                else {
                try d.launch(d.k.gdn_pre, .{ .x = T, .y = z.nconv / 256 }, .{ a.p, cin, cout, l.conv_w.?, l.alog.?, l.dt.?, a.q, a.k, a.v, a.g, a.beta, t32, nul, nul });
                if (r.verifying) { // the inputs, for commit's replay of the kept rows
                    const sb = l.stash.?;
                    const at = StashAt.of(z);
                    try dev.copy(d, sb, a.q, T * s.gdn_k_heads * 128 * 2);
                    try dev.copy(d, off(sb, at.k), a.k, T * s.gdn_k_heads * 128 * 2);
                    try dev.copy(d, off(sb, at.v), a.v, T * z.vd * 2);
                    try dev.copy(d, off(sb, at.g), a.g, T * s.gdn_v_heads * 4);
                    try dev.copy(d, off(sb, at.b), a.beta, T * s.gdn_v_heads * 4);
                    try dev.copy(d, off(sb, at.p), a.p, T * z.gdn_in * 2);
                }
                // verify: the state after the block to rec_save, rec keeps the entry
                try d.launch(d.k.gdn_step, .{ .x = 16, .y = s.gdn_v_heads, .z = 1 }, .{ a.q, a.k, a.v, a.g, a.beta, l.rec.?, if (r.verifying) l.rec_save.? else l.rec.?, a.y, t32, nul, nul, nul, nul });
                try d.launch(d.k.gated_norm, .{ .x = T }, .{ a.y, a.p, l.gnorm.?, a.h, a.ahi, a.alo, EPS, gs, zero, nul, nul, zero });
                }
            }
            try fwd.gemm(d, a.pscr, a.h, a.ahi, a.alo, l.mixer_out, a.r, T);
            try d.launch(d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, l.post_norm, a.h, a.ahi, a.alo, D, EPS, @as(i32, 1), gs, nul, nul, zero, zero });
            try r.moe(l, T);
        }
        r.pos += T;
        return r.head(T, want);
    }

    /// One prompt chunk of WINDOW < rows <= CHUNK at r.pos: norms, GEMMs, attention (attn_prompt) and the MoE over all
    /// rows, the GDN in windows of WINDOW rows.
    fn chunk(r: *Runner, ids: []const i32, want: bool, f: Feed) !i32 {
        if (ids.len <= WINDOW or ids.len > CHUNK) return error.BadWindow;
        if (r.pos + ids.len > r.ctx) return error.ContextFull;
        const d = &r.d;
        const a = &r.acts;
        const m = &r.model;
        const z = r.z;
        const s = z.s;
        const T: u32 = @intCast(ids.len);
        const D: i32 = @intCast(s.hidden);
        const gs: i32 = 64;
        const nul: ?*anyopaque = null;
        const zero: i32 = 0;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, a.ids, ids.ptr, ids.len * 4, null, 0, null));
        errdefer d.settle();
        const rope = try r.feedRope(f, T);
        try r.embed(T);
        try r.inject(f);
        for (m.layers, 0..) |*l, i| {
            try d.launch(d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, l.in_norm, a.h, a.ahi, a.alo, D, EPS, @as(i32, if (i == 0) 0 else 1), gs, nul, nul, zero, zero });
            try fwd.gemm(d, a.pscr, a.h, a.ahi, a.alo, l.mixer_in, a.p, T);
            if (l.attn) {
                const p0: i32 = @intCast(r.pos);
                try d.launch(d.k.attn_prep, .{ .x = T }, .{ a.p, l.qn.?, l.kn.?, m.invf, a.q, l.kc.?, l.vc.?, p0, EPS, nul, nul, nul, zero, r.cap, rope });
                if (r.cap > 0)
                    try d.launch(d.k.attn_prompt8, .{ .x = (T + 7) / 8, .y = s.kv_heads }, .{ a.q, l.kc.?, l.vc.?, a.y, p0, @as(i32, @intCast(T)), @as(f32, 1.0 / 16.0), r.cap })
                else
                    try d.launch(d.k.attn_prompt, .{ .x = (T + 7) / 8, .y = s.kv_heads }, .{ a.q, l.kc.?, l.vc.?, a.y, p0, @as(i32, @intCast(T)), @as(f32, 1.0 / 16.0) });
                try d.launch(d.k.attn_gate, .{ .x = T }, .{ a.y, a.p, a.h, a.ahi, a.alo, gs, zero, nul, nul, zero });
            }
            const pin = l.mixer_in.n;
            const hk = l.mixer_out.k;
            var j: u32 = 0;
            while (!l.attn and j < T) : (j += WINDOW) {
                const w = @min(WINDOW, T - j);
                const w32: i32 = @intCast(w);
                const pj = off(a.p, @as(usize, j) * pin * 2);
                const hj = off(a.h, @as(usize, j) * hk * 2);
                const hij = off(a.ahi, @as(usize, j) * (hk / 64) * 2);
                const loj = off(a.alo, @as(usize, j) * (hk / 64) * 2);
                const cin = l.conv[l.conv_cur].?;
                const cout = l.conv[1 - l.conv_cur].?;
                l.conv_cur = 1 - l.conv_cur;
                try d.launch(d.k.gdn_pre, .{ .x = w, .y = z.nconv / 256 }, .{ pj, cin, cout, l.conv_w.?, l.alog.?, l.dt.?, a.q, a.k, a.v, a.g, a.beta, w32, nul, nul });
                try d.launch(d.k.gdn_step, .{ .x = 16, .y = s.gdn_v_heads, .z = 1 }, .{ a.q, a.k, a.v, a.g, a.beta, l.rec.?, l.rec.?, a.y, w32, nul, nul, nul, nul });
                try d.launch(d.k.gated_norm, .{ .x = w }, .{ a.y, pj, l.gnorm.?, hj, hij, loj, EPS, gs, zero, nul, nul, zero });
            }
            try fwd.gemm(d, a.pscr, a.h, a.ahi, a.alo, l.mixer_out, a.r, T);
            try d.launch(d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, l.post_norm, a.h, a.ahi, a.alo, D, EPS, @as(i32, 1), gs, nul, nul, zero, zero });
            try r.moe(l, T);
        }
        r.pos += T;
        return r.head(T, want);
    }
    /// The prompt form of `moe`: pairs sorted into per-expert tiles, then gate|up and down a tile at a time.
    fn moeTiles(r: *Runner, l: *const Layer, T: u32) !void {
        const d = &r.d;
        const a = &r.acts;
        const s = r.z.s;
        const pairs = T * s.top_k;
        const ff = s.expert_ff;
        const tb = tileBound(s, T);
        try d.launch(r.mk.sums, .{ .x = s.hidden / 64, .y = T }, .{ a.h, a.mxs, s.hidden });
        try d.launch(r.mk.sort, .{ .x = 1 }, .{ a.mids, a.morder, a.mtiles, a.mnt, s.experts, pairs, T });
        try d.launch(r.mk.tile_swiglu, .{ .x = ff / 16, .y = tb }, .{ l.experts.gu_w, l.experts.gu_s, l.experts.gu_b, l.shared.gu_w, l.shared.gu_s, l.shared.gu_b, a.h, a.mxs, a.morder, a.mtiles, a.mnt, a.mact, s.hidden, ff, s.top_k, pairs, s.experts });
        try d.launch(r.mk.sums, .{ .x = ff / 64, .y = pairs + T }, .{ a.mact, a.mas, ff });
        try d.launch(r.mk.tile_down, .{ .x = s.hidden / 32, .y = tb }, .{ l.experts.dn_w, l.experts.dn_s, l.experts.dn_b, l.shared.dn_w, l.shared.dn_s, l.shared.dn_b, a.mact, a.mas, a.morder, a.mtiles, a.mnt, a.my, s.hidden, ff, s.experts });
        try d.launch(r.mk.combine, .{ .x = s.hidden / 64, .y = T }, .{ a.my, a.mwts, a.msw, a.r, s.hidden, s.top_k, pairs });
    }

    /// The M-RoPE positions of a forward's T rows at r.pos: the feed's, else the slot's shift past its images, else
    /// none (every axis at the row's index, the kernels' own rotary).
    fn feedRope(r: *Runner, f: Feed, T: u32) !?*anyopaque {
        if (f.rope) |rows| {
            @memcpy(r.rope_host[0..T], rows);
            return r.ropeRows(T);
        }
        const k = r.shift[r.cur];
        if (k == 0) return null;
        for (r.rope_host[0..T], 0..) |*p, t| p.* = @splat(@as(i32, @intCast(r.pos + t)) + k);
        return r.ropeRows(T);
    }

    /// rope_host's first T rows to acts.rope.
    fn ropeRows(r: *Runner, T: u32) !*anyopaque {
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, r.acts.rope, r.rope_host.ptr, T * 3 * 4, null, 0, null));
        return r.acts.rope;
    }

    /// The feed's tower rows over their placeholders' embeddings in acts.x.
    fn inject(r: *Runner, f: Feed) !void {
        const D: usize = r.z.s.hidden;
        for (f.rows) |x| try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, off(r.acts.x, x.at * D * 4), x.src, x.n * D * 4, null, 0, null));
    }

    /// After a forward of `T` rows: the greedy token of the last row when `want`, else -1 (after a sync).
    /// The embeddings of the T ids at acts.ids into acts.x: MLX 4/64 rows, or an fp16 table.
    fn embed(r: *Runner, T: u32) !void {
        const m = &r.model;
        const a = &r.acts;
        const D: i32 = @intCast(r.z.s.hidden);
        if (m.embed_h) |e| return r.d.launch(r.d.k.embed_dense, .{ .x = T }, .{ a.ids, e, a.x, D });
        try r.d.launch(r.d.k.embed, .{ .x = T }, .{ a.ids, m.embed_w, m.embed_s, m.embed_b, a.x, D });
    }

    /// T rows of h (fp16 [T, D], with their group sums for the 4/64 head) to logits: the GEMM, or fp16 rows.
    fn headRows(r: *Runner, h: *anyopaque, ahi: *anyopaque, alo: *anyopaque, T: u32) !void {
        const m = &r.model;
        const a = &r.acts;
        const v: i32 = @intCast(r.z.s.vocab);
        if (m.head_h) |w| return r.d.launch(r.d.k.dense_rows, .{ .x = (T + 15) / 16, .y = r.z.s.vocab / 16 }, .{ h, w, a.logits, @as(i32, @intCast(T)), v, @as(i32, 0) });
        try fwd.gemm(&r.d, a.pscr, h, ahi, alo, m.head, a.logits, T);
    }

    fn head(r: *Runner, T: u32, want: bool) !i32 {
        const d = &r.d;
        const a = &r.acts;
        const m = &r.model;
        const s = r.z.s;
        const D: i32 = @intCast(s.hidden);
        const nul: ?*anyopaque = null;
        const zero: i32 = 0;
        errdefer d.settle();
        if (!want and r.mtp == null) {
            try d.sync();
            return -1;
        }
        const hgs: i32 = @intCast(m.head.gs);
        try d.launch(d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, m.norm, a.h, a.ahi, a.alo, D, EPS, @as(i32, 1), hgs, nul, nul, zero, zero });
        if (!want) { // MTP: the normed rows only, for the absorb
            try d.sync();
            return -1;
        }
        if (r.verifying) { // every row's token, into block
            try fwd.gemm(d, a.pscr, a.h, a.ahi, a.alo, m.head, a.logits, T);
            try d.launch(d.k.argmax_part, .{ .x = 256, .y = T }, .{ a.logits, @as(i32, @intCast(s.vocab)), a.pv, a.pi });
            try d.launch(d.k.argmax, .{ .x = T }, .{ a.pv, a.pi, a.tok });
            try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &r.block, a.tok, T * 4, null, 0, null));
            try d.sync();
            return r.block[T - 1];
        }
        const first = T - 1;
        const kg = s.hidden / m.head.gs;
        try r.headRows(off(a.h, first * s.hidden * 2), off(a.ahi, first * kg * 2), off(a.alo, first * kg * 2), 1);
        try d.launch(d.k.argmax_part, .{ .x = 256, .y = 1 }, .{ a.logits, @as(i32, @intCast(s.vocab)), a.pv, a.pi });
        try d.launch(d.k.argmax, .{ .x = 1 }, .{ a.pv, a.pi, a.tok });
        var tok: [1]i32 = undefined;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &tok, a.tok, 4, null, 0, null));
        try d.sync();
        return tok[0];
    }

    /// The MoE block of `T` rows: h (the post-attention norm) to r (its output, the residual's next addend).
    fn moe(r: *Runner, l: *const Layer, T: u32) !void {
        const d = &r.d;
        const a = &r.acts;
        const s = r.z.s;
        const pairs = T * s.top_k;
        if (T > WINDOW)
            try d.launch(r.mk.router_tile, .{ .x = (s.experts + 16) / 16, .y = (T + 15) / 16 }, .{ a.h, l.router, l.shared_gate, a.mlog, s.hidden, s.experts, T })
        else
            try d.launch(r.mk.router, .{ .x = s.experts + 1, .y = T }, .{ a.h, l.router, l.shared_gate, a.mlog, s.hidden });
        try d.launch(r.mk.route, .{ .x = T }, .{ a.mlog, a.mids, a.mwts, a.msw, s.experts, s.top_k });
        if (s.expert_ff != s.shared_ff) return error.SharedWidthNotWired; // one act stride for both (Sage: 512, 512)
        if (T > WINDOW) return r.moeTiles(l, T);
        if (T > 1 and !r.verifying) { // several rows: each expert's weights once for all its pairs (verify: the rows' own kernels, as one row)
            try d.launch(r.mk.group_plan, .{ .x = 1 }, .{ a.mids, a.mplan, s.experts, pairs, T });
            try d.launch(r.mk.group_swiglu, .{ .x = s.expert_ff, .y = pairs + 1 }, .{ l.experts.gu_w, l.experts.gu_s, l.experts.gu_b, l.shared.gu_w, l.shared.gu_s, l.shared.gu_b, a.h, a.mplan, a.mact, s.hidden, s.expert_ff, s.top_k, pairs, s.experts });
            try d.launch(r.mk.group_down, .{ .x = s.hidden / 16, .y = pairs + 1 }, .{ l.experts.dn_w, l.experts.dn_s, l.experts.dn_b, l.shared.dn_w, l.shared.dn_s, l.shared.dn_b, a.mact, a.mplan, a.my, s.hidden, s.expert_ff, s.experts });
            return d.launch(r.mk.combine, .{ .x = s.hidden / 64, .y = T }, .{ a.my, a.mwts, a.msw, a.r, s.hidden, s.top_k, pairs });
        }
        const gu_args = .{ l.experts.gu_w, l.experts.gu_s, l.experts.gu_b, l.shared.gu_w, l.shared.gu_s, l.shared.gu_b, a.h, a.mids, a.mact, s.hidden, s.expert_ff, s.top_k, pairs };
        if (s.hidden == 2048) // Sage: x in registers, 4 rows a work-group
            try d.launch(r.mk.swiglu4, .{ .x = s.expert_ff / 4, .y = pairs + T }, gu_args)
        else
            try d.launch(r.mk.swiglu, .{ .x = s.expert_ff, .y = pairs + T }, gu_args);
        if (T == 1 and s.expert_ff == 512) // one row: down and the combine in one kernel
            return d.launch(r.mk.down_combine, .{ .x = s.hidden / (MOE_SG * DC_ROWS) }, .{ l.experts.dn_w, l.experts.dn_s, l.experts.dn_b, l.shared.dn_w, l.shared.dn_s, l.shared.dn_b, a.mact, a.mids, a.mwts, a.msw, a.r, s.hidden, s.top_k });
        const dn_args = .{ l.experts.dn_w, l.experts.dn_s, l.experts.dn_b, l.shared.dn_w, l.shared.dn_s, l.shared.dn_b, a.mact, a.mids, a.my, s.hidden, s.expert_ff, pairs };
        if (s.expert_ff == 512) // Sage: act in registers, 8 rows a sub-group
            try d.launch(r.mk.down8, .{ .x = s.hidden / (MOE_SG * 8), .y = pairs + T }, dn_args)
        else
            try d.launch(r.mk.down, .{ .x = s.hidden / MOE_SG, .y = pairs + T }, dn_args);
        try d.launch(r.mk.combine, .{ .x = s.hidden / 64, .y = T }, .{ a.my, a.mwts, a.msw, a.r, s.hidden, s.top_k, pairs });
    }
};

/// The Dev's kernel table from the Sage images.
fn kernels(d: *Dev, mk: *MoeKernels) !void {
    const mg = try d.g.module(&kb.sage_glue);
    const md = try d.g.module(&kb.sage_decode);
    const mm = try d.g.module(&kb.gemm);
    const mgd = try d.g.module(&kb.gdn);
    const mw = try d.g.module(&kb.ngen_wide);
    const mn = try d.g.module(&kb.ngen_narrow);
    const mm128 = try d.g.module(&kb.gemm128);
    const mw128 = try d.g.module(&kb.ngen_wide128);
    const mn128 = try d.g.module(&kb.ngen_narrow128);
    const mo = try d.g.module(&kb.sage_moe);
    const attn_decode = try d.kernel(md, "attn_decode", 16);
    d.k = .{
        .embed = try d.kernel(mg, "embed", 256),
        .add_rmsnorm = try d.kernel(mg, "add_rmsnorm", 128),
        .add_rmsnorm_dec = try d.kernel(mg, "add_rmsnorm_dec", 512),
        .gdn_pre = try d.kernel(mg, "gdn_pre", 256),
        .gdn_pre_rows = try d.kernel(mg, "gdn_pre_rows", 256),
        .gated_norm = try d.kernel(mg, "gated_norm", 256),
        .attn_prep = try d.kernel(mg, "attn_prep", 256),
        .attn_gate = try d.kernel(mg, "attn_gate", 256),
        .swiglu = try d.kernel(mg, "swiglu", 256),
        .argmax = try d.kernel(mg, "argmax", 256),
        .argmax_part = try d.kernel(mg, "argmax_part", 256),
        .argmax_part_masked = try d.kernel(mg, "argmax_part_masked", 256),
        .tap = try d.kernel(mg, "tap", 256),
        .conv_commit = try d.kernel(mg, "conv_commit", 256),
        .repack = try d.kernel(mg, "repack", 16),
        .gdn_step = try d.kernel(md, "gdn_step", 16),
        .attn_decode = attn_decode,
        .attn_merge = try d.kernel(md, "attn_merge", 16),
        .kv_move = try d.kernel(md, "kv_move", 16),
        .exl3_gemv = try d.kernel(md, "exl3_gemv", 16),
        .rot_heads = try d.kernel(mg, "rot_heads", 16),
        .attn_prompt = try d.kernel(md, "attn_prompt", 128),
        .attn_prompt8 = try d.kernel(md, "attn_prompt8", 128),
        .embed_dense = try d.kernel(mg, "embed_dense", 256),
        .dense_rows = try d.kernel(mg, "dense_rows", 256),
        .repack_gptq = try d.kernel(mg, "repack_gptq", 16),
        .quantize_mlx = try d.kernel(mg, "quantize_mlx", 16),
        .dec_wide_g128 = try d.kernel(mm128, "w4k1_decode", 64),
        .dec_wide64_g128 = try d.kernel(mm128, "w4k1_d64", 64),
        .dec_narrow_g128 = try d.kernel(mm128, "w4k_dsplit", 64),
        .ngen_wide_g128 = try d.kernel(mw128, "w4g_dpb", 256),
        .ngen_narrow_g128 = try d.kernel(mn128, "w4g_s4dpb", 256),
        .ngen_swiglu = try d.kernel(try d.g.module(&kb.ngen_swiglu), "w4g_dpb_swiglu", 256),
        .ngen_swiglu_g128 = try d.kernel(try d.g.module(&kb.ngen_swiglu128), "w4g_dpb_swiglu", 256),
        .rowsums_ff = try d.kernel(mg, "rowsums_ff", 128),
        .fakequant_rows = try d.kernel(mg, "fakequant_rows", 256),
        .a8 = try d.kernel(try d.g.module(&kb.ngen_a8), "w4a8", 256),
        .a8_swiglu = try d.kernel(try d.g.module(&kb.ngen_a8_swiglu), "w4a8_swiglu", 256),
        .quant_a8 = try d.kernel(mg, "quant_a8", 128),
        .quant_a4 = try d.kernel(mg, "quant_a4", 128),
        .a4 = try d.kernel(try d.g.module(&kb.ngen_a4), "w4a4", 256),
        .dec_wide = try d.kernel(mm, "w4k1_decode", 64),
        .dec_wide64 = try d.kernel(mm, "w4k1_d64", 64),
        .dec_narrow = try d.kernel(mm, "w4k_dsplit", 64),
        .gdn_prep = try d.kernel(mgd, "gdn_prep", 64),
        .gdn_state = try d.kernel(mgd, "gdn_state", 256),
        .gdn_out = try d.kernel(mgd, "gdn_out", 128),
        .ngen_wide = try d.kernel(mw, "w4g_dpb", 256),
        .ngen_narrow = try d.kernel(mn, "w4g_s4dpb", 256),
    };
    mk.* = .{
        .router = try d.kernel(mo, "sage_router", 16),
        .route = try d.kernel(mo, "sage_route", 16),
        .swiglu = try d.kernel(mo, "sage_expert_swiglu", SWIGLU_WG),
        .down = try d.kernel(mo, "sage_expert_down", 16 * MOE_SG),
        .down8 = try d.kernel(mo, "sage_expert_down8", 16 * MOE_SG),
        .swiglu4 = try d.kernel(mo, "sage_expert_swiglu4", SWIGLU_WG),
        .down_combine = try d.kernel(mo, "sage_down_combine", 16 * MOE_SG),
        .gdn_one = try d.kernel(mg, "gdn_one", 256),
        .combine = try d.kernel(mo, "sage_moe_combine", 64),
        .attn_span = try d.kernel(md, "attn_decode_span", 16),
        .merge_span = try d.kernel(md, "attn_merge_span", 256),
        .group_plan = try d.kernel(mo, "sage_group_plan", 256),
        .group_swiglu = try d.kernel(mo, "sage_group_swiglu", 16),
        .group_down = try d.kernel(mo, "sage_group_down", 16),
        .router_tile = try d.kernel(mo, "sage_router_tile", 16),
        .sums = try d.kernel(mo, "sage_group_sums", 16),
        .sort = try d.kernel(mo, "sage_moe_sort", 256),
        .tile_swiglu = try d.kernel(mo, "sage_tile_swiglu", 16),
        .tile_down = try d.kernel(mo, "sage_tile_down", 16),
    };
}

fn readFile(path: []const u8) ![]u8 {
    const z = try std.fmt.allocPrintSentinel(alloc, "{s}", .{path}, 0);
    defer alloc.free(z);
    const fd = open(z.ptr, 0);
    if (fd < 0) return error.FileNotFound;
    defer _ = close(fd);
    var out: std.ArrayList(u8) = .empty;
    var buf: [1 << 16]u8 = undefined;
    while (true) {
        const n = read(fd, &buf, buf.len);
        if (n < 0) return error.ReadFailed;
        if (n == 0) break;
        try out.appendSlice(alloc, buf[0..@intCast(n)]);
    }
    return out.toOwnedSlice(alloc);
}

fn readShape(dir: []const u8) !cfgm.Shape {
    const bytes = try readFile(name("{s}/config.json", .{dir}));
    defer alloc.free(bytes);
    const p = try cfgm.parse(alloc, bytes);
    defer p.deinit();
    const s = try cfgm.validate(p.value);
    var buf: [1024]u8 = undefined;
    // the images were built with sage_defines: a checkpoint of another shape would run the wrong kernels
    if (!std.mem.eql(u8, try cfgm.kernelDefines(&buf, s, QPG), cfgm.sage_defines)) return error.ShapeNotBuilt;
    return s;
}

/// One tensor to convert for the MLX loader: a GPTQ projection, a bf16 matrix to quantize, or a norm to shift.
const Job = struct {
    kind: enum { gptq, quantize, plus_one },
    base: []const u8, // the stored name less `.qweight` or `.weight`
    k: usize,
    n: usize,
    w: []u32 = &.{},
    s: []u16 = &.{},
    b: []u16 = &.{},
    fn run(j: *Job, c: *const Checkpoint) !void {
        var nb: [256]u8 = undefined;
        switch (j.kind) {
            .gptq => {
                const qw = try c.get(try std.fmt.bufPrint(&nb, "{s}.qweight", .{j.base}));
                const qz = try c.get(try std.fmt.bufPrint(&nb, "{s}.qzeros", .{j.base}));
                const sc = try c.get(try std.fmt.bufPrint(&nb, "{s}.scales", .{j.base}));
                try sl.gptqAsMlx(qw.bytes, qz.bytes, sc.bytes, j.k, j.n, j.w, j.s, j.b);
            },
            .quantize => try sl.quantizeRows((try c.get(try std.fmt.bufPrint(&nb, "{s}.weight", .{j.base}))).bytes, j.n, j.k, j.w, j.s, j.b),
            .plus_one => sl.plusOne((try c.get(try std.fmt.bufPrint(&nb, "{s}.weight", .{j.base}))).bytes, j.s),
        }
    }
    fn some(jobs: []Job, c: *const Checkpoint, failed: *std.atomic.Value(bool)) void {
        for (jobs) |*j| j.run(c) catch {
            failed.store(true, .release);
            return;
        };
    }
};

/// A W4A16 AutoRound (GPTQ symmetric 4 bit) checkpoint, shown to the MLX loader as an MLX export: each quantized
/// projection as MLX affine 4/64 words, scales and biases (sl.gptqAsMlx: no requantization), the bf16 embedding and
/// lm_head quantized to 4/64 as an MLX export has them, the RMSNorm weights as 1 + w. MTP and vision tensors are not
/// read. False (and nothing changed) for other checkpoints. The checkpoint gets a name index either way.
fn autoRound(c: *Checkpoint) !bool {
    return showAsMlx(c, .main);
}

/// The tensors autoRound converts: `main`, the model less MTP and vision; `mtp`, the MTP head only (every norm as
/// 1 + w, fc and the router gates left as stored, for loadMtp).
fn showAsMlx(c: *Checkpoint, which: enum { main, mtp }) !bool {
    try c.buildIndex();
    if (which == .main and !c.has("layers.0.mlp.shared_expert.gate_proj.qweight")) return false;
    // the embedding and lm_head as 4/64, as MLX has them; XPU_SAGE_HEAD16=1 keeps fp16 copies of the bf16 ones (text
    // NLL 1.9933 against 2.0023 on 1499 tokens, decode about 19% slower: the head reads 1 GB a token)
    const head4 = std.c.getenv("XPU_SAGE_HEAD16") == null;
    var jobs: std.ArrayList(Job) = .empty;
    defer jobs.deinit(alloc);
    for (c.tensors.items[0..c.stored]) |t| {
        const n = t.name;
        const in_mtp = std.mem.startsWith(u8, n, "mtp.");
        if (which == .mtp) {
            if (!in_mtp) continue;
            if (std.mem.endsWith(u8, n, ".qweight")) {
                try jobs.append(alloc, .{ .kind = .gptq, .base = n[0 .. n.len - 8], .k = t.shape[0] * 8, .n = t.shape[1] });
            } else if (std.mem.indexOf(u8, n, "norm") != null and std.mem.endsWith(u8, n, ".weight")) {
                if (!std.mem.eql(u8, t.dtype, "BF16")) return error.UnexpectedDtype;
                try jobs.append(alloc, .{ .kind = .plus_one, .base = n[0 .. n.len - 7], .k = t.shape[0], .n = 1 });
            }
            continue;
        }
        if (in_mtp or std.mem.indexOf(u8, n, "visual.") != null) continue;
        if (std.mem.endsWith(u8, n, ".qweight")) {
            try jobs.append(alloc, .{ .kind = .gptq, .base = n[0 .. n.len - 8], .k = t.shape[0] * 8, .n = t.shape[1] });
        } else if (head4 and (std.mem.endsWith(u8, n, "embed_tokens.weight") or std.mem.eql(u8, n, "lm_head.weight"))) {
            if (!std.mem.eql(u8, t.dtype, "BF16")) return error.UnexpectedDtype;
            try jobs.append(alloc, .{ .kind = .quantize, .base = n[0 .. n.len - 7], .k = t.shape[1], .n = t.shape[0] });
        } else if (std.mem.endsWith(u8, n, "layernorm.weight") or std.mem.endsWith(u8, n, "_norm.weight") or std.mem.eql(u8, Checkpoint.keyOf(n), "norm.weight")) {
            if (!std.mem.eql(u8, t.dtype, "BF16")) return error.UnexpectedDtype;
            try jobs.append(alloc, .{ .kind = .plus_one, .base = n[0 .. n.len - 7], .k = t.shape[0], .n = 1 });
        }
    }
    for (jobs.items) |*j| {
        if (j.kind != .plus_one) j.w = try alloc.alloc(u32, j.n * (j.k / 8));
        j.s = try alloc.alloc(u16, if (j.kind == .plus_one) j.k else j.n * (j.k / 64));
        if (j.kind != .plus_one) j.b = try alloc.alloc(u16, j.n * (j.k / 64));
    }
    // the conversion in threads: contiguous slices of the job list
    var failed = std.atomic.Value(bool).init(false);
    const nt: usize = @min(16, @max(1, std.Thread.getCpuCount() catch 1));
    var threads: [16]std.Thread = undefined;
    const per = (jobs.items.len + nt - 1) / nt;
    for (0..nt) |i| {
        const lo = @min(i * per, jobs.items.len);
        threads[i] = try std.Thread.spawn(.{}, Job.some, .{ jobs.items[lo..@min(lo + per, jobs.items.len)], c, &failed });
    }
    for (threads[0..nt]) |t| t.join();
    if (failed.load(.acquire)) return error.AutoRoundConversion;
    var nb: [256]u8 = undefined;
    for (jobs.items) |j| {
        if (j.kind == .plus_one) {
            try c.put(try std.fmt.bufPrint(&nb, "{s}.weight", .{j.base}), "BF16", &.{j.k}, std.mem.sliceAsBytes(j.s));
            continue;
        }
        try c.put(try std.fmt.bufPrint(&nb, "{s}.weight", .{j.base}), "U32", &.{ j.n, j.k / 8 }, std.mem.sliceAsBytes(j.w));
        try c.put(try std.fmt.bufPrint(&nb, "{s}.scales", .{j.base}), "BF16", &.{ j.n, j.k / 64 }, std.mem.sliceAsBytes(j.s));
        try c.put(try std.fmt.bufPrint(&nb, "{s}.biases", .{j.base}), "BF16", &.{ j.n, j.k / 64 }, std.mem.sliceAsBytes(j.b));
    }
    std.debug.print("  AutoRound{s}: {d} tensors shown as MLX affine 4/64\n", .{ if (which == .mtp) " MTP" else "", jobs.items.len });
    return true;
}

fn addShards(c: *Checkpoint, dir: []const u8) !void {
    const bytes = try readFile(name("{s}/model.safetensors.index.json", .{dir}));
    defer alloc.free(bytes);
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, bytes, .{}) catch return error.BadIndex;
    defer parsed.deinit();
    const map = if (parsed.value == .object) parsed.value.object.get("weight_map") else null;
    if (map == null or map.? != .object) return error.BadIndex;
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(alloc);
    var it = map.?.object.iterator();
    while (it.next()) |e| {
        if (e.value_ptr.* != .string) return error.BadIndex;
        const f = e.value_ptr.string;
        if (seen.contains(f)) continue;
        try seen.put(alloc, f, {});
        const shard = try std.fmt.allocPrintSentinel(alloc, "{s}/{s}", .{ dir, f }, 0);
        defer alloc.free(shard);
        try c.addShard(shard);
    }
}
