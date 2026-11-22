//! Inkling 2 Mini CUDA DSpark block-8 drafter: loads mtp.safetensors, folds Phi/beta on GPU, and runs absorb/hold.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const core = @import("core");
const ops = @import("ops.zig");
const bf12 = @import("bf12.zig");
const config = @import("config.zig");
const cuda_native = @import("cuda_native.zig");
const sys = @import("sys");

const dr = lanes.drafter;
const Stream = lanes.Stream;

pub const hidden: u32 = 2304;
pub const query_width: u32 = 3072;
pub const kv_width: u32 = 1024;
pub const intermediate: u32 = 9216;
pub const rank: u32 = 256;
pub const rnn_width: u32 = 768;
pub const block_rows: u32 = 8;
pub const max_depth: u32 = 7;
pub const max_hold_batch: u32 = 16;
pub const vocab: u32 = 200058;
pub const checkpoint_vocab: u32 = 201024;
pub const absorb_rows: u32 = 64;
/// Batched-absorb cap (items.len > 1 only): total rows across streams fused in one shot.
/// The single-stream path still chunks by absorb_rows, unchanged.
pub const batch_absorb_rows: u32 = 256;
pub const local_ring: usize = 1152;
const tap_list = [_]u32{ 5, 23, 41, 48 };

/// One descriptor per stream for the batched absorb/hold kernels (96 bytes, 16-byte aligned).
/// Mirrors ink2d_lane_desc in dspark.cu field for field.
const LaneDesc = extern struct {
    k: [3]u64,
    v: [3]u64,
    newest: u64,
    src: u64, // absorb only: taps src base (device)
    row0: u32, // absorb only: first fused row in the contiguous batch buffer
    n: u32, // absorb only: row count for this stream
    base_pos: u32, // absorb only: absolute position of row0
    ring: [3]u32,
    _pad: [2]u32,
};

comptime {
    if (@sizeOf(LaneDesc) != 96 or @alignOf(LaneDesc) != 8) @compileError("LaneDesc layout drift");
}

pub const Kernels = struct {
    mod: cuda.Module,
    init_rows: cuda.Function,
    init_rows_batch: cuda.Function,
    pack_draft_x: cuda.Function,
    fuse_taps: cuda.Function,
    block_qkv: cuda.Function,
    block_qkv_dev: cuda.Function,
    context_kv: cuda.Function,
    context_kv_i8: cuda.Function,
    attention: cuda.Function,
    attention_dev: cuda.Function,
    attention_i8: cuda.Function,
    attention_dev_i8: cuda.Function,
    attention_splitk_dev_i8: cuda.Function,
    attention_splitk_merge: cuda.Function,
    residual_norm: cuda.Function,
    swiglu: cuda.Function,
    kappa: cuda.Function,
    recurrent: cuda.Function,
    recurrent_dev: cuda.Function,
    base: cuda.Function,
    logits: cuda.Function,
    top1_gumbel: cuda.Function,
    top1_gumbel_dev: cuda.Function,
    fold_phi_beta: cuda.Function,
    quant_phi_fp8: cuda.Function,
    logits_fp8: cuda.Function,
    logits_fp8_batch: cuda.Function,
    fuse_taps_batch: cuda.Function,
    context_kv_batch: cuda.Function,
    context_kv_batch_i8: cuda.Function,
    block_qkv_dev_batch: cuda.Function,
    attention_dev_batch: cuda.Function,
    attention_dev_batch_i8: cuda.Function,
    attention_splitk_dev_batch_i8: cuda.Function,
    kappa_batch: cuda.Function,
    recurrent_dev_batch: cuda.Function,
    top1_gumbel_dev_batch: cuda.Function,
    top1_prob_dev: cuda.Function,
    top1_prob_dev_batch: cuda.Function,
    top16: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available or cuda.kernels.inkling2_dspark.len == 0) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_dspark);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .init_rows = try mod.function("tf_ink2d_init_rows"),
            .init_rows_batch = try mod.function("tf_ink2d_init_rows_batch"),
            .pack_draft_x = try mod.function("tf_ink2d_pack_draft_x"),
            .fuse_taps = try mod.function("tf_ink2d_fuse_taps"),
            .block_qkv = try mod.function("tf_ink2d_block_qkv"),
            .block_qkv_dev = try mod.function("tf_ink2d_block_qkv_dev"),
            .context_kv = try mod.function("tf_ink2d_context_kv"),
            .context_kv_i8 = try mod.function("tf_ink2d_context_kv_i8"),
            .attention = try mod.function("tf_ink2d_attention"),
            .attention_dev = try mod.function("tf_ink2d_attention_dev"),
            .attention_i8 = try mod.function("tf_ink2d_attention_i8"),
            .attention_dev_i8 = try mod.function("tf_ink2d_attention_dev_i8"),
            .attention_splitk_dev_i8 = try mod.function("tf_ink2d_attention_splitk_dev_i8"),
            .attention_splitk_merge = try mod.function("tf_ink2d_attention_splitk_merge"),
            .residual_norm = try mod.function("tf_ink2d_residual_norm"),
            .swiglu = try mod.function("tf_ink2d_swiglu"),
            .kappa = try mod.function("tf_ink2d_kappa"),
            .recurrent = try mod.function("tf_ink2d_recurrent"),
            .recurrent_dev = try mod.function("tf_ink2d_recurrent_dev"),
            .base = try mod.function("tf_ink2d_base"),
            .logits = try mod.function("tf_ink2d_logits"),
            .top1_gumbel = try mod.function("tf_ink2d_top1_gumbel"),
            .top1_gumbel_dev = try mod.function("tf_ink2d_top1_gumbel_dev"),
            .fold_phi_beta = try mod.function("tf_ink2d_fold_phi_beta"),
            .quant_phi_fp8 = try mod.function("tf_ink2d_quant_phi_fp8"),
            .logits_fp8 = try mod.function("tf_ink2d_logits_fp8"),
            .logits_fp8_batch = try mod.function("tf_ink2d_logits_fp8_batch"),
            .fuse_taps_batch = try mod.function("tf_ink2d_fuse_taps_batch"),
            .context_kv_batch = try mod.function("tf_ink2d_context_kv_batch"),
            .context_kv_batch_i8 = try mod.function("tf_ink2d_context_kv_batch_i8"),
            .block_qkv_dev_batch = try mod.function("tf_ink2d_block_qkv_dev_batch"),
            .attention_dev_batch = try mod.function("tf_ink2d_attention_dev_batch"),
            .attention_dev_batch_i8 = try mod.function("tf_ink2d_attention_dev_batch_i8"),
            .attention_splitk_dev_batch_i8 = try mod.function("tf_ink2d_attention_splitk_dev_batch_i8"),
            .kappa_batch = try mod.function("tf_ink2d_kappa_batch"),
            .recurrent_dev_batch = try mod.function("tf_ink2d_recurrent_dev_batch"),
            .top1_gumbel_dev_batch = try mod.function("tf_ink2d_top1_gumbel_dev_batch"),
            .top1_prob_dev = try mod.function("tf_ink2d_top1_prob_dev"),
            .top1_prob_dev_batch = try mod.function("tf_ink2d_top1_prob_dev_batch"),
            .top16 = try mod.function("tf_ink2d_top16"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }
};

const DenseW = struct { ptr: u64 = 0, mat: ?bf12.Mat = null };

const LayerW = struct {
    dflare: u64,
    input_norm: u64,
    post_norm: u64,
    q_norm: u64,
    k_norm: u64,
    k_norm_ctx: u64,
    q: DenseW,
    k: DenseW,
    v: DenseW,
    k_ctx: DenseW,
    v_ctx: DenseW,
    o: DenseW,
    gate: DenseW,
    up: DenseW,
    down: DenseW,
};

const dsp1_magic: u32 = 0x44535031;
const dsp1_rec_bytes: usize = 36844;

const PendingDump = struct {
    last_pos: u64 = 0,
    last_pending: u32 = 0,
    last_n: u8 = 0,
    drafts: [max_depth]u32 = @splat(0),
    probs: [max_depth]f32 = @splat(0),
    margins: [max_depth]f32 = @splat(0),
    top_toks: [max_depth][16]u32 = @splat(@splat(0)),
    top_vals: [max_depth][16]f32 = @splat(@splat(0)),
    x: [max_depth][hidden]u16 = @splat(@splat(0)),
    lam: [max_depth][rank]u16 = @splat(@splat(0)),
};

fn writeDsp1Record(fp: *std.c.FILE, pd: *const PendingDump, kept: u8, new_pending: u32) void {
    var buf: [dsp1_rec_bytes]u8 = undefined;
    var off: usize = 0;
    const put = struct {
        fn u8v(b: []u8, o: *usize, v: u8) void {
            b[o.*] = v;
            o.* += 1;
        }
        fn u16v(b: []u8, o: *usize, v: u16) void {
            std.mem.writeInt(u16, b[o.*..][0..2], v, .little);
            o.* += 2;
        }
        fn u32v(b: []u8, o: *usize, v: u32) void {
            std.mem.writeInt(u32, b[o.*..][0..4], v, .little);
            o.* += 4;
        }
        fn u64v(b: []u8, o: *usize, v: u64) void {
            std.mem.writeInt(u64, b[o.*..][0..8], v, .little);
            o.* += 8;
        }
        fn f32v(b: []u8, o: *usize, v: f32) void {
            std.mem.writeInt(u32, b[o.*..][0..4], @bitCast(v), .little);
            o.* += 4;
        }
    };
    put.u32v(&buf, &off, dsp1_magic);
    put.u8v(&buf, &off, pd.last_n);
    put.u8v(&buf, &off, kept);
    put.u8v(&buf, &off, 16);
    put.u8v(&buf, &off, 0);
    put.u64v(&buf, &off, pd.last_pos);
    put.u32v(&buf, &off, pd.last_pending);
    put.u32v(&buf, &off, new_pending);
    for (pd.drafts) |v| put.u32v(&buf, &off, v);
    for (pd.probs) |v| put.f32v(&buf, &off, v);
    for (pd.margins) |v| put.f32v(&buf, &off, v);
    for (pd.top_toks) |row| for (row) |v| put.u32v(&buf, &off, v);
    for (pd.top_vals) |row| for (row) |v| put.f32v(&buf, &off, v);
    for (pd.x) |row| for (row) |v| put.u16v(&buf, &off, v);
    for (pd.lam) |row| for (row) |v| put.u16v(&buf, &off, v);
    if (off != dsp1_rec_bytes) unreachable;
    _ = std.c.fwrite(&buf, 1, buf.len, fp);
}

const LaneState = struct {
    k: [3]cuda.DeviceBuffer,
    v: [3]cuda.DeviceBuffer,
    rings: [3]u64,
    newest: cuda.DeviceBuffer,
    from: u64 = 0,
    end: u64 = 0,
    held: [7]u32 = @splat(0),
    held_len: u32 = 0,
    verified_len: u32 = 0, // last heldFn's token count; consumed once by the next absorb, else reset
    held_probs: [max_depth]f32 = @splat(0),
    held_margins: [max_depth]f32 = @splat(0),
    has_probs: bool = false,
    pending_dump: ?*PendingDump = null,

    fn init(d: *const cuda.Driver, capacity: usize, kv_int8: bool) !LaneState {
        const rings = [3]u64{ local_ring, capacity, local_ring };
        var ks: [3]cuda.DeviceBuffer = undefined;
        var vs: [3]cuda.DeviceBuffer = undefined;
        var made: usize = 0;
        errdefer for (0..made) |i| {
            ks[i].free();
            vs[i].free();
        };
        for (0..3) |i| {
            const bytes = if (kv_int8 and i == 1) rings[i] * (kv_width + 8 * 4) else rings[i] * kv_width * 2;
            ks[i] = try cuda.DeviceBuffer.alloc(d, bytes);
            vs[i] = try cuda.DeviceBuffer.alloc(d, bytes);
            made += 1;
        }
        var newest = try cuda.DeviceBuffer.alloc(d, hidden * 2);
        errdefer newest.free();
        return .{ .k = ks, .v = vs, .rings = rings, .newest = newest };
    }

    fn deinit(l: *LaneState) void {
        l.newest.free();
        for (0..3) |i| {
            l.k[i].free();
            l.v[i].free();
        }
    }
};

const HoldGraph = struct { depth: u32, cache: u64, exec: cuda.graph.Exec };

/// Greedy batched-hold graph: one device-only holdBatchCore capture per (S, max_d).
/// All per-lane pointers are read indirectly via d.lane_desc (uploaded prior to launch).
const HoldBatchGraph = struct { s: u32, max_d: u32, exec: cuda.graph.Exec };

fn holdGraphsEnabled() bool {
    return sys.environment.read(bool, "TF_INKLING2_GRAPH", true) catch true;
}

/// E4M3 (bias 7, max 448) byte -> f32, the exact inverse of the device f32_to_e4m3 encoder.
/// Exp 15/mant 7 (never emitted: the encoder saturates) decodes as quiet NaN.
fn f8e4m3ToF32(b: u8) f32 {
    const s: u32 = (@as(u32, b >> 7)) << 31;
    const e: u32 = (b >> 3) & 0xF;
    const m: u32 = b & 7;
    if (e == 0) {
        if (m == 0) return @bitCast(s);
        const v: f32 = @as(f32, @floatFromInt(m)) * 0.001953125; // m * 2^-9, exact
        return if (s == 0) v else -v;
    }
    if (e == 15 and m == 7) return @bitCast(s | 0x7FC00000);
    return @bitCast(s | ((e + 120) << 23) | (m << 20));
}

fn go(f: cuda.Function, s: cuda.Stream, grid: u32, block: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = grid }, .block = .{ .x = block } }, s, a);
}

fn dense(d: *const Drafter, s: cuda.Stream, x: u64, w: DenseW, out: u64, rows: usize, n: usize, k: usize) !void {
    if (w.mat) |*mat| {
        const kb = d.target.kernels.b12 orelse return error.Bf12KernelsMissing;
        // Rows-based sk=1: wide calls run whole rows at once (no partials); narrow calls keep the plan.
        const sk: usize = if (rows >= 64 and mat.n >= 64) 1 else mat.plan.sk;
        const chunk: usize = if (sk > 1) 64 else rows;
        var r0: usize = 0;
        while (r0 < rows) {
            const take = @min(rows - r0, chunk);
            try kb.matmul(s, x + r0 * k * 2, k, mat, out + r0 * n * 2, false, d.b12_part, take);
            r0 += take;
        }
        return;
    }
    try ops.rowdot(&d.target.kernels.glue, s, x, w.ptr, out, rows, n, k, true, false);
}

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    target: *cuda_native.Owned,
    k: Kernels,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    fuse: u64 = 0,
    mask: u64 = 0,
    norm: u64 = 0,
    markov_w1: u64 = 0,
    rnn: u64 = 0,
    rnn_bias: u64 = 0,
    dflare: u64 = 0,
    phi: u64 = 0, // BF16 fold (temporary: freed after the FP8 quant)
    phi_fp8: u64 = 0, // FP8-E4M3 phi[vocab, 256]: 51.2 MB, half the BF16 fold
    phi_scale: u64 = 0, // per-row FP32 amax/448 scales for the fp8 dot
    f8lut: u64 = 0, // host-computed E4M3->f32 table for the fp8 dot
    beta: u64 = 0,
    layers: [3]LayerW = undefined,
    b12_part: u64 = 0,
    // scratch
    ids: u64 = 0,
    resid: u64 = 0,
    next_resid: u64 = 0,
    x: u64 = 0,
    y: u64 = 0,
    q_raw: u64 = 0,
    k_raw: u64 = 0,
    v_raw: u64 = 0,
    q: u64 = 0,
    kb: u64 = 0,
    vb: u64 = 0,
    attn: u64 = 0,
    gate: u64 = 0,
    up: u64 = 0,
    act: u64 = 0,
    base: u64 = 0,
    fused: u64 = 0,
    ck: u64 = 0,
    cv: u64 = 0,
    kappa: u64 = 0,
    state: u64 = 0,
    next_state: u64 = 0,
    z: u64 = 0,
    lambda: u64 = 0,
    logits: u64 = 0,
    id_out: u64 = 0,
    draft_tokens: u64 = 0, // device-side token chain: pending + one id a step, no host reads mid-hold
    draft_probs: u64 = 0, // [16][8]f32: per-step top-1 softmax probability
    draft_margins: u64 = 0, // [16][8]f32: per-step top1 - top2 logit margin
    dump_top_toks: u64 = 0, // dump-only [16][7][16]u32
    dump_top_vals: u64 = 0, // dump-only [16][7][16]f32
    dump_lam: u64 = 0, // dump-only [16][7][256]u16
    dump_fp: ?*std.c.FILE = null,
    use_chances: bool = true,
    prob_power: f64 = 1.0,
    landing: lanes.draft_tree.Landing = .{},
    use_landing: bool = true,
    slack: u32 = 2,
    conf_floor: ?f64 = null,
    copy_first: bool = true,
    use_hold_costs: bool = true,
    hold_costs: [4]lanes.hold_price.HoldCost = undefined,
    n_hold_costs: usize = 0,
    hold_meta: u64 = 0, // [4]u32 {anchor, ln_end, ln_from, position}: the _dev kernels' dynamic scalars
    hold_graphs: std.ArrayList(HoldGraph) = .empty,
    hold_batch_graphs: std.ArrayList(HoldBatchGraph) = .empty,
    lane_desc: u64 = 0, // [16]LaneDesc: per-stream table for the batched absorb/hold kernels
    graphs_disabled: bool = false,
    pool: std.ArrayList(LaneState) = .empty, // unbound lane caches for reserve() to commit up front
    bytes: usize = 0, // fixed device bytes (weights + scratch), accumulated in alloc
    costs: [12]lanes.config.Cost = undefined,
    n_costs: usize = 8,
    step_ms: f64 = 0,
    rule: u64 = 0,
    meta: u64 = 0,
    scores: u64 = 0,
    split_o: u64 = 0,
    split_ml: u64 = 0,
    capacity: usize,
    kv_int8: bool = false,
    lanes: std.AutoHashMapUnmanaged(*Stream, LaneState) = .empty,

    fn alloc(d: *Drafter, drv: *const cuda.Driver, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(drv, @max(bytes, 256));
        errdefer b.free();
        try d.bufs.append(d.gpa, b);
        d.bytes += @max(bytes, 256);
        return b.ptr;
    }

    fn uploadTensor(d: *Drafter, drv: *const cuda.Driver, file: *const core.safetensors.File, name: []const u8, dims: []const usize) !u64 {
        const t = file.names.get(name) orelse return error.DSparkMissingTensor;
        if (t.dtype != .bf16 or t.rank != dims.len) return error.DSparkTensorShape;
        var count: usize = 1;
        for (dims, 0..) |want, i| {
            if (t.shape[i] != want) return error.DSparkTensorShape;
            count *= want;
        }
        const raw = file.map.memory[file.data + t.begin .. file.data + t.end];
        if (raw.len != count * 2) return error.DSparkTensorShape;
        const p = try d.alloc(drv, raw.len);
        try d.bufs.items[d.bufs.items.len - 1].upload(0, raw);
        return p;
    }

    fn uploadDense(d: *Drafter, drv: *const cuda.Driver, file: *const core.safetensors.File, name: []const u8, dims: []const usize, part_floats: *usize) !DenseW {
        const t = file.names.get(name) orelse return error.DSparkMissingTensor;
        if (t.dtype != .bf16 or t.rank != dims.len) return error.DSparkTensorShape;
        for (dims, 0..) |want, i| if (t.shape[i] != want) return error.DSparkTensorShape;
        const raw = file.map.memory[file.data + t.begin .. file.data + t.end];
        if (d.target.kernels.b12 != null and dims.len == 2 and bf12.eligible(dims[0], dims[1])) {
            if (bf12.pack(d.gpa, drv, &d.bufs, raw, dims[0], dims[1])) |mat| {
                part_floats.* = @max(part_floats.*, bf12.partFloats(dims[0], dims[1], 64));
                return .{ .ptr = 0, .mat = mat };
            } else |err| switch (err) {
                error.Bf12EntriesOverflow => {},
                else => return err,
            }
        }
        const p = try d.alloc(drv, raw.len);
        try d.bufs.items[d.bufs.items.len - 1].upload(0, raw);
        return .{ .ptr = p, .mat = null };
    }

    pub fn init(gpa: std.mem.Allocator, io: std.Io, target: *cuda_native.Owned, dir: []const u8) !*Drafter {
        const m = target.weights.c.mtp orelse return error.DSparkNotConfigured;
        if (m.layers != 3 or m.q_heads != 24 or m.kv_heads != 8 or m.head_dim != 128 or m.inter != intermediate or
            m.window != 1024 or m.block != 8 or m.rank != rank or m.rope_theta != 1e6 or m.taps_n != 4 or
            !m.local[0] or m.local[1] or !m.local[2] or !std.mem.eql(u32, m.taps[0..4], &tap_list))
            return error.UnsupportedDSparkConfig;

        const drv = target.model.d;
        var k = try Kernels.load(drv);
        var owned = false;
        errdefer if (!owned) k.unload();
        const d = try gpa.create(Drafter);
        d.* = .{
            .gpa = gpa,
            .target = target,
            .k = k,
            .capacity = target.model.context,
            .kv_int8 = target.model.kv_int8,
        };
        owned = true;
        errdefer d.deinit();

        const path: []u8 = if (std.c.getenv("TF_INKLING2_MTP_PATH")) |mp|
            try gpa.dupe(u8, std.mem.span(mp))
        else if (std.mem.endsWith(u8, dir, ".safetensors"))
            try gpa.dupe(u8, dir)
        else
            try std.fs.path.join(gpa, &.{ dir, "mtp.safetensors" });
        defer gpa.free(path);

        d.use_chances = sys.environment.read(bool, "TF_INKLING2_DSPARK_CHANCES", true) catch true;
        d.prob_power = sys.environment.read(f64, "TF_INKLING2_DSPARK_PROB_POWER", 1.0) catch 1.0;
        d.use_landing = sys.environment.read(bool, "TF_INKLING2_DSPARK_LANDING", true) catch true;
        d.slack = sys.environment.read(u32, "TF_INKLING2_DSPARK_SLACK", 2) catch 2;
        d.conf_floor = sys.environment.read(?f64, "TF_INKLING2_DSPARK_CONF_FLOOR", null) catch null;
        d.copy_first = sys.environment.read(bool, "TF_INKLING2_DSPARK_COPY_FIRST", true) catch true;
        d.use_hold_costs = sys.environment.read(bool, "TF_INKLING2_DSPARK_HOLD_COSTS", true) catch true;
        if (std.c.getenv("TF_INKLING2_DSPARK_DUMP")) |dump_cstr| {
            d.dump_fp = std.c.fopen(dump_cstr, "ab") orelse return error.DSparkDumpOpenFailed;
            std.log.info("dspark: DSP1 trace dump enabled -> {s}", .{std.mem.span(dump_cstr)});
        }
        var file = try core.safetensors.File.open(gpa, io, path);
        defer file.close(io);

        var part_floats: usize = 0;
        d.fuse = try d.uploadTensor(drv, &file, "fuse_weight", &.{ 4, 4 });
        d.mask = try d.uploadTensor(drv, &file, "mask_embed", &.{hidden});
        d.norm = try d.uploadTensor(drv, &file, "norm.weight", &.{hidden});
        const markov_norm = try d.uploadTensor(drv, &file, "markov_head.dflare_norm.weight", &.{hidden});
        const hidden_out = try d.uploadTensor(drv, &file, "markov_head.hidden_out.weight", &.{ hidden, rank });
        const hidden_bias = try d.uploadTensor(drv, &file, "markov_head.hidden_out.bias", &.{hidden});
        d.markov_w1 = try d.uploadTensor(drv, &file, "markov_head.markov_w1.weight", &.{ checkpoint_vocab, rank });
        d.rnn = try d.uploadTensor(drv, &file, "markov_head.rnn_proj.weight", &.{ 3 * rank, 2 * hidden + 2 * rank });
        d.rnn_bias = try d.uploadTensor(drv, &file, "markov_head.rnn_proj.bias", &.{3 * rank});

        for (&d.layers, 0..) |*l, i| {
            var b: [128]u8 = undefined;
            l.* = .{
                .dflare = try d.uploadTensor(drv, &file, try std.fmt.bufPrint(&b, "layers.{d}.dflare_norm.weight", .{i}), &.{hidden}),
                .input_norm = try d.uploadTensor(drv, &file, try std.fmt.bufPrint(&b, "layers.{d}.input_layernorm.weight", .{i}), &.{hidden}),
                .post_norm = try d.uploadTensor(drv, &file, try std.fmt.bufPrint(&b, "layers.{d}.post_attention_layernorm.weight", .{i}), &.{hidden}),
                .q_norm = try d.uploadTensor(drv, &file, try std.fmt.bufPrint(&b, "layers.{d}.self_attn.q_norm.weight", .{i}), &.{128}),
                .k_norm = try d.uploadTensor(drv, &file, try std.fmt.bufPrint(&b, "layers.{d}.self_attn.k_norm.weight", .{i}), &.{128}),
                .k_norm_ctx = try d.uploadTensor(drv, &file, try std.fmt.bufPrint(&b, "layers.{d}.self_attn.k_norm_ctx.weight", .{i}), &.{128}),
                .q = try d.uploadDense(drv, &file, try std.fmt.bufPrint(&b, "layers.{d}.self_attn.q_proj.weight", .{i}), &.{ query_width, hidden }, &part_floats),
                .k = try d.uploadDense(drv, &file, try std.fmt.bufPrint(&b, "layers.{d}.self_attn.k_proj.weight", .{i}), &.{ kv_width, hidden }, &part_floats),
                .v = try d.uploadDense(drv, &file, try std.fmt.bufPrint(&b, "layers.{d}.self_attn.v_proj.weight", .{i}), &.{ kv_width, hidden }, &part_floats),
                .k_ctx = try d.uploadDense(drv, &file, try std.fmt.bufPrint(&b, "layers.{d}.self_attn.k_proj_ctx.weight", .{i}), &.{ kv_width, hidden }, &part_floats),
                .v_ctx = try d.uploadDense(drv, &file, try std.fmt.bufPrint(&b, "layers.{d}.self_attn.v_proj_ctx.weight", .{i}), &.{ kv_width, hidden }, &part_floats),
                .o = try d.uploadDense(drv, &file, try std.fmt.bufPrint(&b, "layers.{d}.self_attn.o_proj.weight", .{i}), &.{ hidden, query_width }, &part_floats),
                .gate = try d.uploadDense(drv, &file, try std.fmt.bufPrint(&b, "layers.{d}.mlp.gate_proj.weight", .{i}), &.{ intermediate, hidden }, &part_floats),
                .up = try d.uploadDense(drv, &file, try std.fmt.bufPrint(&b, "layers.{d}.mlp.up_proj.weight", .{i}), &.{ intermediate, hidden }, &part_floats),
                .down = try d.uploadDense(drv, &file, try std.fmt.bufPrint(&b, "layers.{d}.mlp.down_proj.weight", .{i}), &.{ hidden, intermediate }, &part_floats),
            };
        }
        if (part_floats > 0) d.b12_part = try d.alloc(drv, part_floats * 4);

        const st = target.stream;
        d.dflare = try d.alloc(drv, 4 * hidden * 2);
        for (d.layers, 0..) |l, i| {
            try drv.check(drv.api.cuMemcpyDtoDAsync_v2(d.dflare + i * hidden * 2, l.dflare, hidden * 2, st.handle), "cuMemcpyDtoDAsync");
        }
        try drv.check(drv.api.cuMemcpyDtoDAsync_v2(d.dflare + 3 * hidden * 2, markov_norm, hidden * 2, st.handle), "cuMemcpyDtoDAsync");

        // GPU binary64 fold of phi[vocab, 256] and beta[vocab].
        d.phi = try d.alloc(drv, @as(usize, vocab) * rank * 2);
        const phi_buf_index = d.bufs.items.len - 1; // d.phi's slot, freed after the FP8 quant below
        d.beta = try d.alloc(drv, @as(usize, vocab) * 2);
        var tmp_unembed: ?cuda.DeviceBuffer = null;
        defer if (tmp_unembed) |*u| u.free();
        const unembed_ptr: u64 = if (target.weights.unembed != 0) target.weights.unembed else blk: {
            const mat = target.weights.unembed_b12 orelse return error.MissingUnembed;
            const kb = target.kernels.b12 orelse return error.Bf12KernelsMissing;
            tmp_unembed = try cuda.DeviceBuffer.alloc(drv, mat.bf16Bytes());
            try kb.unpack(st, &mat, tmp_unembed.?.ptr);
            break :blk tmp_unembed.?.ptr;
        };
        var fa: cuda.Args = .{};
        for ([_]u64{ unembed_ptr, hidden_out, hidden_bias, d.phi, d.beta }) |p| fa.add(p);
        fa.add(vocab);
        try go(d.k.fold_phi_beta, st, vocab, 256, &fa);

        // FP8-E4M3 phi + per-row scales + host LUT (bias-7 closed loop with the device encoder).
        d.phi_fp8 = try d.alloc(drv, @as(usize, vocab) * rank);
        d.phi_scale = try d.alloc(drv, @as(usize, vocab) * 4);
        d.f8lut = try d.alloc(drv, 256 * 4);
        var lut: [256]f32 = undefined;
        for (&lut, 0..) |*v, i| v.* = f8e4m3ToF32(@intCast(i));
        try d.bufs.items[d.bufs.items.len - 1].upload(0, std.mem.sliceAsBytes(&lut));
        {
            var qa: cuda.Args = .{};
            for ([_]u64{ d.phi, d.phi_fp8, d.phi_scale }) |p| qa.add(p);
            qa.add(vocab);
            try go(d.k.quant_phi_fp8, st, vocab, 256, &qa);
            try st.synchronize();
        }
        // The BF16 fold served only the quantizer: release its ~100 MB.
        d.bufs.items[phi_buf_index].free();
        _ = d.bufs.swapRemove(phi_buf_index);
        d.phi = 0;
        d.bytes -= @as(usize, vocab) * rank * 2;

        // Allocate scratch (sized for up to max_hold_batch concurrent streams in batched hold)
        const hb: usize = max_hold_batch;
        d.ids = try d.alloc(drv, hb * 4);
        d.resid = try d.alloc(drv, hb * block_rows * hidden * 2);
        d.next_resid = try d.alloc(drv, hb * block_rows * hidden * 2);
        d.x = try d.alloc(drv, hb * block_rows * hidden * 2);
        d.y = try d.alloc(drv, hb * block_rows * hidden * 2);
        d.q_raw = try d.alloc(drv, hb * block_rows * query_width * 2);
        d.k_raw = try d.alloc(drv, hb * block_rows * kv_width * 2);
        d.v_raw = try d.alloc(drv, hb * block_rows * kv_width * 2);
        d.q = try d.alloc(drv, hb * block_rows * query_width * 2);
        d.kb = try d.alloc(drv, hb * block_rows * kv_width * 2);
        d.vb = try d.alloc(drv, hb * block_rows * kv_width * 2);
        d.attn = try d.alloc(drv, hb * block_rows * query_width * 2);
        d.gate = try d.alloc(drv, hb * block_rows * intermediate * 2);
        d.up = try d.alloc(drv, hb * block_rows * intermediate * 2);
        d.act = try d.alloc(drv, hb * block_rows * intermediate * 2);
        d.base = try d.alloc(drv, hb * max_depth * vocab * 2);
        d.fused = try d.alloc(drv, 4 * batch_absorb_rows * hidden * 2);
        d.ck = try d.alloc(drv, batch_absorb_rows * kv_width * 2);
        d.cv = try d.alloc(drv, batch_absorb_rows * kv_width * 2);
        d.kappa = try d.alloc(drv, hb * max_depth * rnn_width * 4);
        d.state = try d.alloc(drv, hb * rank * 2);
        d.next_state = try d.alloc(drv, hb * rank * 2);
        d.z = try d.alloc(drv, hb * rnn_width * 2);
        d.lambda = try d.alloc(drv, hb * rank * 2);
        d.logits = try d.alloc(drv, hb * vocab * 2);
        d.id_out = try d.alloc(drv, 4);
        d.draft_tokens = try d.alloc(drv, hb * 8 * 4);
        d.draft_probs = try d.alloc(drv, hb * 8 * 4);
        d.draft_margins = try d.alloc(drv, hb * 8 * 4);
        if (d.dump_fp != null) {
            d.dump_top_toks = try d.alloc(drv, hb * max_depth * 16 * 4);
            d.dump_top_vals = try d.alloc(drv, hb * max_depth * 16 * 4);
            d.dump_lam = try d.alloc(drv, hb * max_depth * rank * 2);
        }
        d.hold_meta = try d.alloc(drv, hb * 16);
        d.lane_desc = try d.alloc(drv, max_hold_batch * @sizeOf(LaneDesc));
        d.rule = try d.alloc(drv, hb * @sizeOf(cuda.sample.Rule));
        d.meta = try d.alloc(drv, hb * 16);
        d.scores = try d.alloc(drv, 256);
        d.split_o = try d.alloc(drv, hb * block_rows * 24 * 16 * 128 * 4);
        d.split_ml = try d.alloc(drv, hb * block_rows * 24 * 16 * 2 * 4);
        try st.synchronize();
        try d.calibrate(io, st, drv);
        return d;
    }

    /// Startup timing: target window costs for widths 1..8, then the draft step cost from 3 timed
    /// 7-step recurrent_dev+logits+top1 rounds over scratch (results discarded). Resets the target after.
    fn calibrate(d: *Drafter, io: std.Io, st: cuda.Stream, drv: *const cuda.Driver) !void {
        const dummy = [8]u32{ 1000, 2741, 5183, 8912, 14207, 23451, 41903, 78219 };
        for (1..9) |w| {
            const width: usize = w;
            const record = width > 1;
            for (0..2) |_| try d.target.model.forward(&d.target.kernels, st, dummy[0..width], 0, 0, width, record, 0);
            try st.synchronize();
            const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
            for (0..4) |_| try d.target.model.forward(&d.target.kernels, st, dummy[0..width], 0, 0, width, record, 0);
            try st.synchronize();
            const t1 = std.Io.Clock.awake.now(io).toNanoseconds();
            d.costs[width - 1] = .{ .width = @intCast(width), .ms = @as(f64, @floatFromInt(t1 - t0)) / 1e6 / 4.0 };
        }
        d.n_costs = 8;
        if (d.target.model.segments >= 2) {
            const BatchedSeg = @import("loop_engine.zig").Model.BatchedSeg;
            var dummy128: [128]u32 = undefined;
            for (&dummy128, 0..) |*v, i| v.* = @intCast(1000 + (i * 7919) % 50000);
            const max_w = d.target.model.segments * 8;
            for ([_]usize{ 16, 32, 64, 128 }) |w| {
                if (w > max_w) break;
                const n_segs = @min(d.target.model.segments, @max(@as(usize, 2), (w + 7) / 8));
                var segs: [16]BatchedSeg = undefined;
                var off: usize = 0;
                const base_take = w / n_segs;
                const rem = w % n_segs;
                for (0..n_segs) |si| {
                    const take = base_take + @intFromBool(si < rem);
                    segs[si] = .{ .tokens = dummy128[off .. off + take], .p0 = 0, .slot = @intCast(si) };
                    off += take;
                }
                try d.target.model.forwardBatched(&d.target.kernels, st, segs[0..n_segs], w, true, 0);
                try st.synchronize();
                const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
                for (0..3) |_| try d.target.model.forwardBatched(&d.target.kernels, st, segs[0..n_segs], w, true, 0);
                try st.synchronize();
                const t1 = std.Io.Clock.awake.now(io).toNanoseconds();
                d.costs[d.n_costs] = .{ .width = @intCast(w), .ms = @as(f64, @floatFromInt(t1 - t0)) / 1e6 / 3.0 };
                d.n_costs += 1;
            }
        }
        // One draft-step round = 7 recurrent_dev+logits+top1 steps on a fixed valid token; timing only.
        var tok: u32 = 1;
        try cuda.copy_source.enqueue(drv, d.draft_tokens, std.mem.asBytes(&tok), st);
        try drv.check(drv.api.cuMemsetD8Async(d.state, 0, rank * 2, st.handle), "cuMemsetD8Async");
        try drv.check(drv.api.cuMemsetD8Async(d.base, 0, max_depth * vocab * 2, st.handle), "cuMemsetD8Async");
        try drv.check(drv.api.cuMemsetD8Async(d.kappa, 0, max_depth * rnn_width * 4, st.handle), "cuMemsetD8Async");
        var step_ns: i96 = 0;
        for (0..4) |rep| {
            const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
            var second_bank = false;
            for (0..max_depth) |step| {
                const cur = if (second_bank) d.next_state else d.state;
                const nxt = if (second_bank) d.state else d.next_state;
                var a: cuda.Args = .{};
                for ([_]u64{ cur, d.kappa + step * rnn_width * 4, d.rnn, d.markov_w1, d.z, nxt, d.lambda }) |p| a.add(p);
                a.add(d.draft_tokens);
                try go(d.k.recurrent_dev, st, 1, 256, &a);
                var b: cuda.Args = .{};
                for ([_]u64{ d.base + step * vocab * 2, d.phi_fp8, d.phi_scale, d.beta, d.f8lut, d.lambda, d.logits }) |p| b.add(p);
                b.add(vocab);
                try go(d.k.logits_fp8, st, (vocab + 7) / 8, 256, &b);
                var c: cuda.Args = .{};
                c.add(d.logits);
                c.add(@as(f32, 0.0));
                c.add(@as(u64, 0));
                c.add(@as(u32, 1));
                c.add(vocab);
                c.add(d.id_out);
                try go(d.k.top1_gumbel, st, 1, 256, &c);
                second_bank = !second_bank;
            }
            try st.synchronize();
            if (rep > 0) step_ns += std.Io.Clock.awake.now(io).toNanoseconds() - t0;
        }
        d.step_ms = @as(f64, @floatFromInt(step_ns)) / 1e6 / 21.0;

        // Calibrate DSpark's real hold block pass (S=1, and S=4 if segments >= 4) at depth=1 and depth=7.
        d.n_hold_costs = 0;
        const hold_ids = [4]i32{ 1, 1, 1, 1 };
        try cuda.copy_source.enqueue(drv, d.ids, std.mem.sliceAsBytes(&hold_ids), st);
        const hold_streams: []const u32 = if (d.target.model.segments >= 4) &.{ 1, 4 } else &.{1};
        for (hold_streams) |hs| {
            const total_rows = @as(usize, hs) * block_rows;
            for ([_]u32{ 1, max_depth }) |hd| {
                for (0..2) |rep| {
                    if (rep == 1) try st.synchronize();
                    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
                    try ops.embed(&d.target.kernels.glue, st, d.ids, hs, d.target.weights.embed, d.target.weights.embed_norm, d.y, hidden, 1e-6);
                    for (d.layers) |layer| {
                        try dense(d, st, d.x, layer.q, d.q_raw, total_rows, query_width, hidden);
                        try dense(d, st, d.x, layer.k, d.k_raw, total_rows, kv_width, hidden);
                        try dense(d, st, d.x, layer.v, d.v_raw, total_rows, kv_width, hidden);
                        try dense(d, st, d.attn, layer.o, d.y, total_rows, hidden, query_width);
                        try dense(d, st, d.x, layer.gate, d.gate, total_rows, intermediate, hidden);
                        try dense(d, st, d.x, layer.up, d.up, total_rows, intermediate, hidden);
                        try dense(d, st, d.act, layer.down, d.y, total_rows, hidden, intermediate);
                    }
                    try d.target.model.dense(&d.target.kernels, st, d.y, d.target.weights.unembed, d.target.weights.unembed_b12, d.base, @as(usize, hs) * hd, vocab, hidden, true, false);
                    try st.synchronize();
                    if (rep == 1) {
                        const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e6 + @as(f64, @floatFromInt(hd)) * d.step_ms;
                        d.hold_costs[d.n_hold_costs] = .{ .streams = hs, .depth = hd, .ms = ms };
                        d.n_hold_costs += 1;
                    }
                }
            }
        }

        try d.target.model.reset(st);
        try st.synchronize();
        d.target.model.clearGraphs();
        std.log.info("dspark: calibrated step_ms={d:.3}ms", .{d.step_ms});
        for (d.costs[0..d.n_costs]) |c| std.log.info("dspark: calibrated width={d} ms={d:.3}", .{ c.width, c.ms });
        for (d.hold_costs[0..d.n_hold_costs]) |hc| std.log.info("dspark: calibrated hold_cost streams={d} depth={d} ms={d:.3}", .{ hc.streams, hc.depth, hc.ms });
    }

    pub fn deinit(d: *Drafter) void {
        if (d.dump_fp) |fp| _ = std.c.fclose(fp);
        d.clearHoldGraphs();
        var it = d.lanes.valueIterator();
        while (it.next()) |ln| {
            if (ln.pending_dump) |pd| d.gpa.destroy(pd);
            ln.deinit();
        }
        d.lanes.deinit(d.gpa);
        for (d.pool.items) |*ln| {
            if (ln.pending_dump) |pd| d.gpa.destroy(pd);
            ln.deinit();
        }
        d.pool.deinit(d.gpa);
        for (d.bufs.items) |*b| b.free();
        d.bufs.deinit(d.gpa);
        d.k.unload();
        d.gpa.destroy(d);
    }

    /// Fixed device bytes (weights + scratch, committed once at init).
    pub fn fixedBytes(d: *const Drafter) usize {
        return d.bytes;
    }

    /// Per-lane device bytes (KV rings + newest row), matching factsFn's stream_bytes.
    pub fn laneBytes(d: *const Drafter) usize {
        const global_bytes = if (d.kv_int8) d.capacity * (kv_width + 8 * 4) * 2 else d.capacity * kv_width * 4;
        return local_ring * 2 * kv_width * 4 + global_bytes + hidden * 2;
    }

    /// Commit `n` lane caches up front so admission-time memory is really resident (no late OOMs).
    /// Lane states are unbound until openFn assigns them to streams.
    pub fn reserve(d: *Drafter, n: u32) !void {
        while (d.pool.items.len + d.lanes.count() < n) {
            var ln = try LaneState.init(d.target.model.d, d.capacity, d.kv_int8);
            errdefer ln.deinit();
            try d.pool.append(d.gpa, ln);
        }
        std.log.info("dspark: reserved {d} lanes (fixed={d}B, lane={d}B)", .{ n, d.fixedBytes(), d.laneBytes() });
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{
            .kind = "DSpark",
            .ptr = d,
            .vtable = &.{
                .taps = tapsFn,
                .facts = factsFn,
                .open = openFn,
                .absorb = absorbFn,
                .hold = holdFn,
                .held = heldFn,
                .release = releaseFn,
                .chances = chancesFn,
            },
        };
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }
    fn tapsFn(_: *anyopaque) []const u32 {
        return &tap_list;
    }
    fn factsFn(ptr: *anyopaque) dr.Facts {
        const d = cast(ptr);
        return .{
            .depth = 7,
            .step_ms = d.step_ms,
            .batched = true,
            .plain_guard = true,
            .fixed_block = true,
            .chances = d.use_chances,
            .confidence_floor = d.conf_floor,
            .copy_first = d.copy_first,
            .slack = d.slack,
            .hold_costs = if (d.use_hold_costs) d.hold_costs[0..d.n_hold_costs] else &.{},
            .stream_bytes = d.laneBytes(),
            .window_costs = d.costs[0..d.n_costs],
            .lane_costs = d.costs[0..d.n_costs],
        };
    }
    fn chancesFn(ptr: *anyopaque, stream: *Stream, out: []f64) anyerror!bool {
        const d = cast(ptr);
        if (!d.use_chances) return false;
        const ln = d.lanes.getPtr(stream) orelse return error.DraftStreamUnknown;
        if (!ln.has_probs or ln.held_len == 0 or out.len > ln.held_len) return false;
        var cum: f64 = 1.0;
        for (out, 0..) |*o, k| {
            if (d.prob_power == 1.0 and d.use_landing) {
                const own = @log(std.math.clamp(@as(f64, ln.held_probs[k]), 1e-6, 1.0));
                const cal = d.landing.chance(stream.sampling != null, own);
                cum *= std.math.clamp(cal, 0.0, 1.0);
            } else {
                const p = std.math.clamp(@as(f64, ln.held_probs[k]), 0.0, 1.0);
                cum *= if (d.prob_power == 1.0) p else std.math.pow(f64, p, d.prob_power);
            }
            o.* = cum;
        }
        return true;
    }
    fn openFn(ptr: *anyopaque, stream: *Stream) anyerror!void {
        const d = cast(ptr);
        if (d.lanes.contains(stream)) return error.DraftStreamOpen;
        if (d.pool.pop()) |ln| {
            var pooled = ln;
            pooled.from = 0;
            pooled.end = 0;
            pooled.held_len = 0;
            pooled.verified_len = 0;
            pooled.has_probs = false;
            if (pooled.pending_dump) |pd| {
                d.gpa.destroy(pd);
                pooled.pending_dump = null;
            }
            errdefer d.pool.append(d.gpa, pooled) catch {
                d.purgeHoldGraphs(pooled.k[0].ptr);
                pooled.deinit();
            };
            try d.lanes.put(d.gpa, stream, pooled);
        } else {
            var ln = try LaneState.init(d.target.model.d, d.capacity, d.kv_int8);
            errdefer ln.deinit();
            try d.lanes.put(d.gpa, stream, ln);
        }
    }
    fn releaseFn(ptr: *anyopaque, stream: *Stream) void {
        const d = cast(ptr);
        if (d.lanes.fetchRemove(stream)) |entry| {
            var ln = entry.value;
            ln.from = 0;
            ln.end = 0;
            ln.held_len = 0;
            ln.verified_len = 0;
            ln.has_probs = false;
            if (ln.pending_dump) |pd| {
                d.gpa.destroy(pd);
                ln.pending_dump = null;
            }
            d.pool.append(d.gpa, ln) catch {
                d.purgeHoldGraphs(ln.k[0].ptr);
                ln.deinit();
            };
        }
    }
    fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const d = cast(ptr);
        for (streams, out) |stream, tokens| {
            const ln = d.lanes.getPtr(stream) orelse return error.DraftStreamUnknown;
            if (tokens.len > ln.held_len) return error.DraftHeldFewer;
            @memcpy(tokens, ln.held[0..tokens.len]);
            ln.verified_len = @intCast(tokens.len);
        }
    }

    fn flushDump(d: *Drafter, ln: *LaneState, item: dr.Absorb) void {
        const fp = d.dump_fp orelse return;
        if (item.prompt or item.follow.len == 0) {
            if (ln.pending_dump) |pd| {
                d.gpa.destroy(pd);
                ln.pending_dump = null;
            }
            return;
        }
        if (ln.pending_dump) |pd| {
            const kept: u8 = @min(@as(u8, @intCast(item.features.rows -| 1)), pd.last_n);
            const new_pending: u32 = item.follow[item.follow.len - 1];
            writeDsp1Record(fp, pd, kept, new_pending);
            d.gpa.destroy(pd);
            ln.pending_dump = null;
        }
    }

    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        if (items.len == 0) return;
        for (items) |item| {
            if (d.lanes.getPtr(item.stream)) |ln| {
                // One observation per stream per absorb: the verified window's kept rows land, the first
                // miss past them does not. Runs before flushDump/ln.end updates on both the batched and
                // the per-stream path; the reset makes a second absorb without an intervening held a no-op.
                if (!item.prompt and ln.has_probs and ln.verified_len > 0 and item.features.rows > 0) {
                    const sampled = item.stream.sampling != null;
                    const vlen = @min(ln.verified_len, ln.held_len);
                    const kept = @min(item.features.rows - 1, vlen);
                    for (0..kept) |k| {
                        const own = @log(std.math.clamp(@as(f64, ln.held_probs[k]), 1e-6, 1.0));
                        d.landing.observe(sampled, own, true);
                    }
                    if (kept < vlen) {
                        const own = @log(std.math.clamp(@as(f64, ln.held_probs[kept]), 1e-6, 1.0));
                        d.landing.observe(sampled, own, false);
                    }
                }
                ln.verified_len = 0;
                ln.has_probs = false;
                d.flushDump(ln, item);
            }
        }
        if (d.absorbBatchable(items)) {
            try d.absorbBatch(items);
            return;
        }
        const st = d.target.stream;
        const row_bytes = 4 * hidden * 2;
        for (items) |item| {
            const ln = d.lanes.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            const features = item.features;
            if (features.space != .device or features.dtype != .bf16 or features.row_bytes != row_bytes) return error.DraftFeaturesUnsupported;
            if (item.rows != null) return error.TreesNotBuilt;
            if (features.rows == 0 or features.rows > 1048576) return error.DraftFeaturesUnsupported;
            if (item.start < ln.end) return error.DraftContextGap;
            if (item.start > ln.end) ln.from = item.start;
            const end = item.start + features.rows;
            if (end > d.capacity) return error.DraftContextGap;
            var pos = item.start;
            while (pos < end) {
                const n: u32 = @intCast(@min(absorb_rows, end - pos));
                const src = features.buffer + features.offset + (pos - item.start) * row_bytes;
                var fa: cuda.Args = .{};
                for ([_]u64{ src, d.fuse, d.dflare, d.fused, ln.newest.ptr }) |p| fa.add(p);
                fa.add(n);
                try go(d.k.fuse_taps, st, 4 * n, 256, &fa);
                for (d.layers, 0..) |layer, i| {
                    const c = d.fused + i * n * hidden * 2;
                    try dense(d, st, c, layer.k_ctx, d.ck, n, kv_width, hidden);
                    try dense(d, st, c, layer.v_ctx, d.cv, n, kv_width, hidden);
                    var ca: cuda.Args = .{};
                    for ([_]u64{ d.ck, d.cv, layer.k_norm_ctx, ln.k[i].ptr, ln.v[i].ptr }) |p| ca.add(p);
                    ca.add(@as(u32, @intCast(pos)));
                    ca.add(n);
                    ca.add(@as(u32, @intCast(ln.rings[i])));
                    try go(if (d.kv_int8 and i == 1) d.k.context_kv_i8 else d.k.context_kv, st, 8 * n, 128, &ca);
                }
                pos += n;
            }
            ln.end = end;
            ln.held_len = 0;
        }
    }

    /// Read-only batchability probe: same errors as the per-stream loop would raise, no mutation.
    /// Batched iff 2..16 streams and every item fits in one batch_absorb_rows shot.
    fn absorbBatchable(d: *const Drafter, items: []const dr.Absorb) bool {
        if (items.len < 2 or items.len > max_hold_batch) return false;
        const row_bytes = 4 * hidden * 2;
        var total: usize = 0;
        for (items, 0..) |item, idx| {
            for (items[0..idx]) |prev| if (prev.stream == item.stream) return false;
            const ln = d.lanes.getPtr(item.stream) orelse return false;
            const features = item.features;
            if (features.space != .device or features.dtype != .bf16 or features.row_bytes != row_bytes) return false;
            if (item.rows != null) return false;
            if (features.rows == 0 or features.rows > batch_absorb_rows) return false;
            if (item.start < ln.end) return false;
            if (item.start + features.rows > d.capacity) return false;
            total += features.rows;
            if (total > batch_absorb_rows) return false;
        }
        return true;
    }

    /// One fuse_taps_batch over all streams' rows, one k_ctx/v_ctx dense per layer across total_rows
    /// (1x weight read), one context_kv_batch scatter per layer. Bit-identical to the per-stream loop.
    fn absorbBatch(d: *Drafter, items: []const dr.Absorb) !void {
        const st = d.target.stream;
        const S = items.len;
        var descs: [max_hold_batch]LaneDesc = undefined;
        var total: u32 = 0;
        for (items, 0..) |item, s| {
            const ln = d.lanes.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            const end = item.start + item.features.rows;
            if (item.start > ln.end) ln.from = item.start;
            descs[s] = .{
                .k = .{ ln.k[0].ptr, ln.k[1].ptr, ln.k[2].ptr },
                .v = .{ ln.v[0].ptr, ln.v[1].ptr, ln.v[2].ptr },
                .newest = ln.newest.ptr,
                .src = item.features.buffer + item.features.offset,
                .row0 = total,
                .n = @intCast(item.features.rows),
                .base_pos = @intCast(item.start),
                .ring = .{ @intCast(ln.rings[0]), @intCast(ln.rings[1]), @intCast(ln.rings[2]) },
                ._pad = .{ 0, 0 },
            };
            total += @as(u32, @intCast(item.features.rows));
            ln.end = end;
            ln.held_len = 0;
        }
        try cuda.copy_source.enqueue(d.target.model.d, d.lane_desc, std.mem.sliceAsBytes(descs[0..S]), st);
        {
            var fa: cuda.Args = .{};
            for ([_]u64{ d.lane_desc, d.fuse, d.dflare, d.fused }) |p| fa.add(p);
            fa.add(@as(u32, @intCast(S)));
            fa.add(total);
            try go(d.k.fuse_taps_batch, st, 4 * total, 256, &fa);
        }
        for (d.layers, 0..) |layer, i| {
            const c = d.fused + i * total * hidden * 2;
            try dense(d, st, c, layer.k_ctx, d.ck, total, kv_width, hidden);
            try dense(d, st, c, layer.v_ctx, d.cv, total, kv_width, hidden);
            var ca: cuda.Args = .{};
            for ([_]u64{ d.ck, d.cv, layer.k_norm_ctx, d.lane_desc }) |p| ca.add(p);
            ca.add(@as(u32, @intCast(i)));
            ca.add(@as(u32, @intCast(S)));
            ca.add(total);
            try go(if (d.kv_int8 and i == 1) d.k.context_kv_batch_i8 else d.k.context_kv_batch, st, 8 * total, 128, &ca);
        }
    }

    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        const drv = d.target.model.d;
        const st = d.target.stream;
        if (items.len == 0) return;
        if (items.len == 1) {
            const item = items[0];
            const ln = d.lanes.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            if (item.depth == 0 or item.position == 0) return error.DraftDepth;
            const anchor = item.position - 1;
            if (ln.end != anchor) return error.DraftContextGap;
            const depth: u32 = @min(item.depth, max_depth);
            const rule_opt = cuda.sample.ruleFor(item.sampling);
            // Uploads before any capture: ids, the chain's first token, the _dev scalars.
            const id_val: i32 = @intCast(item.pending);
            try cuda.copy_source.enqueue(drv, d.ids, std.mem.asBytes(&id_val), st);
            var first: u32 = item.pending;
            try cuda.copy_source.enqueue(drv, d.draft_tokens, std.mem.asBytes(&first), st);
            var meta = [4]u32{ @intCast(anchor), @intCast(ln.end), @intCast(ln.from), @intCast(item.position) };
            try cuda.copy_source.enqueue(drv, d.hold_meta, std.mem.asBytes(&meta), st);
            if (rule_opt == null and d.dump_fp == null and holdGraphsEnabled() and !d.graphs_disabled) {
                if (d.holdGraph(st, ln, item, depth)) |exec| {
                    try exec.launchOn(st);
                } else |err| {
                    std.log.warn("dspark: hold graph capture failed ({s}); falling back to eager holds", .{@errorName(err)});
                    d.graphs_disabled = true;
                    try d.holdCore(st, drv, ln, item, depth, null);
                }
            } else {
                if (rule_opt) |r| try cuda.copy_source.enqueue(drv, d.rule, std.mem.asBytes(&r), st);
                try d.holdCore(st, drv, ln, item, depth, rule_opt);
            }
            try drv.check(drv.api.cuMemcpyDtoHAsync_v2(&ln.held, d.draft_tokens + 4, @as(usize, depth) * 4, st.handle), "cuMemcpyDtoHAsync_v2");
            if (rule_opt == null) {
                try drv.check(drv.api.cuMemcpyDtoHAsync_v2(&ln.held_probs, d.draft_probs, @as(usize, depth) * 4, st.handle), "cuMemcpyDtoHAsync_v2");
                try drv.check(drv.api.cuMemcpyDtoHAsync_v2(&ln.held_margins, d.draft_margins, @as(usize, depth) * 4, st.handle), "cuMemcpyDtoHAsync_v2");
            }
            var dump_pd: ?*PendingDump = null;
            if (d.dump_fp != null and rule_opt == null) {
                if (ln.pending_dump) |old| d.gpa.destroy(old);
                const pd = try d.gpa.create(PendingDump);
                pd.* = .{ .last_pos = anchor, .last_pending = item.pending, .last_n = @intCast(depth) };
                try drv.check(drv.api.cuMemcpyDtoHAsync_v2(&pd.top_toks, d.dump_top_toks, @as(usize, depth) * 16 * 4, st.handle), "cuMemcpyDtoHAsync_v2");
                try drv.check(drv.api.cuMemcpyDtoHAsync_v2(&pd.top_vals, d.dump_top_vals, @as(usize, depth) * 16 * 4, st.handle), "cuMemcpyDtoHAsync_v2");
                try drv.check(drv.api.cuMemcpyDtoHAsync_v2(&pd.x, d.x + hidden * 2, @as(usize, depth) * hidden * 2, st.handle), "cuMemcpyDtoHAsync_v2");
                try drv.check(drv.api.cuMemcpyDtoHAsync_v2(&pd.lam, d.dump_lam, @as(usize, depth) * rank * 2, st.handle), "cuMemcpyDtoHAsync_v2");
                dump_pd = pd;
            }
            try st.synchronize();
            ln.held_len = depth;
            ln.verified_len = 0;
            ln.has_probs = (rule_opt == null);
            if (dump_pd) |pd| {
                @memcpy(pd.drafts[0..depth], ln.held[0..depth]);
                @memcpy(pd.probs[0..depth], ln.held_probs[0..depth]);
                @memcpy(pd.margins[0..depth], ln.held_margins[0..depth]);
                ln.pending_dump = pd;
            }
            return;
        }
        var off: usize = 0;
        while (off < items.len) {
            const take = @min(items.len - off, max_hold_batch);
            try d.holdBatch(st, drv, items[off .. off + take]);
            off += take;
        }
    }

    fn holdBatch(d: *Drafter, st: cuda.Stream, drv: *const cuda.Driver, items: []const dr.Hold) !void {
        const S = items.len;
        var lanes_ptrs: [max_hold_batch]*LaneState = undefined;
        var depths: [max_hold_batch]u32 = undefined;
        var rules: [max_hold_batch]?cuda.sample.Rule = undefined;
        var ids: [max_hold_batch]i32 = undefined;
        var tokens_init: [max_hold_batch * 8]u32 = @splat(0);
        var metas: [max_hold_batch][4]u32 = undefined;
        var descs: [max_hold_batch]LaneDesc = undefined;
        var max_d: u32 = 0;
        var sampled: u16 = 0;
        for (items, 0..) |item, s| {
            const ln = d.lanes.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            if (item.depth == 0 or item.position == 0) return error.DraftDepth;
            const anchor = item.position - 1;
            if (ln.end != anchor) return error.DraftContextGap;
            const depth: u32 = @min(item.depth, max_depth);
            lanes_ptrs[s] = ln;
            depths[s] = depth;
            max_d = @max(max_d, depth);
            rules[s] = cuda.sample.ruleFor(item.sampling);
            if (rules[s] != null) sampled |= @as(u16, 1) << @intCast(s);
            ids[s] = @intCast(item.pending);
            tokens_init[s * 8] = item.pending;
            metas[s] = .{ @intCast(anchor), @intCast(ln.end), @intCast(ln.from), @intCast(item.position) };
            descs[s] = .{
                .k = .{ ln.k[0].ptr, ln.k[1].ptr, ln.k[2].ptr },
                .v = .{ ln.v[0].ptr, ln.v[1].ptr, ln.v[2].ptr },
                .newest = ln.newest.ptr,
                .src = 0,
                .row0 = 0,
                .n = 0,
                .base_pos = 0,
                .ring = .{ @intCast(ln.rings[0]), @intCast(ln.rings[1]), @intCast(ln.rings[2]) },
                ._pad = .{ 0, 0 },
            };
        }
        // Uploads before any capture: ids, draft_tokens, hold_meta, per-stream rules, lane descs.
        const rule_sz = @sizeOf(cuda.sample.Rule);
        try cuda.copy_source.enqueue(drv, d.ids, std.mem.sliceAsBytes(ids[0..S]), st);
        try cuda.copy_source.enqueue(drv, d.draft_tokens, std.mem.sliceAsBytes(tokens_init[0 .. S * 8]), st);
        try cuda.copy_source.enqueue(drv, d.hold_meta, std.mem.sliceAsBytes(metas[0..S]), st);
        try cuda.copy_source.enqueue(drv, d.lane_desc, std.mem.sliceAsBytes(descs[0..S]), st);
        for (0..S) |s| {
            if (rules[s]) |r| try cuda.copy_source.enqueue(drv, d.rule + s * rule_sz, std.mem.asBytes(&r), st);
        }
        if (sampled == 0 and d.dump_fp == null and holdGraphsEnabled() and !d.graphs_disabled) {
            if (d.holdBatchGraph(st, items, max_d)) |exec| {
                try exec.launchOn(st);
            } else |err| {
                std.log.warn("dspark: batched hold graph capture failed ({s}); falling back to eager", .{@errorName(err)});
                d.graphs_disabled = true;
                try d.holdBatchCore(st, drv, items, 0);
            }
        } else {
            try d.holdBatchCore(st, drv, items, sampled);
        }
        var host_toks: [max_hold_batch * 8]u32 = undefined;
        var host_probs: [max_hold_batch * 8]f32 = undefined;
        var host_margins: [max_hold_batch * 8]f32 = undefined;
        try drv.check(drv.api.cuMemcpyDtoHAsync_v2(&host_toks, d.draft_tokens, S * 8 * 4, st.handle), "cuMemcpyDtoHAsync_v2");
        if (sampled == 0) {
            try drv.check(drv.api.cuMemcpyDtoHAsync_v2(&host_probs, d.draft_probs, S * 8 * 4, st.handle), "cuMemcpyDtoHAsync_v2");
            try drv.check(drv.api.cuMemcpyDtoHAsync_v2(&host_margins, d.draft_margins, S * 8 * 4, st.handle), "cuMemcpyDtoHAsync_v2");
        }
        try st.synchronize();
        for (0..S) |s| {
            @memcpy(lanes_ptrs[s].held[0..depths[s]], host_toks[s * 8 + 1 .. s * 8 + 1 + depths[s]]);
            lanes_ptrs[s].held_len = depths[s];
            lanes_ptrs[s].verified_len = 0;
            lanes_ptrs[s].has_probs = (sampled == 0);
            if (sampled == 0) {
                @memcpy(lanes_ptrs[s].held_probs[0..depths[s]], host_probs[s * 8 .. s * 8 + depths[s]]);
                @memcpy(lanes_ptrs[s].held_margins[0..depths[s]], host_margins[s * 8 .. s * 8 + depths[s]]);
            }
        }
    }

    /// The batched hold's device launches, no host reads and no uploads: captured once per
    /// (S, max_d, lane set) for all-greedy batches, run eagerly otherwise. `sampled` bits mark the
    /// streams that draw from a rule (eager only); `items` is only touched for those streams.
    fn holdBatchCore(d: *Drafter, st: cuda.Stream, drv: *const cuda.Driver, items: []const dr.Hold, sampled: u16) !void {
        const S = items.len;
        var max_d: u32 = 0;
        for (items) |item| max_d = @max(max_d, @min(item.depth, max_depth));
        const rule_sz = @sizeOf(cuda.sample.Rule);
        const total_rows = S * block_rows;
        try ops.embed(&d.target.kernels.glue, st, d.ids, S, d.target.weights.embed, d.target.weights.embed_norm, d.y, hidden, 1e-6);
        {
            var a: cuda.Args = .{};
            for ([_]u64{ d.y, d.mask, d.x }) |p| a.add(p);
            a.add(@as(u32, @intCast(S)));
            try go(d.k.init_rows_batch, st, @intCast(total_rows), 256, &a);
        }
        {
            var a: cuda.Args = .{};
            for ([_]u64{ d.x, 0, d.layers[0].input_norm, d.resid, d.x }) |p| a.add(p);
            a.add(@as(u32, @intCast(total_rows)));
            a.add(@as(u32, 1));
            try go(d.k.residual_norm, st, @intCast(total_rows), 256, &a);
        }
        for (d.layers, 0..) |layer, i| {
            try dense(d, st, d.x, layer.q, d.q_raw, total_rows, query_width, hidden);
            try dense(d, st, d.x, layer.k, d.k_raw, total_rows, kv_width, hidden);
            try dense(d, st, d.x, layer.v, d.v_raw, total_rows, kv_width, hidden);
            {
                var a: cuda.Args = .{};
                for ([_]u64{ d.q_raw, d.k_raw, d.v_raw, layer.q_norm, layer.k_norm, d.q, d.kb, d.vb, d.hold_meta }) |p| a.add(p);
                try go(d.k.block_qkv_dev_batch, st, @intCast(S * block_rows * 40), 128, &a);
            }
            if (d.kv_int8 and i == 1) {
                var a: cuda.Args = .{};
                for ([_]u64{ d.q, d.kb, d.vb, d.lane_desc, d.split_o, d.split_ml, d.hold_meta }) |p| a.add(p);
                try go(d.k.attention_splitk_dev_batch_i8, st, @intCast(S * 8 * 16), 256, &a);
                var m: cuda.Args = .{};
                for ([_]u64{ d.split_o, d.split_ml, d.attn }) |p| m.add(p);
                try go(d.k.attention_splitk_merge, st, @intCast(S * block_rows * 24), 128, &m);
            } else {
                var a: cuda.Args = .{};
                for ([_]u64{ d.q, d.kb, d.vb, d.lane_desc, d.scores, d.attn, d.hold_meta }) |p| a.add(p);
                a.add(@as(u32, @intCast(i)));
                a.add(@as(u32, @intFromBool(i != 1)));
                a.add(@as(u32, @intCast(d.capacity + block_rows)));
                try go(d.k.attention_dev_batch, st, @intCast(S * block_rows * 24), 128, &a);
            }
            try dense(d, st, d.attn, layer.o, d.y, total_rows, hidden, query_width);
            {
                var a: cuda.Args = .{};
                for ([_]u64{ d.y, d.resid, layer.post_norm, d.next_resid, d.x }) |p| a.add(p);
                a.add(@as(u32, @intCast(total_rows)));
                a.add(@as(u32, 0));
                try go(d.k.residual_norm, st, @intCast(total_rows), 256, &a);
            }
            try dense(d, st, d.x, layer.gate, d.gate, total_rows, intermediate, hidden);
            try dense(d, st, d.x, layer.up, d.up, total_rows, intermediate, hidden);
            {
                var a: cuda.Args = .{};
                for ([_]u64{ d.gate, d.up, d.act }) |p| a.add(p);
                a.add(@as(u32, @intCast(total_rows * intermediate)));
                try go(d.k.swiglu, st, @intCast((total_rows * intermediate + 255) / 256), 256, &a);
            }
            try dense(d, st, d.act, layer.down, d.y, total_rows, hidden, intermediate);
            const next_norm = if (i + 1 < 3) d.layers[i + 1].input_norm else d.norm;
            {
                var a: cuda.Args = .{};
                for ([_]u64{ d.y, d.next_resid, next_norm, d.resid, d.x }) |p| a.add(p);
                a.add(@as(u32, @intCast(total_rows)));
                a.add(@as(u32, 0));
                try go(d.k.residual_norm, st, @intCast(total_rows), 256, &a);
            }
        }
        {
            var a: cuda.Args = .{};
            a.add(d.x);
            a.add(d.y);
            a.add(@as(u32, @intCast(S)));
            a.add(max_d);
            try go(d.k.pack_draft_x, st, @intCast(S * max_d), 256, &a);
        }
        try d.target.model.dense(&d.target.kernels, st, d.y, d.target.weights.unembed, d.target.weights.unembed_b12, d.base, S * max_d, vocab, hidden, true, false);
        {
            var a: cuda.Args = .{};
            for ([_]u64{ d.x, d.lane_desc, d.rnn, d.rnn_bias, d.kappa }) |p| a.add(p);
            a.add(@as(u32, @intCast(S)));
            a.add(max_d);
            try go(d.k.kappa_batch, st, @intCast((S * max_d * rnn_width + 255) / 256), 256, &a);
        }
        try drv.check(drv.api.cuMemsetD8Async(d.state, 0, S * rank * 2, st.handle), "cuMemsetD8Async");
        var second_bank = false;
        for (0..max_d) |step| {
            const cur_base = if (second_bank) d.next_state else d.state;
            const nxt_base = if (second_bank) d.state else d.next_state;
            {
                var a: cuda.Args = .{};
                for ([_]u64{ cur_base, d.kappa, d.rnn, d.markov_w1, d.z, nxt_base, d.lambda, d.draft_tokens }) |p| a.add(p);
                a.add(max_d);
                a.add(@as(u32, @intCast(step)));
                try go(d.k.recurrent_dev_batch, st, @intCast(S), 256, &a);
            }
            {
                var a: cuda.Args = .{};
                for ([_]u64{ d.base, d.phi_fp8, d.phi_scale, d.beta, d.f8lut, d.lambda, d.logits }) |p| a.add(p);
                a.add(vocab);
                a.add(@as(u32, @intCast(S)));
                a.add(max_d);
                a.add(@as(u32, @intCast(step)));
                try go(d.k.logits_fp8_batch, st, (vocab + 7) / 8, 256, &a);
            }
            if (sampled == 0) {
                var a: cuda.Args = .{};
                for ([_]u64{ d.logits, d.draft_tokens, d.draft_probs, d.draft_margins }) |p| a.add(p);
                a.add(@as(u32, @intCast(step)));
                a.add(@as(u32, @intCast(S)));
                try go(d.k.top1_prob_dev_batch, st, @intCast(S), 256, &a);
            } else for (0..S) |s| {
                const lg_ptr = d.logits + s * vocab * 2;
                const tok_ptr = d.draft_tokens + (s * 8 + step + 1) * 4;
                if (sampled & (@as(u16, 1) << @intCast(s)) != 0) {
                    const metadata = [4]i32{ @intCast(items[s].position + step - 1), 0, 0, 0 };
                    try cuda.copy_source.enqueue(drv, d.meta + s * 16, std.mem.asBytes(&metadata), st);
                    try d.target.kernels.draws.draw(st, lg_ptr, vocab, d.rule + s * rule_sz, d.meta + s * 16, 0, tok_ptr, 1, null, null);
                } else {
                    var a: cuda.Args = .{};
                    a.add(lg_ptr);
                    a.add(tok_ptr);
                    a.add(d.hold_meta + s * 16);
                    try go(d.k.top1_gumbel_dev, st, 1, 256, &a);
                }
            }
            second_bank = !second_bank;
        }
    }

    /// One hold's device launches, no host reads and no uploads: captured once per (depth, lane cache)
    /// for greedy holds, run eagerly otherwise. `item` is only touched when rule_opt != null (never captured).
    fn holdCore(d: *Drafter, st: cuda.Stream, drv: *const cuda.Driver, ln: *LaneState, item: dr.Hold, depth: u32, rule_opt: ?cuda.sample.Rule) !void {
        try ops.embed(&d.target.kernels.glue, st, d.ids, 1, d.target.weights.embed, d.target.weights.embed_norm, d.x, hidden, 1e-6);
        {
            var a: cuda.Args = .{};
            for ([_]u64{ d.x, d.mask, d.x }) |p| a.add(p);
            try go(d.k.init_rows, st, block_rows, 256, &a);
        }
        {
            var a: cuda.Args = .{};
            for ([_]u64{ d.x, 0, d.layers[0].input_norm, d.resid, d.x }) |p| a.add(p);
            a.add(block_rows);
            a.add(@as(u32, 1));
            try go(d.k.residual_norm, st, block_rows, 256, &a);
        }
        for (d.layers, 0..) |layer, i| {
            try dense(d, st, d.x, layer.q, d.q_raw, block_rows, query_width, hidden);
            try dense(d, st, d.x, layer.k, d.k_raw, block_rows, kv_width, hidden);
            try dense(d, st, d.x, layer.v, d.v_raw, block_rows, kv_width, hidden);
            {
                var a: cuda.Args = .{};
                for ([_]u64{ d.q_raw, d.k_raw, d.v_raw, layer.q_norm, layer.k_norm, d.q, d.kb, d.vb, d.hold_meta }) |p| a.add(p);
                try go(d.k.block_qkv_dev, st, block_rows * 40, 128, &a);
            }
            if (d.kv_int8 and i == 1) {
                var a: cuda.Args = .{};
                for ([_]u64{ d.q, d.kb, d.vb, ln.k[i].ptr, ln.v[i].ptr, d.split_o, d.split_ml, d.hold_meta }) |p| a.add(p);
                a.add(@as(u32, @intCast(ln.rings[i])));
                try go(d.k.attention_splitk_dev_i8, st, 8 * 16, 256, &a);
                var m: cuda.Args = .{};
                for ([_]u64{ d.split_o, d.split_ml, d.attn }) |p| m.add(p);
                try go(d.k.attention_splitk_merge, st, block_rows * 24, 128, &m);
            } else {
                var a: cuda.Args = .{};
                for ([_]u64{ d.q, d.kb, d.vb, ln.k[i].ptr, ln.v[i].ptr, d.scores, d.attn, d.hold_meta }) |p| a.add(p);
                a.add(@as(u32, @intCast(ln.rings[i])));
                a.add(@as(u32, @intFromBool(i != 1)));
                a.add(@as(u32, @intCast(d.capacity + block_rows)));
                try go(d.k.attention_dev, st, block_rows * 24, 128, &a);
            }
            try dense(d, st, d.attn, layer.o, d.y, block_rows, hidden, query_width);
            {
                var a: cuda.Args = .{};
                for ([_]u64{ d.y, d.resid, layer.post_norm, d.next_resid, d.x }) |p| a.add(p);
                a.add(block_rows);
                a.add(@as(u32, 0));
                try go(d.k.residual_norm, st, block_rows, 256, &a);
            }
            try dense(d, st, d.x, layer.gate, d.gate, block_rows, intermediate, hidden);
            try dense(d, st, d.x, layer.up, d.up, block_rows, intermediate, hidden);
            {
                var a: cuda.Args = .{};
                for ([_]u64{ d.gate, d.up, d.act }) |p| a.add(p);
                a.add(@as(u32, block_rows * intermediate));
                try go(d.k.swiglu, st, (block_rows * intermediate + 255) / 256, 256, &a);
            }
            try dense(d, st, d.act, layer.down, d.y, block_rows, hidden, intermediate);
            const next_norm = if (i + 1 < 3) d.layers[i + 1].input_norm else d.norm;
            {
                var a: cuda.Args = .{};
                for ([_]u64{ d.y, d.next_resid, next_norm, d.resid, d.x }) |p| a.add(p);
                a.add(block_rows);
                a.add(@as(u32, 0));
                try go(d.k.residual_norm, st, block_rows, 256, &a);
            }
        }
        try d.target.model.dense(&d.target.kernels, st, d.x + hidden * 2, d.target.weights.unembed, d.target.weights.unembed_b12, d.base, depth, vocab, hidden, true, false);
        {
            var a: cuda.Args = .{};
            for ([_]u64{ d.x + hidden * 2, ln.newest.ptr, d.rnn, d.rnn_bias, d.kappa }) |p| a.add(p);
            a.add(depth);
            try go(d.k.kappa, st, (depth * rnn_width + 255) / 256, 256, &a);
        }
        try drv.check(drv.api.cuMemsetD8Async(d.state, 0, rank * 2, st.handle), "cuMemsetD8Async");
        var second_bank = false;
        for (0..depth) |step| {
            const cur = if (second_bank) d.next_state else d.state;
            const nxt = if (second_bank) d.state else d.next_state;
            {
                var a: cuda.Args = .{};
                for ([_]u64{ cur, d.kappa + step * rnn_width * 4, d.rnn, d.markov_w1, d.z, nxt, d.lambda }) |p| a.add(p);
                a.add(d.draft_tokens + step * 4);
                try go(d.k.recurrent_dev, st, 1, 256, &a);
            }
            {
                var a: cuda.Args = .{};
                for ([_]u64{ d.base + step * vocab * 2, d.phi_fp8, d.phi_scale, d.beta, d.f8lut, d.lambda, d.logits }) |p| a.add(p);
                a.add(vocab);
                try go(d.k.logits_fp8, st, (vocab + 7) / 8, 256, &a);
            }
            if (rule_opt != null) {
                const metadata = [4]i32{ @intCast(item.position + step - 1), 0, 0, 0 };
                try cuda.copy_source.enqueue(drv, d.meta, std.mem.asBytes(&metadata), st);
                try d.target.kernels.draws.draw(st, d.logits, vocab, d.rule, d.meta, 0, d.draft_tokens + (step + 1) * 4, 1, null, null);
            } else {
                var a: cuda.Args = .{};
                a.add(d.logits);
                a.add(d.draft_tokens + (step + 1) * 4);
                a.add(d.draft_probs + step * 4);
                a.add(d.draft_margins + step * 4);
                a.add(d.hold_meta);
                try go(d.k.top1_prob_dev, st, 1, 256, &a);
                if (d.dump_fp != null) {
                    var ta: cuda.Args = .{};
                    for ([_]u64{ d.logits, d.dump_top_toks, d.dump_top_vals }) |p| ta.add(p);
                    ta.add(@as(u32, @intCast(step)));
                    ta.add(depth);
                    try go(d.k.top16, st, 1, 256, &ta);
                    try drv.check(drv.api.cuMemcpyDtoDAsync_v2(d.dump_lam + step * rank * 2, d.lambda, rank * 2, st.handle), "cuMemcpyDtoDAsync_v2");
                }
            }
            second_bank = !second_bank;
        }
    }

    fn holdGraph(d: *Drafter, st: cuda.Stream, ln: *LaneState, item: dr.Hold, depth: u32) !cuda.graph.Exec {
        for (d.hold_graphs.items) |*g| if (g.depth == depth and g.cache == ln.k[0].ptr) return g.exec;
        try cuda.graph.beginCapture(st, .thread_local);
        d.holdCore(st, d.target.model.d, ln, item, depth, null) catch |err| {
            if (cuda.graph.endCapture(st)) |g| {
                var discarded = g;
                discarded.deinit();
            } else |_| {}
            return err;
        };
        var graph = try cuda.graph.endCapture(st);
        defer graph.deinit();
        var exec = try graph.instantiate();
        errdefer exec.deinit();
        try exec.upload(st);
        try d.hold_graphs.append(d.gpa, .{ .depth = depth, .cache = ln.k[0].ptr, .exec = exec });
        return exec;
    }

    fn holdBatchGraph(d: *Drafter, st: cuda.Stream, items: []const dr.Hold, max_d: u32) !cuda.graph.Exec {
        const S: u32 = @intCast(items.len);
        for (d.hold_batch_graphs.items) |*g| {
            if (g.s == S and g.max_d == max_d) return g.exec;
        }
        try cuda.graph.beginCapture(st, .thread_local);
        d.holdBatchCore(st, d.target.model.d, items, 0) catch |err| {
            if (cuda.graph.endCapture(st)) |g| {
                var discarded = g;
                discarded.deinit();
            } else |_| {}
            return err;
        };
        var graph = try cuda.graph.endCapture(st);
        defer graph.deinit();
        var exec = try graph.instantiate();
        errdefer exec.deinit();
        try exec.upload(st);
        try d.hold_batch_graphs.append(d.gpa, .{ .s = S, .max_d = max_d, .exec = exec });
        return exec;
    }

    fn purgeHoldGraphs(d: *Drafter, cache: u64) void {
        var i: usize = 0;
        while (i < d.hold_graphs.items.len) {
            if (d.hold_graphs.items[i].cache == cache) {
                var removed = d.hold_graphs.swapRemove(i);
                removed.exec.deinit();
            } else {
                i += 1;
            }
        }
    }

    fn clearHoldGraphs(d: *Drafter) void {
        for (d.hold_graphs.items) |*g| g.exec.deinit();
        d.hold_graphs.clearAndFree(d.gpa);
        for (d.hold_batch_graphs.items) |*g| g.exec.deinit();
        d.hold_batch_graphs.clearAndFree(d.gpa);
    }
};
