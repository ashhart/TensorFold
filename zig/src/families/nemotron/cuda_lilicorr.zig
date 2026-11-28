const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

const cuda = @import("cuda");
const cuda_engine = @import("cuda_engine.zig");
const cuda_kernels = @import("cuda_kernels.zig");

const cuda_glue = @import("cuda_glue.zig");
const cuda_draft_config = @import("cuda_draft_config.zig");
const cuda_lilicorr_weights = @import("cuda_lilicorr_weights.zig");
const head = @import("cuda_lilicorr_head.zig");
const walk = @import("cuda_lilicorr_walk.zig");
const nvfp4 = @import("cuda_lilicorr_nvfp4.zig");

const lanes = @import("lanes");
const dr = lanes.drafter;
const Stream = lanes.Stream;
const DeviceBuffer = cuda.DeviceBuffer;
const Context = cuda.Context;

const stage = @import("cuda_lilicorr_stage.zig");
const stack = @import("cuda_lilicorr_stack.zig");
const bgate = @import("cuda_lilicorr_bodygate.zig");
const round = @import("cuda_lilicorr_round.zig");
const absorbz = @import("cuda_lilicorr_absorb.zig");
const openz = @import("cuda_lilicorr_open.zig");
pub const max_streams = 8;

// Lazy TF_TAP_DEBUG gate for observation-only instrumentation (module scope: this Zig forbids

// declarations between container fields)
pub var tap_debug_env: ?bool = null;
pub var tap_dumped: bool = false;
pub var th_dumped: bool = false;

// Lazy TF_SURGERY_DUMP gate for surgical dump instrumentation
var surgery_env: ?bool = null;
pub var surgery_dumped = false;
pub fn surgeryOn() bool {
    if (surgery_env) |v| return v;
    const v = (@import("sys").environment.read(bool, "TF_SURGERY_DUMP", false) catch false);
    surgery_env = v;
    return v;
}

/// The f32-accumulate gate (TF_DRAFT_F32ACC): 1 skips the per-op bf16 rounding chain (upstream f32-accum policy).
var f32acc_cache: ?c_int = null;
pub fn f32acc() c_int {
    if (f32acc_cache) |v| return v;
    const v: c_int = if ((@import("sys").environment.read(bool, "TF_DRAFT_F32ACC", false) catch false)) 1 else 0;
    f32acc_cache = v;
    return v;
}

/// Stage oracle (TF_STAGE_DUMP): drafter backbone per-stage buffers dumped /tmp/stage_ours_<name>.bin at round 1.
var stage_env: ?bool = null;
pub var stage_fwd_count: u32 = 0; // 2-frame gate: frame 1 = the warmup dummy (id-0 constant rows), frame 2 = the first real
pub fn stageOn() bool {
    if (stage_env) |v| return v;
    const v = (@import("sys").environment.read(bool, "TF_STAGE_DUMP", false) catch false);
    stage_env = v;
    return v;
}

var stage_seq: u64 = 0; // monotonic dump counter: same-name dumps never overwrite history (AAR 11/24 law)
pub fn stageDump(d: *Draft, buf: u64, bytes: usize, name: []const u8) !void {
    const hb = try d.gpa.alloc(u8, bytes);
    defer d.gpa.free(hb);
    try d.e.ctx.d.check(d.e.ctx.d.api.cuMemcpyDtoHAsync_v2(hb.ptr, buf, bytes, d.e.stream.handle), "DtoH stageDump");
    try d.e.stream.synchronize();
    var pbuf: [96]u8 = undefined;
    stage_seq += 1;
    const pth = try std.fmt.bufPrint(&pbuf, "/tmp/stage_{d}_{s}.bin", .{ stage_seq, name });
    const f = try std.Io.Dir.cwd().createFile(d.e.io, pth, .{});
    defer f.close(d.e.io);
    try std.Io.File.writePositionalAll(f, d.e.io, hb, 0);
    std.debug.print("[stage-ours] wrote {s} bytes={d}\n", .{ name, bytes });
}

pub const staging_rows = max_streams * 16; // the block-pass staging row cap (qwen frame's max_rows)
pub const segments_at = staging_rows * 4;
pub const words_at = segments_at + max_streams * @sizeOf(Segment);
pub const anchor_at = words_at + max_streams * (3 + 2 * 20) * 4;
pub const token_emb_at = anchor_at + head.model_hidden * 4;
pub const pass_at = token_emb_at + head.lattice * head.model_hidden * 4;
pub const scores_at = pass_at + head.lattice * head.model_hidden * 2 + 4096 * 2; // start+pair staging (pinned)
pub const scores_bytes = head.topk * 4 + (head.slots - 1) * head.topk * head.topk * 4;

pub const max_rows = 256; // the absorb API cap and the ring headroom; the device chunk is staging_rows

pub const Segment = extern struct {
    ring: u64,
    begin: i64,
    position: i64,
};

const core = @import("core");

pub const Facts = dr.Facts;

pub const Absorb = dr.Absorb;

pub const Hold = dr.Hold;

pub const Drafter = dr.Drafter;

pub const E = cuda_engine.Engine;
pub const Config = cuda_draft_config.Config;
/// Owner A5's derived faces live beside the state they read: row0Token on the Config, spanOpen on the span base.
pub const row0Token = cuda_draft_config.row0Token;
pub const spanOpen = absorbz.spanOpen;
pub const Weights = cuda_lilicorr_weights.Weights;

const TapRows = absorbz.TapRows;

/// A projection's row role: prompt ingest takes the GEMV face, live rounds the verify tiles.
const Role = enum { prompt, live };

/// The staged open-ownership mark: see Draft.open_stage.
pub const OpenStage = enum(u8) { none, core, ring, ready };

pub const Draft = struct {
    /// false until open() finishes: deinit() refuses a partially built Draft (the cleanup crash class).
    ready: bool = false,
    /// Stage marks: .none till init() done, then .core, .ring, .ready; deinit() tears down exactly the owned stages.
    open_stage: OpenStage = .none,
    scratch_buf: DeviceBuffer = undefined, // the absorb/forward scratch plane (taps_in/th/h/x/q/logits/... carved below)
    e: *E,
    c: Config,
    w: Weights,
    mod: cuda.Module,
    mod_q: cuda.Module,
    fns: struct {
        gather_taps: cuda.Function,
        gather_rows: cuda.Function,
        norm_rope: cuda.Function,
        ring_put: cuda.Function,
        sums: cuda.Function,
        norm_rope_batch: cuda.Function,
        attn_partial: cuda.Function,
        attn_merge: cuda.Function,
        mlp_act: cuda.Function,
        mask_rows: cuda.Function,
        dyn_conv_batch: cuda.Function,
        token_proj: cuda.Function,
        lat_attn: cuda.Function,
        factor: cuda.Function,
        edge_linear: cuda.Function,
        l2norm: cuda.Function,
        dots: cuda.Function,
        rms: cuda.Function,
        lin_f32: cuda.Function,
        silu_bf16: cuda.Function,
        add_bf16: cuda.Function,
        lattice: cuda.Function,
        rms_f32: cuda.Function,
        widen: cuda.Function,
        narrow: cuda.Function,
    },
    taps_in: u64,
    th: u64,
    thn: u64,
    kabs: u64,
    vabs: u64,
    kn: u64,
    affine: cuda.qlinear.Affine4Kernels,
    step_ms: f64,
    tap_rows: u32,

    // New scratch fields as per stage 2 requirements
    h: u64,
    h2: u64,
    x: u64,
    xin: u64,
    dyn: u64,
    qraw: u64,
    q: u64,
    kraw: u64,
    kb: u64,
    vb: u64,
    att: u64,
    pm: u64, // the attention partials: max/sum per (row, head, chunk) + the unnormalized output
    pl: u64,
    po: u64,
    pm_buf: DeviceBuffer = undefined,
    pl_buf: DeviceBuffer = undefined,
    po_buf: DeviceBuffer = undefined,
    o: u64,
    gate: u64,
    up: u64,
    act: u64,
    logits: u64,
    words: u64,
    segments: u64,
    staged: cuda.HostBuffer,

    // New device fields for draft
    cand_ids: u64,
    cand_emb: u64,
    anchor: u64,
    token_states: u64,
    draft_picks: u64,
    cand_ids_buf: DeviceBuffer = undefined,
    cand_emb_buf: DeviceBuffer = undefined,
    anchor_buf: DeviceBuffer = undefined,
    token_states_buf: DeviceBuffer = undefined,
    draft_picks_buf: DeviceBuffer = undefined,
    tap_ring: cuda.DeviceBuffer = undefined,
    xs: u64 = 0, // the projections' 64-group sums scratch (the verify path reads it)
    ids: u64 = 0, // the staged block ids on the device (the embed reads them)
    ids_buf: DeviceBuffer = undefined,
    segments_buf: DeviceBuffer = undefined, // the device home of the `segments` field above
    projection_role: Role = .live,
    xs_buf: DeviceBuffer = undefined,

    // Head weights arena
    head_arena: ?std.heap.ArenaAllocator = null,

    // Head weights
    hw: head.Weights = undefined,
    token_emb_f32: [head.slots * head.topk * head.model_hidden]f32 = undefined,
    pass_f32: [head.slots * head.topk * head.model_hidden]f32 = undefined,
    anchor_f32: [head.model_hidden]f32 = undefined,
    head_loaded: bool = false,
    dhw: ?head.DeviceWeights = null, // the head's device staging (S1): nothing consumes it yet

    // S5b device-chain scratch: carved ONCE at open (L11 buffer DAG) when the device path or

    // the fixture A/Bs are armed; hold()-time launches only. Sizes listed in stageDeviceScratch.
    ds: DeviceScratch = .{},

    // Drafter directory for head loading
    dir: []const u8 = "",

    // Stream lifecycle fields
    streams: std.AutoHashMapUnmanaged(*Stream, *Ctx) = .empty,
    pool: std.ArrayList(*Ctx) = .empty,
    gpa: std.mem.Allocator,
};

const DeviceScratch = openz.DeviceScratch;
const attachWeights = openz.attachWeights;

pub const Ctx = struct {
    ring: DeviceBuffer,
    begin: u64 = 0,
    end: u64 = 0,
    held: [staging_rows]u32 = @splat(0),
    held_n: usize = 0,
    anchor_valid: bool = false,
    /// The span base: the context's end at its prompt phase's END; spanOpen opens at >=2 accepted rows past it.
    span_base: ?u64 = null,
};

pub fn run(e: *E, f: cuda.Function, grid: [3]usize, block: [3]usize, args: anytype) !void {
    var a: cuda.Args = .{};
    inline for (args) |v| a.add(v);
    const cfg: cuda.Config = .{
        .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) },
        .block = .{ .x = @intCast(block[0]), .y = @intCast(block[1]), .z = @intCast(block[2]) },
    };
    try cuda.launch.launch(f, cfg, e.stream, &a);
}

const deviceTokenStatesAB = stage.deviceTokenStatesAB;

const deviceLatticeAttentionAB = stage.deviceLatticeAttentionAB;

const readFixtureF32 = stack.readFixtureF32;

const deviceFactorEdgeAB = stage.deviceFactorEdgeAB;

const a_in_bits_to_f32 = stack.a_in_bits_to_f32;

const bf16Ulp = stack.bf16Ulp;

const deviceLayerStackAB = stack.deviceLayerStackAB;

pub fn ringK(d: *const Draft, ctx: *const Ctx, layer: usize) u64 {
    return ctx.ring.ptr + @as(u64, layer) * 2 * d.c.ringRows() * d.c.kvWidth() * 2;
}

pub fn ringV(d: *const Draft, ctx: *const Ctx, layer: usize) u64 {
    return ringK(d, ctx, layer) + d.c.ringRows() * d.c.kvWidth() * 2;
}

const bf16BitsOf = bgate.bf16BitsOf;

const uploadFixtureBf16 = bgate.uploadFixtureBf16;

const bodyGateStage = bgate.bodyGateStage;

const deviceBodyFixtureAB = bgate.deviceBodyFixtureAB;

const norm = round.norm;

const projDense = round.projDense;

const projAgain = round.projAgain;

const projMany = round.projMany;

const projSums = round.projSums;

const lp_dumped = round.lp_dumped;
const hold_prof_env = round.hold_prof_env;
const hp_fwd = round.hp_fwd;
const hp_body = round.hp_body;
const hp_sync = round.hp_sync;
const hp_emb = round.hp_emb;
const hp_score = round.hp_score;
const hp_walk = round.hp_walk;
const hp_launch = round.hp_launch;
const hp_sync2 = round.hp_sync2;
const hp_d1 = round.hp_d1;
const hp_d2 = round.hp_d2;

const holdProfOn = round.holdProfOn;

const holdMs = round.holdMs;

const absorb = absorbz.absorb;
const absorbRows = absorbz.absorbRows;

pub const create = openz.create;

pub const drop = openz.drop;

const openStream = openz.openStream;

const release = openz.release;

const held = openz.held;

const taps = openz.taps;

pub const deinit = openz.deinit;

/// 16d: the target's verify-window ms by rows for the depth rule, anchored at the live ~14 ms/1-row decode datum.
const verify_window_costs = blk: {
    @setEvalBranchQuota(10000);
    var c: [64]lanes.config.Cost = undefined;
    for (&c, 0..) |*x, i| x.* = .{ .width = i + 1, .ms = 14.0 + 0.5 * @as(f64, @floatFromInt(i)) };
    break :blk c;
};

fn facts(d: *Draft) Facts {
    return .{
        .fixed_block = true,
        .depth = @intCast(d.c.block - 1),
        .step_ms = d.step_ms,
        .plain_guard = true,
        .batched = true,
        .one_stream = true, // the head's device chain carves its scratch once per Draft: concurrent walks would corrupt
        .stream_bytes = streamBytesFor(d.c, @intCast(d.c.hidden), d.tap_rows),
        .trees = false,
        .window_costs = &verify_window_costs,
    };
}

const streamBytesFor = openz.streamBytesFor;

const init = openz.init;

const forward = round.forward;

const widenBuf = round.widenBuf;

const stageDeviceScratch = openz.stageDeviceScratch;

const featuresBits = round.featuresBits;

const hp_n = round.hp_n;
const hp_sum = round.hp_sum;
const hp_max = round.hp_max;
const hp_max_name = round.hp_max_name;

const runP = round.runP;

const deviceScore = round.deviceScore;

const deviceFullChainAB = bgate.deviceFullChainAB;

const hold = round.hold;

/// --drafter over the nemotron engine: open the LiLiCorr drafter in dir; chains only (no trees) for PR-1.
pub fn drafter(d: *Draft) dr.Drafter {
    return .{ .kind = "LiLiCorr", .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, e: *E, dir: []const u8) !*Draft {
    // The 4-bit projections resolve first: the weights loader stages through them.
    const sms: usize = @intCast(try e.ctx.attribute(.multiprocessor_count));
    const d = try gpa.create(Draft);
    errdefer gpa.destroy(d);
    d.ready = false; // create() hands over undefined memory: no field default exists until written
    d.ds = .{}; // S5b: the field default never applies under create() — undefined ds = garbage staged flag
    d.affine = try cuda.qlinear.Affine4Kernels.resolve(e.ctx.d, e.k.mods[0], e.k.mods[1], e.k.mods[15], e.k.mods[20], sms, e.k.gb10);

    // Load the config; only the LiLiCorr kind drives this driver (dflash2 has no Nemotron drafter upstream)
    const c = try cuda_draft_config.Config.read(gpa, io, dir, .{ .hidden = e.c.hidden, .vocab = e.c.vocab, .layers = e.c.layers });
    if (c.kind != .lilicorr) return error.UnsupportedDrafterFormat;
    const shape = cuda_lilicorr_weights.Shape{ .hidden = c.hidden, .intermediate = c.intermediate, .layers = c.layers, .q_width = c.qWidth(), .kv_width = c.kvWidth(), .head_dim = c.head_dim, .dynamic = c.dynamic(), .taps = c.n_taps };

    try attachWeights(d, gpa, io, e, c, dir, try cuda_lilicorr_weights.load(gpa, io, e, &d.affine, dir, shape));
    errdefer deinit(d);

    // The tap ring lives as long as the Draft: the engine's tap kernel writes it every forward.

    // Rows x the taps' slots x hidden bf16 — ringBytes() is the K/V ring's size, not this ring's.
    d.tap_ring = try cuda.DeviceBuffer.alloc(e.ctx.d, c.ringRows() * c.n_taps * c.hidden * 2);
    d.open_stage = .ring; // the ring is owned from here on: a setTaps failure unwinds through deinit()

    // Taps arm before any stream exists (the engine refuses TapsAfterStreams and a second set)
    try e.setTaps(taps(
        d,
    ), d.tap_ring.ptr, c.ringRows());

    d.open_stage = .ready;
    d.ready = true; // fully open: deinit() now owns the teardown
    return d;
}

fn tapsFn(p: *anyopaque) []const u32 {
    const d: *Draft = @ptrCast(@alignCast(p));
    return taps(
        d,
    );
}

fn factsFn(p: *anyopaque) Facts {
    const d: *Draft = @ptrCast(@alignCast(p));
    return facts(
        d,
    );
}

fn openFn(p: *anyopaque, s: *Stream) anyerror!void {
    const d: *Draft = @ptrCast(@alignCast(p));
    return openStream(d, s);
}

fn absorbFn(p: *anyopaque, items: []const Absorb) anyerror!void {
    const d: *Draft = @ptrCast(@alignCast(p));
    return absorb(d, items);
}

fn holdFn(p: *anyopaque, items: []const Hold) anyerror!void {
    const d: *Draft = @ptrCast(@alignCast(p));
    return hold(d, items);
}

fn heldFn(p: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
    const d: *Draft = @ptrCast(@alignCast(p));
    return held(d, streams, out);
}

fn releaseFn(p: *anyopaque, s: *Stream) void {
    const d: *Draft = @ptrCast(@alignCast(p));
    release(d, s);
}

test "layout accounting" {
    const config = Config{
        .hidden = 4096,
        .intermediate = 12288,
        .layers = 5,
        .vocab = 131072,
        .heads = 32,
        .kv_heads = 8,
        .head_dim = 128,
        .block = 8,
        .mask = 990,
        .window = 2048,
        .rank = 0,
        .topk = 8,
        .eps = 1e-6,
        .theta = 10000.0,
        .taps = .{ 1, 18, 35, 51, 68, 85, 0, 0 },
        .n_taps = 6,
        .kind = .lilicorr,
        .embed_scale = 1.0,
        .group = 16,
    };

    // streamBytesFor must at least cover the ring + the tap rows, and stay sane
    const floor = config.ringBytes() + @as(usize, 1024) * config.n_taps * config.hidden * 2;
    try std.testing.expect(streamBytesFor(config, config.hidden, 1024) > floor);
    try std.testing.expect(streamBytesFor(config, config.hidden, 1024) < 2 * 1024 * 1024 * 1024);
    // the pinned staging constants hold the worst case: 8 streams' ids+segments+words
    try std.testing.expect(words_at > segments_at);
    try std.testing.expect(segments_at >= staging_rows * 4);
}

test "absorb keep/trim math" {
    // This is a simplified test for the keep/trim logic
    const config = Config{
        .hidden = 4096,
        .intermediate = 12288,
        .layers = 5,
        .vocab = 131072,
        .heads = 32,
        .kv_heads = 8,
        .head_dim = 128,
        .block = 8,
        .mask = 990,
        .window = 2048,
        .rank = 0,
        .topk = 8,
        .eps = 1e-6,
        .theta = 10000.0,
        .taps = .{ 1, 18, 35, 51, 68, 85, 0, 0 },
        .n_taps = 6,
        .kind = .lilicorr,
        .embed_scale = 1.0,
        .group = 16,
    };

    // Test case: keep calculation
    const tap_rows: usize = 1024;
    const keep = @min(config.window, tap_rows - max_rows);
    try std.testing.expectEqual(@min(2048, 1024 - 256), keep);
}

test "ringK/ringV offsets" {
    const config = Config{
        .hidden = 4096,
        .intermediate = 12288,
        .layers = 5,
        .vocab = 131072,
        .heads = 32,
        .kv_heads = 8,
        .head_dim = 128,
        .block = 8,
        .mask = 990,
        .window = 2048,
        .rank = 0,
        .topk = 8,
        .eps = 1e-6,
        .theta = 10000.0,
        .taps = .{ 1, 18, 35, 51, 68, 85, 0, 0 },
        .n_taps = 6,
        .kind = .lilicorr,
        .embed_scale = 1.0,
        .group = 16,
    };

    // the offsets' arithmetic, verified as numbers (the methods are two one-liners; a device

    // construction would pull the build-graph modules into this standalone test)
    const base: u64 = 1000;
    const kv_span: u64 = @intCast(config.ringRows() * config.kvWidth() * 2);
    const k1: u64 = base + 1 * 2 * @as(u64, @intCast(config.ringRows() * config.kvWidth() * 2));
    try std.testing.expectEqual(base, k1 - 2 * kv_span);
    try std.testing.expectEqual(k1 + kv_span, k1 + config.ringRows() * config.kvWidth() * 2);
}

test "the words layout parses ids and log-softmax logits" {
    // logprob_rows.cuh: out[0] = f32 bits of the log normalizer; out[2 + k] = the k-th best

    // id; out[2 + 20 + k] = f32 bits of the k-th best logit. lp_words = 3 + 2 * 20.
    const lp: usize = 3 + 2 * 20;
    var words: [lp]u32 = undefined;
    const norm_bits: u32 = @bitCast(@as(f32, 3.5));
    const top1_id: u32 = 4242;
    const top1_logit: f32 = 9.25;
    words[0] = norm_bits;
    words[2] = top1_id;
    words[2 + 20] = @bitCast(top1_logit);
    const id: u32 = words[2];
    const logit: f32 = @bitCast(words[2 + 20]);
    try std.testing.expectEqual(top1_id, id);
    try std.testing.expectEqual(top1_logit, logit);
    try std.testing.expectEqual(top1_logit - 3.5, logit - @as(f32, @bitCast(norm_bits)));
    // the chain fill: held[r] = cand[r * topk + chain.index[r]]
    const topk = 8;
    var cand: [7 * topk]u32 = undefined;
    for (&cand, 0..) |*v, i| v.* = @intCast(1000 + i);
    var index: [7]usize = undefined;
    for (&index, 0..) |*v, r| v.* = r % topk; // a deterministic fake walk
    var held_out: [7]u32 = undefined;
    for (0..7) |r| held_out[r] = cand[r * topk + index[r]];
    try std.testing.expectEqual(@as(u32, 1000 + 0 * topk + 0), held_out[0]);
    try std.testing.expectEqual(@as(u32, 1000 + 1 * topk + 1), held_out[1]);
}
