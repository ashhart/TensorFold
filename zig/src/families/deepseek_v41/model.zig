//! Plain V4.1 execution over resident EXL3 weights and source-owned packed caches.
const builtin = @import("builtin");
const std = @import("std");
const mtl = @import("metal");
const checkpoint = @import("core").checkpoint_metal;
const exl3 = @import("core").exl3_projection;
const config = @import("config.zig");
const load = @import("load.zig");
const weights = @import("weights.zig");
const work = @import("workspace.zig");
pub const attention = @import("attention.zig");
const attention_common = @import("attention_common.zig");
pub const kv = @import("kv.zig");
pub const ops = @import("ops.zig");
pub const mhc = @import("mhc.zig");
pub const csa2 = @import("csa2.zig");
pub const csa2_index = @import("csa2_index.zig");
pub const rope = @import("rope.zig");
const route = @import("route.zig");
const engram = @import("engram.zig");
const engram_stream = @import("engram_rows_safetensors.zig");
pub const View = @import("view.zig").View;
const gpu_counters = @import("profiler.zig");
const expert_groups = @import("core").expert_groups;
const scan_probe = @import("scan_probe.zig");
const support = @import("model_support.zig");
const model_drafter = @import("model_drafter.zig");
const model_hash = @import("model_hash.zig");
const model_lifecycle = @import("model_lifecycle.zig");
const model_import = @import("model_import.zig");
const model_propose = @import("model_propose.zig");
const model_draft_attention = @import("model_draft_attention.zig");
const forward_impl = @import("forward_impl.zig");
const moe_impl = @import("model_moe.zig");
pub const nowNs = support.nowNs;
pub const prefill_step = support.prefill_step;
pub const Logits = support.Logits;
pub const Seg = support.Seg;
pub const HidHash = support.HidHash;
pub const mtp_dbg_raw = support.mtp_dbg_raw;
pub const mtp_dbg_qr = support.mtp_dbg_qr;
pub const mtp_dbg_qpre = support.mtp_dbg_qpre;
pub const mtp_dbg_dense = support.mtp_dbg_dense;
pub const mtp_dbg_qpost = support.mtp_dbg_qpost;
pub const mtp_dbg_rot = support.mtp_dbg_rot;
pub const mtp_dbg_sums = support.mtp_dbg_sums;
pub const mtp_dbg_coll = support.mtp_dbg_coll;
pub const mtp_dbg_pre = support.mtp_dbg_pre;
pub const mtp_dbg_pin = support.mtp_dbg_pin;
pub const mtp_dbg_prot = support.mtp_dbg_prot;
pub const mtp_dbg_psum = support.mtp_dbg_psum;
pub const mtp_dbg_total = support.mtp_dbg_total;
pub const mtp_dbg_grouped = support.mtp_dbg_grouped;
pub const mtp_dbg_total2 = support.mtp_dbg_total2;
pub const mtp_dbg_woasums = support.mtp_dbg_woasums;
pub const mtp_dbg_total3 = support.mtp_dbg_total3;
pub const Options = support.Options;
pub const Pipelines = support.Pipelines;
pub const tapsOn = support.tapsOn;
pub const mtpStopPoint = support.mtpStopPoint;
pub const markovBiasOffOn = support.markovBiasOffOn;
pub const dsparkTruePosOn = support.dsparkTruePosOn;
pub const tapsSharedBaseOn = support.tapsSharedBaseOn;
pub const partialTxnOn = support.partialTxnOn;
pub const ringKeepOn = support.ringKeepOn;
pub const max_ring_rows = support.max_ring_rows;
pub const mtpRopeAllOn = support.mtpRopeAllOn;
pub const condRopeOn = support.condRopeOn;
pub const mtpTrunkRing = support.mtpTrunkRing;
pub const condAlignOn = support.condAlignOn;
pub const condRowsWanted = support.condRowsWanted;
pub const concurrentOn = support.concurrentOn;
pub const headRms = support.headRms;
pub const logitsTopK = support.logitsTopK;
pub const Model = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    device: mtl.Device,
    /// The gate GEMM's f16 activation copy when the pack's router gate is f16 (TF_DSV41_ROUTE_GEMM).
    gate_x: ?mtl.Buffer = null,
    /// The gate capture's staging buffers (TF_DSV41_GATE_DUMP): one pair, reused every layer and every token.
    gate_dump_x: ?mtl.Buffer = null,
    gate_dump_l: ?mtl.Buffer = null,
    queue: mtl.Queue,
    residency: mtl.ResidencySet,
    checkpoint: checkpoint.Checkpoint,
    weights: weights.Model,
    cfg: config.Config,
    cfg_arena: std.heap.ArenaAllocator,
    lanes: u32,
    caches: []kv.cache.State,
    /// Slot 0's state; existing single-lane callers keep working unchanged.
    cache: *kv.cache.State,
    /// The drafter's three KV rings (TF_DSV41_MTP_DRAFTS), built on first use; null until then.
    drafter_state: ?*kv.cache.State = null,
    /// The drafter's one-token id (the embedding kernel reads an ids view); shared, four bytes, written a step.
    drafter_ids: ?mtl.Buffer = null,
    /// The Markov chain's picks (a u32 a step), kept apart from the one-token id buffer the embedding reads.
    markov_ids: ?mtl.Buffer = null,
    /// The Markov chain's per-threadgroup bests (ceil(vocab/256) float2), allocated once with the drafts.
    markov_part: ?mtl.Buffer = null,
    /// The last row the conditioning taps captured: the drafter reads that row's three layers.
    taps_row: u32 = 0,
    /// Rows the last tap capture kept (1..128): the draft conditioning's row count.
    taps_n: u32 = 0,
    /// The rows the LAST tap capture kept.
    taps_rows_last: u32 = 0,
    /// The rows of the LAST tap capture that are valid (0 = all of them).
    taps_rows_valid: u32 = 0,
    /// The adaptive depth policy's signal -- the accepted drafts a round, smoothed.
    mtp_accept_slot: [32]f64 = @splat(0.0),
    mtp_accept_seen_slot: [32]u32 = @splat(0),
    /// The round's per-draft acceptance (kept/offered), the hybrid policy's stay-deep signal.
    mtp_round_slot: [32]f64 = @splat(0.0),
    mtp_round_seen_slot: [32]u32 = @splat(0),
    /// The ring's write row for the CURRENT forward.
    taps_base: u32 = 0,
    /// The 128-row conditioning scratch (rows x 5120 bf16) and its packed keys (rows x 528), allocated with the taps.
    cond: ?mtl.Buffer = null,
    cond_wkv: ?mtl.Buffer = null,
    mtp_drafts: [4]u32 = .{ 0, 0, 0, 0 },
    mtp_count: u32 = 0,
    /// The draft round's own logits scratch (eight rows), so its forward never writes the workspace's logits field.
    mtp_logits: ?mtl.Buffer = null,
    /// The drafter's router logits.
    dspark_logits: ?mtl.Buffer = null,
    /// The trunk's own head input (the verify's row 0), for the drafter-vs-trunk hidden probe.
    mtp_trunk_hidden: ?mtl.Buffer = null,
    /// Each slot's Engram cursor as of its beginSlot, restored by rollbackSlot.
    engram_stash: []u8,
    /// Each slot's compressed-path partial scratch as of its beginSlot (TF_DSV41_PARTIAL_TXN).
    partial_stash: []u8,
    /// Each slot's stash of the window ring's next `max_ring_rows` slots.
    ring_stash: mtl.Buffer,
    /// A MERGED TAIL CHUNK'S STASH: the prefix before the mark, so the state at the mark can be copied from it.
    snap_ring: mtl.Buffer,
    taps_stash: mtl.Buffer,
    publish_limit: u32 = 0,
    snap_pos: u32 = 0,
    snap_hist: [4]u32 = .{ 0, 0, 0, 0 },
    snap_taps_row: u32 = 0,
    snap_taps_n: u32 = 0,
    snap_taps_base: u32 = 0,
    snap_taps_last: u32 = 0,
    /// Rows read from the release's shards a window at a time; null when the tables are resident.
    stream: ?*engram_stream.Rows,
    /// The window's staged rows: two layers of [rows, gather_dim] bf16, CPU-filled.
    stream_stage: ?mtl.Buffer,
    workspace: work.Workspace,
    attention_workspace: attention.Workspace,
    /// The drafter's taps: three backbone layers' hidden states, a row's worth of positions at a time.
    taps: mtl.Buffer,
    pipelines: Pipelines,
    admitted_bytes: usize,
    load_seconds: f64 = 0,
    last_gpu_seconds: f64 = 0,
    logits_rows: usize = 0,
    valid: bool = true,
    profile: ?*gpu_counters.Recorder = null,
    /// The fused boundary's stream parity and the pending expand (TF_DSV41_NO_HC_BOUNDARY).
    stream_alt: bool = false,
    pending_expand: bool = false,
    host_prof: bool = false,
    host_hash_ns: u64 = 0,
    host_stage_ns: u64 = 0,
    host_stage_misses: u64 = 0,
    host_stage_faults: u64 = 0,
    host_stage_pf: u8 = 0,
    host_stage_thr: u16 = 0,
    /// The adaptive Engram gather: two cold gathers switch to mapped workers; every 16th probes warmth.
    engram_par: bool = false,
    /// The thread state machine's probe is not the arbiter.
    engram_pf_on: bool = false,
    engram_pf_off: u8 = 0,
    engram_probe_in: u8 = 0,
    engram_probe_every: u8 = 16,
    engram_slow_n: u8 = 0,
    /// The second Engram layer's prompt gather is deferred to its own layer.
    engram_pending: bool = false,
    profile_stage_dispatch: [gpu_counters.stage_count]u64 = @splat(0),
    profile_last_count: u64 = 0,
    profile_open_stage: ?gpu_counters.Stage = null,
    grouped_prefill: bool = true,
    observer: ?Observer = null,
    /// The attention's own per-stage observer (the `stagediff` harness).
    attention_observer: ?attention.DiagnosticObserver = null,
    pub const Observer = struct {
        ctx: *anyopaque,
        encode: *const fn (*anyopaque, mtl.ComputeEncoder, u32, gpu_counters.Stage, *const Model) void,
    };

    pub fn open(gpa: std.mem.Allocator, io: std.Io, o: Options) !Model {
        return model_lifecycle.open(gpa, io, o);
    }
    pub fn deinit(m: *Model) void {
        model_lifecycle.deinit(m);
    }
    pub fn reset(m: *Model) !void {
        return model_lifecycle.reset(m);
    }
    pub fn resetSlot(m: *Model, slot: u32) !void {
        return model_lifecycle.resetSlot(m, slot);
    }
    pub fn beginSnapTail(m: *Model, slot: u32, hist: []const u32, at: u32, from: u32) bool {
        return model_lifecycle.beginSnapTail(m, slot, hist, at, from);
    }
    pub fn snapTailAllowed(m: *const Model) bool {
        return model_lifecycle.snapTailAllowed(m);
    }
    pub fn endSnapTail(m: *Model) void {
        model_lifecycle.endSnapTail(m);
    }
    pub fn snapshotBytes(m: *const Model) u64 {
        return model_lifecycle.snapshotBytes(m);
    }
    pub const Snapshot = struct {
        storage: []u8,
        live: []u8,
        taps: []u8,
        position: u32,
        history: [4]u32,
        taps_row: u32,
        taps_n: u32,
        taps_rows_last: u32,
        taps_rows_valid: u32,
        taps_base: u32,
    };

    pub fn snapshotSave(m: *Model, slot: u32) !*Snapshot {
        return model_lifecycle.snapshotSave(m, slot);
    }
    pub fn snapshotDrop(m: *Model, snap: *Snapshot) void {
        model_lifecycle.snapshotDrop(m, snap);
    }
    pub fn snapshotRestore(m: *Model, slot: u32, snap: *const Snapshot) !void {
        return model_lifecycle.snapshotRestore(m, slot, snap);
    }
    /// A zeroed Snapshot an importer fills (model_import.zig), sealed restorable after `upto` tokens.
    pub fn snapshotBlank(m: *Model) !*Snapshot {
        return model_import.blank(m);
    }
    pub fn snapshotSeal(m: *const Model, snap: *Snapshot, tokens: []const u32, upto: u32) !void {
        return model_import.seal(m, snap, tokens, upto);
    }
    pub fn beginSlot(m: *Model, slot: u32) !void {
        return model_lifecycle.beginSlot(m, slot);
    }
    pub fn commitSlot(m: *Model, slot: u32) !void {
        return model_lifecycle.commitSlot(m, slot);
    }
    pub fn rollbackSlot(m: *Model, slot: u32) !void {
        return model_lifecycle.rollbackSlot(m, slot);
    }
    pub fn portKeep(m: *Model, slot: u32, kept: u32) !void {
        return model_lifecycle.portKeep(m, slot, kept);
    }
    pub fn restoreRing(m: *Model, slot: u32, kept: u32, rows: u32) !void {
        return model_lifecycle.restoreRing(m, slot, kept, rows);
    }
    pub fn forward(m: *Model, ids: []const u32, logits: Logits) ![]const f32 {
        return m.forwardOn(0, ids, logits);
    }

    /// A chunk against lane slot `slot`'s caches and Engram cursor: rows land at that slot's positions.
    pub fn forwardOn(m: *Model, slot: u32, ids: []const u32, logits: Logits) ![]const f32 {
        if (!m.valid) return error.InvalidModelState;
        if (slot >= m.lanes) return error.InvalidLaneSlot;
        return m.forwardSegs(&.{.{ .slot = slot, .ids = ids, .p0 = m.caches[slot].position }}, logits);
    }

    /// One forward over several slots' one-row windows (a shared round's lanes).
    pub fn forwardSegs(m: *Model, segs: []const Seg, logits: Logits) ![]const f32 {
        return forward_impl.forwardSegs(m, segs, logits);
    }

    /// The rows the host may hash.
    pub fn cpuHashRows() u32 {
        return model_hash.cpuHashRows();
    }
    pub fn hashPass(m: *Model, slot: u32, rows: u32) !void {
        return model_hash.hashPass(m, slot, rows);
    }
    pub fn hashPassSegs(m: *Model, segs: []const Seg, rows: u32) !void {
        return model_hash.hashPassSegs(m, segs, rows);
    }
    pub fn engramThreads(m: *Model) usize {
        return model_hash.engramThreads(m);
    }
    pub fn latchStage() bool {
        return model_hash.latchStage();
    }
    pub fn engramPrefetch() bool {
        return model_hash.engramPrefetch();
    }
    pub fn engramSmallThreads() usize {
        return model_hash.engramSmallThreads();
    }
    pub fn engramAdapt() bool {
        return model_hash.engramAdapt();
    }
    pub fn engramPendingOn() bool {
        return model_hash.engramPendingOn();
    }
    pub fn engramPipeOn() bool {
        return model_hash.engramPipeOn();
    }
    pub fn engramPipeRows(m: *Model, rows: u32) bool {
        return model_hash.engramPipeRows(m, rows);
    }
    pub fn prefetchPending(m: *Model, slot: u32, pending: u32) void {
        model_hash.prefetchPending(m, slot, pending);
    }
    pub fn stageFnv(bytes: []const u8) u64 {
        return model_hash.stageFnv(bytes);
    }
    pub fn stageHash(m: *Model, rows: u32, tag: []const u8) void {
        model_hash.stageHash(m, rows, tag);
    }
    pub fn stageLayer(m: *Model, rows: u32, li: usize) !void {
        return model_hash.stageLayer(m, rows, li);
    }
    pub fn stageRows(m: *Model, rows: u32) !void {
        return model_hash.stageRows(m, rows);
    }
    pub fn drafterId(m: *Model, token: u32) !View {
        return model_drafter.drafterId(m, token);
    }
    pub fn drafterRound(m: *Model, pending: u32, rows: u32) !View {
        return model_drafter.drafterRound(m, pending, rows);
    }
    pub fn markovPart(m: *Model) !View {
        return model_drafter.markovPart(m);
    }
    pub fn drafterState(m: *Model) !*kv.cache.State {
        return model_drafter.drafterState(m);
    }
    pub fn tap(m: *Model, enc: mtl.ComputeEncoder, x: View, rows: u32, slot: usize) !void {
        return model_drafter.tap(m, enc, x, rows, slot);
    }
    pub fn tapView(m: *const Model, slot: usize) View {
        return model_drafter.tapView(m, slot);
    }
    pub fn profileSpan(m: *Model, encoder: *mtl.ComputeEncoder, stage: gpu_counters.Stage) !void {
        return model_drafter.profileSpan(m, encoder, stage);
    }
    pub fn normalize(m: *Model, enc: mtl.ComputeEncoder, x: View, w: checkpoint.Tensor, y: View, rows: u32) void {
        model_drafter.normalize(m, enc, x, w, y, rows);
    }
    pub fn mixes(m: *Model, enc: mtl.ComputeEncoder, w: weights.Hc, pre: View, rows: u32) void {
        model_drafter.mixes(m, enc, w, pre, rows);
    }
    pub fn post(m: *Model, enc: mtl.ComputeEncoder, rows: u32) void {
        model_drafter.post(m, enc, rows);
    }
    pub fn project(m: *Model, enc: mtl.ComputeEncoder, matrix: exl3.Matrix, x: View, y: View, rows: u32, xs: usize, ys: usize, output: exl3.Output) !void {
        return model_drafter.project(m, enc, matrix, x, y, rows, xs, ys, output);
    }
    pub fn projectDense8(m: *Model, enc: mtl.ComputeEncoder, matrix: exl3.Matrix, x: View, y: View, rows: u32, xs: usize, ys: usize, output: exl3.Output) !void {
        return model_drafter.projectDense8(m, enc, matrix, x, y, rows, xs, ys, output);
    }
    pub fn projectDebug(m: *Model, enc: mtl.ComputeEncoder, matrix: exl3.Matrix, x: View, y: View, rows: u32, xs: usize, ys: usize, output: exl3.Output, b: mtl.Buffer) !void {
        return model_drafter.projectDebug(m, enc, matrix, x, y, rows, xs, ys, output, b);
    }
    pub fn projectDebugSums(m: *Model, enc: mtl.ComputeEncoder, matrix: exl3.Matrix, x: View, y: View, rows: u32, xs: usize, ys: usize, output: exl3.Output, b: mtl.Buffer) !void {
        return model_drafter.projectDebugSums(m, enc, matrix, x, y, rows, xs, ys, output, b);
    }
    pub fn expertPasses() u32 {
        return model_drafter.expertPasses();
    }
    pub fn moeTileRows() u32 {
        return model_drafter.moeTileRows();
    }
    pub fn draftAttention(m: *Model, enc0: mtl.ComputeEncoder, blk: *const weights.Layer, state: *kv.cache.State, slot: u32, layer: u32, rows: u32, position: u32, dbg: ?mtl.Buffer, prof: ?*Model.CondProf) !void {
        return model_draft_attention.draftAttention(m, enc0, blk, state, slot, layer, rows, position, dbg, prof);
    }

    /// The drafter's conditioning, split into its own command buffers.
    pub const CondProf = struct {
        queue: mtl.Queue,
        cb: mtl.CommandBuffer,
        enc: mtl.ComputeEncoder,
        on: bool,
        /// The BLOCKS' split (TF_DSV41_BLOCK_PROF=1) -- the drafter's three blocks have never been split.
        blk_on: bool = false,
        /// The block the parts belong to (the MoE's own calls do not know it).
        block: u32 = 0,
        total_ms: f64 = 0,
        parts: u32 = 0,

        /// End the buffer, commit it, wait, print the part's GPU window, and start a fresh buffer.
        pub fn split(self: *CondProf, tag: []const u8, label: []const u8, block: u32, enc: *mtl.ComputeEncoder) void {
            enc.end();
            self.cb.commit();
            self.cb.wait();
            const ms = self.cb.gpuSeconds() * 1e3;
            self.total_ms += ms;
            self.parts += 1;
            std.log.info("{s} part {s} b{d}: {d:.3} ms", .{ tag, label, block, ms });
            self.cb = self.queue.commandBuffer();
            const e = self.cb.compute(if (concurrentOn()) .concurrent else .serial);
            self.enc = e;
            enc.* = e;
        }
    };

    /// The adaptive depth policy's warm-up cache: a running mean over the first `k` samples, then the shipped EMA.
    pub var mtp_warm_cache: ?u32 = null;
    pub fn mtpWarmRounds() u32 {
        return model_drafter.mtpWarmRounds();
    }
    pub fn noteAcceptSlot(m: *Model, slot: u32, hit: bool) void {
        model_drafter.noteAcceptSlot(m, slot, hit);
    }
    pub fn noteAcceptRatioSlot(m: *Model, slot: u32, ratio: f64) void {
        model_drafter.noteAcceptRatioSlot(m, slot, ratio);
    }
    pub fn mtpRoundSlot(m: *Model, slot: u32) f64 {
        return model_drafter.mtpRoundSlot(m, slot);
    }
    pub fn mtpAcceptSlot(m: *Model, slot: u32) f64 {
        return model_drafter.mtpAcceptSlot(m, slot);
    }
    pub fn resetAcceptSlot(m: *Model, slot: u32) void {
        model_drafter.resetAcceptSlot(m, slot);
    }
    pub fn proposeTokens(m: *Model, slot: u32, pending: u32, count: u32) ![4]u32 {
        return model_propose.proposeTokens(m, slot, pending, count);
    }

    pub fn streamView(m: *const Model) View {
        return m.workspace.view(if (m.stream_alt) .stream_next else .stream);
    }
    /// One fused boundary: the pending branch expanded into the other stream buffer, its mix partials, then the split.
    pub fn boundary(m: *Model, enc: mtl.ComputeEncoder, bp: *const mhc.BoundaryPipelines, w: weights.Hc, pre_in: View, pre_out: View, norm: checkpoint.Tensor, rows: u32, with_mixes: bool) void {
        if (std.c.getenv("TF_DSV41_SKIP_MHC") != null) return;
        const ws = &m.workspace;
        const cur = if (m.stream_alt) ws.view(.stream_next) else ws.view(.stream);
        const next = if (m.stream_alt) ws.view(.stream) else ws.view(.stream_next);
        const expand = m.pending_expand;
        mhc.encodeBoundary(enc, bp, .{ .rows = rows, .expand = expand, .mixes = with_mixes, .eps = m.cfg.eps, .hc_eps = m.cfg.hc_eps, .iters = m.cfg.hc_sinkhorn_iters }, .{
            .x_old = cur,
            .x_new = next,
            .branch = ws.view(.branch),
            .post = ws.view(.post),
            .comb = ws.view(.comb),
            .fn_ = tensor(w.projection),
            .part = ws.view(.hc_part),
            .base = tensor(w.base),
            .scale = tensor(w.scale),
            .pre_in = pre_in,
            .norm = tensor(norm),
            .normed = ws.view(.normalized),
            .pre_out = pre_out,
            .post_out = ws.view(.post),
            .comb_out = ws.view(.comb),
        });
        if (expand) m.stream_alt = !m.stream_alt;
        m.pending_expand = false;
    }

    /// TF_DSV41_GATE_TREE=0 forces the shipped tf-dense-col gate; the default is the bit-exact tiled tree.
    pub fn gateTreeOff() bool {
        const S = struct {
            var cached: ?bool = null;
        };
        if (S.cached) |v| return v;
        var v = false;
        if (std.c.getenv("TF_DSV41_GATE_TREE")) |e| v = std.mem.eql(u8, std.mem.span(e), "0");
        S.cached = v;
        return v;
    }

    /// TF_DSV41_GATE_WIDE=1 takes the wide (staged) gate of, whose order and logits are the shipped ones.
    pub fn gateWideOn() bool {
        const S = struct {
            var cached: ?bool = null;
        };
        if (S.cached) |v| return v;
        var v = false;
        if (std.c.getenv("TF_DSV41_GATE_WIDE")) |e| v = !std.mem.eql(u8, std.mem.span(e), "0");
        S.cached = v;
        return v;
    }

    pub fn moe(m: *Model, enc: *mtl.ComputeEncoder, e: ?*mtl.CommandBuffer, w: *const weights.Moe, rows: u32, layer_index: u32, topk: u32, prof: ?*CondProf) !void {
        return moe_impl.moe(m, enc, e, w, rows, layer_index, topk, prof);
    }
};

pub fn tensor(t: checkpoint.Tensor) View {
    return .{ .buffer = t.buffer, .offset = t.offset };
}
pub fn asF32(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

pub fn convert(v: View) exl3.View {
    return .{ .buffer = v.buffer, .offset = v.offset };
}

/// TF_DSV41_STAGE_CB: close the command buffer at a stage boundary, log its GPU seconds under `label`.
pub var split_stage_dispatches: u64 = 0;
pub var split_stage_threadgroups: u64 = 0;
pub var gate_dump_done: bool = false;
pub var route_ids_done_pair: bool = false;
pub var route_ids_done_prompt: bool = false;
pub var plan_hist_done: bool = false;
pub var shared_split_logged: bool = false;
pub var gate_dump_done2: bool = false;
/// The tree-price instrument's round join (verify round R <-> proposal round R).
pub var price_round: u64 = 0;
pub var gate_dump_done3: bool = false;
pub fn splitStage(m: *Model, enc: *mtl.ComputeEncoder, e: ?*mtl.CommandBuffer, label: []const u8, index: usize) !void {
    if (std.c.getenv("TF_DSV41_STAGE_CB") == null) return;
    if (e == null) return;
    mtl.command.count_enabled = true;
    const dispatches = mtl.command.dispatch_count.load(.monotonic);
    enc.end();
    e.?.commit();
    e.?.wait();
    try completed(e.?.*);
    // The threadgroup count beside the dispatch count -- the parallelism ledger.
    const groups = mtl.command.dispatch_threadgroups.load(.monotonic);
    std.log.info("stage {s} layer {d}: gpu {d:7.2} ms, dispatches {d}, threadgroups {d}", .{ label, index, e.?.gpuSeconds() * 1e3, dispatches - split_stage_dispatches, groups - split_stage_threadgroups });
    split_stage_dispatches = dispatches;
    split_stage_threadgroups = groups;
    e.?.* = m.queue.commandBufferUnretained();
    enc.* = e.?.compute(.serial);
    for (m.checkpoint.shards.items) |shard| enc.useResource(shard.buffer, mtl.ResourceUsage.read);
}

pub fn completed(cb: mtl.CommandBuffer) !void {
    if (cb.failure()) |message| {
        std.log.err("DeepSeek V4.1 GPU execution failed: {s}", .{message});
        return error.GpuExecutionFailed;
    }
}
