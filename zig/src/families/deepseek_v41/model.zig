// ! Plain V4.1 execution over resident EXL3 weights and source-owned packed caches.
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
const model_lifecycle = @import("model_lifecycle.zig");
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
    /// The gate GEMM's f16 activation copy when the pack's router gate is f16 (TF_DSV41_ROUTE_GEMM). One buffer sized
    /// to the chunk, reused a layer: bf16 -> f16 is exact, and the weight keeps its own 10 bits.
    gate_x: ?mtl.Buffer = null,
    /// The gate capture's staging buffers (TF_DSV41_GATE_DUMP): one pair, reused every layer and every token, because
    /// a buffer freed while a command buffer still names it is a GPU invalid resource.
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
    /// The last row the conditioning taps captured: the drafter reads that row's three layers. The tap ring's next
    /// write row and how many of its 128 rows are live (a rolling window across forwards).
    taps_row: u32 = 0,
    /// Rows the last tap capture kept (1..128): the draft conditioning's row count.
    taps_n: u32 = 0,
    /// The rows the LAST tap capture kept. They are the tail of the taps ring and they are consecutive positions, so
    /// the conditioning can publish the window's missing positions at once.
    taps_rows_last: u32 = 0,
    /// The rows of the LAST tap capture that are valid (0 = all of them). The rollback keeps a prefix of the verify's
    /// capture: the rejected rows' taps are not the committed history, so the conditioning publishes the prefix, not
    /// the capture's tail.
    taps_rows_valid: u32 = 0,
    /// The adaptive depth policy's signal -- the accepted drafts a round, smoothed. The per-lane signal (the first
    /// draft's acceptance) and its counters.
    mtp_accept_slot: [32]f64 = @splat(0.0),
    mtp_accept_seen_slot: [32]u32 = @splat(0),
    /// The round's per-draft acceptance (kept/offered), the hybrid policy's stay-deep signal.
    mtp_round_slot: [32]f64 = @splat(0.0),
    mtp_round_seen_slot: [32]u32 = @splat(0),
    /// The ring's write row for the CURRENT forward. The three target layers SHARE it: the taps buffer is interleaved
    /// by position (row p = the three layers' values), so the capture's ring advances ONCE a forward.
    taps_base: u32 = 0,
    /// The 128-row conditioning scratch (rows x 5120 bf16) and its packed keys (rows x 528), allocated with the taps.
    cond: ?mtl.Buffer = null,
    cond_wkv: ?mtl.Buffer = null,
    mtp_drafts: [4]u32 = .{ 0, 0, 0, 0 },
    mtp_count: u32 = 0,
    /// The draft round's own logits scratch (eight rows), so its forward never writes the workspace's logits field.
    mtp_logits: ?mtl.Buffer = null,
    /// The drafter's router logits (its own 128 columns a row; the backbone's 384-wide rows would overlap the
    /// drafter's five).
    dspark_logits: ?mtl.Buffer = null,
    /// The trunk's own head input (the verify's row 0), for the drafter-vs-trunk hidden probe.
    mtp_trunk_hidden: ?mtl.Buffer = null,
    /// Each slot's Engram cursor as of its beginSlot, restored by rollbackSlot.
    engram_stash: []u8,
    /// Each slot's compressed-path partial scratch as of its beginSlot (TF_DSV41_PARTIAL_TXN), restored by
    /// rollbackSlot: a rejected window leaves the half-block accumulator holding its last draft row, and the re-feed
    /// of the block that straddles the rollback position would read it.
    partial_stash: []u8,
    /// Each slot's stash of the window ring's next `max_ring_rows` slots, taken before a verify and put back for the
    /// rows the round rejects. `lanes * layers * max_ring_rows * ring_stride` bytes.
    ring_stash: mtl.Buffer,
    /// A MERGED TAIL CHUNK'S STASH. When a pass runs its tail mark inside the chunk (no second forward), the chunk
    /// publishes its PREFIX -- the rows before the mark -- into this slot's copy of the window ring and the taps
    /// ring, then publishes the whole chunk into the live ones as usual; the state at the mark is then copied from
    /// the stash. `publish_limit` is the prefix row count while such a chunk runs (0: no merged chunk), and
    /// `snap_pos`/`snap_hist`/`snap_taps_*` are the mark's re-pointed metadata.
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
    /// Layer 0's wq_b input (the post-q_norm `qr`), for the GPTQ calibration dump. Null unless TF_DSV41_DUMP_WQB_X is
    /// set.
    wqb_x_debug: ?mtl.Buffer = null,
    /// Layer 0's MoE input (the experts' output-error dump).
    moe_x_debug: ?mtl.Buffer = null,
    moe_x_rows: u32 = 0,
    pipelines: Pipelines,
    admitted_bytes: usize,
    load_seconds: f64 = 0,
    last_gpu_seconds: f64 = 0,
    logits_rows: usize = 0,
    valid: bool = true,
    profile: ?*gpu_counters.Recorder = null,
    /// The fused boundary's stream parity and the pending expand (TF_DSV41_HC_BOUNDARY).
    stream_alt: bool = false,
    pending_expand: bool = false,
    host_prof: bool = false,
    host_hash_ns: u64 = 0,
    host_stage_ns: u64 = 0,
    host_stage_misses: u64 = 0,
    host_stage_faults: u64 = 0,
    host_stage_pf: u8 = 0,
    host_stage_thr: u16 = 0,
    /// The adaptive Engram gather's state. The decode-sized list starts single-threaded; two consecutive slow small
    /// gathers (cold pages) switch it to the mapped path's workers, and every 16th parallel call one single-threaded
    /// gather probes whether the stream has turned warm. So the warm band never leaves the old path.
    /// TF_DSV41_ENGRAM_ADAPT=0 is the same-binary control.
    engram_par: bool = false,
    /// The thread state machine's probe is not the arbiter: a prefetched window makes the next window's probe look
    /// warm. TF_DSV41_ENGRAM_PF=0 disables the whole arm.
    engram_pf_on: bool = false,
    engram_pf_off: u8 = 0,
    engram_probe_in: u8 = 0,
    engram_probe_every: u8 = 16,
    engram_slow_n: u8 = 0,
    /// The second Engram layer's prompt gather is deferred to its own layer, so the GPU runs the earlier layers while
    /// the read happens. Set by `stageRows` on a prompt-sized chunk.
    engram_pending: bool = false,
    profile_stage_dispatch: [gpu_counters.stage_count]u64 = @splat(0),
    profile_last_count: u64 = 0,
    profile_open_stage: ?gpu_counters.Stage = null,
    grouped_prefill: bool = true,
    observer: ?Observer = null,
    /// The attention's own per-stage observer (the `stagediff` harness). Null on every default path; installing it
    /// only adds captures, never arithmetic.
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

    /// One forward over several slots' one-row windows (a shared round's lanes). The row-wise layers run once over
    /// every segment's rows; the attention core runs once a segment against its own slot state. One-row segments
    /// only, and no instruments: the rung exists to amortise the dense attention weights, not to batch prefill or to
    /// run a profiler.
    pub fn forwardSegs(m: *Model, segs: []const Seg, logits: Logits) ![]const f32 {
        return forward_impl.forwardSegs(m, segs, logits);
    }

    /// The rows the host may hash. The decode/verify window (<= 8 rows) pays the GPU's commit+wait before its ids are
    /// readable on the host; its hash is 48 integer reductions a row, so the host computes it where the staging reads
    /// it and the round trip disappears. The prompt's thousands-row chunks are cheaper on the GPU.
    /// `TF_DSV41_CPU_HASH=0` is the shipped GPU pass.
    pub fn cpuHashRows() u32 {
        const S = struct {
            var cached: i64 = -1;
        };
        if (S.cached < 0) S.cached = if (std.c.getenv("TF_DSV41_CPU_HASH")) |e|
            (std.fmt.parseInt(i64, std.mem.span(e), 10) catch 8)
        else
            8;
        return @intCast(@max(0, S.cached));
    }

    /// The streamed path's own hash pass: the ids must be readable on the host, so the shipped form is a second
    /// command buffer; a decode-sized window takes the host's own copy of the same integer hash.
    pub fn hashPass(m: *Model, slot: u32, rows: u32) !void {
        const p = &m.pipelines;
        const ws = &m.workspace;
        if (rows <= cpuHashRows()) {
            try engram.encodeHashHost(try ws.historyView(slot), ws.view(.ids), ws.view(.token_map), ws.view(.engram_ids_0), ws.view(.engram_ids_1), rows);
            try m.stageRows(rows);
            return;
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const e = m.queue.commandBufferUnretained();
        var enc = e.compute(.serial);
        for (engram.hash.layers, 0..) |meta, index| {
            try engram.encodeHash(enc, &p.engram, meta, ws.view(.ids), ws.view(.token_map), null, try ws.historyView(slot), ws.view(if (index == 0) .engram_ids_0 else .engram_ids_1), rows);
        }
        try engram.encodeCommit(enc, &p.engram, ws.view(.ids), ws.view(.token_map), null, try ws.historyView(slot), rows);
        enc.end();
        e.commit();
        e.wait();
        try completed(e);
        try m.stageRows(rows);
    }

    /// The batch's hash pass -- each segment's ids at its own rows, against its own slot's Engram history, then the
    /// one staging read over all rows.
    pub fn hashPassSegs(m: *Model, segs: []const Seg, rows: u32) !void {
        const ws = &m.workspace;
        var r0: usize = 0;
        for (segs) |sg| {
            const cols: usize = engram.hash.columns;
            try engram.encodeHashHost(try ws.historyView(sg.slot), ws.view(.ids).shifted(r0 * 4), ws.view(.token_map), ws.view(.engram_ids_0).shifted(r0 * cols * 4), ws.view(.engram_ids_1).shifted(r0 * cols * 4), @intCast(sg.ids.len));
            r0 += sg.ids.len;
        }
        try m.stageRows(rows);
    }

    /// Both Engram layers' rows for this window, read from the release's shards into the staging buffer. Threads the
    /// Engram gather splits its ids across (TF_DSV41_ENGRAM_THREADS, 1 keeps one thread). The default follows the
    /// read path: the mapped gather reads by address, so its workers scale; the positional reads share one I/O
    /// context and serialise on it, and one thread is their best case.
    pub fn engramThreads(m: *Model) usize {
        // The mapped default is sixteen for the large gather too.
        var v: usize = if (m.stream) |s| (if (s.mapped()) 16 else 1) else 1;
        if (std.c.getenv("TF_DSV41_ENGRAM_THREADS")) |e| {
            v = std.fmt.parseInt(usize, std.mem.span(e), 10) catch v;
            if (v == 0 or v > 64) v = 1;
        }
        return v;
    }

    /// The latch's coldness clock. Default: the whole stage (prefetch + gather). `=gather` restores The post-prefetch
    /// signal (the same-binary control).
    pub fn latchStage() bool {
        const S = struct {
            var cached: i8 = 0;
        };
        if (S.cached == 0) S.cached = if (std.c.getenv("TF_DSV41_ENGRAM_LATCH")) |e|
            (if (std.mem.eql(u8, std.mem.span(e), "gather")) -1 else 1)
        else
            1;
        return S.cached > 0;
    }

    /// The whole-window page-in (TF_DSV41_ENGRAM_PF). Default on; =0 is the same-binary control.
    pub fn engramPrefetch() bool {
        const S = struct {
            var cached: i8 = 0;
        };
        if (S.cached == 0) S.cached = if (std.c.getenv("TF_DSV41_ENGRAM_PF")) |e|
            (if (std.mem.eql(u8, std.mem.span(e), "0")) -1 else 1)
        else
            1;
        return S.cached > 0;
    }

    /// The decode-sized list's workers. Sixteen is the gather's own thread table's height.
    /// TF_DSV41_ENGRAM_SMALL_THREADS is the arm (1..16).
    pub fn engramSmallThreads() usize {
        const S = struct {
            var cached: usize = 0;
        };
        if (S.cached != 0) return S.cached;
        var v: usize = 16;
        if (std.c.getenv("TF_DSV41_ENGRAM_SMALL_THREADS")) |e| {
            v = std.fmt.parseInt(usize, std.mem.span(e), 10) catch v;
            if (v == 0 or v > 16) v = 16;
        }
        S.cached = v;
        return v;
    }

    pub fn engramAdapt() bool {
        const S = struct {
            var cached: i8 = 0;
        };
        if (S.cached == 0) S.cached = if (std.c.getenv("TF_DSV41_ENGRAM_ADAPT")) |e|
            (if (std.mem.eql(u8, std.mem.span(e), "0")) -1 else 1)
        else
            1;
        return S.cached > 0;
    }

    /// Warm the pending row's Engram pages while the drafter's own forward runs. The pending token is known before
    /// `proposeTokens` and is the verify window's FIRST row, so its ids can be hashed from the standing record (a
    /// peek that does not advance it) and its pages handed to madvise now; the next verify's `stageRows` recomputes
    /// the same ids and finds the pages resident. Only the page-in moves: no id, byte, reduction or reply changes.
    /// `TF_DSV41_ENGRAM_PEND=0` is the same-binary control.
    pub fn engramPendingOn() bool {
        const S = struct {
            var cached: i8 = 0;
        };
        if (S.cached == 0) S.cached = if (std.c.getenv("TF_DSV41_ENGRAM_PEND")) |e|
            (if (std.mem.eql(u8, std.mem.span(e), "0")) -1 else 1)
        else
            1;
        return S.cached > 0;
    }

    /// Stage the second Engram layer's rows at its own layer instead of before the forward. The ids are already known
    /// (the hash pass ran); only the read's clock moves. `=0` is the control.
    pub fn engramPipeOn() bool {
        // Read per forward: the same process can run the control and the arm.
        if (std.c.getenv("TF_DSV41_ENGRAM_PIPE")) |e| return !std.mem.eql(u8, std.mem.span(e), "0");
        return true;
    }

    pub fn engramPipeRows(m: *Model, rows: u32) bool {
        if (rows <= 8 or !engramPipeOn()) return false;
        if (m.stream == null or m.stream_stage == null) return false;
        if (engram.hash.layers.len != 2) return false;
        if (engram.hash.layers[1].layer_id <= engram.hash.layers[0].layer_id) return false;
        if (m.profile != null or m.observer != null) return false;
        return std.c.getenv("TF_DSV41_LAYER_CB") == null and
            std.c.getenv("TF_DSV41_STAGE_CB") == null and
            std.c.getenv("TF_DSV41_ATTN_PROF") == null;
    }

    pub fn prefetchPending(m: *Model, slot: u32, pending: u32) void {
        const src = m.stream orelse return;
        if (!src.mapped()) return;
        const ws = &m.workspace;
        const history = ws.historyView(slot) catch return;
        var ids: [2][engram.hash.columns]u32 = undefined;
        engram.peekHashHost(history, ws.view(.token_map), pending, &ids) catch return;
        for ([_]work.Field{ .engram_ids_0, .engram_ids_1 }, 0..) |_, li| src.prefetch(li, &ids[li]);
    }

    pub fn stageFnv(bytes: []const u8) u64 {
        var h: u64 = 0xcbf29ce484222325;
        for (bytes) |b| {
            h ^= b;
            h *%= 0x100000001b3;
        }
        return h;
    }

    /// `TF_DSV41_ENGRAM_STAGE_HASH=1`; nothing in the default path reads this.
    pub fn stageHash(m: *Model, rows: u32, tag: []const u8) void {
        if (std.c.getenv("TF_DSV41_ENGRAM_STAGE_HASH") == null) return;
        const buf = m.stream_stage orelse return;
        const ws = &m.workspace;
        const layer_bytes = @as(usize, rows) * engram.gather_dim * @sizeOf(u16);
        const base: [*]const u8 = buf.contents();
        const n_ids: usize = @as(usize, rows) * engram.hash.columns;
        const i0v = ws.view(.engram_ids_0);
        const i1v = ws.view(.engram_ids_1);
        const ids_a: [*]const u32 = @ptrCast(@alignCast(i0v.buffer.contents() + i0v.offset));
        const ids_b: [*]const u32 = @ptrCast(@alignCast(i1v.buffer.contents() + i1v.offset));
        std.log.info("engram stage hash {s} rows {d} ids0 {x} ids1 {x} staged0 {x} staged1 {x}", .{
            tag,                                             rows,
            stageFnv(std.mem.sliceAsBytes(ids_a[0..n_ids])), stageFnv(std.mem.sliceAsBytes(ids_b[0..n_ids])),
            stageFnv(base[0..layer_bytes]),                  stageFnv(base[layer_bytes..][0..layer_bytes]),
        });
    }

    pub fn stageLayer(m: *Model, rows: u32, li: usize) !void {
        const src = m.stream orelse return;
        const buf = m.stream_stage orelse return error.EngramNotOpen;
        const ws = &m.workspace;
        const layer_bytes = @as(usize, rows) * engram.gather_dim * @sizeOf(u16);
        const base: [*]u8 = buf.contents();
        const n_ids: usize = @as(usize, rows) * engram.hash.columns;
        const v = ws.view(if (li == 0) work.Field.engram_ids_0 else work.Field.engram_ids_1);
        const at: [*]const u32 = @ptrCast(@alignCast(v.buffer.contents() + v.offset));
        const out = std.mem.bytesAsSlice(u16, base[li * layer_bytes ..][0..layer_bytes]);
        try src.rowsU32Parallel(li, at[0..n_ids], engram.head_dim, out, engramThreads(m));
    }

    pub fn stageRows(m: *Model, rows: u32) !void {
        const stage_start: u64 = if (m.host_prof) nowNs() else 0;
        defer if (m.host_prof) {
            m.host_stage_ns += nowNs() - stage_start;
        };
        m.stageHash(rows, "pre");
        const src = m.stream orelse return;
        const buf = m.stream_stage orelse return error.EngramNotOpen;
        const ws = &m.workspace;
        const cols: usize = engram.hash.columns;
        const layer_bytes = @as(usize, rows) * engram.gather_dim * @sizeOf(u16);
        const base: [*]u8 = buf.contents();
        const n_ids: usize = @as(usize, rows) * cols;
        const small = n_ids < 512;
        // The latch's clock is taken HERE, before the page-in. This clock sees prefetch + gather.
        const w0: u64 = if (small and latchStage()) nowNs() else 0;
        const adapt = small and engramAdapt();
        var threads: usize = engramThreads(m);
        var probed = false;
        if (small) {
            if (!adapt) {
                threads = 1;
            } else if (m.engram_par and m.engram_probe_in > 0) {
                m.engram_probe_in -= 1;
                threads = engramSmallThreads();
            } else {
                threads = 1;
                probed = true;
            }
        }
        // The window's page-ins, only where they pay. The prompt's rows are contiguous and the kernel already
        // readahead-s them, so the large path is left alone; a single-threaded probe must not be prefetched (its time
        // is the page-cache test); the warm band never reaches here.
        const pf_start: u64 = if (small and engramPrefetch()) nowNs() else 0;
        // The stage's own major faults, for the HOST_PROF line; the latch below uses the gather's clock.
        const ru0 = std.posix.getrusage(std.posix.rusage.SELF);
        const pf_on = small and m.engram_pf_on and pf_start != 0;
        if (m.host_prof) {
            m.host_stage_pf = if (pf_on) 1 else 0;
            m.host_stage_thr = @intCast(threads);
        }
        if (pf_on) {
            for ([_]work.Field{ .engram_ids_0, .engram_ids_1 }, 0..) |field, li| {
                const v = ws.view(field);
                const at: [*]const u32 = @ptrCast(@alignCast(v.buffer.contents() + v.offset));
                src.prefetch(li, at[0..n_ids]);
            }
        }
        const t0: u64 = if (adapt) nowNs() else 0;
        const g0: u64 = if (small) nowNs() else 0;
        const misses0: u64 = if (m.host_prof) src.stats.misses else 0;
        if (!small and engramPipeRows(m, rows)) {
            // The first layer now, the second at its own layer (the GPU runs the earlier layers while that read
            // happens). Same ids, same bytes, same decode -- only the clock moves.
            try m.stageLayer(rows, 0);
            m.engram_pending = true;
        } else {
            for ([_]work.Field{ .engram_ids_0, .engram_ids_1 }, 0..) |field, li| {
                const v = ws.view(field);
                const at: [*]const u32 = @ptrCast(@alignCast(v.buffer.contents() + v.offset));
                const ids = at[0..n_ids];
                const out = std.mem.bytesAsSlice(u16, base[li * layer_bytes ..][0..layer_bytes]);
                try src.rowsU32Parallel(li, ids, engram.head_dim, out, threads);
            }
        }
        m.stageHash(rows, "stage");
        if (adapt) {
            const dt = nowNs() - t0;
            if (probed) {
                if (dt > 1_500_000) {
                    m.engram_slow_n += 1;
                    if (m.engram_slow_n >= 2) {
                        m.engram_par = true;
                        m.engram_probe_in = m.engram_probe_every;
                        m.engram_probe_every = if (m.engram_probe_every >= 128) 128 else m.engram_probe_every * 2;
                        m.engram_slow_n = 0;
                    }
                } else {
                    m.engram_slow_n = 0;
                    m.engram_par = false;
                    m.engram_probe_in = 0;
                    m.engram_probe_every = 16;
                }
            }
        }
        if (g0 != 0) {
            // The latch's signal is the gather's own clock, taken AFTER the prefetch: the fault counters cannot
            // serve, because a prefetched window's I/O is the madvise call and its reads are then minor faults, so
            // the page-in hides the very coldness the latch must see. The off-side is delayed three windows:
            // consecutive decode windows share most of their n-grams, so one resident window between two cold ones is
            // the norm, not the stream turning warm.
            const gather = nowNs() - (if (w0 != 0) w0 else g0);
            if (gather > 400_000) {
                m.engram_pf_on = true;
                m.engram_pf_off = 0;
            } else if (gather < 200_000) {
                if (m.engram_pf_off < 3) m.engram_pf_off += 1;
                if (m.engram_pf_off >= 3) m.engram_pf_on = false;
            } else {
                m.engram_pf_off = 0;
            }
        }
        if (m.host_prof) {
            m.host_stage_misses +%= src.stats.misses -% misses0;
            const faults1: i64 = std.posix.getrusage(std.posix.rusage.SELF).majflt;
            if (faults1 > ru0.majflt) m.host_stage_faults +%= @intCast(faults1 - ru0.majflt);
        }
        src.recordWindow(rows);
    }

    /// Copy `rows` rows of the running hidden state into tap `slot` (0..2 for the layers 37, 38 and 39 the drafter's
    /// main_proj reads concatenated). A prompt chunk has thousands of rows and drafts nothing. The drafter's token
    /// id, written where the embedding kernel reads it (the buffer is CPU-shared).
    pub fn drafterId(m: *Model, token: u32) !View {
        if (m.drafter_ids == null) {
            m.drafter_ids = try m.device.buffer(4, mtl.ResourceOptions.shared);
            m.markov_ids = try m.device.buffer(8 * 4, mtl.ResourceOptions.shared); // five Markov steps, a u32 each
            m.residency.add(m.drafter_ids.?);
        }
        std.mem.writeInt(u32, m.drafter_ids.?.contents()[0..4], token, .little);
        return .{ .buffer = m.drafter_ids.? };
    }

    /// The drafter's round of ids: the pending token, then the release's noise token (128799) a row, as
    /// `DSparkBlock.forward_embed` fills `draft_input_ids` (inference/model.py, revision 314113f8).
    pub fn drafterRound(m: *Model, pending: u32, rows: u32) !View {
        if (m.drafter_ids == null) {
            if (m.drafter_ids) |old| old.deinit();
            // The buffer is sized for the widest form (the five-row arm), not for the rows the first call happened to
            // use, so the arm cannot overflow it.
            m.drafter_ids = try m.device.buffer(@max((rows + 1) * 4, 6 * 4), mtl.ResourceOptions.shared);
            m.residency.add(m.drafter_ids.?);
        }
        const bytes = m.drafter_ids.?.contents();
        var r: u32 = 0;
        while (r < rows) : (r += 1)
            std.mem.writeInt(u32, bytes[@as(usize, r) * 4 ..][0..4], if (r == 0) pending else 128799, .little);
        return .{ .buffer = m.drafter_ids.? };
    }

    /// The Markov chain's partial bests: ceil(vocab/256) float2 entries, allocated once.
    pub fn markovPart(m: *Model) !View {
        if (m.markov_part == null) {
            const groups: usize = (129280 + 255) / 256;
            m.markov_part = try m.device.buffer(groups * 8, mtl.ResourceOptions.shared);
            m.residency.add(m.markov_part.?);
        }
        return .{ .buffer = m.markov_part.? };
    }

    /// The drafter's state: `drafter_layers` rings, allocated once and kept for the model's life.
    pub fn drafterState(m: *Model) !*kv.cache.State {
        if (m.drafter_state) |state| return state;
        const state = try m.gpa.create(kv.cache.State);
        errdefer m.gpa.destroy(state);
        state.* = try kv.cache.State.init(m.device, .{ .capacity = m.cache.plan.capacity, .layers = kv.cache.drafter_layers });
        m.drafter_state = state;
        return state;
    }

    pub fn tap(m: *Model, enc: mtl.ComputeEncoder, x: View, rows: u32, slot: usize) !void {
        if (rows > 16 or !tapsOn()) return;
        ops.encodeCopy(enc, &m.pipelines.ops, x, m.tapView(slot), .{
            .rows = rows,
            .width = 5120,
            .x_stride = 5120,
            .y_stride = 5120,
        });
        enc.barrier();
    }
    pub fn tapView(m: *const Model, slot: usize) View {
        return .{ .buffer = m.taps, .offset = slot * 16 * 5120 * 2 };
    }
    pub fn profileSpan(m: *Model, encoder: *mtl.ComputeEncoder, stage: gpu_counters.Stage) !void {
        const recorder = m.profile orelse return;
        if (!recorder.enabled()) return;
        const counted = mtl.command.dispatch_count.load(.monotonic);
        if (m.profile_open_stage) |prev| m.profile_stage_dispatch[@backingInt(prev)] += counted - m.profile_last_count;
        if (recorder.activeSpan()) try recorder.endSpan(encoder);
        try recorder.beginSpan(encoder, stage);
        m.profile_open_stage = stage;
        m.profile_last_count = counted;
        for (m.checkpoint.shards.items) |shard| encoder.useResource(shard.buffer, mtl.ResourceUsage.read);
    }

    pub fn normalize(m: *Model, enc: mtl.ComputeEncoder, x: View, w: checkpoint.Tensor, y: View, rows: u32) void {
        ops.encodeRms(enc, &m.pipelines.ops, x, tensor(w), y, .{ .rows = rows, .width = 5120, .x_stride = 5120, .y_stride = 5120, .eps = m.cfg.eps });
        enc.barrier();
    }

    pub fn mixes(m: *Model, enc: mtl.ComputeEncoder, w: weights.Hc, pre: View, rows: u32) void {
        if (std.c.getenv("TF_DSV41_SKIP_MHC") != null) return;
        const ws = &m.workspace;
        mhc.encodeRms(enc, &m.pipelines.mhc, ws.view(.stream), ws.view(.inv), .{ .rows = rows, .width = 5120, .stream_stride = 20480, .eps = m.cfg.eps });
        mhc.encodeMixDense(enc, &m.pipelines.mhc, ws.view(.stream), tensor(w.projection), ws.view(.mixes), .{ .rows = rows, .k = 20480, .n = 24, .x_stride = 20480 });
        enc.barrier();
        mhc.encodeSplit(enc, &m.pipelines.mhc, ws.view(.mixes), ws.view(.inv), tensor(w.scale), tensor(w.base), pre, ws.view(.post), ws.view(.comb), .{ .rows = rows, .eps = m.cfg.hc_eps });
        enc.barrier();
    }

    pub fn post(m: *Model, enc: mtl.ComputeEncoder, rows: u32) void {
        if (std.c.getenv("TF_DSV41_SKIP_MHC") != null) return;
        const ws = &m.workspace;
        mhc.encodePost(enc, &m.pipelines.mhc, ws.view(.branch), ws.view(.stream), ws.view(.post), ws.view(.comb), ws.view(.stream), .{ .rows = rows, .width = 5120, .branch_stride = 5120, .residual_stride = 20480, .y_stride = 20480 });
        enc.barrier();
    }

    pub fn project(m: *Model, enc: mtl.ComputeEncoder, matrix: exl3.Matrix, x: View, y: View, rows: u32, xs: usize, ys: usize, output: exl3.Output) !void {
        // The prompt-row int8 twin. Decode and every small chunk keep the trellis, so the gate is the call's own row
        // count; the sidecar is attached only for the attention's dense projections.
        if (matrix.int8) |side| {
            if (rows >= 1024 and std.c.getenv("TF_DSV41_I8_PREFILL") != null) {
                try exl3.encodePrefillInt8(&m.pipelines.projection, enc, matrix, side, convert(x), convert(y), m.workspace.projection(), .{ .count = rows, .x_stride = xs, .y_stride = ys, .output = output }, matrix.int8_direct);
                enc.barrier();
                return;
            }
        }
        try exl3.encode(&m.pipelines.projection, enc, matrix, convert(x), convert(y), m.workspace.projection(), .{ .count = rows, .x_stride = xs, .y_stride = ys, .output = output });
        enc.barrier();
    }

    /// The same projection with the CALLER's eight-row dense hint -- the tensor-unit dense tile from eight rows up
    /// instead of the shipped sixty-four.
    pub fn projectDense8(m: *Model, enc: mtl.ComputeEncoder, matrix: exl3.Matrix, x: View, y: View, rows: u32, xs: usize, ys: usize, output: exl3.Output) !void {
        try exl3.encode(&m.pipelines.projection, enc, matrix, convert(x), convert(y), m.workspace.projection(), .{ .count = rows, .x_stride = xs, .y_stride = ys, .output = output, .dense8 = true });
        enc.barrier();
    }

    /// The projection's own three views -- the input it reads, the rotated input it leaves in the scratch, and the
    /// sums the GEMM wrote -- dumped where the call made them, so a row's fault is localised to the stage that
    /// produced it.
    pub fn projectDebug(m: *Model, enc: mtl.ComputeEncoder, matrix: exl3.Matrix, x: View, y: View, rows: u32, xs: usize, ys: usize, output: exl3.Output, b: mtl.Buffer) !void {
        try exl3.encode(&m.pipelines.projection, enc, matrix, convert(x), convert(y), m.workspace.projection(), .{ .count = rows, .x_stride = xs, .y_stride = ys, .output = output });
        enc.barrier();
        const sc = m.workspace.projection();
        ops.encodeDump(enc, &m.pipelines.ops, x, .{ .buffer = b, .offset = mtp_dbg_pin * 4 }, @intCast(rows * xs), true);
        const rot = View{ .buffer = sc.buffer, .offset = sc.offset };
        ops.encodeDump(enc, &m.pipelines.ops, rot, .{ .buffer = b, .offset = mtp_dbg_prot * 4 }, @intCast(rows * matrix.k), true);
        const sums_off = (@as(usize, rows) * matrix.k * 2 + 15) & ~@as(usize, 15);
        const sums = View{ .buffer = sc.buffer, .offset = sc.offset + sums_off };
        ops.encodeDump(enc, &m.pipelines.ops, sums, .{ .buffer = b, .offset = mtp_dbg_psum * 4 }, @intCast(rows * matrix.n), false);
        enc.barrier();
    }

    /// The wo_a group's own GEMM sums, before the inverse rotation, where the overflow is born.
    pub fn projectDebugSums(m: *Model, enc: mtl.ComputeEncoder, matrix: exl3.Matrix, x: View, y: View, rows: u32, xs: usize, ys: usize, output: exl3.Output, b: mtl.Buffer) !void {
        try exl3.encode(&m.pipelines.projection, enc, matrix, convert(x), convert(y), m.workspace.projection(), .{ .count = rows, .x_stride = xs, .y_stride = ys, .output = output });
        enc.barrier();
        const sc = m.workspace.projection();
        const sums_off = (@as(usize, rows) * matrix.k * 2 + 15) & ~@as(usize, 15);
        const sums = View{ .buffer = sc.buffer, .offset = sc.offset + sums_off };
        ops.encodeDump(enc, &m.pipelines.ops, sums, .{ .buffer = b, .offset = mtp_dbg_woasums * 4 }, @intCast(rows * matrix.n), false);
        enc.barrier();
    }

    /// Rows a routed expert's tensor-unit tile covers (TF_DSV41_MOE_TILE_ROWS, 0 = the kernel's 64). A taller tile
    /// decodes each weight once for more rows; it is the routed stage's one free shape.
    pub fn expertPasses() u32 {
        const v = std.c.getenv("TF_DSV41_DOUBLE_EXPERTS") orelse return 1;
        const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 2;
        return @max(1, @min(n, 8));
    }

    pub fn moeTileRows() u32 {
        // The grouped plan pads each expert's block to the tile's own width, so the plan's tile and the routed GEMM's
        // tile are one number (64 by default, 128 for the taller tile).
        return expert_groups.tileRows();
    }

    /// The drafter block's attention: the four projections with their two norms, the round's own keys packed into the
    /// chunk, the scan over the stage's ring plus that chunk with no compressor and no indexer, the eight `wo_a`
    /// slices, `wo_b`, and the publish. The order is the release's `DSparkAttention.forward` (inference/model.py,
    /// revision 314113f8) with our own ring and shapes; its big scratch lives in `.logits` from row 1 on.
    pub fn draftAttention(m: *Model, enc0: mtl.ComputeEncoder, blk: *const weights.Layer, state: *kv.cache.State, slot: u32, layer: u32, rows: u32, position: u32, dbg: ?mtl.Buffer, prof: ?*Model.CondProf) !void {
        return model_draft_attention.draftAttention(m, enc0, blk, state, slot, layer, rows, position, dbg, prof);
    }

    /// The drafter's conditioning, split into its own command buffers so that each part's GPU window can be read on
    /// its own.
    pub const CondProf = struct {
        queue: mtl.Queue,
        cb: mtl.CommandBuffer,
        enc: mtl.ComputeEncoder,
        on: bool,
        /// The BLOCKS' split (TF_DSV41_BLOCK_PROF=1) -- the drafter's three blocks have never been split, and they
        /// are the larger half of the drafter's forward.
        blk_on: bool = false,
        /// The block the parts belong to (the MoE's own calls do not know it).
        block: u32 = 0,
        total_ms: f64 = 0,
        parts: u32 = 0,

        /// End the buffer, commit it, wait, print the part's GPU window, and start a fresh buffer. The encoder comes
        /// by POINTER, so the caller's own copy is updated too -- the blocks' parts are cut from inside the attention
        /// and the MoE, whose callers encode into the fresh encoder. The tag names the section ("cond" or "blk"), the
        /// label the part.
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

    /// TF_DSV41_MTP_DRAFTS: five rows a round (the pending token and four fillers) through three blocks of the
    /// drafter's own weights, then the shared head; the head writes from `.logits` row 2 on. THE FIRST DRAFT's
    /// ACCEPTANCE, smoothed -- the adaptive depth policy's signal. It is the fraction of rounds whose kept count is
    /// at least two (the pending token plus the first draft), and it is the SAME at every depth, while the yield is
    /// not: the yield scales with the depth, so one threshold cannot separate the novel text from the predictable
    /// text. The policy's warm-up length -- a running mean over the first `k` samples, then the shipped EMA. `TF_DSV41_MTP_ADAPT_WARM=0` is the pure EMA (the
    /// same-binary control).
    var mtp_warm_cache: ?u32 = null;
    pub fn mtpWarmRounds() u32 {
        if (mtp_warm_cache) |v| return v;
        const raw = std.c.getenv("TF_DSV41_MTP_ADAPT_WARM") orelse {
            mtp_warm_cache = 4;
            return 4;
        };
        const n = std.fmt.parseInt(u32, std.mem.span(raw), 10) catch 0;
        mtp_warm_cache = @min(n, 16);
        return mtp_warm_cache.?;
    }

    /// The lane's first-draft acceptance (the adaptive depth policy's signal), smoothed.
    pub fn noteAcceptSlot(m: *Model, slot: u32, hit: bool) void {
        if (slot >= m.mtp_accept_slot.len) return;
        const v: f64 = if (hit) 1.0 else 0.0;
        // THE FIRST SAMPLE SETS THE SIGNAL (a request's first round is a valid observation of the first draft's
        // acceptance). The pure-EMA form (the signal starts at zero) costs the PREDICTABLE text ten rounds of the
        // cheap window before it passes UP -- a tenth of the warm probe's rounds at the cheap rate.
        if (m.mtp_accept_seen_slot[slot] == 0) {
            m.mtp_accept_slot[slot] = v;
        } else if (m.mtp_accept_seen_slot[slot] < mtpWarmRounds()) {
            const n: f64 = @floatFromInt(m.mtp_accept_seen_slot[slot] + 1);
            m.mtp_accept_slot[slot] = (m.mtp_accept_slot[slot] * (n - 1.0) + v) / n;
        } else {
            m.mtp_accept_slot[slot] = m.mtp_accept_slot[slot] * 0.85 + v * 0.15;
        }
        m.mtp_accept_seen_slot[slot] += 1;
    }

    /// THE ROUND'S OWN PER-DRAFT ACCEPTANCE as the policy's signal (the same EMA and the same first-sample rule as
    /// `noteAcceptSlot`). At depth 1 it IS the first-draft hit (kept/1); at depth 3 it falls well below that on novel
    /// text -- where the first-draft sample stays high because a deep window's first draft is usually accepted while
    /// its second and third are not. That is the depth-awareness the policy needs to leave the deep window when the
    /// deep window stops paying.
    pub fn noteAcceptRatioSlot(m: *Model, slot: u32, ratio: f64) void {
        if (slot >= m.mtp_round_slot.len) return;
        const v = std.math.clamp(ratio, 0.0, 1.0);
        if (m.mtp_round_seen_slot[slot] == 0) {
            m.mtp_round_slot[slot] = v;
        } else {
            m.mtp_round_slot[slot] = m.mtp_round_slot[slot] * 0.85 + v * 0.15;
        }
        m.mtp_round_seen_slot[slot] += 1;
    }

    /// The round's per-draft acceptance (the hybrid policy's stay-deep signal).
    pub fn mtpRoundSlot(m: *Model, slot: u32) f64 {
        if (slot >= m.mtp_round_slot.len) return 0;
        return m.mtp_round_slot[slot];
    }

    /// The lane's first-draft acceptance (the adaptive depth policy's signal).
    pub fn mtpAcceptSlot(m: *Model, slot: u32) f64 {
        if (slot >= m.mtp_accept_slot.len) return 0;
        return m.mtp_accept_slot[slot];
    }

    /// The signal is per REQUEST -- a new request starts from a clean signal.
    pub fn resetAcceptSlot(m: *Model, slot: u32) void {
        if (slot >= m.mtp_accept_slot.len) return;
        m.mtp_accept_slot[slot] = 0;
        m.mtp_accept_seen_slot[slot] = 0;
        m.mtp_round_slot[slot] = 0;
        m.mtp_round_seen_slot[slot] = 0;
    }

    pub fn proposeTokens(m: *Model, slot: u32, pending: u32, count: u32) ![4]u32 {
        return model_propose.proposeTokens(m, slot, pending, count);
    }

    pub fn streamView(m: *const Model) View {
        return m.workspace.view(if (m.stream_alt) .stream_next else .stream);
    }
    pub fn streamNextView(m: *const Model) View {
        return m.workspace.view(if (m.stream_alt) .stream else .stream_next);
    }
    /// One fused boundary: the pending branch expanded into the other stream buffer, its mix partials, then the
    /// split: this boundary's pre/post/comb from the partials and the lagged `pre_in` collapse into `normed`.
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

/// TF_DSV41_STAGE_CB: close the command buffer at a stage boundary, log its GPU seconds under `label`, and start the
/// next one.
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
