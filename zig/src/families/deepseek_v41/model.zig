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
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const c = try config.Config.read(arena.allocator(), io, o.dir);
        const position_limit: u32 = @intCast(c.rope.max_positions);
        if (o.context > position_limit) return error.ContextFull;
        const capacity = if (o.context == 0) position_limit else o.context;
        const from_config = c.engram_table_dir.items();
        const resident_dir: ?[]const u8 = o.table_dir orelse (if (from_config.len == 0) null else from_config);
        if (o.stream_dir == null and resident_dir == null) return error.MissingEngramTables;
        const plan_dir: ?[]const u8 = if (o.stream_dir == null) (resident_dir orelse return error.MissingEngramTables) else null;
        var plan = try load.Plan.open(gpa, io, o.dir, plan_dir);
        defer plan.deinit();
        const streaming = o.stream_dir != null;
        try weights.validate(&plan, streaming);
        const map_blob = try std.Io.Dir.cwd().readFileAlloc(io, o.token_map, arena.allocator(), .limited(2 << 20));
        const map = try engram.hash.TokenMap.parse(map_blob);
        const lanes = o.lanes;
        // A draft round runs five rows through the same workspace and attention scratch; at decode the plan is one
        // row, so both are widened to the round's width (TF_DSV41_MTP_DRAFTS). A drafted round needs a plan wider
        // than one row: `exact_width` is this plan's rows, and the lane core will not build a window the plan cannot
        // hold. So both drafting knobs widen it, not just the MTP proposer.
        const drafted = std.c.getenv("TF_DSV41_MTP_DRAFTS") != null or std.c.getenv("TF_DSV41_LOOKUP_DRAFTS") != null;
        const round_rows: u32 = if (drafted) @max(o.chunk_rows, 8) else o.chunk_rows;
        const wp = try work.Plan.initLanes(round_rows, lanes);
        const kp = kv.cache.Plan{ .capacity = capacity, .layers = 40 };
        const workspace_bytes = try std.math.add(usize, wp.bytes, attention.Workspace.bytes(o.chunk_rows));
        const cache_bytes = try std.math.add(usize, kp.residentBytes(), weights.sizeofDescriptors());
        const execution_bytes = try std.math.add(usize, workspace_bytes, try std.math.mul(usize, cache_bytes, lanes));
        const rotary_bytes = attention.Pipelines.residentBytes(capacity, c);
        const needed = try std.math.add(usize, plan.bytes, try std.math.add(usize, execution_bytes, rotary_bytes));
        // The int8 twin lives beside the trellis it supersedes at prompt rows; it is additive until the trellis
        // attention weights can be dropped, so the admission must see it.
        const int8_extra: usize = if (std.c.getenv("TF_DSV41_I8_PREFILL") != null) 5 << 30 else 0;
        const needed_all = try std.math.add(usize, needed, int8_extra);
        if (o.ram_limit_bytes) |limit| if (needed_all > limit) {
            std.log.err("the resident payload needs {d} bytes; the load limit is {d}", .{ needed_all, limit });
            return error.LoadLimitExceeded;
        };
        const load_started = std.Io.Timestamp.now(io, .awake);
        const device = try mtl.Device.init();
        errdefer device.deinit();
        const margin: u64 = 8 << 30;
        const available = device.maxWorkingSet() -| device.allocated() -| margin;
        if (needed_all > available) {
            std.log.err("full residency needs {d} bytes plus {d} margin; Metal admission allows {d}", .{ needed_all, margin, available });
            return error.InsufficientResidentMemory;
        }
        const max_buffer = mtl.objc.msg(u64, device.id, "maxBufferLength", .{});
        for (plan.shards.items) |shard| if (shard.end - shard.begin > max_buffer) return error.ShardExceedsMetalBufferLimit;
        std.log.info("full resident admission: weights={d}, total={d}, context={d}, headroom={d}", .{ plan.bytes, needed, capacity, available - needed });
        var pipelines = try Pipelines.init(device, gpa, capacity, c);
        errdefer pipelines.deinit();
        const queue = try device.queue();
        errdefer queue.deinit();
        const residency = try device.residencySet(plan.shards.items.len + 128);
        errdefer residency.deinit();
        var cp = try plan.materialize(gpa, device, residency, queue);
        errdefer cp.deinit();
        var bound = try weights.bind(device, &cp, streaming);
        errdefer bound.deinit();
        // The prompt-row int8 sidecar. Flag-gated; the trellis matrices stay bound for decode and for any chunk under
        // the rows gate.
        if (std.c.getenv("TF_DSV41_I8_PREFILL") != null) {
            const sidecar = std.c.getenv("TF_DSV41_I8_DIR") orelse return error.MissingInt8Sidecar;
            try weights.attachInt8(gpa, io, device, &bound, std.mem.span(sidecar));
        }
        const caches = try gpa.alloc(kv.cache.State, lanes);
        errdefer gpa.free(caches);
        const engram_stash = try gpa.alloc(u8, lanes * 16);
        errdefer gpa.free(engram_stash);
        @memset(engram_stash, 0);
        const partial_stash = try gpa.alloc(u8, lanes * kv.cache.Plan.partialBytes(kp));
        errdefer gpa.free(partial_stash);
        @memset(partial_stash, 0);
        const ring_stash_bytes = @as(usize, lanes) * kp.layers * max_ring_rows * kv.cache.ring_stride;
        const ring_stash = try device.buffer(ring_stash_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer ring_stash.deinit();
        @memset(ring_stash.contents()[0..ring_stash_bytes], 0);
        for (caches) |*state| {
            state.* = try kv.cache.State.init(device, kp);
            errdefer state.deinit();
        }
        var workspace = try work.Workspace.init(device, wp, map.bytes);
        errdefer workspace.deinit();
        var aw = try attention.Workspace.init(device, round_rows);
        errdefer aw.deinit();
        const taps = try device.buffer(3 * 128 * 5120 * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer taps.deinit();
        const snap_ring = try device.buffer(@as(usize, lanes) * kp.layers * kv.cache.window * kv.cache.ring_stride, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer snap_ring.deinit();
        @memset(snap_ring.contents()[0..snap_ring.length()], 0);
        const taps_stash = try device.buffer(3 * 128 * 5120 * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer taps_stash.deinit();
        @memset(taps_stash.contents()[0..taps_stash.length()], 0);
        // The conditioning scratch: 128 rows of 5120 bf16, then 128 rows of 512 bf16 keys, then 128 packed rows. The
        // draft round's own logits scratch: a five-row forward keeps its head rows and Markov reads here, not in the
        // workspace's logits field (one row at decode, which a draft forward overran).
        const mtp_logits = try device.buffer(8 * 129280 * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        const cond_bytes = 128 * 5120 * 2 + 128 * 512 * 2 + 128 * 528;
        const cond = try device.buffer(cond_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        // The fused `wkv` dispatch's scratch. The input window is written three times (one per block), the output
        // comes back as three 128-row windows, and the ids are constant: routes name the block, the input and output
        // maps are the identity (the grouped tile's rows are the descriptors' own rows).
        const cond_wkv_bytes = 3 * 128 * 5120 * 2 + 3 * 128 * 512 * 2 + 3 * 3 * 128 * 4;
        const cond_wkv = try device.buffer(cond_wkv_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        @memset(cond_wkv.contents()[0..cond_wkv_bytes], 0);
        {
            const ids: [*]u32 = @ptrCast(@alignCast(cond_wkv.contents() + 3 * 128 * 5120 * 2 + 3 * 128 * 512 * 2));
            for (0..384) |r| {
                ids[r] = @intCast(r / 128);
                ids[384 + r] = @intCast(r);
                ids[768 + r] = @intCast(r);
            }
        }
        errdefer cond_wkv.deinit();
        errdefer taps.deinit();
        residency.add(taps);
        residency.add(cond_wkv);
        var stream_rows: ?*engram_stream.Rows = null;
        var stream_stage: ?mtl.Buffer = null;
        // Invariant: this block is the only place stream_rows and stream_stage are created, and weights.bind nulls
        // Engram.table/scales on the same condition. Keep the three in step: no live Model may have a null table with
        // a null stage, or the serve would gather from nothing.
        if (o.stream_dir) |dir| {
            const engram_layers = [_]u32{ engram.hash.layers[0].layer_id, engram.hash.layers[1].layer_id };
            const cap: u32 = if (std.c.getenv("TF_DSV41_ENGRAM_CACHE")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch o.stream_rows_cache else o.stream_rows_cache;
            stream_rows = try engram_stream.Rows.open(gpa, io, dir, 0, 1, &engram_layers, cap);
            errdefer stream_rows.?.close();
            const stage_bytes = try std.math.mul(usize, wp.rows, 2 * engram.gather_dim * @sizeOf(u16));
            stream_stage = try device.buffer(stage_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            std.log.info("engram stream: rows from {s}, {d} cached a layer, {d} staging bytes a layer", .{ dir, cap, stage_bytes / 2 });
        }
        bound.registerResidency(residency);
        for (caches) |*state| state.registerResidency(residency);
        residency.add(workspace.buffer);
        residency.add(workspace.shared_maps);
        aw.registerResidency(residency);
        if (stream_stage) |buf| residency.add(buf);
        residency.add(ring_stash);
        pipelines.attention.registerResidency(residency);
        // Layer 0's MoE input, 2,048 rows x 5,120 f32.
        var moe_x_debug: ?mtl.Buffer = null;
        if (std.c.getenv("TF_DSV41_DUMP_MOE_X") != null) {
            const b = try device.buffer(2048 * 5120 * 4, mtl.ResourceOptions.shared);
            errdefer b.deinit();
            moe_x_debug = b;
            residency.add(b);
        }
        // Layer 0's wq_b input, 2,048 rows x 1,280 f32, for the GPTQ Hessian.
        var wqb_x_debug: ?mtl.Buffer = null;
        if (std.c.getenv("TF_DSV41_DUMP_WQB_X") != null) {
            const b = try device.buffer(2048 * 1280 * 4, mtl.ResourceOptions.shared);
            errdefer b.deinit();
            wqb_x_debug = b;
            residency.add(b);
            attention_common.wqb_x_debug = b;
        }
        residency.commit();
        residency.requestResidency();
        queue.addResidencySet(residency);
        return .{ .gpa = gpa, .io = io, .device = device, .queue = queue, .residency = residency, .checkpoint = cp, .weights = bound, .cfg = c, .cfg_arena = arena, .lanes = lanes, .caches = caches, .cache = &caches[0], .engram_stash = engram_stash, .partial_stash = partial_stash, .ring_stash = ring_stash, .snap_ring = snap_ring, .taps_stash = taps_stash, .stream = stream_rows, .stream_stage = stream_stage, .workspace = workspace, .attention_workspace = aw, .taps = taps, .cond = cond, .cond_wkv = cond_wkv, .wqb_x_debug = wqb_x_debug, .moe_x_debug = moe_x_debug, .mtp_logits = mtp_logits, .pipelines = pipelines, .admitted_bytes = needed, .load_seconds = @as(f64, @floatFromInt(std.Io.Timestamp.now(io, .awake).toNanoseconds() - load_started.toNanoseconds())) / 1e9 };
    }

    pub fn deinit(m: *Model) void {
        m.queue.removeResidencySet(m.residency);
        m.residency.deinit();
        if (m.drafter_ids) |ids| ids.deinit();
        if (m.markov_part) |part| part.deinit();
        if (m.drafter_state) |state| {
            state.deinit();
            m.gpa.destroy(state);
        }
        if (m.stream) |rows| rows.close();
        if (m.stream_stage) |buf| buf.deinit();
        m.attention_workspace.deinit();
        m.workspace.deinit();
        for (m.caches) |*state| state.deinit();
        m.gpa.free(m.caches);
        m.gpa.free(m.engram_stash);
        m.gpa.free(m.partial_stash);
        m.ring_stash.deinit();
        m.snap_ring.deinit();
        m.taps_stash.deinit();
        if (m.gate_x) |b| b.deinit();
        if (m.dspark_logits) |b| b.deinit();
        if (m.cond_wkv) |b| b.deinit();
        if (m.wqb_x_debug) |b| {
            b.deinit();
            attention_common.wqb_x_debug = null;
        }
        if (m.moe_x_debug) |b| b.deinit();
        m.weights.deinit();
        m.checkpoint.deinit();
        m.pipelines.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.cfg_arena.deinit();
    }

    pub fn reset(m: *Model) !void {
        for (0..m.lanes) |slot| try m.resetSlot(@intCast(slot));
    }

    /// One lane slot back to empty: its cursor, its mutable cache bytes and its Engram history.
    pub fn resetSlot(m: *Model, slot: u32) !void {
        if (slot >= m.lanes) return error.InvalidLaneSlot;
        const state = &m.caches[slot];
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        m.valid = false;
        m.logits_rows = 0;
        state.position = 0;
        m.attention_workspace.selection_layer = null;
        m.attention_workspace.selection_rows = 0;
        m.attention_workspace.candidate_rows = 0;
        m.last_gpu_seconds = 0;
        state.saved_position = null;
        @memset(state.live.contents()[0..state.plan.mutableBytes()], 0);
        const cb = m.queue.commandBufferUnretained();
        const enc = cb.compute(.serial);
        engram.encodeReset(enc, &m.pipelines.engram, try m.workspace.historyView(slot));
        enc.end();
        cb.commit();
        cb.wait();
        try completed(cb);
        m.valid = true;
        m.logits_rows = 0;
    }

    /// A KEPT PROMPT STATE. One slot's whole cache -- the packed main/index tables and the mutable bytes (the window
    /// rings, the compressed path's partials and the raw-row captures) -- plus its cursor, the four-token Engram
    /// history the next chunk's hash pass reads, and the taps ring's rows and cursor (the drafter's conditioning).
    /// Every copied buffer is `shared` and every forward that wrote it committed and waited, so the copy is exactly
    /// the state a pass at `position` stands on: a restored state continues the prompt where the saved one stood, and
    /// `resumed` equals `fresh` token for token.
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

    /// START A MERGED TAIL CHUNK. `hist` ends at `at` (the last prompt tokens, newest last), the chunk starts at
    /// `from` and runs past `at`. The stash is primed with the live ring and taps -- a prefix shorter than the window
    /// keeps the older slots -- the mark's metadata is recorded, and `publish_limit` makes the chunk's per-layer
    /// publish and tap capture also write its PREFIX into the stash. Returns false when the mark's history cannot be
    /// built; the caller then keeps the cut.
    pub fn beginSnapTail(m: *Model, slot: u32, hist: []const u32, at: u32, from: u32) bool {
        if (hist.len < 3) return false;
        const state = &m.caches[slot];
        const q = at - from;
        const n: u32 = @min(q, kv.cache.window);
        // THE RECORD IS THE HASH PASS'S, NOT RAW TOKENS: `int tail[3]` of COMPRESSED ids, newest first, then `uint
        // available` (saturates at 3). A pass standing at `at` holds the three tokens before it through the same
        // token map the hash pass reads (The first form wrote raw prompt tokens here: the Engram history at the
        // resumed position then read 77/304/223 instead of 2460/174/259 and the turn diverged).
        const tm = m.workspace.view(.token_map);
        const map = engram.hash.TokenMap{ .bytes = (tm.buffer.contents() + tm.offset)[0 .. engram.hash.raw_vocab * 4] };
        var tail: [3]u32 = .{ 0, 0, 0 };
        for (0..3) |k| tail[k] = map.get(hist[hist.len - 1 - k]) catch return false;
        m.publish_limit = q;
        m.snap_pos = at;
        for (&m.snap_hist, 0..) |*v, k| v.* = if (k < 3) tail[k] else 3;
        m.snap_taps_base = if (condAlignOn()) from % kv.cache.window else m.taps_row;
        m.snap_taps_row = (m.taps_row + n) % kv.cache.window;
        m.snap_taps_n = @min(m.taps_n + n, kv.cache.window);
        m.snap_taps_last = n;
        const ring_bytes = @as(usize, state.plan.layers) * kv.cache.window * kv.cache.ring_stride;
        @memcpy(m.snap_ring.contents()[@as(usize, slot) * ring_bytes ..][0..ring_bytes], state.live.contents()[0..ring_bytes]);
        @memcpy(m.taps_stash.contents()[0..m.taps.length()], m.taps.contents()[0..m.taps.length()]);
        m.attention_workspace.publish_limit = q;
        m.attention_workspace.publish_stash = .{ .buffer = m.snap_ring, .offset = @as(usize, slot) * ring_bytes };
        return true;
    }

    /// A merged tail chunk re-points the taps ring BY POSITION; the cursor arm (TF_DSV41_COND_NOALIGN=1) is not
    /// position-addressed, so the merge stays off there.
    pub fn snapTailAllowed(m: *const Model) bool {
        _ = m;
        return condAlignOn();
    }

    pub fn endSnapTail(m: *Model) void {
        m.publish_limit = 0;
        m.attention_workspace.publish_limit = 0;
        m.attention_workspace.publish_stash = null;
    }

    /// The bytes one kept state holds: fixed, whatever its position (the slot's whole cache).
    pub fn snapshotBytes(m: *const Model) u64 {
        const plan = m.caches[0].plan;
        return @as(u64, plan.packedBytes()) + plan.mutableBytes() + m.taps.length();
    }

    pub fn snapshotSave(m: *Model, slot: u32) !*Snapshot {
        if (slot >= m.lanes) return error.InvalidLaneSlot;
        const state = &m.caches[slot];
        const snap = try m.gpa.create(Snapshot);
        errdefer m.gpa.destroy(snap);
        const storage = try m.gpa.dupe(u8, state.storage.contents()[0..state.plan.packedBytes()]);
        errdefer m.gpa.free(storage);
        const live = try m.gpa.dupe(u8, state.live.contents()[0..state.plan.mutableBytes()]);
        errdefer m.gpa.free(live);
        // A merged tail chunk's copy is THE STATE AT THE MARK, not at the chunk end: the taps ring and the window
        // ring come from the stash the chunk's prefix published, the four-token history is the caller's, and the
        // compressed partials at a tile mark are empty.
        const merging = m.publish_limit != 0;
        const taps_src = if (merging) m.taps_stash else m.taps;
        const taps = try m.gpa.dupe(u8, taps_src.contents()[0..m.taps.length()]);
        errdefer m.gpa.free(taps);
        const view = try m.workspace.historyView(slot);
        const head: [*]const u8 = @ptrCast(@alignCast(view.buffer.contents() + view.offset));
        var history: [4]u32 = undefined;
        for (&history, 0..) |*v, i| v.* = std.mem.readInt(u32, head[i * 4 ..][0..4], .little);
        snap.* = .{
            .storage = storage,
            .live = live,
            .taps = taps,
            .position = if (merging) m.snap_pos else state.position,
            .history = if (merging) m.snap_hist else history,
            .taps_row = if (merging) m.snap_taps_row else m.taps_row,
            .taps_n = if (merging) m.snap_taps_n else m.taps_n,
            .taps_rows_last = if (merging) m.snap_taps_last else m.taps_rows_last,
            .taps_rows_valid = if (merging) 0 else m.taps_rows_valid,
            .taps_base = if (merging) m.snap_taps_base else m.taps_base,
        };
        if (merging) {
            const ring_bytes = @as(usize, state.plan.layers) * kv.cache.window * kv.cache.ring_stride;
            const from = @as(usize, slot) * ring_bytes;
            @memcpy(live[0..ring_bytes], m.snap_ring.contents()[from..][0..ring_bytes]);
            const off = kv.cache.Plan.partialOffset(state.plan);
            if (off < live.len) @memset(live[off..], 0);
        }
        if (std.c.getenv("TF_DSV41_SNAP_HASH") != null) {
            const ring_bytes = @as(usize, state.plan.layers) * kv.cache.window * kv.cache.ring_stride;
            var h = std.hash.Wyhash.init(0);
            h.update(live[0..ring_bytes]);
            var hl = std.hash.Wyhash.init(1);
            hl.update(live[0 .. kv.cache.window * kv.cache.ring_stride]);
            var hs = std.hash.Wyhash.init(2);
            hs.update(storage);
            var ht = std.hash.Wyhash.init(3);
            ht.update(taps);
            std.log.info("snap hash: pos {d} merging {} ring {x} layer0 {x} storage {x} taps {x} hist {d} {d} {d} {d} tapsrow {d} tapsn {d} last {d}", .{ snap.position, merging, h.final(), hl.final(), hs.final(), ht.final(), snap.history[0], snap.history[1], snap.history[2], snap.history[3], snap.taps_row, snap.taps_n, snap.taps_rows_last });
        }
        return snap;
    }

    pub fn snapshotDrop(m: *Model, snap: *Snapshot) void {
        m.gpa.free(snap.storage);
        m.gpa.free(snap.live);
        m.gpa.free(snap.taps);
        m.gpa.destroy(snap);
    }

    /// Make one slot's cache the saved state's: the next chunk starts at its position. The caller resets the slot
    /// first, so nothing a previous request left where the state does not cover it survives.
    pub fn snapshotRestore(m: *Model, slot: u32, snap: *const Snapshot) !void {
        if (slot >= m.lanes) return error.InvalidLaneSlot;
        const state = &m.caches[slot];
        if (snap.storage.len != state.plan.packedBytes() or snap.live.len != state.plan.mutableBytes() or snap.taps.len != m.taps.length()) return error.SnapshotLayout;
        if (snap.position > state.plan.capacity) return error.SnapshotOutOfRange;
        @memcpy(state.storage.contents()[0..snap.storage.len], snap.storage);
        @memcpy(state.live.contents()[0..snap.live.len], snap.live);
        @memcpy(m.taps.contents()[0..snap.taps.len], snap.taps);
        const view = try m.workspace.historyView(slot);
        const head: [*]u8 = @ptrCast(@alignCast(view.buffer.contents() + view.offset));
        for (snap.history, 0..) |v, i| std.mem.writeInt(u32, head[i * 4 ..][0..4], v, .little);
        state.position = snap.position;
        state.saved_position = null;
        m.taps_row = snap.taps_row;
        m.taps_n = snap.taps_n;
        m.taps_rows_last = snap.taps_rows_last;
        m.taps_rows_valid = snap.taps_rows_valid;
        m.taps_base = snap.taps_base;
        m.attention_workspace.selection_layer = null;
        m.attention_workspace.selection_rows = 0;
        m.attention_workspace.candidate_rows = 0;
        m.valid = true;
    }

    /// A keep/rollback transaction over one slot's cache cursor and its 16-byte Engram history.
    pub fn beginSlot(m: *Model, slot: u32) !void {
        if (slot >= m.lanes) return error.InvalidLaneSlot;
        const view = try m.workspace.historyView(slot);
        const history: [*]u8 = @ptrCast(@alignCast(view.buffer.contents() + view.offset));
        @memcpy(m.engram_stash[slot * 16 ..][0..16], history[0..16]);
        try m.caches[slot].begin();
        // Stash the window ring's next slots before the verify writes them. The keep path puts back the rows the
        // round rejects; without it a wrapped ring's stale slots alias positions the next forward still reads.
        if (ringKeepOn()) {
            const state = &m.caches[slot];
            const stash_bytes = @as(usize, state.plan.layers) * max_ring_rows * kv.cache.ring_stride;
            const cb = m.queue.commandBufferUnretained();
            var enc = cb.compute(.serial);
            var enc_open = true;
            errdefer if (enc_open) enc.end();
            const ring = View{ .buffer = state.live, .offset = 0 };
            const stash = View{ .buffer = m.ring_stash, .offset = @as(usize, slot) * stash_bytes };
            try m.pipelines.attention.cache.encodeRingMove(enc, stash, ring, .{ .layers = state.plan.layers, .move_rows = max_ring_rows, .max_rows = max_ring_rows, .window = kv.cache.window, .position = state.position, .stride_words = kv.cache.ring_stride / 4, .first = 0, .to_stash = 1 });
            enc.end();
            enc_open = false;
            cb.commit();
            cb.wait();
            try completed(cb);
        }
        if (partialTxnOn()) {
            const state = &m.caches[slot];
            const len = kv.cache.Plan.partialBytes(state.plan);
            const base = kv.cache.Plan.partialOffset(state.plan);
            @memcpy(m.partial_stash[@as(usize, slot) * len ..][0..len], state.live.contents()[base .. base + len]);
        }
    }

    pub fn commitSlot(m: *Model, slot: u32) !void {
        if (slot >= m.lanes) return error.InvalidLaneSlot;
        try m.caches[slot].commit();
    }

    pub fn rollbackSlot(m: *Model, slot: u32) !void {
        if (slot >= m.lanes) return error.InvalidLaneSlot;
        const view = try m.workspace.historyView(slot);
        const history: [*]u8 = @ptrCast(@alignCast(view.buffer.contents() + view.offset));
        try m.caches[slot].rollback();
        if (partialTxnOn()) {
            const state = &m.caches[slot];
            const len = kv.cache.Plan.partialBytes(state.plan);
            const base = kv.cache.Plan.partialOffset(state.plan);
            @memcpy(state.live.contents()[base .. base + len], m.partial_stash[@as(usize, slot) * len ..][0..len]);
        }
        @memcpy(history[0..16], m.engram_stash[slot * 16 ..][0..16]);
    }

    /// The no-re-feed port. It restores the compressed path's open row from the capture (the group that straddles the
    /// new position), advances the Engram's cursor over the kept rows, and marks the taps' valid prefix (the rejected
    /// rows' taps are not the committed history).
    pub fn portKeep(m: *Model, slot: u32, kept: u32) !void {
        if (slot >= m.lanes) return error.InvalidLaneSlot;
        // The adaptive depth policy's signal. The verify's kept rows are the accepted drafts plus the pending token,
        // so `kept - 1` is the accepted drafts this round. The policy is a POLICY (which depth to run), not a
        // numerics change: it picks between two already-characterised windows (depth 1 is token-for-token identical
        // to the plain path; depth 3 is the standing one).
        const state = &m.caches[slot];
        const pos = state.position + kept;
        // The pool's groups are cut at even positions, so a new position that is odd leaves the group {pos-1, pos}
        // open: the open row (pos-1) must be the row the capture holds.
        if (kept > 0 and pos % 2 == 1) {
            const e = m.queue.commandBufferUnretained();
            var enc = e.compute(.serial);
            for ([_]u32{ 2, 8, 14 }) |owner| {
                const source = try state.source(owner, 2);
                const at = source.capture.offset + @as(usize, (pos - 1) % kv.cache.capture_rows) * 2 * kv.cache.capture_width * 4;
                const values = View{ .buffer = source.capture.buffer, .offset = at };
                const gates = View{ .buffer = source.capture.buffer, .offset = at + kv.cache.capture_width * 4 };
                try m.pipelines.attention.cache.restorePool(enc, values, gates, source.partial, .{ .rows = 1, .width = kv.cache.capture_width, .ratio = 2, .position = pos - 1, .stride = kv.cache.capture_width, .input_f32 = 1, .eps = m.cfg.eps });
            }
            enc.end();
            e.commit();
            e.wait();
            try completed(e);
        }
        try state.append(kept);
        if (m.stream != null and std.c.getenv("TF_DSV41_SKIP_ENGRAM") == null) try m.hashPass(slot, kept);
        // The taps: the conditioning publishes the capture's valid prefix (the kept rows).
        m.taps_rows_valid = kept;
    }

    /// Put back the window ring slots that the verify's rejected rows wrote. Called after `rollbackSlot` (so
    /// `position` is the window's start) and before the next forward. The stash was taken by `beginSlot`; rows
    /// `kept..rows-1` are the rejected ones.
    pub fn restoreRing(m: *Model, slot: u32, kept: u32, rows: u32) !void {
        if (!ringKeepOn()) return;
        if (slot >= m.lanes or rows > max_ring_rows or kept > rows) return;
        const state = &m.caches[slot];
        const move = rows - kept;
        if (move == 0) return;
        const stash_bytes = @as(usize, state.plan.layers) * max_ring_rows * kv.cache.ring_stride;
        const cb = m.queue.commandBufferUnretained();
        var enc = cb.compute(.serial);
        var enc_open = true;
        errdefer if (enc_open) enc.end();
        const ring = View{ .buffer = state.live, .offset = 0 };
        const stash = View{ .buffer = m.ring_stash, .offset = @as(usize, slot) * stash_bytes };
        try m.pipelines.attention.cache.encodeRingMove(enc, ring, stash, .{ .layers = state.plan.layers, .move_rows = move, .max_rows = max_ring_rows, .window = kv.cache.window, .position = state.position, .stride_words = kv.cache.ring_stride / 4, .first = kept, .to_stash = 0 });
        enc.end();
        enc_open = false;
        cb.commit();
        cb.wait();
        try completed(cb);
    }

    /// One command buffer per chunk; no operator waits or reads routes back to the CPU.
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
        const dspark = m.weights.dspark orelse return error.NoDrafter;
        // Bisect arms: TF_DSV41_MTP_STUB returns before the draft forward, TF_DSV41_MTP_STUB_COND after the
        // conditioning. Both leave the engine's hand-off and verify in place, so they separate the machinery from the
        // draft round's own execution.
        if (std.c.getenv("TF_DSV41_MTP_STUB") != null) return .{ 0, 0, 0, 0 };
        // The pending row is this round's first verified row; start its page-in now, under the drafter's own GPU
        // window, instead of at the verify's stageRows.
        if (engramPendingOn()) prefetchPending(m, slot, pending);
        // The draft round runs its own boundaries through the shared coefficient views and the stream parity: a main
        // forward re-initialises them, but the *parity* chooses which stream the next boundary reads, so the model
        // fields are saved and put back around the round.
        const alt0 = m.stream_alt;
        const pend0 = m.pending_expand;
        defer {
            m.stream_alt = alt0;
            m.pending_expand = pend0;
        }
        const ws = &m.workspace;
        const p = &m.pipelines;
        const state = try m.drafterState();
        // The conditioning's cursor is THIS slot's committed position, not lane 0's.
        const slot_pos = m.caches[slot].position;
        state.position = slot_pos;
        // THE DRAFTER RUNS `depth + 1` ROWS, NOT FIVE. The chain's step s reads the logits row s and the bias of the
        // token before it, and writes the pick into IDS[s+1]; the hand-off takes picks 0..count-1 (the lane's depth,
        // at most three). The five-row form computed two rows whose picks (the fourth and the fifth) the hand-off
        // never read, and every extra row the drafter runs is published into its own 128-row attention window at its
        // own position -- so the extra rows are not free: they displace the real history the drafter's rows attend
        // over. The rows are the ones the chain reads plus the one the chain's own step needs, so `depth + 1` rows is
        // the form. TF_DSV41_MTP_ROWS restores the five-row form for a same-binary A/B.
        const n: u32 = if (std.c.getenv("TF_DSV41_MTP_ROWS")) |v|
            (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 5)
        else
            @min(@max(count, 1) + 1, 5);
        var out: [4]u32 = @splat(0);
        var cb = m.queue.commandBuffer();
        var enc = cb.compute(if (concurrentOn()) .concurrent else .serial);
        // The conditioning's per-part GPU split, and the host's own encode time against the GPU's window. Neither
        // flag changes a byte of the shipped path.
        var cond_prof = CondProf{ .queue = m.queue, .cb = cb, .enc = enc, .on = std.c.getenv("TF_DSV41_COND_PROF") != null, .blk_on = std.c.getenv("TF_DSV41_BLOCK_PROF") != null };
        const host_prof = std.c.getenv("TF_DSV41_HOST_PROF") != null;
        const t_start: f64 = if (host_prof) mtl.clock.seconds() else 0;
        // The conditioning: the three target layers' stream means (the taps, one 15,360-wide row) through main_proj
        // and main_norm, then each stage's keys into its ring -- what the draft rows attend to. The release's
        // `Transformer.forward_spec` (inference/model.py, revision 314113f8) is the same pass: `forward_embed` makes
        // the noise-token draft rows and `DSparkBlock.forward` threads `pre_mix` and `main_x` through the stage. The
        // conditioning: the last `nc` rows of the rolling tap ring (the decode window's own rows, captured a token a
        // forward) through main_proj and main_norm, then each stage's keys into its ring. The ring once
        // held only the current pass's row (`taps_n = rows = 1` at decode), so 127 of the 128 window slots the draft
        // rows read were uninitialised packed cache and the filler rows went NaN.
        const pos_now = slot_pos;
        // The captured rows are the last `taps_rows_last` rows of the last forward, at their own positions [pos_now -
        // cap_n, pos_now). The release writes each main row at its own position and ropes the draft rows at the
        // positions after the sequence; the earlier form published the valid prefix at the suffix's positions,
        // shifting the context by the rows the round kept.
        const cap_n: u32 = @min(m.taps_rows_last, 128);
        const cap_pos: u32 = pos_now -| cap_n;
        const true_pos = dsparkTruePosOn();
        // The capture's valid rows: the rollback's prefix (the rejected rows' taps are not the committed history), or
        // the whole capture when no rollback ran (the whole window landed).
        const cap_rows: u32 = @min(@min(m.taps_rows_last, m.taps_n), 128);
        // The conditioning publishes the capture's whole rows at their own positions, so the ring holds the main
        // model's rows everywhere; the earlier form published only the kept prefix, leaving a hole at every rejected
        // position. TF_DSV41_DSPARK_CAPTURE_ALL=0 restores the prefix form.
        const dspark_capture_all: u32 = if (std.c.getenv("TF_DSV41_DSPARK_CAPTURE_ALL")) |v|
            (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 1)
        else
            1;
        const tail: u32 = if (std.c.getenv("TF_DSV41_DSPARK_COND_ONE") != null)
            1
        else if (m.taps_rows_valid != 0)
            @min(m.taps_rows_valid, cap_rows)
        else
            cap_rows;
        // The published rows' positions: the capture's own (the release's form) or the earlier shifted form.
        const pub_pos: u32 = if (true_pos) cap_pos else pos_now - tail;
        // The drafter's own rows' rope position: the next token's position (the release ropes every draft row at
        // start_pos + seqlen + i) or the earlier slot position.
        const draft_pos: u32 = if (true_pos) cap_pos + tail else slot_pos;
        // The windowed context's row count. With the aligned ring the rows are the positions' own -- the ring's slots
        // are the positions. The count is rounded down to a multiple of four so the dense multi-row body takes it; 1
        // keeps the one-row body.
        var nc: u32 = 1;
        if (condRowsWanted() != 0) {
            const have: u32 = if (condAlignOn()) @min(pos_now, 128) else @min(m.taps_n, pos_now);
            const want: u32 = @min(condRowsWanted(), have);
            nc = if (want >= 4) (want / 4) * 4 else want;
        }
        var round = DraftRound{
            .m = m,
            .dspark = dspark,
            .slot = slot,
            .pending = pending,
            .count = count,
            .n = n,
            .state = state,
            .pos_now = pos_now,
            .cap_n = cap_n,
            .cap_pos = cap_pos,
            .true_pos = true_pos,
            .cap_rows = cap_rows,
            .dspark_capture_all = dspark_capture_all,
            .tail = tail,
            .pub_pos = pub_pos,
            .draft_pos = draft_pos,
            .nc = nc,
            .enc = enc,
            .cb = cb,
            .cond_prof = &cond_prof,
        };
        try round.conditionStage();
        enc = round.enc;
        cb = round.cb;
        // THE DRAFTER'S STOP POINT 1 (after the conditioning). The arm commits what was encoded, so the command
        // buffer's completion fires.
        if (mtpStopPoint() == 1 or std.c.getenv("TF_DSV41_MTP_STUB_COND") != null) {
            enc.end();
            cb.commit();
            cb.wait();
            std.log.info("mtp stop 1: rows {d} gpu {d:.2} ms", .{ n, cb.gpuSeconds() * 1e3 });
            try completed(cb);
            return .{ 0, 0, 0, 0 };
        }
        // THE DRAFTER'S OWN EARLY COMMIT. Same mechanism and same guarantees as: a command-buffer boundary is a full
        // barrier, the queue is in-order, and the final wait implies the earlier buffer. `TF_DSV41_MTP_PIPE=0` is the
        // same-binary control; an instrument that owns the buffer (`COND_PROF`, `BLOCK_PROF`) keeps the shipped
        // single-buffer form.
        var pipe_prev: mtl.CommandBuffer = undefined;
        var pipe_prev_set = false;
        const mtp_pipe: bool = blk: {
            if (cond_prof.on or cond_prof.blk_on) break :blk false;
            const raw = std.c.getenv("TF_DSV41_MTP_PIPE") orelse break :blk true;
            break :blk !std.mem.eql(u8, std.mem.span(raw), "0");
        };
        if (mtp_pipe) {
            enc.end();
            cb.commit();
            pipe_prev = cb;
            pipe_prev_set = true;
            cb = m.queue.commandBuffer();
            enc = cb.compute(if (concurrentOn()) .concurrent else .serial);
            cond_prof.cb = cb;
            cond_prof.enc = enc;
        }
        const ids = try m.drafterRound(pending, n);
        ops.encodeEmbedding(enc, &p.ops, ids, tensor(m.weights.embed), ws.view(.branch), .{ .rows = n, .width = 5120, .table_stride = 5120, .y_stride = 5120, .copies = 1 });
        enc.barrier();
        if (cond_prof.blk_on) cond_prof.split("blk", "prologue/embed", 0, &enc);
        const dbg: ?mtl.Buffer = if (std.c.getenv("TF_DSV41_MTP_DEBUG_ROWS") != null) try m.device.buffer(mtp_dbg_total3 * 4, mtl.ResourceOptions.shared) else null;
        defer if (dbg) |b| b.deinit();
        if (dbg) |b| ops.encodeDump(enc, &p.ops, ws.view(.branch), .{ .buffer = b, .offset = 0 }, 5 * 5120, true);
        // The mHC's expanded stream and its `pre` vector carry across the drafter's blocks, as the trunk's do (the
        // expand is once, before the loop, and the release's DSparkBlock threads `pre_mix` through each stage).
        // Expanding inside the loop reset the stream to the branch and the pre to one-hot at every block.
        mhc.encodeExpand(enc, &p.mhc, ws.view(.branch), ws.view(.stream), ws.view(.previous_pre), .{ .rows = n, .width = 5120, .x_stride = 5120, .stream_stride = 20480, .pre_stride = 4 });
        if (cond_prof.blk_on) cond_prof.split("blk", "prologue/expand", 0, &enc);
        round.enc = enc;
        round.cb = cb;
        try round.blockStage(dbg);
        enc = round.enc;
        cb = round.cb;
        // THE DRAFTER'S STOP POINT 2 (after the three blocks). The arm commits what was encoded, so the command
        // buffer's completion fires.
        if (mtpStopPoint() == 2 or std.c.getenv("TF_DSV41_MTP_STUB_BLOCKS") != null) {
            enc.end();
            cb.commit();
            cb.wait();
            std.log.info("mtp stop 2: rows {d} gpu {d:.2} ms", .{ n, cb.gpuSeconds() * 1e3 });
            try completed(cb);
            return .{ 0, 0, 0, 0 };
        }
        const logits = try round.headStage();
        enc = round.enc;
        cb = round.cb;
        // THE DRAFTER'S STOP POINT 3 (after the head projection). The arm commits what was encoded, so the command
        // buffer's completion fires.
        if (mtpStopPoint() == 3 or std.c.getenv("TF_DSV41_MTP_STUB_HEAD") != null) {
            enc.end();
            cb.commit();
            cb.wait();
            std.log.info("mtp stop 3: rows {d} gpu {d:.2} ms", .{ n, cb.gpuSeconds() * 1e3 });
            try completed(cb);
            return .{ 0, 0, 0, 0 };
        }
        const markov_run = try round.markovStage(logits);
        enc = round.enc;
        cb = round.cb;
        const price_dbg = markov_run.price_dbg;
        defer if (price_dbg) |b| b.deinit();
        const groups = markov_run.groups;
        // The drafter's logits row 0 against the trunk's own row-0 logits (the verify's first row -- the same
        // prediction the drafter's row 0 makes). The drafter's forward runs after the verify, so the workspace's
        // `.logits` row 0 still holds the trunk's row-0 logits (the drafter's own wo_a writes from row 1 on).
        const top_buf: ?mtl.Buffer = if (std.c.getenv("TF_DSV41_MTP_DEBUG_TOP") != null)
            try m.device.buffer(2 * 129280 * 4 + 5120 * 4, mtl.ResourceOptions.shared)
        else
            null;
        defer if (top_buf) |b| b.deinit();
        if (top_buf) |b| {
            ops.encodeDump(enc, &p.ops, logits, .{ .buffer = b, .offset = 0 }, 129280, false);
            ops.encodeDump(enc, &p.ops, ws.view(.logits), .{ .buffer = b, .offset = 129280 * 4 }, 129280, false);
            ops.encodeDump(enc, &p.ops, ws.view(.normalized), .{ .buffer = b, .offset = 2 * 129280 * 4 }, 5120, false);
            enc.barrier();
        }
        // The host's own encode time, the wait, and the GPU's window -- so the drafter's cost can be split into "the
        // GPU's work" and "the host's encoding" (and, under the conditioning's split, the gap between them).
        const t_encoded: f64 = if (host_prof) mtl.clock.seconds() else 0;
        enc.end();
        cb.commit();
        cb.wait();
        const t_done: f64 = if (host_prof) mtl.clock.seconds() else 0;
        try completed(cb);
        if (price_dbg) |b| {
            const f: [*]const f32 = @ptrCast(@alignCast(b.contents()));
            const g2: usize = @as(usize, groups) * 2;
            var ps: usize = 0;
            while (ps < n) : (ps += 1) {
                var v1: f32 = -std.math.inf(f32);
                var id1: u32 = 0;
                var v2: f32 = -std.math.inf(f32);
                var id2: u32 = 0;
                for (0..groups) |g| {
                    const v = f[ps * g2 + g * 2];
                    const id: u32 = @bitCast(f[ps * g2 + g * 2 + 1]);
                    if (v > v1 or (v == v1 and id < id1)) {
                        v2 = v1;
                        id2 = id1;
                        v1 = v;
                        id1 = id;
                    } else if (v > v2 or (v == v2 and id < id2)) {
                        v2 = v;
                        id2 = id;
                    }
                }
                std.log.info("treeprice propose round {d} step {d} top2 {d} {d}", .{ price_round, ps, id1, id2 });
            }
        }
        // The early buffer completed in queue order; check it and count its GPU time so the `mtp forward` line keeps
        // meaning the whole drafter forward.
        var gpu_extra_ms: f64 = 0;
        if (pipe_prev_set) {
            try completed(pipe_prev);
            gpu_extra_ms = pipe_prev.gpuSeconds() * 1e3;
        }
        // The drafter's own forward's GPU time and wall. The stage timer covers the trunk's layers, not this forward
        // -- and the conditioning's cost (the windowed context's projection, the keys, the rope and the publish)
        // lives here.
        if (host_prof) {
            const encode_ms = (t_encoded - t_start) * 1e3;
            const wait_ms = (t_done - t_encoded) * 1e3;
            if (cond_prof.on or cond_prof.blk_on) {
                cond_prof.total_ms += cb.gpuSeconds() * 1e3;
                std.log.info("mtp forward: rows {d} gpu {d:.3} ms (split sum, {d} parts) encode {d:.2} ms wait {d:.2} ms", .{ n, cond_prof.total_ms, cond_prof.parts + 1, encode_ms, wait_ms });
            } else {
                std.log.info("mtp forward: rows {d} gpu {d:.2} ms encode {d:.2} ms wait {d:.2} ms", .{ n, cb.gpuSeconds() * 1e3 + gpu_extra_ms, encode_ms, wait_ms });
            }
        }
        if (!host_prof and cond_prof.blk_on) {
            cond_prof.total_ms += cb.gpuSeconds() * 1e3;
            std.log.info("mtp forward: rows {d} gpu {d:.3} ms (split sum, {d} parts)", .{ n, cond_prof.total_ms, cond_prof.parts + 1 });
        }
        if (top_buf) |b| {
            const f: [*]const f32 = @ptrCast(@alignCast(b.contents()));
            var dout: [5]u32 = undefined;
            var dval: [5]f32 = undefined;
            var tout: [5]u32 = undefined;
            var tval: [5]f32 = undefined;
            logitsTopK(f, 129280, &dout, &dval);
            logitsTopK(f + 129280, 129280, &tout, &tval);
            std.log.info("mtp top: drafter argmax={d} vals={d:.4} {d:.4} {d:.4} ids={d} {d} {d} {d} {d}", .{ dout[0], dval[0], dval[1], dval[2], dout[0], dout[1], dout[2], dout[3], dout[4] });
            std.log.info("mtp top: trunk   argmax={d} vals={d:.4} {d:.4} {d:.4} ids={d} {d} {d} {d} {d}", .{ tout[0], tval[0], tval[1], tval[2], tout[0], tout[1], tout[2], tout[3], tout[4] });
            std.log.info("mtp top: agree={} drafter@trunk_argmax={d:.4} trunk@drafter_argmax={d:.4}", .{ dout[0] == tout[0], f[@intCast(tout[0])], f[129280 + @as(usize, @intCast(dout[0]))] });
            // The drafter's head input against the trunk's own head input (the verify's row 0): the cosine says
            // whether the MTP module's hidden lives in the trunk's hidden space at all.
            if (m.mtp_trunk_hidden) |tb| {
                const t: [*]const f32 = @ptrCast(@alignCast(tb.contents()));
                const dh = f + 2 * 129280;
                var dot: f64 = 0;
                var dn: f64 = 0;
                var tn: f64 = 0;
                var c: usize = 0;
                while (c < 5120) : (c += 1) {
                    dot += @as(f64, dh[c]) * @as(f64, t[c]);
                    dn += @as(f64, dh[c]) * @as(f64, dh[c]);
                    tn += @as(f64, t[c]) * @as(f64, t[c]);
                }
                std.log.info("mtp hidden: drafter rms={d:.5} trunk rms={d:.5} cosine={d:.5}", .{ @sqrt(dn / 5120.0), @sqrt(tn / 5120.0), dot / @sqrt(dn * tn) });
            }
        }
        if (std.c.getenv("TF_DSV41_MTP_DEBUG_LOGITS") != null) {
            {
                const ev = m.weights.embed;
                std.log.info("mtp embed buffer {d} bytes offset {d}", .{ ev.buffer.length(), ev.offset });
                const tp: [*]const u16 = @ptrCast(@alignCast(ev.buffer.contents() + ev.offset));
                var kk: usize = 0;
                while (kk < 3) : (kk += 1) {
                    const base = (128799 + kk) * 5120;
                    std.log.info("mtp embed row {d}: {d:.5} {d:.5} {d:.5}", .{ 128799 + kk, asF32(tp[base]), asF32(tp[base + 1]), asF32(tp[base + 2]) });
                }
            }
            if (m.drafter_ids) |idsbuf| {
                const ip: [*]const u32 = @ptrCast(@alignCast(idsbuf.contents()));
                std.log.info("mtp ids: {d} {d} {d} {d} {d} (buffer {d} bytes)", .{ ip[0], ip[1], ip[2], ip[3], ip[4], idsbuf.length() });
            }
            if (dbg) |b| {
                const f: [*]const f32 = @ptrCast(@alignCast(b.contents()));
                inline for (.{ "embed", "attn_in", "attn_out", "moe_in", "moe_out" }, 0..) |name, seg| {
                    var rr: usize = 0;
                    while (rr < 5) : (rr += 1) {
                        var bad: usize = 0;
                        var cc: usize = 0;
                        var acc: f64 = 0;
                        while (cc < 5120) : (cc += 1) {
                            if (!std.math.isFinite(f[(seg * 5 + rr) * 5120 + cc])) bad += 1;
                            acc += @as(f64, f[(seg * 5 + rr) * 5120 + cc]) * @as(f64, f[(seg * 5 + rr) * 5120 + cc]);
                        }
                        // The row's RMS: the attention's contribution (attn_out - embed) and the MoE's are read off
                        // the row magnitudes -- a drafter whose attention contributes nothing reads a flat logits row
                        // (the round's instrument).
                        std.log.info("mtp {s} row {d}: nonfinite={d} rms={d:.5} first={d:.5} {d:.5}", .{ name, rr, bad, @sqrt(acc / 5120.0), f[(seg * 5 + rr) * 5120], f[(seg * 5 + rr) * 5120 + 1] });
                    }
                }
                // The drafter's key rows (the pack's input), its query's two stages, the decoded key table the scan
                // reads, and the attention's output: a row's first non-finite column localises it.
                {
                    var rrk: usize = 0;
                    while (rrk < 5) : (rrk += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 512) : (cc += 1) {
                            const v = f[mtp_dbg_raw + rrk * 512 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp key row {d}: nonfinite={d} maxabs={d:.5}", .{ rrk, bad, mx });
                    }
                    var rrq: usize = 0;
                    while (rrq < 5) : (rrq += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 1280) : (cc += 1) {
                            const v = f[mtp_dbg_qr + rrq * 1280 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp qr row {d}: nonfinite={d} maxabs={d:.5}", .{ rrq, bad, mx });
                    }
                    var rrp: usize = 0;
                    while (rrp < 5) : (rrp += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 32768) : (cc += 1) {
                            const v = f[mtp_dbg_qpre + rrp * 32768 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp qpre row {d}: nonfinite={d} maxabs={d:.5}", .{ rrp, bad, mx });
                    }
                    // The drafter's query's per-head RMS -- the instrument's read of the scale the softmax saw (flat
                    // when every head is tiny, saturated when one is huge).
                    {
                        const r = headRms(f + mtp_dbg_qpre, 64, 512);
                        std.log.info("mtp qpre head rms: min={d:.5} max={d:.5} mean={d:.5}", .{ r.min, r.max, r.mean });
                    }
                    var rrd: usize = 0;
                    while (rrd < 133) : (rrd += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 512) : (cc += 1) {
                            const v = f[mtp_dbg_dense + rrd * 512 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        if (bad != 0 or rrd >= 126) std.log.info("mtp dense row {d}: nonfinite={d} maxabs={d:.5}", .{ rrd, bad, mx });
                    }
                    // THE RING'S DECODED KEYS: the RMS of a ring row and the cosine between two ADJACENT positions
                    // (the slots 127 and 0 are one position apart across the ring's wrap). If the ring's keys are one
                    // vector, the drafter's softmax cannot discriminate the history.
                    {
                        const ring0 = mtp_dbg_dense;
                        const ring1 = mtp_dbg_dense + 127 * 512;
                        const ring2 = mtp_dbg_dense + 126 * 512;
                        var a0: f64 = 0;
                        var a1: f64 = 0;
                        var a2: f64 = 0;
                        var d01: f64 = 0;
                        var c01: f64 = 0;
                        var c02: f64 = 0;
                        var cc: usize = 0;
                        while (cc < 512) : (cc += 1) {
                            const v0 = f[ring0 + cc];
                            const v1 = f[ring1 + cc];
                            const v2 = f[ring2 + cc];
                            a0 += @as(f64, v0) * @as(f64, v0);
                            a1 += @as(f64, v1) * @as(f64, v1);
                            a2 += @as(f64, v2) * @as(f64, v2);
                            d01 += (@as(f64, v1) - @as(f64, v0)) * (@as(f64, v1) - @as(f64, v0));
                            c01 += @as(f64, v0) * @as(f64, v1);
                            c02 += @as(f64, v0) * @as(f64, v2);
                        }
                        std.log.info("mtp ring keys: rms(slot0)={d:.5} rms(slot127)={d:.5} rms(slot126)={d:.5} cos(0,127)={d:.5} cos(0,126)={d:.5} rmsdiff(0,127)={d:.5}", .{ @sqrt(a0 / 512.0), @sqrt(a1 / 512.0), @sqrt(a2 / 512.0), c01 / @sqrt(a0 * a1), c02 / @sqrt(a0 * a2), @sqrt(d01 / 512.0) });
                    }
                    var rrz: usize = 0;
                    while (rrz < 5) : (rrz += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 32768) : (cc += 1) {
                            const v = f[mtp_dbg_qpost + rrz * 32768 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp qpost row {d}: nonfinite={d} maxabs={d:.5}", .{ rrz, bad, mx });
                    }
                    var rrc: usize = 0;
                    while (rrc < 5) : (rrc += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 5120) : (cc += 1) {
                            const v = f[mtp_dbg_coll + rrc * 5120 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp collapsed row {d}: nonfinite={d} maxabs={d:.5}", .{ rrc, bad, mx });
                    }
                    var rre: usize = 0;
                    while (rre < 5) : (rre += 1) {
                        var bad: usize = 0;
                        var cc: usize = 0;
                        while (cc < 4) : (cc += 1) {
                            if (!std.math.isFinite(f[mtp_dbg_pre + rre * 4 + cc])) bad += 1;
                        }
                        std.log.info("mtp pre row {d}: nonfinite={d} vals={d:.5} {d:.5} {d:.5} {d:.5}", .{ rre, bad, f[mtp_dbg_pre + rre * 4], f[mtp_dbg_pre + rre * 4 + 1], f[mtp_dbg_pre + rre * 4 + 2], f[mtp_dbg_pre + rre * 4 + 3] });
                    }
                    var rrn: usize = 0;
                    while (rrn < 5) : (rrn += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 5120) : (cc += 1) {
                            const v = f[(1 * 5 + rrn) * 5120 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp norm row {d}: nonfinite={d} maxabs={d:.5}", .{ rrn, bad, mx });
                    }
                    var rri: usize = 0;
                    while (rri < 5) : (rri += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 5120) : (cc += 1) {
                            const v = f[mtp_dbg_pin + rri * 5120 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp pin row {d}: nonfinite={d} maxabs={d:.5}", .{ rri, bad, mx });
                    }
                    var rro: usize = 0;
                    while (rro < 5) : (rro += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 5120) : (cc += 1) {
                            const v = f[mtp_dbg_prot + rro * 5120 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp prot row {d}: nonfinite={d} maxabs={d:.5}", .{ rro, bad, mx });
                    }
                    var rrs: usize = 0;
                    while (rrs < 5) : (rrs += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 512) : (cc += 1) {
                            const v = f[mtp_dbg_psum + rrs * 512 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp psum row {d}: nonfinite={d} maxabs={d:.5}", .{ rrs, bad, mx });
                    }
                    var rrt: usize = 0;
                    while (rrt < 5) : (rrt += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 5120) : (cc += 1) {
                            const v = f[mtp_dbg_rot + rrt * 5120 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp rot row {d}: nonfinite={d} maxabs={d:.5}", .{ rrt, bad, mx });
                    }
                    var rrg: usize = 0;
                    while (rrg < 5) : (rrg += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 8192) : (cc += 1) {
                            const v = f[mtp_dbg_grouped + rrg * 8192 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp grouped row {d}: nonfinite={d} maxabs={d:.5}", .{ rrg, bad, mx });
                    }
                    var rrw: usize = 0;
                    while (rrw < 5) : (rrw += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 1024) : (cc += 1) {
                            const v = f[mtp_dbg_woasums + rrw * 1024 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp woasums row {d}: nonfinite={d} maxabs={d:.5}", .{ rrw, bad, mx });
                    }
                    var rru: usize = 0;
                    while (rru < 5) : (rru += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 512) : (cc += 1) {
                            const v = f[mtp_dbg_sums + rru * 512 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp sums row {d}: nonfinite={d} maxabs={d:.5}", .{ rru, bad, mx });
                    }
                }
            }
            inline for (.{ .{ "branch", work.Field.branch }, .{ "normalized", work.Field.normalized }, .{ "collapsed", work.Field.collapsed } }) |spec| {
                const v = ws.view(spec[1]);
                const b: [*]const u16 = @ptrCast(@alignCast(v.buffer.contents() + v.offset));
                var rrz: usize = 0;
                while (rrz < 5) : (rrz += 1) {
                    const w = b[rrz * 5120 ..];
                    const a0 = asF32(w[0]);
                    const a1 = asF32(w[1]);
                    const a2 = asF32(w[2]);
                    var bad: usize = 0;
                    var cc: usize = 0;
                    while (cc < 5120) : (cc += 1) {
                        if (!std.math.isFinite(asF32(w[cc]))) bad += 1;
                    }
                    std.log.info("mtp {s} row {d}: nonfinite={d} first={d:.5} {d:.5} {d:.5}", .{ spec[0], rrz, bad, a0, a1, a2 });
                }
            }
            const f: [*]const f32 = @ptrCast(@alignCast(m.mtp_logits.?.contents()));
            var row: usize = 0;
            while (row < 5) : (row += 1) {
                var bad: usize = 0;
                var best: f32 = -std.math.inf(f32);
                var best_col: usize = 0;
                var col: usize = 0;
                while (col < 129280) : (col += 1) {
                    const v = f[row * 129280 + col];
                    if (!std.math.isFinite(v)) bad += 1 else if (v > best) {
                        best = v;
                        best_col = col;
                    }
                }
                std.log.info("mtp logits row {d}: nonfinite={d} max={d:.5} at {d}", .{ row, bad, best, best_col });
            }
        }
        const idbytes = m.markov_ids.?.contents();
        if (std.c.getenv("TF_DSV41_DRAFT_DEBUG") != null) {
            const picks: [*]const u32 = @ptrCast(@alignCast(idbytes));
            std.log.info("mtp picks: {d} {d} {d} {d} {d} {d}", .{ picks[0], picks[1], picks[2], picks[3], picks[4], picks[5] });
        }
        const rows_out: u32 = @min(count, n);
        var r: u32 = 0;
        while (r < rows_out) : (r += 1) out[r] = std.mem.readInt(u32, idbytes[(r + 1) * 4 ..][0..4], .little);
        return out;
    }

    /// The release's `Transformer.forward_spec` as this lane runs it: `forward_embed`'s conditioning,
    /// the three `DSparkBlock` stages, `forward_head`'s projection and Markov chain. Each stage keeps the
    /// lane's arms and its own dispatch shapes; the sequence is the release's.
    const MarkovRun = struct {
        price_dbg: ?mtl.Buffer,
        groups: u32,
    };

    const DraftRound = struct {
        m: *Model,
        dspark: weights.Dspark,
        slot: u32,
        pending: u32,
        count: u32,
        n: u32,
        state: *kv.cache.State,
        pos_now: u32,
        cap_n: u32,
        cap_pos: u32,
        true_pos: bool,
        cap_rows: u32,
        dspark_capture_all: u32,
        tail: u32,
        pub_pos: u32,
        draft_pos: u32,
        nc: u32,
        enc: mtl.ComputeEncoder,
        cb: mtl.CommandBuffer,
        cond_prof: *CondProf,

        /// `Transformer.forward_spec`'s conditioning half (`DSparkBlock.forward_embed`): the target layers'
        /// stream means through `main_proj` and `main_norm`, then each draft stage's keys at their own
        /// positions into its ring. The release writes one main row per step; this lane publishes the
        /// capture's whole window so the draft rows read history instead of the uninitialised ring.
        pub fn conditionStage(self: *DraftRound) !void {
            const m = self.m;
            const dspark = &self.dspark;
            const p = &m.pipelines;
            const ws = &m.workspace;
            const aw = &m.attention_workspace;
            const state = self.state;
            const pos_now = self.pos_now;
            const cap_n = self.cap_n;
            const cap_pos = self.cap_pos;
            const true_pos = self.true_pos;
            const cap_rows = self.cap_rows;
            const draft_pos = self.draft_pos;
            const dspark_capture_all = self.dspark_capture_all;
            const tail = self.tail;
            const pub_pos = self.pub_pos;
            const nc = self.nc;
            const cond_prof = self.cond_prof;
            var enc = self.enc;
            var cb = self.cb;
            defer {
                self.enc = enc;
                self.cb = cb;
            }
            if (nc > 1) {
                const base = m.cond.?;
                const cond = View{ .buffer = base, .offset = 0 };
                const raw = View{ .buffer = base, .offset = 128 * 5120 * 2 };
                const chunk = View{ .buffer = base, .offset = 128 * 5120 * 2 + 128 * 512 * 2 };
                // The ring is circular: the oldest live row is `first`, and the last `nA` rows are contiguous there. With
                // the aligned ring the window's first row is the slot of the position pos_now - nc.
                const first: u32 = if (condAlignOn()) (pos_now - nc) % 128 else (m.taps_row + 128 - nc) % 128;
                const n_a: u32 = @min(nc, 128 - first);
                // The eight-row dense hint -- this projection's window wraps, so the smaller of its two pieces runs on
                // 1..63 rows, and the tile's gate is 64.
                try m.projectDense8(enc, dspark.main_proj, .{ .buffer = m.taps, .offset = @as(usize, first) * 3 * 5120 * 2 }, cond, n_a, 3 * 5120, 5120, .bf16);
                if (n_a < nc)
                    try m.projectDense8(enc, dspark.main_proj, .{ .buffer = m.taps, .offset = 0 }, View{ .buffer = cond.buffer, .offset = cond.offset + @as(usize, n_a) * 5120 * 2 }, nc - n_a, 3 * 5120, 5120, .bf16);
                m.normalize(enc, cond, dspark.main_norm, cond, nc);
                enc.barrier();
                // The conditioning's parts, one command buffer each.
                if (cond_prof.on) cond_prof.split("cond", "tapproj+norm", 0, &enc);
                cb = cond_prof.cb;
                // THE THREE PER-BLOCK `wkv` PROJECTIONS AS ONE GROUPED INDEXED DISPATCH. The grouped form lays the window
                // out three times (one expert a 128-row tile group) and takes one dispatch with three times the
                // threadgroups, so the same bytes decode in parallel. TF_DSV41_NO_COND_WKV_FUSE=1 restores the three
                // calls (the A/B).
                const hd = m.cfg.head_dim;
                const fuse_wkv = m.cond_wkv != null and dspark.cond_wkv_descriptors != null and
                    std.c.getenv("TF_DSV41_NO_COND_WKV_FUSE") == null;
                var raw_kv = [3]View{ raw, raw, raw };
                if (fuse_wkv) {
                    const cb3 = m.cond_wkv.?;
                    const in3 = View{ .buffer = cb3, .offset = 0 };
                    const out3 = View{ .buffer = cb3, .offset = 3 * 128 * 5120 * 2 };
                    const ids_off: usize = 3 * 128 * 5120 * 2 + 3 * 128 * hd * 2;
                    for (0..3) |g| ops.encodeCopy(enc, &p.ops, cond, .{ .buffer = cb3, .offset = @as(usize, g) * 128 * 5120 * 2 }, .{ .rows = nc, .width = 5120, .x_stride = 5120, .y_stride = 5120 });
                    enc.barrier();
                    try exl3.encodeIndexed(&p.projection, enc, .{ .descriptors = .{ .buffer = dspark.cond_wkv_descriptors.?, .offset = 0 }, .count = 3, .k = 5120, .n = hd }, convert(in3), convert(out3), m.workspace.projection(), .{
                        .count = 3 * 128,
                        .x_stride = 5120,
                        .y_stride = hd,
                        .output = .bf16,
                        .input_ids = .{ .buffer = cb3, .offset = ids_off + 384 * 4 },
                        .input_rows = 3 * 128,
                        .output_ids = .{ .buffer = cb3, .offset = ids_off + 768 * 4 },
                        .output_rows = 3 * 128,
                        .routes = .{ .ids = .{ .buffer = cb3, .offset = ids_off }, .repeat = 1 },
                        .grouped = true,
                        .tile_rows = 64,
                    });
                    enc.barrier();
                    if (cond_prof.on) cond_prof.split("cond", "wkv3", 0, &enc);
                    cb = cond_prof.cb;
                    for (0..3) |g| raw_kv[g] = .{ .buffer = out3.buffer, .offset = out3.offset + @as(usize, g) * 128 * hd * 2 };
                }
                var s_i: u32 = 0;
                while (s_i < dspark.blocks.len) : (s_i += 1) {
                    const blk = &dspark.blocks[s_i];
                    if (!fuse_wkv) {
                        const cond_wkv_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_COND_WKV") != null) 2 else 1;
                        for (0..cond_wkv_passes) |_| try m.project(enc, blk.attn.wkv, cond, raw, nc, 5120, hd, .bf16);
                        // The conditioning's parts, one command buffer each.
                        if (cond_prof.on) cond_prof.split("cond", "wkv", s_i, &enc);
                        cb = cond_prof.cb;
                    }
                    const raw_g = if (fuse_wkv) raw_kv[@intCast(s_i)] else raw;
                    ops.encodeRms(enc, &p.ops, raw_g, tensor(blk.attn.kv_norm), raw_g, .{ .rows = nc, .width = hd, .x_stride = hd, .y_stride = hd, .eps = m.cfg.eps });
                    enc.barrier();
                    if (cond_prof.on) cond_prof.split("cond", "rms", s_i, &enc);
                    cb = cond_prof.cb;
                    // The release roped the main rows' keys at their own positions before they enter the draft ring.
                    if (condRopeOn()) {
                        try p.attention.rotary.encode(enc, raw_g, .{ .rows = nc, .heads = 1, .width = hd, .rotary = m.cfg.rope_dim, .stride = hd, .position = pos_now - nc, .step = 1, .inverse = 0, .spectrum = .normal });
                        enc.barrier();
                    }
                    // The conditioning's parts, one command buffer each.
                    if (cond_prof.on) cond_prof.split("cond", "rope", s_i, &enc);
                    cb = cond_prof.cb;
                    try p.attention.cache.encodePack(enc, raw_g, chunk, .{ .rows = nc, .width = hd, .stride = hd, .packed_stride = kv.cache.ring_stride, .block = 32, .kind = .window_fp8 });
                    enc.barrier();
                    // The conditioning's parts, one command buffer each.
                    if (cond_prof.on) cond_prof.split("cond", "pack", s_i, &enc);
                    cb = cond_prof.cb;
                    try p.attention.cache.publishWindow(enc, chunk, try state.ring(s_i), nc, pos_now - nc);
                    enc.barrier();
                    if (cond_prof.on) cond_prof.split("cond", "publish", s_i, &enc);
                    cb = cond_prof.cb;
                }
            } else {
                // The release ropes the committed main rows' keys at their own position before they enter the stage's
                // ring (its DSparkAttention applies the rotary to `main_kv`); TF_DSV41_DSPARK_NO_RING_ROPE=1 restores the
                // earlier unroped publish.
                const dspark_ring_rope = std.c.getenv("TF_DSV41_DSPARK_NO_RING_ROPE") == null;
                if (std.c.getenv("TF_DSV41_MTP_DEBUG_ROWS") != null) std.log.info(
                    "mtp cond: taps_row={d} taps_n={d} pos_now={d} publish_pos={d} tap_row={d} ring_rope={} || cap_n={d} cap_pos={d} tail={d} valid={d} pub_pos={d} draft_pos={d} true_pos={}",
                    .{ m.taps_row, m.taps_n, pos_now, pos_now -| 1, (m.taps_row + 127) % 128, dspark_ring_rope, cap_n, cap_pos, tail, m.taps_rows_valid, pub_pos, draft_pos, true_pos },
                );
                // The conditioning publishes the taps ring's whole tail -- the rows the last capture kept, which are
                // consecutive positions -- at their own positions. After a prompt the first round published one of the
                // prompt's 128 tail rows, so the window was uninitialised for the first 128 rounds.
                // TF_DSV41_DSPARK_COND_ONE restores the earlier one-row form. The capture's rows are the ring's tail, but
                // the rollback keeps only a PREFIX of the capture: the rejected rows' taps are not the committed history,
                // so the conditioning publishes the prefix, not the tail. The prefix's rows are the capture's FIRST rows,
                // so their positions are the capture's own -- [cap_pos, cap_pos + tail) -- and the ring's slots follow
                // from them. The earlier form's [pos_now - tail, pos_now) are the SUFFIX's positions. The published rows
                // are the ring's [base, base+tail); the capture's first row is its base, and the rows may wrap the ring's
                // end -- then the second piece starts at row zero. With the aligned ring the capture's first row is the
                // slot of the position pos_now - tail (the kept prefix's first position).
                const base: u32 = if (condAlignOn())
                    (pos_now - tail) % 128
                else if (m.taps_rows_valid != 0)
                    (m.taps_row + 128 - cap_rows) % 128
                else
                    (m.taps_row + 128 - tail) % 128;
                // The published rows are the capture's whole (the main model's rows at their own positions, the rejected
                // ones included) or only the kept prefix (the earlier form). The publish's position is the capture's
                // first row's position either way, so the rows land where they belong.
                const pub_rows: u32 = if (dspark_capture_all != 0) cap_rows else tail;
                const n_a: u32 = @min(pub_rows, 128 - base);
                // The doubling arm for the conditioning's tap projection (idempotent: the same input gives the same
                // output, so the reply is unchanged and only the cost doubles).
                const cond_proj_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_COND_PROWS") != null) 2 else 1;
                for (0..cond_proj_passes) |_| try m.project(enc, dspark.main_proj, .{ .buffer = m.taps, .offset = @as(u64, base) * 3 * 5120 * 2 }, ws.view(.shared_down), n_a, 3 * 5120, 5120, .bf16);
                if (n_a < pub_rows) {
                    const dst = ws.view(.shared_down);
                    try m.project(enc, dspark.main_proj, .{ .buffer = m.taps, .offset = 0 }, View{ .buffer = dst.buffer, .offset = dst.offset + @as(usize, n_a) * 5120 * 2 }, pub_rows - n_a, 3 * 5120, 5120, .bf16);
                }
                m.normalize(enc, ws.view(.shared_down), dspark.main_norm, ws.view(.shared_down), pub_rows);
                enc.barrier();
                const chunk = aw.selection.windowChunk();
                var s_i: u32 = 0;
                while (s_i < dspark.blocks.len) : (s_i += 1) {
                    const blk = &dspark.blocks[s_i];
                    const raw = aw.view(.raw_kv);
                    try m.project(enc, blk.attn.wkv, ws.view(.shared_down), raw, pub_rows, 5120, m.cfg.head_dim, .bf16);
                    ops.encodeRms(enc, &p.ops, raw, tensor(blk.attn.kv_norm), raw, .{ .rows = pub_rows, .width = m.cfg.head_dim, .x_stride = m.cfg.head_dim, .y_stride = m.cfg.head_dim, .eps = m.cfg.eps });
                    enc.barrier();
                    if (dspark_ring_rope) {
                        // Every published row carries its rope phase. The rope covered `tail` rows (the kept prefix)
                        // while the pack wrote `pub_rows`, so rows past the first were packed unroped and the drafter's
                        // attention read unroped keys against a roped query. =1 ropes at pub_pos (the earlier positions);
                        // =2 ropes at cap_pos (their own positions, the release's form).
                        const rope_mode = mtpRopeAllOn();
                        const rope_rows: u32 = if (rope_mode != 0) pub_rows else tail;
                        const rope_pos: u32 = if (rope_mode == 2) cap_pos else pub_pos;
                        try p.attention.rotary.encode(enc, raw, .{ .rows = rope_rows, .heads = 1, .width = m.cfg.head_dim, .rotary = m.cfg.rope_dim, .stride = m.cfg.head_dim, .position = rope_pos, .step = 1, .inverse = 0, .spectrum = .normal });
                        enc.barrier();
                    }
                    try p.attention.cache.encodePack(enc, raw, chunk, .{ .rows = pub_rows, .width = m.cfg.head_dim, .stride = m.cfg.head_dim, .packed_stride = kv.cache.ring_stride, .block = 32, .kind = .window_fp8 });
                    enc.barrier();
                    // The instrument arm TF_DSV41_DSPARK_NO_COND skips the publish, so the drafter's ring keeps whatever
                    // it held (the prompt's tail) -- a STALE context. If the drafter's head input and the trunk's barely
                    // move, the ring's contents never reach the drafter's output.
                    if (std.c.getenv("TF_DSV41_DSPARK_NO_COND") == null)
                        // The value 2 computes the capture's whole rows but publishes only the kept prefix, so the arm
                        // separates the ring's CONTENT from the extra rows' COMPUTATION.
                        try p.attention.cache.publishWindow(enc, chunk, try state.ring(s_i), if (dspark_capture_all == 2) tail else pub_rows, pub_pos);
                    enc.barrier();
                }
            }
        }

        /// `Transformer.forward_spec`'s stage loop: three `DSparkBlock.forward` calls, each with the mHC
        /// expand/collapse around its attention and MoE. The release threads `pre_mix` and `main_x` through
        /// the same three blocks; the bodies are this lane's dispatch shapes.
        pub fn blockStage(self: *DraftRound, dbg: ?mtl.Buffer) !void {
            const m = self.m;
            const dspark = &self.dspark;
            const state = self.state;
            const slot = self.slot;
            const n = self.n;
            const draft_pos = self.draft_pos;
            const cond_prof = self.cond_prof;
            const ws = &m.workspace;
            const p = &m.pipelines;
            var enc = self.enc;
            var cb = self.cb;
            defer {
                self.enc = enc;
                self.cb = cb;
            }
            var k: u32 = 0;
            while (k < dspark.blocks.len) : (k += 1) {
                const blk = &dspark.blocks[k];
                cond_prof.block = k;
                m.mixes(enc, blk.hc_attn, ws.view(.attention_pre), n);
                mhc.encodeCollapse(enc, &p.mhc, ws.view(.stream), ws.view(.previous_pre), ws.view(.collapsed), .{ .rows = n, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
                enc.barrier();
                if (cond_prof.blk_on) cond_prof.split("blk", "mhc_attn", k, &enc);
                // The mHC's collapsed rows (the norm's input) and their magnitudes.
                if (dbg) |b| {
                    if (k == 0) {
                        ops.encodeDump(enc, &p.ops, ws.view(.collapsed), .{ .buffer = b, .offset = mtp_dbg_coll * 4 }, n * 5120, true);
                        ops.encodeDump(enc, &p.ops, ws.view(.previous_pre), .{ .buffer = b, .offset = mtp_dbg_pre * 4 }, n * 4, false);
                    }
                }
                m.normalize(enc, ws.view(.collapsed), blk.attn_norm, ws.view(.normalized), n);
                if (cond_prof.blk_on) cond_prof.split("blk", "attn_norm", k, &enc);
                if (dbg) |b| if (k == 0) ops.encodeDump(enc, &p.ops, ws.view(.normalized), .{ .buffer = b, .offset = 1 * 5 * 5120 * 4 }, 5 * 5120, true);
                try m.draftAttention(enc, blk, state, slot, k, n, @intCast(if (std.c.getenv("TF_DSV41_DSPARK_POS1") != null) (draft_pos -| 1) else draft_pos), dbg, cond_prof);
                enc = cond_prof.enc;
                if (dbg) |b| if (k == 0) ops.encodeDump(enc, &p.ops, ws.view(.branch), .{ .buffer = b, .offset = 2 * 5 * 5120 * 4 }, 5 * 5120, true);
                m.post(enc, n);
                m.mixes(enc, blk.hc_ffn, ws.view(.previous_pre), n);
                mhc.encodeCollapse(enc, &p.mhc, ws.view(.stream), ws.view(.attention_pre), ws.view(.collapsed), .{ .rows = n, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
                enc.barrier();
                if (cond_prof.blk_on) cond_prof.split("blk", "mhc_ffn", k, &enc);
                m.normalize(enc, ws.view(.collapsed), blk.ffn_norm, ws.view(.normalized), n);
                if (cond_prof.blk_on) cond_prof.split("blk", "ffn_norm", k, &enc);
                if (dbg) |b| if (k == 0) ops.encodeDump(enc, &p.ops, ws.view(.normalized), .{ .buffer = b, .offset = 3 * 5 * 5120 * 4 }, 5 * 5120, true);
                try m.moe(&enc, null, &blk.moe, n, 0, m.cfg.dspark_topk, cond_prof);
                m.post(enc, n);
                // The branch after the MoE (the stage contributions: the attention's output against the MoE's), so the
                // row's composition is read off the magnitudes.
                if (dbg) |b| if (k == 0) ops.encodeDump(enc, &p.ops, ws.view(.branch), .{ .buffer = b, .offset = 4 * 5 * 5120 * 4 }, 5 * 5120, true);
                enc = cond_prof.enc;
                cb = cond_prof.cb;
            }
        }

        /// `DSparkBlock.forward_head`'s projection half: the last block's mHC collapse and norm, then the
        /// shared head into the drafter's logits rows.
        pub fn headStage(self: *DraftRound) !View {
            const m = self.m;
            const p = &m.pipelines;
            const ws = &m.workspace;
            const dspark = &self.dspark;
            const n = self.n;
            var enc = self.enc;
            defer self.enc = enc;
            mhc.encodeCollapse(enc, &p.mhc, ws.view(.stream), ws.view(.previous_pre), ws.view(.collapsed), .{ .rows = n, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
            enc.barrier();
            m.normalize(enc, ws.view(.collapsed), dspark.norm, ws.view(.normalized), n);
            const logits = View{ .buffer = m.mtp_logits.?, .offset = 2 * 129280 * 4 };
            try m.project(enc, m.weights.head, ws.view(.normalized), logits, n, 5120, 129280, .f32);
            enc.barrier();
            return logits;
        }

        /// `DSparkBlock.forward_head`'s chain: row k's logits plus the rank-256 Markov bias of the token
        /// before it, its argmax the next token. The release runs the same loop over `block_size`.
        pub fn markovStage(self: *DraftRound, logits: View) !MarkovRun {
            const m = self.m;
            const p = &m.pipelines;
            const dspark = &self.dspark;
            const n = self.n;
            const pending = self.pending;
            var enc = self.enc;
            defer self.enc = enc;
            // The Markov-biased chain (the head chain): row k's logits plus the rank-256 bias of the token before it, its
            // argmax the next token; five steps from the pending token to four drafts. The release's
            // `DSparkBlock.forward_head` runs the same loop (inference/model.py, revision 314113f8).
            const emb = dspark.markov_embed orelse return error.NoMarkov;
            const head_embed = dspark.markov_head orelse return error.NoMarkov;
            if (m.markov_ids == null) {
                m.markov_ids = try m.device.buffer(8 * 4, mtl.ResourceOptions.shared);
                m.residency.add(m.markov_ids.?);
            }
            // The chain's step 0 reads the token before the first draft; its picks land in this buffer, seeded here.
            std.mem.writeInt(u32, m.markov_ids.?.contents()[0..4], pending, .little);
            const ids_view = View{ .buffer = m.markov_ids.?, .offset = 0 };
            const part = try m.markovPart();
            const groups: u32 = (129280 + 255) / 256;
            // The tree price needs the *biased* logits' top-2, and PART is the only place the biased logits survive (the
            // pick consumes them). PART is overwritten by the next step, so each step's 505 float2 bests are copied into
            // its own region before the barrier.
            const price_dbg: ?mtl.Buffer = if (std.c.getenv("TF_DSV41_TREE_PRICE") != null)
                try m.device.buffer(5 * 505 * 8, mtl.ResourceOptions.shared)
            else
                null;
            var step: u32 = 0;
            while (step < n) : (step += 1) {
                ops.encodeMarkov(enc, &p.ops, logits, tensor(emb), tensor(head_embed), ids_view, part, .{ .vocab = 129280, .rank = 256, .step = @intCast(step), .pad = @intFromBool(markovBiasOffOn()) });
                if (price_dbg) |b| ops.encodeDump(enc, &p.ops, part, .{ .buffer = b, .offset = @as(u64, step) * groups * 8 }, groups * 2, false);
                enc.barrier();
                ops.encodeMarkovPick(enc, &p.ops, part, ids_view, groups, step);
                enc.barrier();
            }
            return .{ .price_dbg = price_dbg, .groups = groups };
        }
    };

    /// The stream the boundary state currently lives in (the fused path swaps two buffers).
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
