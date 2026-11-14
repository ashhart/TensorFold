//! The model's remaining methods.
const builtin = @import("builtin");
const std = @import("std");
const mtl = @import("metal");
const checkpoint = @import("core").checkpoint_metal;
const exl3 = @import("core").exl3_projection;
const config = @import("config.zig");
const load = @import("load.zig");
const weights = @import("weights.zig");
const work = @import("workspace.zig");
const attention = @import("attention.zig");
const attention_common = @import("attention_common.zig");
const kv = @import("kv.zig");
const ops = @import("ops.zig");
const mhc = @import("mhc.zig");
const csa2 = @import("csa2.zig");
const csa2_index = @import("csa2_index.zig");
const rope = @import("rope.zig");
const route = @import("route.zig");
const engram = @import("engram.zig");
const engram_stream = @import("engram_rows_safetensors.zig");
const gpu_counters = @import("profiler.zig");
const expert_groups = @import("core").expert_groups;
const scan_probe = @import("scan_probe.zig");
const projection_knobs = @import("projection_knobs.zig");
const model = @import("model.zig");
const forward_cone = @import("forward_cone.zig");
const Model = model.Model;
const CondProf = Model.CondProf;
const Snapshot = Model.Snapshot;
const Pipelines = model.Pipelines;
const Options = model.Options;
const View = model.View;
const Logits = model.Logits;
const Seg = model.Seg;
const HidHash = model.HidHash;
const tensor = model.tensor;
const asF32 = model.asF32;
const convert = model.convert;
const splitStage = model.splitStage;
const completed = model.completed;
const nowNs = model.nowNs;
const prefill_step = model.prefill_step;
const mtp_dbg_raw = model.mtp_dbg_raw;
const mtp_dbg_qr = model.mtp_dbg_qr;
const mtp_dbg_qpre = model.mtp_dbg_qpre;
const mtp_dbg_dense = model.mtp_dbg_dense;
const mtp_dbg_qpost = model.mtp_dbg_qpost;
const mtp_dbg_rot = model.mtp_dbg_rot;
const mtp_dbg_sums = model.mtp_dbg_sums;
const mtp_dbg_coll = model.mtp_dbg_coll;
const mtp_dbg_pre = model.mtp_dbg_pre;
const mtp_dbg_pin = model.mtp_dbg_pin;
const mtp_dbg_prot = model.mtp_dbg_prot;
const mtp_dbg_psum = model.mtp_dbg_psum;
const mtp_dbg_total = model.mtp_dbg_total;
const mtp_dbg_grouped = model.mtp_dbg_grouped;
const mtp_dbg_total2 = model.mtp_dbg_total2;
const mtp_dbg_woasums = model.mtp_dbg_woasums;
const mtp_dbg_total3 = model.mtp_dbg_total3;
const tapsOn = model.tapsOn;
const mtpStopPoint = model.mtpStopPoint;
const markovBiasOffOn = model.markovBiasOffOn;
const dsparkTruePosOn = model.dsparkTruePosOn;
const tapsSharedBaseOn = model.tapsSharedBaseOn;
const partialTxnOn = model.partialTxnOn;
const ringKeepOn = model.ringKeepOn;
const max_ring_rows = model.max_ring_rows;
const mtpRopeAllOn = model.mtpRopeAllOn;
const condRopeOn = model.condRopeOn;
const mtpTrunkRing = model.mtpTrunkRing;
const condAlignOn = model.condAlignOn;
const condRowsWanted = model.condRowsWanted;
const concurrentOn = model.concurrentOn;
const headRms = model.headRms;
const logitsTopK = model.logitsTopK;

pub fn open(gpa: std.mem.Allocator, io: std.Io, o: Options) !Model {
    try @import("env_validate.zig").startup();
    const mtp_enabled = try @import("sys").environment.read(bool, "TF_DSV41_MTP_DRAFTS", false);
    projection_knobs.apply();
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
    // A draft round runs five rows through the same workspace and attention scratch; at decode the plan is one row.
    const drafted = mtp_enabled or std.c.getenv("TF_DSV41_LOOKUP_DRAFTS") != null;
    const round_rows: u32 = if (drafted) @max(o.chunk_rows, 8) else o.chunk_rows;
    const wp = try work.Plan.initLanes(round_rows, lanes);
    const kp = kv.cache.Plan{ .capacity = capacity, .layers = 40 };
    const workspace_bytes = try std.math.add(usize, wp.bytes, attention.Workspace.bytes(o.chunk_rows));
    const cache_bytes = try std.math.add(usize, kp.residentBytes(), weights.sizeofDescriptors());
    const execution_bytes = try std.math.add(usize, workspace_bytes, try std.math.mul(usize, cache_bytes, lanes));
    const rotary_bytes = attention.Pipelines.residentBytes(capacity, c);
    const needed = try std.math.add(usize, plan.bytes, try std.math.add(usize, execution_bytes, rotary_bytes));
    // The int8 twin lives beside the trellis it supersedes at prompt rows.
    const int8_extra: usize = if (exl3.knobs.i8_prefill) 5 << 30 else 0;
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
    // The prompt-row int8 sidecar.
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
    // The conditioning scratch: 128 rows of 5120 bf16, then 128 rows of 512 bf16 keys, then 128 packed rows.
    const mtp_logits = try device.buffer(8 * 129280 * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer mtp_logits.deinit();
    const cond_bytes = 128 * 5120 * 2 + 128 * 512 * 2 + 128 * 528;
    const cond = try device.buffer(cond_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer cond.deinit();
    // The fused `wkv` dispatch's scratch.
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
    residency.add(taps);
    residency.add(cond_wkv);
    var stream_rows: ?*engram_stream.Rows = null;
    var stream_stage: ?mtl.Buffer = null;
    // Invariant: only this block creates stream_rows and stream_stage; keep them in step with weights.bind.
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
    residency.commit();
    residency.requestResidency();
    queue.addResidencySet(residency);
    return .{ .gpa = gpa, .io = io, .device = device, .queue = queue, .residency = residency, .checkpoint = cp, .weights = bound, .mtp_enabled = mtp_enabled, .cfg = c, .cfg_arena = arena, .lanes = lanes, .caches = caches, .cache = &caches[0], .engram_stash = engram_stash, .partial_stash = partial_stash, .ring_stash = ring_stash, .snap_ring = snap_ring, .taps_stash = taps_stash, .stream = stream_rows, .stream_stage = stream_stage, .workspace = workspace, .attention_workspace = aw, .taps = taps, .cond = cond, .cond_wkv = cond_wkv, .mtp_logits = mtp_logits, .pipelines = pipelines, .admitted_bytes = needed, .load_seconds = @as(f64, @floatFromInt(std.Io.Timestamp.now(io, .awake).toNanoseconds() - load_started.toNanoseconds())) / 1e9 };
}
pub fn deinit(m: *Model) void {
    forward_cone.release(m);
    m.queue.removeResidencySet(m.residency);
    m.residency.deinit();
    if (m.drafter_ids) |ids| ids.deinit();
    if (m.markov_ids) |ids| ids.deinit();
    if (m.markov_part) |part| part.deinit();
    if (m.drafter_state) |state| {
        state.deinit();
        m.gpa.destroy(state);
    }
    @import("model_lookahead.zig").deinit(m);
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
    m.taps.deinit();
    if (m.cond) |b| b.deinit();
    if (m.mtp_logits) |b| b.deinit();
    if (m.gate_x) |b| b.deinit();
    if (m.dspark_logits) |b| b.deinit();
    if (m.cond_wkv) |b| b.deinit();
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
pub fn resetSlot(m: *Model, slot: u32) !void {
    forward_cone.drop(m, slot);
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

/// A KEPT PROMPT STATE: one slot's whole cache, cursor, Engram history and taps, so resumed equals fresh.
pub fn beginSnapTail(m: *Model, slot: u32, hist: []const u32, at: u32, from: u32) bool {
    if (hist.len < 3) return false;
    const state = &m.caches[slot];
    const q = at - from;
    const n: u32 = @min(q, kv.cache.window);
    // THE RECORD IS THE HASH PASS'S, NOT RAW TOKENS: `int tail[3]` of COMPRESSED ids, newest first.
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
/// The tail mark is captured inside its chunk, except under the cone: that cuts at the mark and replays there.
pub fn snapTailAllowed(m: *Model) bool {
    return condAlignOn() and !forward_cone.active(m);
}
pub fn endSnapTail(m: *Model) void {
    m.publish_limit = 0;
    m.attention_workspace.publish_limit = 0;
    m.attention_workspace.publish_stash = null;
}
pub fn snapshotBytes(m: *const Model) u64 {
    const plan = m.caches[0].plan;
    return @as(u64, plan.packedBytes()) + plan.mutableBytes() + m.taps.length();
}
pub fn snapshotSave(m: *Model, slot: u32) !*Snapshot {
    try forward_cone.flush(m, slot);
    if (slot >= m.lanes) return error.InvalidLaneSlot;
    const state = &m.caches[slot];
    const snap = try m.gpa.create(Snapshot);
    errdefer m.gpa.destroy(snap);
    const storage = try m.gpa.dupe(u8, state.storage.contents()[0..state.plan.packedBytes()]);
    errdefer m.gpa.free(storage);
    const live = try m.gpa.dupe(u8, state.live.contents()[0..state.plan.mutableBytes()]);
    errdefer m.gpa.free(live);
    // A merged tail chunk's copy is THE STATE AT THE MARK, not at the chunk end.
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
pub fn snapshotRestore(m: *Model, slot: u32, snap: *const Snapshot) !void {
    forward_cone.drop(m, slot);
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
pub fn beginSlot(m: *Model, slot: u32) !void {
    if (slot >= m.lanes) return error.InvalidLaneSlot;
    const view = try m.workspace.historyView(slot);
    const history: [*]u8 = @ptrCast(@alignCast(view.buffer.contents() + view.offset));
    @memcpy(m.engram_stash[slot * 16 ..][0..16], history[0..16]);
    try m.caches[slot].begin();
    // Stash the window ring's next slots before the verify writes them.
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
pub fn portKeep(m: *Model, slot: u32, kept: u32) !void {
    if (slot >= m.lanes) return error.InvalidLaneSlot;
    // The adaptive depth policy's signal.
    const state = &m.caches[slot];
    const pos = state.position + kept;
    // The pool's groups are cut at even positions, so a new position that is odd leaves the group {pos-1, pos} open.
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
