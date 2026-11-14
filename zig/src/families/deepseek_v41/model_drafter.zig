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
const model = @import("model.zig");
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

fn drafterIds(m: *Model, rows: u32) !mtl.Buffer {
    const bytes = @max((@as(usize, rows) + 1) * 4, 6 * 4);
    if (m.drafter_ids) |ids| {
        if (ids.length() >= bytes) return ids;
    }
    const ids = try m.device.buffer(bytes, mtl.ResourceOptions.shared);
    m.residency.add(ids);
    if (m.drafter_ids) |old| {
        m.residency.remove(old);
        old.deinit();
    }
    m.drafter_ids = ids;
    return ids;
}

pub fn drafterId(m: *Model, token: u32) !View {
    const ids = try drafterIds(m, 1);
    std.mem.writeInt(u32, ids.contents()[0..4], token, .little);
    return .{ .buffer = ids };
}
pub fn drafterRound(m: *Model, pending: u32, rows: u32) !View {
    const ids = try drafterIds(m, rows);
    const bytes = ids.contents();
    var r: u32 = 0;
    while (r < rows) : (r += 1)
        std.mem.writeInt(u32, bytes[@as(usize, r) * 4 ..][0..4], if (r == 0) pending else 128799, .little);
    return .{ .buffer = ids };
}
pub fn markovPart(m: *Model) !View {
    if (m.markov_part == null) {
        const groups: usize = (129280 + 255) / 256;
        m.markov_part = try m.device.buffer(groups * 8, mtl.ResourceOptions.shared);
        m.residency.add(m.markov_part.?);
    }
    return .{ .buffer = m.markov_part.? };
}
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
    // The prompt-row int8 twin.
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
pub fn projectDense8(m: *Model, enc: mtl.ComputeEncoder, matrix: exl3.Matrix, x: View, y: View, rows: u32, xs: usize, ys: usize, output: exl3.Output) !void {
    try exl3.encode(&m.pipelines.projection, enc, matrix, convert(x), convert(y), m.workspace.projection(), .{ .count = rows, .x_stride = xs, .y_stride = ys, .output = output, .dense8 = true });
    enc.barrier();
}
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
pub fn projectDebugSums(m: *Model, enc: mtl.ComputeEncoder, matrix: exl3.Matrix, x: View, y: View, rows: u32, xs: usize, ys: usize, output: exl3.Output, b: mtl.Buffer) !void {
    try exl3.encode(&m.pipelines.projection, enc, matrix, convert(x), convert(y), m.workspace.projection(), .{ .count = rows, .x_stride = xs, .y_stride = ys, .output = output });
    enc.barrier();
    const sc = m.workspace.projection();
    const sums_off = (@as(usize, rows) * matrix.k * 2 + 15) & ~@as(usize, 15);
    const sums = View{ .buffer = sc.buffer, .offset = sc.offset + sums_off };
    ops.encodeDump(enc, &m.pipelines.ops, sums, .{ .buffer = b, .offset = mtp_dbg_woasums * 4 }, @intCast(rows * matrix.n), false);
    enc.barrier();
}
pub fn expertPasses() u32 {
    const v = std.c.getenv("TF_DSV41_DOUBLE_EXPERTS") orelse return 1;
    const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 2;
    return @max(1, @min(n, 8));
}
pub fn moeTileRows() u32 {
    // The grouped plan pads each expert's block to the tile's own width.
    return expert_groups.tileRows();
}
pub fn mtpWarmRounds() u32 {
    if (Model.mtp_warm_cache) |v| return v;
    const raw = std.c.getenv("TF_DSV41_MTP_ADAPT_WARM") orelse {
        Model.mtp_warm_cache = 4;
        return 4;
    };
    const n = std.fmt.parseInt(u32, std.mem.span(raw), 10) catch 0;
    Model.mtp_warm_cache = @min(n, 16);
    return Model.mtp_warm_cache.?;
}
pub fn noteAcceptSlot(m: *Model, slot: u32, hit: bool) void {
    if (slot >= m.mtp_accept_slot.len) return;
    const v: f64 = if (hit) 1.0 else 0.0;
    // THE FIRST SAMPLE SETS THE SIGNAL: a request's first round is a valid observation of the first draft's acceptance.
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
pub fn mtpRoundSlot(m: *Model, slot: u32) f64 {
    if (slot >= m.mtp_round_slot.len) return 0;
    return m.mtp_round_slot[slot];
}
pub fn mtpAcceptSlot(m: *Model, slot: u32) f64 {
    if (slot >= m.mtp_accept_slot.len) return 0;
    return m.mtp_accept_slot[slot];
}
pub fn resetAcceptSlot(m: *Model, slot: u32) void {
    if (slot >= m.mtp_accept_slot.len) return;
    m.mtp_accept_slot[slot] = 0;
    m.mtp_accept_seen_slot[slot] = 0;
    m.mtp_round_slot[slot] = 0;
    m.mtp_round_seen_slot[slot] = 0;
}
