//! Plain V4.1 execution over resident EXL3 weights and source-owned packed caches.
const std = @import("std");
const mtl = @import("metal");
const checkpoint = @import("../../core/checkpoint_metal.zig");
const exl3 = @import("../../core/exl3_projection.zig");
const config = @import("config.zig");
const load = @import("load.zig");
const weights = @import("weights.zig");
const work = @import("workspace.zig");
pub const attention = @import("attention.zig");
pub const kv = @import("kv.zig");
pub const ops = @import("ops.zig");
pub const mhc = @import("mhc.zig");
pub const csa2 = @import("csa2.zig");
pub const csa2_index = @import("csa2_index.zig");
pub const rope = @import("rope.zig");
const route = @import("route.zig");
const engram = @import("engram.zig");
const View = @import("view.zig").View;
const gpu_counters = @import("profiler.zig");
const expert_groups = @import("../../core/expert_groups.zig");

pub const prefill_step = 256;
pub const Logits = enum { none, last, all };
pub const Options = struct {
    dir: []const u8,
    table_dir: ?[]const u8 = null,
    token_map: []const u8,
    context: u32 = 0,
    chunk_rows: u32 = prefill_step,
    /// Lane slots, each with its own caches and Engram cursor; 1 keeps the single-lane layout.
    lanes: u32 = 1,
    /// Refuse the load past this many bytes; null admits to Metal's working set minus the margin.
    ram_limit_bytes: ?u64 = null,
};

const Pipelines = struct {
    projection: exl3.Pipelines,
    ops: ops.Pipelines,
    mhc: mhc.Pipelines,
    route: route.Pipelines,
    engram: engram.Pipelines,
    attention: attention.Pipelines,
    groups: expert_groups.Pipelines,

    fn init(device: mtl.Device, capacity: u32, c: config.Config) !Pipelines {
        var ep = try exl3.Pipelines.load(device);
        errdefer ep.deinit();
        var op = try ops.Pipelines.load(device);
        errdefer op.deinit();
        var hp = try mhc.Pipelines.load(device);
        errdefer hp.deinit();
        var rp = try route.Pipelines.load(device);
        errdefer rp.deinit();
        var gp = try engram.Pipelines.load(device);
        errdefer gp.deinit();
        var groups = try expert_groups.Pipelines.load(device);
        errdefer groups.deinit();
        const ap = try attention.Pipelines.load(device, capacity, c);
        return .{ .projection = ep, .ops = op, .mhc = hp, .route = rp, .engram = gp, .attention = ap, .groups = groups };
    }

    fn deinit(p: *Pipelines) void {
        p.attention.deinit();
        p.engram.deinit();
        p.route.deinit();
        p.groups.deinit();
        p.mhc.deinit();
        p.ops.deinit();
        p.projection.deinit();
    }
};

/// Serialized unique owner; submissions finish before return/release, cancellation is between bounded chunks.
pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
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
    /// Each slot's Engram cursor as of its beginSlot, restored by rollbackSlot.
    engram_stash: []u8,
    workspace: work.Workspace,
    attention_workspace: attention.Workspace,
    pipelines: Pipelines,
    admitted_bytes: usize,
    load_seconds: f64 = 0,
    last_gpu_seconds: f64 = 0,
    logits_rows: usize = 0,
    valid: bool = true,
    profile: ?*gpu_counters.Recorder = null,
    grouped_prefill: bool = true,
    observer: ?Observer = null,
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
        const table_dir = o.table_dir orelse (if (from_config.len == 0) return error.MissingEngramTables else from_config);
        var plan = try load.Plan.open(gpa, io, o.dir, table_dir);
        defer plan.deinit();
        try weights.validate(&plan);
        const map_blob = try std.Io.Dir.cwd().readFileAlloc(io, o.token_map, arena.allocator(), .limited(2 << 20));
        const map = try engram.hash.TokenMap.parse(map_blob);
        const lanes = o.lanes;
        const wp = try work.Plan.initLanes(o.chunk_rows, lanes);
        const kp = kv.cache.Plan{ .capacity = capacity, .layers = 40 };
        const workspace_bytes = try std.math.add(usize, wp.bytes, attention.Workspace.bytes(o.chunk_rows));
        const cache_bytes = try std.math.add(usize, kp.residentBytes(), weights.sizeofDescriptors());
        const execution_bytes = try std.math.add(usize, workspace_bytes, try std.math.mul(usize, cache_bytes, lanes));
        const rotary_bytes = attention.Pipelines.residentBytes(capacity, c);
        const needed = try std.math.add(usize, plan.bytes, try std.math.add(usize, execution_bytes, rotary_bytes));
        if (o.ram_limit_bytes) |limit| if (needed > limit) {
            std.log.err("the resident payload needs {d} bytes; the load limit is {d}", .{ needed, limit });
            return error.LoadLimitExceeded;
        };
        const load_started = std.Io.Timestamp.now(io, .awake);
        const device = try mtl.Device.init();
        errdefer device.deinit();
        const margin: u64 = 8 << 30;
        const available = device.maxWorkingSet() -| device.allocated() -| margin;
        if (needed > available) {
            std.log.err("full residency needs {d} bytes plus {d} margin; Metal admission allows {d}", .{ needed, margin, available });
            return error.InsufficientResidentMemory;
        }
        const max_buffer = mtl.objc.msg(u64, device.id, "maxBufferLength", .{});
        for (plan.shards.items) |shard| if (shard.end - shard.begin > max_buffer) return error.ShardExceedsMetalBufferLimit;
        std.log.info("full resident admission: weights={d}, total={d}, context={d}, headroom={d}", .{ plan.bytes, needed, capacity, available - needed });
        var pipelines = try Pipelines.init(device, capacity, c);
        errdefer pipelines.deinit();
        const queue = try device.queue();
        errdefer queue.deinit();
        const residency = try device.residencySet(plan.shards.items.len + 128);
        errdefer residency.deinit();
        var cp = try plan.materialize(gpa, device, residency, queue);
        errdefer cp.deinit();
        var bound = try weights.bind(device, &cp);
        errdefer bound.deinit();
        const caches = try gpa.alloc(kv.cache.State, lanes);
        errdefer gpa.free(caches);
        const engram_stash = try gpa.alloc(u8, lanes * 16);
        errdefer gpa.free(engram_stash);
        @memset(engram_stash, 0);
        for (caches) |*state| {
            state.* = try kv.cache.State.init(device, kp);
            errdefer state.deinit();
        }
        var workspace = try work.Workspace.init(device, wp, map.bytes);
        errdefer workspace.deinit();
        var aw = try attention.Workspace.init(device, o.chunk_rows);
        errdefer aw.deinit();
        bound.registerResidency(residency);
        for (caches) |*state| state.registerResidency(residency);
        residency.add(workspace.buffer);
        aw.registerResidency(residency);
        pipelines.attention.registerResidency(residency);
        residency.commit();
        residency.requestResidency();
        queue.addResidencySet(residency);
        return .{ .gpa = gpa, .device = device, .queue = queue, .residency = residency, .checkpoint = cp, .weights = bound, .cfg = c, .cfg_arena = arena, .lanes = lanes, .caches = caches, .cache = &caches[0], .engram_stash = engram_stash, .workspace = workspace, .attention_workspace = aw, .pipelines = pipelines, .admitted_bytes = needed, .load_seconds = @as(f64, @floatFromInt(std.Io.Timestamp.now(io, .awake).toNanoseconds() - load_started.toNanoseconds())) / 1e9 };
    }

    pub fn deinit(m: *Model) void {
        m.queue.removeResidencySet(m.residency);
        m.residency.deinit();
        m.attention_workspace.deinit();
        m.workspace.deinit();
        for (m.caches) |*state| state.deinit();
        m.gpa.free(m.caches);
        m.gpa.free(m.engram_stash);
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

    /// A keep/rollback transaction over one slot's cache cursor and its 16-byte Engram history.
    pub fn beginSlot(m: *Model, slot: u32) !void {
        if (slot >= m.lanes) return error.InvalidLaneSlot;
        const view = try m.workspace.historyView(slot);
        const history: [*]u8 = @ptrCast(@alignCast(view.buffer.contents() + view.offset));
        @memcpy(m.engram_stash[slot * 16 ..][0..16], history[0..16]);
        try m.caches[slot].begin();
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
        @memcpy(history[0..16], m.engram_stash[slot * 16 ..][0..16]);
    }

    /// One command buffer per chunk; no operator waits or reads routes back to the CPU.
    pub fn forward(m: *Model, ids: []const u32, logits: Logits) ![]const f32 {
        return m.forwardOn(0, ids, logits);
    }

    /// A chunk against lane slot `slot`'s caches and Engram cursor: rows land at that slot's positions.
    pub fn forwardOn(m: *Model, slot: u32, ids: []const u32, logits: Logits) ![]const f32 {
        if (!m.valid) return error.InvalidModelState;
        if (slot >= m.lanes) return error.InvalidLaneSlot;
        const state = &m.caches[slot];
        if (ids.len == 0 or ids.len > m.workspace.plan.rows) return error.InvalidPrefillChunk;
        if (state.position > state.plan.capacity or
            ids.len > state.plan.capacity - state.position) return error.ContextFull;
        for (ids) |id| if (id >= 129280) return error.UnsupportedToken;
        m.logits_rows = 0;
        @memcpy(m.workspace.tokenIds(ids.len), ids);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const rows: u32 = @intCast(ids.len);
        const e = m.queue.commandBufferUnretained();
        var enc = e.compute(.serial);
        for (m.checkpoint.shards.items) |shard| enc.useResource(shard.buffer, mtl.ResourceUsage.read);
        var encoding = true;
        var record_started = false;
        errdefer {
            if (m.profile) |recorder| {
                if (record_started) {
                    if (encoding) recorder.cancelEncoding(&enc) catch {} else recorder.discardAfterCompletion() catch {};
                } else if (encoding) enc.end();
            } else if (encoding) enc.end();
        }
        if (m.profile) |recorder| {
            try recorder.beginCommand(e);
            record_started = true;
        }
        try m.profileSpan(&enc, .other);
        const p = &m.pipelines;
        const ws = &m.workspace;
        for (engram.hash.layers, 0..) |meta, index| {
            try engram.encodeHash(enc, &p.engram, meta, ws.view(.ids), ws.view(.token_map), null, try ws.historyView(slot), ws.view(if (index == 0) .engram_ids_0 else .engram_ids_1), rows);
        }
        try engram.encodeCommit(enc, &p.engram, ws.view(.ids), ws.view(.token_map), null, try ws.historyView(slot), rows);
        ops.encodeEmbedding(enc, &p.ops, ws.view(.ids), tensor(m.weights.embed), ws.view(.collapsed), .{ .rows = rows, .width = 5120, .table_stride = 5120, .y_stride = 5120, .copies = 1 });
        enc.barrier();
        mhc.encodeExpand(enc, &p.mhc, ws.view(.collapsed), ws.view(.stream), ws.view(.previous_pre), .{ .rows = rows, .width = 5120, .x_stride = 5120, .stream_stride = 20480 });
        enc.barrier();
        for (&m.weights.layers, 0..) |*layer, index| {
            if (layer.engram) |g| {
                try m.profileSpan(&enc, .engram);
                try engram.encodeGather(enc, &p.engram, .{ .weight = tensor(g.table), .scale = tensor(g.scales), .rows = @intCast(g.table.shape[0]) }, ws.view(if (index == 1) .engram_ids_0 else .engram_ids_1), ws.view(.engram_gather), rows, 6144);
                try m.project(enc, g.wkv, ws.view(.engram_gather), ws.view(.engram_kv), rows, 6144, 25600, .bf16);
                try engram.encodeGate(enc, &p.engram, .{ .q_weight = tensor(g.q_weight), .k_weight = tensor(g.k_weight), .dim = 5120, .copies = 4, .weight_dtype = switch (g.q_weight.dtype) {
                    .bf16 => .bf16,
                    else => .f32,
                }, .hidden_stride = 20480, .kv_stride = 25600, .output_stride = 20480, .eps = m.cfg.eps }, ws.view(.stream), ws.view(.engram_kv), null, ws.view(.stream), rows);
            }
            try m.profileSpan(&enc, .mhc);
            m.mixes(enc, layer.hc_attn, ws.view(.attention_pre), rows);
            mhc.encodeCollapse(enc, &p.mhc, ws.view(.stream), ws.view(.previous_pre), ws.view(.collapsed), .{ .rows = rows, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
            enc.barrier();
            m.normalize(enc, ws.view(.collapsed), layer.attn_norm, ws.view(.normalized), rows);
            try m.profileSpan(&enc, .attention);
            try attention.encode(&p.attention, enc, &layer.attn, state, &m.attention_workspace, ws.view(.normalized), ws.view(.branch), &p.projection, ws.projection(), @intCast(index), rows, m.cfg);
            enc.barrier();
            try m.profileSpan(&enc, .mhc);
            m.post(enc, rows);
            if (m.observer) |observer| observer.encode(observer.ctx, enc, @intCast(index), .attention, m);
            m.mixes(enc, layer.hc_ffn, ws.view(.previous_pre), rows);
            mhc.encodeCollapse(enc, &p.mhc, ws.view(.stream), ws.view(.attention_pre), ws.view(.collapsed), .{ .rows = rows, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
            enc.barrier();
            m.normalize(enc, ws.view(.collapsed), layer.ffn_norm, ws.view(.normalized), rows);
            try m.profileSpan(&enc, .moe);
            try m.moe(enc, &layer.moe, rows);
            try m.profileSpan(&enc, .mhc);
            m.post(enc, rows);
            if (m.observer) |observer| observer.encode(observer.ctx, enc, @intCast(index), .moe, m);
        }
        if (logits != .none) {
            try m.profileSpan(&enc, .head);
            const output_rows: u32 = if (logits == .all) rows else 1;
            const first: usize = if (logits == .all) 0 else rows - 1;
            mhc.encodeCollapse(enc, &p.mhc, ws.view(.stream).shifted(first * 20480 * 2), ws.view(.previous_pre).shifted(first * 4 * 4), ws.view(.collapsed), .{ .rows = output_rows, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
            enc.barrier();
            m.normalize(enc, ws.view(.collapsed), m.weights.norm, ws.view(.normalized), output_rows);
            try m.project(enc, m.weights.head, ws.view(.normalized), ws.view(.logits), output_rows, 5120, 129280, .f32);
        }
        if (m.profile) |recorder| {
            if (recorder.activeSpan()) try recorder.endSpan(&enc);
            try recorder.finishEncoding(&enc);
        } else enc.end();
        encoding = false;
        m.valid = false;
        e.commit();
        e.wait();
        try completed(e);
        if (m.profile) |recorder| _ = try recorder.readAfterCompletion();
        try state.append(rows);
        m.last_gpu_seconds = e.gpuSeconds();
        m.valid = true;
        m.logits_rows = if (logits == .all) rows else if (logits == .last) 1 else 0;
        return ws.logits(m.logits_rows);
    }

    fn profileSpan(m: *Model, encoder: *mtl.ComputeEncoder, stage: gpu_counters.Stage) !void {
        const recorder = m.profile orelse return;
        if (!recorder.enabled()) return;
        if (recorder.activeSpan()) try recorder.endSpan(encoder);
        try recorder.beginSpan(encoder, stage);
        for (m.checkpoint.shards.items) |shard| encoder.useResource(shard.buffer, mtl.ResourceUsage.read);
    }

    fn normalize(m: *Model, enc: mtl.ComputeEncoder, x: View, w: checkpoint.Tensor, y: View, rows: u32) void {
        ops.encodeRms(enc, &m.pipelines.ops, x, tensor(w), y, .{ .rows = rows, .width = 5120, .x_stride = 5120, .y_stride = 5120, .eps = m.cfg.eps });
        enc.barrier();
    }

    fn mixes(m: *Model, enc: mtl.ComputeEncoder, w: weights.Hc, pre: View, rows: u32) void {
        const ws = &m.workspace;
        mhc.encodeRms(enc, &m.pipelines.mhc, ws.view(.stream), ws.view(.inv), .{ .rows = rows, .width = 5120, .stream_stride = 20480, .eps = m.cfg.eps });
        ops.encodeDense(enc, &m.pipelines.ops, ws.view(.stream), tensor(w.projection), ws.view(.mixes), .{ .rows = rows, .k = 20480, .n = 24, .x_stride = 20480, .w_stride = 20480, .y_stride = 24, .w_dtype = .f32, .y_dtype = .f32 });
        enc.barrier();
        mhc.encodeSplit(enc, &m.pipelines.mhc, ws.view(.mixes), ws.view(.inv), tensor(w.scale), tensor(w.base), pre, ws.view(.post), ws.view(.comb), .{ .rows = rows, .eps = m.cfg.hc_eps });
        enc.barrier();
    }

    fn post(m: *Model, enc: mtl.ComputeEncoder, rows: u32) void {
        const ws = &m.workspace;
        mhc.encodePost(enc, &m.pipelines.mhc, ws.view(.branch), ws.view(.stream), ws.view(.post), ws.view(.comb), ws.view(.stream), .{ .rows = rows, .width = 5120, .branch_stride = 5120, .residual_stride = 20480, .y_stride = 20480 });
        enc.barrier();
    }

    fn project(m: *Model, enc: mtl.ComputeEncoder, matrix: exl3.Matrix, x: View, y: View, rows: u32, xs: usize, ys: usize, output: exl3.Output) !void {
        try exl3.encode(&m.pipelines.projection, enc, matrix, convert(x), convert(y), m.workspace.projection(), .{ .count = rows, .x_stride = xs, .y_stride = ys, .output = output });
        enc.barrier();
    }

    fn moe(m: *Model, enc: mtl.ComputeEncoder, w: *const weights.Moe, rows: u32) !void {
        const ws = &m.workspace;
        const p = &m.pipelines;
        ops.encodeDense(enc, &p.ops, ws.view(.normalized), tensor(w.gate), ws.view(.router_logits), .{ .rows = rows, .k = 5120, .n = 384, .x_stride = 5120, .w_stride = 5120, .y_stride = 384, .w_dtype = switch (w.gate.dtype) {
            .bf16 => .bf16,
            else => .f16,
        }, .y_dtype = .f32 });
        enc.barrier();
        route.encode(enc, &p.route, ws.view(.router_logits), tensor(w.bias), ws.view(.route_ids), ws.view(.route_weights), .{ .rows = rows, .bias_dtype = switch (w.bias.dtype) {
            .f32 => .f32,
            .f16 => .f16,
            .bf16 => .bf16,
            else => .f16,
        } });
        enc.barrier();
        const grouped: ?expert_groups.Scratch = if (m.grouped_prefill and rows >= 8) try ws.grouping(rows) else null;
        var routed_rows = exl3.Rows{ .count = @as(usize, rows) * 6, .x_stride = 5120, .y_stride = 2304, .input_rows = rows, .output_rows = @as(usize, rows) * 6, .routes = .{ .ids = convert(ws.view(.route_ids)), .repeat = 6 } };
        if (grouped) |g| {
            try expert_groups.encode(enc, &p.groups, convert(ws.view(.route_ids)), g, 6);
            routed_rows.count = g.plan.padded;
            routed_rows.input_ids = g.view(.token_ids);
            routed_rows.output_ids = g.view(.slot_ids);
            routed_rows.routes = .{ .ids = g.view(.expert_ids) };
            routed_rows.grouped = true;
        }
        for (0..2) |i| {
            try exl3.encodeIndexed(&p.projection, enc, .{ .descriptors = .{ .buffer = w.descriptors[i] }, .count = 384, .k = 5120, .n = 2304 }, convert(ws.view(.normalized)), convert(ws.view(if (i == 0) .gate else .up)), ws.projection(), routed_rows);
            enc.barrier();
        }
        ops.encodeSwiGLU(enc, &p.ops, ws.view(.gate), ws.view(.up), ws.view(.route_weights), ws.view(.activation), .{ .rows = rows * 6, .width = 2304, .input_stride = 2304, .output_stride = 2304, .weighted = true });
        enc.barrier();
        routed_rows.x_stride = 2304;
        routed_rows.y_stride = 5120;
        routed_rows.routes.?.repeat = 1;
        routed_rows.input_rows = @as(usize, rows) * 6;
        if (grouped) |g| routed_rows.input_ids = g.view(.slot_ids);
        try exl3.encodeIndexed(&p.projection, enc, .{ .descriptors = .{ .buffer = w.descriptors[2] }, .count = 384, .k = 2304, .n = 5120 }, convert(ws.view(.activation)), convert(ws.view(.routed)), ws.projection(), routed_rows);
        enc.barrier();
        try m.project(enc, w.shared[0], ws.view(.normalized), ws.view(.shared_gate), rows, 5120, 2304, .bf16);
        try m.project(enc, w.shared[1], ws.view(.normalized), ws.view(.shared_up), rows, 5120, 2304, .bf16);
        ops.encodeSwiGLU(enc, &p.ops, ws.view(.shared_gate), ws.view(.shared_up), ws.view(.route_weights), ws.view(.shared_activation), .{ .rows = rows, .width = 2304, .input_stride = 2304, .output_stride = 2304 });
        enc.barrier();
        try m.project(enc, w.shared[2], ws.view(.shared_activation), ws.view(.shared_down), rows, 2304, 5120, .bf16);
        ops.encodeCombine(enc, &p.ops, ws.view(.routed), ws.view(.shared_down), ws.view(.route_ids), ws.view(.branch), .{ .rows = rows, .width = 5120, .routed_row_stride = 6 * 5120, .slot_stride = 5120, .shared_stride = 5120, .y_stride = 5120 });
        enc.barrier();
    }
};

fn tensor(t: checkpoint.Tensor) View {
    return .{ .buffer = t.buffer, .offset = t.offset };
}
fn convert(v: View) exl3.View {
    return .{ .buffer = v.buffer, .offset = v.offset };
}
fn completed(cb: mtl.CommandBuffer) !void {
    if (cb.failure()) |message| {
        std.log.err("DeepSeek V4.1 GPU execution failed: {s}", .{message});
        return error.GpuExecutionFailed;
    }
}
