//! Per-layer logs outlive verification; a single layer scratch slab is reused by caller-ordered encoders.
const std = @import("std");
const metal = @import("metal");
const abi = @import("gdn_contract.zig");
const gdn = @import("gdn.zig");
const Plan = @import("gdn_plan.zig").Plan;
const Budget = @import("gdn_pool.zig").Budget;
const Ref = @import("projection.zig").Ref;
const Tensor = @import("gpu_weights.zig").Tensor;
const profile = @import("core").gpu_profile;
const Kernels = @import("gdn_library.zig").Kernels;

const Field = enum { state, next_state, history, next_history, qkv, query, key, value, decay, mixing, snapshots, tails, recurrent };

pub const Metadata = struct {
    parents: Ref,
    windows: Ref,
    row_slots: Ref,
    segments: Ref,
    keeps: Ref,
    kept_rows: Ref,

    pub fn upload(m: Metadata, plan: Plan) !void {
        inline for (.{ "parents", "windows", "row_slots", "segments", "keeps", "kept_rows" }) |name| {
            const r = @field(m, name);
            const bytes = std.mem.sliceAsBytes(@field(plan, name));
            if (r.offset > r.buffer.length() or bytes.len > r.buffer.length() - r.offset) return error.ShortGdnMetadata;
        }
        inline for (.{ "parents", "windows", "row_slots", "segments", "keeps", "kept_rows" }) |name| {
            const r = @field(m, name);
            const bytes = std.mem.sliceAsBytes(@field(plan, name));
            @memcpy(r.buffer.contents()[r.offset..][0..bytes.len], bytes);
        }
    }
};

pub const Scalars = struct {
    conv: gdn.Span,
    a_log: gdn.Span,
    dt: gdn.Span,
    norm: gdn.Span,
    kinds: abi.WeightKinds,

    pub fn from(g: @import("gpu_weights.zig").Gdn) !Scalars {
        return .{ .conv = tensorSpan(g.conv), .a_log = tensorSpan(g.a_log), .dt = tensorSpan(g.dt), .norm = tensorSpan(g.norm), .kinds = .{ .conv = try abi.storage(g.conv.dtype), .a_log = try abi.storage(g.a_log.dtype), .dt = try abi.storage(g.dt.dtype), .norm = try abi.storage(g.norm.dtype) } };
    }
};

pub const DevicePlan = struct { host: Plan, metadata: Metadata };
/// z, a and b in their own rows, or (zba_stride > 0) views into one fused zba row.
pub const Refs = struct { z: Ref, a: Ref, b: Ref, output: Ref, scalars: Scalars, zba_stride: u32 = 0 };

pub const Pool = struct {
    budget: Budget,
    buffers: [std.enums.values(Field).len]metal.Buffer,
    kernels: Kernels,
    profiler: ?*profile.Trace = null,

    pub fn init(device: metal.Device, shape: abi.Shape, layers: usize, slots: usize, capacity: usize) !Pool {
        const budget = try Budget.init(shape, layers, slots, capacity);
        if (capacity % 16 != 0) return error.BadGdnProjectionCapacity;
        const widths = [_]usize{ budget.state, budget.state, budget.history, budget.history, budget.projections, budget.query, budget.key, budget.value, budget.decay, budget.mixing, budget.snapshots, budget.conv_tails, capacity * shape.value * 2 };
        var buffers: [widths.len]metal.Buffer = undefined;
        var count: usize = 0;
        errdefer for (buffers[0..count]) |buffer| buffer.deinit();
        for (widths, &buffers) |bytes, *buffer| {
            buffer.* = try device.buffer(@max(bytes, 16), metal.ResourceOptions.shared | metal.ResourceOptions.untracked);
            count += 1;
        }
        for ([_]Field{ .state, .history }) |field| @memset(buffers[@backingInt(field)].contents()[0..widths[@backingInt(field)]], 0);
        const kernels = try Kernels.init(device);
        return .{ .budget = budget, .buffers = buffers, .kernels = kernels };
    }

    pub fn deinit(pool: *Pool) void {
        pool.kernels.deinit();
        for (pool.buffers) |buffer| buffer.deinit();
    }

    /// Borrow committed bytes only after all encoded work has completed.
    pub fn committed(pool: *const Pool, layer: usize, slot: usize, recurrent: bool) ![]const u8 {
        const span = try pool.committedSpan(layer, slot, recurrent);
        return span.buffer.contents()[span.offset..][0..span.bytes];
    }

    /// Where a slot's committed state (recurrent) or conv history lives, for copies on the GPU.
    pub fn committedSpan(pool: *const Pool, layer: usize, slot: usize, recurrent: bool) !struct { buffer: metal.Buffer, offset: usize, bytes: usize } {
        const off = if (recurrent) try pool.budget.stateOffset(layer, slot) else try pool.budget.historyOffset(layer, slot);
        const bytes = if (recurrent) pool.budget.shape.state * 4 else pool.budget.shape.conv * 2;
        return .{ .buffer = pool.buffers[@backingInt(if (recurrent) Field.state else Field.history)], .offset = off, .bytes = bytes };
    }

    pub fn projected(pool: *const Pool, layer: usize) !Ref {
        const span = try pool.layerSpan(.qkv, layer);
        return .{ .buffer = span.buffer, .offset = span.offset };
    }

    pub fn forward(pool: *Pool, layer: usize, plan: DevicePlan, refs: Refs, e: metal.ComputeEncoder) !void {
        try @import("gdn_pool.zig").validatePlan(pool.budget.shape, pool.budget.slots, pool.budget.rows, plan.host);
        const prepared = try pool.round(layer, plan, refs);
        try plan.metadata.upload(plan.host);
        var pre = profile.begin(pool.profiler, e, .gdn_prepare);
        prepared.prepare(pre.encoder);
        try pre.end();
        var scan = profile.begin(pool.profiler, e, .gdn_scan);
        prepared.forward(scan.encoder);
        try scan.end();
    }

    /// Publish a fully kept chain after its per-position update; prompt chains take the wide scan.
    pub fn forwardChain(pool: *Pool, layer: usize, plan: DevicePlan, refs: Refs, input_encoder: metal.ComputeEncoder, prompt: bool) !void {
        try @import("gdn_pool.zig").validatePlan(pool.budget.shape, pool.budget.slots, pool.budget.rows, plan.host);
        for (plan.host.segments, plan.host.keeps) |segment, keep| {
            if (keep.rows != segment.rows) return error.NotAcceptedChain;
            for (0..segment.rows) |row| if (plan.host.parents[segment.first + row] != (if (row == 0) @as(i32, -1) else @as(i32, @intCast(row - 1)))) return error.NotAcceptedChain;
        }
        const prepared = try pool.round(layer, plan, refs);
        try plan.metadata.upload(plan.host);
        var pre = profile.begin(pool.profiler, input_encoder, .gdn_prepare);
        errdefer pre.cancel();
        prepared.prepare(pre.encoder);
        try pre.end();
        var scan = profile.begin(pool.profiler, input_encoder, .gdn_scan);
        errdefer scan.cancel();
        const wide = prompt and pool.budget.shape.c.dk == 128 and pool.budget.shape.c.dv % 4 == 0;
        prepared.scanChain(scan.encoder, if (wide) pool.kernels.chain_wide else pool.kernels.chain, wide);
        try scan.end();
        var gate = profile.begin(pool.profiler, input_encoder, .gdn_gate);
        errdefer gate.cancel();
        prepared.gate(gate.encoder);
        try gate.end();
        var publication = profile.begin(pool.profiler, input_encoder, .state_publish);
        errdefer publication.cancel();
        const e = publication.encoder;
        prepared.commitConv(e);
        const p = prepared.p;
        e.setPipeline(pool.kernels.publish);
        gdn.bind(e, .{ prepared.b.next_state, prepared.b.next_history }, 0);
        e.setBuffer(plan.metadata.keeps.buffer, plan.metadata.keeps.offset, 2);
        e.setValue(p, 3);
        gdn.bind(e, .{ prepared.b.state, prepared.b.history }, 4);
        e.dispatchThreads(.{ .width = pool.budget.shape.state, .height = plan.host.keeps.len }, .{ .width = 32 });
        e.barrier();
        try publication.end();
    }

    pub fn commit(pool: *Pool, layer: usize, plan: DevicePlan, e: metal.ComputeEncoder) !void {
        try @import("gdn_pool.zig").validatePlan(pool.budget.shape, pool.budget.slots, pool.budget.rows, plan.host);
        try plan.metadata.upload(plan.host);
        return pool.commitDevice(layer, plan, e);
    }

    /// Caller-generated keep descriptors remain on the GPU while the admitted host plan fixes dispatch bounds.
    pub fn commitDevice(pool: *Pool, layer: usize, plan: DevicePlan, e: metal.ComputeEncoder) !void {
        try pool.commitScan(layer, plan, e);
        e.barrier();
        try pool.commitPublish(layer, plan, e);
        e.barrier();
    }

    /// A commit's replay and conv window, unfenced; the replay writes in place when every keep stays in its slot.
    pub fn commitScan(pool: *Pool, layer: usize, plan: DevicePlan, e: metal.ComputeEncoder) !void {
        try @import("gdn_pool.zig").validatePlan(pool.budget.shape, pool.budget.slots, pool.budget.rows, plan.host);
        const kinds = abi.WeightKinds{ .conv = .bf16, .a_log = .bf16, .dt = .bf16, .norm = .bf16 };
        const p = try pool.budget.shape.params(plan.host.parents.len, pool.budget.slots, kinds);
        const state = try pool.layerSpan(.state, layer);
        const history = try pool.layerSpan(.history, layer);
        const next_history = try pool.layerSpan(.next_history, layer);
        const m = plan.metadata;
        e.setPipeline(pool.kernels.pipelines.replay);
        gdn.bind(e, .{ try pool.layerSpan(.key, layer), try pool.layerSpan(.value, layer), try pool.layerSpan(.decay, layer), try pool.layerSpan(.mixing, layer), state }, 0);
        e.setBuffer(m.kept_rows.buffer, m.kept_rows.offset, 5);
        e.setBuffer(m.keeps.buffer, m.keeps.offset, 6);
        e.setValue(p, 7);
        const out = if (inPlace(plan.host)) state else try pool.layerSpan(.next_state, layer);
        e.setBuffer(out.buffer, out.offset, 8);
        e.dispatchThreads(.{ .width = 32, .height = p.dv, .depth = p.nv * plan.host.keeps.len }, .{ .width = 32, .height = 4 });
        e.setPipeline(pool.kernels.pipelines.conv_commit);
        gdn.bind(e, .{ history, try pool.layerSpan(.qkv, layer) }, 0);
        e.setBuffer(m.kept_rows.buffer, m.kept_rows.offset, 2);
        e.setBuffer(m.keeps.buffer, m.keeps.offset, 3);
        e.setBuffer(m.windows.buffer, m.windows.offset, 4);
        e.setValue(p, 5);
        e.setBuffer(next_history.buffer, next_history.offset, 6);
        e.dispatchThreads(.{ .width = pool.budget.shape.conv, .height = plan.host.keeps.len }, .{ .width = 32 });
    }

    /// After commitScan's fence: the new conv window (and the state, when it went to next_state) into place, unfenced.
    pub fn commitPublish(pool: *Pool, layer: usize, plan: DevicePlan, e: metal.ComputeEncoder) !void {
        const kinds = abi.WeightKinds{ .conv = .bf16, .a_log = .bf16, .dt = .bf16, .norm = .bf16 };
        const p = try pool.budget.shape.params(plan.host.parents.len, pool.budget.slots, kinds);
        const state = try pool.layerSpan(.state, layer);
        const history = try pool.layerSpan(.history, layer);
        const next_history = try pool.layerSpan(.next_history, layer);
        const m = plan.metadata;
        if (inPlace(plan.host)) {
            e.setPipeline(pool.kernels.publish_conv);
            gdn.bind(e, .{next_history}, 0);
            e.setBuffer(m.keeps.buffer, m.keeps.offset, 1);
            e.setValue(p, 2);
            e.setBuffer(history.buffer, history.offset, 3);
            e.dispatchThreads(.{ .width = pool.budget.shape.conv, .height = plan.host.keeps.len }, .{ .width = 32 });
            return;
        }
        e.setPipeline(pool.kernels.publish);
        gdn.bind(e, .{ try pool.layerSpan(.next_state, layer), next_history }, 0);
        e.setBuffer(m.keeps.buffer, m.keeps.offset, 2);
        e.setValue(p, 3);
        e.setBuffer(state.buffer, state.offset, 4);
        e.setBuffer(history.buffer, history.offset, 5);
        e.dispatchThreads(.{ .width = pool.budget.shape.state, .height = plan.host.keeps.len }, .{ .width = 32 });
    }

    pub fn clearSlot(pool: *Pool, slot: u32, e: metal.ComputeEncoder) !void {
        if (slot >= pool.budget.slots) return error.BadGdnPoolIndex;
        const k = abi.WeightKinds{ .conv = .bf16, .a_log = .bf16, .dt = .bf16, .norm = .bf16 };
        const p = try pool.budget.shape.params(1, pool.budget.slots, k);
        e.setPipeline(pool.kernels.clear);
        e.setBuffer(pool.get(.state), 0, 0);
        e.setBuffer(pool.get(.history), 0, 1);
        e.setValue(p, 2);
        e.setValue(slot, 3);
        e.dispatchThreads(.{ .width = pool.budget.shape.state, .height = pool.budget.layers }, .{ .width = 32 });
        e.barrier();
    }

    fn round(pool: *Pool, layer: usize, device_plan: DevicePlan, refs: Refs) !gdn.Round {
        const plan = device_plan.host;
        if (plan.parents.len > pool.budget.rows) return error.BadGdnPoolIndex;
        var p = try pool.budget.shape.params(plan.parents.len, pool.budget.slots, refs.scalars.kinds);
        p.zba_stride = refs.zba_stride;
        const bytes = try abi.Bytes.init(pool.budget.shape, p, refs.scalars.kinds);
        const m = device_plan.metadata;
        const zs: usize = refs.zba_stride;
        const heads = pool.budget.shape.c.v_heads;
        const b = gdn.Buffers{
            .qkv = try pool.layerSpan(.qkv, layer),
            .a = try refSpan(refs.a, if (zs == 0) bytes.heads else ((p.rows - 1) * zs + heads) * 2),
            .b = try refSpan(refs.b, if (zs == 0) bytes.heads else ((p.rows - 1) * zs + heads) * 2),
            .z = try refSpan(refs.z, if (zs == 0) bytes.value else ((p.rows - 1) * zs + pool.budget.shape.value) * 2),
            .conv_weight = refs.scalars.conv,
            .a_log = refs.scalars.a_log,
            .dt = refs.scalars.dt,
            .norm = refs.scalars.norm,
            .history = try pool.layerSpan(.history, layer),
            .state = try pool.layerSpan(.state, layer),
            .query = try pool.layerSpan(.query, layer),
            .key = try pool.layerSpan(.key, layer),
            .value = try pool.layerSpan(.value, layer),
            .decay = try pool.layerSpan(.decay, layer),
            .mixing = try pool.layerSpan(.mixing, layer),
            .tails = try refSpan(.{ .buffer = pool.get(.tails) }, bytes.tails),
            .snapshots = try refSpan(.{ .buffer = pool.get(.snapshots) }, bytes.snapshots),
            .recurrent = try refSpan(.{ .buffer = pool.get(.recurrent) }, bytes.value),
            .output = try refSpan(refs.output, bytes.value),
            .next_state = try pool.layerSpan(.next_state, layer),
            .next_history = try pool.layerSpan(.next_history, layer),
            .parents = try refSpan(m.parents, plan.parents.len * 4),
            .windows = try refSpan(m.windows, plan.windows.len * 4),
            .row_slots = try refSpan(m.row_slots, plan.row_slots.len * 4),
            .segments = try refSpan(m.segments, plan.segments.len * @sizeOf(abi.Segment)),
            .keeps = try refSpan(m.keeps, plan.keeps.len * @sizeOf(abi.Keep)),
            .kept_rows = try refSpan(m.kept_rows, plan.kept_rows.len * 4),
        };
        return gdn.Round.init(pool.budget.shape, refs.scalars.kinds, p, plan, b, pool.kernels.pipelines);
    }

    fn get(pool: *const Pool, field: Field) metal.Buffer {
        return pool.buffers[@backingInt(field)];
    }

    fn layerSpan(pool: *const Pool, field: Field, layer: usize) !gdn.Span {
        if (layer >= pool.budget.layers) return error.BadGdnPoolIndex;
        const buffer = pool.get(field);
        const bytes = buffer.length() / pool.budget.layers;
        return .{ .buffer = buffer, .offset = layer * bytes, .bytes = bytes };
    }
};

fn tensorSpan(t: Tensor) gdn.Span {
    return .{ .buffer = t.buffer, .offset = t.offset, .bytes = t.bytes };
}

fn inPlace(plan: Plan) bool {
    for (plan.keeps) |keep| if (keep.state_slot != keep.next_slot) return false;
    return true;
}

fn refSpan(r: Ref, bytes: usize) !gdn.Span {
    if (r.offset > r.buffer.length() or bytes > r.buffer.length() - r.offset) return error.BadGdnBuffer;
    return .{ .buffer = r.buffer, .offset = r.offset, .bytes = bytes };
}
