//! Inkling 2 Mini DSpark: committed target features, three-layer proposal blocks, folded Markov logits.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const gpu_rule = lanes.gpu_rule;
const ck = @import("../../../core/checkpoint_metal.zig");
const Model = @import("model.zig").Model;
const dense_units = @import("dense_units.zig");
const fwd = @import("forward.zig");
const glue = @import("glue.zig");
const k = @import("dspark_kernels.zig");
const weight_loader = @import("dspark_weights.zig");
const state = @import("dspark_state.zig");

const dr = lanes.drafter;
const Stream = lanes.Stream;
const Ref = k.Ref;
const tap_list = [_]u32{ 5, 23, 41, 48 };
/// Drafts a hold offers at most: the block after its anchor. The depth rule prices how many a round takes.
const hold_depth: u32 = state.block - 1;
/// Streams one greedy hold can chain on the GPU; a wider hold samples on the host as sampled streams do.
const chain_streams = 16;
const q4_off_q_c: usize = 0;
const q4_off_k_c: usize = q4_off_q_c + state.query * (state.hidden / 2);
const q4_off_v_c: usize = q4_off_k_c + state.kv * (state.hidden / 2);
const q4_off_o_c: usize = q4_off_v_c + state.kv * (state.hidden / 2);
const q4_off_gate_c: usize = q4_off_o_c + state.hidden * (state.query / 2);
const q4_off_up_c: usize = q4_off_gate_c + state.intermediate * (state.hidden / 2);
const q4_off_down_c: usize = q4_off_up_c + state.intermediate * (state.hidden / 2);
const q4_layer_codes: usize = q4_off_down_c + state.hidden * (state.intermediate / 2);
const q4_off_q_p: usize = 0;
const q4_off_k_p: usize = q4_off_q_p + state.query * (state.hidden / 16);
const q4_off_v_p: usize = q4_off_k_p + state.kv * (state.hidden / 16);
const q4_off_o_p: usize = q4_off_v_p + state.kv * (state.hidden / 16);
const q4_off_gate_p: usize = q4_off_o_p + state.hidden * (state.query / 16);
const q4_off_up_p: usize = q4_off_gate_p + state.intermediate * (state.hidden / 16);
const q4_off_down_p: usize = q4_off_up_p + state.intermediate * (state.hidden / 16);
const q4_layer_params: usize = q4_off_down_p + state.hidden * (state.intermediate / 16);

fn at(buf: mtl.Buffer, off: usize) Ref {
    return .{ .buf = buf, .off = off };
}
fn r(t: ck.Tensor) Ref {
    return .{ .buf = t.buffer, .off = t.offset };
}
fn dense2(m: *const Model, e: mtl.ComputeEncoder, w0: ck.Tensor, w1: ck.Tensor, input: Ref, y0: Ref, y1: Ref, rows: usize) void {
    const k_dim = w0.shape[w0.rank - 1];
    if (m.dense_units) |*du| if (k_dim % 128 == 0) {
        const b = r(w0);
        dense_units.encode6(du, e, .{ .xq = input, .xk = input, .xv = input, .wq = b, .wk = r(w1), .wv = b, .wr = b, .ws = b, .wg = b, .yq = y0, .yk = y1, .yv = y0, .yr = y0, .ys = y0, .yg = y0 }, .{ .rows = @intCast(rows), .k = @intCast(k_dim), .nq = @intCast(w0.shape[0]), .nk = @intCast(w1.shape[0]), .nv = 0, .nr = 0, .ns = 0, .ng = 0 }) catch unreachable;
        return;
    };
    fwd.dense(m, e, w0, input, y0, rows);
    fwd.dense(m, e, w1, input, y1, rows);
}
fn dense3(m: *const Model, e: mtl.ComputeEncoder, w0: ck.Tensor, w1: ck.Tensor, w2: ck.Tensor, input: Ref, y0: Ref, y1: Ref, y2: Ref, rows: usize) void {
    const k_dim = w0.shape[w0.rank - 1];
    if (m.dense_units) |*du| if (k_dim % 128 == 0) {
        const b = r(w0);
        dense_units.encode6(du, e, .{ .xq = input, .xk = input, .xv = input, .wq = b, .wk = r(w1), .wv = r(w2), .wr = b, .ws = b, .wg = b, .yq = y0, .yk = y1, .yv = y2, .yr = y0, .ys = y0, .yg = y0 }, .{ .rows = @intCast(rows), .k = @intCast(k_dim), .nq = @intCast(w0.shape[0]), .nk = @intCast(w1.shape[0]), .nv = @intCast(w2.shape[0]), .nr = 0, .ns = 0, .ng = 0 }) catch unreachable;
        return;
    };
    fwd.dense(m, e, w0, input, y0, rows);
    fwd.dense(m, e, w1, input, y1, rows);
    fwd.dense(m, e, w2, input, y2, rows);
}
fn failed(cb: mtl.CommandBuffer, what: []const u8) !void {
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("inkling2_loop DSpark {s}: {s}", .{ what, msg });
        return error.GpuFailed;
    }
}

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    target: *const Model,
    weights: weight_loader.Loaded,
    pipelines: k.Pipelines,
    scratch: state.Scratch,
    /// Scratch for the second and later streams of one hold; the first uses `scratch`, as absorb does.
    more: std.ArrayList(state.Scratch) = .empty,
    /// The greedy chain's kernels, and each held stream's eight tokens: the anchor, then its drafts.
    chain: k.Chain = undefined,
    chain_tokens: mtl.Buffer = undefined,
    /// 2-bit screened greedy head pipelines and pre-packed unembed/phi buffers.
    screen: k.Screen = undefined,
    u_codes: mtl.Buffer = undefined,
    u_params: mtl.Buffer = undefined,
    p_codes: mtl.Buffer = undefined,
    p_params: mtl.Buffer = undefined,
    q4_codes: ?mtl.Buffer = null,
    q4_params: ?mtl.Buffer = null,
    cand_ids: mtl.Buffer = undefined,
    cand_out_ids: mtl.Buffer = undefined,
    cand_logits: mtl.Buffer = undefined,
    scores: ?mtl.Buffer = null,
    capacity: usize = 1,
    /// A draft's share of a full hold, timed by `measure`.
    step_ms: f64 = 0,
    contexts: std.AutoHashMapUnmanaged(*Stream, state.Lane) = .empty,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, target: *const Model, drafter_dir: []const u8) !*Drafter {
        _ = io; // Native factory convention; Checkpoint.addFile uses platform file calls.
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = .{
            .gpa = gpa,
            .target = target,
            .weights = weight_loader.load(gpa, target, drafter_dir) catch |err| return err,
            .pipelines = undefined,
            .scratch = undefined,
        };
        errdefer d.weights.deinit();
        d.pipelines = try k.Pipelines.load(device);
        errdefer d.pipelines.deinit();
        d.scratch = try state.Scratch.init(device);
        errdefer d.scratch.deinit();
        d.chain = try k.Chain.load(device, @import("inkling2_metal").inkling2_dspark);
        errdefer d.chain.deinit();
        d.chain_tokens = try device.buffer(chain_streams * 8 * @sizeOf(u32), mtl.ResourceOptions.shared);
        errdefer d.chain_tokens.deinit();
        d.screen = try k.Screen.load(device, @import("inkling2_metal").inkling2_dspark);
        errdefer d.screen.deinit();
        d.u_codes = try device.buffer(state.vocabulary * (state.hidden / 16) * @sizeOf(u32), mtl.ResourceOptions.private);
        errdefer d.u_codes.deinit();
        d.u_params = try device.buffer(state.vocabulary * (state.hidden / 64) * 4, mtl.ResourceOptions.private);
        errdefer d.u_params.deinit();
        d.p_codes = try device.buffer(state.vocabulary * (state.rank / 16) * @sizeOf(u32), mtl.ResourceOptions.private);
        errdefer d.p_codes.deinit();
        d.p_params = try device.buffer(state.vocabulary * (state.rank / 64) * 4, mtl.ResourceOptions.private);
        errdefer d.p_params.deinit();
        d.cand_ids = try device.buffer(256 * @sizeOf(u32), mtl.ResourceOptions.private);
        errdefer d.cand_ids.deinit();
        d.cand_out_ids = try device.buffer(257 * @sizeOf(u32), mtl.ResourceOptions.private);
        errdefer d.cand_out_ids.deinit();
        d.cand_logits = try device.buffer(257 * 2, mtl.ResourceOptions.private);
        errdefer d.cand_logits.deinit();
        if (d.screen.q4_units != null) {
            d.q4_codes = try device.buffer(3 * q4_layer_codes, mtl.ResourceOptions.private);
            errdefer if (d.q4_codes) |b| b.deinit();
            d.q4_params = try device.buffer(3 * q4_layer_params, mtl.ResourceOptions.private);
            errdefer if (d.q4_params) |b| b.deinit();
        }
        {
            const cb = target.queue.commandBuffer();
            const e = cb.compute(.serial);
            k.q2Pack(&d.screen, e, r(target.weights.unembed), at(d.u_codes, 0), at(d.u_params, 0), state.vocabulary, state.hidden / 64);
            k.q2Pack(&d.screen, e, at(d.weights.phi, 0), at(d.p_codes, 0), at(d.p_params, 0), state.vocabulary, state.rank / 64);
            if (d.q4_codes) |qc| {
                const qp = d.q4_params.?;
                for (d.weights.weights.layers, 0..) |layer, i| {
                    const cb0 = i * q4_layer_codes;
                    const pb0 = i * q4_layer_params;
                    k.q4Pack(&d.screen, e, r(layer.q), at(qc, cb0 + q4_off_q_c), at(qp, pb0 + q4_off_q_p), state.query, state.hidden / 64);
                    k.q4Pack(&d.screen, e, r(layer.k), at(qc, cb0 + q4_off_k_c), at(qp, pb0 + q4_off_k_p), state.kv, state.hidden / 64);
                    k.q4Pack(&d.screen, e, r(layer.v), at(qc, cb0 + q4_off_v_c), at(qp, pb0 + q4_off_v_p), state.kv, state.hidden / 64);
                    k.q4Pack(&d.screen, e, r(layer.o), at(qc, cb0 + q4_off_o_c), at(qp, pb0 + q4_off_o_p), state.hidden, state.query / 64);
                    k.q4Pack(&d.screen, e, r(layer.gate), at(qc, cb0 + q4_off_gate_c), at(qp, pb0 + q4_off_gate_p), state.intermediate, state.hidden / 64);
                    k.q4Pack(&d.screen, e, r(layer.up), at(qc, cb0 + q4_off_up_c), at(qp, pb0 + q4_off_up_p), state.intermediate, state.hidden / 64);
                    k.q4Pack(&d.screen, e, r(layer.down), at(qc, cb0 + q4_off_down_c), at(qp, pb0 + q4_off_down_p), state.hidden, state.intermediate / 64);
                }
            }
            e.end();
            cb.commit();
            try failed(cb, "q2/q4 pack");
        }
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        var it = d.contexts.valueIterator();
        while (it.next()) |lane| lane.deinit();
        d.contexts.deinit(d.gpa);
        if (d.scores) |scores| scores.deinit();
        if (d.q4_params) |b| b.deinit();
        if (d.q4_codes) |b| b.deinit();
        d.cand_logits.deinit();
        d.cand_out_ids.deinit();
        d.cand_ids.deinit();
        d.p_params.deinit();
        d.p_codes.deinit();
        d.u_params.deinit();
        d.u_codes.deinit();
        d.screen.deinit();
        d.chain_tokens.deinit();
        d.chain.deinit();
        for (d.more.items) |*extra| extra.deinit();
        d.more.deinit(d.gpa);
        d.scratch.deinit();
        d.pipelines.deinit();
        d.weights.deinit();
        d.gpa.destroy(d);
    }

    pub fn fixedBytes(d: *const Drafter) u64 {
        const q4b: u64 = if (d.q4_codes) |b| b.length() + d.q4_params.?.length() else 0;
        return d.weights.fixedBytes() + state.scratchBytes() + d.u_codes.length() + d.u_params.length() + d.p_codes.length() + d.p_params.length() + q4b;
    }

    pub fn scoreBytes(capacity: usize) u64 {
        return state.scoreBytes(capacity);
    }

    pub fn streamBytes(capacity: usize) u64 {
        return state.laneBytes(capacity);
    }

    /// The selected target window fixes every new lane's global ring and the shared attention score allocation.
    pub fn setContext(d: *Drafter, window: u32) !void {
        const capacity: usize = @intCast(window);
        if (capacity == 0 or capacity > 1048576) return error.DSparkLaneGeometry;
        if (d.contexts.count() != 0) return error.DSparkContextActive;
        const scores = try d.target.device.buffer(@intCast(state.scoreBytes(capacity)), mtl.ResourceOptions.private);
        if (d.scores) |old| old.deinit();
        d.scores = scores;
        d.capacity = capacity;
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    pub fn saveSnap(d: *Drafter, stream: *Stream, at_pos: u32, last_tap: ?mtl.Buffer, last_tap_start: u64, last_tap_rows: usize) !?state.Snap {
        const lane = d.contexts.getPtr(stream) orelse return null;
        if (lane.end == at_pos) return try state.Snap.save(d.target.device, d.target.queue, lane, null);
        if (lane.end + 1 == at_pos) {
            const tap = last_tap orelse return error.SnapshotOutOfStep;
            if (last_tap_rows != 1 or last_tap_start != lane.end) return error.SnapshotOutOfStep;
            return try state.Snap.save(d.target.device, d.target.queue, lane, tap);
        }
        return error.SnapshotOutOfStep;
    }

    pub fn restoreSnap(d: *Drafter, stream: *Stream, snap: ?*const state.Snap, at_pos: usize) !void {
        const lane = d.contexts.getPtr(stream) orelse return;
        const s = snap orelse return error.DSparkSnapshotMissing;
        const expected_end = s.end + @as(u64, if (s.tail_off != null) 1 else 0);
        if (expected_end != at_pos) return error.DSparkSnapshotGeometry;
        try state.Snap.restore(lane, d.target.queue, s);
        if (s.tail_off) |tail_off| {
            errdefer {
                lane.from = 0;
                lane.end = 0;
                lane.held_len = 0;
            }
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            d.absorbRows(e, lane, at(s.buf, tail_off), s.end, 1);
            e.end();
            cb.commit();
            try failed(cb, "restore tail");
            lane.end = s.end + 1;
            lane.held_len = 0;
        }
    }

    pub fn resetLane(d: *Drafter, stream: *Stream) void {
        if (d.contexts.getPtr(stream)) |lane| {
            lane.from = 0;
            lane.end = 0;
            lane.held_len = 0;
        }
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(_: *anyopaque) []const u32 {
        return &tap_list;
    }

    fn factsFn(ptr: *anyopaque) dr.Facts {
        const d = cast(ptr);
        // No plain rounds: one draft pays from about a third landing, and the guard's estimate near that line is noise.
        return .{ .depth = hold_depth, .step_ms = d.step_ms, .plain_guard = false, .batched = true, .stream_bytes = @intCast(state.laneBytes(d.capacity)) };
    }

    fn openFn(ptr: *anyopaque, stream: *Stream) anyerror!void {
        const d = cast(ptr);
        if (d.contexts.contains(stream)) return error.DraftStreamOpen;
        const lane = try state.Lane.init(d.target.device, d.capacity);
        errdefer {
            var owned = lane;
            owned.deinit();
        }
        try d.contexts.put(d.gpa, stream, lane);
    }

    fn releaseFn(ptr: *anyopaque, stream: *Stream) void {
        const d = cast(ptr);
        if (d.contexts.fetchRemove(stream)) |entry| {
            var lane = entry.value;
            lane.deinit();
        }
    }

    fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const d = cast(ptr);
        for (streams, out) |stream, tokens| {
            const lane = d.contexts.getPtr(stream) orelse return error.DraftStreamUnknown;
            if (tokens.len > lane.held_len) return error.DraftHeldFewer;
            @memcpy(tokens, lane.held[0..tokens.len]);
        }
    }

    fn absorbRows(d: *Drafter, e: mtl.ComputeEncoder, lane: *const state.Lane, taps: Ref, start: u64, rows: usize) void {
        const n: u32 = @intCast(rows);
        k.fuseTaps(&d.pipelines, e, taps, r(d.weights.weights.fuse), at(d.weights.dflare, 0), at(d.scratch.fused, 0), at(lane.newest, 0), n);
        for (d.weights.weights.layers, 0..) |layer, i| {
            const c = at(d.scratch.fused, i * rows * state.hidden * 2);
            dense2(d.target, e, layer.k_ctx, layer.v_ctx, c, at(d.scratch.ck, 0), at(d.scratch.cv, 0), rows);
            k.commitContext(&d.pipelines, e, at(d.scratch.ck, 0), at(d.scratch.cv, 0), r(layer.k_norm_ctx), at(lane.k[i], 0), at(lane.v[i], 0), @intCast(start), n, @intCast(lane.rings[i]));
        }
    }

    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        if (items.len == 0) return;
        const cb = d.target.queue.commandBuffer();
        const e = cb.compute(.serial);
        const row_bytes = 4 * state.hidden * 2;
        for (items) |item| {
            const lane = d.contexts.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            const features = item.features;
            if (features.space != .device or features.dtype != .bf16 or features.row_bytes != row_bytes) return error.DraftFeaturesUnsupported;
            if (item.rows != null) return error.TreesNotBuilt;
            if (features.rows == 0 or features.rows > 1048576) return error.DraftFeaturesUnsupported;
            if (item.start < lane.end) return error.DraftContextGap;
            if (item.start > lane.end) lane.from = item.start;
            const end = item.start + features.rows;
            if (end > 1048576) return error.DraftContextGap;
            const source = mtl.Buffer{ .id = @ptrFromInt(features.buffer) };
            var position = item.start;
            while (position < end) {
                const rows: usize = @intCast(@min(state.absorb_rows, end - position));
                const offset = features.offset + @as(usize, @intCast(position - item.start)) * row_bytes;
                d.absorbRows(e, lane, at(source, offset), position, rows);
                position += rows;
            }
            lane.end = end;
            lane.held_len = 0;
        }
        e.end();
        cb.commit();
        try failed(cb, "absorb");
    }

    fn encodeBlockCore(d: *Drafter, e: mtl.ComputeEncoder, s: *const state.Scratch, lane: *const state.Lane, pending: u32, anchor: u64, depth: u32) void {
        s.ids.slice(u32, 1)[0] = pending;
        glue.embedNorm(&d.target.glue, e, at(s.ids, 0), r(d.target.weights.embed), r(d.target.weights.embed_norm), at(s.x, 0), 1);
        k.initializeRows(&d.pipelines, e, at(s.x, 0), r(d.weights.weights.mask), at(s.x, 0));
        k.residualNorm(&d.pipelines, e, at(s.x, 0), at(s.resid, 0), r(d.weights.weights.layers[0].input_norm), at(s.resid, 0), at(s.x, 0), 8, true);
        for (d.weights.weights.layers, 0..) |layer, i| {
            if (d.q4_codes) |qc| {
                const qp = d.q4_params.?;
                const cb0 = i * q4_layer_codes;
                const pb0 = i * q4_layer_params;
                k.q4Units(&d.screen, e, at(qc, cb0 + q4_off_q_c), at(qp, pb0 + q4_off_q_p), at(s.x, 0), at(s.q_raw, 0), at(s.k_raw, 0), at(s.v_raw, 0), state.hidden, state.query, state.kv, state.kv);
                k.blockQkv(&d.pipelines, e, at(s.q_raw, 0), at(s.k_raw, 0), at(s.v_raw, 0), r(layer.q_norm), r(layer.k_norm), at(s.q, 0), at(s.kb, 0), at(s.vb, 0), @intCast(anchor));
                k.attend(&d.pipelines, e, at(s.q, 0), at(s.kb, 0), at(s.vb, 0), at(lane.k[i], 0), at(lane.v[i], 0), at(d.scores.?, 0), at(s.attn, 0), @intCast(lane.end), @intCast(lane.rings[i]), i != 1, @intCast(lane.from));
                k.q4Units(&d.screen, e, at(qc, cb0 + q4_off_o_c), at(qp, pb0 + q4_off_o_p), at(s.attn, 0), at(s.y, 0), at(s.y, 0), at(s.y, 0), state.query, state.hidden, 0, 0);
                k.residualNorm(&d.pipelines, e, at(s.y, 0), at(s.resid, 0), r(layer.post_norm), at(s.next_resid, 0), at(s.x, 0), 8, false);
                k.q4Units(&d.screen, e, at(qc, cb0 + q4_off_gate_c), at(qp, pb0 + q4_off_gate_p), at(s.x, 0), at(s.gate, 0), at(s.up, 0), at(s.gate, 0), state.hidden, state.intermediate, state.intermediate, 0);
                k.swiglu(&d.pipelines, e, at(s.gate, 0), at(s.up, 0), at(s.act, 0), 8 * state.intermediate);
                k.q4Units(&d.screen, e, at(qc, cb0 + q4_off_down_c), at(qp, pb0 + q4_off_down_p), at(s.act, 0), at(s.y, 0), at(s.y, 0), at(s.y, 0), state.intermediate, state.hidden, 0, 0);
            } else {
                dense3(d.target, e, layer.q, layer.k, layer.v, at(s.x, 0), at(s.q_raw, 0), at(s.k_raw, 0), at(s.v_raw, 0), 8);
                k.blockQkv(&d.pipelines, e, at(s.q_raw, 0), at(s.k_raw, 0), at(s.v_raw, 0), r(layer.q_norm), r(layer.k_norm), at(s.q, 0), at(s.kb, 0), at(s.vb, 0), @intCast(anchor));
                k.attend(&d.pipelines, e, at(s.q, 0), at(s.kb, 0), at(s.vb, 0), at(lane.k[i], 0), at(lane.v[i], 0), at(d.scores.?, 0), at(s.attn, 0), @intCast(lane.end), @intCast(lane.rings[i]), i != 1, @intCast(lane.from));
                fwd.dense(d.target, e, layer.o, at(s.attn, 0), at(s.y, 0), 8);
                k.residualNorm(&d.pipelines, e, at(s.y, 0), at(s.resid, 0), r(layer.post_norm), at(s.next_resid, 0), at(s.x, 0), 8, false);
                dense2(d.target, e, layer.gate, layer.up, at(s.x, 0), at(s.gate, 0), at(s.up, 0), 8);
                k.swiglu(&d.pipelines, e, at(s.gate, 0), at(s.up, 0), at(s.act, 0), 8 * state.intermediate);
                fwd.dense(d.target, e, layer.down, at(s.act, 0), at(s.y, 0), 8);
            }
            const next_norm = if (i + 1 < d.weights.weights.layers.len) d.weights.weights.layers[i + 1].input_norm else d.weights.weights.norm;
            k.residualNorm(&d.pipelines, e, at(s.y, 0), at(s.next_resid, 0), r(next_norm), at(s.resid, 0), at(s.x, 0), 8, false);
        }
        k.kappa(&d.pipelines, e, at(s.x, state.hidden * 2), at(lane.newest, 0), r(d.weights.weights.rnn), r(d.weights.weights.rnn_bias), at(s.kappa, 0), depth);
    }

    fn encodeBlock(d: *Drafter, e: mtl.ComputeEncoder, s: *const state.Scratch, lane: *const state.Lane, pending: u32, anchor: u64, depth: u32) void {
        d.encodeBlockCore(e, s, lane, pending, anchor, depth);
        k.base(&d.pipelines, e, r(d.target.weights.unembed), at(s.x, state.hidden * 2), at(s.base, 0), depth, state.vocabulary);
    }

    fn recurrent(d: *Drafter, e: mtl.ComputeEncoder, s: *const state.Scratch, step: usize, previous: u32, second_bank: bool) void {
        const current = if (second_bank) s.next_state else s.state;
        const next = if (second_bank) s.state else s.next_state;
        k.recurrentStep(&d.pipelines, e, at(current, 0), at(s.kappa, step * 3 * state.rank * 4), r(d.weights.weights.rnn), r(d.weights.weights.markov_w1), at(s.z, 0), at(next, 0), at(s.lambda, 0), previous);
        k.finalLogits(&d.pipelines, e, at(s.base, step * state.vocabulary * 2), at(d.weights.phi, 0), at(d.weights.beta, 0), at(s.lambda, 0), at(s.logits, 0));
    }

    fn slot(d: *Drafter, index: usize) !*const state.Scratch {
        if (index == 0) return &d.scratch;
        while (d.more.items.len < index) {
            var extra = try state.Scratch.init(d.target.device);
            errdefer extra.deinit();
            try d.more.append(d.gpa, extra);
        }
        return &d.more.items[index - 1];
    }

    /// One stream's chained hold (greedy or sampled): block + 2-bit screened exact head; `held` takes the drafts.
    fn encodeChain(d: *Drafter, e: mtl.ComputeEncoder, s: *const state.Scratch, lane: *const state.Lane, held: Ref, pending: u32, anchor: u64, depth: usize, sampling: ?lanes.Sampling) void {
        d.encodeBlockCore(e, s, lane, pending, anchor, @intCast(depth));
        k.q2Proj(&d.screen, e, at(d.u_codes, 0), at(d.u_params, 0), at(s.x, state.hidden * 2), at(s.base, 0), @intCast(depth), state.vocabulary);
        var second_bank = false;
        for (0..depth) |step| {
            const current = if (second_bank) s.next_state else s.state;
            const next = if (second_bank) s.state else s.next_state;
            k.chainStep(&d.chain, e, at(current, 0), at(s.kappa, step * 3 * state.rank * 4), r(d.weights.weights.rnn), r(d.weights.weights.markov_w1), at(s.z, 0), at(next, 0), at(s.lambda, 0), held, @intCast(step));
            k.q2Screen(&d.screen, e, at(s.base, step * state.vocabulary * 2), at(d.p_codes, 0), at(d.p_params, 0), at(d.weights.beta, 0), at(s.lambda, 0), at(d.cand_ids, 0));
            k.q2Refine(&d.screen, e, at(d.cand_ids, 0), r(d.target.weights.unembed), at(s.x, (1 + step) * state.hidden * 2), at(d.weights.phi, 0), at(d.weights.beta, 0), at(s.lambda, 0), at(d.cand_logits, 0), at(d.cand_out_ids, 0));
            if (sampling) |st| {
                const rule = st.gpuRule();
                const sh = rule.seedHalves();
                k.q2Sample(&d.screen, e, at(d.cand_logits, 0), at(d.cand_out_ids, 0), held, .{ .step = @intCast(step), .position = @intCast(anchor + 1 + step), .seed_lo = sh[0], .seed_hi = sh[1], .top_k = rule.top_k, .inv_t = rule.inv_t, .top_p = rule.top_p, .min_log = rule.min_log });
            } else {
                k.q2Pick(&d.screen, e, at(d.cand_logits, 0), at(d.cand_out_ids, 0), held, @intCast(step));
            }
            second_bank = !second_bank;
        }
    }

    /// A full hold on an empty context, the third of three runs timed: the depth rule prices a draft at its share.
    pub fn measure(d: *Drafter) !void {
        var lane = try state.Lane.init(d.target.device, d.capacity);
        defer lane.deinit();
        const held = Ref{ .buf = d.chain_tokens, .off = 0 };
        var ms: f64 = 0;
        for (0..3) |attempt| {
            var t0: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &t0);
            @memset(d.scratch.state.slice(u16, state.rank), 0);
            d.chain_tokens.slice(u32, 8)[0] = 2;
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            d.encodeChain(e, &d.scratch, &lane, held, 2, 0, hold_depth, null);
            e.end();
            cb.commit();
            try failed(cb, "timed hold");
            var t1: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &t1);
            if (attempt == 2) ms = @as(f64, @floatFromInt(t1.sec - t0.sec)) * 1e3 + @as(f64, @floatFromInt(t1.nsec - t0.nsec)) / 1e6;
        }
        d.step_ms = ms / @as(f64, @floatFromInt(hold_depth));
    }

    /// Chained streams (greedy or sampled): blocks and all draft steps in one command buffer on the GPU.
    fn holdChained(d: *Drafter, items: []const dr.Hold) !void {
        const tokens = d.chain_tokens.slice(u32, chain_streams * 8);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        d.target.queue.roundBegin("dspark_hold");
        defer d.target.queue.roundEnd();
        const cb = d.target.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (items, 0..) |item, i| {
            const depth: usize = @min(item.depth, 7);
            const s = try d.slot(i);
            const lane = d.contexts.getPtr(item.stream).?;
            const held = Ref{ .buf = d.chain_tokens, .off = i * 8 * @sizeOf(u32) };
            tokens[i * 8] = item.pending;
            d.encodeChain(e, s, lane, held, item.pending, item.position - 1, depth, item.sampling);
        }
        e.end();
        cb.commit();
        try failed(cb, "chained block");
        for (items, 0..) |item, i| {
            const depth: usize = @min(item.depth, 7);
            const lane = d.contexts.getPtr(item.stream).?;
            for (0..depth) |step| {
                const token = tokens[i * 8 + step + 1];
                if (token >= state.vocabulary) return error.DraftNonfinite;
                lane.held[step] = token;
            }
            lane.held_len = @intCast(depth);
        }
    }

    /// Every stream's block and first step in one command buffer, then one command buffer a later step for all of them.
    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        if (items.len == 0) return;
        var deepest: usize = 0;
        for (items, 0..) |item, i| {
            const lane = d.contexts.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            if (item.depth == 0 or item.position == 0) return error.DraftDepth;
            if (lane.end != item.position - 1) return error.DraftContextGap;
            deepest = @max(deepest, @min(item.depth, 7));
            @memset((try d.slot(i)).state.slice(u16, state.rank), 0);
        }
        if (items.len <= chain_streams) return d.holdChained(items);
        const previous = try d.gpa.alloc(u32, items.len);
        defer d.gpa.free(previous);
        const values = try d.gpa.alloc(f32, state.vocabulary);
        defer d.gpa.free(values);
        var second_bank = false;
        for (0..deepest) |step| {
            {
                const pool = mtl.objc.Pool.push();
                defer pool.pop();
                const cb = d.target.queue.commandBuffer();
                const e = cb.compute(.serial);
                for (items, 0..) |item, i| {
                    const depth: usize = @min(item.depth, 7);
                    if (step >= depth) continue;
                    const s = try d.slot(i);
                    if (step == 0) {
                        const lane = d.contexts.getPtr(item.stream).?;
                        d.encodeBlock(e, s, lane, item.pending, item.position - 1, @intCast(depth));
                        previous[i] = item.pending;
                    }
                    d.recurrent(e, s, step, previous[i], second_bank);
                }
                e.end();
                cb.commit();
                try failed(cb, if (step == 0) "block" else "recurrent step");
            }
            for (items, 0..) |item, i| {
                const depth: usize = @min(item.depth, 7);
                if (step >= depth) continue;
                const lane = d.contexts.getPtr(item.stream).?;
                const raw = (try d.slot(i)).logits.slice(u16, state.vocabulary);
                for (raw, values) |bits, *value| value.* = @bitCast(@as(u32, bits) << 16);
                const base_position = std.math.cast(u32, item.position) orelse return error.DraftDepth;
                const position = std.math.add(u32, base_position, @intCast(step)) catch return error.DraftDepth;
                previous[i] = try gpu_rule.sample(d.gpa, values, item.sampling, position, null);
                if (previous[i] >= state.vocabulary) return error.DraftNonfinite;
                lane.held[step] = previous[i];
                if (step + 1 == depth) lane.held_len = @intCast(depth);
            }
            second_bank = !second_bank;
        }
    }
};

test "Mini taps and block depth match checkpoint section eight" {
    try std.testing.expectEqualSlices(u32, &.{ 5, 23, 41, 48 }, &tap_list);
    try std.testing.expectEqual(@as(u32, 48), tap_list[3]);
    try std.testing.expectEqual(@as(u64, 768 * 1032), Drafter.scoreBytes(1024));
}
