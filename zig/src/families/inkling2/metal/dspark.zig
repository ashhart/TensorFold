//! Inkling 2 Mini DSpark: committed target features, three-layer proposal blocks, folded Markov logits.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const gpu_rule = lanes.gpu_rule;
const ck = @import("../../../core/checkpoint_metal.zig");
const Model = @import("model.zig").Model;
const fwd = @import("forward.zig");
const glue = @import("glue.zig");
const k = @import("dspark_kernels.zig");
const weight_loader = @import("dspark_weights.zig");
const state = @import("dspark_state.zig");

const dr = lanes.drafter;
const Stream = lanes.Stream;
const Ref = k.Ref;
const tap_list = [_]u32{ 5, 23, 41, 48 };
/// Drafts a hold offers: four rows a stream keep four streams' shared round at 16 of the forward's 32 rows.
const hold_depth: u32 = 3;

fn at(buf: mtl.Buffer, off: usize) Ref {
    return .{ .buf = buf, .off = off };
}
fn r(t: ck.Tensor) Ref {
    return .{ .buf = t.buffer, .off = t.offset };
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
    scores: ?mtl.Buffer = null,
    capacity: usize = 1,
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
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        var it = d.contexts.valueIterator();
        while (it.next()) |lane| lane.deinit();
        d.contexts.deinit(d.gpa);
        if (d.scores) |scores| scores.deinit();
        d.scratch.deinit();
        d.pipelines.deinit();
        d.weights.deinit();
        d.gpa.destroy(d);
    }

    pub fn fixedBytes(d: *const Drafter) u64 {
        return d.weights.fixedBytes() + state.scratchBytes();
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

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(_: *anyopaque) []const u32 {
        return &tap_list;
    }

    fn factsFn(ptr: *anyopaque) dr.Facts {
        const d = cast(ptr);
        return .{ .depth = hold_depth, .step_ms = 0, .plain_guard = true, .stream_bytes = @intCast(state.laneBytes(d.capacity)) };
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
            fwd.dense(d.target, e, layer.k_ctx, c, at(d.scratch.ck, 0), rows);
            fwd.dense(d.target, e, layer.v_ctx, c, at(d.scratch.cv, 0), rows);
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

    fn encodeBlock(d: *Drafter, e: mtl.ComputeEncoder, lane: *const state.Lane, pending: u32, anchor: u64, depth: u32) void {
        const s = &d.scratch;
        s.ids.slice(u32, 1)[0] = pending;
        glue.embedNorm(&d.target.glue, e, at(s.ids, 0), r(d.target.weights.embed), r(d.target.weights.embed_norm), at(s.x, 0), 1);
        k.initializeRows(&d.pipelines, e, at(s.x, 0), r(d.weights.weights.mask), at(s.x, 0));
        k.residualNorm(&d.pipelines, e, at(s.x, 0), at(s.resid, 0), r(d.weights.weights.layers[0].input_norm), at(s.resid, 0), at(s.x, 0), 8, true);
        for (d.weights.weights.layers, 0..) |layer, i| {
            fwd.dense(d.target, e, layer.q, at(s.x, 0), at(s.q_raw, 0), 8);
            fwd.dense(d.target, e, layer.k, at(s.x, 0), at(s.k_raw, 0), 8);
            fwd.dense(d.target, e, layer.v, at(s.x, 0), at(s.v_raw, 0), 8);
            k.blockQkv(&d.pipelines, e, at(s.q_raw, 0), at(s.k_raw, 0), at(s.v_raw, 0), r(layer.q_norm), r(layer.k_norm), at(s.q, 0), at(s.kb, 0), at(s.vb, 0), @intCast(anchor));
            k.attend(&d.pipelines, e, at(s.q, 0), at(s.kb, 0), at(s.vb, 0), at(lane.k[i], 0), at(lane.v[i], 0), at(d.scores.?, 0), at(s.attn, 0), @intCast(lane.end), @intCast(lane.rings[i]), i != 1, @intCast(lane.from));
            fwd.dense(d.target, e, layer.o, at(s.attn, 0), at(s.y, 0), 8);
            k.residualNorm(&d.pipelines, e, at(s.y, 0), at(s.resid, 0), r(layer.post_norm), at(s.next_resid, 0), at(s.x, 0), 8, false);
            fwd.dense(d.target, e, layer.gate, at(s.x, 0), at(s.gate, 0), 8);
            fwd.dense(d.target, e, layer.up, at(s.x, 0), at(s.up, 0), 8);
            k.swiglu(&d.pipelines, e, at(s.gate, 0), at(s.up, 0), at(s.act, 0), 8 * state.intermediate);
            fwd.dense(d.target, e, layer.down, at(s.act, 0), at(s.y, 0), 8);
            const next_norm = if (i + 1 < d.weights.weights.layers.len) d.weights.weights.layers[i + 1].input_norm else d.weights.weights.norm;
            k.residualNorm(&d.pipelines, e, at(s.y, 0), at(s.next_resid, 0), r(next_norm), at(s.resid, 0), at(s.x, 0), 8, false);
        }
        k.base(&d.pipelines, e, r(d.target.weights.unembed), at(s.x, state.hidden * 2), at(s.base, 0), depth, state.vocabulary);
        k.kappa(&d.pipelines, e, at(s.x, state.hidden * 2), at(lane.newest, 0), r(d.weights.weights.rnn), r(d.weights.weights.rnn_bias), at(s.kappa, 0), depth);
    }

    fn recurrent(d: *Drafter, e: mtl.ComputeEncoder, step: usize, previous: u32, second_bank: bool) void {
        const s = &d.scratch;
        const current = if (second_bank) s.next_state else s.state;
        const next = if (second_bank) s.state else s.next_state;
        k.recurrentStep(&d.pipelines, e, at(current, 0), at(s.kappa, step * 3 * state.rank * 4), r(d.weights.weights.rnn), r(d.weights.weights.markov_w1), at(s.z, 0), at(next, 0), at(s.lambda, 0), previous);
        k.finalLogits(&d.pipelines, e, at(s.base, step * state.vocabulary * 2), at(d.weights.phi, 0), at(d.weights.beta, 0), at(s.lambda, 0), at(s.logits, 0));
    }

    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        for (items) |item| {
            const lane = d.contexts.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            if (item.depth == 0 or item.position == 0) return error.DraftDepth;
            const anchor = item.position - 1;
            if (lane.end != anchor) return error.DraftContextGap;
            const depth: usize = @min(item.depth, 7);
            @memset(d.scratch.state.slice(u16, state.rank), 0);
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            d.encodeBlock(e, lane, item.pending, anchor, @intCast(depth));
            d.recurrent(e, 0, item.pending, false);
            e.end();
            cb.commit();
            try failed(cb, "block");
            const values = try d.gpa.alloc(f32, state.vocabulary);
            defer d.gpa.free(values);
            var previous = item.pending;
            var second_bank = false;
            for (0..depth) |step| {
                if (step != 0) {
                    second_bank = !second_bank;
                    const next_cb = d.target.queue.commandBuffer();
                    const next_e = next_cb.compute(.serial);
                    d.recurrent(next_e, step, previous, second_bank);
                    next_e.end();
                    next_cb.commit();
                    try failed(next_cb, "recurrent step");
                }
                const raw = d.scratch.logits.slice(u16, state.vocabulary);
                for (raw, values) |bits, *value| value.* = @bitCast(@as(u32, bits) << 16);
                const base_position = std.math.cast(u32, item.position) orelse return error.DraftDepth;
                const position = std.math.add(u32, base_position, @intCast(step)) catch return error.DraftDepth;
                previous = try gpu_rule.sample(d.gpa, values, item.sampling, position, null);
                if (previous >= state.vocabulary) return error.DraftNonfinite;
                lane.held[step] = previous;
            }
            lane.held_len = @intCast(depth);
        }
    }
};

test "Mini taps and block depth match checkpoint section eight" {
    try std.testing.expectEqualSlices(u32, &.{ 5, 23, 41, 48 }, &tap_list);
    try std.testing.expectEqual(@as(u32, 48), tap_list[3]);
    try std.testing.expectEqual(@as(u64, 768 * 1032), Drafter.scoreBytes(1024));
}
