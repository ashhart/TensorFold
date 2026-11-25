//! Inkling 2 Mini DSpark: batched async context absorb, online-softmax block pass, and GPU chain draw.
const std = @import("std");
const builtin = @import("builtin");
const mtl = @import("metal");
const lanes = @import("lanes");
const ck = @import("../../../core/checkpoint_metal.zig");
const Model = @import("model.zig").Model;
const fwd = @import("forward.zig");
const glue = @import("glue.zig");
const k = @import("dspark_kernels.zig");
const dspark_sample = @import("dspark_sample.zig");
const weight_loader = @import("dspark_weights.zig");
const state = @import("dspark_state.zig");
const st = @import("state.zig");
const lane_snap = @import("dspark_snap.zig");

const dr = lanes.drafter;
const Stream = lanes.Stream;
const Ref = k.Ref;
const tap_list = [_]u32{ 5, 23, 41, 48 };
const draft_prior = [_]f64{ 0.55, 0.45, 0.35 };

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
fn sampledOf(s: ?lanes.Sampling) ?lanes.Sampling {
    const t = s orelse return null;
    return if (t.temperature > 0) t else null;
}

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    target: *const Model,
    weights: weight_loader.Loaded,
    pipelines: k.Pipelines,
    sampler: dspark_sample.Sampler,
    scratch: state.Scratch,
    capacity: usize = 1,
    step_ms: f64 = 0,
    hold_costs: [8]lanes.depth.HoldCost = undefined,
    hold_timed: usize = 0,
    contexts: std.AutoHashMapUnmanaged(*Stream, state.Lane) = .empty,
    absorb_fences: [3]mtl.counters.Fence = undefined,
    absorb_cbs: [3]?mtl.CommandBuffer = @splat(null),
    absorb_ring: usize = 0,
    last_absorb: ?usize = null,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, target: *const Model, drafter_dir: []const u8) !*Drafter {
        _ = io;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = .{
            .gpa = gpa,
            .target = target,
            .weights = weight_loader.load(gpa, target, drafter_dir) catch |err| return err,
            .pipelines = undefined,
            .sampler = undefined,
            .scratch = undefined,
        };
        errdefer d.weights.deinit();
        d.pipelines = try k.Pipelines.load(device);
        errdefer d.pipelines.deinit();
        d.sampler = try dspark_sample.Sampler.load(device);
        errdefer d.sampler.deinit();
        d.scratch = try state.Scratch.init(device);
        errdefer d.scratch.deinit();
        var made: usize = 0;
        errdefer for (d.absorb_fences[0..made]) |f| f.deinit();
        for (&d.absorb_fences) |*f| {
            f.* = try mtl.counters.Fence.init(device);
            made += 1;
        }
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        d.settle() catch {};
        for (&d.absorb_cbs) |*slot| if (slot.*) |cb| {
            cb.release();
            slot.* = null;
        };
        for (d.absorb_fences) |f| f.deinit();
        var it = d.contexts.valueIterator();
        while (it.next()) |lane| lane.deinit();
        d.contexts.deinit(d.gpa);
        d.scratch.deinit();
        d.sampler.deinit();
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

    /// Settles pending absorbs and updates the per-lane global ring capacity before any stream opens.
    pub fn setContext(d: *Drafter, window: u32) !void {
        const capacity: usize = @intCast(window);
        if (capacity == 0 or capacity > 1048576) return error.DSparkLaneGeometry;
        if (d.contexts.count() != 0) return error.DSparkContextActive;
        try d.settle();
        d.capacity = capacity;
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .ptr = d, .vtable = &.{
            .taps = tapsFn,
            .facts = factsFn,
            .open = openFn,
            .absorb = absorbFn,
            .hold = holdFn,
            .clear_proposals = clearProposalsFn,
            .held = heldFn,
            .chances = chancesFn,
            .release = releaseFn,
        } };
    }

    pub fn saveSnap(d: *Drafter, stream: *Stream, at_pos: u32, tap: st.Tap, tap_gen: u64) !?state.Snap {
        const lane = d.contexts.getPtr(stream) orelse return null;
        // absorbs run asynchronously: the rings and lane.end must be final before they are copied
        try d.settle();
        const tail = try lane_snap.resolveTailTap(tap, at_pos, lane.end, tap_gen);
        return try state.Snap.save(d.target.device, d.target.queue, lane, tail);
    }

    pub fn restoreSnap(d: *Drafter, stream: *Stream, snap: ?*const state.Snap, at_pos: usize) !void {
        const lane = d.contexts.getPtr(stream) orelse return;
        const s = snap orelse return error.DSparkSnapshotMissing;
        const expected_end = std.math.add(u64, s.end, if (s.tail_off != null) @as(u64, 1) else 0) catch return error.DSparkSnapshotGeometry;
        if (expected_end != at_pos or expected_end > lane.rings[1]) return error.DSparkSnapshotGeometry;
        try d.settle();
        try state.Snap.restore(lane, d.target.queue, s);
        lane.sampled = false;
        if (s.tail_off) |tail_off| {
            errdefer lane_snap.clearLane(lane);
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            const one = [_]AbsorbSlice{.{ .lane = lane, .taps = at(s.buf, tail_off), .start = s.end, .rows = 1, .row_off = 0 }};
            d.absorbBatch(e, &one, 1);
            e.end();
            cb.commit();
            try failed(cb, "restore tail");
            lane.end = s.end + 1;
            lane.held_len = 0;
        }
    }

    pub fn resetLane(d: *Drafter, stream: *Stream) void {
        d.settle() catch {};
        if (d.contexts.getPtr(stream)) |lane| lane_snap.clearLane(lane);
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }
    fn tapsFn(_: *anyopaque) []const u32 {
        return &tap_list;
    }

    fn factsFn(ptr: *anyopaque) dr.Facts {
        const d = cast(ptr);
        return .{ .depth = 7, .step_ms = d.step_ms, .prior = &draft_prior, .plain_guard = false, .batched = true, .chances = true, .copy_first = true, .hold_costs = d.hold_costs[0..d.hold_timed], .stream_bytes = @intCast(state.laneBytes(d.capacity)) };
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
        d.settle() catch {};
        if (d.contexts.fetchRemove(stream)) |entry| {
            var lane = entry.value;
            lane.deinit();
        }
    }

    fn clearProposalsFn(ptr: *anyopaque, stream: *Stream) void {
        const d = cast(ptr);
        d.settle() catch {};
        if (d.contexts.getPtr(stream)) |lane| {
            lane.held_len = 0;
            lane.sampled = false;
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

    fn chancesFn(ptr: *anyopaque, stream: *Stream, out: []f64) anyerror!bool {
        const d = cast(ptr);
        const lane = d.contexts.getPtr(stream) orelse return false;
        if (lane.held_len == 0 or lane.sampled or out.len > lane.held_len) return false;
        var reach: f64 = 1;
        for (out, lane.probs[0..out.len]) |*o, p| {
            reach *= @as(f64, p);
            o.* = reach;
        }
        return true;
    }

    fn settle(d: *Drafter) !void {
        const start = d.absorb_ring % 3;
        var first_err: ?anyerror = null;
        for (0..3) |offset| {
            const idx = (start + offset) % 3;
            const cb = d.absorb_cbs[idx] orelse continue;
            d.absorb_cbs[idx] = null;
            defer cb.release();
            cb.wait();
            if (cb.failure()) |msg| {
                std.log.err("inkling2_loop DSpark absorb: {s}", .{msg});
                if (first_err == null) first_err = error.GpuFailed;
            }
        }
        d.last_absorb = null;
        if (first_err) |err| return err;
    }

    const AbsorbSlice = struct { lane: *state.Lane, taps: Ref, start: u64, rows: usize, row_off: usize };

    fn absorbBatch(d: *Drafter, e: mtl.ComputeEncoder, batch: []const AbsorbSlice, total_rows: usize) void {
        for (batch) |b| {
            const c = at(d.scratch.fused, b.row_off * state.hidden * 2);
            k.fuseTaps(&d.pipelines, e, b.taps, r(d.weights.weights.fuse), at(d.weights.dflare, 0), c, at(b.lane.newest, 0), @intCast(b.rows), @intCast(total_rows));
        }
        e.barrier();
        for (d.weights.weights.layers, 0..) |layer, i| {
            const c = at(d.scratch.fused, i * total_rows * state.hidden * 2);
            const ck_ref = at(d.scratch.ck, i * total_rows * state.kv * 2);
            const cv_ref = at(d.scratch.cv, i * total_rows * state.kv * 2);
            fwd.dense(d.target, e, layer.k_ctx, c, ck_ref, total_rows);
            fwd.dense(d.target, e, layer.v_ctx, c, cv_ref, total_rows);
        }
        e.barrier();
        for (batch) |b| for (d.weights.weights.layers, 0..) |layer, i| {
            const ck_ref = at(d.scratch.ck, (i * total_rows + b.row_off) * state.kv * 2);
            const cv_ref = at(d.scratch.cv, (i * total_rows + b.row_off) * state.kv * 2);
            k.commitContext(&d.pipelines, e, ck_ref, cv_ref, r(layer.k_norm_ctx), at(b.lane.k[i], 0), at(b.lane.v[i], 0), @intCast(b.start), @intCast(b.rows), @intCast(b.lane.rings[i]));
        };
    }

    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        if (items.len == 0) return;
        const row_bytes = 4 * state.hidden * 2;
        const positions = try d.gpa.alloc(u64, items.len);
        defer d.gpa.free(positions);
        for (items, positions) |item, *pos| {
            const lane = d.contexts.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            const f = item.features;
            if (f.space != .device or f.dtype != .bf16 or f.row_bytes != row_bytes) return error.DraftFeaturesUnsupported;
            if (item.rows != null) return error.TreesNotBuilt;
            if (f.rows == 0 or f.rows > 1048576) return error.DraftFeaturesUnsupported;
            if (item.start < lane.end or item.start + f.rows > 1048576) return error.DraftContextGap;
            pos.* = item.start;
        }
        const slot = d.absorb_ring % 3;
        if (d.absorb_cbs[slot]) |old_cb| {
            d.absorb_cbs[slot] = null;
            defer old_cb.release();
            try failed(old_cb, "absorb");
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = d.target.queue.commandBuffer();
        const e = cb.compute(.concurrent);
        if (d.last_absorb) |prev| if (!d.target.queue.isMetal4()) d.absorb_fences[prev].waitCompute(e);
        var need_barrier = false;
        var idx: usize = 0;
        while (idx < items.len) {
            if (need_barrier) e.barrier();
            var slices: [state.max_streams]AbsorbSlice = undefined;
            var n: usize = 0;
            var total_rows: usize = 0;
            while (idx < items.len and n < state.max_streams and total_rows < state.absorb_rows) {
                const item = items[idx];
                const end = item.start + item.features.rows;
                const rem = end - positions[idx];
                const rows: usize = @intCast(@min(@as(u64, state.absorb_rows - total_rows), rem));
                const source = mtl.Buffer{ .id = @ptrFromInt(item.features.buffer) };
                const off = item.features.offset + @as(usize, @intCast(positions[idx] - item.start)) * row_bytes;
                slices[n] = .{ .lane = d.contexts.getPtr(item.stream).?, .taps = at(source, off), .start = positions[idx], .rows = rows, .row_off = total_rows };
                positions[idx] += rows;
                total_rows += rows;
                n += 1;
                if (positions[idx] == end) idx += 1;
            }
            d.absorbBatch(e, slices[0..n], total_rows);
            need_barrier = true;
        }
        for (items) |item| {
            const lane = d.contexts.getPtr(item.stream).?;
            if (item.start > lane.end) lane.from = item.start;
            lane.end = item.start + item.features.rows;
            lane.held_len = 0;
        }
        if (!d.target.queue.isMetal4()) d.absorb_fences[slot].update(e);
        e.end();
        cb.commit();
        d.absorb_cbs[slot] = cb.retain();
        d.last_absorb = slot;
        d.absorb_ring +%= 1;
    }

    fn encodeBatchBlock(d: *Drafter, e: mtl.ComputeEncoder, lanes_in: []const *const state.Lane, anchors: []const u32, max_depth: usize) void {
        const s = &d.scratch;
        const n = lanes_in.len;
        const rows = n * state.block;
        var ctx_k: [state.max_streams]Ref = undefined;
        var ctx_v: [state.max_streams]Ref = undefined;
        var committed: [state.max_streams]u32 = undefined;
        var ring: [state.max_streams]u32 = undefined;
        var from: [state.max_streams]u32 = undefined;
        var newest_fs: [state.max_streams]Ref = undefined;
        for (lanes_in, 0..) |lane, i| {
            committed[i] = @intCast(lane.end);
            from[i] = @intCast(lane.from);
            newest_fs[i] = at(lane.newest, 0);
        }
        glue.embedNorm(&d.target.glue, e, at(s.ids, 0), r(d.target.weights.embed), r(d.target.weights.embed_norm), at(s.y, 0), @intCast(n));
        k.initializeRows(&d.pipelines, e, at(s.y, 0), r(d.weights.weights.mask), at(s.x, 0), @intCast(n));
        k.residualNorm(&d.pipelines, e, at(s.x, 0), at(s.resid, 0), r(d.weights.weights.layers[0].input_norm), at(s.resid, 0), at(s.x, 0), @intCast(rows), true);
        for (d.weights.weights.layers, 0..) |layer, li| {
            for (lanes_in, 0..) |lane, si| {
                ctx_k[si] = at(lane.k[li], 0);
                ctx_v[si] = at(lane.v[li], 0);
                ring[si] = @intCast(lane.rings[li]);
            }
            fwd.dense(d.target, e, layer.q, at(s.x, 0), at(s.q_raw, 0), rows);
            fwd.dense(d.target, e, layer.k, at(s.x, 0), at(s.k_raw, 0), rows);
            fwd.dense(d.target, e, layer.v, at(s.x, 0), at(s.v_raw, 0), rows);
            k.blockQkv(&d.pipelines, e, at(s.q_raw, 0), at(s.k_raw, 0), at(s.v_raw, 0), r(layer.q_norm), r(layer.k_norm), at(s.q, 0), at(s.kb, 0), at(s.vb, 0), anchors);
            k.attend(&d.pipelines, e, at(s.q, 0), at(s.kb, 0), at(s.vb, 0), at(s.attn, 0), ctx_k[0..n], ctx_v[0..n], committed[0..n], ring[0..n], from[0..n], li != 1);
            fwd.dense(d.target, e, layer.o, at(s.attn, 0), at(s.y, 0), rows);
            k.residualNorm(&d.pipelines, e, at(s.y, 0), at(s.resid, 0), r(layer.post_norm), at(s.next_resid, 0), at(s.x, 0), @intCast(rows), false);
            fwd.dense(d.target, e, layer.gate, at(s.x, 0), at(s.gate, 0), rows);
            fwd.dense(d.target, e, layer.up, at(s.x, 0), at(s.up, 0), rows);
            k.swiglu(&d.pipelines, e, at(s.gate, 0), at(s.up, 0), at(s.act, 0), @intCast(rows * state.intermediate));
            fwd.dense(d.target, e, layer.down, at(s.act, 0), at(s.y, 0), rows);
            const next_norm = if (li + 1 < d.weights.weights.layers.len) d.weights.weights.layers[li + 1].input_norm else d.weights.weights.norm;
            k.residualNorm(&d.pipelines, e, at(s.y, 0), at(s.next_resid, 0), r(next_norm), at(s.resid, 0), at(s.x, 0), @intCast(rows), false);
        }
        const single = n == 1;
        const base_in = at(s.x, if (single) state.hidden * 2 else 0);
        const base_rows: usize = if (single) max_depth else rows;
        k.base(&d.pipelines, e, r(d.target.weights.unembed), base_in, at(s.base, 0), @intCast(base_rows), state.vocabulary);
        k.kappa(&d.pipelines, e, base_in, newest_fs[0..n], at(d.weights.rnn_kappa, 0), r(d.weights.weights.rnn_bias), at(s.kappa, 0), @intCast(max_depth), if (single) 7 else 8, if (single) 0 else 1);
    }

    fn runHoldBatch(d: *Drafter, lane_ptrs: []const *const state.Lane, anchors: []const u32, base_pos: []const u32, samps: []const ?lanes.Sampling, flags: u32, max_depth: usize) !void {
        const s = &d.scratch;
        const count = lane_ptrs.len;
        @memset(s.state.slice(u16, count * state.rank), 0);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = d.target.queue.commandBuffer();
        const e = cb.compute(.serial);
        if (d.last_absorb) |prev| if (!d.target.queue.isMetal4()) d.absorb_fences[prev].waitCompute(e);
        d.encodeBatchBlock(e, lane_ptrs, anchors, max_depth);
        var second_bank = false;
        const single = count == 1;
        const base_stride: u32 = if (single) state.vocabulary else 8 * state.vocabulary;
        for (0..max_depth) |step| {
            if (step != 0) second_bank = !second_bank;
            const cur = if (second_bank) s.next_state else s.state;
            const nxt = if (second_bank) s.state else s.next_state;
            k.recurrentStep(&d.pipelines, e, at(cur, 0), at(s.kappa, step * count * 3 * state.rank * 4), at(d.weights.rnn_rec, 0), r(d.weights.weights.markov_w1), at(s.z, 0), at(nxt, 0), at(s.lambda, 0), at(s.held, 0), @intCast(count), 0, @intCast(step), true);
            const base_off = (if (single) step else 1 + step) * state.vocabulary * 2;
            const la: k.LogitsArgs = .{ .count = state.vocabulary, .phi_stride = weight_loader.vocabulary, .base_stride = base_stride, .streams = @intCast(count), .step = @intCast(step), .mode = 0, .flags = flags };
            k.finalLogits(&d.pipelines, e, at(s.base, base_off), at(d.weights.phi, 0), at(d.weights.beta, 0), at(s.lambda, 0), at(s.logits, 0), at(s.group_stats, 0), at(s.held, 0), at(s.probs, 0), at(s.sample_tok, 0), la);
            if (flags != 0) d.sampler.encodeRuns(e, s.logits, s.sample_tok, s.sample_ck, s.sample_ci, samps, base_pos, @intCast(step));
            var la1 = la;
            la1.mode = 1;
            k.finalLogits(&d.pipelines, e, at(s.base, base_off), at(d.weights.phi, 0), at(d.weights.beta, 0), at(s.lambda, 0), at(s.logits, 0), at(s.group_stats, 0), at(s.held, 0), at(s.probs, 0), at(s.sample_tok, 0), la1);
        }
        e.end();
        cb.commit();
        try failed(cb, "hold");
    }

    pub fn measure(d: *Drafter) !void {
        return d.warm(1);
    }

    /// Warms DSpark absorb and hold pipelines and times hold passes at representative widths and depths 1 and 7.
    pub fn warm(d: *Drafter, streams: u32) !void {
        try d.settle();
        const n_lanes: usize = @min(@max(@as(usize, streams), 1), state.max_streams);
        var temp: [state.max_streams]state.Lane = undefined;
        var made: usize = 0;
        defer for (temp[0..made]) |*lane| lane.deinit();
        for (temp[0..n_lanes]) |*lane| {
            lane.* = try state.Lane.init(d.target.device, 64);
            made += 1;
        }
        const tap_bytes = 4 * state.hidden * 2;
        const tap_buf = try d.target.device.buffer(tap_bytes, mtl.ResourceOptions.shared);
        defer tap_buf.deinit();
        @memset(tap_buf.slice(u8, tap_bytes), 0);
        var slices: [state.max_streams]AbsorbSlice = undefined;
        for (temp[0..n_lanes], slices[0..n_lanes], 0..) |*lane, *sl, i| {
            lane.from = 0;
            lane.end = 1;
            sl.* = .{ .lane = lane, .taps = at(tap_buf, 0), .start = 0, .rows = 1, .row_off = i };
        }
        {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.concurrent);
            d.absorbBatch(e, slices[0..n_lanes], n_lanes);
            e.end();
            cb.commit();
            try failed(cb, "warm absorb");
        }
        var lane_ptrs: [state.max_streams]*const state.Lane = undefined;
        for (temp[0..n_lanes], lane_ptrs[0..n_lanes]) |*lane, *p| p.* = lane;
        const zeros: [state.max_streams]u32 = @splat(0);
        const ones: [state.max_streams]u32 = @splat(1);
        const nosamp: [state.max_streams]?lanes.Sampling = @splat(null);
        @memset(d.scratch.ids.slice(u32, n_lanes), 2);
        for (0..n_lanes) |i| d.scratch.held.slice(u32, n_lanes * 8)[i * 8] = 2;
        try d.runHoldBatch(lane_ptrs[0..1], ones[0..1], zeros[0..1], nosamp[0..1], 0, 7);
        var counts: [4]usize = undefined;
        var n_counts: usize = 0;
        for ([_]usize{ 1, 4, 8, 16 }) |c| if (c == 1 or streams >= c) {
            counts[n_counts] = c;
            n_counts += 1;
        };
        if (streams > 1 and n_counts == 1) {
            counts[n_counts] = @min(@as(usize, streams), 16);
            n_counts += 1;
        }
        d.hold_timed = 0;
        var ms1: f64 = 0;
        var ms7: f64 = 0;
        for (counts[0..n_counts]) |sc| {
            for ([_]usize{ 1, 7 }) |depth| {
                var ms: f64 = 0;
                for (0..2) |run| {
                    const t0 = nowNs();
                    var off: usize = 0;
                    while (off < sc) {
                        const batch = @min(state.max_streams, sc - off);
                        try d.runHoldBatch(lane_ptrs[0..batch], ones[0..batch], zeros[0..batch], nosamp[0..batch], 0, depth);
                        off += batch;
                    }
                    if (run == 1) ms = @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
                }
                if (sc == 1 and depth == 1) ms1 = ms;
                if (sc == 1 and depth == 7) ms7 = ms;
                d.hold_costs[d.hold_timed] = .{ .streams = @intCast(sc), .depth = @intCast(depth), .ms = ms };
                d.hold_timed += 1;
            }
        }
        d.step_ms = @max(ms7 - ms1, 0.0) / 6.0;
    }

    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        const s = &d.scratch;
        var off: usize = 0;
        while (off < items.len) {
            const count: usize = @min(@as(usize, state.max_streams), items.len - off);
            const pass = items[off .. off + count];
            off += count;
            var lane_ptrs: [state.max_streams]*const state.Lane = undefined;
            var mut_lanes: [state.max_streams]*state.Lane = undefined;
            var anchors: [state.max_streams]u32 = undefined;
            var base_pos: [state.max_streams]u32 = undefined;
            var samps: [state.max_streams]?lanes.Sampling = undefined;
            var flags: u32 = 0;
            var max_depth: usize = 0;
            for (pass, 0..) |item, i| {
                const lane = d.contexts.getPtr(item.stream) orelse return error.DraftStreamUnknown;
                if (item.depth == 0 or item.position == 0) return error.DraftDepth;
                const anchor = item.position - 1;
                if (lane.end != anchor) return error.DraftContextGap;
                const pos0 = std.math.cast(u32, item.position) orelse return error.DraftDepth;
                const depth: usize = @min(item.depth, 7);
                _ = std.math.add(u32, pos0, @intCast(depth - 1)) catch return error.DraftDepth;
                lane_ptrs[i] = lane;
                mut_lanes[i] = lane;
                anchors[i] = @intCast(anchor);
                base_pos[i] = pos0;
                samps[i] = sampledOf(item.sampling);
                if (samps[i] != null) flags |= @as(u32, 1) << @intCast(i);
                max_depth = @max(max_depth, depth);
            }
            const ids_host = s.ids.slice(u32, count);
            const held_host = s.held.slice(u32, count * 8);
            for (pass, 0..) |item, i| {
                ids_host[i] = item.pending;
                held_host[i * 8] = item.pending;
            }
            try d.runHoldBatch(lane_ptrs[0..count], anchors[0..count], base_pos[0..count], samps[0..count], flags, max_depth);
            try d.settle();
            const probs_host = s.probs.slice(f32, count * 8);
            for (pass, mut_lanes[0..count], 0..) |item, lane, i| {
                const depth: usize = @min(item.depth, 7);
                for (0..depth) |step| {
                    const tok = held_host[i * 8 + step + 1];
                    if (tok >= state.vocabulary) return error.DraftNonfinite;
                    lane.held[step] = tok;
                    lane.probs[step] = probs_host[i * 8 + step];
                }
                lane.held_len = @intCast(depth);
                lane.sampled = samps[i] != null;
            }
        }
    }
};

fn nowNs() i128 {
    if (comptime !builtin.os.tag.isDarwin()) return 0;
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

test "Mini taps and block depth match checkpoint section eight" {
    try std.testing.expectEqualSlices(u32, &.{ 5, 23, 41, 48 }, &tap_list);
    try std.testing.expectEqual(@as(u32, 48), tap_list[3]);
    try std.testing.expectEqual(@as(u64, 0), Drafter.scoreBytes(1024));
}

test "DSpark exposes batched priced chances, sampled fallback, and copy_first" {
    if (!builtin.os.tag.isDarwin()) return error.SkipZigTest;
    var d: Drafter = .{ .gpa = std.testing.allocator, .target = undefined, .weights = undefined, .pipelines = undefined, .sampler = undefined, .scratch = undefined, .capacity = 128, .step_ms = 0.25 };
    d.hold_costs[0] = .{ .streams = 1, .depth = 1, .ms = 1.5 };
    d.hold_timed = 1;
    defer d.contexts.deinit(std.testing.allocator);
    const dr_vt = d.drafter();
    const f = dr_vt.facts();
    try std.testing.expectEqual(@as(u32, 7), f.depth);
    try std.testing.expectEqual(@as(usize, 3), f.prior.len);
    try std.testing.expect(!f.plain_guard and f.batched and f.chances and f.copy_first);
    try std.testing.expectEqual(@as(?f64, null), f.confidence_floor);
    try std.testing.expectEqual(@as(f64, 0.25), f.step_ms);
    try std.testing.expectEqual(@as(usize, 1), f.hold_costs.len);
    try std.testing.expect(dr_vt.vtable.clear_proposals != null);
    var fake_stream: u8 align(@alignOf(Stream)) = 0;
    const s: *Stream = @ptrCast(@alignCast(&fake_stream));
    var lane: state.Lane = .{ .k = undefined, .v = undefined, .rings = .{ 1, 1, 1 }, .newest = undefined, .held_len = 3 };
    lane.probs[0..3].* = .{ 0.8, 0.5, 0.25 };
    try d.contexts.put(std.testing.allocator, s, lane);
    var out: [3]f64 = undefined;
    try std.testing.expect(try dr_vt.chances(s, &out));
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), out[2], 1e-6);
    d.contexts.getPtr(s).?.sampled = true;
    try std.testing.expect(!try dr_vt.chances(s, &out));
    dr_vt.vtable.clear_proposals.?(dr_vt.ptr, s);
    try std.testing.expect(!d.contexts.getPtr(s).?.sampled);
    try std.testing.expect(!try dr_vt.chances(s, &out));
}

test {
    _ = lane_snap;
}
