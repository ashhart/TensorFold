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
const st = @import("state.zig");

comptime {
    if (st.feature_row_bytes != state.tap_row_bytes) @compileError("DSpark tail tap row bytes mismatch");
}

const dr = lanes.drafter;
const Stream = lanes.Stream;
const Ref = k.Ref;
const tap_list = [_]u32{ 5, 23, 41, 48 };
/// Drafts a hold offers at most: the block after its anchor. The depth rule prices how many a round takes.
const hold_depth: u32 = state.block - 1;
/// Streams one greedy hold can chain on the GPU; a wider hold samples on the host as sampled streams do.
const chain_streams = 16;

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
    /// Scratch for the second and later streams of one hold; the first uses `scratch`, as absorb does.
    more: std.ArrayList(state.Scratch) = .empty,
    /// The greedy chain's kernels, and each held stream's eight tokens: the anchor, then its drafts.
    chain: k.Chain = undefined,
    chain_tokens: mtl.Buffer = undefined,
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
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        var it = d.contexts.valueIterator();
        while (it.next()) |lane| lane.deinit();
        d.contexts.deinit(d.gpa);
        if (d.scores) |scores| scores.deinit();
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

    pub fn saveSnap(d: *Drafter, stream: *Stream, at_pos: u32, tap: st.Tap, tap_gen: u64) !?state.Snap {
        const lane = d.contexts.getPtr(stream) orelse return null;
        const tail = try resolveTailTap(tap, at_pos, lane.end, tap_gen);
        return try state.Snap.save(d.target.device, d.target.queue, lane, tail);
    }

    pub fn restoreSnap(d: *Drafter, stream: *Stream, snap: ?*const state.Snap, at_pos: usize) !void {
        const lane = d.contexts.getPtr(stream) orelse return;
        const s = snap orelse return error.DSparkSnapshotMissing;
        const expected_end = std.math.add(u64, s.end, if (s.tail_off != null) @as(u64, 1) else 0) catch return error.DSparkSnapshotGeometry;
        if (expected_end != at_pos or expected_end > lane.rings[1]) return error.DSparkSnapshotGeometry;
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

    fn encodeBlock(d: *Drafter, e: mtl.ComputeEncoder, s: *const state.Scratch, lane: *const state.Lane, pending: u32, anchor: u64, depth: u32) void {
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

    /// One stream's greedy hold: its block, then each step's recurrent pass, logits and pick; `held` takes the drafts.
    fn encodeChain(d: *Drafter, e: mtl.ComputeEncoder, s: *const state.Scratch, lane: *const state.Lane, held: Ref, pending: u32, anchor: u64, depth: usize) void {
        d.encodeBlock(e, s, lane, pending, anchor, @intCast(depth));
        var second_bank = false;
        for (0..depth) |step| {
            const current = if (second_bank) s.next_state else s.state;
            const next = if (second_bank) s.state else s.next_state;
            k.chainStep(&d.chain, e, at(current, 0), at(s.kappa, step * 3 * state.rank * 4), r(d.weights.weights.rnn), r(d.weights.weights.markov_w1), at(s.z, 0), at(next, 0), at(s.lambda, 0), held, @intCast(step));
            k.finalLogits(&d.pipelines, e, at(s.base, step * state.vocabulary * 2), at(d.weights.phi, 0), at(d.weights.beta, 0), at(s.lambda, 0), at(s.logits, 0));
            k.chainPick(&d.chain, e, at(s.logits, 0), held, @intCast(step));
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
            d.encodeChain(e, &d.scratch, &lane, held, 2, 0, hold_depth);
            e.end();
            cb.commit();
            try failed(cb, "timed hold");
            var t1: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &t1);
            if (attempt == 2) ms = @as(f64, @floatFromInt(t1.sec - t0.sec)) * 1e3 + @as(f64, @floatFromInt(t1.nsec - t0.nsec)) / 1e6;
        }
        d.step_ms = ms / @as(f64, @floatFromInt(hold_depth));
    }

    /// Greedy streams: the blocks and every step in one command buffer, each draft picked on the GPU for the next.
    fn holdChained(d: *Drafter, items: []const dr.Hold) !void {
        const tokens = d.chain_tokens.slice(u32, chain_streams * 8);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = d.target.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (items, 0..) |item, i| {
            const depth: usize = @min(item.depth, 7);
            const s = try d.slot(i);
            const lane = d.contexts.getPtr(item.stream).?;
            const held = Ref{ .buf = d.chain_tokens, .off = i * 8 * @sizeOf(u32) };
            tokens[i * 8] = item.pending;
            d.encodeChain(e, s, lane, held, item.pending, item.position - 1, depth);
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
        var greedy = items.len <= chain_streams;
        for (items) |item| greedy = greedy and item.sampling == null;
        if (greedy) return d.holdChained(items);
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

pub fn resolveTailTap(tap: st.Tap, at_pos: u32, lane_end: u64, tap_gen: u64) !?state.Tail {
    if (lane_end == at_pos) return null;
    if (lane_end + 1 != at_pos) return error.SnapshotOutOfStep;
    const buf = tap.buf orelse return error.SnapshotOutOfStep;
    if (tap.gen) |g| if (g != tap_gen) return error.StaleInkling2Taps;
    if (lane_end < tap.start) return error.SnapshotOutOfStep;
    const rows_before = std.math.cast(usize, lane_end - tap.start) orelse return error.SnapshotOutOfStep;
    if (rows_before >= tap.rows) return error.SnapshotOutOfStep;
    const delta = std.math.mul(usize, rows_before, state.tap_row_bytes) catch return error.SnapshotOutOfStep;
    const src_off = std.math.add(usize, tap.off, delta) catch return error.SnapshotOutOfStep;
    const src_end = std.math.add(usize, src_off, state.tap_row_bytes) catch return error.SnapshotOutOfStep;
    if (src_end > buf.length()) return error.SnapshotOutOfStep;
    return .{ .buf = buf, .off = src_off };
}

test "Mini taps and block depth match checkpoint section eight" {
    try std.testing.expectEqualSlices(u32, &.{ 5, 23, 41, 48 }, &tap_list);
    try std.testing.expectEqual(@as(u32, 48), tap_list[3]);
    try std.testing.expectEqual(@as(u64, 768 * 1032), Drafter.scoreBytes(1024));
}

test "resolveTailTap preserves non-first-stream tap offset and validates generation and bounds" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    const buf = try device.buffer(6 * state.tap_row_bytes, mtl.ResourceOptions.shared);
    defer buf.deinit();
    const shared_tap: st.Tap = .{ .buf = buf, .off = 2 * state.tap_row_bytes, .start = 10, .rows = 3, .gen = 7 };
    try std.testing.expectEqual(@as(?state.Tail, null), try resolveTailTap(shared_tap, 13, 13, 7));
    const tail = (try resolveTailTap(shared_tap, 13, 12, 7)).?;
    try std.testing.expectEqual(buf.id, tail.buf.id);
    try std.testing.expectEqual(4 * state.tap_row_bytes, tail.off);
    const stable_tap: st.Tap = .{ .buf = buf, .off = 2 * state.tap_row_bytes, .start = 10, .rows = 3, .gen = null };
    try std.testing.expectEqual(4 * state.tap_row_bytes, (try resolveTailTap(stable_tap, 13, 12, 99)).?.off);
    try std.testing.expectError(error.StaleInkling2Taps, resolveTailTap(shared_tap, 13, 12, 8));
    try std.testing.expectError(error.SnapshotOutOfStep, resolveTailTap(shared_tap, 14, 13, 7));
    try std.testing.expectError(error.SnapshotOutOfStep, resolveTailTap(shared_tap, 13, 11, 7));
    try std.testing.expectError(error.SnapshotOutOfStep, resolveTailTap(shared_tap, 10, 9, 7));
    const truncated: st.Tap = .{ .buf = buf, .off = 5 * state.tap_row_bytes + 1, .start = 10, .rows = 1, .gen = 7 };
    try std.testing.expectError(error.SnapshotOutOfStep, resolveTailTap(truncated, 11, 10, 7));
}
