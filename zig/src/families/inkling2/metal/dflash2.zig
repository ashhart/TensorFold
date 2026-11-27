//! Inkling 2 Mini DFlash 2 drafter on Metal: five Qwen3+conv layers, sliding 2048 context rings, top-16 path selector.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const sys = @import("sys");
const ck = @import("../../../core/checkpoint_metal.zig");
const Model = @import("model.zig").Model;
const fwd = @import("forward.zig");
const w = @import("dflash2_weights.zig");
const k = @import("dflash2_kernels.zig");
const state = @import("dflash2_state.zig");
const st = @import("state.zig");

const dr = lanes.drafter;
const Stream = lanes.Stream;
const Ref = k.Ref;
const tap_list = [_]u32{ 5, 23, 41, 48 };
const draft_prior = [_]f64{ 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55 };

pub const scratchBytes = state.scratchBytes;
pub const streamBytes = state.streamBytes;

fn at(buf: mtl.Buffer, off: usize) Ref {
    return .{ .buf = buf, .off = off };
}
fn r(t: ck.Tensor) Ref {
    return .{ .buf = t.buffer, .off = t.offset };
}

fn now() f64 {
    var ts: sys.c.timespec = undefined;
    _ = sys.c.clock_gettime(sys.c.CLOCK.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1000.0 + @as(f64, @floatFromInt(ts.nsec)) * 1e-6;
}

fn failed(cb: mtl.CommandBuffer, what: []const u8) !void {
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("inkling2_loop DFlash2 {s}: {s}", .{ what, msg });
        return error.GpuFailed;
    }
}

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    target: *const Model,
    loaded: w.Loaded,
    pipelines: k.Pipelines,
    scratch: state.Scratch,
    contexts: std.AutoHashMapUnmanaged(*Stream, state.Lane) = .empty,
    pending: ?mtl.CommandBuffer = null,
    step_ms: f64 = 0,
    use_embed_norm: bool = true,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, target: *const Model, drafter_dir: []const u8) !*Drafter {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = .{
            .gpa = gpa,
            .target = target,
            .loaded = try w.load(gpa, io, target, drafter_dir),
            .pipelines = undefined,
            .scratch = undefined,
        };
        errdefer d.loaded.deinit();
        d.pipelines = try k.Pipelines.load(device);
        errdefer d.pipelines.deinit();
        d.scratch = try state.Scratch.init(device);
        errdefer d.scratch.deinit();
        try d.warm();
        std.log.info("inkling2_loop DFlash2: loaded {s} (block={d}, top_k={d}, block_ms={d:.2})", .{
            drafter_dir, d.loaded.config.block, d.loaded.config.top_k, d.step_ms * 7.0,
        });
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        d.settle() catch {};
        var it = d.contexts.valueIterator();
        while (it.next()) |lane| lane.deinit();
        d.contexts.deinit(d.gpa);
        d.scratch.deinit();
        d.pipelines.deinit();
        d.loaded.deinit();
        d.gpa.destroy(d);
    }

    pub fn fixedBytes(d: *const Drafter) u64 {
        return d.loaded.fixedBytes() + scratchBytes();
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .kind = "DFlash2", .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    pub fn saveSnap(d: *Drafter, stream: *Stream, at_pos: u32, tap: st.Tap, tap_gen: u64) !?state.Snap {
        const lane = d.contexts.getPtr(stream) orelse return null;
        try d.settle();
        const tail = try state.resolveTailTap(tap, at_pos, lane.end, tap_gen);
        return try state.Snap.save(d.target.device, d.target.queue, lane, tail);
    }

    pub fn restoreSnap(d: *Drafter, stream: *Stream, snap: ?*const state.Snap, at_pos: usize) !void {
        const lane = d.contexts.getPtr(stream) orelse return;
        const s = snap orelse return error.DFlash2SnapshotMissing;
        const expected_end = std.math.add(u64, s.end, if (s.tail_off != null) @as(u64, 1) else 0) catch return error.DFlash2SnapshotGeometry;
        if (expected_end != at_pos or expected_end > state.max_context) return error.DFlash2SnapshotGeometry;
        try d.settle();
        try state.Snap.restore(lane, d.target.queue, s);
        if (s.tail_off) |tail_off| {
            errdefer state.clearLane(lane);
            const buf = s.buf orelse return error.DFlash2SnapshotGeometry;
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            d.absorbRows(e, lane, at(buf, tail_off), s.end, 1);
            e.end();
            cb.commit();
            try failed(cb, "restore tail");
            lane.end = s.end + 1;
            lane.held_len = 0;
        }
    }

    pub fn resetLane(d: *Drafter, stream: *Stream) void {
        d.settle() catch {};
        if (d.contexts.getPtr(stream)) |lane| state.clearLane(lane);
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(_: *anyopaque) []const u32 {
        return &tap_list;
    }

    fn factsFn(ptr: *anyopaque) dr.Facts {
        const d = cast(ptr);
        return .{ .fixed_block = true, .depth = w.block_size - 1, .step_ms = d.step_ms, .prior = &draft_prior, .plain_guard = false, .stream_bytes = @intCast(streamBytes()) };
    }

    pub fn settle(d: *Drafter) !void {
        const cb = d.pending orelse return;
        d.pending = null;
        defer cb.release();
        try failed(cb, "absorb");
    }

    fn openFn(ptr: *anyopaque, stream: *Stream) anyerror!void {
        const d = cast(ptr);
        if (d.contexts.contains(stream)) return error.DraftStreamOpen;
        var lane = try state.Lane.init(d.target.device);
        errdefer lane.deinit();
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

    fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const d = cast(ptr);
        for (streams, out) |stream, tokens| {
            const lane = d.contexts.getPtr(stream) orelse return error.DraftStreamUnknown;
            if (tokens.len > lane.held_len) return error.DraftHeldFewer;
            @memcpy(tokens, lane.held[0..tokens.len]);
        }
    }

    pub fn absorbRows(d: *Drafter, e: mtl.ComputeEncoder, lane: *const state.Lane, taps: Ref, start: u64, rows: usize) void {
        const c = &d.loaded.config;
        const s = &d.scratch;
        fwd.dense(d.target, e, d.loaded.weights.fc, taps, at(s.fcx, 0), rows);
        k.norm16(&d.pipelines, e, at(s.fcx, 0), r(d.loaded.weights.hidden_norm), at(s.hctx, 0), .{ .rows = @intCast(rows), .width = c.hidden, .eps = c.eps, .add = 0 });
        for (d.loaded.weights.layers, 0..) |layer, i| {
            fwd.dense(d.target, e, layer.k_proj, at(s.hctx, 0), at(s.ck, 0), rows);
            fwd.dense(d.target, e, layer.v_proj, at(s.hctx, 0), at(s.cv, 0), rows);
            k.ctxKv(&d.pipelines, e, at(s.ck, 0), at(s.cv, 0), r(layer.k_norm), at(lane.k[i], 0), at(lane.v[i], 0), .{
                .rows = @intCast(rows),
                .pos0 = @intCast(start),
                .ring = @intCast(w.ring_size),
                .eps = c.eps,
                .theta = c.theta,
            });
        }
    }

    fn promptHorizon(stream: *const Stream, start: u64) u64 {
        var horizon: u64 = stream.prompt_len;
        for (stream.reuse.marks) |m| {
            if (m > start and m < horizon) horizon = m;
        }
        return horizon;
    }

    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        try d.settle();
        if (items.len == 0) return;
        const row_bytes = w.taps_n * w.hidden * 2;
        const keep_window: u64 = w.window_size + w.block_size;
        for (items) |item| {
            const lane = d.contexts.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            const features = item.features;
            if (features.space != .device or features.dtype != .bf16 or features.row_bytes != row_bytes) return error.DraftFeaturesUnsupported;
            if (item.rows != null) return error.TreesNotBuilt;
            if (features.rows == 0 or features.rows > state.max_context) return error.DraftFeaturesUnsupported;
            if (item.start < lane.end or item.start > state.max_context or features.rows > state.max_context - item.start) return error.DraftContextGap;
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = d.target.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (items) |item| {
            const lane = d.contexts.getPtr(item.stream).?;
            const features = item.features;
            if (item.start > lane.end) lane.from = item.start;
            const end = item.start + features.rows;
            const horizon = if (item.prompt) promptHorizon(item.stream, item.start) else end;
            const keep_from = if (item.prompt) @max(item.start, horizon -| keep_window) else item.start;
            const next_from = @min(keep_from, end);
            if (next_from > lane.from and next_from > lane.end) lane.from = next_from;
            const source = mtl.Buffer{ .id = @ptrFromInt(features.buffer) };
            var position = keep_from;
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
        d.pending = cb.retain();
    }

    pub fn encodeBlock(d: *Drafter, e: mtl.ComputeEncoder, lane: *const state.Lane, pending: u32, anchor: u64) void {
        const c = &d.loaded.config;
        const s = &d.scratch;
        const n = w.block_size;
        const ids = s.ids.slice(u32, n);
        ids[0] = pending;
        for (1..n) |j| ids[j] = c.mask;
        s.seq.slice(u32, n)[0] = pending;
        k.embed(&d.pipelines, e, at(s.ids, 0), r(d.target.weights.embed), r(d.target.weights.embed_norm), at(s.resid, 0), .{
            .rows = n,
            .width = c.hidden,
            .use_norm = @intFromBool(d.use_embed_norm),
            .eps = c.eps,
        });
        const lo = @max(lane.from, (anchor + 1) -| c.window);
        const retained: u32 = @intCast(anchor - lo);
        const keys: u32 = retained + @as(u32, n);
        const chunks: u32 = (keys + 127) / 128;
        const attend_args: k.AttendArgs = .{
            .rows = n,
            .anchor = @intCast(anchor),
            .from = @intCast(lane.from),
            .lo = @intCast(lo),
            .retained = retained,
            .keys = keys,
            .window = c.window,
            .ring = @intCast(w.ring_size),
            .chunks = chunks,
            .scale = 1.0 / @sqrt(128.0),
        };
        for (d.loaded.weights.layers, 0..) |layer, i| {
            k.norm(&d.pipelines, e, at(s.resid, 0), at(s.yc, 0), r(layer.in_norm), at(s.x, 0), .{ .rows = n, .width = c.hidden, .eps = c.eps, .add = @intFromBool(i > 0) });
            fwd.dense(d.target, e, layer.attn_conv.proj, at(s.x, 0), at(s.dyn, 0), n);
            k.conv(&d.pipelines, e, at(s.x, 0), at(s.dyn, 0), r(layer.attn_conv.base), at(s.xc, 0), .{ .rows = n, .width = c.hidden, .block = n, .groups = w.conv_groups, .side = 0 });
            fwd.dense(d.target, e, layer.q_proj, at(s.xc, 0), at(s.q_raw, 0), n);
            fwd.dense(d.target, e, layer.k_proj, at(s.xc, 0), at(s.k_raw, 0), n);
            fwd.dense(d.target, e, layer.v_proj, at(s.xc, 0), at(s.v_raw, 0), n);
            k.blockQkv(&d.pipelines, e, at(s.q_raw, 0), at(s.k_raw, 0), at(s.v_raw, 0), r(layer.q_norm), r(layer.k_norm), at(s.q, 0), at(s.kb, 0), at(s.vb, 0), .{ .rows = n, .block = n, .pos0 = @intCast(anchor), .eps = c.eps, .theta = c.theta });
            k.attend(&d.pipelines, e, at(s.q, 0), at(s.kb, 0), at(s.vb, 0), at(lane.k[i], 0), at(lane.v[i], 0), at(s.part, 0), at(s.attn, 0), attend_args);
            fwd.dense(d.target, e, layer.o_proj, at(s.attn, 0), at(s.y, 0), n);
            k.conv(&d.pipelines, e, at(s.y, 0), at(s.dyn, 0), r(layer.attn_conv.base), at(s.yc, 0), .{ .rows = n, .width = c.hidden, .block = n, .groups = w.conv_groups, .side = 1 });
            k.norm(&d.pipelines, e, at(s.resid, 0), at(s.yc, 0), r(layer.post_norm), at(s.x, 0), .{ .rows = n, .width = c.hidden, .eps = c.eps, .add = 1 });
            fwd.dense(d.target, e, layer.mlp_conv.proj, at(s.x, 0), at(s.dyn, 0), n);
            k.conv(&d.pipelines, e, at(s.x, 0), at(s.dyn, 0), r(layer.mlp_conv.base), at(s.xc, 0), .{ .rows = n, .width = c.hidden, .block = n, .groups = w.conv_groups, .side = 0 });
            fwd.dense(d.target, e, layer.gate_proj, at(s.xc, 0), at(s.gate, 0), n);
            fwd.dense(d.target, e, layer.up_proj, at(s.xc, 0), at(s.up, 0), n);
            k.siluMul(&d.pipelines, e, at(s.gate, 0), at(s.up, 0), at(s.act, 0), n * w.intermediate);
            fwd.dense(d.target, e, layer.down_proj, at(s.act, 0), at(s.y, 0), n);
            k.conv(&d.pipelines, e, at(s.y, 0), at(s.dyn, 0), r(layer.mlp_conv.base), at(s.yc, 0), .{ .rows = n, .width = c.hidden, .block = n, .groups = w.conv_groups, .side = 1 });
        }
        k.norm(&d.pipelines, e, at(s.resid, 0), at(s.yc, 0), r(d.loaded.weights.norm), at(s.x, 0), .{ .rows = n, .width = c.hidden, .eps = c.eps, .add = 1 });
        const draft_rows = n - 1;
        const x_draft = at(s.x, w.hidden * 2);
        fwd.dense(d.target, e, d.target.weights.unembed, x_draft, at(s.logits, 0), draft_rows);
        fwd.dense(d.target, e, d.loaded.weights.hidden_proj, x_draft, at(s.hid, 0), draft_rows);
        k.topk16(&d.pipelines, e, at(s.logits, 0), at(s.tval, 0), at(s.tid, 0), .{ .rows = draft_rows, .vocab = w.valid_vocab, .k = c.top_k, .stride = w.padded_vocab });
        k.walk(&d.pipelines, e, at(s.tval, 0), at(s.tid, 0), at(s.hid, 0), r(d.loaded.weights.pred_cb), r(d.loaded.weights.succ_cb), at(s.seq, 0), .{ .slots = draft_rows, .rank = c.rank, .top = c.top_k, .vocab = w.valid_vocab, .softcap = c.softcap });
    }

    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        try d.settle();
        for (items) |item| {
            const lane = d.contexts.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            if (item.depth == 0 or item.position == 0) return error.DraftDepth;
            const anchor = item.position - 1;
            if (lane.end != anchor) return error.DraftContextGap;
            const depth: usize = @min(item.depth, w.block_size - 1);
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            d.encodeBlock(e, lane, item.pending, anchor);
            e.end();
            cb.commit();
            try failed(cb, "hold");
            const seq = d.scratch.seq.slice(u32, w.block_size);
            for (seq[1..][0..depth], lane.held[0..depth]) |tok, *dst| {
                if (tok >= w.valid_vocab) return error.DraftNonfinite;
                dst.* = tok;
            }
            lane.held_len = @intCast(depth);
        }
    }

    fn warm(d: *Drafter) !void {
        var lane = try state.Lane.init(d.target.device);
        defer lane.deinit();
        var ms: f64 = 0;
        for (0..3) |run| {
            const t0 = now();
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            d.encodeBlock(e, &lane, 2, 0);
            e.end();
            cb.commit();
            try failed(cb, "warm");
            if (run == 2) ms = now() - t0;
        }
        d.step_ms = ms / @as(f64, @floatFromInt(w.block_size - 1));
    }
};
