//! Inkling 2 Mini DFlash 2 drafter on Metal: five Qwen3+conv layers, 2048 rings, candidate trees, and fused rounds.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const sys = @import("sys");
const Model = @import("model.zig").Model;
const w = @import("dflash2_weights.zig");
const k = @import("dflash2_kernels.zig");
const bf16 = @import("dflash2_bf16.zig");
const st = @import("dflash2_state.zig");
const tree = @import("dflash2_tree.zig");
const enc = @import("dflash2_encode.zig");

const dr = lanes.drafter;
const Stream = lanes.Stream;
const tap_list = [_]u32{ 5, 23, 41, 48 };
const draft_prior = [_]f64{ 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55 };

pub const max_batch_streams = st.max_batch_streams;
pub const scratchBytes = st.scratchBytes;
pub const streamBytes = st.streamBytes;

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
    fast: bf16.Pipelines,
    scratch: st.Scratch,
    max_batch: usize,
    tree_nodes: usize = 0,
    contexts: std.AutoHashMapUnmanaged(*Stream, st.Lane) = .empty,
    pending: ?mtl.CommandBuffer = null,
    spec: st.SpecState = .{},
    landing: tree.Landing = .{},
    spec_hits: u64 = 0,
    spec_misses: u64 = 0,
    block_ms: f64 = 0,
    step_ms: f64 = 0,
    use_embed_norm: bool = true,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, target: *const Model, drafter_dir: []const u8, bits: u8) !*Drafter {
        return initFor(gpa, io, device, target, drafter_dir, bits, max_batch_streams);
    }

    pub fn initFor(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, target: *const Model, drafter_dir: []const u8, bits: u8, max_streams: usize) !*Drafter {
        if (bits != 0 and bits != 4) return error.UnsupportedDraftBits;
        if (max_streams == 0 or max_streams > max_batch_streams) return error.InvalidDraftBatch;
        var loaded = try w.load(gpa, io, device, drafter_dir, bits);
        errdefer loaded.deinit();
        const d = try initWithLoaded(gpa, target, loaded, max_streams);
        std.log.info("inkling2_loop DFlash2: loaded {s} (bits={d}, fixed_mib={d:.1}, block={d}, tree={d}, top_k={d}, batch={d}, block_ms={d:.2})", .{
            drafter_dir,
            if (bits == 4) @as(u8, 4) else 16,
            @as(f64, @floatFromInt(d.fixedBytes())) / (1 << 20),
            d.loaded.config.block,
            d.tree_nodes,
            d.loaded.config.top_k,
            d.max_batch,
            d.block_ms,
        });
        return d;
    }

    fn initWithLoaded(gpa: std.mem.Allocator, target: *const Model, loaded: w.Loaded, max_streams: usize) !*Drafter {
        if (max_streams == 0 or max_streams > max_batch_streams) return error.InvalidDraftBatch;
        const tree_nodes = try tree.nodes();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .target = target, .loaded = loaded, .pipelines = undefined, .fast = undefined, .scratch = undefined, .max_batch = max_streams, .tree_nodes = tree_nodes };
        d.pipelines = try k.Pipelines.load(target.device);
        errdefer d.pipelines.deinit();
        d.fast = try bf16.Pipelines.load(target.device);
        errdefer d.fast.deinit();
        d.scratch = try st.Scratch.init(target.device, max_streams);
        errdefer d.scratch.deinit();
        try d.warm();
        return d;
    }

    pub fn enableDraftHeadQ4(d: *Drafter, used_bytes: u64, limit_bytes: u64) !void {
        if (d.loaded.draft_head_q4 != null) return;
        try d.loaded.enableDraftHeadQ4(d.target, used_bytes, limit_bytes);
        if (d.loaded.draft_head_q4 != null) try d.warm();
    }

    pub fn deinit(d: *Drafter) void {
        d.settle() catch {};
        var it = d.contexts.valueIterator();
        while (it.next()) |lane| lane.deinit();
        d.contexts.deinit(d.gpa);
        d.scratch.deinit();
        d.fast.deinit();
        d.pipelines.deinit();
        d.loaded.deinit();
        d.gpa.destroy(d);
    }

    pub fn fixedBytes(d: *const Drafter) u64 {
        return d.loaded.fixedBytes() + scratchBytes(d.max_batch);
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{
            .kind = "DFlash2",
            .ptr = d,
            .vtable = &.{
                .taps = tapsFn,
                .facts = factsFn,
                .open = openFn,
                .absorb = absorbFn,
                .hold = holdFn,
                .draft_round = draftRoundFn,
                .speculate_early = speculateEarlyFn,
                .held = heldFn,
                .release = releaseFn,
                .tree = treeFn,
                .chances = chancesFn,
            },
        };
    }

    pub const testing = if (@import("builtin").is_test) struct {
        pub fn init(gpa: std.mem.Allocator, target: *const Model, loaded: w.Loaded, max_streams: usize) !*Drafter {
            return initWithLoaded(gpa, target, loaded, max_streams);
        }
        pub fn ring(d: *const Drafter, s: *Stream, layer: usize, is_v: bool) ?mtl.Buffer {
            const lane = d.contexts.getPtr(s) orelse return null;
            return if (is_v) lane.v[layer] else lane.k[layer];
        }
        pub fn benchmarkEncode(d: *Drafter, blocks_count: usize, reps: usize) !f64 {
            var stack_lanes: [max_batch_streams]st.Lane = undefined;
            var made: usize = 0;
            errdefer for (0..made) |i| stack_lanes[i].deinit();
            var blocks: [max_batch_streams]st.Block = undefined;
            for (0..blocks_count) |i| {
                stack_lanes[i] = try st.Lane.init(d.target.device);
                made += 1;
                blocks[i] = .{ .lane = &stack_lanes[i], .pending = 2, .anchor = 0, .from = 0 };
            }
            defer for (0..made) |i| stack_lanes[i].deinit();
            var best: f64 = std.math.inf(f64);
            for (0..3) |iter| {
                const t0 = now();
                const pool = mtl.objc.Pool.push();
                defer pool.pop();
                const cb = d.target.queue.commandBuffer();
                const e = cb.compute(.concurrent);
                var ended = false;
                errdefer if (!ended) e.end();
                for (0..reps) |_| {
                    try enc.encodeBlocks(d, e, blocks[0..blocks_count]);
                    e.barrier();
                }
                e.end();
                ended = true;
                cb.commit();
                try failed(cb, "bench");
                const per = (now() - t0) / @as(f64, @floatFromInt(reps));
                if (iter > 0 and per < best) best = per;
            }
            return best;
        }
    } else struct {};

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(_: *anyopaque) []const u32 {
        return &tap_list;
    }

    fn factsFn(ptr: *anyopaque) dr.Facts {
        const d = cast(ptr);
        return .{
            .fixed_block = true,
            .depth = w.block_size - 1,
            .step_ms = d.step_ms,
            .prior = &draft_prior,
            .plain_guard = false,
            .batched = true,
            .stream_bytes = @intCast(streamBytes()),
            .tree_nodes = @intCast(d.tree_nodes),
        };
    }

    fn settle(d: *Drafter) !void {
        var first_err: ?anyerror = null;
        if (d.pending) |cb| {
            d.pending = null;
            defer cb.release();
            failed(cb, "absorb") catch |err| {
                first_err = err;
            };
        }
        if (d.spec.cb) |cb| {
            d.spec.clear();
            defer cb.release();
            failed(cb, "spec_early") catch |err| {
                if (first_err == null) first_err = err;
            };
        }
        if (first_err) |err| return err;
    }

    fn openFn(ptr: *anyopaque, stream: *Stream) anyerror!void {
        const d = cast(ptr);
        if (d.contexts.contains(stream)) return error.DraftStreamOpen;
        var lane = try st.Lane.init(d.target.device);
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

    fn treeFn(ptr: *anyopaque, stream: *Stream, gpa: std.mem.Allocator) anyerror!?lanes.stream.Held {
        const d = cast(ptr);
        const lane = d.contexts.getPtr(stream) orelse return error.DraftStreamUnknown;
        if (lane.grown == 0) return null;
        const tokens = try gpa.dupe(u32, lane.held[0..lane.grown]);
        errdefer gpa.free(tokens);
        const parents = if (lanes.accept.isChain(lane.parents[0..lane.grown])) null else try gpa.dupe(i32, lane.parents[0..lane.grown]);
        return .{ .count = lane.grown, .tokens = tokens, .parents = parents };
    }

    fn chancesFn(ptr: *anyopaque, stream: *Stream, out: []f64) anyerror!bool {
        const d = cast(ptr);
        const lane = d.contexts.getPtr(stream) orelse return error.DraftStreamUnknown;
        if (lane.grown == 0 or out.len > lane.grown) return false;
        const sampled = if (stream.sampling) |s| s.temperature > 0 else false;
        for (out, lane.parents[0..out.len], lane.own[0..out.len]) |*o, parent, own| {
            o.* = d.landing.chance(sampled, own) * (if (parent >= 0) out[@intCast(parent)] else 1.0);
        }
        return true;
    }

    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        try d.settle();
        if (items.len == 0) return;
        var stack_work: [max_batch_streams]st.AbsorbWork = undefined;
        const work = if (items.len <= stack_work.len) stack_work[0..items.len] else try d.gpa.alloc(st.AbsorbWork, items.len);
        defer if (items.len > stack_work.len) d.gpa.free(work);
        if (try enc.prepareAbsorbs(d, items, work)) {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            try enc.encodeAbsorbs(d, cb, work);
            cb.commit();
            d.pending = cb.retain();
        }
        enc.commitAbsorbs(d, items, work);
    }

    fn finishHold(d: *Drafter, pass: []const dr.Hold) !void {
        const count = pass.len;
        if (d.tree_nodes > 0) {
            const total_rows = count * w.block_size;
            const tids = d.scratch.tid.slice(u32, total_rows * w.top_k);
            const tvals = d.scratch.tval.slice(f32, total_rows * w.top_k);
            const hid = d.scratch.hid.slice(u16, total_rows * w.rank);
            const pred_cb = d.loaded.weights.pred_cb.host(u16);
            const succ_cb = d.loaded.weights.succ_cb.host(u16);
            const draft_rows = w.block_size - 1;
            for (pass, 0..) |item, bi| {
                const lane = d.contexts.getPtr(item.stream).?;
                const row_base = bi * w.block_size + 1;
                const room: usize = if (item.stream.context.items.len >= item.stream.prompt_len)
                    @intCast(@max(0, item.stream.draftRoom() - 1))
                else
                    item.depth;
                const limit = @min(d.tree_nodes, @max(@as(usize, item.depth), @min(d.tree_nodes, room)));
                const sampled = if (item.sampling) |s| s.temperature > 0 else false;
                const allow_branch = sampled or !item.stream.shared_round;
                const c: tree.Candidates = .{
                    .tids = tids[row_base * w.top_k ..][0 .. draft_rows * w.top_k],
                    .tvals = tvals[row_base * w.top_k ..][0 .. draft_rows * w.top_k],
                    .hid = hid[row_base * w.rank ..][0 .. draft_rows * w.rank],
                    .topk = d.loaded.config.top_k,
                    .rank = d.loaded.config.rank,
                    .valid_vocab = w.valid_vocab,
                    .pred_cb = pred_cb,
                    .succ_cb = succ_cb,
                    .softcap = d.loaded.config.softcap,
                    .sampling = item.sampling,
                    .position = item.position,
                };
                const grown = tree.growWith(c, item.pending, draft_rows, limit, allow_branch, &lane.held, &lane.parents, &lane.own);
                if (grown == 0) return error.DraftNonfinite;
                lane.grown = @intCast(grown);
                lane.held_len = @intCast(grown);
            }
            return;
        }
        const seq = d.scratch.seq.slice(u32, count * w.block_size);
        for (pass, 0..) |item, bi| {
            const depth: usize = @min(item.depth, w.block_size - 1);
            for (seq[bi * w.block_size + 1 ..][0..depth]) |tok| if (tok >= w.valid_vocab) return error.DraftNonfinite;
        }
        for (pass, 0..) |item, bi| {
            const lane = d.contexts.getPtr(item.stream).?;
            const depth: usize = @min(item.depth, w.block_size - 1);
            @memcpy(lane.held[0..depth], seq[bi * w.block_size + 1 ..][0..depth]);
            lane.grown = 0;
            lane.held_len = @intCast(depth);
        }
    }

    fn runHolds(d: *Drafter, items: []const dr.Hold) !void {
        var at_idx: usize = 0;
        while (at_idx < items.len) {
            const count = @min(items.len - at_idx, d.max_batch);
            const pass = items[at_idx..][0..count];
            var blocks: [max_batch_streams]st.Block = undefined;
            for (pass, blocks[0..count]) |item, *blk| {
                const lane = d.contexts.getPtr(item.stream).?;
                blk.* = .{ .lane = lane, .pending = item.pending, .anchor = item.position - 1, .from = lane.from };
            }
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.concurrent);
            var ended = false;
            errdefer if (!ended) e.end();
            try enc.encodeBlocks(d, e, blocks[0..count]);
            e.end();
            ended = true;
            cb.commit();
            try failed(cb, "hold");
            try d.finishHold(pass);
            at_idx += count;
        }
    }

    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        try d.settle();
        if (items.len == 0) return;
        for (items) |item| {
            const lane = d.contexts.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            if (item.depth == 0 or item.position == 0) return error.DraftDepth;
            if (lane.end != item.position - 1) return error.DraftContextGap;
        }
        try d.runHolds(items);
    }

    fn draftRoundFn(ptr: *anyopaque, absorbs: []const dr.Absorb, holds: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        if (d.spec.cb) |cb| {
            if (d.spec.matches(absorbs, holds)) {
                d.spec.clear();
                defer cb.release();
                var stack_work: [max_batch_streams]st.AbsorbWork = undefined;
                _ = try enc.prepareAbsorbs(d, absorbs, stack_work[0..absorbs.len]);
                try failed(cb, "spec_early");
                enc.commitAbsorbs(d, absorbs, stack_work[0..absorbs.len]);
                try d.finishHold(holds);
                d.spec_hits += 1;
                return;
            }
            d.spec_misses += 1;
        }
        try d.settle();
        if (holds.len == 0) return absorbFn(ptr, absorbs);
        if (absorbs.len == 0) return holdFn(ptr, holds);
        var stack_work: [max_batch_streams]st.AbsorbWork = undefined;
        const work = if (absorbs.len <= stack_work.len) stack_work[0..absorbs.len] else try d.gpa.alloc(st.AbsorbWork, absorbs.len);
        defer if (absorbs.len > stack_work.len) d.gpa.free(work);
        const has_rows = try enc.prepareAbsorbs(d, absorbs, work);
        for (holds) |item| {
            const lane = d.contexts.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            if (item.depth == 0 or item.position == 0) return error.DraftDepth;
            if (enc.postAbsorb(absorbs, work, item.stream, lane).end != item.position - 1) return error.DraftContextGap;
        }
        const first = @min(holds.len, d.max_batch);
        var blocks: [max_batch_streams]st.Block = undefined;
        for (holds[0..first], blocks[0..first]) |item, *blk| {
            const lane = d.contexts.getPtr(item.stream).?;
            const eff = enc.postAbsorb(absorbs, work, item.stream, lane);
            blk.* = .{ .lane = lane, .pending = item.pending, .anchor = item.position - 1, .from = eff.from };
        }
        {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            if (has_rows) try enc.encodeAbsorbs(d, cb, work);
            const e = cb.compute(.concurrent);
            var ended = false;
            errdefer if (!ended) e.end();
            try enc.encodeBlocks(d, e, blocks[0..first]);
            e.end();
            ended = true;
            cb.commit();
            try failed(cb, "draft_round");
        }
        enc.commitAbsorbs(d, absorbs, work);
        try d.finishHold(holds[0..first]);
        if (holds.len > first) try d.runHolds(holds[first..]);
    }

    fn speculateEarlyFn(ptr: *anyopaque, absorbs: []const dr.Absorb, holds: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        try d.settle();
        if (holds.len == 0 or absorbs.len > d.max_batch or holds.len > d.max_batch) return;
        var work: [max_batch_streams]st.AbsorbWork = undefined;
        for (absorbs, 0..) |it, i| {
            const lane = d.contexts.getPtr(it.stream) orelse return;
            if (it.prompt or it.start != lane.end) return;
            for (absorbs[0..i]) |prev| if (prev.stream == it.stream) return;
        }
        const has_rows = enc.prepareAbsorbs(d, absorbs, work[0..absorbs.len]) catch return;
        var blocks: [max_batch_streams]st.Block = undefined;
        for (holds, blocks[0..holds.len]) |item, *blk| {
            const lane = d.contexts.getPtr(item.stream) orelse return;
            if (item.depth == 0 or item.position == 0 or item.lanes != null) return;
            const eff = enc.postAbsorb(absorbs, work[0..absorbs.len], item.stream, lane);
            if (eff.end != item.position - 1) return;
            blk.* = .{ .lane = lane, .pending = item.pending, .anchor = item.position - 1, .from = eff.from };
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = d.target.queue.commandBuffer();
        if (has_rows) try enc.encodeAbsorbs(d, cb, work[0..absorbs.len]);
        const e = cb.compute(.concurrent);
        var ended = false;
        errdefer if (!ended) e.end();
        try enc.encodeBlocks(d, e, blocks[0..holds.len]);
        e.end();
        ended = true;
        cb.commit();
        d.spec.record(cb.retain(), absorbs, holds);
    }

    fn warm(d: *Drafter) !void {
        var lane = try st.Lane.init(d.target.device);
        defer lane.deinit();
        var ms: f64 = 0;
        for (0..3) |run| {
            const t0 = now();
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.concurrent);
            var ended = false;
            errdefer if (!ended) e.end();
            try enc.encodeBlocks(d, e, &.{.{ .lane = &lane, .pending = 2, .anchor = 0, .from = 0 }});
            e.end();
            ended = true;
            cb.commit();
            try failed(cb, "warm");
            if (run == 2) ms = now() - t0;
        }
        d.block_ms = ms;
        const denom: f64 = @floatFromInt(if (d.tree_nodes > 0) d.tree_nodes else w.block_size - 1);
        d.step_ms = ms / denom;
    }
};

test "DFlash2 stream and scratch accounting fit fixed ring geometry" {
    try std.testing.expectEqual(@as(usize, 2176), w.ring_size);
    try std.testing.expectEqual(@as(u64, 5 * 2 * 2176 * 1024 * 2), streamBytes());
    try std.testing.expect(scratchBytes(1) > 0);
    try std.testing.expect(scratchBytes(4) > scratchBytes(1));
    try std.testing.expect(scratchBytes(max_batch_streams) >= scratchBytes(4));
    const dev: mtl.Device = undefined;
    const target: *const Model = undefined;
    try std.testing.expectError(error.UnsupportedDraftBits, Drafter.init(std.testing.allocator, std.testing.io, dev, target, "unused", 8));
    try std.testing.expectError(error.InvalidDraftBatch, Drafter.initFor(std.testing.allocator, std.testing.io, dev, target, "unused", 0, 0));
    try std.testing.expectError(error.InvalidDraftBatch, Drafter.initFor(std.testing.allocator, std.testing.io, dev, target, "unused", 0, max_batch_streams + 1));
}
