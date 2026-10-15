//! z-lab's DFlash drafter on the target's taps: a stream's context keys, a block forward of up to 16 rows, held chains.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const cfg = @import("config.zig");
const dw = @import("dflash_weights.zig");
const lane = @import("../../core/lane_projection.zig");
const ops = @import("draft_ops");
const attention = @import("../../core/shared_attention.zig");
const round = @import("tree_round_gpu");
const weights = @import("weights.zig");
const fwd = @import("forward.zig");
const Model = @import("model.zig").Model;

const dr = lanes.drafter;
const Stream = lanes.Stream;
const Ref = lane.Ref;
const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const private = mtl.ResourceOptions.private | mtl.ResourceOptions.untracked;

/// A block's rows at most (the pending token and 15 drafts); the rows one batched hold or absorb pass takes.
pub const block_rows = 16;
const absorb_rows = lane.reg_rows;
const hold_rows = lane.reg_rows;
/// Streams one hold or absorb call takes, and host-written rows one absorb call takes before it flushes.
const max_items = 32;
const pos_rows = 1024;

pub const Options = struct { capacity: usize };

/// One stream's drafter state: context keys and values a layer, and the chain it holds.
const Lane = struct {
    k: [8]mtl.Buffer,
    v: [8]mtl.Buffer,
    capacity: usize, // positions a full layer's cache holds
    end: u64 = 0, // context rows absorbed: positions [0, end)
    held: [block_rows]u32 = @splat(0),
    held_len: u32 = 0,
};

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    target: *Model,
    config: dw.Config,
    w: dw.Weights,
    options: Options,
    ops: ops.Ops,
    attention: attention.Attention,
    picks: round.Ops,
    projections: weights.Projections = .{},
    lanes: std.AutoHashMapUnmanaged(*Stream, *Lane) = .empty,
    step_ms: f64 = 0,
    block_ms: [2]f64 = .{ 0, 0 }, // a 1-draft and a 15-draft block's wall time at load
    trace: bool = false, // GEMMA_DRAFT_TRACE: each hold's anchor, depth and drafts on stderr
    /// An absorb's command buffer, committed without a wait: the hold that follows (the same queue) runs after it.
    pending: ?mtl.CommandBuffer = null,
    // the block forward's scratch (one block at a time; items of a call run in order on one command buffer)
    ids: mtl.Buffer,
    pos: mtl.Buffer,
    apos: mtl.Buffer, // an absorb's positions, apart from a block's (its command buffer may still be running)
    xs: mtl.Buffer, // the next projection's group sums, written by the kernel that made its input
    mask: mtl.Buffer,
    ones: mtl.Buffer,
    pick_buf: mtl.Buffer,
    h: mtl.Buffer,
    x: mtl.Buffer,
    qkv: mtl.Buffer,
    q: mtl.Buffer,
    q2: mtl.Buffer,
    k16: mtl.Buffer,
    k16r: mtl.Buffer,
    attn: mtl.Buffer,
    o: mtl.Buffer,
    gate_up: mtl.Buffer,
    act: mtl.Buffer,
    logits: mtl.Buffer,
    // the absorb scratch: 128 context rows at a time
    fcx: mtl.Buffer,
    hctx: mtl.Buffer,
    kv: mtl.Buffer,
    kn: mtl.Buffer,
    knr: mtl.Buffer,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, target: *Model, dir: []const u8, options: Options) !*Drafter {
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = undefined;
        d.gpa = gpa;
        d.io = io;
        d.target = target;
        d.options = options;
        d.projections = .{};
        d.lanes = .empty;
        d.step_ms = 0;
        d.block_ms = .{ 0, 0 };
        d.pending = null;
        d.trace = std.c.getenv("GEMMA_DRAFT_TRACE") != null;
        errdefer d.projections.deinit(gpa); // warm's blocks fill it
        d.config = try dw.Config.read(gpa, io, dir);
        const c = &d.config;
        d.w = try dw.load(gpa, target.device, io, dir, c);
        errdefer d.w.deinit(gpa);
        d.ops = try ops.Ops.init(target.device);
        errdefer d.ops.deinit();
        const keys_max = @max(options.capacity + block_rows, c.window - 1 + block_rows);
        d.attention = try attention.Attention.init(gpa, target.device, .{ .query_heads = c.heads, .kv_heads = c.kv_heads, .head_dim = c.head_dim, .rotary_dim = c.head_dim, .capacity = keys_max, .output = .bf16 });
        errdefer d.attention.deinit();
        d.picks = try round.Ops.init(target.device);
        errdefer d.picks.deinit();
        const dev = target.device;
        const hid = c.hidden;
        const qw = c.heads * c.head_dim;
        const kw = c.kv_heads * c.head_dim;
        const r = hold_rows;
        var made: usize = 0;
        errdefer d.release(made);
        inline for (.{ .{ "ids", max_items * block_rows * 4, true }, .{ "pos", max_items * block_rows * 8, true }, .{ "apos", pos_rows * 8, true }, .{ "xs", (c.tap_count * hid / 64) * absorb_rows * 4, false }, .{ "mask", max_items * block_rows * (c.window - 1 + block_rows), true }, .{ "ones", block_rows * keys_max, true }, .{ "pick_buf", max_items * block_rows * @sizeOf(round.contract.Pick), true }, .{ "h", r * hid * 2, false }, .{ "x", r * hid * 2, false }, .{ "qkv", r * (qw + 2 * kw) * 2, false }, .{ "q", r * qw * 2, false }, .{ "q2", r * qw * 2, false }, .{ "k16", r * kw * 2, false }, .{ "k16r", r * kw * 2, false }, .{ "attn", r * qw * 2, false }, .{ "o", r * hid * 2, false }, .{ "gate_up", r * 2 * c.intermediate * 2, false }, .{ "act", r * c.intermediate * 2, false }, .{ "logits", r * cfg.vocab * 2, false }, .{ "fcx", absorb_rows * hid * 2, false }, .{ "hctx", absorb_rows * hid * 2, d.trace }, .{ "kv", absorb_rows * 2 * kw * 2, false }, .{ "kn", absorb_rows * kw * 2, false }, .{ "knr", absorb_rows * kw * 2, false } }) |f| {
            @field(d, f[0]) = try dev.buffer(f[1], if (f[2]) shared else private);
            made += 1;
        }
        @memset(d.ones.slice(u8, block_rows * keys_max), 0xff);
        try d.warm(io);
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        d.settle() catch {};
        var it = d.lanes.valueIterator();
        while (it.next()) |l| d.dropLane(l.*);
        d.lanes.deinit(d.gpa);
        d.release(std.math.maxInt(usize));
        d.projections.deinit(d.gpa);
        d.picks.deinit();
        d.attention.deinit();
        d.ops.deinit();
        d.w.deinit(d.gpa);
        d.gpa.destroy(d);
    }

    /// Releases the first `made` buffers in init's order, so a failed init frees what it made.
    fn release(d: *Drafter, made: usize) void {
        var i: usize = 0;
        inline for (.{ "ids", "pos", "apos", "xs", "mask", "ones", "pick_buf", "h", "x", "qkv", "q", "q2", "k16", "k16r", "attn", "o", "gate_up", "act", "logits", "fcx", "hctx", "kv", "kn", "knr" }) |f| {
            if (i < made) @field(d, f).deinit();
            i += 1;
        }
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(ptr: *anyopaque) []const u32 {
        const d = cast(ptr);
        return d.config.taps[0..d.config.tap_count];
    }

    /// Acceptance by depth, measured on the Python reference (114 rounds, 3.5 a round), until a stream has its own.
    const prior = [_]f64{ 0.77, 0.64, 0.71, 0.82, 0.84, 0.81, 0.9, 0.9, 0.9, 0.9, 0.83, 0.8, 0.83, 0.8, 0.8 };

    /// Chains of up to 15, always drafted, no plain guard: a block costs the same at any depth, a guard bills per row.
    fn factsFn(ptr: *anyopaque) dr.Facts {
        const d = cast(ptr);
        return .{ .depth = block_rows - 1, .step_ms = d.step_ms, .prior = &prior, .plain_guard = false, .batched = true, .stream_bytes = d.laneBytes(d.options.capacity) };
    }

    fn laneBytes(d: *const Drafter, capacity: usize) usize {
        const c = &d.config;
        var bytes: usize = 0;
        for (c.sliding[0..c.layers]) |sliding| bytes += 2 * (if (sliding) c.window - 1 + block_rows else capacity + block_rows) * c.kv_heads * c.head_dim * 2;
        return bytes;
    }

    fn openFn(ptr: *anyopaque, s: *Stream) anyerror!void {
        const d = cast(ptr);
        if (d.lanes.contains(s)) return error.DraftStreamOpen;
        const capacity = @min(d.options.capacity, (s.prompt_len + s.max_new + 2 * block_rows + 63) / 64 * 64);
        const l = try d.newLane(capacity);
        errdefer d.dropLane(l);
        try d.lanes.put(d.gpa, s, l);
    }

    fn newLane(d: *Drafter, capacity: usize) !*Lane {
        const c = &d.config;
        const l = try d.gpa.create(Lane);
        errdefer d.gpa.destroy(l);
        l.* = .{ .k = undefined, .v = undefined, .capacity = capacity };
        var made: usize = 0;
        errdefer for (0..made) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        };
        for (0..c.layers) |i| {
            const bytes = (if (c.sliding[i]) c.window - 1 + block_rows else capacity + block_rows) * c.kv_heads * c.head_dim * 2;
            l.k[i] = try d.target.device.buffer(bytes, private);
            l.v[i] = d.target.device.buffer(bytes, private) catch |e| {
                l.k[i].deinit();
                return e;
            };
            made += 1;
        }
        return l;
    }

    fn dropLane(d: *Drafter, l: *Lane) void {
        for (0..d.config.layers) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        }
        d.gpa.destroy(l);
    }

    fn releaseFn(ptr: *anyopaque, s: *Stream) void {
        const d = cast(ptr);
        if (d.lanes.fetchRemove(s)) |e| d.dropLane(e.value);
    }

    fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const d = cast(ptr);
        for (streams, out) |s, o| {
            const l = d.lanes.get(s) orelse return error.DraftStreamUnknown;
            if (o.len > l.held_len) return error.DraftHeldFewer;
            @memcpy(o, l.held[0..o.len]);
        }
    }

    /// A projection; `summed`: its input's group sums are in d.xs (stride rows), written by the input's own kernel.
    fn project(d: *Drafter, e: mtl.ComputeEncoder, l: dw.Linear, x: Ref, y: Ref, rows: usize, summed: bool) !void {
        if (l.bf16) return d.ops.linear(e, .{ .buf = x.buf, .off = x.off }, .{ .buf = l.w }, .{ .buf = y.buf, .off = y.off }, .{ .rows = @intCast(rows), .n = @intCast(l.n), .k = @intCast(l.k), .x_stride = @intCast(l.k), .w_stride = @intCast(l.k), .y_stride = @intCast(l.n) });
        const p = try d.projections.get(d.gpa, d.target.device, l.n, l.k);
        if (summed) return p.encodeSums(e, x, .{ .buf = l.w }, .{ .buf = l.sb }, .{ .ref = .{ .buf = d.xs }, .stride = @intCast(rows) }, y, @intCast(rows));
        try p.encode(e, x, .{ .buf = l.w }, .{ .buf = l.sb }, y, @intCast(rows));
    }

    /// A hidden-wide RMS norm (w times the normalized row, the target's kernel) writing the next projection's sums.
    fn normSums(d: *Drafter, e: mtl.ComputeEncoder, x: Ref, gain: Ref, y: Ref, rows: usize) void {
        e.setPipeline(d.target.kernels.sliding.rms);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(gain.buf, gain.off, 1);
        e.setBuffer(y.buf, y.off, 2);
        e.setValue(d.config.eps, 3);
        e.setBuffer(d.xs, 0, 4);
        e.setValue(@as(i32, @intCast(rows)), 5);
        e.dispatchGroups(.{ .width = rows }, .{ .width = 1024 });
    }

    /// Waits for a pending absorb's command buffer (a hold ordered after it on the queue has run it already).
    fn settle(d: *Drafter) !void {
        const cb = d.pending orelse return;
        d.pending = null;
        defer mtl.objc.release(cb.id); // retained when it was left pending (an autoreleased object otherwise)
        cb.wait();
        if (cb.failure()) |msg| return failed(msg);
    }

    fn norm(d: *Drafter, e: mtl.ComputeEncoder, x: Ref, gain: Ref, y: Ref, rows: usize, heads: u32, dim: u32, x_stride: u32, y_stride: u32) !void {
        try d.ops.norm(e, .{ .buf = x.buf, .off = x.off }, .{ .buf = gain.buf, .off = gain.off }, .{ .buf = y.buf, .off = y.off }, .{ .rows = @intCast(rows), .heads = heads, .dim = dim, .x_stride = x_stride, .y_stride = y_stride, .eps = d.config.eps });
    }

    /// Rows of `x` (pitch `stride` values) into cache slots from `first`: position % ring on a ring, else the position.
    fn store(d: *Drafter, e: mtl.ComputeEncoder, x: Ref, width: u32, stride: u32, cache: mtl.Buffer, rows: usize, first: u64, ring: u32) void {
        e.setPipeline(d.target.kernels.sliding.draft_store);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(cache, 0, 1);
        e.setValue([4]u32{ stride, width, @intCast(first), ring }, 2);
        e.dispatchThreads(.{ .width = width, .height = rows }, .{ .width = 256 });
    }

    fn add(d: *Drafter, e: mtl.ComputeEncoder, h: mtl.Buffer, y: mtl.Buffer, count: usize) void {
        e.setPipeline(d.target.kernels.sliding.draft_add);
        e.setBuffer(h, 0, 0);
        e.setBuffer(y, 0, 1);
        e.dispatchThreads(.{ .width = count }, .{ .width = 256 });
    }

    /// Up to 128 rows of one absorb pass: each piece's rows (one stream, one source) sit at `row` of the pass.
    const Piece = struct { lane: *Lane, src: Ref, rows: usize, p0: u64, row: usize };

    /// Rows [start, start + rows) replace the context from `start`, in passes of pieces; committed, not awaited.
    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        const c = &d.config;
        const t0 = std.Io.Clock.awake.now(d.io);
        defer if (d.trace) std.debug.print("absorb of {d} streams: {d:.2} ms\n", .{ items.len, @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(d.io)).toNanoseconds())) / 1e6 });
        try d.settle(); // the positions buffer is rewritten below
        var cb = d.target.queue.commandBuffer();
        var e = cb.compute(.serial);
        var used: usize = 0; // host position rows written this command buffer
        var pieces: [max_items]Piece = undefined;
        var count: usize = 0;
        var rows: usize = 0;
        for (items) |it| {
            const l = d.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
            const f = it.features;
            if (f.space != .device or f.dtype != .bf16 or f.row_bytes != c.tap_count * c.hidden * 2) return error.DraftFeaturesUnsupported;
            if (it.start > l.end or it.start + f.rows > l.capacity) return error.DraftContextGap;
            const src = mtl.Buffer{ .id = @ptrFromInt(f.buffer) };
            var at: usize = 0;
            while (at < f.rows) : (at += absorb_rows) {
                const n = @min(absorb_rows, f.rows - at);
                if (rows + n > absorb_rows or count == pieces.len) { // the pass is full: run it
                    if (used + rows > pos_rows) { // and the positions buffer with it: a new command buffer
                        e.end();
                        cb.commit();
                        cb.wait();
                        if (cb.failure()) |msg| return failed(msg);
                        cb = d.target.queue.commandBuffer();
                        e = cb.compute(.serial);
                        used = 0;
                    }
                    try d.absorbPass(e, pieces[0..count], rows, used);
                    used += rows;
                    count = 0;
                    rows = 0;
                }
                pieces[count] = .{ .lane = l, .src = .{ .buf = src, .off = f.offset + at * f.row_bytes }, .rows = n, .p0 = it.start + at, .row = rows };
                count += 1;
                rows += n;
            }
            if (d.trace) {
                const first = src.slice(u16, (f.offset + f.row_bytes) / 2)[f.offset / 2 ..][0..4];
                std.debug.print("absorb {s}: rows [{d}, {d}) replace from end {d}; first tap bf16 {any}\n", .{ it.stream.id, it.start, it.start + f.rows, l.end, first.* });
            }
            l.end = it.start + f.rows;
            l.held_len = 0;
        }
        if (count > 0) {
            if (used + rows > pos_rows) {
                e.end();
                cb.commit();
                cb.wait();
                if (cb.failure()) |msg| return failed(msg);
                cb = d.target.queue.commandBuffer();
                e = cb.compute(.serial);
                used = 0;
            }
            try d.absorbPass(e, pieces[0..count], rows, used);
        }
        e.end();
        cb.commit();
        _ = mtl.objc.retain(cb.id);
        d.pending = cb;
    }

    /// One pass: fc of each piece into its rows, then hidden_norm, kv, k_norm and RoPE over all rows, stores a piece.
    fn absorbPass(d: *Drafter, e: mtl.ComputeEncoder, pieces: []const Piece, rows: usize, pos_at: usize) !void {
        const c = &d.config;
        const kw = c.kv_heads * c.head_dim;
        const positions = d.apos.slice(u64, pos_rows)[pos_at..][0..rows];
        for (pieces) |pc| {
            for (positions[pc.row..][0..pc.rows], 0..) |*p, i| p.* = pc.p0 + i;
            try d.project(e, d.w.fc, pc.src, .{ .buf = d.fcx, .off = pc.row * c.hidden * 2 }, pc.rows, false);
        }
        const pos = Ref{ .buf = d.apos, .off = pos_at * 8 };
        d.normSums(e, .{ .buf = d.fcx }, d.w.hidden_norm, .{ .buf = d.hctx }, rows);
        for (0..c.layers) |i| {
            const lw = &d.w.layers[i];
            try d.project(e, lw.kv, .{ .buf = d.hctx }, .{ .buf = d.kv }, rows, true);
            try d.norm(e, .{ .buf = d.kv }, lw.k_norm, .{ .buf = d.kn }, rows, c.kv_heads, c.head_dim, 2 * kw, kw);
            try d.ops.rope(e, .{ .buf = d.kn }, .{ .buf = pos.buf, .off = pos.off }, .{ .buf = d.knr }, .{ .rows = @intCast(rows), .heads = c.kv_heads, .dim = c.head_dim, .x_stride = kw, .y_stride = kw, .theta = c.theta });
            const ring: u32 = if (c.sliding[i]) c.window - 1 else 0;
            for (pieces) |pc| {
                d.store(e, .{ .buf = d.knr, .off = pc.row * kw * 2 }, kw, kw, pc.lane.k[i], pc.rows, pc.p0, ring);
                d.store(e, .{ .buf = d.kv, .off = (pc.row * 2 * kw + kw) * 2 }, kw, 2 * kw, pc.lane.v[i], pc.rows, pc.p0, ring);
            }
        }
    }

    fn failed(msg: [*:0]const u8) anyerror {
        std.log.err("gemma4 drafter: command buffer failed: {s}", .{msg});
        return error.GpuFailed;
    }

    /// The sliding layers' mask for item `slot`: ring slots by the position each holds, the block causal.
    fn slidingMask(d: *Drafter, slot: usize, anchor: u64, rows: usize) Ref {
        const ring: u64 = d.config.window - 1;
        const keys = ring + rows;
        const base = slot * block_rows * (ring + block_rows);
        const m = d.mask.slice(u8, max_items * block_rows * (ring + block_rows))[base..][0 .. rows * keys];
        for (0..rows) |i| {
            const row = m[i * keys ..][0..keys];
            for (0..ring) |s| {
                const held = anchor > s; // a slot past the anchor was never written
                const p = if (held) anchor - 1 - ((anchor - 1 - s) % ring) else 0;
                row[s] = @intFromBool(held and p + ring >= anchor + i);
            }
            for (0..rows) |j| row[ring + j] = @intFromBool(j <= i);
        }
        return .{ .buf = d.mask, .off = base };
    }

    /// A stream's block in a batch: its rows [row, row + depth + 1) of the pass, its slot's host regions.
    const Block = struct { lane: *Lane, pending: u32, anchor: u64, depth: u32, slot: usize, row: usize };

    /// A batch of blocks [pending, mask x depth] through the drafter together, attention a block; picks a row.
    fn encodeBlocks(d: *Drafter, e: mtl.ComputeEncoder, blocks: []const Block, rows: usize) !void {
        const c = &d.config;
        const t = d.target;
        const hid = c.hidden;
        const qw = c.heads * c.head_dim;
        const kw = c.kv_heads * c.head_dim;
        const ids = d.ids.slice(u32, max_items * block_rows)[0..rows];
        const positions = d.pos.slice(u64, max_items * block_rows)[0..rows];
        var masks: [max_items]Ref = undefined;
        for (blocks, 0..) |b, i| {
            ids[b.row] = b.pending;
            @memset(ids[b.row + 1 ..][0..b.depth], c.mask);
            for (positions[b.row..][0 .. b.depth + 1], 0..) |*p, j| p.* = b.anchor + j;
            masks[i] = d.slidingMask(b.slot, b.anchor, b.depth + 1);
        }
        e.setPipeline(t.kernels.sliding.embed);
        e.setBuffer(d.ids, 0, 0);
        e.setBuffer(t.weights.embed_w.buf, t.weights.embed_w.off, 1);
        e.setBuffer(t.weights.embed_s.buf, t.weights.embed_s.off, 2);
        e.setBuffer(t.weights.embed_b.buf, t.weights.embed_b.off, 3);
        e.setBuffer(d.h, 0, 4);
        e.setValue(fwd.embedScale(), 5);
        e.dispatchThreads(.{ .width = hid / 2, .height = rows }, .{ .width = 256 });
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
        const pos = Ref{ .buf = d.pos };
        for (0..c.layers) |i| {
            const lw = &d.w.layers[i];
            const sliding = c.sliding[i];
            d.normSums(e, .{ .buf = d.h }, lw.input_norm, .{ .buf = d.x }, rows);
            try d.project(e, lw.qkv, .{ .buf = d.x }, .{ .buf = d.qkv }, rows, true);
            try d.norm(e, .{ .buf = d.qkv }, lw.q_norm, .{ .buf = d.q }, rows, c.heads, c.head_dim, qw + 2 * kw, qw);
            try d.norm(e, .{ .buf = d.qkv, .off = qw * 2 }, lw.k_norm, .{ .buf = d.k16 }, rows, c.kv_heads, c.head_dim, qw + 2 * kw, kw);
            try d.ops.rope(e, .{ .buf = d.q }, .{ .buf = pos.buf, .off = pos.off }, .{ .buf = d.q2 }, .{ .rows = @intCast(rows), .heads = c.heads, .dim = c.head_dim, .x_stride = qw, .y_stride = qw, .theta = c.theta });
            try d.ops.rope(e, .{ .buf = d.k16 }, .{ .buf = pos.buf, .off = pos.off }, .{ .buf = d.k16r }, .{ .rows = @intCast(rows), .heads = c.kv_heads, .dim = c.head_dim, .x_stride = kw, .y_stride = kw, .theta = c.theta });
            for (blocks, 0..) |b, bi| {
                const n = b.depth + 1;
                const first: u64 = if (sliding) c.window - 1 else b.anchor;
                d.store(e, .{ .buf = d.k16r, .off = b.row * kw * 2 }, kw, kw, b.lane.k[i], n, first, 0);
                d.store(e, .{ .buf = d.qkv, .off = (b.row * (qw + 2 * kw) + qw + kw) * 2 }, kw, qw + 2 * kw, b.lane.v[i], n, first, 0);
                const keys: usize = @intCast(first + n);
                const mask = if (sliding) masks[bi] else Ref{ .buf = d.ones };
                try d.attention.encode(e, n, keys, keys, scale, .{ .buf = d.q2, .off = b.row * qw * 2 }, .{ .buf = b.lane.k[i] }, .{ .buf = b.lane.v[i] }, .{ .buf = mask.buf, .off = mask.off }, .{ .buf = d.attn, .off = b.row * qw * 2 });
            }
            try d.project(e, lw.o, .{ .buf = d.attn }, .{ .buf = d.o }, rows, false);
            d.add(e, d.h, d.o, rows * hid);
            d.normSums(e, .{ .buf = d.h }, lw.post_attention, .{ .buf = d.x }, rows);
            try d.project(e, lw.gate_up, .{ .buf = d.x }, .{ .buf = d.gate_up }, rows, true);
            e.setPipeline(t.kernels.sliding.draft_swiglu);
            e.setBuffer(d.gate_up, 0, 0);
            e.setBuffer(d.act, 0, 1);
            e.setValue(@as(u32, c.intermediate), 2);
            e.setBuffer(d.xs, 0, 3);
            e.setValue(@as(i32, @intCast(rows)), 4);
            e.dispatchGroups(.{ .width = c.intermediate / 64, .height = rows }, .{ .width = 64 });
            try d.project(e, lw.down, .{ .buf = d.act }, .{ .buf = d.o }, rows, true);
            d.add(e, d.h, d.o, rows * hid);
        }
        d.normSums(e, .{ .buf = d.h }, d.w.norm, .{ .buf = d.x }, rows);
        const head = try t.projections.get(t.gpa, t.device, t.weights.head.n, t.weights.head.k);
        try head.encodeSums(e, .{ .buf = d.x }, .{ .buf = t.weights.head.w }, .{ .buf = t.weights.head.sb }, .{ .ref = .{ .buf = d.xs }, .stride = @intCast(rows) }, .{ .buf = d.logits }, @intCast(rows));
        var first: usize = 0;
        while (first < rows) : (first += 16) {
            const n = @min(16, rows - first);
            try d.picks.argmax(e, .{ .buf = d.logits, .off = first * cfg.vocab * 2 }, .{ .buf = d.pick_buf, .off = first * @sizeOf(round.contract.Pick) }, .{ .rows = @intCast(n), .vocab = cfg.vocab, .stride = cfg.vocab });
        }
    }

    /// Each stream's chain of `depth` drafts after `pending` at `position`, the streams in batches of up to 128 rows.
    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        if (items.len > max_items) return error.DraftTooManyStreams;
        const t0 = std.Io.Clock.awake.now(d.io);
        defer if (d.trace) std.debug.print("hold of {d} streams: {d:.2} ms\n", .{ items.len, @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(d.io)).toNanoseconds())) / 1e6 });
        var at: usize = 0;
        while (at < items.len) {
            var blocks: [max_items]Block = undefined;
            var count: usize = 0;
            var rows: usize = 0;
            while (at < items.len) : (at += 1) {
                const it = items[at];
                const l = d.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
                if (it.depth == 0 or it.depth >= block_rows or it.position == 0) return error.DraftDepth;
                const anchor = it.position - 1;
                if (l.end != anchor or anchor + block_rows > l.capacity + block_rows) return error.DraftContextGap;
                if (rows + it.depth + 1 > hold_rows) break;
                blocks[count] = .{ .lane = l, .pending = it.pending, .anchor = anchor, .depth = it.depth, .slot = count, .row = rows };
                count += 1;
                rows += it.depth + 1;
            }
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            try d.encodeBlocks(e, blocks[0..count], rows);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| return failed(msg);
            try d.settle();
            const picks = d.pick_buf.slice(round.contract.Pick, max_items * block_rows);
            for (blocks[0..count], items[at - count .. at]) |b, it| {
                for (picks[b.row + 1 ..][0..b.depth], 0..) |p, i| {
                    if (p.nonfinite != 0 or p.token >= cfg.vocab) return error.DraftNonfinite;
                    b.lane.held[i] = p.token;
                }
                b.lane.held_len = b.depth;
                if (d.trace) std.debug.print("draft {s} row {d} anchor {d} pending {d} depth {d}: {any}\n", .{ it.stream.id, b.row, b.anchor, b.pending, b.depth, b.lane.held[0..b.depth] });
            }
        }
    }

    /// Blocks on an empty context: the pipelines compiled, then a 1-draft and a 15-draft block timed (the second runs).
    fn warm(d: *Drafter, io: std.Io) !void {
        const l = try d.newLane(64);
        defer d.dropLane(l);
        var ms: [2]f64 = undefined;
        for ([_]u32{ block_rows - 1, 1, block_rows - 1, 1, block_rows - 1 }, 0..) |depth, i| {
            const t0 = std.Io.Clock.awake.now(io);
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            try d.encodeBlocks(e, &.{.{ .lane = l, .pending = 2, .anchor = 0, .depth = depth, .slot = 0, .row = 0 }}, depth + 1);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| return failed(msg);
            if (i >= 3) ms[i - 3] = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
        }
        d.block_ms = ms;
        d.step_ms = @max(ms[1] - ms[0], 0) / @as(f64, block_rows - 2);
    }
};
