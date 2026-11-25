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

const dr = lanes.drafter;
const Stream = lanes.Stream;
const Ref = k.Ref;
const tap_list = [_]u32{ 5, 23, 41, 48 };
const draft_prior = [_]f64{ 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55 };
const absorb_rows: usize = 256;
const max_chunks: usize = (w.window_size + w.block_size + 127) / 128;

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

const scratch_fields = [_][]const u8{
    "ids",  "resid", "x",  "xc",   "dyn", "q_raw", "k_raw",  "v_raw", "q",    "kb",  "vb",  "part",
    "attn", "y",     "yc", "gate", "up",  "act",   "logits", "hid",   "tval", "tid", "seq", "fcx",
    "hctx", "ck",    "cv",
};

fn scratchSizes() [scratch_fields.len]usize {
    const n = w.block_size;
    const d = n - 1;
    const a = absorb_rows;
    const bf = 2;
    return .{
        n * 4,
        n * w.hidden * 4,
        n * w.hidden * bf,
        n * w.hidden * bf,
        n * w.dyn_width * bf,
        n * w.query_width * bf,
        n * w.kv_width * bf,
        n * w.kv_width * bf,
        n * w.query_width * bf,
        n * w.kv_width * bf,
        n * w.kv_width * bf,
        n * w.q_heads * max_chunks * 132 * 4,
        n * w.query_width * bf,
        n * w.hidden * bf,
        n * w.hidden * bf,
        n * w.intermediate * bf,
        n * w.intermediate * bf,
        n * w.intermediate * bf,
        d * w.padded_vocab * bf,
        d * w.rank * bf,
        d * w.top_k * 4,
        d * w.top_k * 4,
        n * 4,
        a * w.hidden * bf,
        a * w.hidden * bf,
        a * w.kv_width * bf,
        a * w.kv_width * bf,
    };
}

pub fn scratchBytes() u64 {
    var total: u64 = 0;
    for (scratchSizes()) |s| total += s;
    return total;
}

pub fn streamBytes() u64 {
    return @as(u64, w.layers_n) * 2 * w.ring_size * w.kv_width * 2;
}

const Scratch = struct {
    ids: mtl.Buffer,
    resid: mtl.Buffer,
    x: mtl.Buffer,
    xc: mtl.Buffer,
    dyn: mtl.Buffer,
    q_raw: mtl.Buffer,
    k_raw: mtl.Buffer,
    v_raw: mtl.Buffer,
    q: mtl.Buffer,
    kb: mtl.Buffer,
    vb: mtl.Buffer,
    part: mtl.Buffer,
    attn: mtl.Buffer,
    y: mtl.Buffer,
    yc: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    act: mtl.Buffer,
    logits: mtl.Buffer,
    hid: mtl.Buffer,
    tval: mtl.Buffer,
    tid: mtl.Buffer,
    seq: mtl.Buffer,
    fcx: mtl.Buffer,
    hctx: mtl.Buffer,
    ck: mtl.Buffer,
    cv: mtl.Buffer,

    fn init(device: mtl.Device) !Scratch {
        @setEvalBranchQuota(10_000);
        const ns = scratchSizes();
        var s: Scratch = undefined;
        var made: usize = 0;
        errdefer inline for (scratch_fields, 0..) |f, i| if (i < made) @field(s, f).deinit();
        inline for (scratch_fields, 0..) |f, i| {
            @field(s, f) = try device.buffer(@max(ns[i], 16), mtl.ResourceOptions.shared);
            made += 1;
        }
        return s;
    }

    fn deinit(s: *Scratch) void {
        inline for (scratch_fields) |f| @field(s, f).deinit();
        s.* = undefined;
    }
};

const Lane = struct {
    k: [w.layers_n]mtl.Buffer,
    v: [w.layers_n]mtl.Buffer,
    from: u64 = 0,
    end: u64 = 0,
    held: [w.block_size - 1]u32 = @splat(0),
    held_len: u32 = 0,

    fn init(device: mtl.Device) !Lane {
        var l: Lane = .{ .k = undefined, .v = undefined };
        const bytes = w.ring_size * w.kv_width * 2;
        var made: usize = 0;
        errdefer for (0..made) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        };
        for (0..w.layers_n) |i| {
            l.k[i] = try device.buffer(bytes, mtl.ResourceOptions.private);
            l.v[i] = device.buffer(bytes, mtl.ResourceOptions.private) catch |err| {
                l.k[i].deinit();
                return err;
            };
            made += 1;
        }
        return l;
    }

    fn deinit(l: *Lane) void {
        for (0..w.layers_n) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        }
        l.* = undefined;
    }
};

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    target: *const Model,
    loaded: w.Loaded,
    pipelines: k.Pipelines,
    scratch: Scratch,
    contexts: std.AutoHashMapUnmanaged(*Stream, Lane) = .empty,
    pending: ?mtl.CommandBuffer = null,
    step_ms: f64 = 0,
    use_embed_norm: bool = true,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, target: *const Model, drafter_dir: []const u8, bits: u8) !*Drafter {
        if (bits != 0 and bits != 4) return error.UnsupportedDraftBits;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = .{
            .gpa = gpa,
            .target = target,
            .loaded = try w.load(gpa, io, device, drafter_dir, bits),
            .pipelines = undefined,
            .scratch = undefined,
        };
        errdefer d.loaded.deinit();
        d.pipelines = try k.Pipelines.load(device);
        errdefer d.pipelines.deinit();
        d.scratch = try Scratch.init(device);
        errdefer d.scratch.deinit();
        try d.warm();
        std.log.info("inkling2_loop DFlash2: loaded {s} (bits={d}, fixed_mib={d:.1}, block={d}, top_k={d}, block_ms={d:.2})", .{
            drafter_dir,
            if (bits == 4) @as(u8, 4) else 16,
            @as(f64, @floatFromInt(d.fixedBytes())) / (1 << 20),
            d.loaded.config.block,
            d.loaded.config.top_k,
            d.step_ms * 7.0,
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

    fn settle(d: *Drafter) !void {
        const cb = d.pending orelse return;
        d.pending = null;
        defer cb.release();
        try failed(cb, "absorb");
    }

    fn openFn(ptr: *anyopaque, stream: *Stream) anyerror!void {
        const d = cast(ptr);
        if (d.contexts.contains(stream)) return error.DraftStreamOpen;
        var lane = try Lane.init(d.target.device);
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

    fn project(d: *const Drafter, e: mtl.ComputeEncoder, linear: w.Linear, input: Ref, output: Ref, rows: usize) !void {
        if (d.loaded.copies) |*copies| {
            return copies.project(e, linear.key, linear.n, linear.k, input, output, rows);
        }
        fwd.dense(d.target, e, linear.bf16 orelse return error.DraftPreparationUnqualified, input, output, rows);
    }

    fn absorbRows(d: *Drafter, e: mtl.ComputeEncoder, lane: *const Lane, taps: Ref, start: u64, rows: usize) !void {
        const c = &d.loaded.config;
        const s = &d.scratch;
        try d.project(e, d.loaded.weights.fc, taps, at(s.fcx, 0), rows);
        k.norm16(&d.pipelines, e, at(s.fcx, 0), r(d.loaded.weights.hidden_norm), at(s.hctx, 0), .{ .rows = @intCast(rows), .width = c.hidden, .eps = c.eps, .add = 0 });
        for (d.loaded.weights.layers, 0..) |layer, i| {
            try d.project(e, layer.k_proj, at(s.hctx, 0), at(s.ck, 0), rows);
            try d.project(e, layer.v_proj, at(s.hctx, 0), at(s.cv, 0), rows);
            k.ctxKv(&d.pipelines, e, at(s.ck, 0), at(s.cv, 0), r(layer.k_norm), at(lane.k[i], 0), at(lane.v[i], 0), .{
                .rows = @intCast(rows),
                .pos0 = @intCast(start),
                .ring = @intCast(w.ring_size),
                .eps = c.eps,
                .theta = c.theta,
            });
        }
    }

    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        try d.settle();
        if (items.len == 0) return;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = d.target.queue.commandBuffer();
        const e = cb.compute(.serial);
        var ended = false;
        errdefer if (!ended) e.end();
        const row_bytes = w.taps_n * w.hidden * 2;
        const keep_window: u64 = w.window_size + w.block_size;
        for (items) |item| {
            const lane = d.contexts.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            const features = item.features;
            if (features.space != .device or features.dtype != .bf16 or features.row_bytes != row_bytes) return error.DraftFeaturesUnsupported;
            if (item.rows != null) return error.TreesNotBuilt;
            if (features.rows == 0 or features.rows > 1048576) return error.DraftFeaturesUnsupported;
            if (item.start < lane.end) return error.DraftContextGap;
            if (item.start > lane.end) lane.from = item.start;
            const end = item.start + features.rows;
            const keep_from = if (item.prompt) @max(item.start, item.stream.prompt_len -| keep_window) else item.start;
            if (keep_from > lane.from and keep_from > lane.end) lane.from = keep_from;
            const source = mtl.Buffer{ .id = @ptrFromInt(features.buffer) };
            var position = keep_from;
            while (position < end) {
                const rows: usize = @intCast(@min(absorb_rows, end - position));
                const offset = features.offset + @as(usize, @intCast(position - item.start)) * row_bytes;
                try d.absorbRows(e, lane, at(source, offset), position, rows);
                position += rows;
            }
            lane.end = end;
            lane.held_len = 0;
        }
        e.end();
        ended = true;
        cb.commit();
        d.pending = cb.retain();
    }

    fn encodeBlock(d: *Drafter, e: mtl.ComputeEncoder, lane: *const Lane, pending: u32, anchor: u64) !void {
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
            try d.project(e, layer.attn_conv.proj, at(s.x, 0), at(s.dyn, 0), n);
            k.conv(&d.pipelines, e, at(s.x, 0), at(s.dyn, 0), r(layer.attn_conv.base), at(s.xc, 0), .{ .rows = n, .width = c.hidden, .block = n, .groups = w.conv_groups, .side = 0 });
            try d.project(e, layer.q_proj, at(s.xc, 0), at(s.q_raw, 0), n);
            try d.project(e, layer.k_proj, at(s.xc, 0), at(s.k_raw, 0), n);
            try d.project(e, layer.v_proj, at(s.xc, 0), at(s.v_raw, 0), n);
            k.blockQkv(&d.pipelines, e, at(s.q_raw, 0), at(s.k_raw, 0), at(s.v_raw, 0), r(layer.q_norm), r(layer.k_norm), at(s.q, 0), at(s.kb, 0), at(s.vb, 0), .{ .rows = n, .block = n, .pos0 = @intCast(anchor), .eps = c.eps, .theta = c.theta });
            k.attend(&d.pipelines, e, at(s.q, 0), at(s.kb, 0), at(s.vb, 0), at(lane.k[i], 0), at(lane.v[i], 0), at(s.part, 0), at(s.attn, 0), attend_args);
            try d.project(e, layer.o_proj, at(s.attn, 0), at(s.y, 0), n);
            k.conv(&d.pipelines, e, at(s.y, 0), at(s.dyn, 0), r(layer.attn_conv.base), at(s.yc, 0), .{ .rows = n, .width = c.hidden, .block = n, .groups = w.conv_groups, .side = 1 });
            k.norm(&d.pipelines, e, at(s.resid, 0), at(s.yc, 0), r(layer.post_norm), at(s.x, 0), .{ .rows = n, .width = c.hidden, .eps = c.eps, .add = 1 });
            try d.project(e, layer.mlp_conv.proj, at(s.x, 0), at(s.dyn, 0), n);
            k.conv(&d.pipelines, e, at(s.x, 0), at(s.dyn, 0), r(layer.mlp_conv.base), at(s.xc, 0), .{ .rows = n, .width = c.hidden, .block = n, .groups = w.conv_groups, .side = 0 });
            try d.project(e, layer.gate_proj, at(s.xc, 0), at(s.gate, 0), n);
            try d.project(e, layer.up_proj, at(s.xc, 0), at(s.up, 0), n);
            k.siluMul(&d.pipelines, e, at(s.gate, 0), at(s.up, 0), at(s.act, 0), n * w.intermediate);
            try d.project(e, layer.down_proj, at(s.act, 0), at(s.y, 0), n);
            k.conv(&d.pipelines, e, at(s.y, 0), at(s.dyn, 0), r(layer.mlp_conv.base), at(s.yc, 0), .{ .rows = n, .width = c.hidden, .block = n, .groups = w.conv_groups, .side = 1 });
        }
        k.norm(&d.pipelines, e, at(s.resid, 0), at(s.yc, 0), r(d.loaded.weights.norm), at(s.x, 0), .{ .rows = n, .width = c.hidden, .eps = c.eps, .add = 1 });
        const draft_rows = n - 1;
        const x_draft = at(s.x, w.hidden * 2);
        fwd.dense(d.target, e, d.target.weights.unembed, x_draft, at(s.logits, 0), draft_rows);
        try d.project(e, d.loaded.weights.hidden_proj, x_draft, at(s.hid, 0), draft_rows);
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
            var ended = false;
            errdefer if (!ended) e.end();
            try d.encodeBlock(e, lane, item.pending, anchor);
            e.end();
            ended = true;
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
        var lane = try Lane.init(d.target.device);
        defer lane.deinit();
        var ms: f64 = 0;
        for (0..3) |run| {
            const t0 = now();
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            var ended = false;
            errdefer if (!ended) e.end();
            try d.encodeBlock(e, &lane, 2, 0);
            e.end();
            ended = true;
            cb.commit();
            try failed(cb, "warm");
            if (run == 2) ms = now() - t0;
        }
        d.step_ms = ms / @as(f64, @floatFromInt(w.block_size - 1));
    }
};

test "DFlash2 stream and scratch accounting fit fixed ring geometry" {
    try std.testing.expectEqual(@as(usize, 2176), w.ring_size);
    try std.testing.expectEqual(@as(u64, 5 * 2 * 2176 * 1024 * 2), streamBytes());
    try std.testing.expect(scratchBytes() > 0);
    const dev: mtl.Device = undefined;
    const target: *const Model = undefined;
    try std.testing.expectError(error.UnsupportedDraftBits, Drafter.init(std.testing.allocator, std.testing.io, dev, target, "unused", 8));
}
