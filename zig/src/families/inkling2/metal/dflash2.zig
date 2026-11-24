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
pub const max_batch_streams: usize = k.max_batch;

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
    "attn", "y",     "yc", "gate", "up",  "act",   "logits", "hid",   "tval", "tid", "seq", "taps",
    "fcx",  "hctx",  "ck", "cv",
};

fn scratchSizes(max_streams: usize) [scratch_fields.len]usize {
    const streams = @max(@as(usize, 1), @min(max_streams, max_batch_streams));
    const n = streams * w.block_size;
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
        n * w.padded_vocab * bf,
        n * w.rank * bf,
        n * w.top_k * 4,
        n * w.top_k * 4,
        n * 4,
        if (streams > 1) a * w.taps_n * w.hidden * bf else 0,
        a * w.hidden * bf,
        a * w.hidden * bf,
        a * w.kv_width * bf,
        a * w.kv_width * bf,
    };
}

pub fn scratchBytes(max_streams: usize) u64 {
    var total: u64 = 0;
    for (scratchSizes(max_streams)) |s| total += s;
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
    taps: mtl.Buffer,
    fcx: mtl.Buffer,
    hctx: mtl.Buffer,
    ck: mtl.Buffer,
    cv: mtl.Buffer,

    fn init(device: mtl.Device, max_streams: usize) !Scratch {
        @setEvalBranchQuota(10_000);
        const ns = scratchSizes(max_streams);
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
    max_batch: usize,
    contexts: std.AutoHashMapUnmanaged(*Stream, Lane) = .empty,
    pending: ?mtl.CommandBuffer = null,
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
        std.log.info("inkling2_loop DFlash2: loaded {s} (bits={d}, fixed_mib={d:.1}, block={d}, top_k={d}, batch={d}, block_ms={d:.2})", .{
            drafter_dir,
            if (bits == 4) @as(u8, 4) else 16,
            @as(f64, @floatFromInt(d.fixedBytes())) / (1 << 20),
            d.loaded.config.block,
            d.loaded.config.top_k,
            d.max_batch,
            d.step_ms * 7.0,
        });
        return d;
    }

    fn initWithLoaded(gpa: std.mem.Allocator, target: *const Model, loaded: w.Loaded, max_streams: usize) !*Drafter {
        if (max_streams == 0 or max_streams > max_batch_streams) return error.InvalidDraftBatch;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .target = target, .loaded = loaded, .pipelines = undefined, .scratch = undefined, .max_batch = max_streams };
        d.pipelines = try k.Pipelines.load(target.device);
        errdefer d.pipelines.deinit();
        d.scratch = try Scratch.init(target.device, max_streams);
        errdefer d.scratch.deinit();
        try d.warm();
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
        return d.loaded.fixedBytes() + scratchBytes(d.max_batch);
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .kind = "DFlash2", .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    pub const testing = if (@import("builtin").is_test) struct {
        pub fn init(gpa: std.mem.Allocator, target: *const Model, loaded: w.Loaded, max_streams: usize) !*Drafter {
            return initWithLoaded(gpa, target, loaded, max_streams);
        }
        pub fn ring(d: *const Drafter, s: *Stream, layer: usize, is_v: bool) ?mtl.Buffer {
            const lane = d.contexts.getPtr(s) orelse return null;
            return if (is_v) lane.v[layer] else lane.k[layer];
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
        return .{ .fixed_block = true, .depth = w.block_size - 1, .step_ms = d.step_ms, .prior = &draft_prior, .plain_guard = false, .batched = true, .stream_bytes = @intCast(streamBytes()) };
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

    const AbsorbSeg = struct { lane: *const Lane, src: mtl.Buffer, src_off: usize, pos0: u64, rows: usize };
    const AbsorbWork = struct { lane: *Lane, prev_end: u64, start: u64, keep_from: u64, end: u64, src: mtl.Buffer, offset: usize };
    const Block = struct { lane: *const Lane, pending: u32, anchor: u64 };

    fn absorbBatch(d: *Drafter, e: mtl.ComputeEncoder, taps: Ref, segs: []const AbsorbSeg, total_rows: usize) !void {
        const c = &d.loaded.config;
        const s = &d.scratch;
        try d.project(e, d.loaded.weights.fc, taps, at(s.fcx, 0), total_rows);
        k.norm16(&d.pipelines, e, at(s.fcx, 0), r(d.loaded.weights.hidden_norm), at(s.hctx, 0), .{ .rows = @intCast(total_rows), .width = c.hidden, .eps = c.eps, .add = 0 });
        var ctx_segs: [max_batch_streams]k.CtxSeg = undefined;
        var row_off: u32 = 0;
        for (segs, 0..) |seg, bi| {
            ctx_segs[bi] = .{ .row0 = row_off, .rows = @intCast(seg.rows), .pos0 = @intCast(seg.pos0) };
            row_off += @intCast(seg.rows);
        }
        const ctx_args: k.CtxArgs = .{ .rows = @intCast(total_rows), .segs = @intCast(segs.len), .ring = @intCast(w.ring_size), .eps = c.eps, .theta = c.theta };
        for (d.loaded.weights.layers, 0..) |layer, i| {
            try d.project(e, layer.k_proj, at(s.hctx, 0), at(s.ck, 0), total_rows);
            try d.project(e, layer.v_proj, at(s.hctx, 0), at(s.cv, 0), total_rows);
            var rings: [max_batch_streams]k.Ring = undefined;
            for (segs, 0..) |seg, bi| rings[bi] = .{ .k = at(seg.lane.k[i], 0), .v = at(seg.lane.v[i], 0) };
            k.ctxKv(&d.pipelines, e, at(s.ck, 0), at(s.cv, 0), r(layer.k_norm), rings[0..segs.len], ctx_args, ctx_segs[0..segs.len]);
        }
    }

    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        try d.settle();
        if (items.len == 0) return;
        const row_bytes = w.taps_n * w.hidden * 2;
        const keep_window: u64 = w.window_size + w.block_size;
        var stack_work: [max_batch_streams]AbsorbWork = undefined;
        const work = if (items.len <= stack_work.len) stack_work[0..items.len] else try d.gpa.alloc(AbsorbWork, items.len);
        defer if (items.len > stack_work.len) d.gpa.free(work);
        var has_rows = false;
        for (items, work, 0..) |item, *wi, ii| {
            const lane = d.contexts.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            const features = item.features;
            if (features.space != .device or features.dtype != .bf16 or features.row_bytes != row_bytes) return error.DraftFeaturesUnsupported;
            if (item.rows != null) return error.TreesNotBuilt;
            if (features.rows == 0 or features.rows > 1048576) return error.DraftFeaturesUnsupported;
            var prev_end = lane.end;
            for (items[0..ii], work[0..ii]) |prev_it, prev_w| if (prev_it.stream == item.stream) {
                prev_end = prev_w.end;
            };
            if (item.start < prev_end) return error.DraftContextGap;
            const end = item.start + features.rows;
            const keep_from = if (item.prompt) @max(item.start, item.stream.prompt_len -| keep_window) else item.start;
            if (keep_from < end) has_rows = true;
            wi.* = .{ .lane = lane, .prev_end = prev_end, .start = item.start, .keep_from = keep_from, .end = end, .src = .{ .id = @ptrFromInt(features.buffer) }, .offset = features.offset };
        }
        if (has_rows) {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            var active_blit: ?mtl.BlitEncoder = null;
            var active_compute: ?mtl.ComputeEncoder = null;
            errdefer {
                if (active_blit) |b| b.end();
                if (active_compute) |e| e.end();
            }
            const seg_limit = @min(d.max_batch, max_batch_streams);
            var idx: usize = 0;
            var pos: u64 = work[0].keep_from;
            while (idx < work.len) {
                var segs: [max_batch_streams]AbsorbSeg = undefined;
                var seg_count: usize = 0;
                var total_rows: usize = 0;
                var contiguous = true;
                while (idx < work.len and seg_count < seg_limit and total_rows < absorb_rows) {
                    const wi = work[idx];
                    if (pos >= wi.end) {
                        idx += 1;
                        if (idx < work.len) pos = work[idx].keep_from;
                        continue;
                    }
                    var same_lane = false;
                    for (segs[0..seg_count]) |prev_s| if (prev_s.lane == wi.lane) {
                        same_lane = true;
                        break;
                    };
                    if (same_lane) break;
                    const take: usize = @intCast(@min(absorb_rows - total_rows, wi.end - pos));
                    const off = wi.offset + @as(usize, @intCast(pos - wi.start)) * row_bytes;
                    if (seg_count > 0) {
                        const prev = segs[seg_count - 1];
                        if (wi.src.id != prev.src.id or off != prev.src_off + prev.rows * row_bytes) contiguous = false;
                    }
                    segs[seg_count] = .{ .lane = wi.lane, .src = wi.src, .src_off = off, .pos0 = pos, .rows = take };
                    seg_count += 1;
                    total_rows += take;
                    pos += take;
                    if (pos >= wi.end) {
                        idx += 1;
                        if (idx < work.len) pos = work[idx].keep_from;
                    }
                }
                if (seg_count == 0) break;
                const taps_ref = if (contiguous) at(segs[0].src, segs[0].src_off) else blk: {
                    const b = cb.blit();
                    active_blit = b;
                    var dst_row: usize = 0;
                    for (segs[0..seg_count]) |seg| {
                        b.copy(seg.src, seg.src_off, d.scratch.taps, dst_row * row_bytes, seg.rows * row_bytes);
                        dst_row += seg.rows;
                    }
                    b.end();
                    active_blit = null;
                    break :blk at(d.scratch.taps, 0);
                };
                const e = cb.compute(.serial);
                active_compute = e;
                try d.absorbBatch(e, taps_ref, segs[0..seg_count], total_rows);
                e.end();
                active_compute = null;
            }
            cb.commit();
            d.pending = cb.retain();
        }
        for (work) |wi| {
            if (wi.start > wi.prev_end) wi.lane.from = wi.start;
            if (wi.keep_from > wi.lane.from and wi.keep_from > wi.prev_end) wi.lane.from = wi.keep_from;
            wi.lane.end = wi.end;
            wi.lane.held_len = 0;
        }
    }

    fn encodeBlocks(d: *Drafter, e: mtl.ComputeEncoder, blocks: []const Block) !void {
        std.debug.assert(blocks.len > 0 and blocks.len <= d.max_batch);
        const c = &d.loaded.config;
        const s = &d.scratch;
        const n = w.block_size;
        const rows = blocks.len * n;
        const ids = s.ids.slice(u32, rows);
        const seq = s.seq.slice(u32, rows);
        for (blocks, 0..) |blk, bi| {
            const row0 = bi * n;
            ids[row0] = blk.pending;
            @memset(ids[row0 + 1 ..][0 .. n - 1], c.mask);
            seq[row0] = blk.pending;
        }
        k.embed(&d.pipelines, e, at(s.ids, 0), r(d.target.weights.embed), r(d.target.weights.embed_norm), at(s.resid, 0), .{
            .rows = @intCast(rows),
            .width = c.hidden,
            .use_norm = @intFromBool(d.use_embed_norm),
            .eps = c.eps,
        });
        var block_qkv_args: k.BlockQkvArgs = .{ .rows = @intCast(rows), .block = @intCast(n), .eps = c.eps, .theta = c.theta, .pos0 = @splat(0) };
        var streams: [max_batch_streams]k.AttendStream = @splat(.{ .anchor = 0, .from = 0, .lo = 0, .retained = 0, .keys = 0, .chunks = 0 });
        var max_stream_chunks: u32 = 1;
        for (blocks, 0..) |blk, bi| {
            block_qkv_args.pos0[bi] = @intCast(blk.anchor);
            const lo = @max(blk.lane.from, (blk.anchor + 1) -| c.window);
            const retained: u32 = @intCast(blk.anchor - lo);
            const keys: u32 = retained + @as(u32, n);
            const chunks: u32 = (keys + 127) / 128;
            max_stream_chunks = @max(max_stream_chunks, chunks);
            streams[bi] = .{
                .anchor = @intCast(blk.anchor),
                .from = @intCast(blk.lane.from),
                .lo = @intCast(lo),
                .retained = retained,
                .keys = keys,
                .chunks = chunks,
            };
        }
        const attend_args: k.AttendArgs = .{ .rows = @intCast(rows), .block = @intCast(n), .window = c.window, .ring = @intCast(w.ring_size), .max_chunks = max_stream_chunks, .scale = 1.0 / @sqrt(128.0) };
        for (d.loaded.weights.layers, 0..) |layer, i| {
            k.norm(&d.pipelines, e, at(s.resid, 0), at(s.yc, 0), r(layer.in_norm), at(s.x, 0), .{ .rows = @intCast(rows), .width = c.hidden, .eps = c.eps, .add = @intFromBool(i > 0) });
            try d.project(e, layer.attn_conv.proj, at(s.x, 0), at(s.dyn, 0), rows);
            k.conv(&d.pipelines, e, at(s.x, 0), at(s.dyn, 0), r(layer.attn_conv.base), at(s.xc, 0), .{ .rows = @intCast(rows), .width = c.hidden, .block = n, .groups = w.conv_groups, .side = 0 });
            try d.project(e, layer.q_proj, at(s.xc, 0), at(s.q_raw, 0), rows);
            try d.project(e, layer.k_proj, at(s.xc, 0), at(s.k_raw, 0), rows);
            try d.project(e, layer.v_proj, at(s.xc, 0), at(s.v_raw, 0), rows);
            k.blockQkv(&d.pipelines, e, at(s.q_raw, 0), at(s.k_raw, 0), at(s.v_raw, 0), r(layer.q_norm), r(layer.k_norm), at(s.q, 0), at(s.kb, 0), at(s.vb, 0), block_qkv_args);
            var rings: [max_batch_streams]k.Ring = undefined;
            for (blocks, 0..) |blk, bi| rings[bi] = .{ .k = at(blk.lane.k[i], 0), .v = at(blk.lane.v[i], 0) };
            k.attend(&d.pipelines, e, at(s.q, 0), at(s.kb, 0), at(s.vb, 0), rings[0..blocks.len], at(s.part, 0), at(s.attn, 0), attend_args, streams[0..blocks.len]);
            try d.project(e, layer.o_proj, at(s.attn, 0), at(s.y, 0), rows);
            k.conv(&d.pipelines, e, at(s.y, 0), at(s.dyn, 0), r(layer.attn_conv.base), at(s.yc, 0), .{ .rows = @intCast(rows), .width = c.hidden, .block = n, .groups = w.conv_groups, .side = 1 });
            k.norm(&d.pipelines, e, at(s.resid, 0), at(s.yc, 0), r(layer.post_norm), at(s.x, 0), .{ .rows = @intCast(rows), .width = c.hidden, .eps = c.eps, .add = 1 });
            try d.project(e, layer.mlp_conv.proj, at(s.x, 0), at(s.dyn, 0), rows);
            k.conv(&d.pipelines, e, at(s.x, 0), at(s.dyn, 0), r(layer.mlp_conv.base), at(s.xc, 0), .{ .rows = @intCast(rows), .width = c.hidden, .block = n, .groups = w.conv_groups, .side = 0 });
            try d.project(e, layer.gate_proj, at(s.xc, 0), at(s.gate, 0), rows);
            try d.project(e, layer.up_proj, at(s.xc, 0), at(s.up, 0), rows);
            k.siluMul(&d.pipelines, e, at(s.gate, 0), at(s.up, 0), at(s.act, 0), @intCast(rows * w.intermediate));
            try d.project(e, layer.down_proj, at(s.act, 0), at(s.y, 0), rows);
            k.conv(&d.pipelines, e, at(s.y, 0), at(s.dyn, 0), r(layer.mlp_conv.base), at(s.yc, 0), .{ .rows = @intCast(rows), .width = c.hidden, .block = n, .groups = w.conv_groups, .side = 1 });
        }
        k.norm(&d.pipelines, e, at(s.resid, 0), at(s.yc, 0), r(d.loaded.weights.norm), at(s.x, 0), .{ .rows = @intCast(rows), .width = c.hidden, .eps = c.eps, .add = 1 });
        fwd.dense(d.target, e, d.target.weights.unembed, at(s.x, 0), at(s.logits, 0), rows);
        try d.project(e, d.loaded.weights.hidden_proj, at(s.x, 0), at(s.hid, 0), rows);
        k.topk16(&d.pipelines, e, at(s.logits, 0), at(s.tval, 0), at(s.tid, 0), .{ .rows = @intCast(rows), .vocab = w.valid_vocab, .k = c.top_k, .stride = w.padded_vocab });
        k.walk(&d.pipelines, e, at(s.tval, 0), at(s.tid, 0), at(s.hid, 0), r(d.loaded.weights.pred_cb), r(d.loaded.weights.succ_cb), at(s.seq, 0), .{ .slots = n - 1, .rank = c.rank, .top = c.top_k, .vocab = w.valid_vocab, .stride = n, .row0 = 1, .softcap = c.softcap }, @intCast(blocks.len));
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
        var at_idx: usize = 0;
        while (at_idx < items.len) {
            const count = @min(items.len - at_idx, d.max_batch);
            const pass = items[at_idx..][0..count];
            var blocks: [max_batch_streams]Block = undefined;
            for (pass, blocks[0..count]) |item, *blk| {
                blk.* = .{ .lane = d.contexts.getPtr(item.stream).?, .pending = item.pending, .anchor = item.position - 1 };
            }
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            var ended = false;
            errdefer if (!ended) e.end();
            try d.encodeBlocks(e, blocks[0..count]);
            e.end();
            ended = true;
            cb.commit();
            try failed(cb, "hold");
            const seq = d.scratch.seq.slice(u32, count * w.block_size);
            for (pass, 0..) |item, bi| {
                const depth: usize = @min(item.depth, w.block_size - 1);
                for (seq[bi * w.block_size + 1 ..][0..depth]) |tok| if (tok >= w.valid_vocab) return error.DraftNonfinite;
            }
            for (pass, 0..) |item, bi| {
                const lane = d.contexts.getPtr(item.stream).?;
                const depth: usize = @min(item.depth, w.block_size - 1);
                @memcpy(lane.held[0..depth], seq[bi * w.block_size + 1 ..][0..depth]);
                lane.held_len = @intCast(depth);
            }
            at_idx += count;
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
            try d.encodeBlocks(e, &.{.{ .lane = &lane, .pending = 2, .anchor = 0 }});
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
    try std.testing.expect(scratchBytes(1) > 0);
    try std.testing.expect(scratchBytes(4) > scratchBytes(1));
    try std.testing.expect(scratchBytes(max_batch_streams) >= scratchBytes(4));
    const dev: mtl.Device = undefined;
    const target: *const Model = undefined;
    try std.testing.expectError(error.UnsupportedDraftBits, Drafter.init(std.testing.allocator, std.testing.io, dev, target, "unused", 8));
    try std.testing.expectError(error.InvalidDraftBatch, Drafter.initFor(std.testing.allocator, std.testing.io, dev, target, "unused", 0, 0));
    try std.testing.expectError(error.InvalidDraftBatch, Drafter.initFor(std.testing.allocator, std.testing.io, dev, target, "unused", 0, max_batch_streams + 1));
}
