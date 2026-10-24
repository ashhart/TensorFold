//! z-lab's DFlash drafter for the 31B on CUDA (#1048 step 11): the target's taps as context keys, blocks of 16 rows.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const lanes = @import("lanes");
const cfg = @import("cuda_config.zig");
const fwd = @import("cuda_forward.zig");
const kern = @import("cuda_kernels.zig");
const be = @import("cuda_backend.zig");
const wts = @import("cuda_weights.zig");
const snap = @import("cuda_snap.zig");
const aw = @import("cuda_assistant_weights.zig");

const dr = lanes.drafter;
const Stream = lanes.Stream;
const Affine4 = cuda.qlinear.Affine4;
const Pick = cuda.tree_round.Pick;
const DeviceBuffer = cuda.DeviceBuffer;

/// A block's rows (the pending token and 15 drafts), and the blocks one hold runs together.
pub const block_rows = 16;
const max_blocks = 8;
const rows_most = block_rows * max_blocks;
/// Context rows one absorb pass takes (a prompt chunk's).
const absorb_most = fwd.prompt_rows;
const dim = 128;

/// The stock drafter's geometry (z-lab/gemma-4-31B-it-DFlash), admitted as Metal's dflash_weights.Config.
pub const Config = struct {
    taps: [8]u32 = @splat(0),
    tap_count: usize = 0,
    layers: usize,
    hidden: usize,
    intermediate: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block: usize,
    mask: u32,
    window: usize,
    eps: f32,
    theta: f64,
    /// Each layer's kind: true for a sliding layer (causal within the block), false for a full one.
    sliding: [8]bool = @splat(true),

    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, geo: cfg.Geometry) !Config {
        var parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        defer parsed.deinit();
        const o = if (parsed.value == .object) parsed.value.object else return error.BadDraftConfig;
        const archs = o.get("architectures") orelse return error.BadDraftConfig;
        if (archs != .array or archs.array.items.len != 1 or archs.array.items[0] != .string or !std.mem.eql(u8, archs.array.items[0].string, "DFlashDraftModel")) return error.NotDFlashV1;
        const d = if (o.get("dflash_config")) |v| (if (v == .object) v.object else return error.BadDraftConfig) else return error.BadDraftConfig;
        var c: Config = .{ .layers = @intCast(try integer(o, "num_hidden_layers")), .hidden = @intCast(try integer(o, "hidden_size")), .intermediate = @intCast(try integer(o, "intermediate_size")), .heads = @intCast(try integer(o, "num_attention_heads")), .kv_heads = @intCast(try integer(o, "num_key_value_heads")), .head_dim = @intCast(try integer(o, "head_dim")), .block = @intCast(if (d.get("block_size") != null) try integer(d, "block_size") else try integer(o, "block_size")), .mask = @intCast(try integer(d, "mask_token_id")), .window = @intCast(try integer(o, "sliding_window")), .eps = @floatCast(try number(o, "rms_norm_eps")), .theta = try number(o, "rope_theta") };
        if (c.hidden != geo.hidden) return error.DraftForAnotherTarget;
        const taps = d.get("target_layer_ids") orelse return error.BadDraftConfig;
        if (taps != .array or taps.array.items.len == 0 or taps.array.items.len > 8) return error.BadDraftConfig;
        for (taps.array.items, 0..) |t, i| {
            if (t != .integer or t.integer < 0) return error.BadDraftConfig;
            if (t.integer >= geo.layers) return error.DraftForAnotherTarget;
            c.taps[i] = @intCast(t.integer);
        }
        c.tap_count = taps.array.items.len;
        if (c.layers == 0 or c.layers > 8) return error.BadDraftConfig;
        if (o.get("layer_types")) |kinds| {
            if (kinds != .array or kinds.array.items.len != c.layers) return error.BadDraftConfig;
            for (kinds.array.items, 0..) |k, i| c.sliding[i] = k == .string and std.mem.eql(u8, k.string, "sliding_attention");
        }
        if (o.get("is_causal") != null) return error.UnsupportedDraftMask; // the vendored default decides the block's mask
        if (c.head_dim != dim or c.heads % c.kv_heads != 0 or c.heads / c.kv_heads > 8 or c.block != block_rows or c.window < 32) return error.UnsupportedDraftShape;
        if (c.hidden % 64 != 0 or c.intermediate % 64 != 0 or (c.heads * c.head_dim) % 64 != 0 or (c.kv_heads * c.head_dim) % 64 != 0) return error.UnsupportedDraftShape;
        if (c.mask >= cfg.vocab) return error.BadDraftConfig;
        return c;
    }
};

fn integer(o: std.json.ObjectMap, key: []const u8) !i64 {
    const v = o.get(key) orelse return error.BadDraftConfig;
    return if (v == .integer) v.integer else error.BadDraftConfig;
}

fn number(o: std.json.ObjectMap, key: []const u8) !f64 {
    const v = o.get(key) orelse return error.BadDraftConfig;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadDraftConfig,
    };
}

/// Whether `dir` holds a DFlash drafter (its config's single architecture).
pub fn isDFlash(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !bool {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    return std.mem.indexOf(u8, bytes, "\"DFlashDraftModel\"") != null;
}

const Layer = struct { q: Affine4, k: Affine4, v: Affine4, o: Affine4, gate: Affine4, up: Affine4, down: Affine4, input_norm: u64, post_attention: u64, q_norm: u64, k_norm: u64 };

/// The drafter's bf16 linears made MLX affine 4-bit on the host (Metal's default), norms as stored.
const Weights = struct {
    fc: Affine4,
    hidden_norm: u64,
    norm: u64,
    layers: [8]Layer,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
};

fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, k: *const kern.Kernels, dir: []const u8, c: *const Config, w: *Weights) !void {
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    var x: aw.Loader = .{ .gpa = gpa, .d = d, .s = s, .k = k, .ck = &ck, .bufs = &w.buffers };
    const h = c.hidden;
    const qw = c.heads * c.head_dim;
    const kw = c.kv_heads * c.head_dim;
    w.fc = try x.affine4("fc.weight", h, c.tap_count * h);
    w.hidden_norm = try x.vector("hidden_norm.weight", h);
    w.norm = try x.vector("norm.weight", h);
    var buf: [96]u8 = undefined;
    for (w.layers[0..c.layers], 0..) |*l, i| {
        const name = struct {
            fn of(b: []u8, layer: usize, suffix: []const u8) ![]const u8 {
                return std.fmt.bufPrint(b, "layers.{d}.{s}", .{ layer, suffix });
            }
        }.of;
        l.q = try x.affine4(try name(&buf, i, "self_attn.q_proj.weight"), qw, h);
        l.k = try x.affine4(try name(&buf, i, "self_attn.k_proj.weight"), kw, h);
        l.v = try x.affine4(try name(&buf, i, "self_attn.v_proj.weight"), kw, h);
        l.o = try x.affine4(try name(&buf, i, "self_attn.o_proj.weight"), h, qw);
        l.gate = try x.affine4(try name(&buf, i, "mlp.gate_proj.weight"), c.intermediate, h);
        l.up = try x.affine4(try name(&buf, i, "mlp.up_proj.weight"), c.intermediate, h);
        l.down = try x.affine4(try name(&buf, i, "mlp.down_proj.weight"), h, c.intermediate);
        l.input_norm = try x.vector(try name(&buf, i, "input_layernorm.weight"), h);
        l.post_attention = try x.vector(try name(&buf, i, "post_attention_layernorm.weight"), h);
        l.q_norm = try x.vector(try name(&buf, i, "self_attn.q_norm.weight"), dim);
        l.k_norm = try x.vector(try name(&buf, i, "self_attn.k_norm.weight"), dim);
    }
}

/// One stream's context: each layer's keys then values, [kv heads, capacity, 128] bf16; the chain it holds.
const Lane = struct {
    kv: DeviceBuffer,
    capacity: usize,
    end: u64 = 0, // context rows absorbed: positions [0, end)
    held: [block_rows]u32 = @splat(0),
    held_len: u32 = 0,
};

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    m: *fwd.Model,
    k: *const kern.Kernels,
    gb: *be.Gemma,
    s: cuda.Stream,
    c: Config,
    w: Weights,
    ba: cuda.block_attention.BlockAttention,
    lanes: std.AutoHashMapUnmanaged(*Stream, *Lane) = .empty,
    step_ms: f64 = 0,
    costs: [be.max_window]lanes.config.Cost = undefined,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    x: Scratch = undefined,

    const Scratch = struct { inv: u64, ids: u64, pos: u64, table: u64, lens: u64, picks: u64, xs: u64, fcx: u64, h: [2]u64, hs: u64, normed: u64, q: u64, kk: u64, vv: u64, qh: u64, kb: u64, vb: u64, part: u64, att: u64, o: u64, gate: u64, up: u64, act: u64, y: u64, logits: u64, junk: u64 };

    pub fn init(gpa: std.mem.Allocator, io: std.Io, gb: *be.Gemma, dir: []const u8) !*Drafter {
        const m = gb.m;
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(bytes);
        const c = try Config.parse(gpa, bytes, m.geo);
        const x = try gpa.create(Drafter);
        errdefer gpa.destroy(x);
        x.* = .{ .gpa = gpa, .io = io, .d = gb.d, .m = m, .k = gb.k, .gb = gb, .s = gb.s, .c = c, .w = .{ .fc = undefined, .hidden_norm = 0, .norm = 0, .layers = undefined }, .ba = undefined };
        errdefer x.w.buffers.deinit(gpa);
        errdefer for (x.w.buffers.items) |*b| b.free();
        try load(gpa, io, x.d, x.s, x.k, dir, &x.c, &x.w);
        x.ba = try cuda.block_attention.BlockAttention.load(x.d);
        errdefer x.ba.unload();
        errdefer x.buffers.deinit(gpa);
        errdefer for (x.buffers.items) |*b| b.free();
        const h = c.hidden;
        const qw = c.heads * dim;
        const kw = c.kv_heads * dim;
        const r = @max(rows_most, absorb_most);
        var inv: [dim / 2]f32 = undefined;
        for (&inv, 0..) |*f, i| f.* = @floatCast(1.0 / std.math.pow(f64, c.theta, 2.0 * @as(f64, @floatFromInt(i)) / dim));
        x.x = .{
            .inv = try x.alloc(dim / 2 * 4),
            .ids = try x.alloc(rows_most * 4),
            .pos = try x.alloc(r * 4),
            .table = try x.alloc(c.layers * max_blocks * 2 * 8),
            .lens = try x.alloc(max_blocks * 4),
            .picks = try x.alloc(rows_most * @sizeOf(Pick)),
            .xs = try x.alloc(r * (c.tap_count * h / 64) * 4),
            .fcx = try x.alloc(r * h * 2),
            .h = .{ try x.alloc(r * h * 2), try x.alloc(r * h * 2) },
            .hs = try x.alloc(r * h * 2),
            .normed = try x.alloc(r * h * 2),
            .q = try x.alloc(rows_most * qw * 2),
            .kk = try x.alloc(r * kw * 2),
            .vv = try x.alloc(r * kw * 2),
            .qh = try x.alloc(rows_most * qw * 2),
            .kb = try x.alloc(rows_most * kw * 2),
            .vb = try x.alloc(rows_most * kw * 2),
            .part = try x.alloc(rows_most * qw * 2),
            .att = try x.alloc(rows_most * qw * 2),
            .o = try x.alloc(rows_most * h * 2),
            .gate = try x.alloc(rows_most * c.intermediate * 2),
            .up = try x.alloc(rows_most * c.intermediate * 2),
            .act = try x.alloc(rows_most * c.intermediate * 2),
            .y = try x.alloc(rows_most * h * 2),
            .logits = try x.alloc(rows_most * cfg.vocab * 2),
            .junk = try x.alloc(r * kw * 2),
        };
        try (DeviceBuffer{ .d = x.d, .ptr = x.x.inv, .len = dim / 2 * 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(&inv));
        try m.setTaps(c.taps[0..c.tap_count]); // every forward keeps them from now on, the warm's windows too
        try x.warm();
        return x;
    }

    fn alloc(x: *Drafter, bytes: usize) !u64 {
        const b = try DeviceBuffer.alloc(x.d, @max(bytes, 256));
        errdefer {
            var y = b;
            y.free();
        }
        try x.buffers.append(x.gpa, b);
        return b.ptr;
    }

    pub fn deinit(x: *Drafter) void {
        x.s.synchronize() catch {};
        var it = x.lanes.valueIterator();
        while (it.next()) |l| x.dropLane(l.*);
        x.lanes.deinit(x.gpa);
        for (x.buffers.items) |*b| b.free();
        x.buffers.deinit(x.gpa);
        for (x.w.buffers.items) |*b| b.free();
        x.w.buffers.deinit(x.gpa);
        x.ba.unload();
        x.gpa.destroy(x);
    }

    pub fn drafter(x: *Drafter) dr.Drafter {
        return .{ .ptr = x, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(ptr: *anyopaque) []const u32 {
        const x = cast(ptr);
        return x.c.taps[0..x.c.tap_count];
    }

    /// Acceptance by depth, measured on the Python reference (114 rounds, 3.5 a round), until a stream has its own.
    const prior = [_]f64{ 0.77, 0.64, 0.71, 0.82, 0.84, 0.81, 0.9, 0.9, 0.9, 0.9, 0.83, 0.8, 0.83, 0.8, 0.8 };

    /// Chains of up to 15, no plain guard: a block costs the same at any depth (Metal's facts), the windows timed.
    fn factsFn(ptr: *anyopaque) dr.Facts {
        const x = cast(ptr);
        return .{ .depth = block_rows - 1, .step_ms = x.step_ms, .prior = &prior, .plain_guard = false, .batched = true, .stream_bytes = x.rowBytes() * x.m.context, .window_costs = &x.costs };
    }

    /// A context position's bytes over every layer's keys and values.
    fn rowBytes(x: *const Drafter) usize {
        return x.c.layers * 2 * x.c.kv_heads * dim * 2;
    }

    /// Layer i's keys (or values) of a lane: [kv heads, capacity, 128].
    fn cache(x: *const Drafter, l: *const Lane, i: usize, values: bool) u64 {
        return l.kv.ptr + (2 * i + @intFromBool(values)) * x.c.kv_heads * l.capacity * dim * 2;
    }

    fn newLane(x: *Drafter, capacity: usize) !*Lane {
        const l = try x.gpa.create(Lane);
        errdefer x.gpa.destroy(l);
        l.* = .{ .kv = try DeviceBuffer.alloc(x.d, x.rowBytes() * capacity), .capacity = capacity };
        return l;
    }

    fn dropLane(x: *Drafter, l: *Lane) void {
        l.kv.free();
        x.gpa.destroy(l);
    }

    fn openFn(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = cast(ptr);
        if (x.lanes.contains(s)) return error.DraftStreamOpen;
        const capacity = @min(x.m.context, (s.prompt_len + s.max_new + 2 * block_rows + 63) / 64 * 64);
        const l = try x.newLane(capacity);
        errdefer x.dropLane(l);
        try x.lanes.put(x.gpa, s, l);
    }

    fn releaseFn(ptr: *anyopaque, s: *Stream) void {
        const x = cast(ptr);
        if (x.lanes.fetchRemove(s)) |e| {
            x.s.synchronize() catch {};
            x.dropLane(e.value);
        }
    }

    fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const x = cast(ptr);
        for (streams, out) |s, o| {
            const l = x.lanes.get(s) orelse return error.DraftStreamUnknown;
            if (o.len > l.held_len) return error.DraftHeldFewer;
            @memcpy(o, l.held[0..o.len]);
        }
    }

    fn upload(x: *Drafter, dst: u64, bytes: []const u8) !void {
        try x.d.check(x.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, x.s.handle), "cuMemcpyHtoDAsync");
    }

    /// in [rows, l.k] through `l`: verify tiles on the group sums at xs, or the prompt GEMM past 32 rows.
    fn project(x: *Drafter, l: Affine4, in: u64, xs: ?u64, out: u64, rows: usize) !void {
        if (rows > x.k.affine.verifyRows() or xs == null) return x.k.affine.prompt(x.s, in, l, out, rows);
        try x.k.affine.verify(x.s, in, xs.?, l, out, rows);
    }

    fn copy2d(x: *Drafter, dst: u64, dst_pitch: usize, src: u64, src_pitch: usize, width: usize, rows: usize) !void {
        const p: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = src_pitch, .dst = dst, .dst_pitch = dst_pitch, .width = width, .height = rows };
        try x.d.check(x.d.api.cuMemcpy2DAsync_v2(&p, x.s.handle), "cuMemcpy2DAsync");
    }

    /// Tap rows [p0, p0 + n) (`src`, packed) into lane `l`: fc, hidden_norm, each layer's k (normed, roped) and v.
    fn absorbRows(x: *Drafter, l: *Lane, src: u64, n: usize, p0: u64) !void {
        const c = &x.c;
        const s = x.x;
        const h = c.hidden;
        const kw = c.kv_heads * dim;
        var pos: [absorb_most]i32 = undefined;
        for (pos[0..n], 0..) |*p, i| p.* = @intCast(p0 + i);
        try x.upload(s.pos, std.mem.sliceAsBytes(pos[0..n]));
        if (n <= x.k.affine.verifyRows()) try x.k.act.groupSums(x.s, src, c.tap_count * h, s.xs, n, c.tap_count * h);
        try x.project(x.w.fc, src, if (n <= x.k.affine.verifyRows()) s.xs else null, s.fcx, n);
        try x.k.norms.addRmsTail(x.s, s.fcx, null, x.w.hidden_norm, s.hs, s.normed, s.xs, n, h, c.eps, .{ .round = true });
        for (0..c.layers) |i| {
            const lw = &x.w.layers[i];
            try x.project(lw.k, s.normed, s.xs, s.kk, n);
            try x.project(lw.v, s.normed, s.xs, s.vv, n);
            const src_: cuda.qk_rows.Src = .{ .q = s.kk, .q_ld = @intCast(kw), .q_hs = dim, .k = s.kk, .k_ld = @intCast(kw), .k_hs = dim, .v = s.vv, .v_ld = @intCast(kw), .v_hs = dim };
            const shape: cuda.qk_rows.Shape = .{ .norms = lw.k_norm, .norm_k = lw.k_norm, .norms_bf16 = true, .inv = s.inv, .rot = dim, .eps = c.eps, .round = true };
            const at = p0 * dim * 2;
            try x.k.qk.rowsFrom(x.s, src_, .{ .h = c.kv_heads, .hk = c.kv_heads, .dim = dim }, shape, s.pos, null, .{ .q = s.junk, .k = x.cache(l, i, false) + at, .v = x.cache(l, i, true) + at, .kv_ld = dim, .kv_hs = l.capacity * dim }, n);
        }
    }

    /// Rows [start, start + rows) replace the context from `start`, in passes of up to a prompt chunk's rows.
    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const x = cast(ptr);
        const c = &x.c;
        for (items) |it| {
            const l = x.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
            const f = it.features;
            if (f.space != .device or f.dtype != .bf16 or f.row_bytes != c.tap_count * c.hidden * 2) return error.DraftFeaturesUnsupported;
            if (it.start > l.end) try x.resumeLane(it.stream, l, it.start); // a prompt pass that resumed a kept state
            if (it.start + f.rows > l.capacity) return error.DraftContextGap;
            var at: usize = 0;
            while (at < f.rows) : (at += absorb_most) {
                const n = @min(absorb_most, f.rows - at);
                try x.absorbRows(l, f.buffer + f.offset + at * f.row_bytes, n, it.start + at);
            }
            l.end = it.start + f.rows;
            l.held_len = 0;
        }
    }

    /// A stream's block in a hold: its lane, its pending token, the anchor (the block's first position), its depth.
    const Block = struct { lane: *Lane, pending: u32, anchor: u64, depth: u32 };

    /// Blocks of `width` rows each, [pending, mask...] at anchor.., through the drafter together; picks a row.
    fn blocks(x: *Drafter, bs: []const Block, width: usize) !void {
        const c = &x.c;
        const s = x.x;
        const h = c.hidden;
        const qw = c.heads * dim;
        const kw = c.kv_heads * dim;
        const g = c.heads / c.kv_heads;
        const rows = bs.len * width;
        var ids: [rows_most]i32 = undefined;
        var pos: [rows_most]i32 = undefined;
        var lens: [max_blocks]i32 = undefined;
        var table: [8 * max_blocks * 2]u64 = undefined;
        for (bs, 0..) |b, j| {
            for (0..width) |r| {
                ids[j * width + r] = @intCast(if (r == 0) b.pending else c.mask);
                pos[j * width + r] = @intCast(b.anchor + r);
            }
            lens[j] = @intCast(b.anchor);
        }
        try x.upload(s.ids, std.mem.sliceAsBytes(ids[0..rows]));
        try x.upload(s.pos, std.mem.sliceAsBytes(pos[0..rows]));
        try x.upload(s.lens, std.mem.sliceAsBytes(lens[0..bs.len]));
        try x.k.embed.affine(x.s, s.ids, x.m.w.embed_w, x.m.w.embed_s, x.m.w.embed_b, s.h[0], rows, h, .{ .bits = @intCast(x.m.w.embed_bits), .group = cfg.group, .scale = fwd.embedScale(h) });
        try x.k.norms.addRmsTail(x.s, s.h[0], null, x.w.layers[0].input_norm, s.hs, s.normed, s.xs, rows, h, c.eps, .{ .round = true });
        var cur: usize = 0;
        for (0..c.layers) |i| {
            const lw = &x.w.layers[i];
            try x.project(lw.q, s.normed, s.xs, s.q, rows);
            try x.project(lw.k, s.normed, s.xs, s.kk, rows);
            try x.project(lw.v, s.normed, s.xs, s.vv, rows);
            const src: cuda.qk_rows.Src = .{ .q = s.q, .q_ld = @intCast(qw), .q_hs = dim, .k = s.kk, .k_ld = @intCast(kw), .k_hs = dim, .v = s.vv, .v_ld = @intCast(kw), .v_hs = dim };
            const shape: cuda.qk_rows.Shape = .{ .norms = lw.q_norm, .norm_k = lw.k_norm, .norms_bf16 = true, .inv = s.inv, .rot = dim, .eps = c.eps, .round = true };
            // head-major rows for block_attention: q [heads, rows, 128], k and v [kv heads, rows, 128]
            try x.k.qk.rowsFrom(x.s, src, .{ .h = c.heads, .hk = c.kv_heads, .dim = dim }, shape, s.pos, null, .{ .q = s.qh, .k = s.kb, .v = s.vb, .q_ld = dim, .q_hs = rows * dim, .kv_ld = dim, .kv_hs = rows * dim }, rows);
            // one call a key head (block_attention reads a context's heads [hk, len, 128]; ours are capacity apart)
            for (0..c.kv_heads) |kh| {
                for (bs, 0..) |b, j| {
                    table[2 * j] = x.cache(b.lane, i, false) + kh * b.lane.capacity * dim * 2;
                    table[2 * j + 1] = x.cache(b.lane, i, true) + kh * b.lane.capacity * dim * 2;
                }
                try x.upload(s.table, std.mem.sliceAsBytes(table[0 .. 2 * bs.len])); // after the last call read it
                const sh: cuda.block_attention.Shape = .{ .h = g, .hk = 1, .rows = rows, .block = width, .window = if (c.sliding[i]) c.window - 1 else std.math.maxInt(i32), .scale = 1 / @sqrt(@as(f32, dim)), .causal = c.sliding[i] };
                try x.ba.run(x.s, s.qh + kh * g * rows * dim * 2, s.kb + kh * rows * dim * 2, s.vb + kh * rows * dim * 2, s.table, s.lens, s.part, sh);
                try x.copy2d(s.att + kh * g * dim * 2, qw * 2, s.part, g * dim * 2, g * dim * 2, rows);
            }
            try x.k.act.groupSums(x.s, s.att, qw, s.xs, rows, qw);
            try x.project(lw.o, s.att, s.xs, s.o, rows);
            try x.k.norms.addRmsTail(x.s, s.o, s.h[cur], lw.post_attention, s.h[1 - cur], s.normed, s.xs, rows, h, c.eps, .{ .round = true });
            cur = 1 - cur;
            try x.project(lw.gate, s.normed, s.xs, s.gate, rows);
            try x.project(lw.up, s.normed, s.xs, s.up, rows);
            try x.k.act.swigluRound(x.s, s.gate, c.intermediate, s.up, c.intermediate, s.act, s.xs, rows, c.intermediate);
            try x.project(lw.down, s.act, s.xs, s.y, rows);
            const next = if (i + 1 < c.layers) x.w.layers[i + 1].input_norm else x.w.norm;
            try x.k.norms.addRmsTail(x.s, s.y, s.h[cur], next, s.h[1 - cur], s.normed, s.xs, rows, h, c.eps, .{ .round = true });
            cur = 1 - cur;
        }
        try x.head(s.normed, s.xs, rows);
        try x.k.round.argmax(x.s, s.logits, s.picks, rows, cfg.vocab, cfg.vocab);
    }

    /// The target's tied head over the drafter's normed rows.
    fn head(x: *Drafter, in: u64, xs: u64, rows: usize) !void {
        const hd = x.m.w.head;
        const tile: usize = switch (hd) {
            .a4 => x.k.affine.verifyRows(),
            .an => 16,
        };
        var at: usize = 0;
        while (at < rows) : (at += tile) {
            const n = @min(tile, rows - at);
            const xa = in + at * hd.k() * 2;
            const sa = xs + at * (hd.k() / 64) * 4;
            const oa = x.x.logits + at * cfg.vocab * 2;
            switch (hd) {
                .a4 => |q| try x.k.affine.verify(x.s, xa, sa, q, oa, n),
                .an => |q| try x.k.wide.decode(x.s, xa, sa, q, oa, n),
            }
        }
    }

    /// Each stream's chain of `depth` drafts after `pending` at `position`, up to eight blocks together.
    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const x = cast(ptr);
        var at: usize = 0;
        while (at < items.len) {
            var bs: [max_blocks]Block = undefined;
            var n: usize = 0;
            var width: usize = 1;
            while (at < items.len and n < max_blocks) : (at += 1) {
                const it = items[at];
                const l = x.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
                if (it.depth == 0 or it.depth >= block_rows or it.position == 0) return error.DraftDepth;
                if (it.lanes != null) return error.DraftTreesUnsupported;
                const anchor = it.position - 1;
                if (l.end != anchor or anchor + block_rows > l.capacity + block_rows) return error.DraftContextGap;
                bs[n] = .{ .lane = l, .pending = it.pending, .anchor = anchor, .depth = it.depth };
                width = @max(width, it.depth + 1);
                n += 1;
            }
            try x.blocks(bs[0..n], width);
            var picks: [rows_most]Pick = undefined;
            try x.s.synchronize();
            try x.d.check(x.d.api.cuMemcpyDtoH_v2(&picks, x.x.picks, n * width * @sizeOf(Pick)), "cuMemcpyDtoH");
            for (bs[0..n], 0..) |b, j| {
                for (picks[j * width + 1 ..][0..b.depth], 0..) |p, i| {
                    if (p.nonfinite != 0 or p.token >= cfg.vocab) return error.DraftNonfinite;
                    b.lane.held[i] = p.token;
                }
                b.lane.held_len = b.depth;
            }
        }
    }

    /// Bytes a kept prompt state's drafter context at `at` takes.
    pub fn keptBytes(x: *const Drafter, at: usize) u64 {
        return x.rowBytes() * at;
    }

    /// Stream `s`'s context [0, at) into `k.draft` when its lane stands at the state's position (a prompt pass's mark).
    pub fn keep(x: *Drafter, s: *const Stream, k: *snap.Saved) !void {
        const l = x.lanes.get(@constCast(s)) orelse return;
        if (l.end != k.at) return;
        var buf = try DeviceBuffer.alloc(x.d, x.keptBytes(k.at));
        errdefer buf.free();
        try x.copyLane(l, k.at, buf.ptr, true);
        k.draft = buf;
    }

    /// The lane takes the context kept with the state its stream's prompt pass resumed at `start`.
    fn resumeLane(x: *Drafter, s: *Stream, l: *Lane, start: u64) !void {
        const saved = s.reuse.saved orelse return error.DraftContextGap;
        const k: *const snap.Saved = @ptrCast(@alignCast(saved));
        if (s.cached != start or k.at != start or k.draft == null or l.end != 0 or start > l.capacity) return error.DraftContextGap;
        try x.copyLane(l, start, k.draft.?.ptr, false);
        l.end = start;
    }

    /// Positions [0, at) of every layer's heads between the lane and a packed buffer (`out`: lane to buffer).
    fn copyLane(x: *Drafter, l: *const Lane, at: usize, buf: u64, out: bool) !void {
        const n = at * dim * 2;
        var to: usize = 0;
        for (0..x.c.layers) |i| for ([_]bool{ false, true }) |values| for (0..x.c.kv_heads) |kh| {
            const live = x.cache(l, i, values) + kh * l.capacity * dim * 2;
            const src = if (out) live else buf + to;
            const dst = if (out) buf + to else live;
            if (n > 0) try x.d.check(x.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, x.s.handle), "cuMemcpyDtoDAsync");
            to += n;
        };
    }

    /// Blocks on an empty lane timed (1 and 15 drafts), then the target's windows of 1 to 16 rows on slot 0.
    fn warm(x: *Drafter) !void {
        const l = try x.newLane(64);
        defer x.dropLane(l);
        var ms: [2]f64 = undefined;
        for ([_]u32{ block_rows - 1, 1, block_rows - 1, 1, block_rows - 1 }, 0..) |depth, i| {
            const t0 = std.Io.Clock.awake.now(x.io);
            try x.blocks(&.{.{ .lane = l, .pending = 2, .anchor = 0, .depth = depth }}, depth + 1);
            try x.s.synchronize();
            if (i >= 3) ms[i - 3] = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(x.io)).toNanoseconds())) / 1e6;
        }
        x.step_ms = @max(ms[1] - ms[0], 0) / @as(f64, block_rows - 2);
        const tokens: [be.max_window]u32 = @splat(2);
        for (&x.costs, 1..) |*c, n| {
            var best: f64 = std.math.inf(f64);
            for (0..4) |i| {
                const t0 = std.Io.Clock.awake.now(x.io);
                try x.gb.run(&.{.{ .slot = 0, .p0 = 128, .tokens = tokens[0..n] }}, false, 0, n);
                try x.s.synchronize();
                const t = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(x.io)).toNanoseconds())) / 1e6;
                if (i > 0) best = @min(best, t);
            }
            c.* = .{ .width = @intCast(n), .ms = best };
        }
    }
};

test "the stock drafter config is admitted against the 31B and refused against the 12B" {
    const stock =
        \\{"architectures":["DFlashDraftModel"],"dflash_config":{"mask_token_id":4,"target_layer_ids":[1,12,23,35,46,57]},"block_size":16,
        \\"hidden_size":5376,"intermediate_size":10752,"num_hidden_layers":5,"num_attention_heads":64,"num_key_value_heads":8,"head_dim":128,
        \\"sliding_window":2048,"rms_norm_eps":1e-6,"rope_theta":1000000,"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"]}
    ;
    const c = try Config.parse(std.testing.allocator, stock, cfg.dense31b);
    try std.testing.expectEqual(@as(usize, 6), c.tap_count);
    try std.testing.expectEqual(@as(u32, 57), c.taps[5]);
    try std.testing.expect(c.sliding[3] and !c.sliding[4]);
    try std.testing.expectError(error.DraftForAnotherTarget, Config.parse(std.testing.allocator, stock, cfg.unified12b));
    const causal = try std.mem.replaceOwned(u8, std.testing.allocator, stock, "\"hidden_size\"", "\"is_causal\":false,\"hidden_size\"");
    defer std.testing.allocator.free(causal);
    try std.testing.expectError(error.UnsupportedDraftMask, Config.parse(std.testing.allocator, causal, cfg.dense31b));
}
