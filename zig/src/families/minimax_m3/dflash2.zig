//! incoai's DFlash2 drafter for MiniMax-M3.1-Flash (SGLang's DFlash2DraftModel): 8-row blocks over the target's taps.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const ks = @import("kernel_sources");
const ck = @import("../../core/checkpoint_metal.zig");
const wts = @import("weights.zig");
const fwd = @import("forward.zig");
const Model = @import("model.zig").Model;
const now = @import("dspark.zig").now;

const dr = lanes.drafter;
const Stream = lanes.Stream;
const B = fwd.B;
const b = fwd.b;
const bo = fwd.bo;
const t = fwd.t;
const A = fwd.A;

/// The drafter's shape from config.json's `dflash2` (tools/convert.py writes it).
pub const Config = struct {
    layers: u32 = 5,
    heads: u32 = 64,
    kv_heads: u32 = 8,
    hidden: u32 = 6144,
    ffn: u32 = 12288,
    window: u32 = 2048,
    block: u32 = 8,
    mask: u32 = 200061,
    rank: u32 = 256,
    top: u32 = 32,
    group: u32 = 16,
    taps: [6]u32 = .{ 5, 16, 26, 37, 47, 58 },
    theta: f32 = 5e6,
    eps: f32 = 1e-6,

    /// Ring slots of a layer's context: the window and a block past it, whole 128-key blocks.
    pub fn ring(c: *const Config) u32 {
        return std.mem.alignForward(u32, c.window + c.block, 128);
    }

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !?Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        var doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        defer doc.deinit();
        const d = doc.value.object.get("dflash2") orelse return null;
        if (d != .object) return error.BadDFlash2Config;
        const o = d.object;
        const int = struct {
            fn of(x: std.json.ObjectMap, k: []const u8) !u32 {
                const v = x.get(k) orelse return error.BadDFlash2Config;
                if (v != .integer or v.integer <= 0) return error.BadDFlash2Config;
                return @intCast(v.integer);
            }
            fn f(x: std.json.ObjectMap, k: []const u8) !f32 {
                const v = x.get(k) orelse return error.BadDFlash2Config;
                return switch (v) {
                    .integer => |i| @floatFromInt(i),
                    .float => |q| @floatCast(q),
                    else => error.BadDFlash2Config,
                };
            }
        };
        var c = Config{
            .layers = try int.of(o, "num_hidden_layers"),
            .heads = try int.of(o, "num_attention_heads"),
            .kv_heads = try int.of(o, "num_key_value_heads"),
            .hidden = try int.of(o, "hidden_size"),
            .ffn = try int.of(o, "intermediate_size"),
            .window = try int.of(o, "sliding_window"),
            .block = try int.of(o, "block_size"),
            .mask = try int.of(o, "mask_token_id"),
            .rank = try int.of(o, "selector_rank"),
            .top = try int.of(o, "selector_top_k"),
            .group = try int.of(o, "conv_group_size"),
            .theta = try int.f(o, "rope_theta"),
            .eps = try int.f(o, "rms_norm_eps"),
        };
        const taps = o.get("target_layer_ids") orelse return error.BadDFlash2Config;
        if (taps != .array or taps.array.items.len != c.taps.len) return error.UnsupportedDFlash2;
        for (taps.array.items, &c.taps) |x, *y| {
            if (x != .integer or x.integer < 0) return error.BadDFlash2Config;
            y.* = @intCast(x.integer);
        }
        if ((try int.of(o, "head_dim")) != 128 or (try int.of(o, "conv_kernel_size")) != 2 or c.layers > 8 or c.block != 8 or
            c.rank != 256 or c.top != 32 or c.heads % c.kv_heads != 0 or c.hidden % c.group != 0) return error.UnsupportedDFlash2;
        return c;
    }
};

/// Rows a block takes (the trained block).
pub const max_block = 8;
/// Context rows one absorb pass projects at most (a prompt chunk's).
const absorb_rows = 2048;

const Conv = struct { base: wts.Tensor, proj: wts.Linear };
const Layer = struct { in_norm: wts.Tensor, post_norm: wts.Tensor, q_norm: wts.Tensor, k_norm: wts.Tensor, qkv: wts.Linear, o: wts.Linear, gu: wts.Linear, down: wts.Linear, attn_conv: Conv, mlp_conv: Conv };

const Weights = struct {
    fc: wts.Linear,
    hidden_norm: wts.Tensor,
    norm: wts.Tensor,
    ctx_kv: wts.Linear,
    hproj: wts.Linear,
    pred: wts.Tensor,
    succ: wts.Tensor,
    layers: [8]Layer,
};

fn tensor(c: *const ck.Checkpoint, name: []const u8) !wts.Tensor {
    const x = try c.get(name);
    if (x.dtype != .bf16) return error.UnexpectedDFlash2Tensor;
    return x;
}

fn affine(c: *const ck.Checkpoint, base: []const u8, n: usize, k: usize) !wts.Linear {
    var buf: [128]u8 = undefined;
    const w = try c.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{base}));
    const s = try c.get(try std.fmt.bufPrint(&buf, "{s}.scales", .{base}));
    const z = try c.get(try std.fmt.bufPrint(&buf, "{s}.biases", .{base}));
    return wts.checkedAffine(w, s, z, 1, n, k);
}

fn load(ckp: *const ck.Checkpoint, c: *const Config, vocab: usize) !Weights {
    const h: usize = c.hidden;
    const kvw: usize = c.kv_heads * 128;
    const dyn: usize = 4 * (h / c.group);
    var w: Weights = .{
        .fc = try affine(ckp, "dflash2.fc", h, h * c.taps.len),
        .hidden_norm = try tensor(ckp, "dflash2.hidden_norm.weight"),
        .norm = try tensor(ckp, "dflash2.norm.weight"),
        .ctx_kv = try affine(ckp, "dflash2.ctx_kv_proj", c.layers * 2 * kvw, h),
        .hproj = try affine(ckp, "dflash2.hidden_projection", c.rank, h),
        .pred = try tensor(ckp, "dflash2.predecessor_codebook"),
        .succ = try tensor(ckp, "dflash2.successor_codebook"),
        .layers = undefined,
    };
    if (w.pred.count() != vocab * c.rank or w.succ.count() != vocab * c.rank) return error.UnexpectedDFlash2Tensor;
    for (w.layers[0..c.layers], 0..) |*l, i| {
        var nb: [128]u8 = undefined;
        const name = struct {
            fn of(buf: []u8, layer: usize, z: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, "dflash2.layers.{d}.{s}", .{ layer, z });
            }
        }.of;
        l.* = .{
            .in_norm = try tensor(ckp, try name(&nb, i, "input_layernorm.weight")),
            .post_norm = try tensor(ckp, try name(&nb, i, "post_attention_layernorm.weight")),
            .q_norm = try tensor(ckp, try name(&nb, i, "self_attn.q_norm.weight")),
            .k_norm = try tensor(ckp, try name(&nb, i, "self_attn.k_norm.weight")),
            .qkv = try affine(ckp, try name(&nb, i, "self_attn.qkv_proj"), (c.heads + 2 * c.kv_heads) * 128, h),
            .o = try affine(ckp, try name(&nb, i, "self_attn.o_proj"), h, c.heads * 128),
            .gu = try affine(ckp, try name(&nb, i, "mlp.gate_up_proj"), 2 * @as(usize, c.ffn), h),
            .down = try affine(ckp, try name(&nb, i, "mlp.down_proj"), h, c.ffn),
            .attn_conv = .{ .base = try tensor(ckp, try name(&nb, i, "attention_conv.base_kernel")), .proj = try affine(ckp, try name(&nb, i, "attention_conv.kernel_projection"), dyn, h) },
            .mlp_conv = .{ .base = try tensor(ckp, try name(&nb, i, "mlp_conv.base_kernel")), .proj = try affine(ckp, try name(&nb, i, "mlp_conv.kernel_projection"), dyn, h) },
        };
        if (l.attn_conv.base.count() != 4 * h or l.mlp_conv.base.count() != 4 * h) return error.UnexpectedDFlash2Tensor;
    }
    return w;
}

/// One stream's context: each layer's keys and values for the last positions, a ring of Config.ring() slots.
const Lane = struct {
    k: [8]mtl.Buffer,
    v: [8]mtl.Buffer,
    end: u64 = 0, // context rows absorbed: positions [from, end)
    from: u64 = 0, // the first position held (a resumed prompt state's drafter context is not kept)
    held: [max_block]u32 = @splat(0),
    held_len: u32 = 0,
};

const names = [_][:0]const u8{ "df_norm", "df_norm16", "df_prep", "df_conv", "df_silu", "df_topk", "df_walk" };
const Kernel = enum { norm, norm16, prep, conv, silu, topk, walk };

/// The drafter's scratch, fields in init's order.
const fields = [_][]const u8{ "ids", "resid", "x", "xc", "dyn", "qkv", "q", "attn", "y", "yc", "gu", "act", "logits", "pm", "pl", "po", "tv", "ti", "tv2", "ti2", "uv", "ui", "hid", "seq", "fcx", "hctx", "kv", "sums", "vdst", "mdst" };

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    target: *Model,
    config: Config,
    w: Weights,
    pipes: [names.len]mtl.Pipeline,
    lanes: std.AutoHashMapUnmanaged(*Stream, *Lane) = .empty,
    follow: ?mtl.ResidencySet.Follow = null,
    step_ms: f64 = 0,
    s: [fields.len]mtl.Buffer = undefined,

    pub fn init(gpa: std.mem.Allocator, target: *Model, config: Config) !*Drafter {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .target = target, .config = config, .w = try load(&target.checkpoint, &config, target.config.vocab), .pipes = undefined };
        const c = &d.config;
        if (c.hidden != target.config.hidden) return error.DFlash2ForAnotherTarget;
        for (c.taps) |l| if (l >= target.config.layers) return error.DFlash2ForAnotherTarget;
        {
            const lib = try mtl.Library.fromSource(target.device, ks.minimax_m3_dflash2, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (d.pipes[0..made]) |p| p.deinit();
            for (names, 0..) |n, i| {
                d.pipes[i] = try mtl.Pipeline.init(target.device, lib, n, false);
                made += 1;
            }
        }
        errdefer for (d.pipes) |p| p.deinit();
        const h: usize = c.hidden;
        const r: usize = max_block;
        const qw: usize = c.heads * 128;
        const kvw: usize = c.kv_heads * 128;
        const nblk: usize = c.ring() / 128 + 1;
        const vocab: usize = target.config.vocab;
        const chunks: usize = (vocab + 2047) / 2048;
        const dyn: usize = 4 * (h / c.group);
        const sizes = [fields.len]usize{ r * 4, r * h * 4, r * h * 2, r * h * 2, r * dyn * 2, r * (qw + 2 * kvw) * 2, r * qw * 2, r * qw * 2, r * h * 2, r * h * 2, r * 2 * c.ffn * 2, r * c.ffn * 2, r * vocab * 2, c.heads * r * nblk * 4, c.heads * r * nblk * 4, c.heads * r * nblk * 128 * 4, r * chunks * 32 * 4, r * chunks * 32 * 4, r * 64 * 4, r * 64 * 4, r * 32 * 4, r * 32 * 4, r * c.rank * 2, 16 * 4, absorb_rows * h * 2, absorb_rows * h * 2, absorb_rows * c.layers * 2 * kvw * 2, absorb_rows * h * c.taps.len / 32 * 4, 16, 16 };
        var made: usize = 0;
        errdefer for (d.s[0..made]) |x| x.deinit();
        for (sizes, 0..) |n, i| {
            d.s[i] = try target.device.buffer(n, mtl.ResourceOptions.shared);
            made += 1;
        }
        try d.warm();
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        d.setResident(null);
        var it = d.lanes.valueIterator();
        while (it.next()) |l| d.dropLane(l.*);
        d.lanes.deinit(d.gpa);
        for (d.s) |x| x.deinit();
        for (d.pipes) |p| p.deinit();
        d.gpa.destroy(d);
    }

    fn buf(d: *const Drafter, comptime f: []const u8) mtl.Buffer {
        inline for (fields, 0..) |n, i| if (comptime std.mem.eql(u8, n, f)) return d.s[i];
        @compileError("no drafter buffer " ++ f);
    }

    fn pipe(d: *const Drafter, k: Kernel) mtl.Pipeline {
        return d.pipes[@backingInt(k)];
    }

    pub fn setResident(d: *Drafter, hook: ?mtl.ResidencySet.Follow) void {
        if (d.follow) |f| {
            for (d.s) |x| f.leave(f.ctx, x);
            var it = d.lanes.valueIterator();
            while (it.next()) |l| d.laneResident(l.*, f, false);
        }
        d.follow = hook;
        if (hook) |f| {
            for (d.s) |x| f.join(f.ctx, x);
            var it = d.lanes.valueIterator();
            while (it.next()) |l| d.laneResident(l.*, f, true);
        }
    }

    fn laneResident(d: *const Drafter, l: *const Lane, f: mtl.ResidencySet.Follow, on: bool) void {
        for (0..d.config.layers) |i| for ([_]mtl.Buffer{ l.k[i], l.v[i] }) |x| if (on) f.join(f.ctx, x) else f.leave(f.ctx, x);
    }

    /// Device bytes one stream's context takes.
    pub fn laneBytes(d: *const Drafter) usize {
        const c = &d.config;
        return @as(usize, c.layers) * 2 * c.ring() * c.kv_heads * 128 * 2;
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(ptr: *anyopaque) []const u32 {
        return &cast(ptr).config.taps;
    }

    fn factsFn(ptr: *anyopaque) dr.Facts {
        const d = cast(ptr);
        return .{ .depth = d.config.block - 1, .step_ms = d.step_ms, .plain_guard = true, .stream_bytes = d.laneBytes() };
    }

    fn newLane(d: *Drafter) !*Lane {
        const l = try d.gpa.create(Lane);
        errdefer d.gpa.destroy(l);
        l.* = .{ .k = undefined, .v = undefined };
        const bytes = @as(usize, d.config.ring()) * d.config.kv_heads * 128 * 2;
        var made: usize = 0;
        errdefer for (0..made) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        };
        for (0..d.config.layers) |i| {
            l.k[i] = try d.target.device.buffer(bytes, mtl.ResourceOptions.private);
            l.v[i] = d.target.device.buffer(bytes, mtl.ResourceOptions.private) catch |e| {
                l.k[i].deinit();
                return e;
            };
            made += 1;
        }
        if (d.follow) |f| d.laneResident(l, f, true);
        return l;
    }

    fn dropLane(d: *Drafter, l: *Lane) void {
        if (d.follow) |f| d.laneResident(l, f, false);
        for (0..d.config.layers) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        }
        d.gpa.destroy(l);
    }

    fn openFn(ptr: *anyopaque, s: *Stream) anyerror!void {
        const d = cast(ptr);
        if (d.lanes.contains(s)) return error.DraftStreamOpen;
        const l = try d.newLane();
        errdefer d.dropLane(l);
        try d.lanes.put(d.gpa, s, l);
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

    fn glue(e: mtl.ComputeEncoder, p: mtl.Pipeline, args: anytype, bufs: []const B, grid: [3]usize, group: [3]usize) void {
        e.setPipeline(p);
        for (bufs, 0..) |r, slot| if (r.args) e.setValue(args, slot) else e.setBuffer(r.buf, r.off, slot);
        e.dispatchGroups(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(group[0], group[1], group[2]));
    }

    fn threads(e: mtl.ComputeEncoder, p: mtl.Pipeline, args: anytype, bufs: []const B, grid: [3]usize) void {
        e.setPipeline(p);
        for (bufs, 0..) |r, slot| if (r.args) e.setValue(args, slot) else e.setBuffer(r.buf, r.off, slot);
        e.dispatchThreads(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(256, 1, 1));
    }

    fn project(d: *const Drafter, e: mtl.ComputeEncoder, w: wts.Linear, in: B, out: B, rows: usize) !void {
        // a block's rows on the tensor units: each weight (the target head's too) read once for all of them
        return fwd.projectRows(e, d.target, d.buf("sums"), w, in, out, rows, 3);
    }

    const NArgs = extern struct { rows: u32, width: u32, eps: f32, add: u32 };
    const PArgs = extern struct { rows: u32, heads: u32, kv_heads: u32, pos0: u32, stride: u32, q_off: u32, k_off: u32, v_off: u32, ring: u32, eps: f32, theta: f32 };
    const VArgs = extern struct { rows: u32, pos0: u32, heads: u32, groups: u32, blocks: u32, first: u32, hi: u32, window: u32, ring: u32, scale: f32 };
    const CArgs = extern struct { rows: u32, width: u32, block: u32, groups: u32, side: u32 };

    /// The context rows [p0, p0 + rows) from `src` (taps x hidden bf16 each) into every layer's ring.
    fn absorbRows(d: *const Drafter, e: mtl.ComputeEncoder, l: *const Lane, src: B, p0: u64, rows: usize) !void {
        const c = &d.config;
        const kvw = c.kv_heads * 128;
        try d.project(e, d.w.fc, src, b(d.buf("fcx")), rows);
        glue(e, d.pipe(.norm16), NArgs{ .rows = @intCast(rows), .width = c.hidden, .eps = c.eps, .add = 0 }, &.{ b(d.buf("fcx")), t(d.w.hidden_norm), A, b(d.buf("hctx")) }, .{ rows, 1, 1 }, .{ 256, 1, 1 });
        try d.project(e, d.w.ctx_kv, b(d.buf("hctx")), b(d.buf("kv")), rows);
        for (0..c.layers) |i| {
            const pa = PArgs{ .rows = @intCast(rows), .heads = 0, .kv_heads = c.kv_heads, .pos0 = @intCast(p0), .stride = c.layers * 2 * kvw, .q_off = 0, .k_off = @intCast(i * 2 * kvw), .v_off = @intCast(i * 2 * kvw + kvw), .ring = c.ring(), .eps = c.eps, .theta = c.theta };
            glue(e, d.pipe(.prep), pa, &.{ b(d.buf("kv")), t(d.w.layers[i].q_norm), t(d.w.layers[i].k_norm), A, b(d.buf("q")), b(l.k[i]), b(l.v[i]) }, .{ c.kv_heads, rows, 1 }, .{ 32, 1, 1 });
        }
    }

    /// Rows [start, start + rows) replace the context from `start`; rows older than a window and a block are skipped.
    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        const c = &d.config;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = d.target.queue.commandBuffer();
        const e = cb.compute(.serial);
        const width: usize = c.taps.len * c.hidden * 2;
        for (items) |it| {
            const l = d.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
            const f = it.features;
            if (f.space != .device or f.dtype != .bf16 or f.row_bytes != width) return error.DraftFeaturesUnsupported;
            if (it.start > l.end) l.from = it.start; // the rows before it were restored, not absorbed
            const src = mtl.Buffer{ .id = @ptrFromInt(f.buffer) };
            const end = it.start + f.rows;
            const keep_from = if (it.prompt) @max(it.start, (it.stream.prompt_len -| (c.window + c.block))) else it.start;
            var at: u64 = keep_from;
            while (at < end) {
                const n: usize = @intCast(@min(absorb_rows, end - at));
                try d.absorbRows(e, l, bo(src, f.offset + @as(usize, @intCast(at - it.start)) * width), at, n);
                at += n;
            }
            l.end = end;
            l.held_len = 0;
        }
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| return failed(msg);
    }

    /// x (n rows, the sublayer's normed input) through `conv`'s prepare: its coefficients into dyn, side 0 into xc.
    fn convPrepare(d: *const Drafter, e: mtl.ComputeEncoder, conv: Conv, n: usize) !void {
        const c = &d.config;
        try d.project(e, conv.proj, b(d.buf("x")), b(d.buf("dyn")), n);
        threads(e, d.pipe(.conv), CArgs{ .rows = @intCast(n), .width = c.hidden, .block = c.block, .groups = c.hidden / c.group, .side = 0 }, &.{ b(d.buf("x")), b(d.buf("dyn")), t(conv.base), A, b(d.buf("xc")) }, .{ c.hidden, n, 1 });
    }

    /// The block [pending, mask x 7] at positions anchor.. through the drafter; seq[1..] its drafts (rows 1-7's path).
    fn encodeBlock(d: *const Drafter, e: mtl.ComputeEncoder, l: *const Lane, pending: u32, anchor: u64) !void {
        const c = &d.config;
        const m = d.target;
        const n: usize = c.block;
        const h: usize = c.hidden;
        const qw: usize = c.heads * 128;
        const kvw: usize = c.kv_heads * 128;
        const ids = d.buf("ids").slice(u32, max_block);
        ids[0] = pending;
        @memset(ids[1..n], c.mask);
        d.buf("seq").slice(u32, 16)[0] = pending;
        const E = m.weights.embed;
        const MArgs = extern struct { rows: u32 = 0, width: u32 = 0, heads: u32 = 0, dim: u32 = 0, pos: u32 = 0, stride: u32 = 0, bits: u32 = 0, group: u32 = 0, eps: f32 = 0, alpha: f32 = 0, limit: f32 = 0, scale: f32 = 0 };
        glue(e, m.pipe(.embed), MArgs{ .rows = @intCast(n), .width = c.hidden, .bits = E.bits, .group = @intCast(E.group) }, &.{ b(d.buf("ids")), bo(E.w, E.w_off), bo(E.scales, E.s_off), bo(E.biases, E.b_off), A, b(d.buf("resid")) }, .{ (h + 255) / 256, n, 1 }, .{ 256, 1, 1 });
        const lo = @max((anchor + 1) -| c.window, l.from);
        const hi = anchor + n;
        const first = lo / 128;
        const blocks = (hi - 1) / 128 - first + 1;
        const groups: u32 = c.hidden / c.group;
        for (0..c.layers) |i| {
            const L = &d.w.layers[i];
            // input norm (the residual takes the last sublayer's output first, from the second layer on)
            glue(e, d.pipe(.norm), NArgs{ .rows = @intCast(n), .width = c.hidden, .eps = c.eps, .add = @intFromBool(i > 0) }, &.{ b(d.buf("resid")), b(d.buf("yc")), t(L.in_norm), A, b(d.buf("x")) }, .{ n, 1, 1 }, .{ 256, 1, 1 });
            try d.convPrepare(e, L.attn_conv, n);
            try d.project(e, L.qkv, b(d.buf("xc")), b(d.buf("qkv")), n);
            const pa = PArgs{ .rows = @intCast(n), .heads = c.heads, .kv_heads = c.kv_heads, .pos0 = @intCast(anchor), .stride = @intCast(qw + 2 * kvw), .q_off = 0, .k_off = @intCast(qw), .v_off = @intCast(qw + kvw), .ring = c.ring(), .eps = c.eps, .theta = c.theta };
            glue(e, d.pipe(.prep), pa, &.{ b(d.buf("qkv")), t(L.q_norm), t(L.k_norm), A, b(d.buf("q")), b(l.k[i]), b(l.v[i]) }, .{ c.heads + c.kv_heads, n, 1 }, .{ 32, 1, 1 });
            const va = VArgs{ .rows = @intCast(n), .pos0 = @intCast(anchor), .heads = c.heads, .groups = c.kv_heads, .blocks = @intCast(blocks), .first = @intCast(first), .hi = @intCast(hi), .window = c.window, .ring = c.ring(), .scale = 1.0 / @sqrt(128.0) };
            glue(e, m.draft_vec, va, &.{ b(d.buf("q")), b(l.k[i]), b(l.v[i]), A, b(d.buf("pm")), b(d.buf("pl")), b(d.buf("po")) }, .{ c.kv_heads, n, blocks }, .{ 32, c.heads / c.kv_heads, 1 });
            const ma = extern struct { rows: u32, pos: u32, heads: u32, groups: u32, topk: u32, scale: f32, sel: u32, all: u32 }{ .rows = @intCast(n), .pos = 0, .heads = c.heads, .groups = c.kv_heads, .topk = @intCast(blocks), .scale = 0, .sel = 0, .all = 0 };
            glue(e, m.pipe(.merge_vec), ma, &.{ b(d.buf("pm")), b(d.buf("pl")), b(d.buf("po")), A, b(d.buf("attn")) }, .{ c.heads, n, 1 }, .{ 32, 1, 1 });
            try d.project(e, L.o, b(d.buf("attn")), b(d.buf("y")), n);
            // the attention conv's finish (side 1, its prepare's coefficients), then the post-attention norm
            threads(e, d.pipe(.conv), CArgs{ .rows = @intCast(n), .width = c.hidden, .block = c.block, .groups = groups, .side = 1 }, &.{ b(d.buf("y")), b(d.buf("dyn")), t(L.attn_conv.base), A, b(d.buf("yc")) }, .{ c.hidden, n, 1 });
            glue(e, d.pipe(.norm), NArgs{ .rows = @intCast(n), .width = c.hidden, .eps = c.eps, .add = 1 }, &.{ b(d.buf("resid")), b(d.buf("yc")), t(L.post_norm), A, b(d.buf("x")) }, .{ n, 1, 1 }, .{ 256, 1, 1 });
            try d.convPrepare(e, L.mlp_conv, n);
            try d.project(e, L.gu, b(d.buf("xc")), b(d.buf("gu")), n);
            threads(e, d.pipe(.silu), extern struct { rows: u32, width: u32 }{ .rows = @intCast(n), .width = c.ffn }, &.{ b(d.buf("gu")), A, b(d.buf("act")) }, .{ c.ffn, n, 1 });
            try d.project(e, L.down, b(d.buf("act")), b(d.buf("y")), n);
            threads(e, d.pipe(.conv), CArgs{ .rows = @intCast(n), .width = c.hidden, .block = c.block, .groups = groups, .side = 1 }, &.{ b(d.buf("y")), b(d.buf("dyn")), t(L.mlp_conv.base), A, b(d.buf("yc")) }, .{ c.hidden, n, 1 });
        }
        glue(e, d.pipe(.norm), NArgs{ .rows = @intCast(n), .width = c.hidden, .eps = c.eps, .add = 1 }, &.{ b(d.buf("resid")), b(d.buf("yc")), t(d.w.norm), A, b(d.buf("x")) }, .{ n, 1, 1 }, .{ 256, 1, 1 });
        // rows 1-7: the head, each row's top 32 (three passes of 2048), the selector's projection and path
        const slots = n - 1;
        const xr = bo(d.buf("x"), h * 2);
        try d.project(e, m.weights.head, xr, b(d.buf("logits")), slots);
        try d.project(e, d.w.hproj, xr, b(d.buf("hid")), slots);
        const vocab: u32 = m.config.vocab;
        const chunks: u32 = (vocab + 2047) / 2048;
        const TArgs = extern struct { n: u32, stride: u32, chunks: u32, from_ids: u32 };
        glue(e, d.pipe(.topk), TArgs{ .n = vocab, .stride = vocab, .chunks = chunks, .from_ids = 0 }, &.{ b(d.buf("logits")), b(d.buf("tv")), b(d.buf("ti")), A, b(d.buf("tv")), b(d.buf("ti")) }, .{ chunks, slots, 1 }, .{ 1024, 1, 1 });
        const second: u32 = (chunks * 32 + 2047) / 2048;
        glue(e, d.pipe(.topk), TArgs{ .n = chunks * 32, .stride = chunks * 32, .chunks = second, .from_ids = 1 }, &.{ b(d.buf("logits")), b(d.buf("tv")), b(d.buf("ti")), A, b(d.buf("tv2")), b(d.buf("ti2")) }, .{ second, slots, 1 }, .{ 1024, 1, 1 });
        glue(e, d.pipe(.topk), TArgs{ .n = second * 32, .stride = second * 32, .chunks = 1, .from_ids = 1 }, &.{ b(d.buf("logits")), b(d.buf("tv2")), b(d.buf("ti2")), A, b(d.buf("uv")), b(d.buf("ui")) }, .{ 1, slots, 1 }, .{ 1024, 1, 1 });
        const WArgs = extern struct { slots: u32, rank: u32, top: u32, vocab: u32 };
        glue(e, d.pipe(.walk), WArgs{ .slots = @intCast(slots), .rank = c.rank, .top = c.top, .vocab = vocab }, &.{ b(d.buf("uv")), b(d.buf("ui")), b(d.buf("hid")), t(d.w.pred), t(d.w.succ), A, b(d.buf("seq")) }, .{ 1, 1, 1 }, .{ c.rank, 1, 1 });
    }

    /// Each stream's `depth` drafts after `pending` at `position`: a whole block each (its rows see each other).
    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        const c = &d.config;
        for (items) |it| {
            const l = d.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
            if (it.depth == 0 or it.depth > c.block - 1 or it.position == 0) return error.DraftDepth;
            const anchor = it.position - 1;
            if (l.end != anchor) return error.DraftContextGap;
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            try d.encodeBlock(e, l, it.pending, anchor);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| return failed(msg);
            const seq = d.buf("seq").slice(u32, 16);
            for (seq[1..][0..it.depth], l.held[0..it.depth]) |x, *y| {
                if (x >= d.target.config.vocab) return error.DraftNonfinite;
                y.* = x;
            }
            l.held_len = it.depth;
        }
    }

    /// The pipelines' first launch, then a block timed (the depth rule prices a draft at a block's time over its rows).
    fn warm(d: *Drafter) !void {
        const l = try d.newLane();
        defer d.dropLane(l);
        var ms: f64 = 0;
        for (0..3) |run| {
            const t0 = now();
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            try d.encodeBlock(e, l, 2, 0);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| return failed(msg);
            if (run == 2) ms = now() - t0;
        }
        d.step_ms = ms / @as(f64, @floatFromInt(d.config.block - 1));
    }

    fn failed(msg: [*:0]const u8) anyerror {
        std.log.err("minimax_m3 DFlash2 drafter: command buffer failed: {s}", .{msg});
        return error.GpuFailed;
    }
};
