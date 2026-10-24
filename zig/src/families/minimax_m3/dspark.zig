//! MiniMax-M3.1's DSpark drafter (ref/dspark.py): 7-row blocks through five layers over the target's taps.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const ks = @import("kernel_sources");
const ck = @import("../../core/checkpoint_metal.zig");
const wts = @import("weights.zig");
const fwd = @import("forward.zig");
const Model = @import("model.zig").Model;

const dr = lanes.drafter;
const Stream = lanes.Stream;
const B = fwd.B;
const b = fwd.b;
const bo = fwd.bo;
const t = fwd.t;
const A = fwd.A;

/// The drafter's shape from config.json's `dspark` (tools/convert.py writes it).
pub const Config = struct {
    layers: u32 = 5,
    heads: u32 = 64,
    kv_heads: u32 = 4,
    hidden: u32 = 6144,
    ffn: u32 = 12288,
    window: u32 = 4096,
    block: u32 = 7,
    noise: u32 = 200058,
    rank: u32 = 256,
    taps: [5]u32 = .{ 3, 17, 31, 45, 59 },
    rotary: u32 = 64,
    theta: f32 = 1e7,
    eps: f32 = 1e-6,
    alpha: f32 = 1.702,
    limit: f32 = 7.0,

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
        const d = doc.value.object.get("dspark") orelse return null;
        if (d != .object) return error.BadDSparkConfig;
        const o = d.object;
        const int = struct {
            fn of(x: std.json.ObjectMap, k: []const u8) !u32 {
                const v = x.get(k) orelse return error.BadDSparkConfig;
                if (v != .integer or v.integer <= 0) return error.BadDSparkConfig;
                return @intCast(v.integer);
            }
            fn f(x: std.json.ObjectMap, k: []const u8) !f32 {
                const v = x.get(k) orelse return error.BadDSparkConfig;
                return switch (v) {
                    .integer => |i| @floatFromInt(i),
                    .float => |q| @floatCast(q),
                    else => error.BadDSparkConfig,
                };
            }
        };
        var c = Config{
            .layers = try int.of(o, "num_hidden_layers"),
            .heads = try int.of(o, "num_attention_heads"),
            .kv_heads = try int.of(o, "num_key_value_heads"),
            .hidden = try int.of(o, "hidden_size"),
            .ffn = try int.of(o, "dspark_ffn_hidden_size"),
            .window = try int.of(o, "sliding_window"),
            .block = try int.of(o, "dspark_block_size"),
            .noise = try int.of(o, "dspark_noise_token_id"),
            .rank = try int.of(o, "dspark_markov_rank"),
            .rotary = try int.of(o, "rotary_dim"),
            .theta = try int.f(o, "rope_theta"),
            .eps = try int.f(o, "rms_norm_eps"),
            .alpha = try int.f(o, "swiglu_alpha"),
            .limit = try int.f(o, "swiglu_limit"),
        };
        const taps = o.get("dspark_target_layer_ids") orelse return error.BadDSparkConfig;
        if (taps != .array or taps.array.items.len != c.taps.len) return error.UnsupportedDSpark;
        for (taps.array.items, &c.taps) |x, *y| {
            if (x != .integer or x.integer < 0) return error.BadDSparkConfig;
            y.* = @intCast(x.integer);
        }
        if ((try int.of(o, "head_dim")) != 128 or c.layers > 8 or c.block > max_block or c.rank != 256 or c.heads % c.kv_heads != 0) return error.UnsupportedDSpark;
        return c;
    }
};

/// Rows a block takes at most (the trained block is 7).
pub const max_block = 8;
/// Context rows one absorb pass projects at most (a prompt chunk's).
const absorb_rows = 2048;

const Layer = struct { in_norm: wts.Tensor, post_norm: wts.Tensor, q_norm: wts.Tensor, k_norm: wts.Tensor, qkv: wts.Linear, o: wts.Linear, gu: wts.Linear, down: wts.Linear };

const Weights = struct {
    fc: wts.Linear,
    hidden_norm: wts.Tensor,
    norm: wts.Tensor,
    ctx_kv: wts.Linear,
    w1: wts.Tensor,
    w2: wts.Linear,
    layers: [8]Layer,
};

fn tensor(c: *const ck.Checkpoint, name: []const u8) !wts.Tensor {
    const x = try c.get(name);
    if (x.dtype != .bf16) return error.UnexpectedDSparkTensor;
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
    var w: Weights = .{
        .fc = try affine(ckp, "dspark.fc", h, h * c.taps.len),
        .hidden_norm = try tensor(ckp, "dspark.hidden_norm.weight"),
        .norm = try tensor(ckp, "dspark.norm.weight"),
        .ctx_kv = try affine(ckp, "dspark.ctx_kv_proj", c.layers * 2 * kvw, h),
        .w1 = try tensor(ckp, "dspark.markov_w1.weight"),
        .w2 = try affine(ckp, "dspark.markov_w2", vocab, c.rank),
        .layers = undefined,
    };
    if (w.w1.count() != vocab * c.rank or w.w2.bits != 8 or w.w2.group != 64) return error.UnexpectedDSparkTensor;
    for (w.layers[0..c.layers], 0..) |*l, i| {
        var pb: [64]u8 = undefined;
        var nb: [128]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "dspark.layers.{d}.", .{i});
        const name = struct {
            fn of(buf: []u8, a: []const u8, z: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, "{s}{s}", .{ a, z });
            }
        }.of;
        l.* = .{
            .in_norm = try tensor(ckp, try name(&nb, p, "input_layernorm.weight")),
            .post_norm = try tensor(ckp, try name(&nb, p, "post_attention_layernorm.weight")),
            .q_norm = try tensor(ckp, try name(&nb, p, "self_attn.q_norm.weight")),
            .k_norm = try tensor(ckp, try name(&nb, p, "self_attn.k_norm.weight")),
            .qkv = try affine(ckp, try name(&nb, p, "self_attn.qkv_proj"), (c.heads + 2 * c.kv_heads) * 128, h),
            .o = try affine(ckp, try name(&nb, p, "self_attn.o_proj"), h, c.heads * 128),
            .gu = try affine(ckp, try name(&nb, p, "mlp.gate_up_proj"), 2 * @as(usize, c.ffn), h),
            .down = try affine(ckp, try name(&nb, p, "mlp.down_proj"), h, c.ffn),
        };
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

const names = [_][:0]const u8{ "mm_norm16", "mm_draft_prep", "mm_draft_vec", "mm_markov_step", "mm_markov_pick" };
const Kernel = enum { norm16, prep, vec, markov_step, markov_pick };

/// The drafter's scratch, fields in init's order.
const fields = [_][]const u8{ "ids", "resid", "x", "qkv", "q", "attn", "y", "gu", "act", "logits", "pm", "pl", "po", "best", "best_id", "seq", "fcx", "hctx", "kv", "sums" };

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
        if (c.hidden != target.config.hidden) return error.DSparkForAnotherTarget;
        for (c.taps) |l| if (l >= target.config.layers) return error.DSparkForAnotherTarget;
        {
            const lib = try mtl.Library.fromSource(target.device, ks.minimax_m3_dspark, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
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
        const sizes = [fields.len]usize{ r * 4, r * h * 4, r * h * 2, r * (qw + 2 * kvw) * 2, r * qw * 2, r * qw * 2, r * h * 2, r * 2 * c.ffn * 2, r * c.ffn * 2, r * vocab * 2, c.heads * r * nblk * 4, c.heads * r * nblk * 4, c.heads * r * nblk * 128 * 4, (vocab + 255) / 256 * 4, (vocab + 255) / 256 * 4, 16 * 4, absorb_rows * h * 2, absorb_rows * h * 2, absorb_rows * c.layers * 2 * kvw * 2, absorb_rows * h * c.taps.len / 32 * 4 };
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
        return .{ .depth = d.config.block, .step_ms = d.step_ms, .plain_guard = true, .stream_bytes = d.laneBytes() };
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

    fn project(d: *const Drafter, e: mtl.ComputeEncoder, w: wts.Linear, in: B, out: B, rows: usize) !void {
        // a block's rows on the tensor units: each weight (the target head's too) read once for all of them
        return fwd.projectRows(e, d.target, d.buf("sums"), w, in, out, rows, 3);
    }

    /// The context rows [p0, p0 + rows) from `src` (taps x hidden bf16 each) into every layer's ring.
    fn absorbRows(d: *const Drafter, e: mtl.ComputeEncoder, l: *const Lane, src: B, p0: u64, rows: usize) !void {
        const c = &d.config;
        const kvw = c.kv_heads * 128;
        try d.project(e, d.w.fc, src, b(d.buf("fcx")), rows);
        glue(e, d.pipe(.norm16), extern struct { rows: u32, width: u32, eps: f32 }{ .rows = @intCast(rows), .width = c.hidden, .eps = c.eps }, &.{ b(d.buf("fcx")), t(d.w.hidden_norm), A, b(d.buf("hctx")) }, .{ rows, 1, 1 }, .{ 256, 1, 1 });
        try d.project(e, d.w.ctx_kv, b(d.buf("hctx")), b(d.buf("kv")), rows);
        for (0..c.layers) |i| {
            const pa = PArgs{ .rows = @intCast(rows), .heads = 0, .kv_heads = c.kv_heads, .pos0 = @intCast(p0), .rotary = c.rotary, .stride = c.layers * 2 * kvw, .q_off = 0, .k_off = @intCast(i * 2 * kvw), .v_off = @intCast(i * 2 * kvw + kvw), .ring = c.ring(), .eps = c.eps, .theta = c.theta };
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
            // only the last window (and a block) of a long pass is ever read again
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

    const PArgs = extern struct { rows: u32, heads: u32, kv_heads: u32, pos0: u32, rotary: u32, stride: u32, q_off: u32, k_off: u32, v_off: u32, ring: u32, eps: f32, theta: f32 };
    const VArgs = extern struct { rows: u32, pos0: u32, heads: u32, groups: u32, blocks: u32, first: u32, hi: u32, window: u32, ring: u32, scale: f32 };
    const MArgs = extern struct { rows: u32 = 0, width: u32 = 0, heads: u32 = 0, dim: u32 = 0, pos: u32 = 0, stride: u32 = 0, bits: u32 = 0, group: u32 = 0, eps: f32 = 0, alpha: f32 = 0, limit: f32 = 0, scale: f32 = 0 };

    /// The block [pending, noise x (block - 1)] at positions anchor.. through the drafter; seq[1..] its drafts.
    fn encodeBlock(d: *const Drafter, e: mtl.ComputeEncoder, l: *const Lane, pending: u32, anchor: u64) !void {
        const c = &d.config;
        const m = d.target;
        const n: usize = c.block;
        const h: usize = c.hidden;
        const qw: usize = c.heads * 128;
        const kvw: usize = c.kv_heads * 128;
        const ids = d.buf("ids").slice(u32, max_block);
        ids[0] = pending;
        @memset(ids[1..n], c.noise);
        d.buf("seq").slice(u32, 16)[0] = pending;
        const E = m.weights.embed;
        glue(e, m.pipe(.embed), MArgs{ .rows = @intCast(n), .width = c.hidden, .bits = E.bits, .group = @intCast(E.group) }, &.{ b(d.buf("ids")), bo(E.w, E.w_off), bo(E.scales, E.s_off), bo(E.biases, E.b_off), A, b(d.buf("resid")) }, .{ (h + 255) / 256, n, 1 }, .{ 256, 1, 1 });
        const nm = MArgs{ .rows = @intCast(n), .width = c.hidden, .eps = c.eps };
        glue(e, m.pipe(.norm), nm, &.{ b(d.buf("resid")), t(d.w.layers[0].in_norm), A, b(d.buf("x")) }, .{ n, 1, 1 }, .{ 256, 1, 1 });
        const lo = @max((anchor + 1) -| c.window, l.from);
        const hi = anchor + n;
        const first = lo / 128;
        const blocks = (hi - 1) / 128 - first + 1;
        for (0..c.layers) |i| {
            const L = &d.w.layers[i];
            try d.project(e, L.qkv, b(d.buf("x")), b(d.buf("qkv")), n);
            const pa = PArgs{ .rows = @intCast(n), .heads = c.heads, .kv_heads = c.kv_heads, .pos0 = @intCast(anchor), .rotary = c.rotary, .stride = @intCast(qw + 2 * kvw), .q_off = 0, .k_off = @intCast(qw), .v_off = @intCast(qw + kvw), .ring = c.ring(), .eps = c.eps, .theta = c.theta };
            glue(e, d.pipe(.prep), pa, &.{ b(d.buf("qkv")), t(L.q_norm), t(L.k_norm), A, b(d.buf("q")), b(l.k[i]), b(l.v[i]) }, .{ c.heads + c.kv_heads, n, 1 }, .{ 32, 1, 1 });
            const va = VArgs{ .rows = @intCast(n), .pos0 = @intCast(anchor), .heads = c.heads, .groups = c.kv_heads, .blocks = @intCast(blocks), .first = @intCast(first), .hi = @intCast(hi), .window = c.window, .ring = c.ring(), .scale = 1.0 / @sqrt(128.0) };
            glue(e, d.pipe(.vec), va, &.{ b(d.buf("q")), b(l.k[i]), b(l.v[i]), A, b(d.buf("pm")), b(d.buf("pl")), b(d.buf("po")) }, .{ c.kv_heads, n, blocks }, .{ 32, c.heads / c.kv_heads, 1 });
            const ma = extern struct { rows: u32, pos: u32, heads: u32, groups: u32, topk: u32, scale: f32, sel: u32, all: u32 }{ .rows = @intCast(n), .pos = 0, .heads = c.heads, .groups = c.kv_heads, .topk = @intCast(blocks), .scale = 0, .sel = 0, .all = 0 };
            glue(e, m.pipe(.merge_vec), ma, &.{ b(d.buf("pm")), b(d.buf("pl")), b(d.buf("po")), A, b(d.buf("attn")) }, .{ c.heads, n, 1 }, .{ 32, 1, 1 });
            try d.project(e, L.o, b(d.buf("attn")), b(d.buf("y")), n);
            glue(e, m.pipe(.add_norm), nm, &.{ b(d.buf("resid")), b(d.buf("y")), t(L.post_norm), A, b(d.buf("x")) }, .{ n, 1, 1 }, .{ 256, 1, 1 });
            try d.project(e, L.gu, b(d.buf("x")), b(d.buf("gu")), n);
            const act = MArgs{ .rows = @intCast(n), .width = c.ffn, .stride = 2 * c.ffn, .alpha = c.alpha, .limit = c.limit };
            e.setPipeline(m.pipe(.swiglu));
            e.setBuffer(d.buf("gu"), 0, 0);
            e.setValue(act, 1);
            e.setBuffer(d.buf("act"), 0, 2);
            e.dispatchThreads(mtl.Size.of(c.ffn, n, 1), mtl.Size.of(256, 1, 1));
            try d.project(e, L.down, b(d.buf("act")), b(d.buf("y")), n);
            const next = if (i + 1 < c.layers) d.w.layers[i + 1].in_norm else d.w.norm;
            glue(e, m.pipe(.add_norm), nm, &.{ b(d.buf("resid")), b(d.buf("y")), t(next), A, b(d.buf("x")) }, .{ n, 1, 1 }, .{ 256, 1, 1 });
        }
        try d.project(e, m.weights.head, b(d.buf("x")), b(d.buf("logits")), n);
        const vocab: u32 = m.config.vocab;
        const groups: u32 = (vocab + 255) / 256;
        const w2 = d.w.w2;
        for (0..n) |j| {
            const sa = extern struct { vocab: u32, rank: u32, step: u32 }{ .vocab = vocab, .rank = c.rank, .step = @intCast(j) };
            glue(e, d.pipe(.markov_step), sa, &.{ b(d.buf("logits")), t(d.w.w1), bo(w2.w, w2.w_off), bo(w2.scales, w2.s_off), bo(w2.biases, w2.b_off), b(d.buf("seq")), A, b(d.buf("best")), b(d.buf("best_id")) }, .{ groups, 1, 1 }, .{ 256, 1, 1 });
            glue(e, d.pipe(.markov_pick), [2]u32{ groups, @intCast(j) }, &.{ b(d.buf("best")), b(d.buf("best_id")), A, b(d.buf("seq")) }, .{ 1, 1, 1 }, .{ 1024, 1, 1 });
        }
    }

    /// Each stream's `depth` drafts after `pending` at `position`: a whole block each (its rows see each other).
    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        const c = &d.config;
        for (items) |it| {
            const l = d.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
            if (it.depth == 0 or it.depth > c.block or it.position == 0) return error.DraftDepth;
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
            var ts0: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &ts0);
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            try d.encodeBlock(e, l, 2, 0);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| return failed(msg);
            var ts1: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &ts1);
            if (run == 2) ms = @as(f64, @floatFromInt(ts1.sec - ts0.sec)) * 1e3 + @as(f64, @floatFromInt(ts1.nsec - ts0.nsec)) / 1e6;
        }
        d.step_ms = ms / @as(f64, @floatFromInt(d.config.block));
    }

    fn failed(msg: [*:0]const u8) anyerror {
        std.log.err("minimax_m3 drafter: command buffer failed: {s}", .{msg});
        return error.GpuFailed;
    }
};

pub fn now() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1e3 + @as(f64, @floatFromInt(ts.nsec)) / 1e6;
}
