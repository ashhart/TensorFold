//! ricursiveAI's DFlash2 drafter for Ripple on Metal (z-lab's DFlash2DraftModel): 16-row blocks over 5 target taps.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const ks = @import("kernel_sources");
const ck = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const cfg = @import("config.zig");
const fwd = @import("forward.zig");
const dc = @import("dflash2_config.zig");
const Model = @import("model.zig").Model;

const dr = lanes.drafter;
const Stream = lanes.Stream;
const B = fwd.B;
const b = fwd.b;
const bo = fwd.bo;
const t = fwd.t;
const A = fwd.A;
pub const Config = dc.Config;

/// Rows a block takes (the trained block).
pub const max_block = 16;
/// Context rows one absorb pass projects at most (a prompt chunk's).
const absorb_rows = 2048;
/// A key block of the draft attention, and the most a row's window spans (the global layer's ring).
const key_block = 128;
const max_key_blocks = dc.global_window / key_block + 2;

/// One stream's context: each layer's keys and values for the last positions, a ring of Config.ring(layer) slots.
const Lane = struct {
    k: [dc.max_layers]mtl.Buffer,
    v: [dc.max_layers]mtl.Buffer,
    end: u64 = 0, // context rows absorbed: positions [from, end)
    from: u64 = 0, // the first position held (a resumed prompt's drafter context starts there)
    held: [max_block]u32 = @splat(0),
    held_len: u32 = 0,
};

/// The drafter's pipelines, in `Kernel` order (the contract's roll call reads them).
pub const kernel_names = [_][:0]const u8{ "rc_df_norm", "rc_df_norm16", "rc_df_embed", "rc_df_prep", "rc_df_conv", "rc_df_silu", "rc_df_vec", "rc_df_topk", "rc_df_walk" };
const Kernel = enum { norm, norm16, embed, prep, conv, silu, vec, topk, walk };

/// The drafter's scratch, fields in init's order.
const fields = [_][]const u8{ "ids", "resid", "x", "xc", "dyn", "qkv", "q", "attn", "y", "yc", "gu", "act", "logits", "pm", "pl", "po", "tv", "ti", "tv2", "ti2", "uv", "ui", "hid", "seq", "fcx", "hctx", "kv" };

const NArgs = extern struct { rows: u32, width: u32, eps: f32, add: u32 };
const PArgs = extern struct { rows: u32, heads: u32, kv_heads: u32, pos0: u32, stride: u32, q_off: u32, k_off: u32, v_off: u32, ring: u32, rope: u32, eps: f32, theta: f32 };
const VArgs = extern struct { rows: u32, pos0: u32, heads: u32, groups: u32, blocks: u32, chunk: u32, first: u32, scale: f32, hi: u32, window: u32, ring: u32, floor: u32 };
const CArgs = extern struct { rows: u32, width: u32, block: u32, groups: u32, side: u32, taps: u32 };

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    target: *Model,
    config: Config,
    checkpoint: ck.Checkpoint,
    w: dc.Weights,
    pipes: [kernel_names.len]mtl.Pipeline,
    lanes: std.AutoHashMapUnmanaged(*Stream, *Lane) = .empty,
    follow: ?mtl.ResidencySet.Follow = null,
    step_ms: f64 = 0,
    s: [fields.len]mtl.Buffer = undefined,

    /// The drafter's config.json, checked against `target` before any weight is read.
    pub fn readConfig(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, target: *const cfg.Config) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(bytes);
        return dc.parse(gpa, bytes, target);
    }

    pub fn init(gpa: std.mem.Allocator, io: std.Io, target: *Model, dir: []const u8, config: Config) !*Drafter {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .target = target, .config = config, .checkpoint = ck.Checkpoint.init(gpa), .w = undefined, .pipes = undefined };
        errdefer d.checkpoint.deinit();
        const files = try shards.shardFiles(gpa, io, dir);
        defer shards.freeShardFiles(gpa, files);
        for (files) |path| try d.checkpoint.addFile(target.device, path, "");
        d.w = try dc.load(&d.checkpoint, &d.config);
        {
            const lib = try mtl.Library.fromSource(target.device, ks.ricursive_dflash2, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (d.pipes[0..made]) |p| p.deinit();
            for (kernel_names, 0..) |n, i| {
                d.pipes[i] = try mtl.Pipeline.init(target.device, lib, n, false);
                made += 1;
            }
        }
        errdefer for (d.pipes) |p| p.deinit();
        const sizes = scratchSizes(&d.config);
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
        d.checkpoint.deinit();
        d.gpa.destroy(d);
    }

    /// Bytes of the scratch, in `fields` order.
    fn scratchSizes(c: *const Config) [fields.len]usize {
        const h: usize = c.hidden;
        const r: usize = max_block;
        const qw: usize = @as(usize, c.heads) * cfg.head_dim;
        const kvw: usize = @as(usize, c.kv_heads) * cfg.head_dim;
        const vocab: usize = c.vocab;
        const chunks: usize = (vocab + 2047) / 2048;
        const parts: usize = @as(usize, c.heads) * r * max_key_blocks;
        return .{ r * 4, r * h * 4, r * h * 2, r * h * 2, r * c.dyn() * 2, r * (qw + 2 * kvw) * 2, r * qw * 2, r * qw * 2, r * h * 2, r * h * 2, r * 2 * c.ffn * 2, r * c.ffn * 2, r * vocab * 2, parts * 4, parts * 4, parts * cfg.head_dim * 4, r * chunks * 32 * 4, r * chunks * 32 * 4, r * 64 * 4, r * 64 * 4, r * 32 * 4, r * 32 * 4, r * c.rank * 2, 16 * 4, absorb_rows * h * 2, absorb_rows * h * 2, absorb_rows * 2 * kvw * 2 };
    }

    /// Device bytes the drafter holds beside its weights: the scratch.
    pub fn fixedBytes(c: *const Config) usize {
        var n: usize = 0;
        for (scratchSizes(c)) |x| n += x;
        return n;
    }

    /// Device bytes one stream's context takes.
    pub fn laneBytes(c: *const Config) usize {
        var n: usize = 0;
        for (0..c.layers) |l| n += @as(usize, c.ring(l)) * c.kv_heads * cfg.head_dim * 2 * 2;
        return n;
    }

    fn buf(d: *const Drafter, comptime f: []const u8) mtl.Buffer {
        inline for (fields, 0..) |n, i| if (comptime std.mem.eql(u8, n, f)) return d.s[i];
        @compileError("no drafter buffer " ++ f);
    }

    fn pipe(d: *const Drafter, k: Kernel) mtl.Pipeline {
        return d.pipes[@backingInt(k)];
    }

    /// The weights' shards and the scratch, for the host's residency set.
    pub fn buffers(d: *const Drafter, gpa: std.mem.Allocator, list: *std.ArrayList(mtl.Buffer)) !void {
        for (d.checkpoint.shards.items) |sh| try list.append(gpa, sh.buffer);
        try list.appendSlice(gpa, &d.s);
    }

    pub fn setResident(d: *Drafter, hook: ?mtl.ResidencySet.Follow) void {
        if (d.follow) |f| {
            var it = d.lanes.valueIterator();
            while (it.next()) |l| d.laneResident(l.*, f, false);
        }
        d.follow = hook;
        if (hook) |f| {
            var it = d.lanes.valueIterator();
            while (it.next()) |l| d.laneResident(l.*, f, true);
        }
    }

    fn laneResident(d: *const Drafter, l: *const Lane, f: mtl.ResidencySet.Follow, on: bool) void {
        for (0..d.config.layers) |i| for ([_]mtl.Buffer{ l.k[i], l.v[i] }) |x| if (on) f.join(f.ctx, x) else f.leave(f.ctx, x);
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .kind = "DFlash2", .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(ptr: *anyopaque) []const u32 {
        return &cast(ptr).config.taps;
    }

    fn factsFn(ptr: *anyopaque) dr.Facts {
        const d = cast(ptr);
        return .{ .depth = d.config.block - 1, .step_ms = d.step_ms, .plain_guard = true, .stream_bytes = laneBytes(&d.config) };
    }

    fn newLane(d: *Drafter) !*Lane {
        const l = try d.gpa.create(Lane);
        errdefer d.gpa.destroy(l);
        l.* = .{ .k = undefined, .v = undefined };
        var made: usize = 0;
        errdefer for (0..made) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        };
        for (0..d.config.layers) |i| {
            const bytes = @as(usize, d.config.ring(i)) * d.config.kv_heads * cfg.head_dim * 2;
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

    /// y = x W^T over `rows` rows, `ldy` apart: the target's row kernels to 32 rows, its GEMM past them.
    fn project(d: *const Drafter, e: mtl.ComputeEncoder, w: @import("weights.zig").Dense, in: B, out: B, rows: usize, ldy: usize) void {
        fwd.denseRows(e, d.target, w, in, out.buf, out.off, rows, ldy);
    }

    /// The context rows [p0, p0 + rows) from `src` (taps x hidden bf16 each) into every layer's ring.
    fn absorbRows(d: *const Drafter, e: mtl.ComputeEncoder, l: *const Lane, src: B, p0: u64, rows: usize) void {
        const c = &d.config;
        const kvw: usize = @as(usize, c.kv_heads) * cfg.head_dim;
        d.project(e, d.w.fc, src, b(d.buf("fcx")), rows, c.hidden);
        glue(e, d.pipe(.norm16), NArgs{ .rows = @intCast(rows), .width = c.hidden, .eps = c.eps, .add = 0 }, &.{ b(d.buf("fcx")), t(d.w.hidden_norm), A, b(d.buf("hctx")) }, .{ rows, 1, 1 }, .{ 256, 1, 1 });
        for (0..c.layers) |i| {
            const L = &d.w.layers[i];
            // each layer's k and v of the normed taps, k normed and rotated, into the layer's ring
            d.project(e, L.k, b(d.buf("hctx")), b(d.buf("kv")), rows, 2 * kvw);
            d.project(e, L.v, b(d.buf("hctx")), bo(d.buf("kv"), kvw * 2), rows, 2 * kvw);
            const pa = PArgs{ .rows = @intCast(rows), .heads = 0, .kv_heads = c.kv_heads, .pos0 = @intCast(p0), .stride = @intCast(2 * kvw), .q_off = 0, .k_off = 0, .v_off = @intCast(kvw), .ring = c.ring(i), .rope = @intFromBool(c.rope[i]), .eps = c.eps, .theta = c.theta };
            glue(e, d.pipe(.prep), pa, &.{ b(d.buf("kv")), t(L.q_norm), t(L.k_norm), A, b(d.buf("q")), b(l.k[i]), b(l.v[i]) }, .{ c.kv_heads, rows, 1 }, .{ 32, 1, 1 });
        }
    }

    /// Rows [start, start + rows) replace the context from `start`; prompt rows past the longest span are skipped.
    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        const c = &d.config;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = d.target.queue.commandBuffer();
        const e = cb.compute(.serial);
        const width: usize = dc.max_taps * @as(usize, c.hidden) * 2;
        for (items) |it| {
            const l = d.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
            const f = it.features;
            if (f.space != .device or f.dtype != .bf16 or f.row_bytes != width) return error.DraftFeaturesUnsupported;
            if (it.start > l.end or l.end == 0) l.from = it.start; // the rows before it were restored, not absorbed
            const src = mtl.Buffer{ .id = @ptrFromInt(f.buffer) };
            const end = it.start + f.rows;
            const keep_from = if (it.prompt) @max(it.start, it.stream.prompt_len -| (dc.global_window + c.block)) else it.start;
            if (keep_from > it.start) l.from = @max(l.from, keep_from); // a long prompt's oldest rows: never absorbed
            var at: u64 = keep_from;
            while (at < end) {
                const n: usize = @intCast(@min(absorb_rows, end - at));
                d.absorbRows(e, l, bo(src, f.offset + @as(usize, @intCast(at - it.start)) * width), at, n);
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

    /// x (n rows, the sublayer's normed input) through `cv`'s prepare: its coefficients into dyn, side 0 into xc.
    fn convPrepare(d: *const Drafter, e: mtl.ComputeEncoder, cv: dc.Conv, n: usize) void {
        const c = &d.config;
        d.project(e, cv.proj, b(d.buf("x")), b(d.buf("dyn")), n, c.dyn());
        d.conv(e, cv, b(d.buf("x")), b(d.buf("xc")), n, 0);
    }

    fn conv(d: *const Drafter, e: mtl.ComputeEncoder, cv: dc.Conv, in: B, out: B, n: usize, side: u32) void {
        const c = &d.config;
        threads(e, d.pipe(.conv), CArgs{ .rows = @intCast(n), .width = c.hidden, .block = c.block, .groups = c.hidden / c.group, .side = side, .taps = c.conv }, &.{ in, b(d.buf("dyn")), t(cv.base), A, out }, .{ c.hidden, n, 1 });
    }

    /// The block [pending, mask x 15] at positions anchor.. through the drafter; seq[1..] its drafts.
    fn encodeBlock(d: *const Drafter, e: mtl.ComputeEncoder, l: *const Lane, pending: u32, anchor: u64) void {
        const c = &d.config;
        const m = d.target;
        const n: usize = c.block;
        const h: usize = c.hidden;
        const qw: usize = @as(usize, c.heads) * cfg.head_dim;
        const kvw: usize = @as(usize, c.kv_heads) * cfg.head_dim;
        const stride = qw + 2 * kvw;
        const ids = d.buf("ids").slice(u32, max_block);
        ids[0] = pending;
        @memset(ids[1..n], c.mask);
        d.buf("seq").slice(u32, 16)[0] = pending;
        const E = m.weights.embed;
        threads(e, d.pipe(.embed), NArgs{ .rows = @intCast(n), .width = c.hidden, .eps = 0, .add = 0 }, &.{ b(d.buf("ids")), bo(E.buf, E.off), A, b(d.buf("resid")) }, .{ h, n, 1 });
        const hi = anchor + n;
        for (0..c.layers) |i| {
            const L = &d.w.layers[i];
            // input norm (the residual takes the last sublayer's output first, from the second layer on)
            glue(e, d.pipe(.norm), NArgs{ .rows = @intCast(n), .width = c.hidden, .eps = c.eps, .add = @intFromBool(i > 0) }, &.{ b(d.buf("resid")), b(d.buf("yc")), t(L.in_norm), A, b(d.buf("x")) }, .{ n, 1, 1 }, .{ 256, 1, 1 });
            d.convPrepare(e, L.attn_conv, n);
            d.project(e, L.q, b(d.buf("xc")), b(d.buf("qkv")), n, stride);
            d.project(e, L.k, b(d.buf("xc")), bo(d.buf("qkv"), qw * 2), n, stride);
            d.project(e, L.v, b(d.buf("xc")), bo(d.buf("qkv"), (qw + kvw) * 2), n, stride);
            const pa = PArgs{ .rows = @intCast(n), .heads = c.heads, .kv_heads = c.kv_heads, .pos0 = @intCast(anchor), .stride = @intCast(stride), .q_off = 0, .k_off = @intCast(qw), .v_off = @intCast(qw + kvw), .ring = c.ring(i), .rope = @intFromBool(c.rope[i]), .eps = c.eps, .theta = c.theta };
            glue(e, d.pipe(.prep), pa, &.{ b(d.buf("qkv")), t(L.q_norm), t(L.k_norm), A, b(d.buf("q")), b(l.k[i]), b(l.v[i]) }, .{ c.heads + c.kv_heads, n, 1 }, .{ 32, 1, 1 });
            const span = c.span(i);
            const lo = @max((anchor + 1) -| span, l.from);
            const first = lo / key_block;
            const blocks = (hi - 1) / key_block - first + 1;
            const va = VArgs{ .rows = @intCast(n), .pos0 = @intCast(anchor), .heads = c.heads, .groups = c.kv_heads, .blocks = @intCast(blocks), .chunk = key_block, .first = @intCast(first), .scale = 1.0 / @sqrt(@as(f32, cfg.head_dim)), .hi = @intCast(hi), .window = span, .ring = c.ring(i), .floor = @intCast(l.from) };
            glue(e, d.pipe(.vec), va, &.{ b(d.buf("q")), b(l.k[i]), b(l.v[i]), A, b(d.buf("pm")), b(d.buf("pl")), b(d.buf("po")) }, .{ c.kv_heads, n, blocks }, .{ 32, c.heads / c.kv_heads, 1 });
            glue(e, m.ll(.merge_vec), va, &.{ b(d.buf("pm")), b(d.buf("pl")), b(d.buf("po")), A, b(d.buf("attn")) }, .{ c.heads, n, 1 }, .{ 1024, 1, 1 });
            d.project(e, L.o, b(d.buf("attn")), b(d.buf("y")), n, h);
            // the attention conv's finish (side 1, its prepare's coefficients), then the post-attention norm
            d.conv(e, L.attn_conv, b(d.buf("y")), b(d.buf("yc")), n, 1);
            glue(e, d.pipe(.norm), NArgs{ .rows = @intCast(n), .width = c.hidden, .eps = c.eps, .add = 1 }, &.{ b(d.buf("resid")), b(d.buf("yc")), t(L.post_norm), A, b(d.buf("x")) }, .{ n, 1, 1 }, .{ 256, 1, 1 });
            d.convPrepare(e, L.mlp_conv, n);
            d.project(e, L.gate, b(d.buf("xc")), b(d.buf("gu")), n, 2 * c.ffn);
            d.project(e, L.up, b(d.buf("xc")), bo(d.buf("gu"), @as(usize, c.ffn) * 2), n, 2 * c.ffn);
            threads(e, d.pipe(.silu), extern struct { rows: u32, width: u32 }{ .rows = @intCast(n), .width = c.ffn }, &.{ b(d.buf("gu")), A, b(d.buf("act")) }, .{ c.ffn, n, 1 });
            d.project(e, L.down, b(d.buf("act")), b(d.buf("y")), n, h);
            d.conv(e, L.mlp_conv, b(d.buf("y")), b(d.buf("yc")), n, 1);
        }
        glue(e, d.pipe(.norm), NArgs{ .rows = @intCast(n), .width = c.hidden, .eps = c.eps, .add = 1 }, &.{ b(d.buf("resid")), b(d.buf("yc")), t(d.w.norm), A, b(d.buf("x")) }, .{ n, 1, 1 }, .{ 256, 1, 1 });
        // rows 1-15: the target's head, each row's top 32 (three passes of 2048), the selector's projection and path
        const slots = n - 1;
        const xr = bo(d.buf("x"), h * 2);
        d.project(e, m.weights.head, xr, b(d.buf("logits")), slots, c.vocab);
        d.project(e, d.w.hproj, xr, b(d.buf("hid")), slots, c.rank);
        const chunks: u32 = (c.vocab + 2047) / 2048;
        const TArgs = extern struct { n: u32, stride: u32, chunks: u32, from_ids: u32 };
        glue(e, d.pipe(.topk), TArgs{ .n = c.vocab, .stride = c.vocab, .chunks = chunks, .from_ids = 0 }, &.{ b(d.buf("logits")), b(d.buf("tv")), b(d.buf("ti")), A, b(d.buf("tv")), b(d.buf("ti")) }, .{ chunks, slots, 1 }, .{ 1024, 1, 1 });
        const second: u32 = (chunks * 32 + 2047) / 2048;
        glue(e, d.pipe(.topk), TArgs{ .n = chunks * 32, .stride = chunks * 32, .chunks = second, .from_ids = 1 }, &.{ b(d.buf("logits")), b(d.buf("tv")), b(d.buf("ti")), A, b(d.buf("tv2")), b(d.buf("ti2")) }, .{ second, slots, 1 }, .{ 1024, 1, 1 });
        glue(e, d.pipe(.topk), TArgs{ .n = second * 32, .stride = second * 32, .chunks = 1, .from_ids = 1 }, &.{ b(d.buf("logits")), b(d.buf("tv2")), b(d.buf("ti2")), A, b(d.buf("uv")), b(d.buf("ui")) }, .{ 1, slots, 1 }, .{ 1024, 1, 1 });
        const WArgs = extern struct { slots: u32, rank: u32, top: u32, vocab: u32 };
        glue(e, d.pipe(.walk), WArgs{ .slots = @intCast(slots), .rank = c.rank, .top = c.top, .vocab = c.vocab }, &.{ b(d.buf("uv")), b(d.buf("ui")), b(d.buf("hid")), t(d.w.pred), t(d.w.succ), A, b(d.buf("seq")) }, .{ 1, 1, 1 }, .{ c.rank, 1, 1 });
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
            d.encodeBlock(e, l, it.pending, anchor);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| return failed(msg);
            const seq = d.buf("seq").slice(u32, 16);
            for (seq[1..][0..it.depth], l.held[0..it.depth]) |x, *y| {
                if (x >= c.vocab) return error.DraftNonfinite;
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
            const t0 = msNow();
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            d.encodeBlock(e, l, 2, 0);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| return failed(msg);
            if (run == 2) ms = msNow() - t0;
        }
        d.step_ms = ms / @as(f64, @floatFromInt(d.config.block - 1));
    }

    fn failed(msg: [*:0]const u8) anyerror {
        std.log.err("ricursive DFlash2 drafter: command buffer failed: {s}", .{msg});
        return error.GpuFailed;
    }
};

fn msNow() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1e3 + @as(f64, @floatFromInt(ts.nsec)) / 1e6;
}
