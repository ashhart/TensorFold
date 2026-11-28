//! Independent serial 1-token Metal reference over on-disk Inkling 2 Small EXL3 packs.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const st = @import("core").safetensors;
const exl3 = @import("core").exl3_format;
const exl3_proj = @import("core").exl3_projection;
const cfg = @import("../config.zig");
const ap = @import("attention_plan.zig");
const attention = @import("attention.zig");
const glue = @import("glue.zig");
const small_ops = @import("small_ops.zig");

const H: usize = 4608;
const Q_HEADS: usize = 48;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const Q_WIDTH: usize = Q_HEADS * HEAD_DIM;
const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;
const D_REL: usize = 16;
const REL_WIDTH: usize = Q_HEADS * D_REL;
const INTER: usize = 2304;
const EXPERTS: usize = 384;
const TOPK: usize = 8;
const BLOCKS: usize = 8;
const TAPS: usize = 8;
const MAX_ATTN_STRIDE: usize = 64;
const Ref = @import("inkling2_metal_ref").Ref;
const RArgs = extern struct { count: u32, boundary: u32 = 0 };
const NormArgs = extern struct { rows: u32, width: u32, heads: u32 = 0, bits: u32 = 0, group: u32 = 0, eps: f32 };
const DenseArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };

pub const Capture = struct {
    gpa: std.mem.Allocator,
    steps: usize,
    vocab: usize,
    logits: []u32,
    taps: []u16,
    tokens: []u32,
    pub fn deinit(c: *Capture) void {
        c.gpa.free(c.logits);
        c.gpa.free(c.taps);
        c.gpa.free(c.tokens);
    }
};

const DenseEntry = struct { buf: mtl.Buffer, off: usize, bytes: usize, dtype: st.DType, rank: u8, shape: [st.max_rank]usize };

const RawDense = struct {
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(mtl.Buffer) = .empty,
    map: std.StringHashMapUnmanaged(DenseEntry) = .empty,

    fn deinit(self: *RawDense) void {
        var it = self.map.keyIterator();
        while (it.next()) |k| self.gpa.free(k.*);
        self.map.deinit(self.gpa);
        for (self.bufs.items) |b| b.deinit();
        self.bufs.deinit(self.gpa);
    }

    fn loadFile(self: *RawDense, device: mtl.Device, path: [:0]const u8) !void {
        const fd = platform.c.open(path, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.OpenFailed;
        defer _ = platform.c.close(fd);
        var head: [8]u8 = undefined;
        try preadAll(fd, &head, 0);
        const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
        if (end < 8) return error.BadSafetensors;
        const fsize = std.math.cast(usize, end) orelse return error.BadSafetensors;
        const hlen = try st.headerSize(std.mem.readInt(u64, &head, .little), fsize);
        var arena = std.heap.ArenaAllocator.init(self.gpa);
        defer arena.deinit();
        const hj = try arena.allocator().alloc(u8, hlen);
        try preadAll(fd, hj, 8);
        const dstart = 8 + hlen;
        const dlen = fsize - dstart;
        const hdr = try st.parseHeaderPrefix(arena.allocator(), hj, dlen, "model.llm.");
        if (hdr.count() == 0) return;
        var lo: usize = dlen;
        var hi: usize = 0;
        for (hdr.values()) |e| {
            lo = @min(lo, e.begin);
            hi = @max(hi, e.end);
        }
        const span = hi - lo;
        const buf = try device.buffer(@max(span, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer buf.deinit();
        try preadAll(fd, buf.contents()[0..span], dstart + lo);
        try self.bufs.append(self.gpa, buf);
        var it = hdr.iterator();
        while (it.next()) |kv| {
            const e = kv.value_ptr.*;
            const kdup = try self.gpa.dupe(u8, kv.key_ptr.*);
            errdefer self.gpa.free(kdup);
            try self.map.put(self.gpa, kdup, .{ .buf = buf, .off = e.begin - lo, .bytes = e.end - e.begin, .dtype = e.dtype, .rank = e.rank, .shape = e.shape });
        }
    }

    fn get(self: *const RawDense, comptime fmt: []const u8, args: anytype, dtype: st.DType, dims: []const usize) !Ref {
        var nb: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&nb, fmt, args);
        const e = self.map.get(name) orelse return error.MissingOracleTensor;
        if (e.dtype != dtype or e.rank != dims.len) return error.WrongOracleTensor;
        for (dims, 0..) |d, i| if (e.shape[i] != d) return error.WrongOracleTensor;
        return .{ .buf = e.buf, .off = e.off };
    }
};

fn preadAll(fd: platform.c.fd_t, dst: []u8, at: usize) !void {
    const chunk: usize = 1 << 30;
    var done: usize = 0;
    while (done < dst.len) {
        const n = platform.c.pread(fd, dst.ptr + done, @min(dst.len - done, chunk), @intCast(at + done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
}

fn fileExists(io: std.Io, path: []const u8) !bool {
    const f = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    f.close(io);
    return true;
}

const LayerExp = struct { w13: mtl.Buffer, w2: mtl.Buffer, shards: []mtl.objc.Id };

fn buildDesc(hdr: st.Header, buf: mtl.Buffer, e: u32, comptime proj: []const u8, k: usize, n: usize) !exl3_proj.ExpertDescriptor {
    var nb: [64]u8 = undefined;
    const te = hdr.get(std.fmt.bufPrint(&nb, "experts.{d}." ++ proj ++ ".trellis", .{e}) catch unreachable) orelse return error.MissingTensor;
    if (te.dtype != .i16 or te.rank != 3 or te.shape[0] != k / 16 or te.shape[1] != n / 16 or te.shape[2] == 0 or te.shape[2] % 16 != 0)
        return error.WrongTensor;
    const bits = te.shape[2] / 16;
    if (bits < 2 or bits > 4) return error.UnsupportedWidth;
    const su = hdr.get(std.fmt.bufPrint(&nb, "experts.{d}." ++ proj ++ ".suh", .{e}) catch unreachable) orelse return error.MissingTensor;
    const sv = hdr.get(std.fmt.bufPrint(&nb, "experts.{d}." ++ proj ++ ".svh", .{e}) catch unreachable) orelse return error.MissingTensor;
    const me = hdr.get(std.fmt.bufPrint(&nb, "experts.{d}." ++ proj ++ ".mul1", .{e}) catch unreachable) orelse return error.UnsupportedCodebook;
    if (me.dtype != .i32 or me.rank != 0 or me.end - me.begin != 4) return error.UnsupportedCodebook;
    if (std.mem.readInt(u32, buf.contents()[me.begin..][0..4], .little) != exl3.Mul1Mul) return error.UnsupportedCodebook;
    return exl3_proj.descriptor(.{
        .trellis = .{ .buffer = buf, .offset = te.begin },
        .suh = .{ .buffer = buf, .offset = su.begin },
        .svh = .{ .buffer = buf, .offset = sv.begin },
        .k = k,
        .n = n,
        .width = .{ .halves = @as(u16, @intCast(bits)) * 2 },
        .codebook = .mul1,
        .suh_format = .f16,
        .svh_format = .f16,
    });
}

fn loadLayerExperts(gpa: std.mem.Allocator, device: mtl.Device, root: []const u8, lo: std.json.ObjectMap, bufs: *std.ArrayList(mtl.Buffer), cache: *std.StringHashMapUnmanaged(mtl.Buffer)) !LayerExp {
    const batches = (lo.get("batches") orelse return error.BadManifest).array.items;
    const w13_buf = try device.buffer(EXPERTS * @sizeOf(exl3_proj.ExpertDescriptor), mtl.ResourceOptions.shared);
    errdefer w13_buf.deinit();
    const w2_buf = try device.buffer(EXPERTS * @sizeOf(exl3_proj.ExpertDescriptor), mtl.ResourceOptions.shared);
    errdefer w2_buf.deinit();
    const shards = try gpa.alloc(mtl.objc.Id, batches.len);
    errdefer gpa.free(shards);
    const w13s = w13_buf.slice(exl3_proj.ExpertDescriptor, EXPERTS);
    const w2s = w2_buf.slice(exl3_proj.ExpertDescriptor, EXPERTS);
    for (batches, 0..) |bv, bi| {
        const bo = bv.object;
        const start: u32 = @intCast(bo.get("start").?.integer);
        const end_e: u32 = @intCast(bo.get("end").?.integer);
        const spath = try std.fs.path.joinZ(gpa, &.{ root, bo.get("file").?.string });
        defer gpa.free(spath);
        const fd = platform.c.open(spath, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.OpenFailed;
        defer _ = platform.c.close(fd);
        var head: [8]u8 = undefined;
        try preadAll(fd, &head, 0);
        const fsize: usize = @intCast(platform.c.lseek(fd, 0, platform.c.SEEK.END));
        const hlen = try st.headerSize(std.mem.readInt(u64, &head, .little), fsize);
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const hj = try arena.allocator().alloc(u8, hlen);
        try preadAll(fd, hj, 8);
        const dstart = 8 + hlen;
        const dlen = fsize - dstart;
        const hdr = try st.parseHeader(arena.allocator(), hj, dlen);
        const sbuf = cache.get(spath) orelse blk: {
            const b = try device.buffer(@max(dlen, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            {
                errdefer b.deinit();
                try preadAll(fd, b.contents()[0..dlen], dstart);
                try bufs.append(gpa, b);
            }
            const key = try gpa.dupe(u8, spath);
            errdefer gpa.free(key);
            try cache.put(gpa, key, b);
            break :blk b;
        };
        shards[bi] = sbuf.id;
        for (start..end_e) |ei| {
            w13s[ei] = try buildDesc(hdr, sbuf, @intCast(ei), "w13", H, 2 * INTER);
            w2s[ei] = try buildDesc(hdr, sbuf, @intCast(ei), "w2", INTER, H);
        }
    }
    return .{ .w13 = w13_buf, .w2 = w2_buf, .shards = shards };
}

const Pipes = struct {
    attn: attention.Kernels,
    glue_p: glue.Pipelines,
    small: small_ops.Pipelines,
    exl3_p: exl3_proj.Pipelines,
    matvec1: mtl.Pipeline,
    norm: mtl.Pipeline,
    copy_bf16: mtl.Pipeline,
    widen: mtl.Pipeline,
    mlp_partial: mtl.Pipeline,

    fn load(gpa: std.mem.Allocator, device: mtl.Device) !Pipes {
        var p: Pipes = undefined;
        p.attn = try attention.Kernels.init(gpa, device);
        errdefer p.attn.deinit();
        p.glue_p = try glue.Pipelines.load(device);
        errdefer p.glue_p.deinit();
        p.small = try small_ops.Pipelines.load(device);
        errdefer p.small.deinit();
        p.exl3_p = try exl3_proj.Pipelines.load(device);
        errdefer p.exl3_p.deinit();
        const slib = try mtl.Library.fromSource(device, sources.spark25_glue, .{
            .language = mtl.CompileOptions.mlx().language,
            .math_functions = .precise,
        });
        defer slib.deinit();
        p.matvec1 = try mtl.Pipeline.init(device, slib, "sp_matvec_1", false);
        errdefer p.matvec1.deinit();
        p.norm = try mtl.Pipeline.init(device, slib, "sp_norm", false);
        errdefer p.norm.deinit();
        const rlib = try mtl.Library.fromSource(device, sources.inkling2_runtime, mtl.CompileOptions.mlx());
        defer rlib.deinit();
        p.copy_bf16 = try mtl.Pipeline.init(device, rlib, "tf_ink2_copy_bf16", false);
        errdefer p.copy_bf16.deinit();
        p.widen = try mtl.Pipeline.init(device, rlib, "tf_ink2_widen", false);
        errdefer p.widen.deinit();
        p.mlp_partial = try mtl.Pipeline.init(device, rlib, "tf_ink2_mlp_partial", false);
        return p;
    }

    fn deinit(p: *Pipes) void {
        p.mlp_partial.deinit();
        p.widen.deinit();
        p.copy_bf16.deinit();
        p.norm.deinit();
        p.matvec1.deinit();
        p.exl3_p.deinit();
        p.small.deinit();
        p.glue_p.deinit();
        p.attn.deinit();
    }
};

fn runOp(pipe: mtl.Pipeline, e: mtl.ComputeEncoder, refs: []const Ref, count: usize) void {
    e.setPipeline(pipe);
    for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(RArgs{ .count = @intCast(count), .boundary = 0 }, refs.len);
    e.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
}
fn matvec1(p: *const Pipes, e: mtl.ComputeEncoder, w: Ref, x: Ref, y: Ref, n: usize, k: usize) void {
    e.setPipeline(p.matvec1);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(w.buf, w.off, 1);
    e.setValue(DenseArgs{ .rows = 1, .n = @intCast(n), .k = @intCast(k), .ldy = @intCast(n) }, 2);
    e.setBuffer(y.buf, y.off, 3);
    e.dispatchGroups(mtl.Size.of((n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
}
fn rmsNorm1(p: *const Pipes, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref) void {
    e.setPipeline(p.norm);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(w.buf, w.off, 1);
    e.setValue(NormArgs{ .rows = 1, .width = H, .eps = 1e-6 }, 2);
    e.setBuffer(y.buf, y.off, 3);
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
}
fn ar1(p: *const Pipes, e: mtl.ComputeEncoder, blocks: Ref, partial: Ref, query: Ref, out: Ref, srcs: u32) void {
    small_ops.ar(&p.small, e, blocks, partial, query, out, srcs, 1, .{ .rows = 1, .blocks_stride = BLOCKS * H, .partial_stride = H, .out_stride = H });
}
fn staticConv1(p: *const Pipes, e: mtl.ComputeEncoder, hist: Ref, stg: Ref, raw: Ref, w: Ref, out: Ref, ch: usize) void {
    runOp(p.copy_bf16, e, &.{ hist, stg }, 3 * ch);
    runOp(p.copy_bf16, e, &.{ raw, .{ .buf = stg.buf, .off = stg.off + 3 * ch * 2 } }, ch);
    glue.convStatic(&p.glue_p, e, stg, w, out, 1, @intCast(ch));
    runOp(p.copy_bf16, e, &.{ .{ .buf = stg.buf, .off = stg.off + ch * 2 }, hist }, 3 * ch);
}
fn convSlot(buf: mtl.Buffer, layer: usize, role: usize) Ref {
    const per_layer: usize = 3 * (KV_WIDTH * 2 + H * 2) * 2;
    const off: usize = switch (role) {
        0 => 0,
        1 => 3 * KV_WIDTH * 2,
        2 => 6 * KV_WIDTH * 2,
        3 => 3 * (KV_WIDTH * 2 + H) * 2,
        else => unreachable,
    };
    return .{ .buf = buf, .off = layer * per_layer + off };
}
fn tapChannelFor(layer: usize) ?usize {
    return if (layer >= 5 and (layer - 5) % 6 == 0 and (layer - 5) / 6 < 7) (layer - 5) / 6 else null;
}

/// Independent serial single-token teacher-forced prompt plus greedy decode capture from raw pack files on disk.
pub fn capture(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, prompt: []const u32, decode_steps: usize) !Capture {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const single_m = try std.fs.path.join(gpa, &.{ dir, "experts", "checkpoint.json" });
    defer gpa.free(single_m);
    const split_m = try std.fs.path.join(gpa, &.{ dir, "checkpoint.json" });
    defer gpa.free(split_m);
    const is_single = try fileExists(io, single_m);
    const cfg_p = if (is_single) try std.fs.path.join(gpa, &.{ dir, "config.json" }) else try std.fs.path.join(gpa, &.{ dir, "auxiliary", "config.json" });
    defer gpa.free(cfg_p);
    const cfg_txt = try std.Io.Dir.cwd().readFileAlloc(io, cfg_p, gpa, .limited(1 << 24));
    defer gpa.free(cfg_txt);
    var why: cfg.Why = .{};
    const parsed_cfg = try cfg.parse(gpa, cfg_txt, &why);
    const n_layers: usize = parsed_cfg.text.layers;
    const vocab: usize = parsed_cfg.text.vocab;
    const unpadded: usize = parsed_cfg.text.unpadded_vocab;
    const block_sz: usize = parsed_cfg.text.block;
    const total_steps = prompt.len + decode_steps;
    const capacity = std.mem.alignForward(usize, total_steps + 4, ap.tile_keys);

    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var pipes = try Pipes.load(gpa, device);
    defer pipes.deinit();

    var dense = RawDense{ .gpa = gpa };
    defer dense.deinit();
    if (is_single) {
        const dp = try std.fs.path.joinZ(gpa, &.{ dir, "model-dense.safetensors" });
        defer gpa.free(dp);
        try dense.loadFile(device, dp);
    } else {
        const gp = try std.fs.path.joinZ(gpa, &.{ dir, "dense", "global.safetensors" });
        defer gpa.free(gp);
        try dense.loadFile(device, gp);
        for (0..n_layers) |li| {
            var lb: [32]u8 = undefined;
            const lp = try std.fs.path.joinZ(gpa, &.{ dir, "dense", std.fmt.bufPrint(&lb, "layer-{d:0>2}.safetensors", .{li}) catch unreachable });
            defer gpa.free(lp);
            try dense.loadFile(device, lp);
        }
    }

    const man_txt = try std.Io.Dir.cwd().readFileAlloc(io, if (is_single) single_m else split_m, gpa, .limited(64 << 20));
    defer gpa.free(man_txt);
    var man_arena = std.heap.ArenaAllocator.init(gpa);
    defer man_arena.deinit();
    const man_root = (try std.json.parseFromSlice(std.json.Value, man_arena.allocator(), man_txt, .{})).value.object;
    const layers_obj = man_root.get("layers").?.object;
    const exp_root = if (is_single) try std.fs.path.join(gpa, &.{ dir, "experts" }) else try gpa.dupe(u8, dir);
    defer gpa.free(exp_root);
    var exp_bufs: std.ArrayList(mtl.Buffer) = .empty;
    defer {
        for (exp_bufs.items) |b| b.deinit();
        exp_bufs.deinit(gpa);
    }
    const lexps = try gpa.alloc(LayerExp, n_layers);
    defer gpa.free(lexps);
    var loaded_exp: usize = 0;
    defer for (lexps[0..loaded_exp]) |*le| {
        le.w13.deinit();
        le.w2.deinit();
        gpa.free(le.shards);
    };
    var shard_cache: std.StringHashMapUnmanaged(mtl.Buffer) = .empty;
    defer {
        var it = shard_cache.keyIterator();
        while (it.next()) |k| gpa.free(k.*);
        shard_cache.deinit(gpa);
    }
    for (0..n_layers) |li| {
        var kb: [16]u8 = undefined;
        const lo = layers_obj.get(std.fmt.bufPrint(&kb, "{d}", .{li}) catch unreachable).?.object;
        lexps[li] = try loadLayerExperts(gpa, device, exp_root, lo, &exp_bufs, &shard_cache);
        loaded_exp += 1;
    }

    const ring_local: usize = ap.ringFor(1024, 1);
    const kv_k = try gpa.alloc(mtl.Buffer, n_layers);
    defer gpa.free(kv_k);
    const kv_v = try gpa.alloc(mtl.Buffer, n_layers);
    defer gpa.free(kv_v);
    var kv_init: usize = 0;
    defer for (0..kv_init) |li| {
        kv_k[li].deinit();
        kv_v[li].deinit();
    };
    for (0..n_layers) |li| {
        const slots = if (li % 6 == 5) capacity else ring_local;
        kv_k[li] = try device.buffer(slots * KV_WIDTH * 2, mtl.ResourceOptions.private);
        errdefer kv_k[li].deinit();
        kv_v[li] = try device.buffer(slots * KV_WIDTH * 2, mtl.ResourceOptions.private);
        kv_init += 1;
    }
    const conv_hist = try device.buffer(n_layers * 3 * (KV_WIDTH * 2 + H * 2) * 2, mtl.ResourceOptions.shared);
    defer conv_hist.deinit();
    @memset(conv_hist.contents()[0 .. n_layers * 3 * (KV_WIDTH * 2 + H * 2) * 2], 0);
    const conv_stg = try device.buffer(4 * H * 2, mtl.ResourceOptions.private);
    defer conv_stg.deinit();

    const s_ids = try device.buffer(16, mtl.ResourceOptions.shared);
    defer s_ids.deinit();
    var bufs: std.ArrayList(mtl.Buffer) = .empty;
    defer {
        for (bufs.items) |b| b.deinit();
        bufs.deinit(gpa);
    }
    const mk = struct {
        fn f(dev: mtl.Device, list: *std.ArrayList(mtl.Buffer), alloc: std.mem.Allocator, len: usize, opts: usize) !mtl.Buffer {
            const b = try dev.buffer(len, opts);
            errdefer b.deinit();
            try list.append(alloc, b);
            return b;
        }
    }.f;
    const sh = mtl.ResourceOptions.shared;
    const pr = mtl.ResourceOptions.private;
    const s_pos = try mk(device, &bufs, gpa, 16, sh);
    const s_seg_of = try mk(device, &bufs, gpa, 16, sh);
    const s_segs = try mk(device, &bufs, gpa, @sizeOf(ap.Seg), sh);
    const s_p = try mk(device, &bufs, gpa, H * 2, pr);
    const s_partial = try mk(device, &bufs, gpa, H * 4, pr);
    const s_blocks = try mk(device, &bufs, gpa, BLOCKS * H * 2, pr);
    const s_ar = try mk(device, &bufs, gpa, 3 * H * 4, pr);
    const s_qn = try mk(device, &bufs, gpa, H * 2, pr);
    const s_kn = try mk(device, &bufs, gpa, H * 2, pr);
    const s_vn = try mk(device, &bufs, gpa, H * 2, pr);
    const s_qr = try mk(device, &bufs, gpa, Q_WIDTH * 2, pr);
    const s_kr = try mk(device, &bufs, gpa, KV_WIDTH * 2, pr);
    const s_vr = try mk(device, &bufs, gpa, KV_WIDTH * 2, pr);
    const s_kc = try mk(device, &bufs, gpa, KV_WIDTH * 2, pr);
    const s_vc = try mk(device, &bufs, gpa, KV_WIDTH * 2, pr);
    const s_rr = try mk(device, &bufs, gpa, REL_WIDTH * 2, pr);
    const s_qp = try mk(device, &bufs, gpa, Q_WIDTH * 2, pr);
    const s_rp = try mk(device, &bufs, gpa, REL_WIDTH * 2, pr);
    const s_sink = try mk(device, &bufs, gpa, Q_HEADS * 2, pr);
    const s_gate = try mk(device, &bufs, gpa, Q_WIDTH * 2, pr);
    const s_att = try mk(device, &bufs, gpa, Q_WIDTH * 2, pr);
    const s_po = try mk(device, &bufs, gpa, Q_HEADS * MAX_ATTN_STRIDE * HEAD_DIM * 4, pr);
    const s_pml = try mk(device, &bufs, gpa, Q_HEADS * MAX_ATTN_STRIDE * 8, pr);
    const s_ad = try mk(device, &bufs, gpa, H * 2, pr);
    const s_ac = try mk(device, &bufs, gpa, H * 2, pr);
    const s_mar = try mk(device, &bufs, gpa, H * 4, pr);
    const s_mn = try mk(device, &bufs, gpa, H * 2, pr);
    const s_rids = try mk(device, &bufs, gpa, TOPK * 4, pr);
    const s_rw = try mk(device, &bufs, gpa, TOPK * 4, pr);
    const s_gu_raw = try mk(device, &bufs, gpa, TOPK * 2 * INTER * 2, pr);
    const s_gu = try mk(device, &bufs, gpa, TOPK * INTER * 2, pr);
    const s_down = try mk(device, &bufs, gpa, TOPK * H * 2, pr);
    const s_exl3 = try mk(device, &bufs, gpa, try exl3_proj.scratchBytes(H, 2 * INTER, TOPK), pr);
    const s_mo = try mk(device, &bufs, gpa, H * 2, pr);
    const s_mc = try mk(device, &bufs, gpa, H * 2, pr);
    const s_taps = try mk(device, &bufs, gpa, TAPS * H * 2, sh);
    const s_logits = try mk(device, &bufs, gpa, unpadded * 4, sh);

    const out_logits = try gpa.alloc(u32, total_steps * unpadded);
    errdefer gpa.free(out_logits);
    const out_taps = try gpa.alloc(u16, total_steps * TAPS * H);
    errdefer gpa.free(out_taps);
    const out_tokens = try gpa.alloc(u32, total_steps);
    errdefer gpa.free(out_tokens);

    const w_embed = try dense.get("model.llm.embed.weight", .{}, .bf16, &.{ vocab, H });
    const w_enorm = try dense.get("model.llm.embed_norm.weight", .{}, .bf16, &.{H});
    const w_norm = try dense.get("model.llm.norm.weight", .{}, .bf16, &.{H});
    const w_unembed = try dense.get("model.llm.unembed.weight", .{}, .bf16, &.{ vocab, H });
    const w_outq = try dense.get("model.llm.attn_res.output_query", .{}, .f32, &.{H});
    const prep = ap.Prep{ .qk_clip = parsed_cfg.text.qk_clip, .rel_clip = parsed_cfg.text.rel_clip, .floor = @floatFromInt(parsed_cfg.text.log_floor), .alpha = parsed_cfg.text.log_alpha, .eps = parsed_cfg.text.eps };

    for (0..total_steps) |step| {
        const tok: u32 = if (step < prompt.len) prompt[step] else blk: {
            const prev = std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(out_logits[(step - 1) * unpadded ..][0..unpadded]));
            var best: usize = 0;
            for (prev[1..], 1..) |v, i| if (v > prev[best]) {
                best = i;
            };
            break :blk @intCast(best);
        };
        out_tokens[step] = tok;
        s_ids.slice(u32, 1)[0] = tok;
        s_pos.slice(i32, 1)[0] = @intCast(step);
        s_seg_of.slice(i32, 1)[0] = 0;
        s_segs.slice(ap.Seg, 1)[0] = .{ .row0 = 0, .n = 1, .p0 = @intCast(step), .off = 0 };
        @memset(s_taps.slice(u16, TAPS * H), 0);

        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        small_ops.embedNorm(&pipes.small, e, .{ .buf = s_ids }, w_embed, w_enorm, .{ .buf = s_p }, 1);
        for (0..n_layers) |li| {
            const global = (li % 6 == 5);
            const boundary = (li % block_sz == 0);
            const blocks_in: u32 = @intCast((li + block_sz - 1) / block_sz);
            if (tapChannelFor(li)) |ch| small_ops.captureTap(&pipes.small, e, .{ .buf = s_p }, .{ .buf = s_taps }, 1, ch);
            runOp(pipes.widen, e, &.{ .{ .buf = s_p }, .{ .buf = s_partial } }, H);
            const qn_w = if (global) try dense.get("model.llm.layers.{d}.q_input_norm.weight", .{li}, .bf16, &.{H}) else try dense.get("model.llm.layers.{d}.attn_norm.weight", .{li}, .bf16, &.{H});
            if (li == 0) {
                runOp(pipes.widen, e, &.{ .{ .buf = s_p }, .{ .buf = s_ar } }, H);
                rmsNorm1(&pipes, e, .{ .buf = s_ar }, qn_w, .{ .buf = s_qn });
            } else if (!global) {
                const aq = try dense.get("model.llm.attn_res.attn_queries.{d}", .{li - 1}, .f32, &.{H});
                ar1(&pipes, e, .{ .buf = s_blocks }, .{ .buf = s_partial }, aq, .{ .buf = s_ar }, blocks_in + 1);
                rmsNorm1(&pipes, e, .{ .buf = s_ar }, qn_w, .{ .buf = s_qn });
            } else {
                const aq = try dense.get("model.llm.attn_res.attn_queries.{d}", .{li - 1}, .f32, &.{ 3, H });
                for (0..3) |role| ar1(&pipes, e, .{ .buf = s_blocks }, .{ .buf = s_partial }, .{ .buf = aq.buf, .off = aq.off + role * H * 4 }, .{ .buf = s_ar, .off = role * H * 4 }, blocks_in + 1);
                rmsNorm1(&pipes, e, .{ .buf = s_ar, .off = 0 }, qn_w, .{ .buf = s_qn });
                rmsNorm1(&pipes, e, .{ .buf = s_ar, .off = H * 4 }, try dense.get("model.llm.layers.{d}.k_input_norm.weight", .{li}, .bf16, &.{H}), .{ .buf = s_kn });
                rmsNorm1(&pipes, e, .{ .buf = s_ar, .off = 2 * H * 4 }, try dense.get("model.llm.layers.{d}.v_input_norm.weight", .{li}, .bf16, &.{H}), .{ .buf = s_vn });
            }
            const kn_ref: Ref = if (global) .{ .buf = s_kn } else .{ .buf = s_qn };
            const vn_ref: Ref = if (global) .{ .buf = s_vn } else .{ .buf = s_qn };
            matvec1(&pipes, e, try dense.get("model.llm.layers.{d}.attn.wq_du.weight", .{li}, .bf16, &.{ Q_WIDTH, H }), .{ .buf = s_qn }, .{ .buf = s_qr }, Q_WIDTH, H);
            matvec1(&pipes, e, try dense.get("model.llm.layers.{d}.attn.wk_dv.weight", .{li}, .bf16, &.{ KV_WIDTH, H }), kn_ref, .{ .buf = s_kr }, KV_WIDTH, H);
            matvec1(&pipes, e, try dense.get("model.llm.layers.{d}.attn.wv_dv.weight", .{li}, .bf16, &.{ KV_WIDTH, H }), vn_ref, .{ .buf = s_vr }, KV_WIDTH, H);
            matvec1(&pipes, e, try dense.get("model.llm.layers.{d}.attn.wr_du.weight", .{li}, .bf16, &.{ REL_WIDTH, H }), .{ .buf = s_qn }, .{ .buf = s_rr }, REL_WIDTH, H);
            matvec1(&pipes, e, try dense.get("model.llm.layers.{d}.attn.attn_sink_proj.weight", .{li}, .bf16, &.{ Q_HEADS, H }), .{ .buf = s_qn }, .{ .buf = s_sink }, Q_HEADS, H);
            matvec1(&pipes, e, try dense.get("model.llm.layers.{d}.attn.gate_attn.weight", .{li}, .bf16, &.{ Q_WIDTH, H }), .{ .buf = s_qn }, .{ .buf = s_gate }, Q_WIDTH, H);

            for ([_]struct { role: usize, raw: Ref, out: Ref, norm: Ref, k_flag: bool }{
                .{ .role = 0, .raw = .{ .buf = s_kr }, .out = .{ .buf = s_kc }, .norm = kn_ref, .k_flag = true },
                .{ .role = 1, .raw = .{ .buf = s_vr }, .out = .{ .buf = s_vc }, .norm = vn_ref, .k_flag = false },
            }) |cv| {
                const hist = convSlot(conv_hist, li, cv.role);
                const stg = Ref{ .buf = conv_stg };
                runOp(pipes.copy_bf16, e, &.{ hist, stg }, 3 * KV_WIDTH);
                runOp(pipes.copy_bf16, e, &.{ cv.raw, .{ .buf = conv_stg, .off = 3 * KV_WIDTH * 2 } }, KV_WIDTH);
                if (!global) {
                    const cw = if (cv.k_flag) try dense.get("model.llm.layers.{d}.attn.k_sconv.weight", .{li}, .bf16, &.{ KV_WIDTH, 1, 4 }) else try dense.get("model.llm.layers.{d}.attn.v_sconv.weight", .{li}, .bf16, &.{ KV_WIDTH, 1, 4 });
                    glue.convStatic(&pipes.glue_p, e, stg, cw, cv.out, 1, KV_WIDTH);
                } else {
                    const sw = if (cv.k_flag) try dense.get("model.llm.layers.{d}.attn.k_sconv.static_w", .{li}, .bf16, &.{ 4, KV_WIDTH }) else try dense.get("model.llm.layers.{d}.attn.v_sconv.static_w", .{li}, .bf16, &.{ 4, KV_WIDTH });
                    const proj = if (cv.k_flag) try dense.get("model.llm.layers.{d}.attn.k_sconv.weight_proj", .{li}, .bf16, &.{ 32, H }) else try dense.get("model.llm.layers.{d}.attn.v_sconv.weight_proj", .{li}, .bf16, &.{ 32, H });
                    small_ops.convDynamic(&pipes.small, e, stg, cv.norm, sw, proj, cv.out, 1, KV_WIDTH);
                }
                runOp(pipes.copy_bf16, e, &.{ .{ .buf = conv_stg, .off = KV_WIDTH * 2 }, hist }, 3 * KV_WIDTH);
            }

            const l0 = if (global) ap.Layer.global(@intCast(capacity)) else ap.Layer.sliding(1024, 1);
            const a_layer = ap.Layer{ .window = l0.window, .ring = @intCast(if (global) capacity else ring_local), .chunk = l0.chunk, .tau = l0.tau };
            const rr = ap.Rows{ .rows = 1, .heads = Q_HEADS, .kv_heads = KV_HEADS, .last_pos = @intCast(step), .streams = 1 };
            const meta = attention.RowMeta{ .pos = .{ .buf = s_pos }, .seg_of = .{ .buf = s_seg_of }, .segs = .{ .buf = s_segs } };
            const kv = attention.Cache{ .k = .{ .buf = kv_k[li] }, .v = .{ .buf = kv_v[li] } };
            const qs = try dense.get("model.llm.layers.{d}.attn.q_l2_clip.scale", .{li}, .bf16, &.{ Q_HEADS, HEAD_DIM });
            const rs = try dense.get("model.llm.layers.{d}.attn.rel_l2_clip.scale", .{li}, .bf16, &.{ Q_HEADS, D_REL });
            const rp = try dense.get("model.llm.layers.{d}.attn.rel_logits_proj.proj", .{li}, .bf16, &.{ D_REL, if (global) parsed_cfg.text.rel_extent else parsed_cfg.text.window });
            attention.prep(&pipes.attn, e, rr, a_layer, prep, meta, .{ .{ .buf = s_qr }, .{ .buf = s_kc }, .{ .buf = s_vc }, .{ .buf = s_rr } }, .{ .{ .buf = qs.buf, .off = qs.off }, .{ .buf = rs.buf, .off = rs.off } }, .{ .buf = s_qp }, .{ .buf = s_rp }, kv);
            try attention.attend(&pipes.attn, e, rr, a_layer, MAX_ATTN_STRIDE, 0, meta, .{ .buf = s_qp }, kv, .{ .buf = s_rp }, .{ .buf = rp.buf, .off = rp.off }, .{ .buf = s_po }, .{ .buf = s_pml }, .{ .buf = s_sink }, .{ .buf = s_gate }, .{ .buf = s_att });

            matvec1(&pipes, e, try dense.get("model.llm.layers.{d}.attn.wo_ud.weight", .{li}, .bf16, &.{ H, Q_WIDTH }), .{ .buf = s_att }, .{ .buf = s_ad }, H, Q_WIDTH);
            staticConv1(&pipes, e, convSlot(conv_hist, li, 2), .{ .buf = conv_stg }, .{ .buf = s_ad }, try dense.get("model.llm.layers.{d}.attn_sconv.weight", .{li}, .bf16, &.{ H, 1, 4 }), .{ .buf = s_ac }, H);
            small_ops.attentionPartial(&pipes.small, e, .{ .buf = s_p }, .{ .buf = s_ac }, .{ .buf = s_blocks }, .{ .buf = s_partial }, 1, if (boundary) blocks_in + 1 else 0);
            ar1(&pipes, e, .{ .buf = s_blocks }, .{ .buf = s_partial }, try dense.get("model.llm.attn_res.mlp_queries.{d}", .{li}, .f32, &.{H}), .{ .buf = s_mar }, blocks_in + 1 + @intFromBool(boundary));
            rmsNorm1(&pipes, e, .{ .buf = s_mar }, try dense.get("model.llm.layers.{d}.mlp_norm.weight", .{li}, .bf16, &.{H}), .{ .buf = s_mn });
            small_ops.router(&pipes.small, e, .{ .buf = s_mn }, try dense.get("model.llm.layers.{d}.mlp.gate.weight", .{li}, .bf16, &.{ EXPERTS, H }), try dense.get("model.llm.layers.{d}.mlp.gate.bias", .{li}, .f32, &.{EXPERTS}), .{ .buf = s_rids }, .{ .buf = s_rw }, 1);
            if (lexps[li].shards.len > 0) e.useResources(lexps[li].shards, mtl.ResourceUsage.read);
            try exl3_proj.encodeIndexed(&pipes.exl3_p, e, .{ .descriptors = .{ .buffer = lexps[li].w13, .offset = 0 }, .count = EXPERTS, .k = H, .n = 2 * INTER }, .{ .buffer = s_mn, .offset = 0 }, .{ .buffer = s_gu_raw, .offset = 0 }, .{ .buffer = s_exl3, .offset = 0 }, .{ .count = TOPK, .x_stride = H, .y_stride = 2 * INTER, .output = .bf16, .routes = .{ .ids = .{ .buffer = s_rids, .offset = 0 }, .repeat = TOPK } });
            glue.swiglu(&pipes.glue_p, e, .{ .buf = s_gu_raw }, .{ .buf = s_gu }, TOPK, INTER);
            try exl3_proj.encodeIndexed(&pipes.exl3_p, e, .{ .descriptors = .{ .buffer = lexps[li].w2, .offset = 0 }, .count = EXPERTS, .k = INTER, .n = H }, .{ .buffer = s_gu, .offset = 0 }, .{ .buffer = s_down, .offset = 0 }, .{ .buffer = s_exl3, .offset = 0 }, .{ .count = TOPK, .x_stride = INTER, .y_stride = H, .output = .bf16, .routes = .{ .ids = .{ .buffer = s_rids, .offset = 0 }, .repeat = 1 } });
            glue.expertCombine(&pipes.glue_p, e, .{ .buf = s_down }, .{ .buf = s_rw }, .{ .buf = s_mo }, 1, H);
            staticConv1(&pipes, e, convSlot(conv_hist, li, 3), .{ .buf = conv_stg }, .{ .buf = s_mo }, try dense.get("model.llm.layers.{d}.mlp_sconv.weight", .{li}, .bf16, &.{ H, 1, 4 }), .{ .buf = s_mc }, H);
            runOp(pipes.mlp_partial, e, &.{ .{ .buf = s_partial }, .{ .buf = s_mc }, .{ .buf = s_p } }, H);
        }
        runOp(pipes.widen, e, &.{ .{ .buf = s_p }, .{ .buf = s_partial } }, H);
        const final_srcs: u32 = @intCast((n_layers + block_sz - 1) / block_sz + 1);
        ar1(&pipes, e, .{ .buf = s_blocks }, .{ .buf = s_partial }, w_outq, .{ .buf = s_mar }, final_srcs);
        rmsNorm1(&pipes, e, .{ .buf = s_mar }, w_norm, .{ .buf = s_mn });
        small_ops.captureTap(&pipes.small, e, .{ .buf = s_mn }, .{ .buf = s_taps }, 1, TAPS - 1);
        small_ops.logits(&pipes.small, e, .{ .buf = s_mn }, w_unembed, .{ .buf = s_logits }, 1, @intCast(unpadded));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailed;
        @memcpy(out_logits[step * unpadded ..][0..unpadded], s_logits.slice(u32, unpadded));
        @memcpy(out_taps[step * TAPS * H ..][0 .. TAPS * H], s_taps.slice(u16, TAPS * H));
    }
    return .{ .gpa = gpa, .steps = total_steps, .vocab = unpadded, .logits = out_logits, .taps = out_taps, .tokens = out_tokens };
}
