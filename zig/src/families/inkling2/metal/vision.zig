//! Inkling 2 Mini's image tower on Metal: BF16 activations and weights, FP32 accumulators, BF16 text-width output.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("inkling2_metal");
const frags = @import("../../../core/frags.zig");
const ck = @import("../../../core/checkpoint_metal.zig");
const config = @import("../config.zig");
const gm = @import("vision_gemm.zig");
const tables = @import("vision_tables.zig");
const image = @import("vision_image.zig");

pub const Prepared = image.Prepared;
pub const prepare = image.prepare;
pub const problem = image.problem;
/// The tensor-unit GEMMs, for the kernel contract's probes (tests/contract/metal_inkling2_vision_*.zig).
pub const gemm = gm;

const Ref = gm.Ref;
const chunk_rows: u32 = 1024; // sequence rows a projection pass multiplies at once (the fp32 sums' scratch)
const score_bytes: usize = 256 << 20; // the attention scores' fp32 scratch

pub const Options = struct {
    /// The most image tokens one encode takes (--vision's image budget): the scratch is sized for it.
    max_tokens: u32 = 4096,
};

const Layer = struct {
    ln1_w: Ref,
    ln1_b: Ref,
    ln2_w: Ref,
    ln2_b: Ref,
    ls1: Ref,
    ls2: Ref,
    w: [4]Ref, // q, k, v, out
    b: [4]Ref,
    fc1_w: Ref,
    fc1_b: Ref,
    fc2_w: Ref,
    fc2_b: Ref,
};

const glue_names = [_][]const u8{ "pack", "embed", "layer_norm", "qkv", "softmax", "heads", "bias_gelu", "gelu_rows", "residual" };
const Glue = enum { pack, embed, layer_norm, qkv, softmax, heads, bias_gelu, gelu_rows, residual };

/// Kernel constants (the structs of kernels/metal/inkling2/vision.metal).
const Rows = extern struct { rows: u32, width: u32, eps: f32 };
const Qkv = extern struct { rows: u32, heads: u32, dim: u32, r0: u32 };
const Heads = extern struct { rows: u32, r0: u32, heads: u32, dim: u32 };
const Embed = extern struct { patches: u32, width: u32 };

const scratch_names = [_][]const u8{ "pix", "pixb", "pe", "taps", "weights", "cs", "sn", "x", "xn", "att", "acc", "q", "k", "v", "s", "o", "mid", "p1" };

pub const Tower = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    cfg: config.Vision,
    max_tokens: u32,
    owned: ?ck.Checkpoint = null,
    lib: mtl.Library,
    gemm: gm.Gemm,
    glue: [glue_names.len]mtl.Pipeline,
    layers: []Layer,
    patch_w: Ref,
    pos_table: Ref,
    cls: Ref,
    cls_pos: Ref,
    pre_w: Ref,
    pre_b: Ref,
    proj_in: Ref,
    proj_out: Ref,
    scratch: [scratch_names.len]mtl.Buffer,
    qchunk: u32,
    follow: ?mtl.ResidencySet.Follow = null,

    /// The tower from `dir`'s checkpoint (the shards holding model.visual.*), sized for `opts.max_tokens` image tokens.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, device: mtl.Device, queue: mtl.Queue, cfg: *const config.Config, opts: Options) !*Tower {
        var checkpoint = ck.Checkpoint.init(gpa);
        errdefer checkpoint.deinit();
        try geometry(cfg.vision, opts);
        try readShards(&checkpoint, io, dir, device);
        const t = try loadFromCheckpoint(gpa, device, queue, cfg, opts, &checkpoint);
        t.owned = checkpoint;
        return t;
    }

    /// The caller keeps the checkpoint and its residency alive through tower deinit.
    pub fn loadFromCheckpoint(gpa: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, cfg: *const config.Config, opts: Options, checkpoint: *const ck.Checkpoint) !*Tower {
        const v = cfg.vision;
        try geometry(v, opts);
        const t = try gpa.create(Tower);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .device = device, .queue = queue, .cfg = v, .max_tokens = opts.max_tokens, .lib = undefined, .gemm = .{}, .glue = undefined, .layers = &.{}, .patch_w = undefined, .pos_table = undefined, .cls = undefined, .cls_pos = undefined, .pre_w = undefined, .pre_b = undefined, .proj_in = undefined, .proj_out = undefined, .scratch = undefined, .qchunk = 0 };
        t.layers = try gpa.alloc(Layer, v.layers);
        errdefer gpa.free(t.layers);
        try t.bind(checkpoint);
        const src = try frags.source(device, gpa, sources.vision);
        defer gpa.free(src);
        t.lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        errdefer t.lib.deinit();
        t.gemm = try gm.Gemm.init(device, t.lib, &.{ .bf16_nt, .bf16f_nt, .f32_nt, .f32_nn });
        errdefer t.gemm.deinit();
        var made: usize = 0;
        errdefer for (t.glue[0..made]) |p| p.deinit();
        inline for (glue_names, 0..) |name, i| {
            t.glue[i] = try mtl.Pipeline.init(device, t.lib, "ik2v_" ++ name, false);
            made += 1;
        }
        try t.allocate();
        return t;
    }

    fn geometry(v: config.Vision, opts: Options) !void {
        if (v.heads == 0 or v.hidden % v.heads != 0 or (v.hidden / v.heads) % 4 != 0 or v.hidden / v.heads > 128)
            return error.UnsupportedVisionGeometry;
        if (v.patch != image.patch or v.merge != image.merge or v.grid != image.grid or opts.max_tokens == 0) return error.UnsupportedVisionGeometry;
    }

    pub fn deinit(t: *Tower) void {
        t.setResident(null);
        for (t.scratch) |b| b.deinit();
        for (t.glue) |p| p.deinit();
        t.gemm.deinit();
        t.lib.deinit();
        t.gpa.free(t.layers);
        if (t.owned) |*weights| weights.deinit();
        t.gpa.destroy(t);
    }

    /// Reads the shards that hold the tower's tensors into the tower's own checkpoint.
    fn readShards(checkpoint: *ck.Checkpoint, io: std.Io, dir: []const u8, device: mtl.Device) !void {
        const gpa = checkpoint.allocator;
        const index_path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
        defer gpa.free(index_path);
        const index = std.Io.Dir.cwd().readFileAlloc(io, index_path, gpa, .limited(256 << 20)) catch |err| switch (err) {
            error.FileNotFound => null,
            else => return err,
        };
        var files: std.ArrayList([]const u8) = .empty;
        defer files.deinit(gpa);
        var doc: ?std.json.Parsed(std.json.Value) = null;
        defer if (doc) |d| d.deinit();
        defer if (index) |b| gpa.free(b);
        if (index) |json| {
            doc = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
            const map = doc.?.value.object.get("weight_map") orelse return error.BadCheckpointIndex;
            if (map != .object) return error.BadCheckpointIndex;
            var it = map.object.iterator();
            while (it.next()) |e| {
                if (!std.mem.startsWith(u8, e.key_ptr.*, "model.visual.") or e.value_ptr.* != .string) continue;
                const name = e.value_ptr.string;
                const seen = for (files.items) |f| {
                    if (std.mem.eql(u8, f, name)) break true;
                } else false;
                if (!seen) try files.append(gpa, name);
            }
        } else try files.append(gpa, "model.safetensors");
        if (files.items.len == 0) return error.NoVisionTensors;
        for (files.items) |name| {
            const joined = try std.fs.path.join(gpa, &.{ dir, name });
            defer gpa.free(joined);
            const path = try gpa.dupeSentinel(u8, joined, 0);
            defer gpa.free(path);
            try checkpoint.addFileSelected(device, path, "", "model.visual.");
        }
    }

    fn tensor(checkpoint: *const ck.Checkpoint, name: []const u8, dims: []const usize) !Ref {
        const x = try checkpoint.get(name);
        if (x.dtype != .bf16 or x.rank != dims.len or !std.mem.eql(usize, x.shape[0..x.rank], dims)) {
            std.log.err("inkling2 vision: {s} is not bf16 {any}", .{ name, dims });
            return error.UnexpectedVisionTensor;
        }
        return .{ .buf = x.buffer, .off = x.offset };
    }

    fn layerTensor(checkpoint: *const ck.Checkpoint, i: usize, comptime tail: []const u8, dims: []const usize) !Ref {
        var name_buf: [128]u8 = undefined;
        return tensor(checkpoint, try std.fmt.bufPrint(&name_buf, "model.visual.backbone.layers.{d}." ++ tail, .{i}), dims);
    }

    fn bind(t: *Tower, checkpoint: *const ck.Checkpoint) !void {
        const v = t.cfg;
        const h: usize = v.hidden;
        const ff: usize = h * v.mlp_ratio;
        const pre = "model.visual.backbone.";
        t.patch_w = try tensor(checkpoint, pre ++ "embeddings.patch_embedding.weight", &.{ h, image.patch_width });
        t.pos_table = try tensor(checkpoint, pre ++ "embeddings.position_embedding", &.{ @as(usize, v.grid) * v.grid, h });
        t.cls = try tensor(checkpoint, pre ++ "embeddings.class_embedding", &.{h});
        t.cls_pos = try tensor(checkpoint, pre ++ "embeddings.cls_position_embedding", &.{h});
        t.pre_w = try tensor(checkpoint, pre ++ "ln_pre.weight", &.{h});
        t.pre_b = try tensor(checkpoint, pre ++ "ln_pre.bias", &.{h});
        t.proj_in = try tensor(checkpoint, "model.visual.proj_in.weight", &.{ v.proj_hidden, h * v.merge * v.merge });
        t.proj_out = try tensor(checkpoint, "model.visual.proj_out.weight", &.{ v.text_hidden, v.proj_hidden });
        for (t.layers, 0..) |*l, i| {
            l.ln1_w = try layerTensor(checkpoint, i, "ln_1.weight", &.{h});
            l.ln1_b = try layerTensor(checkpoint, i, "ln_1.bias", &.{h});
            l.ln2_w = try layerTensor(checkpoint, i, "ln_2.weight", &.{h});
            l.ln2_b = try layerTensor(checkpoint, i, "ln_2.bias", &.{h});
            l.ls1 = try layerTensor(checkpoint, i, "ls_1.gamma", &.{h});
            l.ls2 = try layerTensor(checkpoint, i, "ls_2.gamma", &.{h});
            l.w[0] = try layerTensor(checkpoint, i, "self_attn.q_proj.weight", &.{ h, h });
            l.w[1] = try layerTensor(checkpoint, i, "self_attn.k_proj.weight", &.{ h, h });
            l.w[2] = try layerTensor(checkpoint, i, "self_attn.v_proj.weight", &.{ h, h });
            l.w[3] = try layerTensor(checkpoint, i, "self_attn.out_proj.weight", &.{ h, h });
            l.b[0] = try layerTensor(checkpoint, i, "self_attn.q_proj.bias", &.{h});
            l.b[1] = try layerTensor(checkpoint, i, "self_attn.k_proj.bias", &.{h});
            l.b[2] = try layerTensor(checkpoint, i, "self_attn.v_proj.bias", &.{h});
            l.b[3] = try layerTensor(checkpoint, i, "self_attn.out_proj.bias", &.{h});
            l.fc1_w = try layerTensor(checkpoint, i, "mlp.c_fc.weight", &.{ ff, h });
            l.fc1_b = try layerTensor(checkpoint, i, "mlp.c_fc.bias", &.{ff});
            l.fc2_w = try layerTensor(checkpoint, i, "mlp.c_proj.weight", &.{ h, ff });
            l.fc2_b = try layerTensor(checkpoint, i, "mlp.c_proj.bias", &.{h});
        }
    }

    fn sizes(t: *const Tower) [scratch_names.len]usize {
        const v = t.cfg;
        const tokens: usize = t.max_tokens;
        const P = tokens * v.merge * v.merge;
        const S = P + 1;
        const h: usize = v.hidden;
        const hd = h / v.heads;
        const ff = h * v.mlp_ratio;
        const q: usize = t.qchunk;
        const rows = @min(S, chunk_rows);
        return .{ P * image.patch_width * 4, P * image.patch_width * 2, P * h * 2, P * 4 * 4, P * 4 * 4, S * hd * 4, S * hd * 4, S * h * 2, S * h * 2, S * h * 2, rows * @max(3 * h, ff) * 4, @as(usize, v.heads) * S * hd * 4, @as(usize, v.heads) * S * hd * 4, @as(usize, v.heads) * S * hd * 4, @as(usize, v.heads) * q * S * 4, @as(usize, v.heads) * q * hd * 4, rows * ff * 2, tokens * v.proj_hidden * 2 };
    }

    fn allocate(t: *Tower) !void {
        const v = t.cfg;
        const S: usize = @as(usize, t.max_tokens) * v.merge * v.merge + 1;
        const per_query = @as(usize, v.heads) * S * 4;
        t.qchunk = @intCast(@min(S, @max(64, score_bytes / per_query / 64 * 64)));
        var made: usize = 0;
        errdefer for (t.scratch[0..made]) |b| b.deinit();
        for (t.sizes(), 0..) |n, i| {
            t.scratch[i] = try t.device.buffer(@max(n, 16), mtl.ResourceOptions.shared);
            made += 1;
        }
    }

    fn buf(t: *const Tower, comptime name: []const u8) mtl.Buffer {
        inline for (scratch_names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return t.scratch[i];
        @compileError("no tower buffer " ++ name);
    }

    fn pipe(t: *const Tower, g: Glue) mtl.Pipeline {
        return t.glue[@backingInt(g)];
    }

    /// Device bytes the tower owns: scratch and, only for standalone loads, its weight buffers.
    pub fn bytes(t: *const Tower) u64 {
        var total: u64 = 0;
        for (t.scratch) |b| total += b.length();
        if (t.owned) |weights| for (weights.shards.items) |s| {
            total += s.buffer.length();
        };
        return total;
    }

    /// Makes the residency set follow the tower's buffers (null: stop following, leaving them).
    pub fn setResident(t: *Tower, hook: ?mtl.ResidencySet.Follow) void {
        if (t.follow) |f| {
            for (t.scratch) |b| f.leave(f.ctx, b);
            if (t.owned) |weights| for (weights.shards.items) |s| f.leave(f.ctx, s.buffer);
        }
        t.follow = hook;
        if (hook) |f| {
            for (t.scratch) |b| f.join(f.ctx, b);
            if (t.owned) |weights| for (weights.shards.items) |s| f.join(f.ctx, s.buffer);
        }
    }

    /// A new BF16 buffer [gh * gw, text_hidden] of the projector's raw output (caller releases); one encode at a time.
    pub fn encodeInput(t: *Tower, pixels: []const f32, gh: u32, gw: u32) !mtl.Buffer {
        return t.run(pixels, gh, gw, null, false);
    }

    /// Copy every BF16 residual into `trace` (pre-norm first), for the matched-input FP64 test.
    pub fn traceInput(t: *Tower, pixels: []const f32, gh: u32, gw: u32, trace: mtl.Buffer) !mtl.Buffer {
        if (comptime !@import("builtin").is_test) @compileError("vision trace is only for tests");
        return t.run(pixels, gh, gw, trace, true);
    }

    fn run(t: *Tower, pixels: []const f32, gh: u32, gw: u32, trace: ?mtl.Buffer, comptime capture: bool) !mtl.Buffer {
        const v = t.cfg;
        const tokens: usize = @as(usize, gh) * gw;
        if (tokens == 0 or tokens > t.max_tokens) return error.ImageTooLarge;
        if (pixels.len != tokens * image.token_width) return error.ImageGrid;
        const P: u32 = @intCast(tokens * v.merge * v.merge);
        const hd: u32 = v.hidden / v.heads;
        @memcpy(t.buf("pix").slice(f32, pixels.len), pixels);
        const S: usize = @as(usize, P) + 1;
        tables.fill(gh, gw, hd, t.buf("taps").slice(u32, P * 4), t.buf("weights").slice(f32, P * 4), t.buf("cs").slice(f32, S * hd), t.buf("sn").slice(f32, S * hd));
        const out = try t.device.buffer(tokens * v.text_hidden * 2, mtl.ResourceOptions.shared);
        errdefer out.deinit();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = t.queue.commandBuffer();
        t.encode(cb, P, out, capture, trace);
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("inkling2 vision: command buffer failed: {s}", .{msg});
            return error.GpuFailed;
        }
        return out;
    }

    fn ref(t: *const Tower, comptime name: []const u8) Ref {
        return .{ .buf = t.buf(name) };
    }

    fn encode(t: *Tower, cb: mtl.CommandBuffer, P: u32, out: mtl.Buffer, comptime capture: bool, trace: ?mtl.Buffer) void {
        var e = cb.compute(.serial);
        const v = t.cfg;
        const h: u32 = v.hidden;
        const S: u32 = P + 1;
        const hd: u32 = h / v.heads;
        const ff: u32 = h * v.mlp_ratio;
        const tokens: u32 = P / (v.merge * v.merge);
        // pixels as bf16, the patch projection, then class row + patches with their position embedding
        const n: u32 = P * image.patch_width;
        e.setPipeline(t.pipe(.pack));
        e.setBuffer(t.buf("pix"), 0, 0);
        e.setBuffer(t.buf("pixb"), 0, 1);
        e.setValue(n, 2);
        e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
        t.gemm.run(e, .bf16_nt, t.ref("pixb"), t.patch_w, t.ref("pe"), .{ .m = P, .n = h, .k = image.patch_width, .lda = image.patch_width, .ldb = image.patch_width, .ldd = h });
        e.setPipeline(t.pipe(.embed));
        for ([_]Ref{ t.ref("pe"), t.pos_table, t.ref("taps"), t.ref("weights"), t.cls, t.cls_pos, t.ref("x") }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(Embed{ .patches = P, .width = h }, 7);
        e.dispatchThreads(mtl.Size.of(h, S, 1), mtl.Size.of(256, 1, 1));
        // the pre-LayerNorm's output is the residual stream: x and xn trade places
        var res = t.ref("x");
        var tmp = t.ref("xn");
        t.layerNorm(e, res, t.pre_w, t.pre_b, tmp, S);
        std.mem.swap(Ref, &res, &tmp);
        const trace_bytes: usize = @as(usize, S) * h * 2;
        if (capture) e = copyTrace(cb, e, res, trace.?, 0, trace_bytes);
        const att = t.ref("att");
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        for (t.layers, 1..) |l, stage| {
            t.layerNorm(e, res, l.ln1_w, l.ln1_b, tmp, S);
            var r0: u32 = 0;
            while (r0 < S) : (r0 += chunk_rows) {
                const m = @min(chunk_rows, S - r0);
                for (0..3) |which| t.gemm.run(e, .bf16f_nt, tmp.at(@as(usize, r0) * h * 2), l.w[which], t.ref("acc").at(which * h * 4), .{ .m = m, .n = h, .k = h, .lda = h, .ldb = h, .ldd = 3 * h });
                e.setPipeline(t.pipe(.qkv));
                for ([_]Ref{ t.ref("acc"), l.b[0], l.b[1], l.b[2], t.ref("cs"), t.ref("sn"), t.ref("q"), t.ref("k"), t.ref("v") }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
                e.setValue(Qkv{ .rows = S, .heads = v.heads, .dim = hd, .r0 = r0 }, 9);
                e.dispatchThreads(mtl.Size.of(hd, 3 * v.heads, m), mtl.Size.of(hd, @max(1, 256 / hd), 1));
            }
            var q0: u32 = 0;
            while (q0 < S) : (q0 += t.qchunk) {
                const lc = @min(t.qchunk, S - q0);
                const stride: u32 = S * hd;
                t.gemm.run(e, .f32_nt, t.ref("q").at(@as(usize, q0) * hd * 4), t.ref("k"), t.ref("s"), .{ .m = lc, .n = S, .k = hd, .lda = hd, .ldb = hd, .ldd = S, .batch = v.heads, .sa = stride, .sb = stride, .sd = lc * S });
                e.setPipeline(t.pipe(.softmax));
                e.setBuffer(t.buf("s"), 0, 0);
                e.setValue(S, 1);
                e.setValue(scale, 2);
                e.dispatchGroups(mtl.Size.of(@as(usize, v.heads) * lc, 1, 1), mtl.Size.of(256, 1, 1));
                t.gemm.run(e, .f32_nn, t.ref("s"), t.ref("v"), t.ref("o"), .{ .m = lc, .n = hd, .k = S, .lda = S, .ldb = hd, .ldd = hd, .batch = v.heads, .sa = lc * S, .sb = stride, .sd = lc * hd });
                e.setPipeline(t.pipe(.heads));
                e.setBuffer(t.buf("o"), 0, 0);
                e.setBuffer(att.buf, att.off, 1);
                e.setValue(Heads{ .rows = lc, .r0 = q0, .heads = v.heads, .dim = hd }, 2);
                e.dispatchThreads(mtl.Size.of(h, lc, 1), mtl.Size.of(256, 1, 1));
            }
            r0 = 0;
            while (r0 < S) : (r0 += chunk_rows) {
                const m = @min(chunk_rows, S - r0);
                t.gemm.run(e, .bf16f_nt, att.at(@as(usize, r0) * h * 2), l.w[3], t.ref("acc"), .{ .m = m, .n = h, .k = h, .lda = h, .ldb = h, .ldd = h });
                t.residual(e, l.b[3], l.ls1, res.at(@as(usize, r0) * h * 2), m, h);
            }
            t.layerNorm(e, res, l.ln2_w, l.ln2_b, tmp, S);
            r0 = 0;
            while (r0 < S) : (r0 += chunk_rows) {
                const m = @min(chunk_rows, S - r0);
                t.gemm.run(e, .bf16f_nt, tmp.at(@as(usize, r0) * h * 2), l.fc1_w, t.ref("acc"), .{ .m = m, .n = ff, .k = h, .lda = h, .ldb = h, .ldd = ff });
                e.setPipeline(t.pipe(.bias_gelu));
                e.setBuffer(t.buf("acc"), 0, 0);
                e.setBuffer(l.fc1_b.buf, l.fc1_b.off, 1);
                e.setBuffer(t.buf("mid"), 0, 2);
                e.setValue(ff, 3);
                e.dispatchThreads(mtl.Size.of(ff, m, 1), mtl.Size.of(256, 1, 1));
                t.gemm.run(e, .bf16f_nt, t.ref("mid"), l.fc2_w, t.ref("acc"), .{ .m = m, .n = h, .k = ff, .lda = ff, .ldb = ff, .ldd = h });
                t.residual(e, l.fc2_b, l.ls2, res.at(@as(usize, r0) * h * 2), m, h);
            }
            if (capture) e = copyTrace(cb, e, res, trace.?, stage * trace_bytes, trace_bytes);
        }
        // the merger over the patch rows (rows 1 .. of one image): 4 patches a token
        const merged: u32 = v.merge * v.merge * h;
        t.gemm.run(e, .bf16_nt, res.at(@as(usize, h) * 2), t.proj_in, t.ref("p1"), .{ .m = tokens, .n = v.proj_hidden, .k = merged, .lda = merged, .ldb = merged, .ldd = v.proj_hidden });
        e.setPipeline(t.pipe(.gelu_rows));
        e.setBuffer(t.buf("p1"), 0, 0);
        e.setValue(v.proj_hidden, 1);
        e.dispatchThreads(mtl.Size.of(v.proj_hidden, tokens, 1), mtl.Size.of(256, 1, 1));
        t.gemm.run(e, .bf16_nt, t.ref("p1"), t.proj_out, .{ .buf = out }, .{ .m = tokens, .n = v.text_hidden, .k = v.proj_hidden, .lda = v.proj_hidden, .ldb = v.proj_hidden, .ldd = v.text_hidden });
        e.end();
    }

    fn copyTrace(cb: mtl.CommandBuffer, encoder: mtl.ComputeEncoder, src: Ref, dst: mtl.Buffer, off: usize, n: usize) mtl.ComputeEncoder {
        encoder.end();
        const blit = cb.blit();
        blit.copy(src.buf, src.off, dst, off, n);
        blit.end();
        return cb.compute(.serial);
    }

    fn layerNorm(t: *const Tower, e: mtl.ComputeEncoder, x: Ref, w: Ref, b: Ref, y: Ref, rows: u32) void {
        e.setPipeline(t.pipe(.layer_norm));
        for ([_]Ref{ x, w, b, y }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(Rows{ .rows = rows, .width = t.cfg.hidden, .eps = t.cfg.eps }, 4);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// x rows += lambda * (acc + bias) over the first `rows` rows of the fp32 sums.
    fn residual(t: *const Tower, e: mtl.ComputeEncoder, bias: Ref, lambda: Ref, x: Ref, rows: u32, width: u32) void {
        e.setPipeline(t.pipe(.residual));
        e.setBuffer(t.buf("acc"), 0, 0);
        e.setBuffer(bias.buf, bias.off, 1);
        e.setBuffer(lambda.buf, lambda.off, 2);
        e.setBuffer(x.buf, x.off, 3);
        e.setValue(width, 4);
        e.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(256, 1, 1));
    }
};
