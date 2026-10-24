//! MiniMax-M3's image tower on Metal: patch embedding, 32 CLIP layers with axial RoPE, the merging projector.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const ck = @import("../../core/checkpoint_metal.zig");
const vl = @import("../../core/qwen_vl/qwen_vl.zig");
const picture = @import("../../core/picture/picture.zig");
const Glue = @import("model.zig").Glue;

pub const width = 1280;
pub const heads = 16;
pub const head_dim = 80;
pub const depth = 32;
pub const mlp_inter = 5120;
pub const hidden = 6144; // the projector's width and the language model's
pub const merge = 2;
pub const patch_width = 3 * 2 * 14 * 14; // channel by frame by row by column, the conv's own order
const freqs = 13; // each axis's: 2 * ((head_dim / 3) / 2) / 2
const rotated = 6 * freqs; // 78 of the 80 head dims turn; the last two pass
const theta: f64 = 10000;
const eps: f32 = 1e-5;
pub const max_tokens: u32 = 576; // the processor's max_pixels, 672 x 672: 48 x 48 patches
const max_patches: u32 = max_tokens * merge * merge;
pub const shard = "model-vision.safetensors";
/// The tokenizer's image markers: the processor writes start, the image's tokens, end (the template the token alone).
pub const image_start: u32 = 200029;
pub const image_end: u32 = 200030;

/// The processor's settings as the shared Qwen-VL geometry reads them (smart_resize by 28, CLIP's mean and std).
pub const image_config: vl.Config = .{ .depth = depth, .hidden = width, .heads = heads, .intermediate = mlp_inter, .out_hidden = hidden, .patch = 14, .temporal = 2, .merge = merge, .positions = 0, .image_token = 0, .vision_start = 0, .vision_end = 0, .min_pixels = 4 * 28 * 28, .max_pixels = 672 * 672, .rescale = 1.0 / 255.0, .mean = .{ 0.48145466, 0.4578275, 0.40821073 }, .std = .{ 0.26862954, 0.26130258, 0.27577711 } };

pub const Ref = struct {
    buf: mtl.Buffer,
    off: usize = 0,

    pub fn at(r: Ref, n: usize) Ref {
        return .{ .buf = r.buf, .off = r.off + n };
    }
};

fn ref(t: ck.Tensor) Ref {
    return .{ .buf = t.buffer, .off = t.offset };
}

/// An image's bytes to patches in merge order: torchvision's antialiased bicubic (the fast processor's), then CLIP's.
pub fn prepare(io: std.Io, gpa: std.mem.Allocator, bytes: []const u8, budget: u32) !vl.geometry.Prepared {
    var img = try picture.decode(io, gpa, bytes);
    defer img.deinit(gpa);
    const c = image_config;
    if (budget == 0) return error.ImageTooSmall;
    const b = vl.geometry.bounds(c, @min(budget, max_tokens));
    const size = try vl.geometry.smartResize(@intCast(img.height), @intCast(img.width), c.factor(), b[0], b[1]);
    const resized = try picture.resample.resize(gpa, .torch, .of(img.pixels, img.width, img.height), size.w, size.h);
    defer gpa.free(resized);
    const g = [2]u32{ size.h / c.patch, size.w / c.patch };
    const pixels = try vl.geometry.patchify(gpa, .{ .width = size.w, .height = size.h, .pixels = resized }, c);
    return .{ .pixels = pixels, .gh = g[0], .gw = g[1], .tokens = g[0] * g[1] / (merge * merge), .hash = picture.identity.withGrid(picture.identity.ofBytes(0x3a3a, bytes), g) };
}

/// Head dims moved so the core qkv kernel's pairs (d, d + 40) are MiniMax's (i, i + 39); q . k is unchanged.
fn moved(j: usize) usize {
    return if (j < 39) j else if (j == 39) 78 else if (j < 79) j - 1 else 79;
}

const Layer = struct { ln1_w: Ref, ln1_b: Ref, qkv_w: Ref, qkv_b: Ref, out_w: Ref, out_b: Ref, ln2_w: Ref, ln2_b: Ref, fc1_w: Ref, fc1_b: Ref, fc2_w: Ref, fc2_b: Ref };

pub const Vision = struct {
    gpa: std.mem.Allocator,
    lib: mtl.Library,
    gemm: [3][2][2]mtl.Pipeline, // [bf16 nt, f32 nt, f32 nn][M % 64 == 0][N % 128 == 0]
    norm: mtl.Pipeline,
    qkv: mtl.Pipeline,
    softmax: mtl.Pipeline,
    heads_: mtl.Pipeline,
    gelu: mtl.Pipeline,
    residual: mtl.Pipeline,
    bias: mtl.Pipeline,
    widen: mtl.Pipeline, // the model's glue
    narrow: mtl.Pipeline,
    stacked: mtl.Buffer, // every layer's q | k | v weights (q, k rows moved) then their biases
    scratch: mtl.Buffer,
    chunk: u32, // query rows an attention pass scores
    patch_w: Ref, // [width, patch_width] fp32, as stored
    pre_w: Ref,
    pre_b: Ref,
    proj: [4][2]Ref, // the projector's linear_1, linear_2, then patch_merge_mlp's: weight, bias
    layers: [depth]Layer,
    patches: Ref, // [P, patch_width] fp32
    cos: Ref, // [P, head_dim] fp32
    sin: Ref,
    x: Ref, // [P, width] fp32: the residual stream
    xn: Ref, // [P, width] bf16
    att: Ref,
    qkv_rows: Ref, // [P, 3 width] bf16
    q: Ref, // [heads, P, head_dim] fp32
    k: Ref,
    v: Ref,
    s: Ref, // [heads, chunk, P] fp32
    o: Ref, // [heads, chunk, head_dim] fp32
    mid: Ref, // [P, mlp_inter] bf16
    p1: Ref, // [P, hidden] bf16: the projector's two rows a patch, then [P / 4, hidden] merged
    p2: Ref,

    const glue = .{ .{ "norm", "qwv_layer_norm" }, .{ "qkv", "qwv_qkv" }, .{ "softmax", "qwv_softmax" }, .{ "heads_", "qwv_heads" }, .{ "gelu", "qwv_bias_gelu" }, .{ "residual", "qwv_residual" }, .{ "bias", "qwv_bias" } };

    /// The tower from `c`'s vision shard (already added): the shared Qwen-VL kernels, the q | k | v weights stacked.
    pub fn load(gpa: std.mem.Allocator, device: mtl.Device, c: *const ck.Checkpoint, widen: mtl.Pipeline, narrow: mtl.Pipeline) !*Vision {
        const v = try gpa.create(Vision);
        errdefer gpa.destroy(v);
        v.gpa = gpa;
        v.widen = widen;
        v.narrow = narrow;
        const src = try frags.source(device, gpa, ks.core_qwen_vision);
        defer gpa.free(src);
        v.lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        errdefer v.lib.deinit();
        var made: usize = 0;
        errdefer v.release(made);
        for ([_][]const u8{ "bf16_nt", "f32_nt", "f32_nn" }, 0..) |kind, ki| for (0..2) |m| for (0..2) |n| {
            var name: [32]u8 = undefined;
            v.gemm[ki][m][n] = try mtl.Pipeline.init(device, v.lib, try std.fmt.bufPrint(&name, "qwv_gemm_{s}_{d}{d}", .{ kind, m, n }), false);
            made += 1;
        };
        inline for (glue) |kv| {
            @field(v, kv[0]) = try mtl.Pipeline.init(device, v.lib, kv[1], false);
            made += 1;
        }
        const pre = "vision_tower.vision_model.";
        v.patch_w = ref(try get(c, pre ++ "embeddings.patch_embedding.weight", .f32, width * patch_width));
        v.pre_w = ref(try get(c, pre ++ "pre_layrnorm.weight", .bf16, width));
        v.pre_b = ref(try get(c, pre ++ "pre_layrnorm.bias", .bf16, width));
        const projs = [4]struct { []const u8, usize }{ .{ "multi_modal_projector.linear_1", width }, .{ "multi_modal_projector.linear_2", hidden }, .{ "patch_merge_mlp.linear_1", hidden * merge * merge }, .{ "patch_merge_mlp.linear_2", hidden } };
        for (projs, &v.proj) |p, *dst| {
            var name: [96]u8 = undefined;
            dst[0] = ref(try get(c, try std.fmt.bufPrint(&name, "{s}.weight", .{p[0]}), .bf16, hidden * p[1]));
            dst[1] = ref(try get(c, try std.fmt.bufPrint(&name, "{s}.bias", .{p[0]}), .bf16, hidden));
        }
        const per = 3 * width * width * 2;
        v.stacked = try device.buffer(depth * (per + 3 * width * 2), mtl.ResourceOptions.shared);
        errdefer v.stacked.deinit();
        const dst: [*]u16 = @ptrCast(@alignCast(v.stacked.contents()));
        for (&v.layers, 0..) |*l, i| {
            const at = struct {
                fn t(cp: *const ck.Checkpoint, at_layer: usize, name: []const u8, dtype: ck.DType, count: usize) !ck.Tensor {
                    var b: [128]u8 = undefined;
                    return get(cp, try std.fmt.bufPrint(&b, pre ++ "encoder.layers.{d}.{s}", .{ at_layer, name }), dtype, count);
                }
            }.t;
            const w_at = i * 3 * width * width;
            const b_at = depth * 3 * width * width + i * 3 * width;
            for ([_][]const u8{ "q", "k", "v" }, 0..) |which, part| {
                var nb: [64]u8 = undefined;
                const wt = try at(c, i, try std.fmt.bufPrint(&nb, "self_attn.{s}_proj.weight", .{which}), .bf16, width * width);
                const bt = try at(c, i, try std.fmt.bufPrint(&nb, "self_attn.{s}_proj.bias", .{which}), .bf16, width);
                const ws: [*]const u16 = @ptrCast(@alignCast(wt.buffer.contents() + wt.offset));
                const bs: [*]const u16 = @ptrCast(@alignCast(bt.buffer.contents() + bt.offset));
                for (0..width) |row| {
                    const from = if (part == 2) row else row / head_dim * head_dim + moved(row % head_dim);
                    @memcpy(dst[w_at + (part * width + row) * width ..][0..width], ws[from * width ..][0..width]);
                    dst[b_at + part * width + row] = bs[from];
                }
            }
            l.qkv_w = .{ .buf = v.stacked, .off = w_at * 2 };
            l.qkv_b = .{ .buf = v.stacked, .off = b_at * 2 };
            l.ln1_w = ref(try at(c, i, "layer_norm1.weight", .bf16, width));
            l.ln1_b = ref(try at(c, i, "layer_norm1.bias", .bf16, width));
            l.ln2_w = ref(try at(c, i, "layer_norm2.weight", .bf16, width));
            l.ln2_b = ref(try at(c, i, "layer_norm2.bias", .bf16, width));
            l.out_w = ref(try at(c, i, "self_attn.out_proj.weight", .bf16, width * width));
            l.out_b = ref(try at(c, i, "self_attn.out_proj.bias", .bf16, width));
            l.fc1_w = ref(try at(c, i, "mlp.fc1.weight", .bf16, mlp_inter * width));
            l.fc1_b = ref(try at(c, i, "mlp.fc1.bias", .bf16, mlp_inter));
            l.fc2_w = ref(try at(c, i, "mlp.fc2.weight", .bf16, width * mlp_inter));
            l.fc2_b = ref(try at(c, i, "mlp.fc2.bias", .bf16, width));
        }
        // the scores take at most 256 MiB: every head's `chunk` query rows by P keys in fp32
        const P: usize = max_patches;
        v.chunk = @intCast(std.math.clamp((256 << 20) / (heads * P * 4) / 64 * 64, 64, P));
        const parts = [_]usize{ P * patch_width * 4, P * head_dim * 4, P * head_dim * 4, P * width * 4, P * width * 2, P * width * 2, P * 3 * width * 2, heads * P * head_dim * 4, heads * P * head_dim * 4, heads * P * head_dim * 4, @as(usize, heads) * v.chunk * P * 4, @as(usize, heads) * v.chunk * head_dim * 4, P * mlp_inter * 2, P * hidden * 2, P * hidden * 2 };
        var total: usize = 0;
        for (parts) |n| total += std.mem.alignForward(usize, n, 256);
        v.scratch = try device.buffer(total, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer v.scratch.deinit();
        var off: usize = 0;
        for ([_]*Ref{ &v.patches, &v.cos, &v.sin, &v.x, &v.xn, &v.att, &v.qkv_rows, &v.q, &v.k, &v.v, &v.s, &v.o, &v.mid, &v.p1, &v.p2 }, parts) |view, n| {
            view.* = .{ .buf = v.scratch, .off = off };
            off += std.mem.alignForward(usize, n, 256);
        }
        return v;
    }

    pub fn deinit(v: *Vision) void {
        v.scratch.deinit();
        v.stacked.deinit();
        v.release(3 * 2 * 2 + glue.len);
        v.lib.deinit();
        v.gpa.destroy(v);
    }

    /// The first `n` pipelines in the order `load` makes them.
    fn release(v: *Vision, n: usize) void {
        var i: usize = 0;
        for (&v.gemm) |*a| for (a) |*b| for (b) |p| {
            if (i == n) return;
            p.deinit();
            i += 1;
        };
        inline for (glue) |kv| {
            if (i == n) return;
            @field(v, kv[0]).deinit();
            i += 1;
        }
    }

    /// Bytes of the tower's own buffers (beside the checkpoint's shards).
    pub fn bytes(v: *const Vision) u64 {
        return v.stacked.length() + v.scratch.length();
    }

    /// Buffers to keep resident with the model's (its other weights are in the checkpoint's shards).
    pub fn buffers(v: *const Vision) [2]mtl.Buffer {
        return .{ v.stacked, v.scratch };
    }

    /// One image's patches ([gh * gw, patch_width] fp32, merge order) to its gh * gw / 4 rows at `out` (bf16, hidden).
    pub fn encode(v: *Vision, e: mtl.ComputeEncoder, pixels: []const f32, gh: u32, gw: u32, out: Ref) !void {
        const P: u32 = gh * gw;
        if (P == 0 or gh % merge != 0 or gw % merge != 0 or pixels.len != @as(usize, P) * patch_width) return error.ImageGrid;
        if (P > max_patches) return error.ImageTooLarge;
        v.tables(pixels, gh, gw);
        v.gemmAt(e, 1, v.patches, v.patch_w, v.x, P, width, patch_width, patch_width, patch_width, width, 1, 0, 0, 0);
        v.layerNorm(e, v.x, v.pre_w, v.pre_b, v.xn, P);
        v.copy(e, v.widen, v.xn, v.x, P * width);
        for (v.layers) |l| v.layer(e, l, P);
        // the projector: each patch's rows, then four neighbours' (merge order) as one row of p2 read [P / 4, 4 hidden]
        const T = P / (merge * merge);
        v.copy(e, v.narrow, v.x, v.xn, P * width);
        v.linear(e, v.xn, v.proj[0][0], v.p1, P, hidden, width);
        v.gelu_(e, v.p1, v.proj[0][1], P, hidden);
        v.linear(e, v.p1, v.proj[1][0], v.p2, P, hidden, hidden);
        v.bias_(e, v.p2, v.proj[1][1], P, hidden);
        v.linear(e, v.p2, v.proj[2][0], v.p1, T, hidden, merge * merge * hidden);
        v.gelu_(e, v.p1, v.proj[2][1], T, hidden);
        v.linear(e, v.p1, v.proj[3][0], out, T, hidden, hidden);
        v.bias_(e, out, v.proj[3][1], T, hidden);
    }

    /// Rows of an encoded image over `n` embedded prompt rows (`dst`, the forward's fp32 residual).
    pub fn inject(v: *const Vision, e: mtl.ComputeEncoder, src: Ref, dst: Ref, n: u32) void {
        v.copy(e, v.widen, src, dst, n * hidden);
    }

    /// The host's part: the patches, and each patch's cos and sin over the moved head dims.
    fn tables(v: *Vision, pixels: []const f32, gh: u32, gw: u32) void {
        const P: usize = @as(usize, gh) * gw;
        @memcpy(@as([*]f32, @ptrCast(@alignCast(v.scratch.contents() + v.patches.off)))[0..pixels.len], pixels);
        const cs: [*]f32 = @ptrCast(@alignCast(v.scratch.contents() + v.cos.off));
        const sn: [*]f32 = @ptrCast(@alignCast(v.scratch.contents() + v.sin.off));
        var inv: [freqs]f64 = undefined;
        for (&inv, 0..) |*f, i| f.* = 1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / (2 * freqs));
        for (0..P) |i| {
            const p = vl.grid.place(i, gw, merge);
            const pos = [3]f32{ 0, @floatFromInt(p[0]), @floatFromInt(p[1]) }; // t, h, w: an image is frame 0
            for (0..head_dim / 2) |j| {
                // slot j and j + 40 hold MiniMax's dims j and j + 39 (j < 39): axis j / 13, frequency j % 13
                const a: f32 = if (j < rotated / 2) pos[j / freqs] * @as(f32, @floatCast(inv[j % freqs])) else 0;
                const c, const s = .{ @cos(a), @sin(a) };
                for ([2]usize{ j, j + head_dim / 2 }) |d| {
                    cs[i * head_dim + d] = c;
                    sn[i * head_dim + d] = s;
                }
            }
        }
    }

    fn layer(v: *Vision, e: mtl.ComputeEncoder, l: Layer, P: u32) void {
        v.layerNorm(e, v.x, l.ln1_w, l.ln1_b, v.xn, P);
        v.linear(e, v.xn, l.qkv_w, v.qkv_rows, P, 3 * width, width);
        e.setPipeline(v.qkv);
        for ([_]Ref{ v.qkv_rows, l.qkv_b, v.cos, v.sin, v.q, v.k, v.v }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue([3]u32{ P, heads, head_dim }, 7);
        e.dispatchThreads(mtl.Size.of(head_dim, 3 * heads, P), mtl.Size.of(head_dim, 256 / head_dim, 1));
        var q0: u32 = 0;
        while (q0 < P) : (q0 += v.chunk) {
            const lc = @min(v.chunk, P - q0);
            const stride = @as(usize, P) * head_dim;
            v.gemmAt(e, 1, v.q.at(@as(usize, q0) * head_dim * 4), v.k, v.s, lc, P, head_dim, head_dim, head_dim, P, heads, stride, stride, @as(usize, lc) * P);
            e.setPipeline(v.softmax);
            e.setBuffer(v.s.buf, v.s.off, 0);
            e.setValue(P, 1);
            e.setValue(1.0 / @sqrt(@as(f32, head_dim)), 2);
            e.dispatchGroups(mtl.Size.of(@as(usize, heads) * lc, 1, 1), mtl.Size.of(256, 1, 1));
            v.gemmAt(e, 2, v.s, v.v, v.o, lc, head_dim, P, P, head_dim, head_dim, heads, @as(usize, lc) * P, stride, @as(usize, lc) * head_dim);
            e.setPipeline(v.heads_);
            e.setBuffer(v.o.buf, v.o.off, 0);
            e.setBuffer(v.att.buf, v.att.off, 1);
            e.setValue([4]u32{ lc, q0, heads, head_dim }, 2);
            e.dispatchThreads(mtl.Size.of(width, lc, 1), mtl.Size.of(256, 1, 1));
        }
        v.linear(e, v.att, l.out_w, v.xn, P, width, width);
        v.residual_(e, v.xn, l.out_b, P, width);
        v.layerNorm(e, v.x, l.ln2_w, l.ln2_b, v.xn, P);
        v.linear(e, v.xn, l.fc1_w, v.mid, P, mlp_inter, width);
        v.gelu_(e, v.mid, l.fc1_b, P, mlp_inter);
        v.linear(e, v.mid, l.fc2_w, v.xn, P, width, mlp_inter);
        v.residual_(e, v.xn, l.fc2_b, P, width);
    }

    /// y [M, N] = x [M, K] w^T in bf16 with fp32 sums (w as stored, [N, K]).
    fn linear(v: *Vision, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, M: u32, N: u32, K: u32) void {
        v.gemmAt(e, 0, x, w, y, M, N, K, K, K, N, 1, 0, 0, 0);
    }

    /// tfp::gemm_nax: d [batch, M, N] = a x b (kind 0: bf16 a b^T, 1: fp32 a b^T, 2: fp32 a b), batches `s*` apart.
    fn gemmAt(v: *Vision, e: mtl.ComputeEncoder, kind: usize, a: Ref, b: Ref, d: Ref, M: u32, N: u32, K: u32, lda: u32, ldb: u32, ldd: u32, batch: u32, sa: usize, sb: usize, sd: usize) void {
        const tn = (N + 127) / 128;
        const tm = (M + 63) / 64;
        e.setPipeline(v.gemm[kind][@intFromBool(M % 64 == 0)][@intFromBool(N % 128 == 0)]);
        e.setBuffer(a.buf, a.off, 0);
        e.setBuffer(b.buf, b.off, 1);
        var p: [16]i32 = @splat(0);
        const vals = [_]usize{ M, N, K, lda, ldb, ldd, tn, tm, 2, K / 256, sa, sb, sd, 0, 0 };
        for (vals, 0..) |x, i| p[i] = @intCast(x);
        e.setBytes(std.mem.asBytes(&p), 2);
        e.setBuffer(d.buf, d.off, 3);
        e.dispatchThreads(mtl.Size.of(@as(usize, tn << 2) * 32, (tm + 3) / 4 * 4, @as(usize, batch) * 2), mtl.Size.of(32, 4, 2));
    }

    fn layerNorm(v: *Vision, e: mtl.ComputeEncoder, x: Ref, w: Ref, b: Ref, y: Ref, rows: u32) void {
        e.setPipeline(v.norm);
        for ([_]Ref{ x, w, b, y }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(extern struct { rows: u32, width: u32, eps: f32 }{ .rows = rows, .width = width, .eps = eps }, 4);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// y = gelu(y + b), the exact (erf) GELU.
    fn gelu_(v: *Vision, e: mtl.ComputeEncoder, y: Ref, b: Ref, rows: u32, n: u32) void {
        e.setPipeline(v.gelu);
        e.setBuffer(y.buf, y.off, 0);
        e.setBuffer(b.buf, b.off, 1);
        e.setValue([2]u32{ n, 1 }, 2);
        e.dispatchThreads(mtl.Size.of(n, rows, 1), mtl.Size.of(256, 1, 1));
    }

    fn bias_(v: *Vision, e: mtl.ComputeEncoder, y: Ref, b: Ref, rows: u32, n: u32) void {
        e.setPipeline(v.bias);
        e.setBuffer(y.buf, y.off, 0);
        e.setBuffer(b.buf, b.off, 1);
        e.setValue(n, 2);
        e.dispatchThreads(mtl.Size.of(n, rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// x += y + b on the residual stream.
    fn residual_(v: *Vision, e: mtl.ComputeEncoder, y: Ref, b: Ref, rows: u32, n: u32) void {
        e.setPipeline(v.residual);
        for ([_]Ref{ v.x, y, b }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(n, 3);
        e.dispatchThreads(mtl.Size.of(n, rows, 1), mtl.Size.of(256, 1, 1));
    }

    fn copy(_: *const Vision, e: mtl.ComputeEncoder, p: mtl.Pipeline, src: Ref, dst: Ref, n: u32) void {
        e.setPipeline(p);
        e.setBuffer(src.buf, src.off, 0);
        e.setBuffer(dst.buf, dst.off, 1);
        e.setValue(n, 2);
        e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
    }
};

fn get(c: *const ck.Checkpoint, name: []const u8, dtype: ck.DType, count: usize) !ck.Tensor {
    const t = try c.get(name);
    if (t.dtype != dtype or t.count() != count) {
        std.log.err("minimax_m3 vision: {s} is not {s}[{d}]", .{ name, @tagName(dtype), count });
        return error.UnexpectedMiniMaxVisionTensor;
    }
    return t;
}

test "the moved head dims pair MiniMax's rotary halves and keep every dim once" {
    var seen: [head_dim]bool = @splat(false);
    for (0..head_dim) |j| seen[moved(j)] = true;
    for (seen) |s| try std.testing.expect(s);
    for (0..39) |j| try std.testing.expectEqual(moved(j) + 39, moved(j + head_dim / 2));
    try std.testing.expectEqual([2]usize{ 78, 79 }, [2]usize{ moved(39), moved(79) });
}
