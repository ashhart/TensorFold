//! MiniMax-M3.1-Flash's image tower on CUDA: patch embedding, 32 CLIP layers, axial RoPE, the merging projector.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const kern = @import("cuda_kernels.zig");

const vl = core.qwen_vl;
const picture = core.picture;
const vit = cuda.vit;

pub const width = 1280;
pub const heads = 16;
pub const head_dim = 80;
pub const depth = 32;
pub const mlp_inter = 5120;
pub const hidden = 6144;
pub const merge = 2;
pub const patch_width = 3 * 2 * 14 * 14;
const freqs = 13;
const theta: f64 = 10000;
const eps: f32 = 1e-5;
pub const max_tokens: u32 = 576; // the processor's max_pixels, 672 x 672: 48 x 48 patches
pub const max_patches: u32 = max_tokens * merge * merge;
pub const image_start: u32 = 200029;
pub const image_end: u32 = 200030;
pub const image_token: u32 = 200025;

pub const image_config: vl.Config = .{ .depth = depth, .hidden = width, .heads = heads, .intermediate = mlp_inter, .out_hidden = hidden, .patch = 14, .temporal = 2, .merge = merge, .positions = 0, .image_token = 0, .vision_start = 0, .vision_end = 0, .min_pixels = 4 * 28 * 28, .max_pixels = 672 * 672, .rescale = 1.0 / 255.0, .mean = .{ 0.48145466, 0.4578275, 0.40821073 }, .std = .{ 0.26862954, 0.26130258, 0.27577711 } };

/// A request's image limits (--vision): image tokens in all and images.
pub const Limits = struct { image_tokens: u32, max_images: u32 };

/// Prompt rows [at, at + n) take image rows [row, row + n).
pub const Span = struct { at: u32, n: u32, row: u32 };

/// An image's bytes to patches in merge order: torchvision's antialiased bicubic, then CLIP's mean and std.
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

const Layer = struct { ln1_w: u64, ln1_b: u64, qkv_w: u64, qkv_b: u64, out_w: u64, out_b: u64, ln2_w: u64, ln2_b: u64, fc1_w: u64, fc1_b: u64, fc2_w: u64, fc2_b: u64 };

/// The tower's weights and scratch for images up to max_patches patches.
pub const Tower = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    k: vit.Kernels,
    att: vit.Attention,
    g: vit.Gemms,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    patch_w: u64 = 0, // fp32 [width][patch_width]
    pre_w: u64 = 0,
    pre_b: u64 = 0,
    proj: [4][2]u64 = undefined, // linear_1, linear_2, merge linear_1, merge linear_2: weight, bias
    layers: [depth]Layer = undefined,
    // scratch
    px32: u64 = 0, // f32 [P][patch_width]
    x: u64 = 0, // bf16 [P][width]: the residual
    xn: u64 = 0,
    qkv: u64 = 0, // bf16 [P][3 width]
    o: u64 = 0, // bf16 [P][width]
    mid: u64 = 0, // bf16 [P][mlp_inter]
    p1: u64 = 0, // bf16 [P][hidden]
    p2: u64 = 0,
    acc: u64 = 0, // f32 [P][hidden]: a linear's sums when its bias is not fused
    cs: u64 = 0, // f32 [P][39]
    sn: u64 = 0,
    bytes: u64 = 0,

    fn alloc(t: *Tower, n: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(t.d, n);
        try t.bufs.append(t.gpa, b);
        t.bytes += n;
        return b.ptr;
    }

    fn up(t: *Tower, bytes: []const u8) !u64 {
        const p = try t.alloc(bytes.len);
        try t.d.check(t.d.api.cuMemcpyHtoD_v2(p, bytes.ptr, bytes.len), "cuMemcpyHtoD");
        return p;
    }

    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !*Tower {
        const t = try gpa.create(Tower);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .d = d, .k = try vit.Kernels.load(d), .att = try vit.Attention.load(d), .g = try vit.Gemms.init(gpa, d) };
        errdefer t.deinit();
        var ck = try core.Checkpoint.openModel(gpa, io, dir);
        defer core.checkpoint.cache.close(&ck);
        const pre = "vision_tower.vision_model.";
        {
            const w = try ck.expect(pre ++ "embeddings.patch_embedding.weight", .f32, &.{ width, 3, 2, 14, 14 });
            t.patch_w = try t.up(w.bytes); // fp32, as stored (the conv's [out][channel][frame][row][column] is [width][patch_width])
        }
        t.pre_w = try t.up((try ck.expect(pre ++ "pre_layrnorm.weight", .bf16, &.{width})).bytes);
        t.pre_b = try t.up((try ck.expect(pre ++ "pre_layrnorm.bias", .bf16, &.{width})).bytes);
        const projs = [4]struct { []const u8, usize }{ .{ "multi_modal_projector.linear_1", width }, .{ "multi_modal_projector.linear_2", hidden }, .{ "patch_merge_mlp.linear_1", hidden * merge * merge }, .{ "patch_merge_mlp.linear_2", hidden } };
        for (projs, &t.proj) |p, *dst| {
            var nb: [96]u8 = undefined;
            dst[0] = try t.up((try ck.expect(try std.fmt.bufPrint(&nb, "{s}.weight", .{p[0]}), .bf16, &.{ hidden, p[1] })).bytes);
            dst[1] = try t.up((try ck.expect(try std.fmt.bufPrint(&nb, "{s}.bias", .{p[0]}), .bf16, &.{hidden})).bytes);
        }
        for (&t.layers, 0..) |*l, i| {
            var nb: [128]u8 = undefined;
            const get = struct {
                fn f(c: *core.Checkpoint, buf: *[128]u8, li: usize, comptime rel: []const u8, dims: []const usize) !core.checkpoint.Tensor {
                    return c.expect(try std.fmt.bufPrint(buf, pre ++ "encoder.layers.{d}." ++ rel, .{li}), .bf16, dims);
                }
            }.f;
            l.ln1_w = try t.up((try get(&ck, &nb, i, "layer_norm1.weight", &.{width})).bytes);
            l.ln1_b = try t.up((try get(&ck, &nb, i, "layer_norm1.bias", &.{width})).bytes);
            l.ln2_w = try t.up((try get(&ck, &nb, i, "layer_norm2.weight", &.{width})).bytes);
            l.ln2_b = try t.up((try get(&ck, &nb, i, "layer_norm2.bias", &.{width})).bytes);
            // q | k | v stacked by rows, their biases likewise
            const w = try gpa.alloc(u8, 3 * width * width * 2);
            defer gpa.free(w);
            var b: [3 * width * 2]u8 = undefined;
            inline for (.{ "q", "k", "v" }, 0..) |which, j| {
                @memcpy(w[j * width * width * 2 ..][0 .. width * width * 2], (try get(&ck, &nb, i, "self_attn." ++ which ++ "_proj.weight", &.{ width, width })).bytes);
                @memcpy(b[j * width * 2 ..][0 .. width * 2], (try get(&ck, &nb, i, "self_attn." ++ which ++ "_proj.bias", &.{width})).bytes);
            }
            l.qkv_w = try t.up(w);
            l.qkv_b = try t.up(&b);
            l.out_w = try t.up((try get(&ck, &nb, i, "self_attn.out_proj.weight", &.{ width, width })).bytes);
            l.out_b = try t.up((try get(&ck, &nb, i, "self_attn.out_proj.bias", &.{width})).bytes);
            l.fc1_w = try t.up((try get(&ck, &nb, i, "mlp.fc1.weight", &.{ mlp_inter, width })).bytes);
            l.fc1_b = try t.up((try get(&ck, &nb, i, "mlp.fc1.bias", &.{mlp_inter})).bytes);
            l.fc2_w = try t.up((try get(&ck, &nb, i, "mlp.fc2.weight", &.{ width, mlp_inter })).bytes);
            l.fc2_b = try t.up((try get(&ck, &nb, i, "mlp.fc2.bias", &.{width})).bytes);
        }
        const P: usize = max_patches;
        t.px32 = try t.alloc(P * patch_width * 4);
        t.x = try t.alloc(P * width * 2);
        t.xn = try t.alloc(P * width * 2);
        t.qkv = try t.alloc(P * 3 * width * 2);
        t.o = try t.alloc(P * width * 2);
        t.mid = try t.alloc(P * mlp_inter * 2);
        t.p1 = try t.alloc(P * hidden * 2);
        t.p2 = try t.alloc(P * hidden * 2);
        t.acc = try t.alloc(P * hidden * 4);
        t.cs = try t.alloc(P * 39 * 4);
        t.sn = try t.alloc(P * 39 * 4);
        return t;
    }

    /// The tower's tensors, every one of which load uploads as stored (q | k | v stacked take the same bytes).
    const prefixes = [_][]const u8{ "vision_tower.vision_model.", "multi_modal_projector.", "patch_merge_mlp." };

    /// load's scratch, buffer by buffer: px32; x, xn, o; qkv; mid; p1, p2; acc; cs, sn.
    pub const scratch_bytes: u64 = @as(u64, max_patches) * (patch_width * 4 + 3 * width * 2 + 3 * width * 2 + mlp_inter * 2 + 2 * hidden * 2 + hidden * 4 + 2 * 39 * 4);

    /// The device bytes load takes for the checkpoint at `dir`: its tower tensors as stored, and the scratch.
    pub fn bytesFor(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
        var ck = try core.Checkpoint.openModelPrefixes(gpa, io, dir, &prefixes);
        defer core.checkpoint.cache.close(&ck);
        var total: u64 = scratch_bytes;
        for (ck.files.items) |*f| for (f.names.keys()) |name| {
            total += f.get(name).?.bytes.len;
        };
        return total;
    }

    pub fn deinit(t: *Tower) void {
        for (t.bufs.items) |*b| b.free();
        t.bufs.deinit(t.gpa);
        t.g.deinit();
        t.att.unload();
        t.k.unload();
        t.gpa.destroy(t);
    }

    /// One image's patches (fp32 at px32, merge order) to its gh gw / 4 rows at `out` (bf16, hidden).
    pub fn encode(t: *Tower, mk: *const kern.Kernels, s: cuda.Stream, gh: u32, gw: u32, out: u64) !void {
        const P: usize = @as(usize, gh) * gw;
        if (P == 0 or gh % merge != 0 or gw % merge != 0) return error.ImageGrid;
        if (P > max_patches) return error.ImageTooLarge;
        try t.tables(gh, gw);
        try kern.patchEmbed(mk, s, t.px32, t.patch_w, t.xn, P, width, patch_width);
        try t.k.layerNorm(s, t.xn, t.pre_w, t.pre_b, t.x, P, width, eps);
        const scale: f32 = 1.0 / @sqrt(@as(f32, head_dim));
        for (t.layers) |l| {
            try t.k.layerNorm(s, t.x, l.ln1_w, l.ln1_b, t.xn, P, width, eps);
            try vit.linear(&t.g, &t.k, s, t.acc, t.xn, l.qkv_w, l.qkv_b, t.qkv, P, 3 * width, width);
            try kern.vitRope(mk, s, t.qkv, t.cs, t.sn, P);
            try t.att.full(s, t.qkv, t.o, P, heads, head_dim, scale);
            try vit.linearAct(&t.g, &t.k, s, t.acc, t.o, l.out_w, l.out_b, t.xn, P, width, width, t.x, t.x, .none);
            try t.k.layerNorm(s, t.x, l.ln2_w, l.ln2_b, t.xn, P, width, eps);
            try vit.linearAct(&t.g, &t.k, s, t.acc, t.xn, l.fc1_w, l.fc1_b, t.mid, P, mlp_inter, width, null, t.mid, .gelu);
            try vit.linearAct(&t.g, &t.k, s, t.acc, t.mid, l.fc2_w, l.fc2_b, t.xn, P, width, mlp_inter, t.x, t.x, .none);
        }
        // the projector: each patch's rows, then four neighbours' (merge order, contiguous) as one row
        const T = P / (merge * merge);
        try vit.linearAct(&t.g, &t.k, s, t.acc, t.x, t.proj[0][0], t.proj[0][1], t.p1, P, hidden, width, null, t.p1, .gelu);
        try vit.linear(&t.g, &t.k, s, t.acc, t.p1, t.proj[1][0], t.proj[1][1], t.p2, P, hidden, hidden);
        try vit.linearAct(&t.g, &t.k, s, t.acc, t.p2, t.proj[2][0], t.proj[2][1], t.p1, T, hidden, merge * merge * hidden, null, t.p1, .gelu);
        try vit.linear(&t.g, &t.k, s, t.acc, t.p1, t.proj[3][0], t.proj[3][1], out, T, hidden, hidden);
    }

    /// Each patch's cos and sin over the 39 rotated pairs: axis i / 13 (frame, row, column), frequency i % 13.
    fn tables(t: *Tower, gh: u32, gw: u32) !void {
        _ = gh;
        const P: usize = max_patches;
        const cs = try t.gpa.alloc(f32, P * 39);
        defer t.gpa.free(cs);
        const sn = try t.gpa.alloc(f32, P * 39);
        defer t.gpa.free(sn);
        var inv: [freqs]f64 = undefined;
        for (&inv, 0..) |*f, i| f.* = 1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / (2 * freqs));
        const n = cs.len / 39;
        for (0..n) |i| {
            const p = vl.grid.place(i, gw, merge);
            const pos = [3]f32{ 0, @floatFromInt(p[0]), @floatFromInt(p[1]) };
            for (0..39) |j| {
                const a: f32 = pos[j / freqs] * @as(f32, @floatCast(inv[j % freqs]));
                cs[i * 39 + j] = @cos(a);
                sn[i * 39 + j] = @sin(a);
            }
        }
        try t.d.check(t.d.api.cuMemcpyHtoD_v2(t.cs, cs.ptr, cs.len * 4), "cuMemcpyHtoD");
        try t.d.check(t.d.api.cuMemcpyHtoD_v2(t.sn, sn.ptr, sn.len * 4), "cuMemcpyHtoD");
    }
};

/// A rank's images for a prompt pass: their rows and spans, the tower on the encoding rank, rank 0's offer.
pub const Images = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    tower: ?*Tower = null,
    pixels: cuda.DeviceBuffer, // f32 [max_patches][patch_width]: an image's patches as broadcast to the tower rank
    limits: Limits,
    marks: struct { image: u32, open: u32, close: u32 } = .{ .image = image_token, .open = image_start, .close = image_end },
    mark_ids: [3]u32 = .{ image_start, image_token, image_end },

    pub const part_type = "image";

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, limits: Limits, tower: bool) !*Images {
        const v = try gpa.create(Images);
        errdefer gpa.destroy(v);
        var px = try cuda.DeviceBuffer.alloc(d, @as(usize, max_patches) * patch_width * 4);
        errdefer px.free();
        v.* = .{ .gpa = gpa, .io = io, .d = d, .pixels = px, .limits = limits };
        if (tower) v.tower = try Tower.load(gpa, io, d, dir);
        return v;
    }

    pub fn deinit(v: *Images) void {
        if (v.tower) |t| t.deinit();
        v.pixels.free();
        v.gpa.destroy(v);
    }

    /// The server's processor (on the HTTP thread): at most the tower's 576 tokens an image.
    pub fn prepare(v: *const Images, a: std.mem.Allocator, bytes: []const u8, max: u32) !vl.geometry.Prepared {
        return prepare_(v.io, a, bytes, max);
    }

    pub fn explain(err: anyerror) ?[]const u8 {
        return vl.geometry.problem(err);
    }

    /// Bytes a stream's image rows take beside the tower: a pass's image tokens at the request limit, in bf16.
    pub fn streamBytes(v: *const Images) usize {
        return @as(usize, v.limits.image_tokens) * hidden * 2;
    }
};

/// A sequence's image rows over prompt rows [lo, hi) of its pass, into the residual rows at `resid` (row lo first).
pub fn splice(mk: *const kern.Kernels, s: cuda.Stream, rows: u64, spans: []const Span, lo: usize, hi: usize, resid: u64) !void {
    for (spans) |sp| {
        const a = @max(lo, sp.at);
        const b = @min(hi, sp.at + sp.n);
        if (a >= b) continue;
        try kern.splice(mk, s, rows + (sp.row + (a - sp.at)) * hidden * 2, resid + (a - lo) * hidden * 4, b - a, hidden);
    }
}

const prepare_ = prepare;

test "the tower's processor settings: 672 x 672 at most, 576 tokens an image" {
    try std.testing.expectEqual(@as(u32, 2304), max_patches);
    try std.testing.expectEqual(@as(u64, 672 * 672), image_config.max_pixels);
}
