//! Idol's DINOv3 ViT-H+/16 on CUDA: a 1024 x 1024 image to feature1 [4101, 1280] as idol-sdk's ImageEncoder makes it.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("dino_config.zig");
const host = @import("dino_host.zig");
const wts = @import("dino_cuda_weights.zig");
const ks = @import("dino_cuda_kernels.zig");
const gm = @import("cuda_gemm.zig");
const denoiser = @import("cuda_model.zig");

const W = cfg.width;
const R = cfg.rows;
const T = cfg.patches * cfg.head_dim;

pub const Timing = denoiser.Timing;
pub const seconds = denoiser.seconds;
pub const admitted = denoiser.admitted;

/// The scratch buffer's parts in bytes, in the order of Encoder's views.
const parts = [_]usize{
    R * W * 2,                     R * W * 2, R * 2 * cfg.ffn * 4, R * 3 * W * 2, R * W * 2, R * cfg.ffn * 2,
    cfg.patches * cfg.patch_k * 2, T * 2,     T * 2,               R * W * 2,     R * W * 4,
};

pub fn scratchBytes() usize {
    var total: usize = 0;
    for (parts) |n| total += std.mem.alignForward(usize, n, 256);
    return total;
}

/// transformers' DINOv3ViTModel in bf16, its last_hidden_state, then ImageEncoder's F.layer_norm in fp32.
pub const Encoder = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream_: cuda.Stream,
    k: ks.Kernels,
    gemms: gm.Gemms,
    w: *wts.Weights,
    scratch: cuda.DeviceBuffer,
    marks: [2]cuda.Event,
    // the stream, a norm's output (bf16); a linear's fp32 sums; packed q | k | v, attention's and SwiGLU's rows (bf16)
    x: u64 = 0,
    xn: u64 = 0,
    acc: u64 = 0,
    qkv: u64 = 0,
    att: u64 = 0,
    h: u64 = 0,
    /// the patch embedding's input [4096, 768], and RoPE's cos and sin [4096, 64], bf16
    pix: u64 = 0,
    cos: u64 = 0,
    sin: u64 = 0,
    /// last_hidden_state (bf16) and feature1 (fp32)
    hidden: u64 = 0,
    out: u64 = 0,
    /// The host's copies: the patch rows staged for upload, RoPE's tables, and rows downloaded for `stream`.
    pix_host: []u16 = &.{},
    rope: [2][]u16 = .{ &.{}, &.{} },
    rows_host: []u16 = &.{},

    /// The encoder of timm snapshot folder `dir` on `ctx`'s GPU, if it is one this family admits.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8) !*Encoder {
        const sm = try ctx.capability();
        if (!admitted(sm)) {
            var buf: [256]u8 = undefined;
            const name = ctx.name(&buf) catch "this GPU";
            std.log.err("idol dino: {s} (sm_{d}) is not a GPU this family admits; it runs on sm_120 and sm_121 alone", .{ name, sm });
            return error.UncheckedGpu;
        }
        const d = ctx.d;
        const t = try gpa.create(Encoder);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .d = d, .stream_ = undefined, .k = undefined, .gemms = undefined, .w = undefined, .scratch = undefined, .marks = undefined };
        t.pix_host = try gpa.alloc(u16, cfg.patches * cfg.patch_k);
        errdefer gpa.free(t.pix_host);
        t.rows_host = try gpa.alloc(u16, R * W);
        errdefer gpa.free(t.rows_host);
        t.rope[0] = try gpa.alloc(u16, T);
        errdefer gpa.free(t.rope[0]);
        t.rope[1] = try gpa.alloc(u16, T);
        errdefer gpa.free(t.rope[1]);
        host.rope(t.rope[0], t.rope[1]);
        t.stream_ = try cuda.Stream.init(d, false);
        errdefer t.stream_.deinit();
        t.k = try ks.Kernels.load(d);
        errdefer t.k.unload();
        t.gemms = try gm.Gemms.init(gpa, d);
        errdefer t.gemms.deinit();
        t.w = try wts.load(gpa, io, d, dir);
        errdefer {
            t.w.deinit();
            gpa.destroy(t.w);
        }
        t.scratch = try cuda.DeviceBuffer.alloc(d, scratchBytes());
        errdefer t.scratch.free();
        t.marks[0] = try cuda.Event.init(d, true);
        errdefer t.marks[0].deinit();
        t.marks[1] = try cuda.Event.init(d, true);
        errdefer t.marks[1].deinit();
        var at: usize = t.scratch.ptr;
        const views = [_]*u64{ &t.x, &t.xn, &t.acc, &t.qkv, &t.att, &t.h, &t.pix, &t.cos, &t.sin, &t.hidden, &t.out };
        for (views, parts) |view, n| {
            view.* = at;
            at += std.mem.alignForward(usize, n, 256);
        }
        try t.upload(t.cos, std.mem.sliceAsBytes(t.rope[0]));
        try t.upload(t.sin, std.mem.sliceAsBytes(t.rope[1]));
        return t;
    }

    pub fn deinit(t: *Encoder) void {
        for (&t.marks) |*m| m.deinit();
        t.scratch.free();
        t.w.deinit();
        t.gpa.destroy(t.w);
        t.gemms.deinit();
        t.k.unload();
        t.stream_.deinit();
        for (t.rope) |r| t.gpa.free(r);
        t.gpa.free(t.rows_host);
        t.gpa.free(t.pix_host);
        t.gpa.destroy(t);
    }

    fn upload(t: *const Encoder, at: u64, bytes: []const u8) !void {
        try t.d.check(t.d.api.cuMemcpyHtoDAsync_v2(at, bytes.ptr, bytes.len, t.stream_.handle), "cuMemcpyHtoDAsync");
        try t.stream_.synchronize();
    }

    fn download(t: *const Encoder, out: []u8, at: u64) !void {
        try t.d.check(t.d.api.cuMemcpyDtoHAsync_v2(out.ptr, at, out.len, t.stream_.handle), "cuMemcpyDtoHAsync");
        try t.stream_.synchronize();
    }

    /// The patches of `pixels` [3, 1024, 1024] in [0, 1] on the device, ready for `embed`.
    fn stage(t: *Encoder, pixels: []const f32) !void {
        if (pixels.len != 3 * cfg.image * cfg.image) return error.ImageSize;
        host.patchRows(pixels, t.pix_host);
        try t.upload(t.pix, std.mem.sliceAsBytes(t.pix_host));
    }

    /// feature1 [4101, 1280] fp32 of an image's `pixels` [3, 1024, 1024] in [0, 1] (IdolPipeline's tensor).
    pub fn encode(t: *Encoder, pixels: []const f32, feature1: []f32) !Timing {
        if (feature1.len != R * W) return error.FeatureRows;
        const started = seconds();
        try t.stage(pixels);
        const spent = seconds() - started;
        try t.marks[0].record(t.stream_);
        try t.embed();
        for (t.w.blocks) |b| try t.block(b);
        try ks.final(&t.k, t.stream_, t.x, t.w.norm, t.w.norm_b, t.hidden, t.out, R);
        try t.marks[1].record(t.stream_);
        try t.download(std.mem.sliceAsBytes(feature1), t.out);
        const gpu = @as(f64, try cuda.Event.elapsedMs(t.marks[0], t.marks[1])) / 1000;
        return .{ .wall = seconds() - started, .gpu = gpu, .host = spent };
    }

    /// The embedding alone: the stream's rows [4101, 1280] bf16 for `pixels` (the input of block 0).
    pub fn embedOnly(t: *Encoder, pixels: []const f32) !void {
        try t.stage(pixels);
        try t.embed();
        try t.stream_.synchronize();
    }

    /// Block `index` alone on bf16 `rows` [4101, 1280]; `stream` then holds its output.
    pub fn blockAt(t: *Encoder, index: usize, rows: []const u16) !void {
        if (index >= cfg.layers or rows.len != R * W) return error.BlockRows;
        try t.upload(t.x, std.mem.sliceAsBytes(rows));
        try t.block(t.w.blocks[index]);
        try t.stream_.synchronize();
    }

    /// The last norm and F.layer_norm alone on the last block's bf16 `rows`, into `feature1`.
    pub fn finalAt(t: *Encoder, rows: []const u16, feature1: []f32) !void {
        if (rows.len != R * W or feature1.len != R * W) return error.BlockRows;
        try t.upload(t.x, std.mem.sliceAsBytes(rows));
        try ks.final(&t.k, t.stream_, t.x, t.w.norm, t.w.norm_b, t.hidden, t.out, R);
        try t.download(std.mem.sliceAsBytes(feature1), t.out);
    }

    /// The stream's rows as the last pass left them, bf16 bits (valid until the next call).
    pub fn stream(t: *Encoder) ![]const u16 {
        try t.download(std.mem.sliceAsBytes(t.rows_host), t.x);
        return t.rows_host;
    }

    /// last_hidden_state as the last encode left it, bf16 bits (valid until the next call).
    pub fn lastHidden(t: *Encoder) ![]const u16 {
        try t.download(std.mem.sliceAsBytes(t.rows_host), t.hidden);
        return t.rows_host;
    }

    /// The RoPE tables load computed, bf16 bits [4096, 64] each.
    pub fn ropeTables(t: *const Encoder) [2][]const u16 {
        return .{ t.rope[0], t.rope[1] };
    }

    /// The class token and registers, then the 16 x 16 conv over the patches as one GEMM into rows 5 to 4100.
    fn embed(t: *Encoder) !void {
        try t.d.check(t.d.api.cuMemcpyDtoDAsync_v2(t.x, t.w.prefix, cfg.prefix * W * 2, t.stream_.handle), "cuMemcpyDtoDAsync");
        try t.gemms.run(t.stream_, .bf16, t.pix, t.w.patch, t.acc, cfg.patches, W, cfg.patch_k);
        try ks.bias(&t.k, t.stream_, t.acc, t.w.patch_b, t.x + cfg.prefix * W * 2, cfg.patches, W);
    }

    /// One block: LayerNorm, attention (RoPE on the patches), LayerScale; LayerNorm, SwiGLU, LayerScale.
    fn block(t: *Encoder, b: wts.Block) !void {
        const k = &t.k;
        const s = t.stream_;
        try ks.norm(k, s, t.x, b.norm1, b.norm1_b, t.xn, R);
        try t.gemms.run(s, .bf16, t.xn, b.qkv, t.acc, R, 3 * W, W);
        try ks.heads(k, s, t.acc, t.cos, t.sin, t.qkv, R);
        try ks.attention(k, s, t.qkv, t.att, R, cfg.heads, 1.0 / @sqrt(@as(f32, cfg.head_dim)));
        try t.gemms.run(s, .bf16, t.att, b.proj, t.acc, R, W, W);
        try ks.scaleAdd(k, s, t.x, t.acc, b.proj_b, b.gamma1, R);
        try ks.norm(k, s, t.x, b.norm2, b.norm2_b, t.xn, R);
        try t.gemms.run(s, .bf16, t.xn, b.fc1, t.acc, R, 2 * cfg.ffn, W);
        try ks.swiglu(k, s, t.acc, b.fc1_b, t.h, R);
        try t.gemms.run(s, .bf16, t.h, b.fc2, t.acc, R, W, cfg.ffn);
        try ks.scaleAdd(k, s, t.x, t.acc, b.fc2_b, b.gamma2, R);
    }
};

test "the encoder's scratch: fc1's fp32 sums are the widest part, every part 256-aligned, under 400 MB" {
    try std.testing.expectEqual(@as(usize, 4101), R);
    try std.testing.expectEqual(@as(usize, 167_976_960), parts[2]);
    for (parts) |n| try std.testing.expect(n <= parts[2]);
    try std.testing.expect(scratchBytes() < 400 << 20);
    // the kernels index rows and offsets as c_int
    try std.testing.expect(R * 2 * cfg.ffn < std.math.maxInt(c_int));
}

test "the patch embedding writes rows 5 to 4100: the prefix and the patches fill the stream" {
    try std.testing.expectEqual(R * W, cfg.prefix * W + cfg.patches * W);
    try std.testing.expectEqual(@as(usize, 768), cfg.patch_k);
}
