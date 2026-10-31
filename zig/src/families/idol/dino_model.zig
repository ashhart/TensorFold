//! Idol's DINOv3 ViT-H+/16 on Metal: a 1024 x 1024 image to feature1 [4101, 1280] as idol-sdk's ImageEncoder makes it.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("dino_config.zig");
const host = @import("dino_host.zig");
const wts = @import("dino_weights.zig");
const ks = @import("dino_kernels.zig");
const Ref = ks.Ref;

const W = cfg.width;
const R = cfg.rows;

/// One encode's seconds: wall clock, on the GPU, and the host's part before it.
pub const Timing = struct { wall: f64, gpu: f64, host: f64 };

/// The scratch buffer's parts in bytes, in the order of Encoder's views.
const parts = [_]usize{
    R * W * 2,           R * W * 2,       R * 3 * W * 2,                 R * W * 2,                      R * W * 2,                      R * W * 2, R * W * 2, R * W * 2,
    R * 2 * cfg.ffn * 2, R * cfg.ffn * 2, cfg.patches * cfg.patch_k * 2, cfg.patches * cfg.head_dim * 2, cfg.patches * cfg.head_dim * 2, R * W * 2, R * W * 4,
};

pub fn scratchBytes() usize {
    var total: usize = 0;
    for (parts) |n| total += std.mem.alignForward(usize, n, 256);
    return total;
}

/// transformers' DINOv3ViTModel in bf16, its last_hidden_state, then ImageEncoder's F.layer_norm in fp32.
pub const Encoder = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    k: ks.Kernels,
    w: wts.Weights,
    scratch: mtl.Buffer,
    // bf16 rows: the stream, a norm's output, packed qkv, heads [20, rows, 64], attention rows, a projection's output
    x: Ref,
    xn: Ref,
    qkv: Ref,
    q: Ref,
    k_: Ref,
    v: Ref,
    att: Ref,
    y: Ref,
    /// the MLP's gate | up rows, then its SwiGLU rows
    gu: Ref,
    h: Ref,
    /// the patch embedding's input [4096, 768], and RoPE's cos and sin [4096, 64], bf16
    pix: Ref,
    cos: Ref,
    sin: Ref,
    /// last_hidden_state (bf16) and feature1 (fp32)
    hidden: Ref,
    out: Ref,

    /// The encoder of timm snapshot folder `dir` (config.json, model.safetensors).
    pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8) !*Encoder {
        // qualified on the M5's tensor units alone, as the denoiser is
        if (!device.tensorUnits()) {
            std.log.err("idol dino: {s} has no tensor units; this encoder runs on the M5's alone, the only chip it was checked on", .{device.name()});
            return error.TensorUnitsRequired;
        }
        const t = try gpa.create(Encoder);
        errdefer gpa.destroy(t);
        t.* = undefined;
        t.gpa = gpa;
        t.device = device;
        t.queue = try device.queue();
        errdefer t.queue.deinit();
        t.k = try ks.Kernels.load(gpa, device);
        errdefer t.k.deinit();
        t.w = try wts.load(gpa, io, device, dir);
        errdefer t.w.deinit();
        t.scratch = try device.buffer(scratchBytes(), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        var at: usize = 0;
        const views = [_]*Ref{ &t.x, &t.xn, &t.qkv, &t.q, &t.k_, &t.v, &t.att, &t.y, &t.gu, &t.h, &t.pix, &t.cos, &t.sin, &t.hidden, &t.out };
        for (views, parts) |view, n| {
            view.* = .{ .buf = t.scratch, .off = at };
            at += std.mem.alignForward(usize, n, 256);
        }
        host.rope(halves(t.cos, cfg.patches * cfg.head_dim), halves(t.sin, cfg.patches * cfg.head_dim));
        return t;
    }

    pub fn deinit(t: *Encoder) void {
        t.scratch.deinit();
        t.w.deinit();
        t.k.deinit();
        t.queue.deinit();
        t.gpa.destroy(t);
    }

    fn halves(r: Ref, n: usize) []u16 {
        return @as([*]u16, @ptrCast(@alignCast(r.addr())))[0..n];
    }

    fn floats(r: Ref, n: usize) []f32 {
        return @as([*]f32, @ptrCast(@alignCast(r.addr())))[0..n];
    }

    /// The patches of `pixels` [3, 1024, 1024] in [0, 1] and the prefix rows, ready for `embed`.
    fn stage(t: *Encoder, pixels: []const f32) !void {
        if (pixels.len != 3 * cfg.image * cfg.image) return error.ImageSize;
        host.patchRows(pixels, halves(t.pix, cfg.patches * cfg.patch_k));
    }

    /// feature1 [4101, 1280] fp32 of an image's `pixels` [3, 1024, 1024] in [0, 1] (IdolPipeline's tensor).
    pub fn encode(t: *Encoder, pixels: []const f32, feature1: []f32) !Timing {
        if (feature1.len != R * W) return error.FeatureRows;
        const started = mtl.clock.seconds();
        try t.stage(pixels);
        const spent = mtl.clock.seconds() - started;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = t.queue.commandBuffer();
        const e = cb.compute(.serial);
        t.embed(e);
        for (t.w.blocks) |b| t.block(e, b);
        t.k.final(e, t.x, t.w.norm, t.w.norm_b, t.hidden, t.out, R);
        const timing = try finish(cb, e, started, spent);
        @memcpy(feature1, floats(t.out, R * W));
        return timing;
    }

    /// The embedding alone: the stream's rows [4101, 1280] bf16 for `pixels` (the input of block 0).
    pub fn embedOnly(t: *Encoder, pixels: []const f32) !void {
        try t.stage(pixels);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = t.queue.commandBuffer();
        const e = cb.compute(.serial);
        t.embed(e);
        _ = try finish(cb, e, mtl.clock.seconds(), 0);
    }

    /// Block `index` alone on bf16 `rows` [4101, 1280]; `stream` then holds its output.
    pub fn blockAt(t: *Encoder, index: usize, rows: []const u16) !void {
        if (index >= cfg.layers or rows.len != R * W) return error.BlockRows;
        @memcpy(halves(t.x, R * W), rows);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = t.queue.commandBuffer();
        const e = cb.compute(.serial);
        t.block(e, t.w.blocks[index]);
        _ = try finish(cb, e, mtl.clock.seconds(), 0);
    }

    /// The last norm and F.layer_norm alone on the last block's bf16 `rows`, into `feature1`.
    pub fn finalAt(t: *Encoder, rows: []const u16, feature1: []f32) !void {
        if (rows.len != R * W or feature1.len != R * W) return error.BlockRows;
        @memcpy(halves(t.x, R * W), rows);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = t.queue.commandBuffer();
        const e = cb.compute(.serial);
        t.k.final(e, t.x, t.w.norm, t.w.norm_b, t.hidden, t.out, R);
        _ = try finish(cb, e, mtl.clock.seconds(), 0);
        @memcpy(feature1, floats(t.out, R * W));
    }

    /// The stream's rows as the last pass left them, bf16 bits.
    pub fn stream(t: *const Encoder) []const u16 {
        return halves(t.x, R * W);
    }

    /// last_hidden_state as the last encode left it, bf16 bits.
    pub fn lastHidden(t: *const Encoder) []const u16 {
        return halves(t.hidden, R * W);
    }

    /// The RoPE tables load computed, bf16 bits [4096, 64] each.
    pub fn ropeTables(t: *const Encoder) [2][]const u16 {
        return .{ halves(t.cos, cfg.patches * cfg.head_dim), halves(t.sin, cfg.patches * cfg.head_dim) };
    }

    fn finish(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder, started: f64, spent: f64) !Timing {
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("idol dino: a pass failed on the GPU: {s}", .{msg});
            return error.GpuFailure;
        }
        return .{ .wall = mtl.clock.seconds() - started, .gpu = cb.gpuSeconds(), .host = spent };
    }

    /// The class token and registers, then the 16 x 16 conv over the patches as one GEMM into rows 5 to 4100.
    fn embed(t: *Encoder, e: mtl.ComputeEncoder) void {
        @memcpy(halves(t.x, cfg.prefix * W), halves(t.w.prefix, cfg.prefix * W));
        t.k.linear(e, t.pix, t.w.patch, t.w.patch_b, t.x.at(cfg.prefix * W * 2), cfg.patches, W, cfg.patch_k, W);
    }

    /// One block: LayerNorm, attention (RoPE on the patches), LayerScale; LayerNorm, SwiGLU, LayerScale.
    fn block(t: *const Encoder, e: mtl.ComputeEncoder, b: wts.Block) void {
        const k = &t.k;
        k.norm(e, t.x, b.norm1, b.norm1_b, t.xn, R);
        k.linear(e, t.xn, b.qkv, t.w.zeros, t.qkv, R, 3 * W, W, 3 * W);
        k.heads(e, t.qkv, t.cos, t.sin, t.q, t.k_, t.v, R);
        k.attention(e, t.q, t.k_, t.v, t.att, cfg.heads, R, R);
        k.linear(e, t.att, b.proj, b.proj_b, t.y, R, W, W, W);
        k.scaleAdd(e, t.x, t.y, b.gamma1, R);
        k.norm(e, t.x, b.norm2, b.norm2_b, t.xn, R);
        k.linear(e, t.xn, b.fc1, b.fc1_b, t.gu, R, 2 * cfg.ffn, W, 2 * cfg.ffn);
        k.swiglu(e, t.gu, t.h, R);
        k.linear(e, t.h, b.fc2, b.fc2_b, t.y, R, W, cfg.ffn, W);
        k.scaleAdd(e, t.x, t.y, b.gamma2, R);
    }
};

test "the encoder's scratch fits one GPU buffer, each part 256-aligned" {
    try std.testing.expect(scratchBytes() < 1 << 31);
    try std.testing.expectEqual(@as(usize, 4101), R);
}
