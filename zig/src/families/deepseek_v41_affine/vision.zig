//! DeepSeek-V4.1-Flash's image tower (vision.py): patch embedding, 32 blocks, final norm, 3x3 aligner.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const row = @import("../../core/row_projection.zig");
const am = @import("../../core/affine_mm.zig");
const c = @import("config.zig");
const Model = @import("model.zig").Model;

pub const width = 1024;
pub const heads = 16;
pub const blocks = 32;
pub const inter = 2816;
pub const patch = 14;
pub const patch_values = 3 * patch * patch; // 588
pub const downsample = 3;
pub const max_tokens = 1024; // vision_max_n_token
pub const min_pixels = 295936; // vision_min_pixels (544 x 544)
pub const eps: f32 = 1e-6;
pub const rope_theta: f32 = 10000;

const Tensor = ck.Tensor;
const Linear = row.Weights;

pub const Block = struct { norm1: Tensor, wqkv: Linear, bqkv: Tensor, wo: Linear, bo: Tensor, norm2: Tensor, w1: Linear, w2: Linear };

pub const Weights = struct {
    patch_w: Tensor, // bf16 [1024, 588]
    patch_b: Tensor,
    blocks: [blocks]Block,
    norm: Tensor,
    w1: Linear, // the aligner: [5120, 9216] and [5120, 5120], with biases
    b1: Tensor,
    w2: Linear,
    b2: Tensor,
    start: Tensor, // the span's learned embeddings, bf16 [5120] each
    end: Tensor,
    newline: Tensor,
};

fn get(ckpt: *const ck.Checkpoint, name: []const u8, dtype: ck.DType, shape: []const usize) !Tensor {
    const t = ckpt.get(name) catch |e| {
        std.log.err("deepseek_v41: the checkpoint has no {s} (convert its vision shard again)", .{name});
        return e;
    };
    if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
        std.log.err("{s}: unexpected vision tensor dtype or shape", .{name});
        return error.UnexpectedDeepseekTensor;
    }
    return t;
}

fn affine(ckpt: *const ck.Checkpoint, name: []const u8, n: usize, k: usize, w: c.Width) !Linear {
    var buf: [128]u8 = undefined;
    const q = try get(ckpt, try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), .u32, &.{ n, k * w.bits / 32 });
    const s = try get(ckpt, try std.fmt.bufPrint(&buf, "{s}.scales", .{name}), .bf16, &.{ n, k / w.group });
    const b = try get(ckpt, try std.fmt.bufPrint(&buf, "{s}.biases", .{name}), .bf16, &.{ n, k / w.group });
    const out: Linear = .{ .w = q.buffer, .w_off = q.offset, .scales = s.buffer, .s_off = s.offset, .biases = b.buffer, .b_off = b.offset, .n = n, .k = k, .group = w.group, .bits = w.bits };
    try out.validate();
    return out;
}

/// The tower's weights, or null when the checkpoint has none.
pub fn load(ckpt: *const ck.Checkpoint, dense: c.Width) !?Weights {
    if (!ckpt.has("vision.patch_embed.proj.weight")) return null;
    var w: Weights = undefined;
    w.patch_w = try get(ckpt, "vision.patch_embed.proj.weight", .bf16, &.{ width, patch_values });
    w.patch_b = try get(ckpt, "vision.patch_embed.proj.bias", .bf16, &.{width});
    for (&w.blocks, 0..) |*b, i| {
        var buf: [64]u8 = undefined;
        var nb: [96]u8 = undefined;
        const p = try std.fmt.bufPrint(&buf, "vision.blocks.{d}.", .{i});
        const name = struct {
            fn at(out: []u8, prefix: []const u8, rest: []const u8) ![]const u8 {
                return std.fmt.bufPrint(out, "{s}{s}", .{ prefix, rest });
            }
        }.at;
        b.norm1 = try get(ckpt, try name(&nb, p, "norm1.weight"), .bf16, &.{width});
        b.wqkv = try affine(ckpt, try name(&nb, p, "attn.wqkv"), 3 * width, width, dense);
        b.bqkv = try get(ckpt, try name(&nb, p, "attn.wqkv.bias"), .bf16, &.{3 * width});
        b.wo = try affine(ckpt, try name(&nb, p, "attn.wo"), width, width, dense);
        b.bo = try get(ckpt, try name(&nb, p, "attn.wo.bias"), .bf16, &.{width});
        b.norm2 = try get(ckpt, try name(&nb, p, "norm2.weight"), .bf16, &.{width});
        b.w1 = try affine(ckpt, try name(&nb, p, "mlp.w1"), 2 * inter, width, dense);
        b.w2 = try affine(ckpt, try name(&nb, p, "mlp.w2"), width, inter, dense);
    }
    w.norm = try get(ckpt, "vision.norm.weight", .bf16, &.{width});
    w.w1 = try affine(ckpt, "aligner.w1", c.hidden, width * downsample * downsample, dense);
    w.b1 = try get(ckpt, "aligner.w1.bias", .bf16, &.{c.hidden});
    w.w2 = try affine(ckpt, "aligner.w2", c.hidden, c.hidden, dense);
    w.b2 = try get(ckpt, "aligner.w2.bias", .bf16, &.{c.hidden});
    w.start = try get(ckpt, "image_start", .bf16, &.{c.hidden});
    w.end = try get(ckpt, "image_end", .bf16, &.{c.hidden});
    w.newline = try get(ckpt, "image_newline", .bf16, &.{c.hidden});
    return w;
}

/// The token grid an image's patch grid becomes (the aligner's 3x3 blocks, partial ones padded).
pub fn tokenGrid(n_h: u32, n_w: u32) [2]u32 {
    return .{ (n_h + downsample - 1) / downsample, (n_w + downsample - 1) / downsample };
}

/// The language model's rows an image takes: [start] + (grid width + line break) x grid height + [end].
pub fn spanTokens(n_h: u32, n_w: u32) u32 {
    const g = tokenGrid(n_h, n_w);
    return g[0] * (g[1] + 1) + 2;
}

const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
const LinArgs = extern struct { rows: i32, n: i32, k: i32 };
const RowsArgs = extern struct { n: i32, x_stride: i32, y_stride: i32, eps: f32 };
const UnfoldArgs = extern struct { n_h: i32, n_w: i32, out_w: i32 };

/// One image through tower and aligner: `pixels` bf16 [n_h * n_w, 588] to `out` bf16 [grid cells, 5120].
pub fn run(m: *Model, v: *const Weights, pixels: mtl.Buffer, n_h: u32, n_w: u32, out: mtl.Buffer) !void {
    const n: usize = @as(usize, n_h) * n_w;
    const g = tokenGrid(n_h, n_w);
    const t: usize = @as(usize, g[0]) * g[1];
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const dev = m.device;
    // per image: the stream, a norm's output, qkv, the MLP's hidden rows, its gate's, the unfold and the group sums
    const x = try dev.buffer(n * width * 2, opts);
    defer x.deinit();
    const h = try dev.buffer(n * width * 2, opts);
    defer h.deinit();
    const big = try dev.buffer(@max(n * 3 * width, n * 2 * inter, t * c.hidden) * 2, opts);
    defer big.deinit();
    const mid = try dev.buffer(@max(n * inter, t * 9 * width) * 2, opts);
    defer mid.deinit();
    const o = try dev.buffer(n * width * 2, opts);
    defer o.deinit();
    const sums = try dev.buffer(@max(n * inter, t * 9 * width) / 64 * 4 + 64, opts);
    defer sums.deinit();
    const freq = try dev.buffer(16 * 4, opts);
    defer freq.deinit();
    for (freq.slice(f32, 16), 0..) |*f, i| f.* = @floatCast(1.0 / std.math.pow(f64, rope_theta, @as(f64, @floatFromInt(2 * i)) / 32.0));

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    // unretained: the transient buffers above are freed by their defers, which run only after cb.wait() below
    const cb = m.queue.commandBufferUnretained();
    const e = cb.compute(.serial);
    const k = &m.kernels;
    const lin = struct {
        fn call(enc: mtl.ComputeEncoder, mm: *Model, w: Linear, xi: mtl.Buffer, y: mtl.Buffer, rows: usize, s: mtl.Buffer) void {
            if (mm.kernels.naxPipes(.{ .bits = w.bits, .group = @intCast(w.group) })) |p| if (rows >= 16) {
                const xr = Ref{ .buf = xi };
                am.rowSums(enc, p, @intCast(w.group), xr, Ref{ .buf = s }, @intCast(rows), @intCast(w.k));
                const q = .{ .w = Ref{ .buf = w.w, .off = w.w_off }, .s = Ref{ .buf = w.scales, .off = w.s_off }, .b = Ref{ .buf = w.biases, .off = w.b_off }, .n = @as(u32, @intCast(w.n)), .k = @as(u32, @intCast(w.k)) };
                am.dense(enc, p, xr, Ref{ .buf = s }, q, Ref{ .buf = y }, @intCast(rows));
                return;
            };
            const callr = row.call(&mm.kernels.rows, w, rows, false) catch unreachable;
            enc.setPipeline(callr.pipeline);
            enc.setBuffer(xi, 0, 0);
            enc.setBuffer(w.w, w.w_off, 1);
            enc.setBuffer(w.scales, w.s_off, 2);
            enc.setBuffer(w.biases, w.b_off, 3);
            enc.setBytes(std.mem.asBytes(&callr.dims), 4);
            enc.setBuffer(y, 0, 5);
            enc.dispatchThreads(mtl.Size.of(callr.groups * callr.threads, 1, 1), mtl.Size.of(callr.threads, 1, 1));
        }
        fn bias(enc: mtl.ComputeEncoder, mm: *Model, y: mtl.Buffer, b: Tensor, rows: usize, cols: usize, gelu: bool) void {
            enc.setPipeline(mm.kernels.get("ds_bias"));
            enc.setBuffer(y, 0, 0);
            enc.setBuffer(b.buffer, b.offset, 1);
            enc.setBytes(std.mem.asBytes(&[2]i32{ @intCast(cols), @intFromBool(gelu) }), 2);
            enc.dispatchThreads(mtl.Size.of(cols, rows, 1), mtl.Size.of(256, 1, 1));
        }
        fn rms(enc: mtl.ComputeEncoder, mm: *Model, xi: mtl.Buffer, w: Tensor, y: mtl.Buffer, rows: usize) void {
            enc.setPipeline(mm.kernels.get("ds_rms"));
            enc.setBuffer(xi, 0, 0);
            enc.setBuffer(w.buffer, w.offset, 1);
            enc.setBuffer(y, 0, 2);
            enc.setBytes(std.mem.asBytes(&RowsArgs{ .n = width, .x_stride = width, .y_stride = width, .eps = eps }), 3);
            enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
        }
        fn add(enc: mtl.ComputeEncoder, mm: *Model, xi: mtl.Buffer, y: mtl.Buffer, count: usize) void {
            enc.setPipeline(mm.kernels.get("ds_add"));
            enc.setBuffer(xi, 0, 0);
            enc.setBuffer(y, 0, 1);
            enc.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
        }
    };
    // the patch embedding (bias inside)
    e.setPipeline(k.get("ds_linear_bias"));
    e.setBuffer(pixels, 0, 0);
    e.setBuffer(v.patch_w.buffer, v.patch_w.offset, 1);
    e.setBuffer(v.patch_b.buffer, v.patch_b.offset, 2);
    e.setBuffer(x, 0, 3);
    e.setBytes(std.mem.asBytes(&LinArgs{ .rows = @intCast(n), .n = width, .k = patch_values }), 4);
    e.dispatchGroups(mtl.Size.of(width / 8, (n + 7) / 8, 1), mtl.Size.of(256, 1, 1));
    for (&v.blocks) |*b| {
        lin.rms(e, m, x, b.norm1, h, n);
        lin.call(e, m, b.wqkv, h, big, n, sums);
        lin.bias(e, m, big, b.bqkv, n, 3 * width, false);
        e.setPipeline(k.get("ds_vit_rope"));
        e.setBuffer(big, 0, 0);
        e.setBuffer(freq, 0, 1);
        e.setBytes(std.mem.asBytes(&@as(i32, @intCast(n_w))), 2);
        e.dispatchThreads(mtl.Size.of(32, 2 * heads, n), mtl.Size.of(32, 1, 1));
        e.setPipeline(k.get("ds_vit_attend"));
        e.setBuffer(big, 0, 0);
        e.setBuffer(h, 0, 1);
        e.setBytes(std.mem.asBytes(&@as(i32, @intCast(n))), 2);
        e.dispatchGroups(mtl.Size.of(heads, (n + 7) / 8, 1), mtl.Size.of(256, 1, 1));
        lin.call(e, m, b.wo, h, o, n, sums);
        lin.bias(e, m, o, b.bo, n, width, false);
        lin.add(e, m, x, o, n * width);
        lin.rms(e, m, x, b.norm2, h, n);
        lin.call(e, m, b.w1, h, big, n, sums);
        e.setPipeline(k.get("ds_vit_glu"));
        e.setBuffer(big, 0, 0);
        e.setBuffer(mid, 0, 1);
        e.setBytes(std.mem.asBytes(&@as(i32, inter)), 2);
        e.dispatchThreads(mtl.Size.of(inter, n, 1), mtl.Size.of(256, 1, 1));
        lin.call(e, m, b.w2, mid, o, n, sums);
        lin.add(e, m, x, o, n * width);
    }
    lin.rms(e, m, x, v.norm, h, n);
    // the aligner
    e.setPipeline(k.get("ds_unfold"));
    e.setBuffer(h, 0, 0);
    e.setBuffer(mid, 0, 1);
    e.setBytes(std.mem.asBytes(&UnfoldArgs{ .n_h = @intCast(n_h), .n_w = @intCast(n_w), .out_w = @intCast(g[1]) }), 2);
    e.dispatchThreads(mtl.Size.of(9 * width, t, 1), mtl.Size.of(256, 1, 1));
    lin.call(e, m, v.w1, mid, big, t, sums);
    lin.bias(e, m, big, v.b1, t, c.hidden, true);
    lin.call(e, m, v.w2, big, out, t, sums);
    lin.bias(e, m, out, v.b2, t, c.hidden, false);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.status() != .completed) {
        if (cb.failure()) |why| std.log.err("deepseek_v41: the image tower failed: {s}", .{why});
        return error.DeepseekGpuFailed;
    }
}
