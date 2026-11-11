const std = @import("std");
const gpu = @import("metal.zig");
const mtl = gpu.mtl;
const Tensor = gpu.Tensor;
const Gpu = gpu.Gpu;

pub fn cast(g: *Gpu, x: Tensor, dtype: gpu.DType) !Tensor {
    const out = try g.allocDevice(x.count, dtype);
    errdefer out.deinit();
    try g.dispatch("im_cast", &.{ x, out }, [3]u32{ @intCast(x.count), @backingInt(x.dtype), @backingInt(dtype) }, mtl.Size.of(x.count, 1, 1), mtl.Size.of(256, 1, 1));
    return out;
}

pub fn headsCopy(g: *Gpu, x: Tensor, rows: usize, heads: usize, src_dim: usize, dst_dim: usize, stride: usize, offset: usize) !Tensor {
    return stridedHeads(g, x, rows, heads, src_dim, dst_dim, stride, offset, src_dim);
}

pub fn stridedHeads(g: *Gpu, x: Tensor, rows: usize, heads: usize, src_dim: usize, dst_dim: usize, stride: usize, offset: usize, head_stride: usize) !Tensor {
    if (heads == 0 or x.count != rows * stride or offset + (heads - 1) * head_stride + src_dim > stride) return error.TensorShape;
    const out = try g.allocDevice(rows * heads * dst_dim, x.dtype);
    errdefer out.deinit();
    try g.dispatch("im_heads", &.{ x, out }, [8]u32{ @intCast(rows), @intCast(heads), @intCast(src_dim), @intCast(dst_dim), @intCast(stride), @intCast(offset), @backingInt(x.dtype), @intCast(head_stride) }, mtl.Size.of(dst_dim, heads, rows), mtl.Size.of(@min(128, dst_dim), 1, 1));
    return out;
}

pub fn gelu(g: *Gpu, x: Tensor, exact: bool) !Tensor {
    const out = try g.allocDevice(x.count, x.dtype);
    errdefer out.deinit();
    try g.dispatch("im_gelu", &.{ x, out }, [3]u32{ @intCast(x.count), @backingInt(x.dtype), @intFromBool(exact) }, mtl.Size.of(x.count, 1, 1), mtl.Size.of(256, 1, 1));
    return out;
}

pub fn point(g: *Gpu, a: Tensor, b: Tensor, operation: u32, width: usize) !Tensor {
    const broadcast = operation == 1 or operation == 3;
    if (operation > 4 or width == 0 or a.dtype != b.dtype or (broadcast and b.count != width) or (!broadcast and a.count != b.count)) return error.TensorShape;
    const out = try g.allocDevice(a.count, a.dtype);
    errdefer out.deinit();
    try g.dispatch("im_point", &.{ a, b, out }, [4]u32{ @intCast(a.count), @intCast(width), @backingInt(a.dtype), operation }, mtl.Size.of(a.count, 1, 1), mtl.Size.of(256, 1, 1));
    return out;
}

pub fn norm(g: *Gpu, x: Tensor, weight: Tensor, bias: ?Tensor, width: usize, rms: bool, eps: f32) !Tensor {
    if (width == 0 or x.count % width != 0 or weight.count != width or weight.dtype != x.dtype) return error.TensorShape;
    if (bias) |b| if (b.count != width or b.dtype != x.dtype) return error.TensorShape;
    const out = try g.allocDevice(x.count, x.dtype);
    errdefer out.deinit();
    const Params = extern struct { width: u32, kind: u32, rms: u32, has_bias: u32, eps: f32 };
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const pipeline = try g.pipeline("im_norm");
    const pass = g.command();
    const cb = pass.cb;
    const enc = pass.enc;
    enc.setPipeline(pipeline);
    for ([_]Tensor{ x, weight, bias orelse weight, out }, 0..) |t, i| enc.setBuffer(t.buffer, t.offset, i);
    enc.setValue(Params{ .width = @intCast(width), .kind = @backingInt(x.dtype), .rms = @intFromBool(rms), .has_bias = @intFromBool(bias != null), .eps = eps }, 4);
    enc.dispatchGroups(mtl.Size.of(x.count / width, 1, 1), mtl.Size.of(256, 1, 1));
    try g.finish(cb, enc);
    return out;
}

pub fn attention(g: *Gpu, q: Tensor, k: Tensor, v: Tensor, rows: usize, heads: usize, kv_heads: usize, causal: bool, scale: f32) !Tensor {
    return attentionWindow(g, q, k, v, rows, heads, kv_heads, causal, scale, 0);
}

pub fn attentionWindow(g: *Gpu, q: Tensor, k: Tensor, v: Tensor, rows: usize, heads: usize, kv_heads: usize, causal: bool, scale: f32, window: usize) !Tensor {
    return attentionBatchedWindow(g, q, k, v, rows, heads, kv_heads, causal, scale, window, 1, rows, rows);
}

pub fn attentionBatched(g: *Gpu, q: Tensor, k: Tensor, v: Tensor, rows: usize, heads: usize, kv_heads: usize, causal: bool, scale: f32, batch: usize) !Tensor {
    return attentionBatchedWindow(g, q, k, v, rows, heads, kv_heads, causal, scale, 0, batch, rows, rows);
}

pub fn attentionPadded(g: *Gpu, q: Tensor, k: Tensor, v: Tensor, rows: usize, keys: usize, heads: usize, kv_heads: usize, scale: f32) !Tensor {
    return attentionBatchedWindow(g, q, k, v, rows, heads, kv_heads, true, scale, 0, 1, keys, rows);
}

pub fn attentionCross(g: *Gpu, q: Tensor, k: Tensor, v: Tensor, rows: usize, keys: usize, heads: usize, scale: f32) !Tensor {
    return attentionBatchedWindow(g, q, k, v, rows, heads, heads, false, scale, 0, 1, keys, keys);
}

fn attentionBatchedWindow(g: *Gpu, q: Tensor, k: Tensor, v: Tensor, rows: usize, heads: usize, kv_heads: usize, causal: bool, scale: f32, window: usize, batch: usize, keys: usize, storage_rows: usize) !Tensor {
    if (rows == 0 or heads == 0 or batch == 0 or keys == 0 or keys > storage_rows) return error.AttentionShape;
    const dim = q.count / rows / heads / batch;
    if (kv_heads == 0 or heads % kv_heads != 0 or q.dtype != k.dtype or q.dtype != v.dtype or
        (dim != 32 and dim != 64 and dim != 96 and dim != 128 and dim != 256) or q.count != batch * rows * heads * dim or k.count != batch * storage_rows * kv_heads * dim or k.count != v.count) return error.AttentionShape;
    const out = try g.allocDevice(q.count, q.dtype);
    errdefer out.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var name: [64]u8 = undefined;
    if (window != 0 and ((window != 128 and window != 512) or dim != 64 or q.dtype != .f32 or !causal)) return error.AttentionWindowShape;
    const key = if (window == 128) "im_attention_f32_window64" else if (window == 512) "im_attention_f32_window512" else try std.fmt.bufPrint(&name, "im_attention_{s}_{s}{s}", .{ @tagName(q.dtype), if (causal) "causal" else "full", if (dim == 32) "32" else if (dim == 64) "64" else if (dim == 96) "96" else if (dim == 256) "256" else "" });
    const pipeline = try g.pipeline(key);
    const pass = g.command();
    const cb = pass.cb;
    const enc = pass.enc;
    enc.setPipeline(pipeline);
    for ([_]Tensor{ q, k, v }, 0..) |t, i| enc.setBuffer(t.buffer, t.offset, i);
    const nq = (rows + 63) / 64;
    const nk = (keys + 31) / 32;
    const vals = [_]usize{
        batch,              heads, rows,        keys,                          heads / kv_heads, nq,             nk,                            rows / 64, keys / 32,      rows % 64,          keys % 32, 0,
        rows * heads * dim, dim,   heads * dim, storage_rows * kv_heads * dim, dim,              kv_heads * dim, storage_rows * kv_heads * dim, dim,       kv_heads * dim, rows * heads * dim, dim,       heads * dim,
    };
    var args: [24]i32 = undefined;
    for (vals, 0..) |value, i| args[i] = std.math.cast(i32, value) orelse return error.AttentionTooLarge;
    enc.setValue(args, 3);
    enc.setValue([4]f32{ scale, 0, 0, 0 }, 4);
    enc.setBuffer(out.buffer, 0, 5);
    enc.dispatchThreads(mtl.Size.of(nq * 32, heads * 4, batch), mtl.Size.of(32, 4, 1));
    try g.finish(cb, enc);
    return out;
}
