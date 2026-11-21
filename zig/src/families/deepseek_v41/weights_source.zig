//! The checkpoint-source helpers: names, plain tensors, per-format scales and EXL3 matrices.
const std = @import("std");
const mtl = @import("metal");
const st = @import("core").safetensors;
const exl3 = @import("core").exl3_projection;
const wts = @import("weights.zig");
const Tensor = wts.Tensor;
const Matrix = wts.Matrix;
const AnyMatrix = wts.AnyMatrix;

pub fn named(buffer: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buffer, fmt, args);
}

/// A plain tensor the encoders decode as f32, f16 or bf16; the shape must match exactly.
pub fn plainEither(comptime binding: bool, source: anytype, name: []const u8, shape: []const usize, destination: ?*Tensor) !void {
    const tensor = try source.get(name);
    if ((tensor.dtype != .f32 and tensor.dtype != .f16 and tensor.dtype != .bf16) or tensor.rank != shape.len or
        !std.mem.eql(usize, tensor.shape[0..tensor.rank], shape))
    {
        std.log.err("unsupported tensor {s}: {s} {any}", .{ name, @tagName(tensor.dtype), tensor.shape[0..tensor.rank] });
        return error.UnsupportedTensor;
    }
    if (binding) {
        var count: usize = 1;
        for (shape) |dim| count = try std.math.mul(usize, count, dim);
        if (tensor.bytes != try std.math.mul(usize, count, tensor.dtype.size())) return error.UnsupportedTensor;
        if (tensor.offset % tensor.dtype.size() != 0 or tensor.offset > tensor.buffer.length() or
            tensor.bytes > tensor.buffer.length() - tensor.offset) return error.InvalidTensorView;
        destination.?.* = tensor;
    } else if (tensor.begin % 4 != 0) return error.InvalidTensorView;
}

pub fn plain(comptime binding: bool, source: anytype, name: []const u8, dtype: st.DType, shape: []const usize, destination: ?*Tensor) !void {
    const tensor = try source.get(name);
    if (tensor.dtype != dtype or tensor.rank != shape.len or
        !std.mem.eql(usize, tensor.shape[0..tensor.rank], shape))
    {
        std.log.err("unsupported backbone tensor {s}: {s} {any}", .{ name, @tagName(tensor.dtype), tensor.shape[0..tensor.rank] });
        return error.UnsupportedTensor;
    }
    const offset = if (binding) tensor.offset else tensor.begin;
    if (offset % dtype.size() != 0) return error.InvalidTensorView;
    if (binding) {
        var count: usize = 1;
        for (shape) |dim| count = try std.math.mul(usize, count, dim);
        if (tensor.bytes != try std.math.mul(usize, count, dtype.size())) return error.UnsupportedTensor;
        if (tensor.offset % dtype.size() != 0 or tensor.offset > tensor.buffer.length() or
            tensor.bytes > tensor.buffer.length() - tensor.offset) return error.InvalidTensorView;
        destination.?.* = tensor;
    }
}

pub fn has(source: anytype, name: []const u8) bool {
    return source.has(name);
}

const Scale = struct { name: []const u8, format: exl3.ScaleFormat };
pub fn scale(comptime binding: bool, source: anytype, prefix: []const u8, size: usize, comptime fp: []const u8, comptime signs: []const u8, storage: []u8) !Scale {
    var alternative: [160]u8 = undefined;
    const f = try named(storage, "{s}.{s}", .{ prefix, fp });
    const s = try named(&alternative, "{s}.{s}", .{ prefix, signs });
    const have_fp = has(source, f);
    if (have_fp == has(source, s)) return error.MissingOrAmbiguousExl3Scale;
    const name = if (have_fp) f else try named(storage, "{s}.{s}", .{ prefix, signs });
    const t = try source.get(name);
    if (t.rank != 1 or t.shape[0] != (if (have_fp) size else size / 16) or
        t.dtype != (if (have_fp) st.DType.f16 else st.DType.i16)) return error.UnsupportedExl3Scale;
    const offset = if (binding) t.offset else t.begin;
    if (offset % 2 != 0) return error.InvalidTensorView;
    if (binding) {
        const bytes = (if (have_fp) size else size / 16) * 2;
        if (t.bytes != bytes or t.offset % 2 != 0 or t.offset > t.buffer.length() or
            bytes > t.buffer.length() - t.offset) return error.InvalidTensorView;
    }
    return .{ .name = name, .format = if (have_fp) .f16 else .packed_signs };
}

/// Compressor/indexer matrices ride either storage: packs quantize; rank-split repacks stay dense with rotation scales.
pub fn matrixAny(comptime binding: bool, source: anytype, prefix: []const u8, k: usize, n: usize, destination: ?*AnyMatrix) !void {
    var name_buf: [160]u8 = undefined;
    if (has(source, try named(&name_buf, "{s}.weight", .{prefix}))) {
        const weight_name = try named(&name_buf, "{s}.weight", .{prefix});
        const w = try source.get(weight_name);
        if (w.dtype != .bf16 or w.rank != 2 or w.shape[0] != n or w.shape[1] != k) {
            std.log.err("unsupported dense tensor {s}: {s} {any}", .{ weight_name, @tagName(w.dtype), w.shape[0..w.rank] });
            return error.UnsupportedTensor;
        }
        const su = try source.get(try named(&name_buf, "{s}.suh", .{prefix}));
        const sv = try source.get(try named(&name_buf, "{s}.svh", .{prefix}));
        if (su.dtype != .f16 or su.rank != 1 or su.shape[0] != k or
            sv.dtype != .f16 or sv.rank != 1 or sv.shape[0] != n) return error.UnsupportedExl3Scale;
        if (binding) {
            if (w.offset % 2 != 0) return error.InvalidTensorView;
            destination.?.* = .{ .dense = .{ .weight = w, .suh = su, .svh = sv, .k = k, .n = n } };
        }
        return;
    }
    var m: Matrix = undefined;
    try matrix(binding, source, prefix, k, n, if (binding) &m else null);
    if (binding) destination.?.* = .{ .trellis = m };
}

pub fn matrix(comptime binding: bool, source: anytype, prefix: []const u8, k: usize, n: usize, destination: ?*Matrix) !void {
    var name: [160]u8 = undefined;
    const trellis = try source.get(try named(&name, "{s}.trellis", .{prefix}));
    if (trellis.dtype != .i16 or trellis.rank != 3 or trellis.shape[0] != k / 16 or
        trellis.shape[1] != n / 16 or trellis.shape[2] % 8 != 0) return error.UnsupportedExl3Shape;
    if ((if (binding) trellis.offset else trellis.begin) % 2 != 0) return error.InvalidTensorView;
    const halves = trellis.shape[2] / 8;
    if (halves < 2 or halves > 16 or (halves % 2 != 0 and halves > 7)) return error.UnsupportedExl3Width;
    // Pinned checkpoint uses mul1 throughout; missing markers cannot silently select the legacy codebook.
    const marker = try source.get(try named(&name, "{s}.mul1", .{prefix}));
    if (marker.dtype != .i32 or marker.rank != 0) return error.UnsupportedExl3Marker;
    if ((if (binding) marker.offset else marker.begin) % 4 != 0) return error.InvalidTensorView;
    if (has(source, try named(&name, "{s}.mcg", .{prefix})) or
        has(source, try named(&name, "{s}.3inst", .{prefix})) or
        has(source, try named(&name, "{s}.bias", .{prefix})))
    {
        std.log.err("unexpected exl3 parts beside {s}", .{prefix});
        return error.UnsupportedExl3Parts;
    }
    var su_name: [160]u8 = undefined;
    var sv_name: [160]u8 = undefined;
    const su = try scale(binding, source, prefix, k, "suh", "su", &su_name);
    const sv = try scale(binding, source, prefix, n, "svh", "sv", &sv_name);
    if (binding) {
        const expected = k / 16 * (n / 16) * halves * 16;
        if (trellis.bytes != expected or trellis.offset % 2 != 0 or
            trellis.offset > trellis.buffer.length() or expected > trellis.buffer.length() - trellis.offset)
            return error.InvalidTensorView;
        if (marker.bytes != 4 or marker.offset % 4 != 0 or marker.offset > marker.buffer.length() or
            4 > marker.buffer.length() - marker.offset) return error.InvalidTensorView;
        const marker_bytes = marker.buffer.contents()[marker.offset..][0..4];
        if (std.mem.readInt(u32, marker_bytes, .little) != exl3.format.Mul1Mul) return error.UnsupportedExl3Marker;
        const suh = try source.get(su.name);
        const svh = try source.get(sv.name);
        destination.?.* = .{
            .trellis = .{ .buffer = trellis.buffer, .offset = trellis.offset },
            .suh = .{ .buffer = suh.buffer, .offset = suh.offset },
            .svh = .{ .buffer = svh.buffer, .offset = svh.offset },
            .k = k,
            .n = n,
            .width = .{ .halves = @intCast(halves) },
            .codebook = .mul1,
            .suh_format = su.format,
            .svh_format = sv.format,
        };
        try destination.?.validate();
    }
}
