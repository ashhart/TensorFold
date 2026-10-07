//! Affine storage metadata keeps MLX packed words, scale and bias precision explicit before any device allocation.
const std = @import("std");
const st = @import("../../core/safetensors.zig");

pub const Spec = struct {
    bits: u8,
    group: usize,

    pub fn init(bits: usize, group: usize) !Spec {
        if (bits != 2 and bits != 3 and bits != 4 and bits != 5 and bits != 6 and bits != 8) return error.UnsupportedAffineBits;
        if (group != 32 and group != 64 and group != 128) return error.UnsupportedAffineGroup;
        return .{ .bits = @intCast(bits), .group = group };
    }

    pub fn words(s: Spec, k: usize) !usize {
        if (k == 0 or k % s.group != 0) return error.InvalidAffineWidth;
        const count = try std.math.mul(usize, k, s.bits);
        if (count % 32 != 0) return error.InvalidAffineWidth;
        return count / 32;
    }

    pub fn checkFlashKernel(s: Spec) !void {
        if (s.bits != 6 or s.group != 32) return error.UnsupportedFlashAffineKernel;
    }
};

pub const Shape = struct { experts: usize, n: usize, k: usize, words: usize, groups: usize, dtype: st.DType };

pub fn matrix(weight: st.Entry, scales: st.Entry, biases: st.Entry, spec: Spec) !Shape {
    if (weight.rank != 2 and weight.rank != 3) return error.InvalidAffineRank;
    if (scales.rank != weight.rank or biases.rank != weight.rank or weight.dtype != .u32) return error.InvalidAffineStorage;
    if (scales.dtype != biases.dtype or (scales.dtype != .bf16 and scales.dtype != .f16 and scales.dtype != .f32)) return error.InvalidAffineMetadataPrecision;
    for ([_]st.Entry{ weight, scales, biases }) |tensor| {
        var bytes = tensor.dtype.size();
        for (tensor.shape[0..tensor.rank]) |n| bytes = try std.math.mul(usize, bytes, n);
        if (tensor.end < tensor.begin or tensor.end - tensor.begin != bytes) return error.InvalidAffineByteCount;
    }
    const last = weight.rank - 1;
    if (scales.shape[last] == 0) return error.InvalidAffineWidth;
    const k = try std.math.mul(usize, scales.shape[last], spec.group);
    const words = try spec.words(k);
    for (0..weight.rank) |axis| {
        if (weight.shape[axis] == 0 or scales.shape[axis] == 0 or biases.shape[axis] != scales.shape[axis]) return error.InvalidAffineShape;
        if (axis == last) {
            if (weight.shape[axis] != words) return error.InvalidAffineShape;
        } else if (weight.shape[axis] != scales.shape[axis]) return error.InvalidAffineShape;
    }
    return .{ .experts = if (weight.rank == 3) weight.shape[0] else 1, .n = weight.shape[last - 1], .k = k, .words = words, .groups = k / spec.group, .dtype = scales.dtype };
}

pub fn code6(words: []const u32, index: usize) !u8 {
    const bit = try std.math.mul(usize, index, 6);
    const at = bit / 32;
    if (at >= words.len) return error.PackedCodeOutOfBounds;
    const shift: u5 = @intCast(bit % 32);
    var value = words[at] >> shift;
    if (shift > 26) {
        if (at + 1 >= words.len) return error.PackedCodeOutOfBounds;
        value |= words[at + 1] << @as(u5, @intCast(32 - @as(u32, shift)));
    }
    return @intCast(value & 63);
}

/// The packed code at `index` for `spec.bits`-wide codes over little-endian words in `bytes`, which may be unaligned.
pub fn code(spec: Spec, bytes: []const u8, index: usize) !u8 {
    const bit = try std.math.mul(usize, index, spec.bits);
    const at = bit / 32;
    if ((at + 1) * 4 > bytes.len) return error.PackedCodeOutOfBounds;
    const shift: u5 = @intCast(bit % 32);
    var value: u32 = std.mem.readInt(u32, bytes[at * 4 ..][0..4], .little) >> shift;
    if (shift + spec.bits > 32) {
        if ((at + 2) * 4 > bytes.len) return error.PackedCodeOutOfBounds;
        value |= std.mem.readInt(u32, bytes[(at + 1) * 4 ..][0..4], .little) << @as(u5, @intCast(32 - @as(u32, shift)));
    }
    const mask: u32 = (@as(u32, 1) << @as(u5, @intCast(spec.bits))) - 1;
    return @intCast(value & mask);
}

pub fn setCode6(words: []u32, index: usize, value: u8) !void {
    if (value > 63) return error.PackedCodeOutOfBounds;
    const bit = try std.math.mul(usize, index, 6);
    const at = bit / 32;
    if (at >= words.len or (bit % 32 > 26 and at + 1 >= words.len)) return error.PackedCodeOutOfBounds;
    const shift: u5 = @intCast(bit % 32);
    const mask = @as(u32, 63) << shift;
    words[at] = (words[at] & ~mask) | (@as(u32, value) << shift);
    if (shift > 26) {
        const carry: u5 = @intCast(32 - @as(u32, shift));
        const high_mask = @as(u32, 63) >> carry;
        words[at + 1] = (words[at + 1] & ~high_mask) | (@as(u32, value) >> carry);
    }
}
