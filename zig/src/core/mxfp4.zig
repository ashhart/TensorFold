//! Host-only MXFP4: 32 E2M1 values in 16 bytes and one UE8M0 scale, with explicit nibble order.
const std = @import("std");
const st = @import("safetensors");

pub const Order = enum { low_first, high_first };
pub const group = 32;

/// The finite UE8M0 code denotes 2^(code-127); 255 is the reserved NaN scale.
pub fn scale64(code: u8) error{BadMxfp4Scale}!f64 {
    if (code == 255) return error.BadMxfp4Scale;
    return @bitCast((@as(u64, code) + 896) << 52);
}

fn code64(code: u8) f64 {
    const magnitude = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const value = magnitude[code & 7];
    return if (code & 8 != 0) -value else value;
}

fn nibble(byte: u8, column: usize, order: Order) u8 {
    const low = (column % 2 == 0) == (order == .low_first);
    return if (low) byte & 15 else byte >> 4;
}

/// Decode as FP64, preserving signed zero and finite products beyond FP32's range. No checkpoint is requantized.
pub fn decodeBlock(bytes: *const [16]u8, scale: u8, order: Order, out: *[32]f64) !void {
    const factor = try scale64(scale);
    for (out, 0..) |*value, i| value.* = code64(nibble(bytes[i / 2], i, order)) * factor;
}

/// Borrowed stored tensors [E,N,K/32,16] and [E,N,K/32]; the caller retains their bytes.
pub const Host = struct {
    blocks: []const u8,
    scales: []const u8,
    experts: usize,
    rows: usize,
    groups: usize,
    order: Order,

    pub fn init(blocks: st.Tensor, scales: st.Tensor, order: Order) !Host {
        if (blocks.dtype != .u8 or scales.dtype != .u8 or blocks.rank != 4 or scales.rank != 3 or
            blocks.shape[3] != 16) return error.BadMxfp4Shape;
        var count: usize = 1;
        for (0..3) |i| {
            if (blocks.shape[i] == 0 or blocks.shape[i] != scales.shape[i]) return error.BadMxfp4Shape;
            count = std.math.mul(usize, count, blocks.shape[i]) catch return error.BadMxfp4Shape;
        }
        const size = std.math.mul(usize, count, 16) catch return error.BadMxfp4Shape;
        _ = std.math.mul(usize, blocks.shape[2], group) catch return error.BadMxfp4Shape;
        if (blocks.bytes.len != size or scales.bytes.len != count) return error.BadMxfp4Shape;
        return .{ .blocks = blocks.bytes, .scales = scales.bytes, .experts = blocks.shape[0], .rows = blocks.shape[1], .groups = blocks.shape[2], .order = order };
    }

    pub fn columns(h: Host) usize {
        return h.groups * group;
    }

    pub fn value(h: Host, expert: usize, index: usize, column: usize) !f64 {
        if (expert >= h.experts or index >= h.rows or column >= h.columns()) return error.BadMxfp4Index;
        const block = (expert * h.rows + index) * h.groups + column / group;
        const byte = h.blocks[block * 16 + (column % group) / 2];
        return code64(nibble(byte, column, h.order)) * try scale64(h.scales[block]);
    }

    pub fn row(h: Host, expert: usize, index: usize, out: []f64) !void {
        if (out.len != h.columns()) return error.BadMxfp4Shape;
        for (out, 0..) |*v, column| v.* = try h.value(expert, index, column);
    }
};
