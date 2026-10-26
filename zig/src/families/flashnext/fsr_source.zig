//! Slot addressing for the recorded expert arithmetic. No resident source is changed.
const std = @import("std");
pub const Kind = enum { gate, down };

fn once(a: std.mem.Allocator, text: []const u8, from: []const u8, to: []const u8) ![]const u8 {
    if (std.mem.count(u8, text, from) != 1) return error.FsrKernelPatch;
    return std.mem.replaceOwned(u8, a, text, from, to);
}

/// Each stride is in that component's element units; the pool has nine persistent component slabs.
pub fn patch(a: std.mem.Allocator, original: []const u8, kind: Kind) ![]const u8 {
    if (std.mem.indexOf(u8, original, "FSR_MAP") != null) return error.FsrKernelPatch;
    const parameter = if (kind == .gate) "  device float* WTS [[buffer(16)]],\n" else "  device bfloat16_t* Y [[buffer(9)]],\n";
    const extra = if (kind == .gate)
        "  const device int* FSR_MAP [[buffer(18)]],\n  constant uint* FSR_STRIDES [[buffer(19)]],\n"
    else
        "  const device int* FSR_MAP [[buffer(10)]],\n  constant uint* FSR_STRIDES [[buffer(11)]],\n";
    var text = try once(a, original, parameter, try std.mem.concat(a, u8, &.{ parameter, extra }));
    if (kind == .gate) {
        text = try once(a, text, "(GW, UW, GS, GB, US, UB, e * N + row0, X + r * K, lane, ag, au)", "(GW + size_t(FSR_MAP[e]) * FSR_STRIDES[0], UW + size_t(FSR_MAP[e]) * FSR_STRIDES[3], GS + size_t(FSR_MAP[e]) * FSR_STRIDES[1], GB + size_t(FSR_MAP[e]) * FSR_STRIDES[2], US + size_t(FSR_MAP[e]) * FSR_STRIDES[4], UB + size_t(FSR_MAP[e]) * FSR_STRIDES[5], size_t(row0), X + r * K, lane, ag, au)");
    } else {
        text = try once(a, text, "(DW, DS, DB, size_t(PICK[r * TOPK + k]) * D + d0, x, lane, out)", "(DW + size_t(FSR_MAP[PICK[r * TOPK + k]]) * FSR_STRIDES[6], DS + size_t(FSR_MAP[PICK[r * TOPK + k]]) * FSR_STRIDES[7], DB + size_t(FSR_MAP[PICK[r * TOPK + k]]) * FSR_STRIDES[8], size_t(d0), x, lane, out)");
    }
    return text;
}
