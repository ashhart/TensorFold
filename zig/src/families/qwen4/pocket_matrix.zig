//! Packed base views retain original codes; learned rows and columns use the shared affine decoder.
const std = @import("std");
const st = @import("core").safetensors;
const living = @import("living");
const cfg = @import("config.zig");
pub const Role = enum { gate, up, down };

pub fn role(name: []const u8) ?Role {
    inline for (std.enums.values(Role)) |r| {
        const tail = ".mlp." ++ (if (r == .gate) "gate_proj" else if (r == .up) "up_proj" else "down_proj");
        if (std.mem.endsWith(u8, name, tail)) return r;
    }
    return null;
}

pub fn admit(c: *const cfg.Config, name: []const u8, t: st.Tensor) !void {
    var problem: cfg.Problem = .{};
    const r = try c.quant.of(name, &problem);
    if (r.format != .mlx4 or r.group != 64 or c.hidden % 64 != 0 or c.intermediate % 64 != 0)
        return error.Qwen4PocketFormat;
    if (t.rank != 2 or (t.dtype != .u32 and t.dtype != .bf16)) return error.Qwen4PocketStorage;
}

/// The allocator owns any compacted output rows; input projection views borrow the immutable checkpoint.
pub fn base(a: std.mem.Allocator, c: *const cfg.Config, name: []const u8, t: st.Tensor) ![]const u8 {
    if (c.learned == 0) return t.bytes;
    const r = role(name) orelse return t.bytes;
    try admit(c, name, t);
    const divisor: usize = if (t.dtype == .u32) 8 else 64;
    const scalar: usize = t.dtype.size();
    const own = c.intermediate;
    const wide = try std.math.add(usize, own, c.learned);
    const columns = if (r == .down) wide else c.hidden;
    const rows = if (r == .down) c.hidden else wide;
    if (columns % divisor != 0 or t.shape[0] != rows or t.shape[1] != columns / divisor)
        return error.Qwen4PocketShape;
    const stride = try std.math.mul(usize, columns / divisor, scalar);
    if (t.bytes.len != try std.math.mul(usize, rows, stride)) return error.Qwen4PocketShape;
    if (r != .down) return t.bytes[0 .. own * stride];
    const kept = own / divisor * scalar;
    const out = try a.alloc(u8, rows * kept);
    for (0..rows) |row| @memcpy(out[row * kept ..][0..kept], t.bytes[row * stride ..][0..kept]);
    return out;
}

pub fn decode(c: *const cfg.Config, r: Role, parts: [3]st.Tensor, out: []f32) !void {
    const d = c.hidden;
    const n = c.learned;
    if (n != 64 or out.len != n * d) return error.Qwen4PocketShape;
    const width = if (r == .down) c.intermediate + n else d;
    const rows = if (r == .down) d else c.intermediate + n;
    for (parts, 0..) |t, i| {
        const div: usize = if (i == 0) 8 else 64;
        if (t.rank != 2 or t.dtype != (if (i == 0) st.DType.u32 else .bf16) or
            t.shape[0] != rows or t.shape[1] != width / div or t.bytes.len != rows * width / div * t.dtype.size())
            return error.Qwen4PocketShape;
    }
    const words = std.mem.bytesAsSlice(u32, parts[0].bytes);
    const scales = std.mem.bytesAsSlice(u16, parts[1].bytes);
    const biases = std.mem.bytesAsSlice(u16, parts[2].bytes);
    const q = living.families.qwen4.quant;
    if (r != .down) {
        const at = c.intermediate * d;
        living.affine.dequantize(q, words[at / 8 ..], scales[at / 64 ..], biases[at / 64 ..], out);
    } else {
        var col: [64]f32 = undefined;
        for (0..d) |row| {
            const at = row * width + c.intermediate;
            living.affine.dequantize(q, words[at / 8 ..][0 .. n / 8], scales[at / 64 ..][0 .. n / 64], biases[at / 64 ..][0 .. n / 64], &col);
            for (col, 0..) |value, neuron| out[neuron * d + row] = value;
        }
    }
    for (out) |value| if (!std.math.isFinite(value)) return error.Qwen4PocketNonfinite;
}
