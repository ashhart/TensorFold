//! Dense appended rows and columns preserve base bytes and share one group-64 affine decoder.
const std = @import("std");
const affine = @import("affine.zig");
pub const Role = enum { gate, up, down };
pub const Part = enum { weight, scales, biases };
pub fn role(name: []const u8) ?Role {
    inline for (std.enums.values(Role), .{ "gate_proj", "up_proj", "down_proj" }) |r, suffix| {
        if (std.mem.endsWith(u8, name, "mlp." ++ suffix)) return r;
    }
    return null;
}
pub const View = struct {
    rank: usize,
    shape: [2]usize,
    bytes: []const u8,
    words: bool,
    valid: bool = true,
    pub fn of(t: anytype) View {
        return .{ .rank = t.rank, .shape = t.shape[0..2].*, .bytes = if (@TypeOf(t.bytes) == usize) t.host(u8) else t.bytes, .words = t.dtype == .u32, .valid = t.dtype == .u32 or t.dtype == .bf16 };
    }
};
fn admit(d: usize, own: usize, extra: usize, r: Role, part: Part, v: View) !struct { stride: usize, rows: usize } {
    if (d == 0 or own == 0 or d % 64 != 0 or own % 64 != 0 or extra != 64 or !v.valid or v.rank != 2 or v.words != (part == .weight)) return error.DensePocketShape;
    const cols = if (r == .down) try std.math.add(usize, own, extra) else d;
    const rows = if (r == .down) d else try std.math.add(usize, own, extra);
    const div: usize = if (part == .weight) 8 else 64;
    const scalar: usize = if (part == .weight) 4 else 2;
    const stride = try std.math.mul(usize, cols / div, scalar);
    if (v.shape[0] != rows or v.shape[1] != cols / div or v.bytes.len != try std.math.mul(usize, rows, stride)) return error.DensePocketShape;
    return .{ .stride = stride, .rows = rows };
}
pub fn base(a: std.mem.Allocator, d: usize, own: usize, extra: usize, r: Role, part: Part, v: View) ![]const u8 {
    if (extra == 0) return v.bytes;
    const n = try admit(d, own, extra, r, part, v);
    if (r != .down) return v.bytes[0 .. own * n.stride];
    const kept = own / (if (part == .weight) @as(usize, 8) else 64) * (if (part == .weight) @as(usize, 4) else 2);
    const out = try a.alloc(u8, n.rows * kept);
    for (0..n.rows) |row| @memcpy(out[row * kept ..][0..kept], v.bytes[row * n.stride ..][0..kept]);
    return out;
}
pub fn decode(d: usize, own: usize, extra: usize, r: Role, parts: [3]View, out: []f32) !void {
    if (out.len != extra * d) return error.DensePocketShape;
    for (parts, std.enums.values(Part)) |v, part| _ = try admit(d, own, extra, r, part, v);
    const q: @import("family.zig").Quant = .{ .bits = 4, .group = 64 };
    const w = std.mem.bytesAsSlice(u32, parts[0].bytes);
    const s = std.mem.bytesAsSlice(u16, parts[1].bytes);
    const b = std.mem.bytesAsSlice(u16, parts[2].bytes);
    if (r != .down) {
        const at = own * d;
        affine.dequantize(q, w[at / 8 ..], s[at / 64 ..], b[at / 64 ..], out);
    } else {
        var col: [64]f32 = undefined;
        for (0..d) |row| {
            const at = row * (own + extra) + own;
            affine.dequantize(q, w[at / 8 ..][0..8], s[at / 64 ..][0..1], b[at / 64 ..][0..1], &col);
            for (col, 0..) |v, j| out[j * d + row] = v;
        }
    }
    for (out) |v| if (!std.math.isFinite(v)) return error.DensePocketNonfinite;
}
