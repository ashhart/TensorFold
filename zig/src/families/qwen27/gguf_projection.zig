//! Exact GGUF affine storage for the existing weight-only projection, with six-bit codes widened losslessly.
const std = @import("std");
const core = @import("core");
pub const layout = @import("gguf_layout.zig");
pub const adapter = @import("gguf.zig");
const wo = core.weight_only;
pub const Matrix = struct {
    params: wo.Params,
    words: []const u8,
    scales: []const u8,
    biases: []const u8,
};
pub fn admit(doc: *const core.gguf.Document, p: *adapter.Problem) !void {
    const tag = doc.get("general.file_type") orelse return p.fail("general.file_type: missing", .{});
    const kind = tag.unsigned() catch return p.fail("general.file_type: expected integer", .{});
    if (kind != 15 and kind != 7) return p.fail("general.file_type: unsupported {d} ({s} tensors); expected Q4_K_M (15) or Q8_0 (7)", .{ kind, if (doc.tensors.len == 0) "no" else @tagName(doc.tensors[0].kind) });
    for (doc.tensors) |t| switch (t.kind) {
        .f32 => {},
        .q4_k, .q6_k => if (kind != 15) return p.fail("{s}: {s} is unsupported in Q8_0", .{ t.name, @tagName(t.kind) }),
        .q8_0 => {},
        else => return p.fail("{s}: unsupported GGUF serving quantization {s}", .{ t.name, @tagName(t.kind) }),
    };
}
pub fn code(words: []const u32, bits: usize, i: usize) u8 {
    const at = i * bits;
    const shift: u5 = @intCast(at % 32);
    var v = words[at / 32] >> shift;
    if (at % 32 + bits > 32) v |= words[at / 32 + 1] << @as(u5, @intCast(32 - at % 32));
    return @intCast(v & ((@as(u32, 1) << @as(u5, @intCast(bits))) - 1));
}
/// Returned original arrays borrow the adapter; widened codes belong to the caller's arena.
pub fn prepare(a: std.mem.Allocator, c: @import("config.zig").Config, t: adapter.Tensor, p: *adapter.Problem) !Matrix {
    if (t.name.role != .matrix or t.data != .affine) return p.fail("{s}: unsupported GGUF serving quantization {s}; expected an affine matrix", .{ t.source.name, @tagName(t.source.kind) });
    const q = t.data.affine;
    const format: wo.Format = switch (t.source.kind) {
        .q4_k => .mlx4,
        .q6_k, .q8_0 => .mlx8,
        else => return p.fail("{s}: unsupported GGUF serving quantization {s}", .{ t.source.name, @tagName(t.source.kind) }),
    };
    const bytes = if (q.shape.format.bits == 6) blk: {
        const out = try a.alloc(u8, q.shape.elements);
        for (out, 0..) |*v, i| v.* = code(q.words, 6, i);
        break :blk out;
    } else std.mem.sliceAsBytes(q.words);
    const params: wo.Params = .{ .rows = 1, .n = @intCast(t.name.dims[0]), .k = @intCast(t.name.dims[1]), .format = format, .group = @intCast(q.shape.format.group), .scale_kind = .f32 };
    _ = try wo.validateMetal(params);
    const arrays = try layout.affine(a, c, layout.order(t.name.native), @intCast(params.n), @intCast(params.k), if (format == .mlx4) 4 else 8, @intCast(params.group), .{ .words = bytes, .scales = std.mem.sliceAsBytes(q.scales), .biases = std.mem.sliceAsBytes(q.biases) });
    return .{ .params = params, .words = arrays.words, .scales = arrays.scales, .biases = arrays.biases };
}
