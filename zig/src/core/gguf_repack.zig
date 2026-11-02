//! Compact affine codes with FP32 metadata, written from the GGUF block-layout definitions for the ten plain quants.
const std = @import("std");
const blocks = @import("gguf_repack_blocks.zig");
pub const Kind = @import("gguf_tensor.zig").Kind;
pub const Format = struct { bits: u8, group: usize };
pub fn format(kind: Kind) !Format {
    return switch (kind) {
        .q8_0 => .{ .bits = 8, .group = 32 },
        .q4_0, .q4_1, .q4_k => .{ .bits = 4, .group = 32 },
        .q5_0, .q5_1, .q5_k => .{ .bits = 5, .group = 32 },
        .q6_k => .{ .bits = 6, .group = 16 },
        .q3_k => .{ .bits = 3, .group = 16 },
        .q2_k => .{ .bits = 2, .group = 16 },
        else => error.UnsupportedRepackType,
    };
}
pub const Layout = struct {
    format: Format,
    elements: usize,
    groups: usize,
    words: usize,
};
pub fn layout(kind: Kind, bytes: usize) !Layout {
    const f = try format(kind);
    const block = kind.block();
    if (bytes == 0 or bytes % block.bytes != 0) return error.InvalidBlockLength;
    const n = std.math.mul(usize, bytes / block.bytes, block.values) catch return error.RepackSizeOverflow;
    const words = std.math.mul(usize, n / 32, f.bits) catch return error.RepackSizeOverflow;
    return .{ .format = f, .elements = n, .groups = n / f.group, .words = words };
}
pub const Affine = struct {
    gpa: std.mem.Allocator,
    shape: Layout,
    /// Logical element i occupies bits [i*bits, (i+1)*bits), low bit first; group boundaries need not be words.
    words: []u32,
    scales: []f32,
    biases: []f32,

    pub fn deinit(a: *Affine) void {
        a.gpa.free(a.biases);
        a.gpa.free(a.scales);
        a.gpa.free(a.words);
        a.* = undefined;
    }
};
/// Finite-block repack with owned storage and no narrowing; affine cancellation can change the sign of zero.
pub fn repack(gpa: std.mem.Allocator, kind: Kind, src: []const u8) !Affine {
    const l = try layout(kind, src.len);
    try blocks.validate(kind, src);
    const words = try gpa.alloc(u32, l.words);
    errdefer gpa.free(words);
    const scales = try gpa.alloc(f32, l.groups);
    errdefer gpa.free(scales);
    const biases = try gpa.alloc(f32, l.groups);
    errdefer gpa.free(biases);
    blocks.write(kind, src, l.format.bits, l.format.group, words, scales, biases);
    return .{ .gpa = gpa, .shape = l, .words = words, .scales = scales, .biases = biases };
}
fn overlap(a: []const u8, b: []const u8) bool {
    if (a.len == 0 or b.len == 0) return false;
    const x = @intFromPtr(a.ptr);
    const y = @intFromPtr(b.ptr);
    return if (x <= y) y - x < a.len else x - y < b.len;
}
/// All admission checks finish before any write. Inputs and all three outputs must be disjoint.
pub fn into(kind: Kind, src: []const u8, words: []u32, scales: []f32, biases: []f32) !void {
    const l = try layout(kind, src.len);
    if (words.len != l.words or scales.len != l.groups or biases.len != l.groups) return error.InvalidRepackStorage;
    const parts = [_][]const u8{ src, std.mem.sliceAsBytes(words), std.mem.sliceAsBytes(scales), std.mem.sliceAsBytes(biases) };
    for (parts, 0..) |a, i| for (parts[i + 1 ..]) |b| {
        if (overlap(a, b)) return error.OverlappingRepackStorage;
    };
    try blocks.validate(kind, src);
    blocks.write(kind, src, l.format.bits, l.format.group, words, scales, biases);
}

test {
    _ = @import("gguf_repack_test.zig");
}
