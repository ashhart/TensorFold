//! The family adapter transfers the reviewed core block repacker's owned native affine rows without narrowing.
const std = @import("std");
const gg = @import("../../core/gguf.zig");
pub const source = @embedFile("gguf_repack.zig") ++ @embedFile("../../core/gguf_repack.zig") ++
    @embedFile("../../core/gguf_repack_blocks.zig");
pub const Format = gg.affine.Format;
pub fn format(kind: gg.TensorKind) !Format {
    const f = try gg.affine.format(kind);
    if (!((f.group == 32 and (f.bits == 4 or f.bits == 5 or f.bits == 8)) or
        (f.group == 16 and f.bits == 6))) return error.UnqualifiedGgufAffine;
    return f;
}
pub const Affine = struct {
    n: usize,
    k: usize,
    bits: u8,
    group: usize,
    words: []u32,
    scales: []f32,
    biases: []f32,
    pub fn deinit(p: Affine, a: std.mem.Allocator) void {
        a.free(p.words);
        a.free(p.scales);
        a.free(p.biases);
    }
};
pub const Repacker = struct {
    identity: []const u8 = source,
    context: ?*anyopaque = null,
    run: *const fn (?*anyopaque, std.mem.Allocator, gg.Tensor) anyerror!Affine = native,
};
fn native(_: ?*anyopaque, a: std.mem.Allocator, t: gg.Tensor) !Affine {
    const f = try format(t.kind);
    if (t.rank != 2 or t.shape[0] == 0 or t.shape[1] == 0) return error.BadGgufProjection;
    const k = std.math.cast(usize, t.shape[0]) orelse return error.BadGgufProjection;
    const n = std.math.cast(usize, t.shape[1]) orelse return error.BadGgufProjection;
    const block = t.kind.block();
    if (k % block.values != 0) return error.BadGgufProjection;
    const count = try std.math.mul(usize, n, k);
    if (t.bytes.len != try std.math.mul(usize, count / block.values, block.bytes)) return error.BadGgufProjection;
    var p = try gg.affine.repack(a, t.kind, t.bytes);
    errdefer p.deinit();
    if (p.shape.elements != count or p.shape.format.bits != f.bits or p.shape.format.group != f.group) return error.BadGgufRepackResult;
    return .{ .n = n, .k = k, .bits = f.bits, .group = f.group, .words = p.words, .scales = p.scales, .biases = p.biases };
}
