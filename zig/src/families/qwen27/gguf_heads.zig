//! Invert GGUF's key-fast value-head order without changing any codes or affine metadata.
const std = @import("std");
const Affine = @import("gguf_repack.zig").Affine;
pub const Axis = enum { rows, columns };
pub const Order = struct {
    axis: Axis,
    key_heads: usize,
    value_heads: usize,
    width: usize,
    offset: usize = 0,

    /// GGUF head j is native head (j % key_heads) * ratio + j / key_heads.
    pub fn sourceHead(o: Order, native_head: usize) usize {
        const ratio = o.value_heads / o.key_heads;
        return (native_head % ratio) * o.key_heads + native_head / ratio;
    }
    fn validate(o: Order, extent: usize) !void {
        if (o.key_heads == 0 or o.value_heads == 0 or o.value_heads % o.key_heads != 0 or o.width == 0)
            return error.BadGgufHeadOrder;
        const count = try std.math.mul(usize, o.value_heads, o.width);
        if (o.offset > extent or count > extent - o.offset) return error.BadGgufHeadOrder;
    }
    fn scaled(o: Order, numerator: usize, denominator: usize) !Order {
        if (o.axis == .rows) return o;
        const width = try std.math.mul(usize, o.width, numerator);
        const offset = try std.math.mul(usize, o.offset, numerator);
        if (width % denominator != 0 or offset % denominator != 0) return error.UnalignedGgufHeadOrder;
        var result = o;
        result.width = width / denominator;
        result.offset = offset / denominator;
        return result;
    }
};
/// Allocate the complete result before changing the caller's data; prefix and suffix stay byte-identical.
pub fn matrix(comptime T: type, a: std.mem.Allocator, src: []const T, n: usize, k: usize, o: Order) ![]T {
    if (src.len != try std.math.mul(usize, n, k)) return error.BadGgufHeadOrder;
    try o.validate(if (o.axis == .rows) n else k);
    const dst = try a.dupe(T, src);
    if (o.axis == .rows) {
        for (0..o.value_heads) |head| {
            const to = (o.offset + head * o.width) * k;
            const from = (o.offset + o.sourceHead(head) * o.width) * k;
            @memcpy(dst[to..][0 .. o.width * k], src[from..][0 .. o.width * k]);
        }
    } else {
        for (0..n) |row| {
            for (0..o.value_heads) |head| {
                const to = row * k + o.offset + head * o.width;
                const from = row * k + o.offset + o.sourceHead(head) * o.width;
                @memcpy(dst[to..][0..o.width], src[from..][0..o.width]);
            }
        }
    }
    return dst;
}
/// Whole-head spans must align to packed words and scale groups; no dequantization or requantization is used.
pub fn affine(a: std.mem.Allocator, p: *Affine, o: Order) !void {
    if (p.group == 0 or p.k % p.group != 0 or p.k % 32 != 0) return error.UnalignedGgufHeadOrder;
    const words = try matrix(u32, a, p.words, p.n, p.k / 32 * p.bits, try o.scaled(p.bits, 32));
    errdefer a.free(words);
    const scales = try matrix(f32, a, p.scales, p.n, p.k / p.group, try o.scaled(1, p.group));
    errdefer a.free(scales);
    const biases = try matrix(f32, a, p.biases, p.n, p.k / p.group, try o.scaled(1, p.group));
    a.free(p.words);
    a.free(p.scales);
    a.free(p.biases);
    p.words = words;
    p.scales = scales;
    p.biases = biases;
}
