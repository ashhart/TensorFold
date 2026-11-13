//! Validated attention views and descriptor identity, without GPU work.
const std = @import("std");
const Ref = @import("projection.zig").Ref;
const Cache = @import("attention.zig").Cache;
const Plan = @import("attention_plan.zig").Plan;
const planning = @import("attention_plan.zig");

pub fn span(r: Ref, bytes: usize) !void {
    try spanAligned(r, bytes, 2);
}
pub fn spanAligned(r: Ref, bytes: usize, alignment: usize) !void {
    if (r.offset > r.buffer.length() or bytes > r.buffer.length() - r.offset or r.offset % alignment != 0) return error.AttentionBuffer;
}
pub fn disjoint(a: Ref, an: usize, b: Ref, bn: usize) !void {
    if (a.buffer.id == b.buffer.id and a.offset < b.offset + bn and b.offset < a.offset + an) return error.AttentionAlias;
}
pub fn fingerprint(plan: Plan) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    // Preserve the canonical 27B's descriptor hash; any new numeric shape gets its own identity.
    if (!plan.geometry.sameShape(.{})) {
        const shape = [_]u32{ plan.geometry.heads, plan.geometry.kv_heads, plan.geometry.dim, plan.geometry.rotary };
        hash.update(std.mem.sliceAsBytes(&shape));
    }
    hash.update(std.mem.asBytes(&plan.meta));
    inline for (.{ plan.tile_stream, plan.q_rows, plan.nodes, plan.paths, plan.positions }) |values| hash.update(std.mem.sliceAsBytes(values));
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return digest;
}
pub fn cacheBytes(cache: Cache, key: bool, geometry: planning.Geometry) usize {
    const requested = if (key) cache.key_stride else cache.value_stride;
    const stride: usize = if (requested == 0) @as(usize, cache.capacity) * geometry.dim else requested;
    return ((@as(usize, geometry.kv_heads) - 1) * stride + @as(usize, cache.capacity) * geometry.dim) * 2;
}
/// An operand outside `cache`'s rows; a paged cache's arena holds only pages, so an operand in it at all is an alias.
pub fn apart(cache: Cache, geometry: planning.Geometry, operand: Ref, bytes: usize) !void {
    if (cache.pages != null) {
        if (operand.buffer.id == cache.keys.buffer.id or operand.buffer.id == cache.values.buffer.id) return error.AttentionAlias;
        return;
    }
    try disjoint(cache.keys, cacheBytes(cache, true, geometry), operand, bytes);
    try disjoint(cache.values, cacheBytes(cache, false, geometry), operand, bytes);
}
