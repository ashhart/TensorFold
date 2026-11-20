//! Norm/RoPE views, validated before any direct cache or packed-query write.
const std = @import("std");
const at = @import("attention.zig");
const Ref = @import("projection.zig").Ref;
const views = @import("attention_views.zig");
const span = views.span;
const spanAligned = views.spanAligned;
const disjoint = views.disjoint;
const keyStride = views.keyStride;
const keyBytes = views.keyBytes;
pub fn check(p: at.Preprocess) !void {
    try p.geometry.check();
    const g = p.geometry;
    const q_bytes = @as(usize, p.rows) * g.heads * g.dim * 2;
    const kv_bytes = @as(usize, p.rows) * g.kv_heads * g.dim * 2;
    const gain_bytes = @as(usize, g.dim) * 2;
    if (p.rows == 0 or p.rows > @import("gpu_frame.zig").max_rows or !std.math.isFinite(p.eps) or p.eps <= 0 or !std.math.isFinite(p.theta) or p.theta <= 0) return error.TargetAttentionShape;
    if (keyStride(p) < g.kv_heads * g.dim or @as(u64, p.rows - 1) * keyStride(p) + g.kv_heads * g.dim > std.math.maxInt(u32)) return error.TargetAttentionShape;
    try span(p.qg, 2 * q_bytes);
    try span(p.mixed, keyBytes(p));
    try span(p.q_gain, gain_bytes);
    try span(p.k_gain, gain_bytes);
    try span(p.norm_q, q_bytes);
    try span(p.norm_k, kv_bytes);
    try span(p.query, q_bytes);
    try span(p.rotated_key, kv_bytes);
    if ((p.h_end > 0 or p.w_end > 0) and p.plane < p.rows) return error.TargetAttentionShape;
    const plane_rows: usize = if (p.h_end > 0 or p.w_end > 0) 2 * @as(usize, p.plane) + p.rows else p.rows;
    try spanAligned(p.positions, plane_rows * 4, 4);
    const outputs = [_]struct { ref: Ref, bytes: usize }{ .{ .ref = p.norm_q, .bytes = q_bytes }, .{ .ref = p.norm_k, .bytes = kv_bytes }, .{ .ref = p.query, .bytes = q_bytes }, .{ .ref = p.rotated_key, .bytes = kv_bytes } };
    const inputs = [_]struct { ref: Ref, bytes: usize }{ .{ .ref = p.qg, .bytes = 2 * q_bytes }, .{ .ref = p.mixed, .bytes = keyBytes(p) }, .{ .ref = p.q_gain, .bytes = gain_bytes }, .{ .ref = p.k_gain, .bytes = gain_bytes }, .{ .ref = p.positions, .bytes = plane_rows * 4 } };
    for (outputs, 0..) |dst, i| {
        for (inputs) |src| try disjoint(dst.ref, dst.bytes, src.ref, src.bytes);
        for (outputs[0..i]) |prior| try disjoint(dst.ref, dst.bytes, prior.ref, prior.bytes);
    }
}
/// The direct dispatch writes caches and QA, so these inputs and descriptors must not alias either destination.
pub fn checkCache(p: at.Preprocess, scratch: *const at.Scratch, binding: *const at.Binding, plan: at.Plan, caches: []const at.Cache, out: Ref) !void {
    try check(p);
    if (p.rows != plan.rows or !p.geometry.sameShape(plan.geometry) or p.key_row_stride != 0) return error.TargetAttentionShape;
    const g = p.geometry;
    const qb = @as(usize, p.rows) * g.heads * g.dim * 2;
    const kb = @as(usize, p.rows) * g.kv_heads * g.dim * 2;
    const planes: usize = if (p.h_end > 0 or p.w_end > 0) 2 * @as(usize, p.plane) + p.rows else p.rows;
    const operands = [_]struct { ref: Ref, bytes: usize }{
        .{ .ref = p.qg, .bytes = 2 * qb },            .{ .ref = p.mixed, .bytes = keyBytes(p) },
        .{ .ref = p.q_gain, .bytes = g.dim * 2 },     .{ .ref = p.k_gain, .bytes = g.dim * 2 },
        .{ .ref = p.positions, .bytes = planes * 4 }, .{ .ref = p.norm_q, .bytes = qb },
        .{ .ref = p.norm_k, .bytes = kb },            .{ .ref = p.query, .bytes = qb },
        .{ .ref = p.rotated_key, .bytes = kb },
    };
    const qa = scratch.get(.qa);
    const qab = @as(usize, plan.tiles) * 16 * g.kv_heads * g.dim * 2;
    try spanAligned(qa, qab, 16);
    for (operands) |r| {
        try disjoint(qa, qab, r.ref, r.bytes);
        for (caches) |cache| try views.apart(cache, g, r.ref, r.bytes);
    }
    for ([_]at.Field{ .nodes, .q_rows, .tile_stream, .paths }) |field| {
        const r = scratch.get(field);
        for (operands[5..]) |dst| try disjoint(r, r.buffer.length(), dst.ref, dst.bytes);
    }
    for (binding.buffers) |b| for (operands[5..]) |dst| try disjoint(.{ .buffer = b }, b.length(), dst.ref, dst.bytes);
    try disjoint(qa, qab, out, qb);
    try disjoint(p.mixed, keyBytes(p), out, qb);
    for (caches) |cache| {
        for (scratch.buffers) |b| try views.apart(cache, g, .{ .buffer = b }, b.length());
        for (binding.buffers) |b| try views.apart(cache, g, .{ .buffer = b }, b.length());
    }
}
