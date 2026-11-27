//! A DSA layer's load (weights.load's dsa): GLM-5.3-Flash's gated pooled indexer, full GLM-5.3's roped or shared one.
const std = @import("std");
const cuda = @import("cuda");
const formats = @import("cuda_formats.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const sparse = @import("cuda_sparse.zig");
const shard = @import("cuda_shard.zig");
const mm = @import("cuda_mm.zig");

/// weights.load's dsa(i) (latent): [q_a | kv_a], norms, q_b by heads, kv_b as wk/wv (AbsorbW), o by cols, indexer.
pub fn dsa(L: anytype, i: u32) !dsa_block.Layer {
    const c = &L.w.config;
    const h = L.w.share.mla;
    const qk = c.nope;
    // the MTP layer's may be MXFP8 (FP4-Local's): on the lane as stored, kv_b widened exactly for its halves
    const mtp = i == c.layers;
    const wants: []const formats.Format = if (mtp) &.{ .bf16, .fp8_pb, .mxfp8 } else &.{ .bf16, .fp8_pb };
    var kb: [96]u8 = undefined;
    const kv_name = try std.fmt.bufPrint(&kb, "{s}layers.{d}.self_attn.kv_b_proj", .{ L.prefix, i });
    const kv_mx = mtp and (try L.get("layers.{d}.self_attn.kv_b_proj.weight", .{i})).dtype == .f8_e4m3;
    try L.format("layers.{d}.self_attn.kv_b_proj", .{i}, if (kv_mx) .mxfp8 else .bf16);
    const kv_wide: ?formats.mxfp8.Mxfp8 = if (kv_mx) try formats.mxfp8.Mxfp8.read(L.ck, kv_name) else null;
    const columns = L.w.share.dsa_columns and !mtp;
    const ps: ?shard.Span = if (columns) try shard.Share.outputColumns(c.q_lora + c.kv_lora, L.w.share.rank, L.w.share.world) else null;
    var proj_parts: [2]formats.Part = undefined;
    const proj_count = projectionParts(ps, c.q_lora, &proj_parts);
    if (ps != null) {
        try L.format("layers.{d}.self_attn.q_a_proj", .{i}, .bf16);
        try L.format("layers.{d}.self_attn.kv_a_proj_with_mqa", .{i}, .bf16);
    }
    const proj = try formats.cutProj(L, L.prefix, i, proj_parts[0..proj_count], if (columns) &.{.bf16} else wants, "layers.{d}.dsa.proj");
    try L.whole(try L.get("layers.{d}.self_attn.q_a_layernorm.weight", .{i}), .bf16);
    const q_norm = try L.put("layers.{d}.dsa.q_norm", .{i});
    try L.whole(try L.get("layers.{d}.self_attn.kv_a_layernorm.weight", .{i}), .bf16);
    const kv_norm = try L.put("layers.{d}.dsa.kv_norm", .{i});
    const qh = qk + c.rope; // a q head's rows: its latent-path dims, then its roped ones
    const q_b = try formats.cutProj(L, L.prefix, i, &.{.{ .rel = "self_attn.q_b_proj", .rows = .{ h.lo * qh, h.hi * qh } }}, wants, "layers.{d}.dsa.q_b");
    const kv_b = try L.get("layers.{d}.self_attn.kv_b_proj.weight", .{i});
    const per = qk + c.v_dim; // a head's kv_b rows: its keys, then its values
    const total = c.mla_heads * per;
    for (h.lo..h.hi) |head| try absorbRows(L, kv_wide, kv_b, kv_name, head * per, head * per + qk, total);
    const wk = try L.put("layers.{d}.dsa.absorb.wk", .{i});
    for (h.lo..h.hi) |head| try absorbRows(L, kv_wide, kv_b, kv_name, head * per + qk, (head + 1) * per, total);
    const wv = try L.put("layers.{d}.dsa.absorb.wv", .{i});
    const o = try formats.cutProj(L, L.prefix, i, &.{.{ .rel = "self_attn.o_proj", .cols = .{ h.lo * c.v_dim, h.hi * c.v_dim } }}, wants, "layers.{d}.dsa.o");
    const inv = try ropeInv(L);
    return .{
        .proj = proj,
        .proj_columns = ps,
        .q_norm = q_norm,
        .kv_norm = kv_norm,
        .q_b = q_b,
        .wk = wk,
        .wv = wv,
        .o = o,
        .index = if (c.shared.isSet(i)) shared(c) else try index(L, i, mtp, inv),
        .heads = h.len(),
        .qk = qk,
        .dv = c.v_dim,
        .lw = c.kv_lora,
        .rope = c.rope,
        .rope_inv = inv,
        .q_lora = c.q_lora,
        .dims = c.hidden,
        .eps = c.eps,
        .dense_limit = c.keyWidth(),
    };
}
/// The stored row cuts for one rank's `[q_a | kv_a]` outputs; null keeps the replicated pair.
fn projectionParts(ps: ?shard.Span, q_lora: usize, parts: *[2]formats.Part) usize {
    const sp = ps orelse {
        parts.* = .{ .{ .rel = "self_attn.q_a_proj" }, .{ .rel = "self_attn.kv_a_proj_with_mqa" } };
        return 2;
    };
    var count: usize = 0;
    if (sp.lo < q_lora) {
        parts[count] = .{ .rel = "self_attn.q_a_proj", .rows = .{ sp.lo, @min(sp.hi, q_lora) } };
        count += 1;
    }
    if (sp.hi > q_lora) {
        parts[count] = .{ .rel = "self_attn.kv_a_proj_with_mqa", .rows = .{ @max(sp.lo, q_lora) - q_lora, sp.hi - q_lora } };
        count += 1;
    }
    return count;
}

/// The layer's own indexer: [wk | weights_proj], wq_b and its key LayerNorm; GLM-5.3-Flash's pool gate and ape too.
fn index(L: anytype, i: u32, mtp: bool, inv: u64) !dsa_block.Index {
    const c = &L.w.config;
    const iw: []const formats.Format = if (mtp) &.{ .bf16, .mxfp8 } else &.{.bf16};
    const kw = try formats.cutProj(L, L.prefix, i, &.{ .{ .rel = "self_attn.indexer.wk" }, .{ .rel = "self_attn.indexer.weights_proj" } }, iw, "layers.{d}.dsa.index.kw");
    const qs: ?shard.Span = if (L.w.share.dsa_columns and !mtp) try shard.Share.outputColumns(c.i_heads * c.i_dim, L.w.share.rank, L.w.share.world) else null;
    if (qs != null) try L.format("layers.{d}.self_attn.indexer.wq_b", .{i}, .bf16);
    const qb = try formats.cutProj(L, L.prefix, i, &.{.{ .rel = "self_attn.indexer.wq_b", .rows = if (qs) |sp| .{ sp.lo, sp.hi } else null }}, if (qs != null) &.{.bf16} else iw, "layers.{d}.dsa.index.qb");
    try L.whole(try L.get("layers.{d}.self_attn.indexer.k_norm.weight", .{i}), .bf16);
    const ln_w = try L.put("layers.{d}.dsa.index.ln_w", .{i});
    try L.whole(try L.get("layers.{d}.self_attn.indexer.k_norm.bias", .{i}), .bf16);
    const ln_b = try L.put("layers.{d}.dsa.index.ln_b", .{i});
    var gate: u64 = 0;
    var ape: u64 = 0;
    if (c.kpool != 1) { // pools of 4 sum their keys by a softmax of these gates
        try L.whole(try L.get("layers.{d}.self_attn.indexer.index_kpool_compress_gate", .{i}), .bf16);
        gate = try L.put("layers.{d}.dsa.index.gate", .{i});
        try L.whole(try L.get("layers.{d}.self_attn.indexer.index_kpool_compress_ape", .{i}), .bf16);
        ape = try L.put("layers.{d}.dsa.index.ape", .{i});
    }
    const g: sparse.Geometry = .{ .pool = c.kpool, .topk = c.i_topk / c.kpool, .rope = if (c.dsa) c.rope else 0, .inv = if (c.dsa) inv else 0 };
    var result: dsa_block.Index = .{ .kw = kw, .gate = gate, .ln_w = ln_w, .ln_b = ln_b, .ape = ape, .qb = qb, .heads = c.i_heads, .d = c.i_dim, .geom = g, .apart = c.dsa };
    result.qb_columns = qs;
    return result;
}

/// A shared layer's indexer: none of its own; it attends the token lists its indexed layer left (Config.indexFrom).
fn shared(c: anytype) dsa_block.Index {
    const g: sparse.Geometry = .{ .pool = c.kpool, .topk = c.i_topk / c.kpool };
    return .{ .kw = .{ .bf16 = 0 }, .gate = 0, .ln_w = 0, .ln_b = 0, .ape = 0, .qb = .{ .bf16 = 0 }, .heads = c.i_heads, .d = c.i_dim, .geom = g, .own = false, .apart = true };
}

/// The roped dims' fp32 frequencies (rope_theta), uploaded once for every layer; 0 on GLM-5.3-Flash.
fn ropeInv(L: anytype) !u64 {
    const c = &L.w.config;
    if (c.rope == 0) return 0;
    if (L.w.rope_inv != 0) return L.w.rope_inv;
    if (c.rope != cuda.mla.rope_width) return error.UnexpectedTensor;
    const f = cuda.mla.ropeFreqs(c.rope_theta);
    try L.host.appendSlice(L.gpa, std.mem.sliceAsBytes(&f));
    L.w.rope_inv = try L.put("rope_inv", .{});
    return L.w.rope_inv;
}

/// kv_b's rows [lo, hi) onto the host bytes: bf16 as stored, or MXFP8 (`mx`, the MTP layer's) widened exactly.
fn absorbRows(L: anytype, mx: ?formats.mxfp8.Mxfp8, t: anytype, name: []const u8, lo: usize, hi: usize, total: usize) !void {
    if (mx) |p| return formats.mxfp8.appendWide(L, p, name, lo, hi);
    try L.rows(t, lo, hi, total, .bf16);
}

test "GLM DSA columns: loader rows cover q_a and kv_a exactly once on each rank" {
    var parts: [2]formats.Part = undefined;
    try std.testing.expectEqual(@as(usize, 1), projectionParts(.{ .lo = 0, .hi = 1024 }, 1536, &parts));
    try std.testing.expectEqual(@as(?[2]usize, .{ 0, 1024 }), parts[0].rows);
    try std.testing.expectEqualStrings("self_attn.q_a_proj", parts[0].rel);
    try std.testing.expectEqual(@as(usize, 2), projectionParts(.{ .lo = 1024, .hi = 2048 }, 1536, &parts));
    try std.testing.expectEqual(@as(?[2]usize, .{ 1024, 1536 }), parts[0].rows);
    try std.testing.expectEqual(@as(?[2]usize, .{ 0, 512 }), parts[1].rows);
    try std.testing.expectEqualStrings("self_attn.kv_a_proj_with_mqa", parts[1].rel);
    try std.testing.expectEqual(@as(usize, 2), projectionParts(null, 1536, &parts));
    try std.testing.expectEqual(@as(?[2]usize, null), parts[0].rows);
    try std.testing.expectEqual(@as(?[2]usize, null), parts[1].rows);
}

test "GLM DSA columns: full and half outputs keep the same glm_mm K slices" {
    try std.testing.expectEqual(@as(usize, 8), mm.splitK(2048, 4096));
    try std.testing.expectEqual(mm.splitK(2048, 4096), mm.splitK(1024, 4096));
    try std.testing.expectEqual(@as(usize, 4), mm.splitK(4096, 1536));
    try std.testing.expectEqual(mm.splitK(4096, 1536), mm.splitK(2048, 1536));
}
