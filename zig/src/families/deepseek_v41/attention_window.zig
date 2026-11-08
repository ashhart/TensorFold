//! The attention pass's window stage, outside the per-stage struct.
const std = @import("std");
const mtl = @import("metal");
const exl3 = @import("core").exl3_projection;
const checkpoint = @import("core").checkpoint_metal;
const weights = @import("weights.zig");
const config = @import("config.zig");
const ops = @import("ops.zig");
const csa2 = @import("csa2.zig");
const kv = @import("kv.zig");
const index = @import("csa2_index.zig");
const rope = @import("rope.zig");
const common = @import("attention_common.zig");
const stages = @import("attention_stages.zig");
const Stage = stages.Stage;
const View = common.View;
const State = common.State;
const Seg = common.Seg;
const Prof = common.Prof;
const profLayerOn = common.profLayerOn;
const DiagnosticStage = common.DiagnosticStage;
const DiagnosticObserver = common.DiagnosticObserver;
const Pipelines = common.Pipelines;
const regions = common.regions;
const Region = common.Region;
const Workspace = common.Workspace;
const tensor = common.tensor;
const core = common.core;
const projection = common.projection;
const norm = common.norm;
const normRepeat = common.normRepeat;
const rotate = common.rotate;
const rotateRepeat = common.rotateRepeat;
const attnCountOn = common.attnCountOn;
const attnBegin = common.attnBegin;
const attnMark = common.attnMark;
const precedingIndex = common.precedingIndex;
const partialBytes = common.partialBytes;
const denseKvBytes = common.denseKvBytes;
const splitScan = common.splitScan;

pub fn windowStage(self: *Stage, comptime diagnostic: bool) !void {
    const p = self.p;
    const w = self.w;
    const ep = self.ep;
    const c = self.c;
    const layer = self.layer;
    const rows = self.rows;
    const scratch = self.scratch;
    const x = self.x;
    const position = self.position;
    const ratio = self.ratio;
    const chunk = self.chunk;
    const dense_passes = self.dense_passes;
    const dup_view = self.dup_view;
    const prof = self.prof;
    var e = self.e;
    defer self.e = e;
    const raw = self.raw;
    const qkv_merged = self.qkv_merged;
    if (!qkv_merged) for (0..dense_passes) |_| try projection(ep, e, .{ .trellis = w.wkv }, x, raw, scratch, rows, c.hidden, c.head_dim, .bf16);
    attnMark(layer, "wkv", prof, &e);
    if (diagnostic) try DiagnosticObserver.observe(true, self.observer, e, layer, .kv_projection, raw);
    // The window KV's norm + rope + fp8 pack as ONE dispatch, at the pack's own threadgroup geometry.
    const fuse_kv_glue = rows <= 8 and w.kv_norm.dtype == .bf16 and c.head_dim == 512 and !diagnostic and
        std.c.getenv("TF_DSV41_NO_FUSE_KVGLUE") == null and
        std.c.getenv("TF_DSV41_DOUBLE_ATTN_NORM") == null and
        std.c.getenv("TF_DSV41_DOUBLE_ROPE") == null and
        std.c.getenv("TF_DSV41_DOUBLE_ATTN_PACK") == null;
    if (fuse_kv_glue) {
        const factor_offset: usize = if (ratio == 0) 0 else @as(usize, p.rotary.capacity) * p.rotary.rotary * @sizeOf(f32);
        const factors: View = .{ .buffer = p.rotary.factors, .offset = factor_offset };
        try p.cache.encodeGlue(e, raw, tensor(w.kv_norm), factors, chunk, chunk, .{
            .rows = rows,
            .width = c.head_dim,
            .stride = c.head_dim,
            .packed_stride = kv.cache.ring_stride,
            .block = 32,
            .position = position,
            .rotary = c.rope_dim,
            .write_y = 0,
            .eps = c.eps,
        });
        attnMark(layer, "kv_glue", prof, &e);
    } else {
        norm(p, e, raw, w.kv_norm, rows, c.head_dim, c.eps);
        if (rows == 1 and std.c.getenv("TF_DSV41_DOUBLE_ATTN_NORM") != null) normRepeat(p, e, raw, dup_view, w.kv_norm, rows, c.head_dim, c.eps);
        attnMark(layer, "norm_kv", prof, &e);
        if (diagnostic) try DiagnosticObserver.observe(true, self.observer, e, layer, .kv_norm, raw);
        try rotate(p, e, raw, rows, 1, c.head_dim, position, ratio, 1, false, c);
        if (rows == 1 and std.c.getenv("TF_DSV41_DOUBLE_ROPE") != null) try rotateRepeat(p, e, dup_view, rows, 1, c.head_dim, position, ratio, 1, false, c);
        attnMark(layer, "kv_rope", prof, &e);
        if (diagnostic) try DiagnosticObserver.observe(true, self.observer, e, layer, .kv_rope, raw);
        try p.cache.encodePack(e, raw, chunk, .{ .rows = rows, .width = c.head_dim, .stride = c.head_dim, .packed_stride = kv.cache.ring_stride, .block = 32, .kind = .window_fp8 });
        // TF_DSV41_DOUBLE_ATTN_PACK -- the window's fp8 cache pack's same-output duplicate.
        if (rows == 1 and std.c.getenv("TF_DSV41_DOUBLE_ATTN_PACK") != null)
            try p.cache.encodePack(e, raw, chunk, .{ .rows = rows, .width = c.head_dim, .stride = c.head_dim, .packed_stride = kv.cache.ring_stride, .block = 32, .kind = .window_fp8 });
        attnMark(layer, "pack", prof, &e);
        if (diagnostic) try DiagnosticObserver.observe(true, self.observer, e, layer, .kv_quantized, chunk);
    }
}
