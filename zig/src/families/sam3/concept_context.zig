const std = @import("std");
const runtime = @import("runtime.zig");
const elementwise = @import("elementwise.zig");
const projection = @import("projection.zig");
const norms = @import("norms.zig");
const attention = @import("attention.zig");
const encodings = @import("encodings.zig");

const Runtime = runtime.Runtime;
const Scope = runtime.Scope;
const Tensor = runtime.Tensor;
const FeatureMap = runtime.FeatureMap;

/// Every detector attention: 8 heads of width 32.
pub const heads = 8;
pub const class_layers = 3;
pub const grounding_layers = 6;
pub const norm_eps: f32 = 1e-5;

/// Grid-ordered coarse image tokens, their sine code, and the cue tokens (text rows, then the geometry class token).
pub const Context = struct { image: Tensor, grid_code: Tensor, cues: Tensor, cue_rows: usize };

/// Plain-route layer norm with the detector epsilon.
pub fn layerNorm(s: Scope, leaf: []const u8, x: Tensor) !Tensor {
    return norms.layerNorm(s, leaf, x, norm_eps, .plain);
}

/// Opens a batch when none is open; returns whether this call opened it.
pub fn openBatch(rt: *Runtime) !bool {
    if (rt.gpu.batch != null) return false;
    try rt.gpu.beginBatch();
    return true;
}

/// Closes a batch this call opened, also on error paths.
pub fn closeBatch(rt: *Runtime, opened: bool) void {
    if (opened and rt.gpu.batch != null) rt.gpu.endBatch() catch {};
}

/// Must start with no batch open: the sine-grid constant is built on the host.
pub fn build(rt: *Runtime, coarse: FeatureMap, text_features: Tensor) !Context {
    const c = rt.config;
    const side = c.featureSide();
    const width = c.hidden;
    if (coarse.height != side or coarse.width != side or coarse.channels != width or
        coarse.tensor.count != side * side * width) return error.SamConceptShape;
    if (text_features.count == 0 or text_features.count % width != 0) return error.SamConceptShape;
    const grid_code = try encodings.sineGrid(rt, side, side, width);
    const m = rt.mark();
    errdefer rt.rewind(m, &.{});
    const opened = try openBatch(rt);
    errdefer closeBatch(rt, opened);
    const class_token = try classToken(rt, coarse.tensor, grid_code);
    const cues = try elementwise.concat(rt, &.{ text_features, class_token });
    if (opened) try rt.gpu.endBatch();
    rt.rewind(m, &.{cues});
    return .{ .image = coarse.tensor, .grid_code = grid_code, .cues = cues, .cue_rows = text_features.count / width + 1 };
}

fn classToken(rt: *Runtime, image: Tensor, grid_code: Tensor) !Tensor {
    const geometry = try Scope.root(rt, rt.config.roots.detector).sub("geometry_encoder");
    const embed = try geometry.param("cls_embed.weight");
    if (embed.count != rt.config.hidden) return error.SamConceptShape;
    var token = try layerNorm(geometry, "norm", try projection.affine(geometry, "final_proj", embed));
    const image_keys = try elementwise.add(rt, image, grid_code);
    for (0..class_layers) |j| {
        const layer = try geometry.item("encode", j);
        var a = try layerNorm(layer, "norm1", token);
        token = try elementwise.add(rt, token, try attention.packedHeads(try layer.sub("self_attn"), a, a, a, heads, .{}));
        a = try layerNorm(layer, "norm2", token);
        const looked = try attention.packedHeads(try layer.sub("cross_attn_image"), a, image_keys, image, heads, .{});
        token = try elementwise.add(rt, token, looked);
        a = try layerNorm(layer, "norm3", token);
        token = try elementwise.add(rt, token, try projection.expandContract(layer, a, .relu));
    }
    return layerNorm(geometry, "encode_norm", token);
}

/// Six fusion layers grounding the image tokens in the cue tokens → `[G², hidden]` in grid order.
pub fn ground(rt: *Runtime, context: Context) !Tensor {
    const encoder = try Scope.root(rt, rt.config.roots.detector).sub("transformer.encoder");
    const m = rt.mark();
    errdefer rt.rewind(m, &.{});
    const opened = try openBatch(rt);
    errdefer closeBatch(rt, opened);
    var x = context.image;
    for (0..grounding_layers) |j| {
        const layer = try encoder.item("layers", j);
        var a = try layerNorm(layer, "norm1", x);
        const placed = try elementwise.add(rt, a, context.grid_code);
        x = try elementwise.add(rt, x, try attention.packedHeads(try layer.sub("self_attn"), placed, placed, a, heads, .{}));
        a = try layerNorm(layer, "norm2", x);
        const cued = try attention.packedHeads(try layer.sub("cross_attn_image"), a, context.cues, context.cues, heads, .{});
        x = try elementwise.add(rt, x, cued);
        a = try layerNorm(layer, "norm3", x);
        x = try elementwise.add(rt, x, try projection.expandContract(layer, a, .relu));
        rt.rewind(m, &.{x});
    }
    if (opened) try rt.gpu.endBatch();
    return x;
}
