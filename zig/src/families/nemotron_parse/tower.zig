//! C-RADIOv4-SO400M and Parse's horizontal convolution neck on native HIP kernels.
const std = @import("std");
const hip = @import("hip");
const config = @import("config.zig");
const names = @import("schema.zig");
const Geometry = @import("processor.zig").Geometry;
const Compute = @import("compute.zig").Compute;
const Tensor = hip.ops.Tensor;

const width = config.vision_hidden;
const heads = config.vision_heads;
const head_width = width / heads;
comptime {
    std.debug.assert(head_width * heads == width);
}

/// "<stem>.<name>" in `buf`, valid until the next call; the helpers append ".weight" or ".bias".
fn sub(buf: []u8, stem: []const u8, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.{s}", .{ stem, name });
}

/// One pre-norm ViT layer (arXiv 2010.11929, Eq. 2-3), unmasked, biased projections, erf GELU.
fn block(c: Compute, x: Tensor, stem: []const u8, rows: usize) !void {
    var buf: [224]u8 = undefined;

    // Eq. 2: attention sub-layer.
    const a = try c.norm(x, try sub(&buf, stem, "norm1"), rows, width, config.vision_eps);
    const qkv = try c.linear(a, try sub(&buf, stem, "attn.qkv"), rows, 3 * width, width, true);
    const mixed = try c.global(qkv, rows, heads, head_width);
    const attended = try c.linear(mixed, try sub(&buf, stem, "attn.proj"), rows, width, width, true);
    try c.residual(x, attended, rows * width);

    // Eq. 3: feed-forward sub-layer.
    const b = try c.norm(x, try sub(&buf, stem, "norm2"), rows, width, config.vision_eps);
    const up = try c.linear(b, try sub(&buf, stem, "mlp.fc1"), rows, config.vision_ffn, width, true);
    try c.gelu(up, rows * config.vision_ffn);
    const down = try c.linear(up, try sub(&buf, stem, "mlp.fc2"), rows, width, config.vision_ffn, true);
    try c.residual(x, down, rows * width);
}

/// `out` holds the image's visual prefix [geometry.tokens(),1024] and survives scratch reuse.
pub fn encode(c: Compute, patches: Tensor, geometry: Geometry, out: Tensor) !void {
    const count = geometry.patches();
    const rows = count + config.vision_prefix;
    const hidden = config.vision_hidden;
    const x = try c.alloc(rows * hidden, .bf16);
    const mark = c.ops.arena.mark();
    try positioned(c, patches, geometry, x);
    try c.record("encoder.positioned", x, rows, hidden);
    c.ops.arena.release(mark);
    var buf: [192]u8 = undefined;
    for (0..config.vision_layers) |i| {
        try c.poll();
        const stem = try std.fmt.bufPrint(&buf, names.tower ++ "blocks.{d}", .{i});
        try block(c, x, stem, rows);
        try c.record(try std.fmt.bufPrint(&buf, "encoder.block.{d}", .{i}), x, rows, hidden);
        c.ops.arena.release(mark);
    }
    try neck(c, x, count, out);
    try c.record("encoder.neck.0", out, geometry.tokens(), config.hidden);
}

fn positioned(c: Compute, patches: Tensor, geometry: Geometry, x: Tensor) !void {
    const h = (geometry.content.h + 15) / 16;
    const w = (geometry.content.w + 15) / 16;
    const projected = try c.linear(patches, names.patch_generator ++ "embedder", geometry.patches(), config.vision_hidden, 768, false);
    const field = try c.alloc(h * w * config.vision_hidden, .f32);
    const table = try c.weights.get(names.patch_generator ++ "pos_embed");
    if (@min(h, w) == 1) {
        try c.kernels.resize(c.stream, table, field.ptr, .{ 128, 128 }, .{ h, w }, config.vision_hidden, .area);
    } else {
        const horizontal = try c.alloc(128 * w * config.vision_hidden, .f32);
        try c.kernels.resize(c.stream, table, horizontal.ptr, .{ 128, 128 }, .{ 128, w }, config.vision_hidden, .horizontal);
        try c.kernels.resize(c.stream, horizontal, field.ptr, .{ 128, w }, .{ h, w }, config.vision_hidden, .vertical);
    }
    try c.kernels.position(c.stream, projected, (try c.weights.get(names.patch_generator ++ "cls_token.token")).ptr, field.ptr, x, .{ geometry.padded.h / 16, geometry.padded.w / 16 }, .{ geometry.content.h, geometry.content.w }, config.vision_hidden, config.vision_prefix);
}

fn neck(c: Compute, x: Tensor, patches: usize, out: Tensor) !void {
    const features: Tensor = .{ .ptr = x.ptr + config.vision_prefix * config.vision_hidden * 2, .kind = .bf16 };
    const first = try c.linear(features, "encoder.conv1", patches, config.hidden, config.vision_hidden, true);
    const norm1 = try c.norm(first, "encoder.layer_norm1", patches, config.hidden, config.vision_eps);
    const grouped = try c.alloc(patches * config.hidden, .bf16);
    try c.kernels.layout(c.stream, norm1, grouped, patches, 1, config.hidden, .horizontal_four);
    const reduced = try c.linear(grouped, "encoder.conv2", patches / 4, config.hidden, config.hidden * 4, false);
    try c.kernels.norm(c.stream, reduced, (try c.named("encoder.layer_norm2", "weight")).ptr, (try c.named("encoder.layer_norm2", "bias")).ptr, out, patches / 4, config.hidden, config.vision_eps);
    const summary = try c.linear(x, "encoder.sum_proj", 1, config.hidden, config.vision_hidden * 2, true);
    const last: Tensor = .{ .ptr = out.ptr + patches / 4 * config.hidden * 2, .kind = .bf16 };
    try c.kernels.norm(c.stream, summary, (try c.named("encoder.layer_norm3", "weight")).ptr, (try c.named("encoder.layer_norm3", "bias")).ptr, last, 1, config.hidden, config.vision_eps);
}
