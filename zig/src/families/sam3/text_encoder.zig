const ops = @import("ops.zig");
const Layer = @import("layers.zig").Layer;

fn trace(r: *ops.Runtime, enabled: bool, name: []const u8, value: ops.Tensor, rows: usize) !void {
    if (enabled) try r.observe(name, value, &.{ 1, rows, value.count / rows });
}

fn mlpProjection(layer: Layer, suffix: []const u8, input: ops.Tensor, compensated: bool) !ops.Tensor {
    if (!compensated) return layer.linear(suffix, input);
    const r = layer.runtime;
    const name = try layer.named(suffix);
    const weights = try r.name("{s}.weight", .{name});
    const matrix = try r.store.get(weights);
    if (matrix.rank != 2 or matrix.dim(1) == 0 or input.count % matrix.dim(1) != 0) return error.SamProjectionShape;
    return r.own(try @import("dense_small.zig").run(r.g, input, try r.weight(weights), try r.weight(try r.name("{s}.bias", .{name})), input.count / matrix.dim(1), matrix.dim(0), matrix.dim(1)));
}

fn block(layer: Layer, input: ops.Tensor, width: usize, detailed: bool, compensated: bool) !ops.Tensor {
    const r = layer.runtime;
    const rows = input.count / width;
    const normalized = try layer.norm("ln_1", input, 1e-5);
    try trace(r, detailed, "text.first.norm1", normalized, rows);
    const attention = try layer.named("attn");
    const q = try r.project(attention, normalized, width, 0);
    const k = try r.project(attention, normalized, width, 1);
    const v = try r.project(attention, normalized, width, 2);
    const mixed = try r.attend(q, k, v, 16, width, null, true);
    const projected = try layer.linear("attn.out_proj", mixed);
    try trace(r, detailed, "text.first.projected", projected, rows);
    const residual = try r.add(input, projected);
    try trace(r, detailed, "text.first.residual", residual, rows);
    const norm2 = try layer.norm("ln_2", residual, 1e-5);
    try trace(r, detailed, "text.first.norm2", norm2, rows);
    const expanded = try mlpProjection(layer, "mlp.c_fc", norm2, compensated);
    try trace(r, detailed, "text.first.expanded", expanded, rows);
    const activated = try r.gelu(expanded);
    try trace(r, detailed, "text.first.gelu", activated, rows);
    const contracted = try mlpProjection(layer, "mlp.c_proj", activated, compensated);
    try trace(r, detailed, "text.first.contracted", contracted, rows);
    return r.add(residual, contracted);
}

pub fn encode(r: *ops.Runtime, ids: []const u32) !ops.Tensor {
    return encodeWithMlp(r, ids, false);
}

pub fn encodeWithMlp(r: *ops.Runtime, ids: []const u32, compensated: bool) !ops.Tensor {
    if (ids.len == 0 or ids.len > 32) return error.SamTextLength;
    const root = Layer{ .runtime = r, .prefix = "detector.backbone.language_backbone" };
    const encoder = try root.child("encoder");
    const words = try encoder.parameter("token_embedding.weight");
    const positions = try encoder.parameter("positional_embedding");
    const metadata = try r.store.get(try encoder.named("token_embedding.weight"));
    if (metadata.rank != 2 or metadata.dim(1) != 1024 or positions.count < ids.len * metadata.dim(1)) return error.SamTextEmbeddingShape;
    const width = metadata.dim(1);
    const mark = r.mark();
    errdefer r.trim(mark, &.{});
    var value = try r.alloc(ids.len * width);
    for (ids, 0..) |id, token| {
        if (id >= metadata.dim(0)) return error.SamTokenOutOfRange;
        for (0..width) |channel| value.slice(f32, value.count)[token * width + channel] = words.get(id * width + channel) + positions.get(token * width + channel);
    }
    try r.observe("text.input", value, &.{ 1, ids.len, width });
    try r.g.beginBatch();
    errdefer if (r.g.batch != null) r.g.endBatch() catch {};
    for (0..24) |index| {
        value = try block(try encoder.indexed("transformer.resblocks", index), value, width, index == 0, compensated);
        try r.observe(try r.name("text.layer.{d}", .{index}), value, &.{ 1, ids.len, width });
        r.trim(mark, &.{value});
    }
    value = try root.linear("resizer", try encoder.norm("ln_final", value, 1e-5));
    try r.observe("text.output", value, &.{ 1, ids.len, r.config.hidden });
    try r.g.endBatch();
    r.trim(mark, &.{value});
    return value;
}
