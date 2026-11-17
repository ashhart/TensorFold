const ops = @import("ops.zig");
const Layer = @import("layers.zig").Layer;

const Encoder = struct {
    root: Layer,
    width: usize,
    rows: usize,
    compensated: bool,

    fn trace(self: Encoder, suffix: []const u8, tensor: ops.Tensor, index: usize) !void {
        if (index == 0) try self.root.runtime.observe(suffix, tensor, &.{ 1, self.rows, tensor.count / self.rows });
    }

    fn projection(self: Encoder, weights: Layer, name: []const u8, tensor: ops.Tensor) !ops.Tensor {
        if (!self.compensated) return weights.linear(name, tensor);
        const r = self.root.runtime;
        const prefix = try weights.named(name);
        const weight_key = try r.name("{s}.weight", .{prefix});
        const shape = try r.store.get(weight_key);
        if (shape.rank != 2 or shape.dim(1) == 0 or tensor.count % shape.dim(1) != 0) return error.SamProjectionShape;
        const weight = try r.weight(weight_key);
        const bias = try r.weight(try r.name("{s}.bias", .{prefix}));
        return r.own(try @import("dense_small.zig").run(r.g, tensor, weight, bias, tensor.count / shape.dim(1), shape.dim(0), shape.dim(1)));
    }

    fn layer(self: Encoder, input: ops.Tensor, index: usize) !ops.Tensor {
        const r = self.root.runtime;
        const block = try self.root.indexed("transformer.resblocks", index);
        const norm1 = try block.norm("ln_1", input, 1e-5);
        try self.trace("text.first.norm1", norm1, index);
        const attention_key = try block.named("attn");
        var qkv: [3]ops.Tensor = undefined;
        for (&qkv, 0..) |*tensor, part| tensor.* = try r.project(attention_key, norm1, self.width, @intCast(part));
        const context = try r.attend(qkv[0], qkv[1], qkv[2], 16, self.width, null, true);
        const projected = try block.linear("attn.out_proj", context);
        try self.trace("text.first.projected", projected, index);
        const residual = try r.add(input, projected);
        try self.trace("text.first.residual", residual, index);
        const norm2 = try block.norm("ln_2", residual, 1e-5);
        try self.trace("text.first.norm2", norm2, index);
        const expansion = try self.projection(block, "mlp.c_fc", norm2);
        try self.trace("text.first.expanded", expansion, index);
        const activation = try r.gelu(expansion);
        try self.trace("text.first.gelu", activation, index);
        const contraction = try self.projection(block, "mlp.c_proj", activation);
        try self.trace("text.first.contracted", contraction, index);
        return r.add(residual, contraction);
    }
};

pub fn encode(runtime: *ops.Runtime, tokens: []const u32) !ops.Tensor {
    return encodeWithMlp(runtime, tokens, false);
}

pub fn encodeWithMlp(r: *ops.Runtime, tokens: []const u32, compensated: bool) !ops.Tensor {
    if (tokens.len < 1 or tokens.len > 32) return error.SamTextLength;
    const root = Layer{ .runtime = r, .prefix = "detector.backbone.language_backbone" };
    const weights = try root.child("encoder");
    const vocabulary = try r.store.get(try weights.named("token_embedding.weight"));
    const words = try weights.parameter("token_embedding.weight");
    const positions = try weights.parameter("positional_embedding");
    if (vocabulary.rank != 2 or vocabulary.dim(1) != 1024 or positions.count < tokens.len * vocabulary.dim(1)) return error.SamTextEmbeddingShape;
    const tower = Encoder{ .root = weights, .width = vocabulary.dim(1), .rows = tokens.len, .compensated = compensated };
    const start = r.mark();
    errdefer r.trim(start, &.{});
    var value = try r.alloc(tower.rows * tower.width);
    const input = value.slice(f32, value.count);
    for (tokens, 0..) |id, row| {
        if (id >= vocabulary.dim(0)) return error.SamTokenOutOfRange;
        for (input[row * tower.width ..][0..tower.width], 0..) |*entry, channel| {
            entry.* = words.get(id * tower.width + channel) + positions.get(row * tower.width + channel);
        }
    }
    try r.observe("text.input", value, &.{ 1, tower.rows, tower.width });
    try r.g.beginBatch();
    errdefer if (r.g.batch != null) r.g.endBatch() catch {};
    for (0..24) |index| {
        value = try tower.layer(value, index);
        try r.observe(try r.name("text.layer.{d}", .{index}), value, &.{ 1, tower.rows, tower.width });
        r.trim(start, &.{value});
    }
    const normalized = try weights.norm("ln_final", value, 1e-5);
    value = try root.linear("resizer", normalized);
    try r.observe("text.output", value, &.{ 1, tower.rows, r.config.hidden });
    try r.g.endBatch();
    r.trim(start, &.{value});
    return value;
}
