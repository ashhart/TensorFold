const std = @import("std");
const ops = @import("ops.zig");
const decoder = @import("decoder.zig");
const embeddings = @import("prompt_encoder.zig");
const memory = @import("memory.zig");
const position = @import("position.zig");
const prompts = @import("prompts.zig");
const selection = @import("memory_selection.zig");
const Layer = @import("layers.zig").Layer;

pub const Result = decoder.Result;
pub const Memory = memory.Memory;

fn probabilitySeed(root: Layer, prediction: *Result, mask: prompts.Mask) !void {
    const r = root.runtime;
    const tensor = try r.alloc(mask.values.len);
    var nonempty = false;
    for (tensor.slice(f32, tensor.count), mask.values) |*logit, probability| {
        logit.* = 20 * probability - 10;
        nonempty = nonempty or probability > 0;
    }
    prediction.input_masks = ops.feature(tensor, mask.size.height, mask.size.width, 1);
    const side = 4 * r.config.featureSide();
    prediction.masks = try r.resize(prediction.input_masks.?, side, side, true);
    prediction.scores[0] = if (nonempty) 10 else -10;
    prediction.ious[0] = 1;
    if (!nonempty) prediction.pointer = if (r.config.version == .sam31)
        try root.linear("no_obj_ptr_linear", prediction.pointer)
    else
        try root.parameter("no_obj_ptr");
}

pub fn seed(r: *ops.Runtime, features: [3]ops.Feature, prompt: prompts.Prompt, size: prompts.Size) !Result {
    const start = r.mark();
    errdefer r.trim(start, &.{});
    const root = Layer{ .runtime = r, .prefix = r.config.trackerPrefix() };
    const encoded = try embeddings.encode(r, prompt, size);
    const key = if (r.config.version == .sam31) "interactivity_no_mem_embed" else "no_mem_embed";
    const image = try r.add(features[2].tensor, try root.parameter(key));
    var prediction = try decoder.decode(r, features, image, .{
        .sparse = encoded.sparse,
        .dense = encoded.dense,
        .multimask = prompt.points.len == 1 and prompt.boxes.len == 0 and prompt.mask == null,
    });
    if (prompt.mask) |mask| {
        if (mask.format == .probabilities and prompt.points.len == 0 and prompt.boxes.len == 0)
            try probabilitySeed(root, &prediction, mask);
    }
    var storage: [3]ops.Tensor = undefined;
    r.trim(start, prediction.retained(&storage));
    return prediction;
}

fn temporalPointer(root: Layer, relative: usize, horizon: usize, count: usize) !ops.Tensor {
    const r = root.runtime;
    const key = try r.name("position.pointer.{s}.{d}.{d}.{d}", .{ root.prefix, relative, horizon, count });
    if (r.constant(key)) |cached| return cached;
    const half = r.config.hidden / 2;
    const normalized = @as(f32, @floatFromInt(relative)) / @as(f32, @floatFromInt(@max(1, horizon - 1)));
    const phases = try r.alloc(2 * half);
    const values = phases.slice(f32, phases.count);
    for (0..half) |band| {
        const exponent = @as(f32, @floatFromInt(2 * (band / 2))) / @as(f32, @floatFromInt(half));
        const phase = normalized / std.math.pow(f32, 10000, exponent);
        values[band] = @sin(phase);
        values[half + band] = @cos(phase);
    }
    const projected = try root.linear("obj_ptr_tpos_proj", phases);
    if (projected.count != r.config.memory_width or count % projected.count != 0) return error.SamPointerPositionShape;
    const output = try r.alloc(count);
    const target = output.slice(f32, output.count);
    for (0..count / projected.count) |row| {
        @memcpy(target[row * projected.count ..][0..projected.count], projected.slice(f32, projected.count));
    }
    return if (relative < r.config.pointer_frames) r.remember(key, output) else output;
}

const Retrieval = struct {
    memory: ops.Tensor,
    position: ops.Tensor,
    image: ops.Tensor,
    spatial_rows: usize,
    cache: bool,
    history: []const Memory,
    plan: selection.Plan,

    fn assemble(root: Layer, history: []const Memory, plan: selection.Plan, pointer_positions: []const ops.Tensor) !Retrieval {
        const r = root.runtime;
        const cache = r.config.version == .sam31 and r.precision == .mixed;
        const side = r.config.featureSide();
        const spatial_position = try position.spatial(r, side, side, r.config.memory_width);
        const temporal = try root.parameter("maskmem_tpos_enc");
        var values: [29]ops.Tensor = undefined;
        var positions: [29]ops.Tensor = undefined;
        var images: [11]ops.Tensor = undefined;
        const spatial_count = plan.spatial.len;
        const count = spatial_count + plan.pointers.len;
        for (plan.spatial.constSlice(), 0..) |entry, row| {
            const frame = history[entry.history];
            values[row] = frame.encoded;
            images[row] = frame.image;
            const offset = ops.view(temporal, entry.temporal * r.config.memory_width, r.config.memory_width);
            positions[row] = try r.add(spatial_position, offset);
        }
        var pointer_rows: usize = 0;
        for (plan.pointers.constSlice(), pointer_positions, 0..) |entry, phase, row| {
            const pointer = history[entry.history].pointer;
            values[spatial_count + row] = pointer;
            positions[spatial_count + row] = phase;
            pointer_rows += pointer.count / r.config.memory_width;
        }
        const padding = try r.alloc(pointer_rows * r.config.hidden);
        @memset(padding.slice(f32, padding.count), 0);
        images[spatial_count] = padding;
        return .{
            .memory = try r.concat(values[if (cache) spatial_count else 0..count]),
            .position = try r.concat(positions[0..count]),
            .image = if (r.config.version == .sam31 and !cache) try r.concat(images[0 .. spatial_count + 1]) else padding,
            .spatial_rows = spatial_count * side * side,
            .cache = cache,
            .history = history,
            .plan = plan,
        };
    }

    fn projected(self: Retrieval, layer: Layer, index: usize) ![2]ops.Tensor {
        const r = layer.runtime;
        if (self.cache) {
            var keys: [11]ops.Tensor = undefined;
            var values: [11]ops.Tensor = undefined;
            const count = self.plan.spatial.len;
            for (self.plan.spatial.constSlice(), 0..) |entry, row| {
                const cached = self.history[entry.history].projected orelse return error.MissingSamMemoryProjection;
                keys[row] = cached[index].key;
                values[row] = cached[index].value;
            }
            const bias = try layer.parameter("image_cross_attn_k_proj.bias");
            keys[count] = try r.add(try layer.linear("cross_attn_k_proj", self.memory), bias);
            values[count] = try layer.linear("cross_attn_v_proj", self.memory);
            return .{ try r.add(try r.concat(keys[0 .. count + 1]), self.position), try r.concat(values[0 .. count + 1]) };
        }
        const image_key = try layer.linear("image_cross_attn_k_proj", self.image);
        const memory_key = try layer.linear("cross_attn_k_proj", self.memory);
        const key = try r.add(try r.add(image_key, memory_key), self.position);
        return .{ key, try layer.linear("cross_attn_v_proj", self.memory) };
    }
};

fn singleAttention(root: Layer, query: ops.Tensor, keys: ops.Tensor, values: ops.Tensor, spatial_rows: usize) !ops.Tensor {
    const r = root.runtime;
    const width = r.config.hidden;
    const side = r.config.featureSide();
    const q = try r.rope(try root.linear("q_proj", query), 1, width, side, false, query.count / width);
    const k = try r.rope(try root.linear("k_proj", keys), 1, width, side, false, spatial_rows);
    const v = try root.linear("v_proj", values);
    if (r.observer != null) {
        for ([_][]const u8{ "q", "k", "v" }, [_]ops.Tensor{ q, k, v }) |label, tensor| {
            try r.observe(try root.named(label), tensor, &.{ tensor.count / width, width });
        }
    }
    const context = try r.attend(q, k, v, 1, width, null, false);
    if (r.observer != null) try r.observe(try root.named("attended"), context, &.{ context.count / width, width });
    return root.linear("out_proj", context);
}

fn memoryLayer(root: Layer, input: ops.Tensor, image: ops.Tensor, retrieval: Retrieval, index: usize) !ops.Tensor {
    const r = root.runtime;
    const width = r.config.hidden;
    const side = r.config.featureSide();
    const norm1 = try root.norm("norm1", input, 1e-5);
    if (r.config.version == .sam3) {
        const self_update = try singleAttention(try root.child("self_attn"), norm1, norm1, norm1, side * side);
        const residual = try r.add(input, self_update);
        const norm2 = try root.norm("norm2", residual, 1e-5);
        const key = try r.add(retrieval.memory, retrieval.position);
        const cross_update = try singleAttention(try root.child("cross_attn_image"), norm2, key, retrieval.memory, retrieval.spatial_rows);
        const cross_residual = try r.add(residual, cross_update);
        const norm3 = try root.norm("norm3", cross_residual, 1e-5);
        return r.add(cross_residual, try root.feedForward(norm3, false));
    }
    const heads = r.config.tracker_heads;
    const q = try r.rope(try root.linear("self_attn_q_proj", norm1), heads, width, side, false, side * side);
    const k = try r.rope(try root.linear("self_attn_k_proj", norm1), heads, width, side, false, side * side);
    const v = try root.linear("self_attn_v_proj", norm1);
    const self_update = try root.linear("self_attn_out_proj", try r.attend(q, k, v, heads, width, null, false));
    const residual = try r.add(input, self_update);
    const norm2 = try root.norm("norm2", residual, 1e-5);
    const image_query = try root.linear("image_cross_attn_q_proj", image);
    const memory_query = try root.linear("cross_attn_q_proj", norm2);
    const query = try r.rope(try r.add(image_query, memory_query), heads, width, side, false, side * side);
    const projected = try retrieval.projected(root, index);
    const key = try r.rope(projected[0], heads, width, side, false, retrieval.spatial_rows);
    const cross_update = try root.linear("cross_attn_out_proj", try r.attend(query, key, projected[1], heads, width, null, false));
    const cross_residual = try r.add(residual, cross_update);
    const norm3 = try root.norm("norm3", cross_residual, 1e-5);
    return r.add(cross_residual, try root.feedForward(norm3, true));
}

fn observePlan(r: *ops.Runtime, history: []const Memory, plan: selection.Plan) !void {
    if (r.observer == null) return;
    const start = r.mark();
    defer r.trim(start, &.{});
    const lists = .{ plan.spatial.constSlice(), plan.pointers.constSlice() };
    inline for (lists, .{ "tracker.spatial_plan", "tracker.pointer_plan" }) |entries, label| {
        const tensor = try r.alloc(entries.len * 2);
        const data = tensor.slice(f32, tensor.count);
        for (entries, 0..) |entry, row| {
            data[2 * row] = @floatFromInt(history[entry.history].index);
            data[2 * row + 1] = @floatFromInt(if (@hasField(@TypeOf(entry), "temporal")) entry.temporal else entry.relative);
        }
        try r.observe(label, tensor, &.{ entries.len, 2 });
    }
}

pub fn propagate(r: *ops.Runtime, features: [3]ops.Feature, history: []const Memory, index: usize, frames: usize, direction: selection.Direction, active: u16) !Result {
    const start = r.mark();
    errdefer r.trim(start, &.{});
    const multiplex = r.config.version == .sam31;
    const plan = try selection.select(history, index, frames, direction, multiplex);
    if (plan.spatial.len == 0 or plan.pointers.len == 0) return error.MissingConditioningFrame;
    try observePlan(r, history, plan);
    const root = Layer{ .runtime = r, .prefix = r.config.trackerPrefix() };
    var phases: [19]ops.Tensor = undefined;
    for (plan.pointers.constSlice(), 0..) |entry, row| {
        phases[row] = try temporalPointer(root, entry.relative, @min(frames, r.config.pointer_frames), history[entry.history].pointer.count);
    }
    const side = r.config.featureSide();
    const shape = [_]usize{ 1, side, side, r.config.hidden };
    const spatial = try position.spatial(r, side, side, r.config.hidden);
    const memory_position = try position.spatial(r, side, side, r.config.memory_width);
    try r.observe("tracker.position", spatial, &shape);
    try r.observe("tracker.memory_spatial_position", memory_position, &.{ 1, side, side, r.config.memory_width });
    if (!multiplex) {
        const keys = plan.spatial.len * side * side + plan.pointers.len * r.config.hidden / r.config.memory_width;
        if (r.split_attention) {
            _ = try r.g.splitAttentionWorkspace(side * side, keys, 1, 1, @import("sam3_kernels").attention_split.chunk_blocks);
        } else try r.g.reserveAttention(side * side, keys, 1, 1);
    }
    try r.g.beginBatch();
    errdefer if (r.g.batch != null) r.g.endBatch() catch {};
    const retrieval = try Retrieval.assemble(root, history, plan, phases[0..plan.pointers.len]);
    var value = try r.add(features[2].tensor, try r.scale(spatial, 0.1));
    try r.observe("tracker.input", value, &shape);
    try r.observe("tracker.memory", retrieval.memory, &.{ retrieval.memory.count / r.config.memory_width, r.config.memory_width });
    try r.observe("tracker.memory_pos", retrieval.position, &.{ retrieval.position.count / r.config.memory_width, r.config.memory_width });
    const scratch = r.mark();
    for (0..4) |layer| {
        value = try memoryLayer(try root.indexed("transformer.encoder.layers", layer), value, features[2].tensor, retrieval, layer);
        try r.observe(try r.name("tracker.layers.{d}.output", .{layer}), value, &shape);
        r.trim(scratch, &.{value});
    }
    value = try root.norm("transformer.encoder.norm", value, 1e-5);
    try r.observe("tracker.output", value, &shape);
    try r.g.endBatch();
    const prediction = try decoder.decode(r, features, value, .{ .multiplex = multiplex, .active = active });
    r.trim(start, &.{ prediction.masks.tensor, prediction.pointer });
    return prediction;
}
