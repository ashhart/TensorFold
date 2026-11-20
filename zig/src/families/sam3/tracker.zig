const std = @import("std");
const ops = @import("ops.zig");
const memory = @import("memory.zig");
const decoder = @import("decoder.zig");
const embeddings = @import("prompt_encoder.zig");
const position = @import("position.zig");
const prompts = @import("prompts.zig");
const selection = @import("memory_selection.zig");
const Layer = @import("layers.zig").Layer;
pub const Result = decoder.Result;
pub const Memory = memory.Memory;

fn seedMask(r: *ops.Runtime, output: *Result, mask: prompts.Mask) !void {
    const logits = try r.alloc(mask.values.len);
    var present = false;
    for (mask.values, logits.slice(f32, logits.count)) |probability, *value| {
        present = present or probability > 0;
        value.* = probability * 20 - 10;
    }
    output.input_masks = ops.feature(logits, mask.size.height, mask.size.width, 1);
    const side = r.config.featureSide() * 4;
    output.masks = try r.resize(output.input_masks.?, side, side, true);
    output.scores[0] = if (present) 10 else -10;
    output.ious[0] = 1;
    if (!present) {
        const owner = Layer{ .runtime = r, .prefix = r.config.trackerPrefix() };
        output.pointer = if (r.config.version == .sam31) try owner.linear("no_obj_ptr_linear", output.pointer) else try owner.parameter("no_obj_ptr");
    }
}

pub fn seed(r: *ops.Runtime, features: [3]ops.Feature, prompt: prompts.Prompt, size: prompts.Size) !Result {
    const mark = r.mark();
    errdefer r.trim(mark, &.{});
    const owner = Layer{ .runtime = r, .prefix = r.config.trackerPrefix() };
    const encoded = try embeddings.encode(r, prompt, size);
    const initial = try owner.parameter(if (r.config.version == .sam31) "interactivity_no_mem_embed" else "no_mem_embed");
    var result = try decoder.decode(r, features, try r.add(features[2].tensor, initial), .{
        .sparse = encoded.sparse,
        .dense = encoded.dense,
        .multimask = prompt.mask == null and prompt.points.len == 1 and prompt.boxes.len == 0,
    });
    if (prompt.mask) |mask| if (mask.format == .probabilities and prompt.points.len + prompt.boxes.len == 0) try seedMask(r, &result, mask);
    var retained: [3]ops.Tensor = undefined;
    r.trim(mark, result.retained(&retained));
    return result;
}

fn pointerPosition(owner: Layer, relative: usize, horizon: usize, count: usize) !ops.Tensor {
    const r = owner.runtime;
    const name = try r.name("position.pointer.{s}.{d}.{d}.{d}", .{ owner.prefix, relative, horizon, count });
    if (r.constant(name)) |cached| return cached;
    const width = r.config.hidden;
    const half = width / 2;
    const embedding = try r.alloc(width);
    for (0..half) |frequency| {
        const normalized = @as(f32, @floatFromInt(relative)) / @as(f32, @floatFromInt(@max(1, horizon - 1)));
        const exponent = @as(f32, @floatFromInt(frequency / 2 * 2)) / @as(f32, @floatFromInt(half));
        const angle = normalized / std.math.pow(f32, 10000, exponent);
        embedding.slice(f32, width)[frequency] = @sin(angle);
        embedding.slice(f32, width)[half + frequency] = @cos(angle);
    }
    const encoded = try owner.linear("obj_ptr_tpos_proj", embedding);
    const repeated = try r.alloc(count);
    if (encoded.count != r.config.memory_width or count % encoded.count != 0) return error.SamPointerPositionShape;
    for (0..count / encoded.count) |token| @memcpy(repeated.slice(f32, count)[token * encoded.count ..][0..encoded.count], encoded.slice(f32, encoded.count));
    return if (relative < r.config.pointer_frames) r.remember(name, repeated) else repeated;
}

const Memories = struct {
    values: ops.Tensor,
    positions: ops.Tensor,
    images: ops.Tensor,
    spatial: usize,
    cached: bool,
    history: []const Memory,
    plan: selection.Plan,

    fn gather(r: *ops.Runtime, owner: Layer, history: []const Memory, plan: selection.Plan, pointer_positions: []const ops.Tensor) !Memories {
        const cached = r.config.version == .sam31 and r.precision == .mixed;
        const total = plan.spatial.len + plan.pointers.len;
        const values = try r.a.alloc(ops.Tensor, total);
        const positions = try r.a.alloc(ops.Tensor, total);
        const images = try r.a.alloc(ops.Tensor, plan.spatial.len + 1);
        const side = r.config.featureSide();
        const spatial_position = try position.spatial(r, side, side, r.config.memory_width);
        const temporal = try owner.parameter("maskmem_tpos_enc");
        var pointer_count: usize = 0;
        for (plan.spatial.constSlice(), 0..) |entry, at| {
            const frame = history[entry.history];
            values[at] = frame.encoded;
            images[at] = frame.image;
            positions[at] = try r.add(spatial_position, ops.view(temporal, entry.temporal * r.config.memory_width, r.config.memory_width));
        }
        for (plan.pointers.constSlice(), pointer_positions, 0..) |entry, encoded, at| {
            const pointer = history[entry.history].pointer;
            values[plan.spatial.len + at] = pointer;
            positions[plan.spatial.len + at] = encoded;
            pointer_count += pointer.count / r.config.memory_width;
        }
        const empty = try r.alloc(pointer_count * r.config.hidden);
        @memset(empty.slice(f32, empty.count), 0);
        images[plan.spatial.len] = empty;
        return .{
            .values = try r.concat(if (cached) values[plan.spatial.len..] else values),
            .positions = try r.concat(positions),
            .images = if (r.config.version == .sam31 and !cached) try r.concat(images) else empty,
            .spatial = plan.spatial.len * side * side,
            .cached = cached,
            .history = history,
            .plan = plan,
        };
    }

    fn multiplexKeys(self: Memories, block: Layer, index: usize) ![2]ops.Tensor {
        const r = block.runtime;
        if (!self.cached) {
            const key = try r.add(try block.linear("image_cross_attn_k_proj", self.images), try block.linear("cross_attn_k_proj", self.values));
            return .{ try r.add(key, self.positions), try block.linear("cross_attn_v_proj", self.values) };
        }
        const keys = try r.a.alloc(ops.Tensor, self.plan.spatial.len + 1);
        const values = try r.a.alloc(ops.Tensor, keys.len);
        for (self.plan.spatial.constSlice(), 0..) |entry, at| {
            const cache = self.history[entry.history].projected orelse return error.MissingSamMemoryProjection;
            keys[at] = cache[index].key;
            values[at] = cache[index].value;
        }
        keys[keys.len - 1] = try r.add(try block.linear("cross_attn_k_proj", self.values), try block.parameter("image_cross_attn_k_proj.bias"));
        values[values.len - 1] = try block.linear("cross_attn_v_proj", self.values);
        return .{ try r.add(try r.concat(keys), self.positions), try r.concat(values) };
    }
};

fn attend(layer: Layer, query: ops.Tensor, keys: ops.Tensor, values: ops.Tensor, spatial: usize) !ops.Tensor {
    const r = layer.runtime;
    const width = r.config.hidden;
    const q = try r.rope(try layer.linear("q_proj", query), 1, width, r.config.featureSide(), false, query.count / width);
    const k = try r.rope(try layer.linear("k_proj", keys), 1, width, r.config.featureSide(), false, spatial);
    const v = try layer.linear("v_proj", values);
    if (r.observer != null) {
        try r.observe(try layer.named("q"), q, &.{ q.count / width, width });
        try r.observe(try layer.named("k"), k, &.{ k.count / width, width });
        try r.observe(try layer.named("v"), v, &.{ v.count / width, width });
    }
    const attended = try r.attend(q, k, v, 1, width, null, false);
    if (r.observer != null) try r.observe(try layer.named("attended"), attended, &.{ attended.count / width, width });
    return layer.linear("out_proj", attended);
}

fn step(block: Layer, input: ops.Tensor, image: ops.Tensor, memories: Memories, index: usize) !ops.Tensor {
    const r = block.runtime;
    const side = r.config.featureSide();
    const width = r.config.hidden;
    const norm = try block.norm("norm1", input, 1e-5);
    var output: ops.Tensor = undefined;
    if (r.config.version == .sam3) {
        output = try r.add(input, try attend(try block.child("self_attn"), norm, norm, norm, side * side));
        const query = try block.norm("norm2", output, 1e-5);
        output = try r.add(output, try attend(try block.child("cross_attn_image"), query, try r.add(memories.values, memories.positions), memories.values, memories.spatial));
        const hidden = try block.feedForward(try block.norm("norm3", output, 1e-5), false);
        return r.add(output, hidden);
    }
    const heads = r.config.tracker_heads;
    const q = try r.rope(try block.linear("self_attn_q_proj", norm), heads, width, side, false, side * side);
    const k = try r.rope(try block.linear("self_attn_k_proj", norm), heads, width, side, false, side * side);
    const v = try block.linear("self_attn_v_proj", norm);
    output = try r.add(input, try block.linear("self_attn_out_proj", try r.attend(q, k, v, heads, width, null, false)));
    const query = try r.add(try block.linear("image_cross_attn_q_proj", image), try block.linear("cross_attn_q_proj", try block.norm("norm2", output, 1e-5)));
    const key, const value = try memories.multiplexKeys(block, index);
    const context = try r.attend(try r.rope(query, heads, width, side, false, side * side), try r.rope(key, heads, width, side, false, memories.spatial), value, heads, width, null, false);
    output = try r.add(output, try block.linear("cross_attn_out_proj", context));
    return r.add(output, try block.feedForward(try block.norm("norm3", output, 1e-5), true));
}

fn observePlan(r: *ops.Runtime, history: []const Memory, plan: selection.Plan) !void {
    if (r.observer == null) return;
    const mark = r.mark();
    defer r.trim(mark, &.{});
    const spatial = try r.alloc(plan.spatial.len * 2);
    for (plan.spatial.constSlice(), 0..) |entry, at| {
        spatial.slice(f32, spatial.count)[at * 2] = @floatFromInt(history[entry.history].index);
        spatial.slice(f32, spatial.count)[at * 2 + 1] = @floatFromInt(entry.temporal);
    }
    try r.observe("tracker.spatial_plan", spatial, &.{ plan.spatial.len, 2 });
    const pointers = try r.alloc(plan.pointers.len * 2);
    for (plan.pointers.constSlice(), 0..) |entry, at| {
        pointers.slice(f32, pointers.count)[at * 2] = @floatFromInt(history[entry.history].index);
        pointers.slice(f32, pointers.count)[at * 2 + 1] = @floatFromInt(entry.relative);
    }
    try r.observe("tracker.pointer_plan", pointers, &.{ plan.pointers.len, 2 });
}

pub fn propagate(r: *ops.Runtime, features: [3]ops.Feature, history: []const Memory, index: usize, frames: usize, direction: selection.Direction, active: u16) !Result {
    const mark = r.mark();
    errdefer r.trim(mark, &.{});
    const multiplex = r.config.version == .sam31;
    const plan = try selection.select(history, index, frames, direction, multiplex);
    if (plan.pointers.len == 0 or plan.spatial.len == 0) return error.MissingConditioningFrame;
    try observePlan(r, history, plan);
    const owner = Layer{ .runtime = r, .prefix = r.config.trackerPrefix() };
    const pointer_positions = try r.a.alloc(ops.Tensor, plan.pointers.len);
    for (plan.pointers.constSlice(), pointer_positions) |entry, *encoded| encoded.* = try pointerPosition(owner, entry.relative, @min(frames, r.config.pointer_frames), history[entry.history].pointer.count);
    const side = r.config.featureSide();
    const spatial = try position.spatial(r, side, side, r.config.hidden);
    const memory_position = try position.spatial(r, side, side, r.config.memory_width);
    try r.observe("tracker.position", spatial, &.{ 1, side, side, r.config.hidden });
    try r.observe("tracker.memory_spatial_position", memory_position, &.{ 1, side, side, r.config.memory_width });
    if (!multiplex) {
        const keys = plan.spatial.len * side * side + plan.pointers.len * r.config.hidden / r.config.memory_width;
        if (r.split_attention) {
            _ = try r.g.splitAttentionWorkspace(side * side, keys, 1, 1, @import("sam3_kernels").attention_split.chunk_blocks);
        } else try r.g.reserveAttention(side * side, keys, 1, 1);
    }
    try r.g.beginBatch();
    errdefer if (r.g.batch != null) r.g.endBatch() catch {};
    const memories = try Memories.gather(r, owner, history, plan, pointer_positions);
    var value = try r.add(features[2].tensor, try r.scale(spatial, 0.1));
    try r.observe("tracker.input", value, &.{ 1, side, side, r.config.hidden });
    try r.observe("tracker.memory", memories.values, &.{ memories.values.count / r.config.memory_width, r.config.memory_width });
    try r.observe("tracker.memory_pos", memories.positions, &.{ memories.positions.count / r.config.memory_width, r.config.memory_width });
    const scratch = r.mark();
    for (0..4) |layer_index| {
        value = try step(try owner.indexed("transformer.encoder.layers", layer_index), value, features[2].tensor, memories, layer_index);
        try r.observe(try r.name("tracker.layers.{d}.output", .{layer_index}), value, &.{ 1, side, side, r.config.hidden });
        r.trim(scratch, &.{value});
    }
    value = try owner.norm("transformer.encoder.norm", value, 1e-5);
    try r.observe("tracker.output", value, &.{ 1, side, side, r.config.hidden });
    try r.g.endBatch();
    const output = try decoder.decode(r, features, value, .{ .multiplex = multiplex, .active = active });
    r.trim(mark, &.{ output.masks.tensor, output.pointer });
    return output;
}
