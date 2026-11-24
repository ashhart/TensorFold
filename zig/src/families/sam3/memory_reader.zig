const std = @import("std");
const runtime = @import("runtime.zig");
const memory_bank = @import("memory_bank.zig");
const recall_plan = @import("recall_plan.zig");
const rotary = @import("rotary.zig");
const encodings = @import("encodings.zig");
const elementwise = @import("elementwise.zig");
const projection = @import("projection.zig");
const norms = @import("norms.zig");
const attention = @import("attention.zig");

const Runtime = runtime.Runtime;
const Tensor = runtime.Tensor;
const Scope = runtime.Scope;
const Entry = memory_bank.Entry;
const Plan = recall_plan.Plan;

const layer_count = 4;
const norm_eps: f32 = 1e-5;
const input_scale: f32 = 0.1;
const cached_offsets = 16;
const max_tokens = 16;
const time_width = 256;
const max_slots = recall_plan.max_grid_slots + recall_plan.max_pointer_slots;

pub fn pointerTokens(bucket_size: usize, hidden: usize, memory_width: usize) usize {
    return bucket_size * hidden / memory_width;
}

pub fn rotatedRows(grid_slots: usize, grid_rows: usize) usize {
    return grid_slots * grid_rows;
}

pub fn cachedTimeCode(offset: usize) bool {
    return offset < cached_offsets;
}

fn timeDivisor(frames: usize) usize {
    return @max(1, recall_plan.horizon(frames) -| 1);
}

const Sequence = struct {
    values: Tensor,
    positions: Tensor,
    image: ?Tensor,
    rotated: usize,
    entries: []const Entry,
    recall: *const Plan,
    cached: bool,
};

/// Must be called with no batch open: prepares constants and the attention workspace before its own batch.
pub fn condition(rt: *Runtime, coarse_image: Tensor, entries: []const Entry, recall: *const Plan, frames: usize) !Tensor {
    std.debug.assert(rt.gpu.batch == null);
    const config = rt.config;
    const side = config.featureSide();
    const rows = side * side;
    const hidden = config.hidden;
    const width = config.memory_width;
    const heads = config.tracker_heads;
    const grouped = config.grouped();
    const tokens = pointerTokens(config.bucket_size, hidden, width);
    const grid = recall.gridSlots();
    const pointers = recall.pointerSlots();
    if (grid.len == 0 or pointers.len == 0) return error.MissingPromptedFrame;
    if (tokens == 0 or tokens > max_tokens or hidden > time_width or hidden % heads != 0) return error.SamMemoryShape;
    if (coarse_image.count != rows * hidden) return error.SamMemoryShape;
    for (grid) |slot| if (slot.entry >= entries.len or entries[slot.entry].encoding.count != rows * width or
        entries[slot.entry].coarse_image.count != rows * hidden) return error.SamMemoryShape;
    for (pointers) |slot| if (slot.entry >= entries.len or entries[slot.entry].pointer.count != tokens * width)
        return error.SamMemoryShape;

    const mark = rt.mark();
    errdefer {
        if (rt.gpu.batch != null) rt.gpu.endBatch() catch {};
        rt.rewind(mark, &.{});
    }
    const tracker = Scope.root(rt, config.roots.tracker);
    const table = try rotary.generate(rt, side, hidden / heads, 1);
    const position = try encodings.sineGrid(rt, side, side, hidden);
    const memory_position = try encodings.sineGrid(rt, side, side, width);
    var times: [recall_plan.max_pointer_slots]Tensor = undefined;
    for (pointers, 0..) |slot, l| times[l] = try timeBlock(rt, tracker, slot.offset, recall_plan.horizon(frames), tokens);
    const keys = rotatedRows(grid.len, rows) + pointers.len * tokens;
    if (!grouped) try attention.reserve(rt, rows, keys);
    const cached = grouped and rt.observer == null;
    const zeros: ?Tensor = if (grouped and !cached) try elementwise.hostFill(rt, pointers.len * tokens * hidden, 0) else null;
    try rt.observe("tracker.position", position, &.{ 1, side, side, hidden });
    try rt.observe("tracker.memory_spatial_position", memory_position, &.{ 1, side, side, width });
    const time_rows = try tracker.param("maskmem_tpos_enc");

    try rt.gpu.beginBatch();
    var parts: [max_slots]Tensor = undefined;
    var count: usize = 0;
    if (!cached) for (grid) |slot| {
        parts[count] = entries[slot.entry].encoding;
        count += 1;
    };
    for (pointers) |slot| {
        parts[count] = entries[slot.entry].pointer;
        count += 1;
    }
    const values = try elementwise.concat(rt, parts[0..count]);
    count = 0;
    for (grid) |slot| {
        if ((slot.time_slot + 1) * width > time_rows.count) return error.SamMemoryShape;
        parts[count] = try elementwise.add(rt, memory_position, runtime.part(time_rows, slot.time_slot * width, width));
        count += 1;
    }
    for (pointers, 0..) |_, l| {
        parts[count] = times[l];
        count += 1;
    }
    const positions = try elementwise.concat(rt, parts[0..count]);
    const image: ?Tensor = if (zeros) |rows_of_zeros| blk: {
        for (grid, 0..) |slot, k| parts[k] = entries[slot.entry].coarse_image;
        parts[grid.len] = rows_of_zeros;
        break :blk try elementwise.concat(rt, parts[0 .. grid.len + 1]);
    } else null;
    const input = try elementwise.add(rt, coarse_image, try elementwise.scale(rt, position, input_scale));
    try rt.observe("tracker.input", input, &.{ 1, side, side, hidden });
    try rt.observe("tracker.memory", values, &.{ values.count / width, width });
    try rt.observe("tracker.memory_position", positions, &.{ positions.count / width, width });

    const sequence = Sequence{
        .values = values,
        .positions = positions,
        .image = image,
        .rotated = rotatedRows(grid.len, rows),
        .entries = entries,
        .recall = recall,
        .cached = cached,
    };
    const encoder = try tracker.sub("transformer.encoder");
    const layer_mark = rt.mark();
    var x = input;
    for (0..layer_count) |i| {
        const layer = try encoder.item("layers", i);
        x = if (grouped)
            try groupedLayer(rt, layer, i, x, coarse_image, sequence, table)
        else
            try singleLayer(rt, layer, x, sequence, table);
        try rt.observe(try rt.print("tracker.layers.{d}.output", .{i}), x, &.{ 1, side, side, hidden });
        rt.rewind(layer_mark, &.{x});
    }
    x = try norms.layerNorm(encoder, "norm", x, norm_eps, .plain);
    try rt.observe("tracker.output", x, &.{ 1, side, side, hidden });
    try rt.gpu.endBatch();
    rt.rewind(mark, &.{x});
    return x;
}

fn timeBlock(rt: *Runtime, tracker: Scope, offset: usize, horizon: usize, tokens: usize) !Tensor {
    const keep = cachedTimeCode(offset);
    const name = try rt.print("memory_reader.time.{d}.{d}.{d}", .{ offset, horizon, tokens });
    if (keep) if (rt.constant(name)) |existing| return existing;
    var code: [time_width]f32 = undefined;
    encodings.timeCode(&code, offset, horizon);
    const omega = try rt.upload(&code);
    try rt.gpu.beginBatch();
    errdefer if (rt.gpu.batch != null) rt.gpu.endBatch() catch {};
    const time_row = try projection.affine(tracker, "obj_ptr_tpos_proj", omega);
    var copies: [max_tokens]Tensor = undefined;
    for (copies[0..tokens]) |*copy| copy.* = time_row;
    const block = try elementwise.concat(rt, copies[0..tokens]);
    try rt.gpu.endBatch();
    return if (keep) try rt.keepConstant(name, block) else block;
}

fn rotate(rt: *Runtime, x: Tensor, table: rotary.Table, rotated_rows: usize) !Tensor {
    return rotary.rotate(rt, x, rt.config.tracker_heads, rt.config.hidden, table, rotated_rows);
}

fn attend(rt: *Runtime, q: Tensor, k: Tensor, v: Tensor) !Tensor {
    return attention.attend(rt, q, k, v, rt.config.tracker_heads, rt.config.hidden, .{});
}

fn groupedLayer(rt: *Runtime, layer: Scope, index: usize, x: Tensor, coarse_image: Tensor, sequence: Sequence, table: rotary.Table) !Tensor {
    const rows = x.count / rt.config.hidden;
    const n1 = try norms.layerNorm(layer, "norm1", x, norm_eps, .plain);
    const q = try rotate(rt, try projection.affine(layer, "self_attn_q_proj", n1), table, rows);
    const k = try rotate(rt, try projection.affine(layer, "self_attn_k_proj", n1), table, rows);
    const v = try projection.affine(layer, "self_attn_v_proj", n1);
    const x1 = try elementwise.add(rt, x, try projection.affine(layer, "self_attn_out_proj", try attend(rt, q, k, v)));

    const n2 = try norms.layerNorm(layer, "norm2", x1, norm_eps, .plain);
    const image_query = try projection.affine(layer, "image_cross_attn_q_proj", coarse_image);
    const cross_q = try rotate(rt, try elementwise.add(rt, image_query, try projection.affine(layer, "cross_attn_q_proj", n2)), table, rows);
    var keys: Tensor = undefined;
    var values: Tensor = undefined;
    if (sequence.cached) {
        const grid = sequence.recall.gridSlots();
        var key_parts: [recall_plan.max_grid_slots + 1]Tensor = undefined;
        var value_parts: [recall_plan.max_grid_slots + 1]Tensor = undefined;
        for (grid, 0..) |slot, g| {
            const pairs = sequence.entries[slot.entry].projections orelse return error.MissingMemoryProjections;
            key_parts[g] = pairs[index].key;
            value_parts[g] = pairs[index].value;
        }
        const pointer_keys = try projection.affine(layer, "cross_attn_k_proj", sequence.values);
        key_parts[grid.len] = try elementwise.add(rt, pointer_keys, try layer.param("image_cross_attn_k_proj.bias"));
        value_parts[grid.len] = try projection.affine(layer, "cross_attn_v_proj", sequence.values);
        keys = try elementwise.concat(rt, key_parts[0 .. grid.len + 1]);
        values = try elementwise.concat(rt, value_parts[0 .. grid.len + 1]);
    } else {
        const image = sequence.image orelse return error.SamMemoryShape;
        const image_keys = try projection.affine(layer, "image_cross_attn_k_proj", image);
        keys = try elementwise.add(rt, image_keys, try projection.affine(layer, "cross_attn_k_proj", sequence.values));
        values = try projection.affine(layer, "cross_attn_v_proj", sequence.values);
    }
    const cross_k = try rotate(rt, try elementwise.add(rt, keys, sequence.positions), table, sequence.rotated);
    const attended = try attend(rt, cross_q, cross_k, values);
    const x2 = try elementwise.add(rt, x1, try projection.affine(layer, "cross_attn_out_proj", attended));

    const n3 = try norms.layerNorm(layer, "norm3", x2, norm_eps, .plain);
    return elementwise.add(rt, x2, try projection.expandContract(layer, n3, .gelu));
}

fn singleLayer(rt: *Runtime, layer: Scope, x: Tensor, sequence: Sequence, table: rotary.Table) !Tensor {
    const rows = x.count / rt.config.hidden;
    const self_scope = try layer.sub("self_attn");
    const n1 = try norms.layerNorm(layer, "norm1", x, norm_eps, .plain);
    const q = try rotate(rt, try projection.affine(self_scope, "q_proj", n1), table, rows);
    const k = try rotate(rt, try projection.affine(self_scope, "k_proj", n1), table, rows);
    const v = try projection.affine(self_scope, "v_proj", n1);
    const x1 = try elementwise.add(rt, x, try projection.affine(self_scope, "out_proj", try observedAttention(rt, self_scope, q, k, v)));

    const cross = try layer.sub("cross_attn_image");
    const n2 = try norms.layerNorm(layer, "norm2", x1, norm_eps, .plain);
    const cross_q = try rotate(rt, try projection.affine(cross, "q_proj", n2), table, rows);
    const positioned = try elementwise.add(rt, sequence.values, sequence.positions);
    const cross_k = try rotate(rt, try projection.affine(cross, "k_proj", positioned), table, sequence.rotated);
    const cross_v = try projection.affine(cross, "v_proj", sequence.values);
    const attended = try observedAttention(rt, cross, cross_q, cross_k, cross_v);
    const x2 = try elementwise.add(rt, x1, try projection.affine(cross, "out_proj", attended));

    const n3 = try norms.layerNorm(layer, "norm3", x2, norm_eps, .plain);
    return elementwise.add(rt, x2, try projection.expandContract(layer, n3, .relu));
}

fn observedAttention(rt: *Runtime, block: Scope, q: Tensor, k: Tensor, v: Tensor) !Tensor {
    const hidden = rt.config.hidden;
    try rt.observe(try block.key("q"), q, &.{ q.count / hidden, hidden });
    try rt.observe(try block.key("k"), k, &.{ k.count / hidden, hidden });
    try rt.observe(try block.key("v"), v, &.{ v.count / hidden, hidden });
    const attended = try attend(rt, q, k, v);
    try rt.observe(try block.key("attended"), attended, &.{ attended.count / hidden, hidden });
    return attended;
}

test "SAM memory reader splits pointers into memory-width tokens and rotates grid keys only" {
    try std.testing.expectEqual(@as(usize, 4), pointerTokens(1, 256, 64));
    try std.testing.expectEqual(@as(usize, 16), pointerTokens(16, 256, 256));
    try std.testing.expectEqual(@as(usize, 3 * 5184), rotatedRows(3, 5184));
    try std.testing.expectEqual(@as(usize, 0), rotatedRows(0, 5184));
}

test "SAM memory reader normalises time codes by the pointer horizon and caches small offsets" {
    for ([_]usize{ 120, 16, 2, 1 }, [_]usize{ 15, 15, 1, 1 }) |frames, divisor| {
        try std.testing.expectEqual(divisor, timeDivisor(frames));
        var code: [time_width]f32 = undefined;
        encodings.timeCode(&code, divisor, recall_plan.horizon(frames));
        try std.testing.expectEqual(@sin(@as(f32, 1)), code[0]);
        try std.testing.expectEqual(@cos(@as(f32, 1)), code[time_width / 2]);
    }
    try std.testing.expect(cachedTimeCode(0));
    try std.testing.expect(cachedTimeCode(15));
    try std.testing.expect(!cachedTimeCode(16));
    try std.testing.expect(!cachedTimeCode(26));
}
