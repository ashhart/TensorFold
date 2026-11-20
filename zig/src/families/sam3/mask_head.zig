const std = @import("std");
const runtime = @import("runtime.zig");
const elementwise = @import("elementwise.zig");
const projection = @import("projection.zig");
const norms = @import("norms.zig");
const attention = @import("attention.zig");
const convolution = @import("convolution.zig");
const encodings = @import("encodings.zig");
const outcome = @import("outcome.zig");

const Runtime = runtime.Runtime;
const Tensor = runtime.Tensor;
const FeatureMap = runtime.FeatureMap;
const Scope = runtime.Scope;
const Outcome = outcome.Outcome;
const Pick = outcome.Pick;
const max_slots = outcome.max_slots;

const exchange_heads = 8;
const exchange_eps: f32 = 1e-5;
const upscale_eps: f32 = 1e-6;
const chain_depth = 3;

pub const Request = struct {
    grouped: bool = false,
    active: u16 = 1,
    pick: Pick = .best_alternative,
    dense: ?Tensor = null,
    sparse: ?Tensor = null,
};

const RowMap = struct {
    slots: usize,
    candidates: usize,
    prompt_rows: usize,

    fn learned(self: RowMap) usize {
        return 2 * self.slots + self.slots * self.candidates;
    }
    fn total(self: RowMap) usize {
        return self.learned() + self.prompt_rows;
    }
    fn presenceRow(_: RowMap, slot: usize) usize {
        return slot;
    }
    fn qualityRow(self: RowMap, slot: usize) usize {
        return self.slots + slot;
    }
    fn maskRow(self: RowMap, slot: usize, candidate: usize) usize {
        return 2 * self.slots + slot * self.candidates + candidate;
    }
};

// Without sparse rows the single-object head reads two terminator rows; the grouped head reads none.
fn promptRows(grouped: bool, sparse_rows: ?usize) usize {
    if (sparse_rows) |rows| return rows;
    return if (grouped) 0 else 2;
}

fn enabled(grouped: bool, active: u16, slot: usize) bool {
    return !grouped or (slot < 16 and (active >> @intCast(slot)) & 1 == 1);
}

fn upscales(grouped: bool, active: u16, presence: []const f32) bool {
    for (presence, 0..) |logit, s| if (enabled(grouped, active, s) and outcome.present(logit)) return true;
    return false;
}

fn checkLevels(levels: [3]FeatureMap, side: usize, width: usize) !void {
    for (levels, 0..) |level, i| {
        const expected = side >> @intCast(i);
        if (level.height != expected or level.width != expected or level.channels != width or
            level.tensor.count != expected * expected * width) return error.SamMaskHeadShape;
    }
}

fn learnedRows(rt: *Runtime, head: Scope, grouped: bool, map: RowMap, active: u16) !Tensor {
    const cached = active == 1;
    const name = if (cached) try rt.print("mask_head.rows.{s}.{d}", .{ head.path, active }) else "";
    if (cached) if (rt.constant(name)) |rows| return rows;
    const width = rt.config.hidden;
    const n = map.slots;
    const k = map.candidates;
    const presence = try head.param("obj_score_token.weight");
    const quality = try head.param("iou_token.weight");
    const candidates = try head.param("mask_tokens.weight");
    if (presence.count != n * width or quality.count != n * width or candidates.count != n * k * width)
        return error.SamMaskHeadShape;
    var parts: [2 + max_slots]Tensor = undefined;
    parts[0] = presence;
    parts[1] = quality;
    var count: usize = 2;
    if (grouped) {
        const tracker = Scope.root(rt, rt.config.roots.tracker);
        const occupied = try tracker.param("output_valid_embed");
        const unoccupied = try tracker.param("output_invalid_embed");
        if (occupied.count != n * width or unoccupied.count != n * width) return error.SamMaskHeadShape;
        for (0..n) |s| {
            const vector = runtime.part(if (enabled(true, active, s)) occupied else unoccupied, s * width, width);
            parts[count] = try elementwise.add(rt, runtime.part(candidates, s * k * width, k * width), vector);
            count += 1;
        }
    } else {
        parts[2] = candidates;
        count = 3;
    }
    const rows = try elementwise.concat(rt, parts[0..count]);
    if (!cached) return rows;
    return rt.keepConstant(name, rows);
}

fn exchange(rt: *Runtime, transformer: Scope, t0: Tensor, x0: Tensor, position: Tensor) ![2]Tensor {
    const add = elementwise.add;
    var t = t0;
    var x = x0;
    for (0..2) |b| {
        const mark = rt.mark();
        const layer = try transformer.item("layers", b);
        const self_attention = try layer.sub("self_attn");
        if (b == 0) {
            const attended = try attention.separateHeads(self_attention, t, t, t, exchange_heads);
            t = try norms.layerNorm(layer, "norm1", attended, exchange_eps, .plain);
        } else {
            const coded = try add(rt, t, t0);
            const attended = try attention.separateHeads(self_attention, coded, coded, t, exchange_heads);
            t = try norms.layerNorm(layer, "norm1", try add(rt, t, attended), exchange_eps, .plain);
        }
        const image_code = try add(rt, x, position);
        const to_image = try layer.sub("cross_attn_token_to_image");
        const gathered = try attention.separateHeads(to_image, try add(rt, t, t0), image_code, x, exchange_heads);
        t = try norms.layerNorm(layer, "norm2", try add(rt, t, gathered), exchange_eps, .plain);
        const hidden = try elementwise.relu(rt, try projection.affine(layer, "mlp.lin1", t));
        const mixed = try projection.affine(layer, "mlp.lin2", hidden);
        t = try norms.layerNorm(layer, "norm3", try add(rt, t, mixed), exchange_eps, .plain);
        const to_token = try layer.sub("cross_attn_image_to_token");
        const spread = try attention.separateHeads(to_token, image_code, try add(rt, t, t0), t, exchange_heads);
        x = try norms.layerNorm(layer, "norm4", try add(rt, x, spread), exchange_eps, .plain);
        rt.rewind(mark, &.{ t, x });
    }
    const final = try transformer.sub("final_attn_token_to_image");
    const attended = try attention.separateHeads(final, try add(rt, t, t0), try add(rt, x, position), x, exchange_heads);
    t = try norms.layerNorm(transformer, "norm_final_attn", try add(rt, t, attended), exchange_eps, .plain);
    return .{ t, x };
}

fn upscale(rt: *Runtime, head: Scope, x: Tensor, levels: [3]FeatureMap) !FeatureMap {
    const grid = rt.config.featureSide();
    var high = try convolution.upsample2(head, "output_upscaling.0", FeatureMap.of(x, grid, grid, rt.config.hidden));
    const skip_high = try convolution.conv(head, "conv_s1", levels[1], .{});
    if (skip_high.tensor.count != high.tensor.count) return error.SamMaskHeadShape;
    const joined = try elementwise.add(rt, high.tensor, skip_high.tensor);
    high.tensor = try elementwise.gelu(rt, try norms.layerNorm(head, "output_upscaling.1", joined, upscale_eps, .plain));
    var full = try convolution.upsample2(head, "output_upscaling.3", high);
    const skip_full = try convolution.conv(head, "conv_s0", levels[0], .{});
    if (skip_full.tensor.count != full.tensor.count) return error.SamMaskHeadShape;
    full.tensor = try elementwise.gelu(rt, try elementwise.add(rt, full.tensor, skip_full.tensor));
    return full;
}

fn maskLogits(rt: *Runtime, head: Scope, basis: FeatureMap, token: Tensor, candidate: usize) !Tensor {
    const leaf = try rt.print("output_hypernetworks_mlps.{d}", .{candidate});
    const weights = try projection.reluChain(head, leaf, token, chain_depth);
    if (weights.count != basis.channels) return error.SamMaskHeadShape;
    return projection.matmul(rt, basis.tensor, weights, basis.height * basis.width, 1, basis.channels);
}

// The absent-object pointer: a learned map on SAM3.1, a learned vector on SAM3; always a new held tensor.
pub fn absentPointer(rt: *Runtime, pointer: Tensor) !Tensor {
    const tracker = Scope.root(rt, rt.config.roots.tracker);
    const replaced = switch (rt.config.version) {
        .sam31 => try projection.affine(tracker, "no_obj_ptr_linear", pointer),
        .sam3 => try elementwise.concat(rt, &.{try tracker.param("no_obj_ptr")}),
    };
    if (replaced.count != rt.config.hidden) return error.SamMaskHeadShape;
    return replaced;
}

fn pointerOf(rt: *Runtime, grouped: bool, token: Tensor, is_present: bool) !Tensor {
    const tracker = Scope.root(rt, rt.config.roots.tracker);
    const interactive = rt.config.version == .sam31 and !grouped;
    const leaf = if (interactive) "interactive_obj_ptr_proj" else "obj_ptr_proj";
    const pointer = try projection.reluChain(tracker, leaf, token, chain_depth);
    if (pointer.count != rt.config.hidden) return error.SamMaskHeadShape;
    return if (is_present) pointer else absentPointer(rt, pointer);
}

pub fn decode(rt: *Runtime, levels: [3]FeatureMap, image: Tensor, request: Request) !Outcome {
    const config = rt.config;
    const width = config.hidden;
    const grid = config.featureSide();
    const side = config.maskSide();
    const grouped = request.grouped and config.version == .sam31;
    const head_path: []const u8 = if (!grouped)
        config.roots.prompt_head
    else
        (config.roots.grouped_head orelse return error.SamMaskHeadShape);
    const head = Scope.root(rt, head_path);
    const tokens_info = try head.info("obj_score_token.weight");
    const masks_info = try head.info("mask_tokens.weight");
    const n = tokens_info.dim(0);
    if (tokens_info.rank != 2 or tokens_info.dim(1) != width or n < 1 or n > max_slots) return error.SamMaskHeadShape;
    if (masks_info.rank != 2 or masks_info.dim(1) != width or masks_info.dim(0) < n or masks_info.dim(0) % n != 0)
        return error.SamMaskHeadShape;
    const k = masks_info.dim(0) / n;
    if (image.count != grid * grid * width) return error.SamMaskHeadShape;
    try checkLevels(levels, side, width);
    if (request.dense) |dense| if (dense.count != grid * grid * width) return error.SamMaskHeadShape;
    const sparse_rows: ?usize = if (request.sparse) |sparse| sparse.count / width else null;
    if (request.sparse) |sparse| if (sparse.count == 0 or sparse.count % width != 0) return error.SamMaskHeadShape;
    const map = RowMap{ .slots = n, .candidates = k, .prompt_rows = promptRows(grouped, sparse_rows) };

    const m = rt.mark();
    errdefer rt.rewind(m, &.{});
    const learned = try learnedRows(rt, head, grouped, map, request.active);
    const encoder = Scope.root(rt, config.roots.prompt_encoder);
    const tracker = Scope.root(rt, config.roots.tracker);
    const matrix_key = if (grouped)
        try tracker.key("image_pe_layer.positional_encoding_gaussian_matrix")
    else
        try encoder.key("pe_layer.positional_encoding_gaussian_matrix");
    const position = try encodings.fourierGrid(rt, matrix_key);
    const no_mask: ?Tensor = if (!grouped and request.dense == null) try encoder.param("no_mask_embed.weight") else null;
    const terminator: ?Tensor = if (!grouped and request.sparse == null) try encoder.param("not_a_point_embed.weight") else null;
    if (no_mask) |vector| if (vector.count != width) return error.SamMaskHeadShape;
    if (terminator) |row| if (row.count != width) return error.SamMaskHeadShape;
    try rt.observe("decoder.image", image, &.{ 1, grid, grid, width });
    if (no_mask) |vector| try rt.observe("prompt.no_mask", vector, &.{ 1, width });
    try rt.observe("decoder.position", position, &.{ 1, grid, grid, width });

    try rt.gpu.beginBatch();
    errdefer if (rt.gpu.batch != null) rt.gpu.endBatch() catch {};
    const t0 = if (request.sparse) |sparse|
        try elementwise.concat(rt, &.{ learned, sparse })
    else if (terminator) |row|
        try elementwise.concat(rt, &.{ learned, row, row })
    else
        learned;
    if (t0.count != map.total() * width) return error.SamMaskHeadShape;
    const x0 = if (request.dense) |dense|
        try elementwise.add(rt, image, dense)
    else if (no_mask) |vector|
        try elementwise.add(rt, image, vector)
    else
        image;
    const state = try exchange(rt, try head.sub("transformer"), t0, x0, position);
    const t = state[0];
    const presence_rows = runtime.part(t, map.presenceRow(0) * width, n * width);
    const quality_rows = runtime.part(t, map.qualityRow(0) * width, n * width);
    const presence_logits = try projection.reluChain(head, "pred_obj_score_head", presence_rows, chain_depth);
    var quality = try projection.reluChain(head, "iou_prediction_head", quality_rows, chain_depth);
    if (config.version == .sam3) quality = try elementwise.sigmoid(rt, quality);
    try rt.gpu.endBatch();
    try rt.observe("decoder.quality", quality, &.{ n, k });
    if (presence_logits.count != n or quality.count != n * k) return error.SamMaskHeadShape;

    var result = Outcome{
        .mask_logits = undefined,
        .pointers = undefined,
        .slots = n,
        .presence = @splat(0),
        .quality = @splat(0),
    };
    var picked: [max_slots]usize = undefined;
    var missing_mask = false;
    var missing_pointer = false;
    for (0..n) |s| {
        const row = quality.slice(f32, n * k)[s * k ..][0..k];
        picked[s] = outcome.pick(row, request.pick);
        result.presence[s] = presence_logits.get(s);
        result.quality[s] = row[picked[s]];
        const on = enabled(grouped, request.active, s);
        missing_mask = missing_mask or !(on and outcome.present(result.presence[s]));
        missing_pointer = missing_pointer or !on;
    }
    const fill_mask: ?Tensor = if (missing_mask) try elementwise.hostFill(rt, side * side, outcome.absent_logit) else null;
    const fill_pointer: ?Tensor = if (missing_pointer) try elementwise.hostFill(rt, width, 0) else null;

    try rt.gpu.beginBatch();
    const basis: ?FeatureMap = if (upscales(grouped, request.active, result.presence[0..n]))
        try upscale(rt, head, state[1], levels)
    else
        null;
    if (basis) |b| if (b.height != side or b.width != side) return error.SamMaskHeadShape;
    var slot_logits: [max_slots]Tensor = undefined;
    var pointers: [max_slots]Tensor = undefined;
    for (0..n) |s| {
        const token = runtime.part(t, map.maskRow(s, picked[s]) * width, width);
        const on = enabled(grouped, request.active, s);
        const is_present = outcome.present(result.presence[s]);
        slot_logits[s] = if (on and is_present) try maskLogits(rt, head, basis.?, token, picked[s]) else fill_mask.?;
        pointers[s] = if (on) try pointerOf(rt, grouped, token, is_present) else fill_pointer.?;
    }
    const mask_values = if (n == 1)
        slot_logits[0]
    else
        try elementwise.transpose(rt, try elementwise.concat(rt, slot_logits[0..n]), n, side * side);
    result.mask_logits = FeatureMap.of(mask_values, side, side, n);
    result.pointers = try elementwise.concat(rt, pointers[0..n]);
    try rt.gpu.endBatch();
    var buffer: [3]Tensor = undefined;
    rt.rewind(m, result.tensors(&buffer));
    return result;
}

test "SAM mask head row map for the single-object and grouped heads" {
    const single = RowMap{ .slots = 1, .candidates = 4, .prompt_rows = promptRows(false, null) };
    try std.testing.expectEqual(@as(usize, 6), single.learned());
    try std.testing.expectEqual(@as(usize, 0), single.presenceRow(0));
    try std.testing.expectEqual(@as(usize, 1), single.qualityRow(0));
    for (0..4) |c| try std.testing.expectEqual(2 + c, single.maskRow(0, c));
    try std.testing.expectEqual(@as(usize, 8), single.total());
    try std.testing.expectEqual(@as(usize, 3), promptRows(false, 3));
    const grouped = RowMap{ .slots = 16, .candidates = 3, .prompt_rows = promptRows(true, null) };
    try std.testing.expectEqual(@as(usize, 80), grouped.learned());
    try std.testing.expectEqual(@as(usize, 80), grouped.total());
    for (0..16) |s| {
        try std.testing.expectEqual(s, grouped.presenceRow(s));
        try std.testing.expectEqual(16 + s, grouped.qualityRow(s));
        for (0..3) |c| try std.testing.expectEqual(32 + 3 * s + c, grouped.maskRow(s, c));
    }
}

test "SAM mask head upscales only when an enabled slot is present" {
    const nan = std.math.nan(f32);
    var presence: [16]f32 = @splat(-1);
    presence[0] = nan;
    presence[2] = -2;
    try std.testing.expect(!upscales(true, 0b100, &presence));
    try std.testing.expect(upscales(true, 0b101, &presence));
    presence[2] = 0.5;
    try std.testing.expect(upscales(true, 0b100, &presence));
    for ([_]u16{ 0, 1, 0b110, 0xffff }) |active| {
        try std.testing.expect(upscales(false, active, &.{0.25}));
        try std.testing.expect(!upscales(false, active, &.{-0.25}));
        try std.testing.expect(upscales(false, active, &.{nan}));
    }
}
