const std = @import("std");
const prompts = @import("prompts.zig");
const runtime = @import("runtime.zig");
const elementwise = @import("elementwise.zig");
const spatial = @import("spatial.zig");
const prompt_tokens = @import("prompt_tokens.zig");
const mask_head = @import("mask_head.zig");
const outcome = @import("outcome.zig");
const memory_bank = @import("memory_bank.zig");
const recall_plan = @import("recall_plan.zig");
const memory_reader = @import("memory_reader.zig");

const Runtime = runtime.Runtime;
const Tensor = runtime.Tensor;
const FeatureMap = runtime.FeatureMap;
const Scope = runtime.Scope;
const Outcome = outcome.Outcome;
const Entry = memory_bank.Entry;
const Direction = recall_plan.Direction;

const passthrough_gain: f32 = 20;
const passthrough_offset: f32 = -10;
const passthrough_presence: f32 = 10;
const passthrough_quality: f32 = 1;

// One point alone picks the best alternative candidate; every other prompt takes the primary candidate.
fn promptPick(prompt: prompts.Prompt) outcome.Pick {
    const one_point = prompt.points.len == 1 and prompt.boxes.len == 0 and prompt.mask == null;
    return if (one_point) .best_alternative else .primary;
}

fn passthroughMask(prompt: prompts.Prompt) ?prompts.Mask {
    const mask = prompt.mask orelse return null;
    if (mask.format != .probabilities or prompt.points.len != 0 or prompt.boxes.len != 0) return null;
    return mask;
}

// Writes 20p - 10 per pixel (two roundings) and reports whether any probability is positive.
fn passthroughLogits(probabilities: []const f32, out: []f32) bool {
    var any = false;
    for (out, probabilities) |*logit, p| {
        logit.* = passthrough_gain * p + passthrough_offset;
        any = any or p > 0;
    }
    return any;
}

fn passThrough(rt: *Runtime, head: Outcome, mask: prompts.Mask) !Outcome {
    const height = mask.size.height;
    const width = mask.size.width;
    const side = rt.config.maskSide();
    if (mask.values.len != height * width) return error.SamPromptShape;
    const full = try rt.hostTensor(height * width);
    const any = passthroughLogits(mask.values, full.slice(f32, height * width));
    const full_logits = FeatureMap.of(full, height, width, 1);
    var result = head;
    result.full_logits = full_logits;
    result.mask_logits = try spatial.resample(rt, full_logits, side, side, .bilinear);
    result.presence[0] = if (any) passthrough_presence else -passthrough_presence;
    result.quality[0] = passthrough_quality;
    if (!any) result.pointers = try mask_head.absentPointer(rt, head.pointers);
    return result;
}

pub fn prompted(rt: *Runtime, levels: [3]FeatureMap, prompt: prompts.Prompt, size: prompts.Size) !Outcome {
    const m = rt.mark();
    errdefer rt.rewind(m, &.{});
    const tokens = try prompt_tokens.encode(rt, prompt, size);
    const tracker = Scope.root(rt, rt.config.roots.tracker);
    const no_memory = try tracker.param(if (rt.config.version == .sam31) "interactivity_no_mem_embed" else "no_mem_embed");
    if (no_memory.count != rt.config.hidden) return error.SamPromptShape;
    const image = try elementwise.add(rt, levels[2].tensor, no_memory);
    var result = try mask_head.decode(rt, levels, image, .{
        .grouped = false,
        .active = 1,
        .pick = promptPick(prompt),
        .dense = tokens.dense,
        .sparse = tokens.sparse,
    });
    if (passthroughMask(prompt)) |mask| result = try passThrough(rt, result, mask);
    var buffer: [3]Tensor = undefined;
    rt.rewind(m, result.tensors(&buffer));
    return result;
}

fn observePlan(rt: *Runtime, entries: []const Entry, recall: *const recall_plan.Plan) !void {
    if (rt.observer == null) return;
    const grid = recall.gridSlots();
    const spatial_plan = try rt.hostTensor(grid.len * 2);
    const grid_values = spatial_plan.slice(f32, grid.len * 2);
    for (grid, 0..) |slot, i| {
        grid_values[2 * i] = @floatFromInt(entries[slot.entry].frame);
        grid_values[2 * i + 1] = @floatFromInt(slot.time_slot);
    }
    try rt.observe("tracker.spatial_plan", spatial_plan, &.{ grid.len, 2 });
    const pointers = recall.pointerSlots();
    const pointer_plan = try rt.hostTensor(pointers.len * 2);
    const pointer_values = pointer_plan.slice(f32, pointers.len * 2);
    for (pointers, 0..) |slot, i| {
        pointer_values[2 * i] = @floatFromInt(entries[slot.entry].frame);
        pointer_values[2 * i + 1] = @floatFromInt(slot.offset);
    }
    try rt.observe("tracker.pointer_plan", pointer_plan, &.{ pointers.len, 2 });
}

pub fn tracked(rt: *Runtime, levels: [3]FeatureMap, entries: []const Entry, frame: usize, frames: usize, direction: Direction, active: u16) !Outcome {
    const grouped = rt.config.version == .sam31;
    const recall = try recall_plan.plan(entries, frame, frames, direction, grouped);
    if (recall.grid_len == 0 or recall.pointer_len == 0) return error.MissingPromptedFrame;
    const m = rt.mark();
    errdefer rt.rewind(m, &.{});
    try observePlan(rt, entries, &recall);
    const conditioned = try memory_reader.condition(rt, levels[2].tensor, entries, &recall, frames);
    const request: mask_head.Request = if (grouped)
        .{ .grouped = true, .active = active, .pick = .best_any }
    else
        .{ .grouped = false, .pick = .best_alternative };
    const result = try mask_head.decode(rt, levels, conditioned, request);
    var buffer: [3]Tensor = undefined;
    rt.rewind(m, result.tensors(&buffer));
    return result;
}

test "SAM prompted step picks the best alternative only for one point alone" {
    const point = prompts.Point{ .x = 1, .y = 1 };
    const box = prompts.Box{ .x0 = 0, .y0 = 0, .x1 = 2, .y1 = 2 };
    const mask = prompts.Mask{ .size = .{ .width = 1, .height = 1 }, .values = &.{1} };
    try std.testing.expectEqual(outcome.Pick.best_alternative, promptPick(.{ .points = &.{point} }));
    try std.testing.expectEqual(outcome.Pick.primary, promptPick(.{ .points = &.{ point, point } }));
    try std.testing.expectEqual(outcome.Pick.primary, promptPick(.{ .boxes = &.{box} }));
    try std.testing.expectEqual(outcome.Pick.primary, promptPick(.{ .points = &.{point}, .boxes = &.{box} }));
    try std.testing.expectEqual(outcome.Pick.primary, promptPick(.{ .points = &.{point}, .mask = mask }));
    try std.testing.expectEqual(outcome.Pick.primary, promptPick(.{ .mask = mask }));
}

test "SAM mask passthrough applies only to a probability mask without points or boxes" {
    const point = prompts.Point{ .x = 1, .y = 1 };
    const mask = prompts.Mask{ .size = .{ .width = 1, .height = 1 }, .values = &.{1} };
    const logits = prompts.Mask{ .size = .{ .width = 1, .height = 1 }, .values = &.{1}, .format = .logits };
    try std.testing.expect(passthroughMask(.{ .mask = mask }) != null);
    try std.testing.expect(passthroughMask(.{ .mask = logits }) == null);
    try std.testing.expect(passthroughMask(.{ .mask = mask, .points = &.{point} }) == null);
    try std.testing.expect(passthroughMask(.{ .points = &.{point} }) == null);
    var out: [3]f32 = undefined;
    try std.testing.expect(passthroughLogits(&.{ 0, 0.5, 1 }, &out));
    try std.testing.expectEqual([3]f32{ -10, 0, 10 }, out);
    try std.testing.expect(!passthroughLogits(&.{ 0, 0, 0 }, &out));
}
