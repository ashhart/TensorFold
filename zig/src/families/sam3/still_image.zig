const std = @import("std");
const runtime = @import("runtime.zig");
const variant = @import("variant.zig");
const prompts = @import("prompts.zig");
const mask = @import("mask.zig");
const engine = @import("engine.zig");
const elementwise = @import("elementwise.zig");
const spatial = @import("spatial.zig");
const image_tower = @import("image_tower.zig");
const text_tower = @import("text_tower.zig");
const concept_readout = @import("concept_readout.zig");
const prompt_tokens = @import("prompt_tokens.zig");
const mask_head = @import("mask_head.zig");
const outcome = @import("outcome.zig");

const Model = engine.Model;
const Runtime = runtime.Runtime;
const Version = variant.Version;
const DetectOptions = concept_readout.DetectOptions;

/// Masks at source size, owned by `allocator`.
pub const Prediction = struct {
    allocator: std.mem.Allocator,
    masks: []mask.Mask,

    pub fn deinit(self: *Prediction) void {
        for (self.masks) |m| self.allocator.free(m.pixels);
        self.allocator.free(self.masks);
        self.* = undefined;
    }
};

/// Concept prediction from text; the tokenizer and the options are checked before the text is framed.
pub fn predictText(model: *Model, allocator: std.mem.Allocator, image: prompts.Image, text: []const u8, options: DetectOptions) !Prediction {
    if (model.tokenizer == null) return error.TokenizerNotLoaded;
    try options.validate();
    const ids = try text_tower.frame(allocator, &model.tokenizer.?, text);
    defer allocator.free(ids);
    return predictTokens(model, allocator, image, ids, options);
}

/// Concept prediction from framed token ids.
pub fn predictTokens(model: *Model, allocator: std.mem.Allocator, image: prompts.Image, tokens: []const u32, options: DetectOptions) !Prediction {
    try options.validate();
    try model.mutex.lock(model.io);
    defer model.mutex.unlock(model.io);
    var arena = std.heap.ArenaAllocator.init(model.allocator);
    defer arena.deinit();
    var rt = model.runtime(arena.allocator());
    defer rt.deinit();
    const trunk = try image_tower.encodeImage(&rt, image);
    const levels = try image_tower.levels(&rt, trunk, .detector);
    const text_features = try text_tower.encode(&rt, tokens, .plain);
    const detections = try concept_readout.detect(&rt, levels, text_features, options);
    const masks = try allocator.alloc(mask.Mask, detections.len);
    var made: usize = 0;
    errdefer {
        for (masks[0..made]) |m| allocator.free(m.pixels);
        allocator.free(masks);
    }
    for (detections) |d| {
        var found = try spatial.sourceMask(&rt, allocator, d.mask, image.size, d.score);
        found.box = concept_readout.scaleBox(d.box, image.size.width, image.size.height);
        masks[made] = found;
        made += 1;
    }
    return .{ .allocator = allocator, .masks = masks };
}

/// One mask from points, boxes or a mask prompt on the prompt level set, scored by the chosen candidate's quality.
pub fn predictVisual(model: *Model, allocator: std.mem.Allocator, image: prompts.Image, prompt: prompts.Prompt, best_alternative: bool) !Prediction {
    try image.validate();
    try prompt.validate(image.size);
    try model.mutex.lock(model.io);
    defer model.mutex.unlock(model.io);
    var arena = std.heap.ArenaAllocator.init(model.allocator);
    defer arena.deinit();
    var rt = model.runtime(arena.allocator());
    defer rt.deinit();
    const trunk = try image_tower.encodeImage(&rt, image);
    const levels = try image_tower.levels(&rt, trunk, .prompt);
    const tokens = try prompt_tokens.encode(&rt, prompt, image.size);
    const key = try rt.print("{s}.{s}", .{ rt.config.roots.tracker, noMemoryLeaf(rt.config.version) });
    const no_memory = try rt.param(key);
    if (no_memory.count != rt.config.hidden) return error.SamNoMemoryShape;
    const conditioned = try elementwise.add(&rt, levels[2].tensor, no_memory);
    const decoded = try mask_head.decode(&rt, levels, conditioned, .{
        .grouped = false,
        .active = 1,
        .pick = pickFor(best_alternative),
        .dense = tokens.dense,
        .sparse = tokens.sparse,
    });
    const masks = try allocator.alloc(mask.Mask, 1);
    errdefer allocator.free(masks);
    masks[0] = try spatial.sourceMask(&rt, allocator, decoded.mask_logits, image.size, decoded.quality[0]);
    return .{ .allocator = allocator, .masks = masks };
}

/// Candidate rule of a still-image visual prediction.
pub fn pickFor(best_alternative: bool) outcome.Pick {
    return if (best_alternative) .best_alternative else .primary;
}

/// No-memory vector added to the prompt set's coarse map, under `roots.tracker`.
pub fn noMemoryLeaf(version: Version) []const u8 {
    return switch (version) {
        .sam3 => "no_mem_embed",
        .sam31 => "interactivity_no_mem_embed",
    };
}

const testing = std.testing;

test "SAM still-image candidate rule follows the boolean argument" {
    try testing.expectEqual(outcome.Pick.best_alternative, pickFor(true));
    try testing.expectEqual(outcome.Pick.primary, pickFor(false));
}

test "SAM still-image no-memory key per version" {
    var buffer: [128]u8 = undefined;
    for ([_]Version{ .sam3, .sam31 }, [_][]const u8{ "tracker.no_mem_embed", "tracker.model.interactivity_no_mem_embed" }) |version, expected| {
        const tracker = variant.Config.of(version).roots.tracker;
        try testing.expectEqualStrings(expected, try std.fmt.bufPrint(&buffer, "{s}.{s}", .{ tracker, noMemoryLeaf(version) }));
    }
}
