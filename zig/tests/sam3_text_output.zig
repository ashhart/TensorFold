const std = @import("std");
const sam = @import("sam3");
const fixture = @import("checkpoint_fixture");

/// Framed ids (I64 `[n]`) and text features (F32 `[n, 256]`) of one text, as `token_ids` and `language_features`.
pub fn writeFeatures(a: std.mem.Allocator, io: std.Io, model: *sam.Model, text: []const u8, path: []const u8) !void {
    if (model.tokenizer == null) return error.TokenizerNotLoaded;
    const ids = try sam.text_tower.frame(a, &model.tokenizer.?, text);
    var rt = model.runtime(a);
    defer rt.deinit();
    const features = try sam.text_tower.encode(&rt, ids, .plain);
    if (features.count != ids.len * model.config.hidden) return error.TextFeatureShape;
    const wide = try a.alloc(i64, ids.len);
    for (wide, ids) |*out, id| out.* = id;
    try fixture.write(a, io, .cwd(), path, &.{
        .{ .name = "token_ids", .dtype = "I64", .shape = &.{ids.len}, .bytes = std.mem.sliceAsBytes(wide) },
        .{ .name = "language_features", .dtype = "F32", .shape = &.{ ids.len, model.config.hidden }, .bytes = features.slice(u8, features.count * 4) },
    });
    std.debug.print("token ids {any}\n", .{ids});
}

/// Detections in output order: `masks` U8 `[k, h, w]`, `scores` F32 `[k]`, `boxes_xyxy` F32 `[k, 4]` in source pixels.
pub fn writeDetections(a: std.mem.Allocator, io: std.Io, prediction: sam.Prediction, size: sam.prompts.Size, path: []const u8) !void {
    const k = prediction.masks.len;
    const pixels = try size.pixels();
    const masks = try a.alloc(u8, k * pixels);
    const scores = try a.alloc(f32, k);
    const boxes = try a.alloc(f32, k * 4);
    for (prediction.masks, 0..) |m, d| {
        @memcpy(masks[d * pixels ..][0..pixels], m.pixels);
        scores[d] = m.score;
        @memcpy(boxes[d * 4 ..][0..4], &m.box);
    }
    try fixture.write(a, io, .cwd(), path, &.{
        .{ .name = "masks", .shape = &.{ k, size.height, size.width }, .bytes = masks },
        .{ .name = "scores", .dtype = "F32", .shape = &.{k}, .bytes = std.mem.sliceAsBytes(scores) },
        .{ .name = "boxes_xyxy", .dtype = "F32", .shape = &.{ k, 4 }, .bytes = std.mem.sliceAsBytes(boxes) },
    });
}
