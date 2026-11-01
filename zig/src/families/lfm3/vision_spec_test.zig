//! vision_spec's geometry, tensor table and image rows against the VL config fixture and a real checkpoint's index.
const std = @import("std");
const spec = @import("vision_spec.zig");
const fixtures = @import("config_test.zig");

const testing = std.testing;

test "the VL config fixture reads to the geometry config.zig admits" {
    const v = try spec.read(testing.allocator, fixtures.vl);
    try testing.expect(std.meta.eql(spec.lfm3_vl_2b, v));
    try testing.expectEqual(@as(usize, 64), v.headDim());
    try testing.expectEqual(@as(usize, 512), v.kvWidth());
    try testing.expectEqual(@as(usize, 1536), v.patchWidth());
    try testing.expectEqual(@as(usize, 4096), v.merged());
    try testing.expectEqual(@as(usize, 70), v.tokens(14, 20));
}

test "a vision config without a field, with an unknown mixer or a different merge does not read" {
    const a = testing.allocator;
    const cases = [_][2][]const u8{
        .{ "\"num_key_value_heads\": 8,", "" },
        .{ "\"conv\",\n      \"full_attention\",", "\"conv\",\n      \"sliding_attention\"," },
        .{ "\"downsample_factor\": 2", "\"downsample_factor\": 4" },
    };
    for (cases) |c| {
        const bad = try std.mem.replaceOwned(u8, a, fixtures.vl, c[0], c[1]);
        defer a.free(bad);
        try testing.expect(!std.mem.eql(u8, bad, fixtures.vl));
        try testing.expectError(error.BadLfm3VisionConfig, spec.read(a, bad));
    }
}

test "the table holds LFM3-VL-2B's 138 bf16 tensors, each name once, at the loaders' shapes" {
    const a = testing.allocator;
    const v = spec.lfm3_vl_2b;
    const t = try spec.table(a, v);
    defer spec.freeTable(a, t);
    try testing.expectEqual(@as(usize, 138), t.len);
    try testing.expectEqual(spec.count(v), t.len);
    for (t, 0..) |x, i| for (t[i + 1 ..]) |y| try testing.expect(!std.mem.eql(u8, x.name, y.name));
    const want = [_]struct { []const u8, []const usize }{
        .{ "vision_tower.patchifier.weight", &.{ 1024, 1536 } },
        .{ "vision_tower.embedding_norm.weight", &.{1024} },
        .{ "vision_tower.layers.0.conv.in_proj.weight", &.{ 3072, 1024 } },
        .{ "vision_tower.layers.1.conv.spatial_conv.weight", &.{ 1024, 3, 3, 1 } },
        .{ "vision_tower.layers.2.self_attn.k_proj.weight", &.{ 512, 1024 } },
        .{ "vision_tower.layers.12.self_attn.q_layernorm.weight", &.{64} },
        .{ "vision_tower.layers.13.feed_forward.w2.weight", &.{ 1024, 2560 } },
        .{ "multi_modal_projector.linear_1.weight", &.{ 4096, 4096 } },
        .{ "multi_modal_projector.linear_2.weight", &.{ 2048, 4096 } },
        .{ "multi_modal_projector.linear_2.bias", &.{2048} },
    };
    for (want) |w| {
        const x = for (t) |x| {
            if (std.mem.eql(u8, x.name, w[0])) break x;
        } else return error.TestMissingTensor;
        try testing.expectEqualSlices(usize, w[1], x.shape.slice());
        try testing.expectEqual(spec.Dtype.bf16, x.dtype);
    }
    var buf: [128]u8 = undefined;
    try testing.expectError(error.BadLfm3VisionModule, spec.tensor(&buf, v, 1, .q_proj));
    try testing.expectError(error.BadLfm3VisionModule, spec.tensor(&buf, v, 2, .conv_in));
    try testing.expectError(error.BadLfm3VisionModule, spec.tensor(&buf, v, null, .w1));
    try testing.expectError(error.BadLfm3VisionModule, spec.tensor(&buf, v, 14, .w1));
}

/// The table's names, one renamed when `plant` is set, plus a decoder key that `check` must pass by.
fn keysOf(a: std.mem.Allocator, t: []const spec.Tensor, plant: ?usize) ![][]const u8 {
    const keys = try a.alloc([]const u8, t.len + 1);
    for (t, keys[0..t.len], 0..) |x, *k, i| k.* = if (plant == i) "vision_tower.layers.3.conv.depthwise.weight" else x.name;
    keys[t.len] = "language_model.model.embed_tokens.weight";
    return keys;
}

test "planted: a renamed, missing or extra vision tensor fails the check" {
    const a = testing.allocator;
    const t = try spec.table(a, spec.lfm3_vl_2b);
    defer spec.freeTable(a, t);
    const keys = try keysOf(a, t, null);
    defer a.free(keys);
    try spec.check(t, keys);
    const renamed = try keysOf(a, t, 20);
    defer a.free(renamed);
    try testing.expectError(error.UnexpectedLfm3VisionTensor, spec.check(t, renamed));
    try testing.expectError(error.MissingLfm3VisionTensor, spec.check(t, keys[1..]));
    const extra = try a.alloc([]const u8, keys.len + 1);
    defer a.free(extra);
    @memcpy(extra[0..keys.len], keys);
    extra[keys.len] = "multi_modal_projector.linear_3.weight";
    try testing.expectError(error.UnexpectedLfm3VisionTensor, spec.check(t, extra));
}

/// A real export's directory (config.json and model.safetensors.index.json), e.g. LiquidAI/LFM3-VL-2B-MLX-4bit.
fn realDir() ?[]const u8 {
    const dir = std.c.getenv("TF_LFM3_VL_DIR") orelse return null;
    return std.mem.span(dir);
}

fn readFile(a: std.mem.Allocator, dir: []const u8, file: []const u8) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, file });
    defer a.free(path);
    return std.Io.Dir.cwd().readFileAlloc(testing.io, path, a, .limited(1 << 26));
}

test "the real index (TF_LFM3_VL_DIR) holds the table's names exactly, and a planted rename fails" {
    const dir = realDir() orelse return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v = try spec.read(a, try readFile(a, dir, "config.json"));
    const index = try std.json.parseFromSliceLeaky(std.json.Value, a, try readFile(a, dir, "model.safetensors.index.json"), .{});
    const map = index.object.get("weight_map") orelse return error.TestBadIndex;
    if (map != .object) return error.TestBadIndex;
    const keys = try a.alloc([]const u8, map.object.count());
    var it = map.object.iterator();
    var vision: usize = 0;
    for (keys) |*k| {
        k.* = it.next().?.key_ptr.*;
        vision += @intFromBool(spec.isVision(k.*));
    }
    const t = try spec.table(a, v);
    try testing.expectEqual(t.len, vision);
    try spec.check(t, keys);
    for (keys) |*k| if (spec.isVision(k.*)) {
        k.* = "vision_tower.layers.3.conv.depthwise.weight";
        break;
    };
    try testing.expectError(error.UnexpectedLfm3VisionTensor, spec.check(t, keys));
}

test "image rows: a span's part of each prompt chunk, as image_rows.zig copies it" {
    const s: spec.Span = .{ .at = 10, .rows = 70, .source = 5 };
    try testing.expectEqual(@as(?spec.Copy, null), spec.overlap(s, 0, 10));
    try testing.expectEqual(@as(?spec.Copy, null), spec.overlap(s, 80, 16));
    try testing.expectEqual(spec.Copy{ .source = 5, .row = 6, .rows = 26 }, spec.overlap(s, 4, 32).?);
    try testing.expectEqual(spec.Copy{ .source = 31, .row = 0, .rows = 32 }, spec.overlap(s, 36, 32).?);
    try testing.expectEqual(spec.Copy{ .source = 63, .row = 0, .rows = 12 }, spec.overlap(s, 68, 32).?);
    var total: usize = 0;
    var at: usize = 0;
    while (at < 100) : (at += 32) total += if (spec.overlap(s, at, 32)) |c| c.rows else 0;
    try testing.expectEqual(@as(usize, 70), total);
}

test "image rows: the pictures a chunk reads, packed in order, within the budget" {
    const Img = struct { at: u32, tokens: u32 };
    const imgs = [_]Img{ .{ .at = 2, .tokens = 64 }, .{ .at = 70, .tokens = 70 }, .{ .at = 150, .tokens = 100 } };
    var out: [4]spec.Span = undefined;
    const all = try spec.place(&out, &imgs, 0, 300, 256);
    try testing.expectEqualSlices(spec.Span, &.{ .{ .at = 2, .rows = 64, .source = 0 }, .{ .at = 70, .rows = 70, .source = 64 }, .{ .at = 150, .rows = 100, .source = 134 } }, all);
    const later = try spec.place(&out, &imgs, 66, 150, 256);
    try testing.expectEqualSlices(spec.Span, &.{.{ .at = 70, .rows = 70, .source = 0 }}, later);
    try testing.expectError(error.ImageTooLarge, spec.place(&out, &imgs, 0, 300, 200));
    try testing.expectError(error.TooManyImages, spec.place(out[0..2], &imgs, 0, 300, 256));
}
