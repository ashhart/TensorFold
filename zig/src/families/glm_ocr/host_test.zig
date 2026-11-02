//! Checkpoint admission and independent processor fixtures; no GPU runtime is initialized.
const std = @import("std");
const st = @import("core").safetensors;
const c = @import("config.zig");
const schema = @import("schema.zig");
const checkpoint = @import("checkpoint.zig");
const processor = @import("processor.zig");
const positions = @import("positions.zig");
const a = std.testing.allocator;
const valid = @embedFile("fixtures/config.json");
const preprocessor = @embedFile("fixtures/preprocessor_config.json");

fn replace(text: []const u8, old: []const u8, new: []const u8) ![]u8 {
    const at = std.mem.indexOf(u8, text, old) orelse return error.MissingTestField;
    return std.mem.concat(a, u8, &.{ text[0..at], new, text[at + old.len ..] });
}

test "GLM-OCR rejects incompatible decoder, rotary, tower and quantization metadata" {
    try c.validate(a, valid);
    for ([_]struct { []const u8, []const u8 }{
        .{ "\"glm_ocr\"", "\"glm4v\"" },
        .{ "\"head_dim\": 128", "\"head_dim\": 96" },
        .{ "\"hidden_size\": 1536", "\"hidden_size\": 2048" },
        .{ "\"silu\"", "\"gelu\"" },
        .{ "\"attention_bias\": false", "\"attention_bias\": true" },
        .{ "\"num_nextn_predict_layers\": 1", "\"num_nextn_predict_layers\": 0" },
        .{ "\"partial_rotary_factor\": 1.0", "\"partial_rotary_factor\": 0.5" },
        .{ "\"rope_type\": \"axial\"", "\"rope_type\": \"default\"" },
        .{ "\"spatial_merge_size\": 2", "\"spatial_merge_size\": 4" },
        .{ "\"in_channels\": 3", "\"in_channels\": 1" },
        .{ "\"model_type\": \"glm_ocr\"", "\"model_type\": \"glm_ocr\", \"quantization_config\": {}" },
        .{ "59253", "59254" },
    }) |change| {
        const bad = try replace(valid, change[0], change[1]);
        defer a.free(bad);
        try std.testing.expectError(error.UnsupportedGlmOcrConfig, c.validate(a, bad));
    }
    try std.testing.expectError(error.BadGlmOcrConfig, c.validate(a, "[]"));
}

test "GLM-OCR admits legacy vision defaults without changing the graph" {
    var doc = try std.json.parseFromSlice(std.json.Value, a, valid, .{});
    defer doc.deinit();
    const vision = &doc.value.object.getPtr("vision_config").?.object;
    for ([_][]const u8{ "in_channels", "attention_dropout", "rope_parameters" }) |key|
        try std.testing.expect(vision.swapRemove(key));
    const legacy = try std.json.Stringify.valueAlloc(a, doc.value, .{});
    defer a.free(legacy);
    try c.validate(a, legacy);
}

test "GLM-OCR refuses processor arithmetic changes" {
    try c.validateProcessor(a, preprocessor);
    for ([_]struct { []const u8, []const u8 }{
        .{ "\"patch_size\": 14", "\"patch_size\": 16" },
        .{ "\"temporal_patch_size\": 2", "\"temporal_patch_size\": 1" },
        .{ "\"do_rescale\": true", "\"do_rescale\": false" },
        .{ "9633792", "10000000" },
        .{ "0.48145466", "0.5" },
        .{ "\"merge_size\": 2", "\"merge_size\": 2,\"resample\":2" },
    }) |change| {
        const bad = try replace(preprocessor, change[0], change[1]);
        defer a.free(bad);
        try std.testing.expectError(error.UnsupportedGlmOcrConfig, c.validateProcessor(a, bad));
    }
}

const Fixture = struct {
    arena: std.heap.ArenaAllocator,
    names: st.Header = .empty,
    pub fn tensor(s: *Fixture, name: []const u8, shape: []const usize) !void {
        const ar = s.arena.allocator();
        var dims: [st.max_rank]usize = @splat(1);
        @memcpy(dims[0..shape.len], shape);
        try s.names.put(ar, try ar.dupe(u8, name), .{ .dtype = .bf16, .rank = @intCast(shape.len), .shape = dims, .begin = 0, .end = 0 });
    }
};

test "GLM-OCR validates every tensor including biased patch projections and MTP" {
    var f: Fixture = .{ .arena = .init(a) };
    defer f.arena.deinit();
    try schema.each(&f);
    try std.testing.expectEqual(@as(usize, 526), f.names.count());
    try schema.validate(&f.names);
    for ([_][]const u8{ schema.visual ++ "downsample.weight", schema.mtp ++ "eh_proj.weight", schema.text ++ "layers.0.self_attn.o_proj.weight" }) |name| {
        const original = f.names.get(name).?;
        f.names.getPtr(name).?.shape[1] += 1;
        try std.testing.expectError(error.BadGlmOcrTensor, schema.validate(&f.names));
        f.names.getPtr(name).?.* = original;
        f.names.getPtr(name).?.dtype = .f16;
        try std.testing.expectError(error.BadGlmOcrTensor, schema.validate(&f.names));
        f.names.getPtr(name).?.* = original;
    }
    try f.tensor("unused.weight", &.{1});
    try std.testing.expectError(error.UnexpectedGlmOcrTensor, schema.validate(&f.names));
    _ = f.names.swapRemove("unused.weight");
    _ = f.names.swapRemove(schema.visual ++ "patch_embed.proj.bias");
    try std.testing.expectError(error.MissingGlmOcrTensor, schema.validate(&f.names));
}

test "GLM-OCR opens the downloaded checkpoint read-only without a GPU" {
    const dir = @import("glm_ocr_options").model_dir orelse return error.SkipZigTest;
    var ck = try checkpoint.Checkpoint.open(a, std.testing.io, dir);
    defer ck.close();
    try std.testing.expectEqual(@as(usize, 526), ck.file.names.count());
    try std.testing.expect((try ck.get(schema.visual ++ "patch_embed.proj.weight")).is(.bf16, &.{ 1024, 3, 2, 14, 14 }));
    try std.testing.expect((try ck.get("lm_head.weight")).is(.bf16, &.{ c.vocab, c.hidden }));
}

const Reference = struct {
    cases: []const struct { height: usize, width: usize, grid: [3]usize, tokens: usize, sha256: []const u8 },
    geometry: []const struct { height: usize, width: usize, h: usize, w: usize },
    positions: struct { ids: []const u32, grids: []const [3]usize, rows: []const [3]i32, delta: i32 },
};

test "GLM-OCR pixels and merge order match independent Transformers fingerprints" {
    const doc = try std.json.parseFromSlice(Reference, a, @embedFile("fixtures/processor.json"), .{ .ignore_unknown_fields = true });
    defer doc.deinit();
    for (doc.value.cases) |cell| {
        const pixels = try a.alloc(u8, cell.height * cell.width * 3);
        defer a.free(pixels);
        for (0..cell.height) |y| for (0..cell.width) |x| for (0..3) |ch| {
            pixels[(y * cell.width + x) * 3 + ch] = @truncate(y * 17 + x * 29 + ch * 83);
        };
        var p = try processor.prepareRgb(a, .{ .height = cell.height, .width = cell.width, .pixels = pixels }, 6144);
        defer p.deinit(a);
        try std.testing.expectEqual(cell.grid[1] * 14, p.geometry.h);
        try std.testing.expectEqual(cell.grid[2] * 14, p.geometry.w);
        try std.testing.expectEqual(cell.tokens, p.geometry.tokens());
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(p.pixels), &digest, .{});
        try std.testing.expectEqualStrings(cell.sha256, &std.fmt.bytesToHex(digest, .lower));
    }
    for (doc.value.geometry) |cell| {
        try std.testing.expectEqualDeep(processor.Geometry{ .h = cell.h, .w = cell.w }, try processor.geometry(cell.height, cell.width));
    }
}

test "GLM-OCR image positions match Transformers across multiple non-square images" {
    const doc = try std.json.parseFromSlice(Reference, a, @embedFile("fixtures/processor.json"), .{ .ignore_unknown_fields = true });
    defer doc.deinit();
    const f = doc.value.positions;
    const rows = try a.alloc([3]i32, f.ids.len);
    defer a.free(rows);
    const images = try a.alloc(processor.Geometry, f.grids.len);
    defer a.free(images);
    for (images, f.grids) |*g, grid| g.* = .{ .h = grid[1] * 14, .w = grid[2] * 14 };
    const result = try positions.fill(f.ids, images, rows);
    try std.testing.expectEqual(f.delta, result.delta);
    try std.testing.expectEqualDeep(f.rows, rows);
    try std.testing.expectError(error.BadImageSpan, positions.fill(f.ids, images[0..1], rows));
    const corrupt = try a.dupe(u32, f.ids);
    defer a.free(corrupt);
    corrupt[3] = 7;
    try std.testing.expectError(error.BadImageSpan, positions.fill(corrupt, images, rows));
}

test "GLM-OCR refuses invalid geometry, pixel buffers and budgets before allocation" {
    try std.testing.expectError(error.Invalid, processor.geometry(0, 10));
    try std.testing.expectError(error.Oversized, processor.geometry(std.math.maxInt(usize), 10));
    try std.testing.expectError(error.AspectRatio, processor.geometry(28, 5628));
    const g = try processor.geometry(84, 112);
    try g.fits(12, c.context - 12, c.context);
    try std.testing.expectError(error.ContextExceeded, g.fits(12, c.context - 11, c.context));
    try std.testing.expectError(error.ImageBudget, g.fits(11, 0, c.context));
    try std.testing.expectError(error.BadPixels, processor.prepareRgb(a, .{ .height = 84, .width = 112, .pixels = &.{} }, 6144));
    try std.testing.expectError(error.ImageBudget, processor.prepareRgb(a, .{ .height = 84, .width = 112, .pixels = &.{} }, 1));
}
