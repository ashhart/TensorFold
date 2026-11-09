//! dots.mocr's text decoder on CUDA: the dots recipe behind the shared dense engine (native/cuda.zig's family decls).
const std = @import("std");
const dense = @import("cuda_dense");
const cfg = @import("config.zig");
const fixtures = @import("config_test.zig");

pub const recipe = @import("cuda_recipe.zig");
pub const weights = @import("cuda_weights.zig");
pub const tower_weights = @import("cuda_tower_weights.zig");
pub const kernels = @import("cuda_kernels.zig");
pub const native = dense.Native(recipe);

test {
    std.testing.refAllDecls(tower_weights);
}

test "dots geometry from the released config: 12 query heads over 2 KV heads of 128, two query heads a pattn block" {
    const c = try cfg.Config.parse(std.testing.allocator, fixtures.sample, fixtures.generation);
    const g = recipe.geometryOf(c);
    var plain = g;
    plain.inv_freq = null;
    try std.testing.expectEqual(dense.Geometry{ .hidden = 1536, .layers = 28, .heads = 12, .kv_heads = 2, .head_dim = 128, .vocab = 151936, .eps = 1e-6, .context = 131072, .acc = 2048, .wide = 17920, .mid = 8960, .theta = 1e6 }, plain);
    try std.testing.expectEqual(@as(usize, 64), g.inv_freq.?.len); // Transformers' own table (cuda_recipe.zig)
    const shape = try dense.recipe.attentionShape(g);
    try std.testing.expect(shape.hpc == 2 and shape.grid_y == 6 and shape.rows == 64);
}

test "dots text bytes: embedding, untied head, final norm and 28 layers, the tower left out" {
    const c = try cfg.Config.parse(std.testing.allocator, fixtures.sample, fixtures.generation);
    const h: u64 = 1536;
    const kv: u64 = 256;
    const m: u64 = 8960;
    const layer = 2 * h + (h + 2 * kv) * h + (h + 2 * kv) + h * h + 3 * m * h;
    const want = (2 * 151936 * h + h + 28 * layer) * 2;
    try std.testing.expectEqual(@as(u64, 3554176000), want);
    try std.testing.expectEqual(want, try weights.textBytes(std.testing.allocator, c));
}

test "dots on CUDA: text only, no reasoning block, its config and tensor refusals worded" {
    try std.testing.expect(!native.reasoning and native.keeps_no_prompts);
    // compiles the load and the layer
    const hooks = .{ &native.open, &native.attach, &native.fits, &native.ledgerPool, &native.textWeightBytes };
    inline for (hooks) |h| try std.testing.expect(@intFromPtr(h) != 0);
    try std.testing.expectEqualStrings("dots_ocr", native.model_type);
    try std.testing.expectStringStartsWith(native.explain(null, error.DenseTextOnly).?, "dots.mocr on CUDA reads text prompts only");
    for ([_]anyerror{ error.UnsupportedDotsConfig, error.BadDotsConfig, error.BadEos, error.DotsShape, error.DotsMissingTensor, error.DotsTensorCount, error.DotsIndexMismatch }) |e|
        try std.testing.expect(native.explain(null, e) != null);
}

test "dots on CUDA reads its geometry from the checkpoint's files and refuses another model's config by name" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "generation_config.json", .data = fixtures.generation });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = fixtures.sample });
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    try std.testing.expectEqual(@as(usize, 2048), (try recipe.geometry(a, io, dir)).acc);
    try std.testing.expectError(error.FileNotFound, recipe.weightBytes(a, io, dir)); // no shards: nothing to count
    const other = try std.mem.replaceOwned(u8, a, fixtures.sample, "\"dots_ocr\"", "\"qwen3_5\"");
    defer a.free(other);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = other });
    try std.testing.expectError(error.UnsupportedDotsConfig, recipe.geometry(a, io, dir));
    try std.testing.expect(native.explain(null, error.UnsupportedDotsConfig) != null);
}
