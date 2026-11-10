//! dots.mocr's CUDA image tower on the host: its contract, bytes, inverse frequencies, refusals and patch rows.
const std = @import("std");
const cuda = @import("cuda");
const dense = @import("cuda_dense");
const bf16 = @import("core").bf16_round;
const cfg = @import("config.zig");
const fixtures = @import("config_test.zig");
const image = @import("image.zig");
const recipe = @import("cuda_recipe.zig");
const tw = @import("cuda_tower.zig");
const weights = @import("cuda_tower_weights.zig");
const t = std.testing;

const Tower = tw.Tower;
const native = dense.Native(recipe);

fn config() !cfg.Config {
    return cfg.Config.parse(t.allocator, fixtures.sample, fixtures.generation);
}

const Dir = @import("sparse_checkpoint_test.zig").Dir;

test "dots tower: the engine's contract, its marks and the template's own wrapping" {
    comptime dense.recipe.checkTower(Tower);
    try t.expect(comptime dense.recipe.towerProblem(Tower) == null);
    try t.expectEqual(tw.Marks{ .open = 151666, .image = 151665, .close = 151667 }, tw.marks);
    try t.expectEqual([3]u32{ 151666, 151665, 151667 }, tw.mark_ids);
    try t.expectEqual((try config()).image_token, tw.marks.image);
    try t.expect(Tower.template_wraps and std.mem.eql(u8, Tower.part_type, "image"));
}

test "dots tower: the largest image is the processor's own, 14,400 tokens of 2 x 2 patches" {
    const c = try config();
    const p = try image.Processor.parse(t.allocator, @embedFile("fixtures/preprocessor_config.json"), c);
    try t.expectEqual(@as(u64, tw.max_patches), p.patches.max_pixels / (@as(u64, c.vision.patch) * c.vision.patch));
    try t.expectEqual(@as(u32, tw.max_patches / 4), Tower.default_image_tokens);
}

test "dots tower scratch: 39,592 bytes a patch, the merger's buffers inside the blocks' own" {
    const v = (try config()).vision;
    const one = tw.Scratch.of(v, 256); // 256 rows: every part a whole number of 256-byte blocks
    try t.expectEqual(@as(u64, 256 * 39592 + 256), one.bytes);
    const p = tw.max_patches;
    const s = tw.Scratch.of(v, p);
    const h: u64 = v.hidden;
    try t.expectEqual(@as(u64, p * (592 * 2 + 8 + 4 * 1536 * 2 + 3 * 1536 * 2 + 2 * 4224 * 2) + 256), s.bytes);
    try t.expectEqual(@as(u64, 2_280_499_456), s.bytes);
    try t.expect(s.px == 0 and s.hw == p * 592 * 2 and s.inv == s.bytes - 256);
    try t.expect(s.qkv + p * @as(u64, v.intermediate) * 2 <= s.att); // the SwiGLU's rows in q|k|v's
    try t.expect(s.gu + p / 4 * @as(u64, v.merged()) * 4 <= s.inv); // the merger's f32 sums in fc1|fc3's
    try t.expect(s.att + p / 4 * @as(u64, v.merged()) * 2 <= s.gu); // its first linear in attention's rows
    try t.expect(s.r + p / 4 * @as(u64, v.merged()) * 2 <= s.qkv and s.h + p * h * 2 <= s.xn);
}

test "dots tower bytes: weights 2,524,194,816 on the device, the scratch and workspace beside them" {
    const a = t.allocator;
    const io = t.io;
    var d = try Dir.init(true, null);
    defer d.deinit();
    try t.expectEqual(@as(u64, 2_524_194_816), try Tower.weightBytes(a, io, d.path));
    const open = 57600 * 39592 + 256 + (32 << 20);
    try t.expectEqual(@as(u64, 2_314_053_888), open);
    try t.expectEqual(@as(u64, open), try Tower.openBytes(a, io, d.path));
    var bare = try Dir.init(false, null); // no shards: nothing to count
    defer bare.deinit();
    try t.expectError(error.FileNotFound, Tower.weightBytes(a, io, bare.path));
    try t.expectEqual(@as(u64, open), try Tower.openBytes(a, io, bare.path));
    var short = try Dir.init(true, "vision_tower.merger.mlp.2.bias"); // the planted control: a tower tensor missing
    defer short.deinit();
    try t.expectError(error.DotsTensorCount, Tower.weightBytes(a, io, short.path));
}

/// The first entry where two tables' bits differ.
fn firstDiff(x: []const f32, y: []const f32) ?usize {
    for (x, y, 0..) |p, q, i| if (@as(u32, @bitCast(p)) != @as(u32, @bitCast(q))) return i;
    return null;
}

test "dots tower rotary: the fixture's 32 frequencies equal the float32 table 1 / 10000^(2i / 64) bit for bit" {
    var host: [tw.rope_pairs]f32 = undefined;
    dense.rope.inverse(10000, 64, &host);
    try t.expectEqual(@as(?usize, null), firstDiff(&tw.inv_freq, &host));
    try t.expectEqual(@as(f32, 1), tw.inv_freq[0]);
    var off = host; // the planted control: one entry an ulp away is seen
    off[27] = @bitCast(@as(u32, @bitCast(off[27])) + 1);
    try t.expectEqual(@as(?usize, 27), firstDiff(&tw.inv_freq, &off));
}

test "dots tower refusals: each new error worded, the budget's naming its flag, an unknown one not" {
    for ([_]anyerror{ error.DotsImageSize, error.DotsImageBudget, error.DotsGrid, error.DotsVisionLimits }) |e| {
        try t.expect(Tower.explain(e) != null);
    }
    try t.expect(std.mem.indexOf(u8, Tower.explain(error.DotsImageBudget).?, "(--vision-image-tokens)") != null);
    try t.expectStringStartsWith(Tower.explain(error.AspectRatio).?, "its longer side");
    try t.expect(Tower.explain(error.DotsTowerUnheardOf) == null); // the control: not every error has words
    try t.expect(native.explain(null, error.DotsShape) != null); // the weights' own refusals: the recipe's words
}

fn grid(gh: u32, gw: u32, pixels: []const f32, tokens: u32) dense.recipe.Image {
    return .{ .pixels = pixels, .gh = gh, .gw = gw, .at = 0, .tokens = tokens };
}

test "dots tower images: whole 2 x 2 merge blocks whose pixels and tokens fill the grid, at most 57,600 patches" {
    const v = (try config()).vision;
    const px = try t.allocator.alloc(f32, 4 * 6 * 588);
    defer t.allocator.free(px);
    try t.expectEqual(@as(usize, 24), try tw.rowsOf(v, &grid(4, 6, px, 6))); // the control: a whole grid passes
    try t.expectError(error.DotsGrid, tw.rowsOf(v, &grid(3, 8, px, 6)));
    try t.expectError(error.DotsGrid, tw.rowsOf(v, &grid(4, 6, px, 5)));
    try t.expectError(error.DotsGrid, tw.rowsOf(v, &grid(4, 6, px[1..], 6)));
    try t.expectError(error.DotsGrid, tw.rowsOf(v, &grid(0, 6, px, 0)));
    const most: []const f32 = @as([*]const f32, @ptrFromInt(0x1000))[0 .. 240 * 240 * 588]; // its length alone is read
    try t.expectEqual(@as(usize, 57600), try tw.rowsOf(v, &grid(240, 240, most, 14400)));
    try t.expectError(error.DotsImageBudget, tw.rowsOf(v, &grid(240, 242, most, 14520)));
}

test "dots tower patches: bf16 rows of 592, rounded to nearest even, the four columns past 588 zero" {
    var src: [2 * 588]f32 = undefined;
    for (&src, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i)) * 0.001 - 0.5;
    src[0] = 1.00390625; // half a bf16 step above 1: ties to even, to 1
    src[1] = 1.01171875; // half a step above 1.0078125: to 1.015625
    var dst: [2 * weights.patch_k]u16 = @splat(0xAAAA);
    tw.patchRows(&dst, &src, 588);
    try t.expectEqual(@as(u16, 0x3f80), dst[0]);
    try t.expectEqual(@as(u16, 0x3f82), dst[1]);
    for (0..2) |r| {
        for (0..588) |c| try t.expectEqual(bf16.bits(src[r * 588 + c]), dst[r * weights.patch_k + c]);
        for (588..weights.patch_k) |c| try t.expectEqual(@as(u16, 0), dst[r * weights.patch_k + c]);
    }
}

test "dots tower open: zero limits refused before the config is read; valid ones read the processor's file" {
    const a = t.allocator;
    var d = try Dir.init(false, null); // no preprocessor_config.json
    defer d.deinit();
    const ctx: *const cuda.Context = undefined; // neither call reaches the device
    const s: cuda.Stream = undefined;
    try t.expectError(error.DotsVisionLimits, Tower.open(a, t.io, ctx, s, d.path, .{ .image_tokens = 0, .max_images = 1 }));
    try t.expectError(error.DotsVisionLimits, Tower.open(a, t.io, ctx, s, d.path, .{ .image_tokens = 14400, .max_images = 0 }));
    // the control
    try t.expectError(error.FileNotFound, Tower.open(a, t.io, ctx, s, d.path, .{ .image_tokens = 14400, .max_images = 1 }));
}
