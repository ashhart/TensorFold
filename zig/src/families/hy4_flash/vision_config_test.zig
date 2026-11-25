//! Host tests for Hy4-flash vision config parsing, patchify, bilinear 128x128 taps, and 3D M-RoPE layout.
const std = @import("std");
const vision = @import("vision_config.zig");

const testing = std.testing;

test "vision_config parses release fixture config.json and optional processor_config overrides" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const fixture = @embedFile("config_fixture.json");
    const c = (try vision.Config.parse(a, fixture, null)).?;
    try testing.expectEqual(@as(usize, 1152), c.hidden);
    try testing.expectEqual(@as(usize, 16), c.heads);
    try testing.expectEqual(@as(usize, 72), c.headDim());
    try testing.expectEqual(@as(usize, 27), c.depth);
    try testing.expectEqual(@as(usize, 4304), c.intermediate);
    try testing.expectEqual(@as(usize, 3072), c.out_hidden);
    try testing.expectEqual(@as(usize, 4608), c.mergedWidth());
    try testing.expectEqual(@as(u32, 120118), c.image_start);
    try testing.expectEqual(@as(u32, 120120), c.image_token);
    try testing.expectEqual(@as(u32, 120119), c.image_end);
    try testing.expectEqual(@as(u32, 120124), c.video_token);

    const proc =
        \\{"image_processor":{"size":{"shortest_edge":8192,"longest_edge":262144},"rescale_factor":0.00392156862745098,"image_mean":[0.25,0.5,0.75],"image_std":[0.5,0.25,1.0]}}
    ;
    const ov = (try vision.Config.parse(a, fixture, proc)).?;
    try testing.expectEqual(@as(u64, 8192), ov.min_pixels);
    try testing.expectEqual(@as(u64, 262144), ov.max_pixels);
    try testing.expectEqual([3]f32{ 0.25, 0.5, 0.75 }, ov.mean);
    try testing.expectEqual([3]f32{ 0.5, 0.25, 1.0 }, ov.std_);

    // Text-only config without vision_config returns null.
    try testing.expect((try vision.Config.parse(a, "{\"model_type\":\"hy_v4_flash\"}", null)) == null);
}

test "vision_config refuses unsupported geometry, non-gelu activation, and bad processor parameters" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const fixture = @embedFile("config_fixture.json");
    try testing.expectError(error.UnsupportedHy4Vision, vision.Config.parse(a, "{\"vision_config\":{\"patch_size\":14}}", null));
    try testing.expectError(error.UnsupportedHy4Vision, vision.Config.parse(a, "{\"vision_config\":{\"hidden_size\":1024}}", null));
    try testing.expectError(error.UnsupportedHy4Vision, vision.Config.parse(a, "{\"vision_config\":{\"hidden_act\":\"silu\"}}", null));
    try testing.expectError(error.BadVisionConfig, vision.Config.parse(a, "{\"vision_config\":{\"rms_norm_eps\":0}}", null));
    try testing.expectError(error.BadVisionConfig, vision.Config.parse(a, fixture, "{\"image_std\":[0.5,0.0,0.5]}"));
    try testing.expectError(error.BadVisionConfig, vision.Config.parse(a, fixture, "{\"min_pixels\":65536,\"max_pixels\":4096}"));
    try testing.expectError(error.BadVisionConfig, vision.Config.parse(a, fixture, "{\"rescale_factor\":0}"));
}

test "patchifyRaster and prepareRgb normalize RGB channels and enforce max_tokens budget" {
    const a = testing.allocator;
    const c: vision.Config = .{};
    var pixels: [32 * 32 * 3]u8 = undefined;
    for (0..32) |y| for (0..32) |x| {
        const off = (y * 32 + x) * 3;
        pixels[off] = 0; // -> (0 - 0.5) / 0.5 = -1.0
        pixels[off + 1] = 255; // -> (1 - 0.5) / 0.5 = +1.0
        pixels[off + 2] = if (y < 16 and x < 16) 0 else 255;
    };
    const flat = try vision.patchifyRaster(a, .{ .width = 32, .height = 32, .pixels = &pixels }, c);
    defer a.free(flat);
    try testing.expectEqual(@as(usize, 4 * vision.patch_width), flat.len);
    // Patch (0, 0): channel 0 = -1, channel 1 = +1, channel 2 = -1; Patch (0, 1): channel 2 = +1.
    try testing.expectApproxEqAbs(@as(f32, -1.0), flat[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), flat[256], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -1.0), flat[512], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), flat[vision.patch_width + 512], 1e-6);

    // Budget shrinking: a 96x64 image capped to max_tokens = 2 shrinks to 32x32 (gh=2, gw=2, tokens=2).
    const big = try a.alloc(u8, 96 * 64 * 3);
    defer a.free(big);
    @memset(big, 128);
    const prep = try vision.prepareRgb(a, .{ .width = 96, .height = 64, .pixels = big }, 42, c, 2);
    defer a.free(prep.pixels);
    try testing.expectEqual(@as(u32, 2), prep.gh);
    try testing.expectEqual(@as(u32, 2), prep.gw);
    try testing.expectEqual(@as(u32, 2), prep.tokens);
    try testing.expectEqual(@as(usize, 4 * vision.patch_width), prep.pixels.len);

    // Refusal controls: max_tokens < 2 and malformed RGB buffer.
    try testing.expectError(error.ImageTooSmall, vision.prepareRgb(a, .{ .width = 32, .height = 32, .pixels = &pixels }, 42, c, 1));
    try testing.expectError(error.Invalid, vision.prepareRgb(a, .{ .width = 32, .height = 32, .pixels = pixels[0..10] }, 42, c, 8));
}

test "prepareRgb terminates and respects max_tokens for narrow portrait and landscape images clamped to 32 px" {
    const a = testing.allocator;
    const c: vision.Config = .{};
    // Narrow portrait (h=128, w=16) with 4-token budget: 1 merged col fits at most max_tokens/2 = 2 merged rows.
    const port_px = try a.alloc(u8, 128 * 16 * 3);
    defer a.free(port_px);
    @memset(port_px, 64);
    const port = try vision.prepareRgb(a, .{ .width = 16, .height = 128, .pixels = port_px }, 7, c, 4);
    defer a.free(port.pixels);
    try testing.expectEqual(@as(u32, 4), port.gh);
    try testing.expectEqual(@as(u32, 2), port.gw);
    try testing.expectEqual(@as(u32, 4), port.tokens);

    // Narrow landscape (h=16, w=128) with 4-token budget: 1 merged row fits at most max_tokens-1 = 3 merged cols.
    const land_px = try a.alloc(u8, 16 * 128 * 3);
    defer a.free(land_px);
    @memset(land_px, 192);
    const land = try vision.prepareRgb(a, .{ .width = 128, .height = 16, .pixels = land_px }, 8, c, 4);
    defer a.free(land.pixels);
    try testing.expectEqual(@as(u32, 2), land.gh);
    try testing.expectEqual(@as(u32, 6), land.gw);
    try testing.expectEqual(@as(u32, 4), land.tokens);

    // Extreme 100:1 strips and odd budget (max_tokens = 5): portrait -> 4 tokens (2x1), landscape -> 5 tokens (1x4).
    const tall_px = try a.alloc(u8, 800 * 8 * 3);
    defer a.free(tall_px);
    @memset(tall_px, 32);
    const tall = try vision.prepareRgb(a, .{ .width = 8, .height = 800, .pixels = tall_px }, 9, c, 5);
    defer a.free(tall.pixels);
    try testing.expectEqual(@as(u32, 4), tall.gh);
    try testing.expectEqual(@as(u32, 2), tall.gw);
    try testing.expectEqual(@as(u32, 4), tall.tokens);

    const wide = try vision.prepareRgb(a, .{ .width = 800, .height = 8, .pixels = tall_px }, 10, c, 5);
    defer a.free(wide.pixels);
    try testing.expectEqual(@as(u32, 2), wide.gh);
    try testing.expectEqual(@as(u32, 8), wide.gw);
    try testing.expectEqual(@as(u32, 5), wide.tokens);
}

test "positions produces exact 128x128 identity and exact normalized bilinear 4-taps on smaller grids" {
    const a = testing.allocator;
    const id_idx = try a.alloc(i32, 16384 * 4);
    defer a.free(id_idx);
    const id_wts = try a.alloc(f32, 16384 * 4);
    defer a.free(id_wts);
    vision.positions(128, 128, id_idx, id_wts);
    for (0..16384) |p| {
        try testing.expectEqual(@as(i32, @intCast(p)), id_idx[p * 4]);
        try testing.expectEqual(@as(f32, 1.0), id_wts[p * 4]);
        try testing.expectEqual(@as(f32, 0.0), id_wts[p * 4 + 1] + id_wts[p * 4 + 2] + id_wts[p * 4 + 3]);
    }

    // Non-identity grid (gh=4, gw=8): check closed-form tap values, in-range indices, and unit weight sum.
    var idx: [4 * 8 * 4]i32 = undefined;
    var wts: [4 * 8 * 4]f32 = undefined;
    vision.positions(4, 8, &idx, &wts);
    for (0..32) |p| {
        var sum: f32 = 0.0;
        for (0..4) |t| {
            try testing.expect(idx[p * 4 + t] >= 0 and idx[p * 4 + t] < 16384);
            try testing.expect(wts[p * 4 + t] >= 0.0 and wts[p * 4 + t] <= 1.0);
            sum += wts[p * 4 + t];
        }
        try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);
    }
    // Cell (r=0, col=0): row raw = 15.109756... -> (15, 16); col raw = 7.401234... -> (7, 8).
    try testing.expectEqual(@as(i32, 15 * 128 + 7), idx[0]);
    try testing.expectEqual(@as(i32, 15 * 128 + 8), idx[1]);
    try testing.expectEqual(@as(i32, 16 * 128 + 7), idx[2]);
    try testing.expectEqual(@as(i32, 16 * 128 + 8), idx[3]);
    const dr: f32 = @as(f32, @floatCast(0.5 * 128.0 / 4.1 - 0.5)) - 15.0;
    const dc: f32 = @as(f32, @floatCast(0.5 * 128.0 / 8.1 - 0.5)) - 7.0;
    try testing.expectApproxEqAbs((1.0 - dr) * (1.0 - dc), wts[0], 1e-6);
    try testing.expectApproxEqAbs((1.0 - dr) * dc, wts[1], 1e-6);
    try testing.expectApproxEqAbs(dr * (1.0 - dc), wts[2], 1e-6);
    try testing.expectApproxEqAbs(dr * dc, wts[3], 1e-6);
}

test "layout computes 3D M-RoPE coordinates and rope_delta for single-image, multi-image, and text-only prompts" {
    try testing.expectEqual(@as(u32, 6), vision.tokensForGrid(4, 4)); // 2 x (2 + 1) = 6
    try testing.expectEqual(@as(u32, 4160), vision.tokensForGrid(128, 128)); // 64 x (64 + 1)

    // Single 2x2 merged image (gh=4, gw=4 -> 2x3 = 6 tokens) at offset 1 in an 8-token prompt.
    var src: [8]i32 = undefined;
    var pos3: [24]i32 = undefined;
    const delta = try vision.layout(8, &.{.{ .at = 1, .gh = 4, .gw = 4 }}, &src, &pos3);
    try testing.expectEqualSlices(i32, &.{ -1, 0, 1, 2, 3, 4, 5, -1 }, &src);
    try testing.expectEqualSlices(i32, &.{ 0, 1, 1, 1, 1, 1, 1, 4 }, pos3[0..8]); // T
    try testing.expectEqualSlices(i32, &.{ 0, 1, 1, 1, 2, 2, 2, 4 }, pos3[8..16]); // H
    try testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 1, 2, 3, 4 }, pos3[16..24]); // W
    try testing.expectEqual(@as(i32, -3), delta);

    // Two images separated by 1 text token: im0 at 1 (2x2 -> 2 tok), text at 3, im1 at 4 (4x2 -> 4 tok), text at 8.
    var src2: [9]i32 = undefined;
    var pos2: [27]i32 = undefined;
    const delta2 = try vision.layout(9, &.{ .{ .at = 1, .gh = 2, .gw = 2 }, .{ .at = 4, .gh = 4, .gw = 2 } }, &src2, &pos2);
    try testing.expectEqualSlices(i32, &.{ -1, 0, 1, -1, 2, 3, 4, 5, -1 }, &src2);
    try testing.expectEqualSlices(i32, &.{ 0, 1, 1, 3, 4, 4, 4, 4, 6 }, pos2[0..9]); // T
    try testing.expectEqualSlices(i32, &.{ 0, 1, 1, 3, 4, 4, 5, 5, 6 }, pos2[9..18]); // H
    try testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 4, 5, 4, 5, 6 }, pos2[18..27]); // W
    try testing.expectEqual(@as(i32, -2), delta2); // next=6, cursor=8 -> 6 - 8 = -2

    // Text-only prompt: identity 1D positions and delta 0.
    var src0: [4]i32 = undefined;
    var pos0: [12]i32 = undefined;
    try testing.expectEqual(@as(i32, 0), try vision.layout(4, &.{}, &src0, &pos0));
    try testing.expectEqualSlices(i32, &.{ -1, -1, -1, -1 }, &src0);
    try testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3 }, pos0[0..4]);

    // Refusal controls: bad slice lengths, overlapping runs, out-of-bounds run, and odd patch grid.
    try testing.expectError(error.BadVisionLayout, vision.layout(8, &.{}, src[0..7], &pos3));
    try testing.expectError(error.BadVisionLayout, vision.layout(8, &.{.{ .at = 3, .gh = 4, .gw = 4 }}, &src, &pos3));
    try testing.expectError(error.BadVisionLayout, vision.layout(9, &.{ .{ .at = 1, .gh = 4, .gw = 4 }, .{ .at = 5, .gh = 2, .gw = 2 } }, &src2, &pos2));
    try testing.expectError(error.BadVisionLayout, vision.layout(8, &.{.{ .at = 1, .gh = 3, .gw = 4 }}, &src, &pos3));
}
