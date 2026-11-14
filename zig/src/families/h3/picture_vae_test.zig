//! MiniMax-H3-Image's VAE host rules (picture_vae.zig): config, tiles, blend, statistics, bytes, posterior.
const std = @import("std");
const v = @import("picture_vae.zig");
const testing = std.testing;

fn arena() std.heap.ArenaAllocator {
    return std.heap.ArenaAllocator.init(testing.allocator);
}

fn configJson(a: std.mem.Allocator, change: []const u8) ![]const u8 {
    var mean: []const u8 = "0.5";
    for (1..24) |_| mean = try std.fmt.allocPrint(a, "{s}, 0.5", .{mean});
    const base = try std.fmt.allocPrint(a,
        \\{{"_class_name": "MiniMaxH3ImageVAE", "latent_channels": 24, "latents_mean": [{s}], "latents_std": [{s}],
        \\"vae_tile_size": 256, "vae_tile_overlap_min": 64, "vae_token_drop": 0, "vit_decoder_2d": true,
        \\"vae_decoder_tiling": 1, "vit_decoder_kwargs": {{"dim_head": 64, "ffn_activation_fn": "silu",
        \\"ffn_use_gated": true, "heads": 16, "norm_affine": true, "norm_type": "rms_norm", "num_layers": 24,
        \\"qk_norm_affine": false, "qk_norm_type": "rms_norm", "rope_dim_ratio": 0.5, "rope_theta": 100.0}}}}
    , .{ mean, mean });
    if (change.len == 0) return base;
    const at = std.mem.indexOf(u8, change, "=>").?;
    return std.mem.replaceOwned(u8, a, base, change[0..at], change[at + 2 ..]);
}

test "picture vae config: the released geometry, and every difference refused" {
    var a = arena();
    defer a.deinit();
    const x = a.allocator();
    const c = try v.config(x, try configJson(x, ""));
    try testing.expectEqual(@as(usize, 256), c.tile);
    try testing.expectEqual(@as(usize, 64), c.overlap);
    try testing.expectEqual(@as(f32, 100), c.rope_theta);
    try testing.expectEqual(@as(f32, 0.5), c.std[23]);
    for ([_][]const u8{
        "MiniMaxH3ImageVAE=>AutoencoderKL",                    "\"vae_token_drop\": 0=>\"vae_token_drop\": 1",
        "\"qk_norm_affine\": false=>\"qk_norm_affine\": true", "\"num_layers\": 24=>\"num_layers\": 26",
        "\"rope_dim_ratio\": 0.5=>\"rope_dim_ratio\": 1.0",    "\"vit_decoder_2d\": true=>\"vit_decoder_2d\": false",
    }) |change| try testing.expectError(error.H3UnsupportedVae, v.config(x, try configJson(x, change)));
    for ([_][]const u8{
        "\"vae_tile_size\": 256=>\"vae_tile_size\": 250", "\"vae_tile_overlap_min\": 64=>\"vae_tile_overlap_min\": 256",
        "[0.5, 0.5=>[0.5",                                "\"rope_theta\": 100.0=>\"rope_theta\": 0",
    }) |change| try testing.expectError(error.H3BadVaeConfig, v.config(x, try configJson(x, change)));
}

fn expectSplit(length: usize, starts: []const usize, overlaps: []const usize) !void {
    var a = arena();
    defer a.deinit();
    const s = try v.split(a.allocator(), length, 256, 64);
    try testing.expectEqual(starts.len, s.len);
    for (s, starts, overlaps) |span, start, o| {
        try testing.expectEqual(start, span.start);
        try testing.expectEqual(o, span.overlap);
    }
}

test "picture vae tiles: 1024, 2048 and 4096 pixels at 256 with at least 64 shared" {
    try expectSplit(1024, &.{ 0, 192, 384, 576, 768 }, &.{ 64, 64, 64, 64, 0 });
    try expectSplit(2048, &.{ 0, 176, 352, 528, 704, 880, 1056, 1232, 1408, 1600, 1792 }, &.{ 80, 80, 80, 80, 80, 80, 80, 80, 64, 64, 0 });
    var a = arena();
    defer a.deinit();
    const big = try v.split(a.allocator(), 4096, 256, 64);
    try testing.expectEqual(@as(usize, 21), big.len);
    for (big[0..20]) |s| try testing.expectEqual(@as(usize, 64), s.overlap);
    const one = try v.split(a.allocator(), 192, 256, 64);
    try testing.expectEqual(@as(usize, 1), one.len);
    try testing.expectEqual(@as(usize, 192), one[0].len);
    try testing.expectError(error.H3BadTiles, v.split(a.allocator(), 1000, 256, 64));
}

test "picture vae tiles: every 32-pixel length ends its last tile on the edge, overlaps at least the least" {
    var a = arena();
    defer a.deinit();
    var n: usize = 32;
    while (n <= 8192) : (n += 32) {
        const s = try v.split(a.allocator(), n, 256, 64);
        const last = s[s.len - 1];
        try testing.expectEqual(n, last.start + last.len);
        for (s[0 .. s.len - 1], s[1..]) |x, y| {
            try testing.expect(x.overlap >= 64 and x.overlap % 16 == 0 and x.overlap < 256);
            try testing.expectEqual(x.start + x.len - x.overlap, y.start);
        }
    }
}

test "picture vae assembly: band weights, copies of one picture reassemble it, a band blends its two tiles" {
    var a = arena();
    defer a.deinit();
    const x = a.allocator();
    try testing.expectEqual([2]f32{ 1, 0 }, v.Assembler.weights(0, 80));
    try testing.expectEqual([2]f32{ @floatCast(1.0 - 40.0 / 80.0), @floatCast(40.0 / 80.0) }, v.Assembler.weights(40, 80));
    // a 2-channel 1024 x 1024 picture of small integers, cut into its tiles and put back
    const rows = try v.split(x, 1024, 256, 64);
    const src = try x.alloc(f32, 2 * 1024 * 1024);
    for (src, 0..) |*p, i| p.* = @floatFromInt(i % 7);
    const out = try x.alloc(f32, src.len);
    var asm_ = try v.Assembler.init(x, rows, rows, 2, out);
    const tile = try x.alloc(f32, 2 * 256 * 256);
    for (rows, 0..) |r, ri| for (rows, 0..) |c, ci| {
        for (0..2) |ch| for (0..256) |y| for (0..256) |col| {
            tile[(ch * 256 + y) * 256 + col] = src[(ch * 1024 + r.start + y) * 1024 + c.start + col];
        };
        try asm_.place(ri, ci, tile);
    };
    for (out, src) |o, s| try testing.expectApproxEqAbs(s, o, 1e-5);
    // two tiles of 0 and 1 side by side: column p of the band is 0 * (1 - p / 64) + 1 * p / 64
    const cols = try v.split(x, 448, 256, 64);
    const one = [_]v.Span{.{ .start = 0, .len = 16, .overlap = 0 }};
    const two = try x.alloc(f32, 16 * 448);
    var pair = try v.Assembler.init(x, &one, cols, 1, two);
    const left = try x.alloc(f32, 16 * 256);
    @memset(left, 0);
    const right = try x.alloc(f32, 16 * 256);
    @memset(right, 1);
    try pair.place(0, 0, left);
    try pair.place(0, 1, right);
    try testing.expectEqual(cols[1].start, @as(usize, 192));
    for (0..64) |p| try testing.expectEqual(v.Assembler.weights(p, 64)[1], two[5 * 448 + 192 + p]);
    try testing.expectEqual(@as(f32, 0), two[5 * 448 + 191]);
    try testing.expectEqual(@as(f32, 1), two[5 * 448 + 447]);
}

test "picture vae bytes truncate; statistics go both ways; the posterior is kept as fp16" {
    try testing.expectEqual(@as(u8, 0), v.quantise(0));
    try testing.expectEqual(@as(u8, 127), v.quantise(0.5));
    try testing.expectEqual(@as(u8, 254), v.quantise(254.9 / 255.0));
    try testing.expectEqual(@as(u8, 255), v.quantise(1));
    try testing.expectEqual(@as(u8, 0), v.quantise(-0.25));
    try testing.expectEqual(@as(u8, 255), v.byte(100, 0));
    try testing.expectEqual(@as(u8, 0), v.byte(-100, 2));
    var c: v.Config = .{ .mean = @splat(0.25), .std = @splat(2), .tile = 256, .overlap = 64, .rope_theta = 100 };
    c.mean[1] = -1;
    var lat: [48]f32 = @splat(0.5);
    try v.denormalize(&c, &lat);
    try testing.expectEqual(@as(f32, 1.25), lat[0]);
    try testing.expectEqual(@as(f32, 0), lat[2]);
    try v.normalize(&c, &lat);
    for (lat) |p| try testing.expectEqual(@as(f32, 0.5), p);
    var out: [3]f32 = undefined;
    try v.posterior(&.{ 1, 0, 0 }, &.{ 0, 100, -100 }, &.{ 1.0 / 3.0, 1, 1 }, &out);
    try testing.expectEqual(@as(f32, @floatCast(@as(f16, @floatCast(@as(f32, 1) + 1.0 / 3.0)))), out[0]);
    try testing.expectEqual(@as(f32, @floatCast(@as(f16, @floatCast(@exp(@as(f32, 10)))))), out[1]);
    try testing.expectEqual(@as(f32, @floatCast(@as(f16, @floatCast(@exp(@as(f32, -15)))))), out[2]);
}
