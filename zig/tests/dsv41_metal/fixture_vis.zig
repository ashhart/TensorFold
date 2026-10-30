//! A tiny vision tower's tensors under the release's names as a safetensors image, and synthetic patches for it.
const std = @import("std");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const ref = @import("ref.zig");

const Spec = host.pack.Spec;
pub const vision = pkg.vision;

/// 2 blocks of 2 heads, 4-pixel patches, 3x3 blocks into the tiny model's 256 hidden.
pub fn config() vision.Config {
    return .{ .layers = 2, .dim = 128, .heads = 2, .inter = 192, .patch = 4, .ratio = 3, .max_tokens = 64, .out_dim = 256, .image_token_id = 255, .min_pixels = 1024 };
}

const Gen = enum { norm, unit, bias, delim, f32 };

/// The tower's tensors, the delimiters and each text layer's image routing bias, drawn from `seed`.
pub fn image(a: std.mem.Allocator, cfg: vision.Config, layers: u32, experts: u32, seed: u64) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const al = arena.allocator();
    var specs: std.ArrayList(Spec) = .empty;
    var gens: std.ArrayList(Gen) = .empty;
    var ks: std.ArrayList(usize) = .empty;
    const add = struct {
        fn f(s: *std.ArrayList(Spec), g: *std.ArrayList(Gen), k: *std.ArrayList(usize), x: std.mem.Allocator, name: []const u8, dtype: []const u8, shape: []const usize, gen: Gen, kk: usize) !void {
            try s.append(x, .{ .name = try x.dupe(u8, name), .dtype = dtype, .shape = try x.dupe(usize, shape) });
            try g.append(x, gen);
            try k.append(x, kk);
        }
    }.f;
    const D: usize = cfg.dim;
    const A: usize = cfg.out_dim;
    var nb: [96]u8 = undefined;
    try add(&specs, &gens, &ks, al, "vision.patch_embed.proj.weight", "BF16", &.{ D, cfg.patchValues() }, .unit, cfg.patchValues());
    try add(&specs, &gens, &ks, al, "vision.patch_embed.proj.bias", "BF16", &.{D}, .bias, 1);
    for (0..cfg.layers) |i| {
        const t = [_]struct { []const u8, []const usize, Gen, usize }{
            .{ "norm1.weight", &.{D}, .norm, 1 },                      .{ "attn.wqkv.weight", &.{ 3 * D, D }, .unit, D },
            .{ "attn.wqkv.bias", &.{3 * D}, .bias, 1 },                .{ "attn.wo.weight", &.{ D, D }, .unit, D },
            .{ "attn.wo.bias", &.{D}, .bias, 1 },                      .{ "norm2.weight", &.{D}, .norm, 1 },
            .{ "mlp.w1.weight", &.{ 2 * cfg.inter, D }, .unit, D },    .{ "mlp.w2.weight", &.{ D, cfg.inter }, .unit, cfg.inter },
        };
        for (t) |x| try add(&specs, &gens, &ks, al, try std.fmt.bufPrint(&nb, "vision.blocks.{d}.{s}", .{ i, x[0] }), "BF16", x[1], x[2], x[3]);
    }
    try add(&specs, &gens, &ks, al, "vision.norm.weight", "BF16", &.{D}, .norm, 1);
    try add(&specs, &gens, &ks, al, "aligner.w1.weight", "BF16", &.{ A, cfg.alignIn() }, .unit, cfg.alignIn());
    try add(&specs, &gens, &ks, al, "aligner.w1.bias", "BF16", &.{A}, .bias, 1);
    try add(&specs, &gens, &ks, al, "aligner.w2.weight", "BF16", &.{ A, A }, .unit, A);
    try add(&specs, &gens, &ks, al, "aligner.w2.bias", "BF16", &.{A}, .bias, 1);
    for ([_][]const u8{ "image_start", "image_end", "image_newline" }) |n| try add(&specs, &gens, &ks, al, n, "BF16", &.{A}, .delim, 1);
    for (0..layers) |i| try add(&specs, &gens, &ks, al, try std.fmt.bufPrint(&nb, "layers.{d}.ffn.gate.bias_vl", .{i}), "F32", &.{experts}, .f32, 1);
    const bytes = try host.pack.image(a, specs.items);
    const head: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
    var at = 8 + head;
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (specs.items, gens.items, ks.items) |s, g, k| {
        var n = host.pack.DType.parse(s.dtype).?.size();
        for (s.shape) |d| n *= d;
        const out = bytes[at..][0..n];
        for (0..n / (if (g == .f32) @as(usize, 4) else 2)) |i| {
            const u = rng.float(f32) * 2 - 1;
            switch (g) {
                .norm => std.mem.writeInt(u16, out[2 * i ..][0..2], ref.toBf(1 + 0.1 * u), .little),
                .unit => std.mem.writeInt(u16, out[2 * i ..][0..2], ref.toBf(u * @sqrt(3.0 / @as(f32, @floatFromInt(k)))), .little),
                .bias => std.mem.writeInt(u16, out[2 * i ..][0..2], ref.toBf(0.1 * u), .little),
                .delim => std.mem.writeInt(u16, out[2 * i ..][0..2], ref.toBf(u), .little),
                .f32 => std.mem.writeInt(u32, out[4 * i ..][0..4], @bitCast(u), .little),
            }
        }
        at += n;
    }
    return bytes;
}

/// Patches [gh * gw][patchValues] as the preprocessor makes them: values in [-1, 1] on bf16's grid, with structure.
pub fn patches(a: std.mem.Allocator, cfg: vision.Config, gh: u32, gw: u32, seed: u64) ![]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const out = try a.alloc(f32, @as(usize, gh) * gw * cfg.patchValues());
    for (out, 0..) |*v, i| {
        const smooth = @sin(@as(f32, @floatFromInt(i % 97)) * 0.3);
        v.* = @as(f32, @bitCast(@as(u32, ref.toBf(0.7 * smooth + 0.3 * (rng.float(f32) * 2 - 1))) << 16));
    }
    return out;
}
