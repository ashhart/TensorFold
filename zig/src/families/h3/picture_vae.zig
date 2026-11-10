//! MiniMax-H3-Image's VAE on the host: its config, the tile split and blend, latent statistics, pixels, posterior.
const std = @import("std");
const geometry = @import("geometry.zig");

pub const latent_channels = 24;
/// Pixels a latent cell covers a side.
pub const cell_pixels = 16;
/// The decoder's transformer: 24 layers of 16 heads of 64, a gated FFN of 4,096, 4 register tokens.
pub const layers = 24;
pub const heads = 16;
pub const head_dim = 64;
pub const width = heads * head_dim;
pub const ffn = 4096;
pub const registers = 4;
/// Rotated dims a head: `rope_dim_ratio` of the head; 8 frequencies an axis over two axes.
pub const rotated = 32;
pub const axis_freqs = rotated / 4;
/// Every decoder norm's epsilon, and the encoder's GroupNorm's.
pub const eps: f32 = 1e-5;
pub const group_eps: f32 = 1e-6;
/// ImageNet's statistics, which map decoded values to [0, 1] pixels and pixels to the encoder's input.
pub const pixel_mean = [3]f32{ 0.485, 0.456, 0.406 };
pub const pixel_std = [3]f32{ 0.229, 0.224, 0.225 };

/// `vae/config.json` at the released geometry: the latent statistics, the tiles and the decoder's rotary theta.
pub const Config = struct {
    mean: [latent_channels]f32,
    std: [latent_channels]f32,
    tile: usize,
    overlap: usize,
    rope_theta: f32,
};

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.H3BadVaeConfig,
    };
}

fn get(o: std.json.ObjectMap, key: []const u8) !std.json.Value {
    return o.get(key) orelse error.H3BadVaeConfig;
}

fn expectInt(o: std.json.ObjectMap, key: []const u8, want: i64) !void {
    const v = try get(o, key);
    if (v != .integer or v.integer != want) return error.H3UnsupportedVae;
}

fn expectText(o: std.json.ObjectMap, key: []const u8, want: []const u8) !void {
    const v = try get(o, key);
    if (v != .string or !std.mem.eql(u8, v.string, want)) return error.H3UnsupportedVae;
}

fn expectBool(o: std.json.ObjectMap, key: []const u8, want: bool) !void {
    const v = try get(o, key);
    if (v != .bool or v.bool != want) return error.H3UnsupportedVae;
}

/// Accepts `MiniMaxH3ImageVAE` at the released decoder and tiling, else an error naming what differs.
pub fn config(a: std.mem.Allocator, json: []const u8) !Config {
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return error.H3BadVaeConfig;
    if (doc != .object) return error.H3BadVaeConfig;
    const o = doc.object;
    try expectText(o, "_class_name", "MiniMaxH3ImageVAE");
    try expectInt(o, "latent_channels", latent_channels);
    try expectInt(o, "vae_token_drop", 0);
    try expectBool(o, "vit_decoder_2d", true);
    try expectInt(o, "vae_decoder_tiling", 1);
    const kw = try get(o, "vit_decoder_kwargs");
    if (kw != .object) return error.H3BadVaeConfig;
    const k = kw.object;
    inline for (.{ .{ "num_layers", layers }, .{ "heads", heads }, .{ "dim_head", head_dim } }) |kv| try expectInt(k, kv[0], kv[1]);
    inline for (.{ .{ "norm_type", "rms_norm" }, .{ "qk_norm_type", "rms_norm" }, .{ "ffn_activation_fn", "silu" } }) |kv| try expectText(k, kv[0], kv[1]);
    try expectBool(k, "norm_affine", true);
    try expectBool(k, "qk_norm_affine", false);
    try expectBool(k, "ffn_use_gated", true);
    if (try number(try get(k, "rope_dim_ratio")) != 0.5) return error.H3UnsupportedVae;
    const theta = try number(try get(k, "rope_theta"));
    if (!(theta > 0) or !std.math.isFinite(theta)) return error.H3BadVaeConfig;
    var c: Config = .{ .mean = undefined, .std = undefined, .tile = 0, .overlap = 0, .rope_theta = @floatCast(theta) };
    inline for (.{ .{ "latents_mean", &c.mean }, .{ "latents_std", &c.std } }) |kv| {
        const list = try get(o, kv[0]);
        if (list != .array or list.array.items.len != latent_channels) return error.H3BadVaeConfig;
        for (kv[1], list.array.items) |*d, v| {
            const f = try number(v);
            if (!std.math.isFinite(f)) return error.H3BadVaeConfig;
            d.* = @floatCast(f);
        }
    }
    for (c.std) |s| if (!(s > 0)) return error.H3BadVaeConfig;
    const tile = try get(o, "vae_tile_size");
    const overlap = try get(o, "vae_tile_overlap_min");
    if (tile != .integer or overlap != .integer) return error.H3BadVaeConfig;
    if (tile.integer < cell_pixels or overlap.integer < 0 or overlap.integer >= tile.integer or
        @rem(tile.integer, cell_pixels) != 0 or @rem(overlap.integer, cell_pixels) != 0) return error.H3BadVaeConfig;
    c.tile = @intCast(tile.integer);
    c.overlap = @intCast(overlap.integer);
    return c;
}

/// One tile of an axis: its first pixel, its length and its overlap with the next tile (0 for the last).
pub const Span = struct { start: usize, len: usize, overlap: usize };

/// An axis in tiles overlapping by at least `least`, the spare spread 16 pixels at a time from the first overlap.
pub fn split(a: std.mem.Allocator, length: usize, tile: usize, least: usize) ![]Span {
    if (length == 0 or length % cell_pixels != 0 or tile % cell_pixels != 0 or least % cell_pixels != 0 or least >= tile)
        return error.H3BadTiles;
    if (tile >= length) {
        const one = try a.alloc(Span, 1);
        one[0] = .{ .start = 0, .len = length, .overlap = 0 };
        return one;
    }
    var count = (length + tile - 1) / tile;
    while (count * tile < (count - 1) * least + length) count += 1;
    if (count > geometry.tile_max) return error.H3BadTiles;
    const spare = (count * tile - (count - 1) * least - length) / cell_pixels;
    const spans = try a.alloc(Span, count);
    var start: usize = 0;
    for (spans, 0..) |*s, i| {
        const last = i + 1 == count;
        const extra: usize = if (last) 0 else spare / (count - 1) + @intFromBool(i < spare % (count - 1));
        s.* = .{ .start = start, .len = tile, .overlap = if (last) 0 else least + extra * cell_pixels };
        start += tile - s.overlap;
    }
    return spans;
}

/// Decoded tiles, row-major: top band blended with the tile above, left band with the left one (both as decoded).
pub const Assembler = struct {
    rows: []const Span,
    cols: []const Span,
    channels: usize,
    out: []f32,
    height: usize,
    width: usize,
    above: [][]f32,
    current: [][]f32,
    scratch: []f32,

    pub fn init(a: std.mem.Allocator, rows: []const Span, cols: []const Span, channels: usize, out: []f32) !Assembler {
        const height = rows[rows.len - 1].start + rows[rows.len - 1].len;
        const w = cols[cols.len - 1].start + cols[cols.len - 1].len;
        if (out.len != try geometry.product(&.{ channels, height, w })) return error.H3BadLatentLength;
        const most = try geometry.product(&.{ channels, rows[0].len, cols[0].len });
        const above = try a.alloc([]f32, cols.len);
        const current = try a.alloc([]f32, cols.len);
        for (above, current) |*x, *y| {
            x.* = try a.alloc(f32, most);
            y.* = try a.alloc(f32, most);
        }
        return .{ .rows = rows, .cols = cols, .channels = channels, .out = out, .height = height, .width = w, .above = above, .current = current, .scratch = try a.alloc(f32, most) };
    }

    /// The fp32 weights of band row or column `p` of `extent`: the neighbour's 1 - p / e and this tile's p / e.
    pub fn weights(p: usize, extent: usize) [2]f32 {
        const t = @as(f64, @floatFromInt(p)) / @as(f64, @floatFromInt(extent));
        return .{ @floatCast(1.0 - t), @floatCast(t) };
    }

    /// Tile (r, c) as decoded; tiles must come in row-major order.
    pub fn place(s: *Assembler, r: usize, c: usize, tile: []const f32) !void {
        const th = s.rows[r].len;
        const tw = s.cols[c].len;
        if (tile.len != s.channels * th * tw) return error.H3BadLatentLength;
        if (c == 0 and r > 0) std.mem.swap([][]f32, &s.above, &s.current);
        @memcpy(s.current[c][0..tile.len], tile);
        const t = s.scratch[0..tile.len];
        @memcpy(t, tile);
        if (r > 0) {
            const e = s.rows[r - 1].overlap;
            const up = s.above[c];
            const uh = s.rows[r - 1].len;
            for (0..s.channels) |ch| for (0..e) |p| {
                const w = weights(p, e);
                for (0..tw) |x| {
                    const at = (ch * th + p) * tw + x;
                    t[at] = up[(ch * uh + uh - e + p) * tw + x] * w[0] + t[at] * w[1];
                }
            };
        }
        if (c > 0) {
            const e = s.cols[c - 1].overlap;
            const left = s.current[c - 1];
            const lw = s.cols[c - 1].len;
            for (0..s.channels) |ch| for (0..th) |y| for (0..e) |p| {
                const w = weights(p, e);
                const at = (ch * th + y) * tw + p;
                t[at] = left[(ch * th + y) * lw + lw - e + p] * w[0] + t[at] * w[1];
            };
        }
        const keep_h = th - s.rows[r].overlap;
        const keep_w = tw - s.cols[c].overlap;
        for (0..s.channels) |ch| for (0..keep_h) |y| {
            const to = (ch * s.height + s.rows[r].start + y) * s.width + s.cols[c].start;
            @memcpy(s.out[to..][0..keep_w], t[(ch * th + y) * tw ..][0..keep_w]);
        };
    }
};

/// A normalised latent [24, rows, columns] to the VAE's raw latent, in place: x * std + mean, float32.
pub fn denormalize(c: *const Config, latent: []f32) !void {
    if (latent.len % latent_channels != 0) return error.H3BadLatentLength;
    const plane = latent.len / latent_channels;
    for (0..latent_channels) |ch| for (latent[ch * plane ..][0..plane]) |*v| {
        v.* = v.* * c.std[ch] + c.mean[ch];
    };
}

/// The inverse, in place: (x - mean) / std, float32.
pub fn normalize(c: *const Config, latent: []f32) !void {
    if (latent.len % latent_channels != 0) return error.H3BadLatentLength;
    const plane = latent.len / latent_channels;
    for (0..latent_channels) |ch| for (latent[ch * plane ..][0..plane]) |*v| {
        v.* = (v.* - c.mean[ch]) / c.std[ch];
    };
}

/// A [0, 1] value as a byte: x * 255 clamped, then truncated, not rounded.
pub fn quantise(x: f32) u8 {
    return @intFromFloat(std.math.clamp(x * 255.0, 0.0, 255.0));
}

/// One decoded value of channel `ch` mapped to [0, 1] with ImageNet's statistics, clamped, then quantised.
pub fn byte(decoded: f32, ch: usize) u8 {
    return quantise(std.math.clamp(decoded * pixel_std[ch] + pixel_mean[ch], 0.0, 1.0));
}

/// Decoded [3, height, width] float32 to RGB bytes [height, width, 3].
pub fn pixels(decoded: []const f32, h: usize, w: usize, rgb: []u8) !void {
    const plane = try geometry.product(&.{ h, w });
    if (decoded.len != 3 * plane or rgb.len != 3 * plane) return error.H3BadLatentLength;
    for (0..3) |ch| for (0..plane) |i| {
        rgb[3 * i + ch] = byte(decoded[ch * plane + i], ch);
    };
}

/// The encoder's posterior sample: mean + exp(0.5 * clamp(logvar, -30, 20)) * noise, kept as fp16 does.
pub fn posterior(mean: []const f32, logvar: []const f32, noise: []const f32, out: []f32) !void {
    if (mean.len != logvar.len or mean.len != noise.len or mean.len != out.len) return error.H3BadLatentLength;
    for (out, mean, logvar, noise) |*o, m, lv, n| {
        const s = @exp(0.5 * std.math.clamp(lv, -30.0, 20.0));
        o.* = @floatCast(@as(f16, @floatCast(m + s * n)));
    }
}
