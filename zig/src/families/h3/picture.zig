//! MiniMax-H3-Image's picture rules on the host: canvas, distilled schedule, positions, rotary, packing, tiles.
const std = @import("std");
const geometry = @import("geometry.zig");

/// One transformer row covers this many pixels a side: the VAE's 16 and the 2 x 2 patch.
pub const row_pixels = 32;
/// Latent cells a side per pixel side: the VAE's downsampling.
pub const cell_pixels = 16;
/// Values of one row: 24 latent channels times the 2 x 2 patch.
pub const channels = 24;
pub const row_width = channels * 4;
/// The smallest area drawn by default (2048 x 2048); smaller requests are drawn larger and resized back.
pub const default_min_area: u64 = 2048 * 2048;
pub const min_area_floor: u64 = row_pixels * row_pixels;
pub const min_area_ceiling: u64 = 4096 * 4096;
/// Reference rows are held at this timestep (noise 0.001) at every step.
pub const reference_time: f32 = 0.999;
/// Frequencies a rotary axis has (`rope_inv_freq_len`), three axes, so 48 angles a row and 96 rotated dims.
pub const axis_freqs = 16;
pub const angles = 3 * axis_freqs;
pub const rotated = 2 * angles;

/// x rounded to the nearest integer, ties to even, as Python's round() does on a float64.
pub fn roundEven(x: f64) f64 {
    const f = @floor(x);
    const d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return if (@mod(f, 2) == 0) f else f + 1;
}

pub const Size = struct { width: usize, height: usize };

/// The canvas a picture is drawn on: grown to `min_area` at its ratio when smaller, sides snapped to 32 (half even).
pub fn canvas(width: usize, height: usize, min_area: u64) !Size {
    if (width == 0 or height == 0) return error.H3BadPictureSize;
    if (min_area < min_area_floor or min_area > min_area_ceiling) return error.H3BadMinArea;
    const w: f64 = @floatFromInt(width);
    const h: f64 = @floatFromInt(height);
    const scale = @max(1.0, @sqrt(@as(f64, @floatFromInt(min_area)) / (w * h)));
    const side = struct {
        fn snap(v: f64) !usize {
            const n = @max(1.0, roundEven(v / row_pixels));
            if (n > @as(f64, @floatFromInt(geometry.scalar_max / row_pixels))) return error.H3BadPictureSize;
            return @as(usize, @intFromFloat(n)) * row_pixels;
        }
    };
    return .{ .width = try side.snap(w * scale), .height = try side.snap(h * scale) };
}

/// The distilled schedule (`pdd_config.json`): its sigmas round-tripped through float32 timesteps.
pub const Pdd = struct {
    /// The run's sigmas (steps + 1, the last 0) and the timesteps the model sees, t = 1 - sigma exactly.
    sigmas: []f32,
    timesteps: []f32,
    /// Each step's Euler size, sigma_k - sigma_k+1 in float32.
    deltas: []f32,

    pub fn steps(p: Pdd) usize {
        return p.timesteps.len;
    }
};

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.H3BadPdd,
    };
}

/// `pdd_config.json`: `num_inference_steps` and its steps + 1 `sigmas`, falling strictly from 1 to 0.
pub fn pdd(a: std.mem.Allocator, json: []const u8) !Pdd {
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return error.H3BadPdd;
    if (doc != .object) return error.H3BadPdd;
    const n_field = doc.object.get("num_inference_steps") orelse return error.H3BadPdd;
    if (n_field != .integer or n_field.integer < 1 or n_field.integer > 1024) return error.H3BadPdd;
    const n: usize = @intCast(n_field.integer);
    const list = doc.object.get("sigmas") orelse return error.H3BadPdd;
    if (list != .array or list.array.items.len != n + 1) return error.H3PddSigmaCount;
    const raw = try a.alloc(f64, n + 1);
    for (raw, list.array.items) |*r, v| {
        r.* = try number(v);
        if (!std.math.isFinite(r.*)) return error.H3BadPdd;
    }
    if (raw[0] != 1.0 or raw[n] != 0.0) return error.H3PddEnds;
    for (raw[0..n], raw[1..]) |x, y| if (!(y < x)) return error.H3PddNotFalling;
    return roundTrip(a, raw);
}

/// Each sigma through a float32 timestep: u = f32(1 - sigma), sigma' = 1 - u and t = 1 - sigma' in float32.
pub fn roundTrip(a: std.mem.Allocator, raw: []const f64) !Pdd {
    const n = raw.len - 1;
    const sigmas = try a.alloc(f32, n + 1);
    const timesteps = try a.alloc(f32, n);
    const deltas = try a.alloc(f32, n);
    for (raw[0..n], 0..) |s, k| {
        const u: f32 = @floatCast(1.0 - s);
        sigmas[k] = 1.0 - u;
        timesteps[k] = 1.0 - sigmas[k];
    }
    sigmas[n] = 0;
    for (deltas, 0..) |*d, k| d.* = sigmas[k] - sigmas[k + 1];
    return .{ .sigmas = sigmas, .timesteps = timesteps, .deltas = deltas };
}

/// The plain Euler step the distilled heads were trained with: x + delta * v, in float32.
pub fn euler(x: []f32, velocity: []const f32, delta: f32) void {
    for (x, velocity) |*o, v| o.* += delta * v;
}

/// A picture block's latent grid, in cells (each side even: two cells a row's side).
pub const Grid = struct { rows: usize, columns: usize };

pub fn gridOf(s: Size) !Grid {
    if (s.width % row_pixels != 0 or s.height % row_pixels != 0 or s.width == 0 or s.height == 0) return error.H3BadPictureSize;
    return .{ .rows = s.height / cell_pixels, .columns = s.width / cell_pixels };
}

/// Rows of a grid: one per 2 x 2 cells.
pub fn gridRows(g: Grid) !usize {
    if (g.rows == 0 or g.columns == 0 or g.rows % 2 != 0 or g.columns % 2 != 0) return error.H3BadPictureGrid;
    return geometry.product(&.{ g.rows / 2, g.columns / 2 });
}

/// Entry k of a block's centred axis: cells over the block's area, cells / 2 entries, times 32, in float64.
fn axis(cells: usize, area: f64, k: usize) f64 {
    const r = @as(f64, @floatFromInt(cells)) / area;
    const left = (1.0 - r) / 2.0;
    const right = left + r;
    const n: f64 = @floatFromInt(cells / 2);
    return (left + @as(f64, @floatFromInt(k)) * ((right - left) / n)) * 32.0;
}

/// The packed rows of a picture: `text` rows, then each reference's rows, then the target's.
pub const Layout = struct {
    text: usize,
    references: []const Grid,
    target: Grid,

    pub fn referenceRows(l: Layout) !usize {
        var n: usize = 0;
        for (l.references) |g| n = try geometry.sum(&.{ n, try gridRows(g) });
        return n;
    }

    pub fn rows(l: Layout) !usize {
        return geometry.sum(&.{ l.text, try l.referenceRows(), try gridRows(l.target) });
    }
};

/// Each row's (t, h, w): text i at (i, 0, 0), reference k at t = text + k, the target after the references.
pub fn positions(a: std.mem.Allocator, l: Layout) ![][3]f32 {
    const out = try a.alloc([3]f32, try l.rows());
    for (out[0..l.text], 0..) |*p, i| p.* = .{ @floatFromInt(i), 0, 0 };
    var at = l.text;
    for (l.references, 0..) |g, k| at = block(out, at, g, l.text + k);
    _ = block(out, at, l.target, l.text + l.references.len);
    return out;
}

fn block(out: [][3]f32, from: usize, g: Grid, t: usize) usize {
    const area = @sqrt(@as(f64, @floatFromInt(g.rows)) * @as(f64, @floatFromInt(g.columns)));
    var at = from;
    for (0..g.rows / 2) |r| for (0..g.columns / 2) |c| {
        out[at] = .{ @floatFromInt(t), @floatCast(axis(g.rows, area, r)), @floatCast(axis(g.columns, area, c)) };
        at += 1;
    };
    return at;
}

/// Per row, the cosines and sines of 48 angles (position times `rope.inv_freq`), each laid twice over 96 dims.
pub fn rotary(a: std.mem.Allocator, p: []const [3]f32, inv_freq: []const f32) !struct { cos: []f32, sin: []f32 } {
    if (inv_freq.len != axis_freqs) return error.H3BadRotary;
    const n = try geometry.product(&.{ p.len, rotated });
    const cos = try a.alloc(f32, n);
    const sin = try a.alloc(f32, n);
    for (p, 0..) |row, i| for (0..3) |ax| for (inv_freq, 0..) |f, j| {
        const angle = row[ax] * f;
        const at = i * rotated + ax * axis_freqs + j;
        cos[at] = @cos(angle);
        sin[at] = @sin(angle);
        cos[at + angles] = cos[at];
        sin[at + angles] = sin[at];
    };
    return .{ .cos = cos, .sin = sin };
}

/// A latent [24, rows, columns] (channel-major) as rows of 96: patches row-major, each channel's 2 x 2 in order.
pub fn patchify(latent: []const f32, g: Grid, out: []f32) !void {
    const n = try gridRows(g);
    if (latent.len != n * row_width or out.len != latent.len) return error.H3BadLatentLength;
    const plane = g.rows * g.columns;
    for (0..g.rows / 2) |pr| for (0..g.columns / 2) |pc| for (0..channels) |c| for (0..2) |dy| for (0..2) |dx| {
        out[(pr * (g.columns / 2) + pc) * row_width + c * 4 + dy * 2 + dx] = latent[c * plane + (2 * pr + dy) * g.columns + 2 * pc + dx];
    };
}

/// patchify's inverse.
pub fn unpatchify(rows: []const f32, g: Grid, out: []f32) !void {
    const n = try gridRows(g);
    if (rows.len != n * row_width or out.len != rows.len) return error.H3BadLatentLength;
    const plane = g.rows * g.columns;
    for (0..g.rows / 2) |pr| for (0..g.columns / 2) |pc| for (0..channels) |c| for (0..2) |dy| for (0..2) |dx| {
        out[c * plane + (2 * pr + dy) * g.columns + 2 * pc + dx] = rows[(pr * (g.columns / 2) + pc) * row_width + c * 4 + dy * 2 + dx];
    };
}

/// A dense forward's tile map: the prefix's rows (text and references) in tiles of up to 64, then the target's.
pub const Tiles = struct { slot: []i32, sizes: []i32, prefix: usize, target: usize };

pub fn tiles(a: std.mem.Allocator, prefix_rows: usize, target_rows: usize) !Tiles {
    if (prefix_rows == 0 or target_rows == 0) return error.H3EmptyMedia;
    const rows = try geometry.sum(&.{ prefix_rows, target_rows });
    const prefix = (prefix_rows + geometry.tile_rows - 1) / geometry.tile_rows;
    const target = (target_rows + geometry.tile_rows - 1) / geometry.tile_rows;
    if (try geometry.sum(&.{ prefix, target }) > geometry.tile_max) return error.H3BadTiles;
    const slot = try a.alloc(i32, rows);
    const sizes = try a.alloc(i32, prefix + target);
    for (slot[0..prefix_rows], 0..) |*s, r| s.* = @intCast(r);
    for (slot[prefix_rows..], 0..) |*s, r| s.* = @intCast(prefix * geometry.tile_rows + r);
    for (sizes[0..prefix], 0..) |*s, t| s.* = @intCast(@min(geometry.tile_rows, prefix_rows - t * geometry.tile_rows));
    for (sizes[prefix..], 0..) |*s, t| s.* = @intCast(@min(geometry.tile_rows, target_rows - t * geometry.tile_rows));
    return .{ .slot = slot, .sizes = sizes, .prefix = prefix, .target = target };
}

fn field(o: std.json.ObjectMap, key: []const u8) !std.json.Value {
    return o.get(key) orelse error.H3NotPictureModel;
}

/// A MiniMax-H3-Image folder's `model_index.json`: class and partition checked; returns the step heads' file.
pub fn checkIndex(a: std.mem.Allocator, json: []const u8) ![]const u8 {
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return error.H3NotPictureModel;
    if (doc != .object) return error.H3NotPictureModel;
    const class = try field(doc.object, "_class_name");
    if (class != .string or !std.mem.eql(u8, class.string, "MiniMaxH3ImagePipeline")) return error.H3NotPictureModel;
    const ext = try field(doc.object, "_minimax_h3");
    if (ext != .object) return error.H3NotPictureModel;
    const partition = try field(ext.object, "partition");
    if (partition != .string or !std.mem.eql(u8, partition.string, "image")) return error.H3NotPictureModel;
    const heads = try field(ext.object, "pdd_fused_heads");
    if (heads != .string or heads.string.len == 0 or std.mem.indexOf(u8, heads.string, "..") != null or heads.string[0] == '/')
        return error.H3NotPictureModel;
    return heads.string;
}

/// `transformer/config.json`: audio off, one-frame 2 x 2 patches of 24 channels, 16 rotary frequencies an axis.
pub fn checkTransformer(a: std.mem.Allocator, json: []const u8) !void {
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return error.H3NotPictureModel;
    if (doc != .object) return error.H3NotPictureModel;
    const o = doc.object;
    const audio = try field(o, "audio_enabled");
    if (audio != .bool or audio.bool) return error.H3PictureHasAudio;
    const ints = .{ .{ "latents_dim", channels }, .{ "rope_inv_freq_len", axis_freqs }, .{ "attention_head_dim", 128 } };
    inline for (ints) |kv| {
        const v = try field(o, kv[0]);
        if (v != .integer or v.integer != kv[1]) return error.H3UnsupportedPictureModel;
    }
    const patch = try field(o, "patch_size");
    if (patch != .array or patch.array.items.len != 3) return error.H3UnsupportedPictureModel;
    for (patch.array.items, [_]i64{ 1, 2, 2 }) |v, want| if (v != .integer or v.integer != want) return error.H3UnsupportedPictureModel;
}

/// The shard count of `transformer/model.safetensors.index.json`: every file named model-NNNNN-of-COUNT.
pub fn shardCount(a: std.mem.Allocator, json: []const u8) !usize {
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return error.H3BadShardIndex;
    if (doc != .object) return error.H3BadShardIndex;
    const map = doc.object.get("weight_map") orelse return error.H3BadShardIndex;
    if (map != .object or map.object.count() == 0) return error.H3BadShardIndex;
    var count: ?usize = null;
    for (map.object.values()) |v| {
        if (v != .string) return error.H3BadShardIndex;
        const name = v.string;
        const of = std.mem.indexOf(u8, name, "-of-") orelse return error.H3BadShardIndex;
        if (!std.mem.startsWith(u8, name, "model-") or !std.mem.endsWith(u8, name, ".safetensors")) return error.H3BadShardIndex;
        const n = std.fmt.parseInt(usize, name[of + 4 .. name.len - ".safetensors".len], 10) catch return error.H3BadShardIndex;
        if (n == 0 or (count != null and count.? != n)) return error.H3BadShardIndex;
        count = n;
    }
    return count.?;
}

/// Ours against a reference in float64: cosine, relative error, and the worst row's relative error.
pub const Distance = struct { cosine: f64, relative: f64, worst_row: f64 };

pub fn distance(ours: []const f32, theirs: []const f32, width: usize) !Distance {
    if (ours.len != theirs.len or width == 0 or ours.len % width != 0 or ours.len == 0) return error.H3BadLatentLength;
    var dot: f64 = 0;
    var aa: f64 = 0;
    var bb: f64 = 0;
    var dd: f64 = 0;
    var worst: f64 = 0;
    var row: usize = 0;
    while (row < ours.len) : (row += width) {
        var rd: f64 = 0;
        var rb: f64 = 0;
        for (ours[row..][0..width], theirs[row..][0..width]) |x, y| {
            dot += @as(f64, x) * y;
            aa += @as(f64, x) * x;
            bb += @as(f64, y) * y;
            rd += (@as(f64, x) - y) * (@as(f64, x) - y);
            rb += @as(f64, y) * y;
        }
        dd += rd;
        worst = @max(worst, @sqrt(rd / @max(rb, 1e-30)));
    }
    return .{ .cosine = dot / @sqrt(@max(aa * bb, 1e-300)), .relative = @sqrt(dd / @max(bb, 1e-300)), .worst_row = worst };
}

/// The gates of the reference comparison, fixed before any run (docs/recipes/minimax-h3-image-metal.md).
pub const gates = struct {
    /// Teacher-forced: each step's target velocity on the reference's own input rows.
    pub const step_cosine: f64 = 0.999;
    pub const step_relative: f64 = 0.03;
    /// Free-running: our final latent rows after our own six steps.
    pub const latent_cosine: f64 = 0.99;
    /// The rotary table's bf16 values may differ from the reference's by at most this many ULPs.
    pub const rotary_ulps: u32 = 1;
};

pub fn stepPasses(d: Distance) bool {
    return d.cosine >= gates.step_cosine and d.relative <= gates.step_relative;
}

/// How many bf16 ULPs apart two bf16 values are (sign-magnitude ordered), the way the rotary gate counts them.
pub fn ulps(a: u16, b: u16) u32 {
    const key = struct {
        fn of(v: u16) i32 {
            return if (v & 0x8000 != 0) -@as(i32, v & 0x7fff) else @as(i32, v);
        }
    };
    return @abs(key.of(a) - key.of(b));
}

/// A float32 table's values rounded to bf16, as the heads kernel rounds them.
pub fn bf16(v: f32) u16 {
    return @import("bf16_round").bits(v);
}
