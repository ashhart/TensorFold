//! MiniMax-H3-Image's edit references on the host: sizes, the encoder's input and layers, clean and anchor rows.
const std = @import("std");
const st = @import("safetensors");
const geometry = @import("geometry.zig");
const picture = @import("picture.zig");
const vae = @import("picture_vae.zig");

pub const Size = picture.Size;
pub const Grid = picture.Grid;
const cell = vae.cell_pixels;

/// A reference's sides must be at least this long.
pub const min_side = 32;
/// The short edge of each copy: the VAE's lies in 1024..2048, the Qwen encoder's is exactly 1024.
pub const Bound = struct { least: usize, most: usize };
pub const vae_bound: Bound = .{ .least = 1024, .most = 2048 };
pub const qwen_bound: Bound = .{ .least = 1024, .most = 1024 };
/// The largest VAE copy taken: 64 MP, above any 8:1 reference (2048 x 16384) vLLM-Omni admits.
pub const max_area: u64 = 8192 * 8192;
/// An anchor row keeps this much of the clean row (float32 0.999); the rest is noise.
pub const anchor_clean: f32 = 0.999;
/// 1 - c in float32 (exact), not float32(0.001).
pub const anchor_noise: f32 = 1.0 - anchor_clean;

/// A copy whose short edge lies in `b`: kept when it does, else scaled to the nearer end and snapped to 32.
pub fn bounded(width: usize, height: usize, b: Bound) !Size {
    if (width < min_side or height < min_side) return error.H3ReferenceTooSmall;
    const short = @min(width, height);
    var s: Size = .{ .width = width, .height = height };
    if (short < b.least or short > b.most) {
        const scale = @as(f64, @floatFromInt(if (short < b.least) b.least else b.most)) / @as(f64, @floatFromInt(short));
        s = .{ .width = try snap(width, scale), .height = try snap(height, scale) };
    }
    if (@as(u64, s.width) * s.height > max_area) return error.H3ReferenceTooLarge;
    return s;
}

/// side x scale to the nearest multiple of 32 (halves to even, as Python's round), at least 32.
fn snap(side: usize, scale: f64) !usize {
    const n = @max(1.0, picture.roundEven(@as(f64, @floatFromInt(side)) * scale / picture.row_pixels));
    if (n > @as(f64, @floatFromInt(max_area / picture.row_pixels))) return error.H3ReferenceTooLarge;
    return @as(usize, @intFromFloat(n)) * picture.row_pixels;
}

/// The encoder's input size: the VAE copy cropped top-left to multiples of 16 pixels.
pub fn cropped(s: Size) !Size {
    if (s.width < cell or s.height < cell) return error.H3ReferenceTooSmall;
    return .{ .width = s.width / cell * cell, .height = s.height / cell * cell };
}

/// The encoder's latent grid of a cropped input: a cell every 16 pixels, rows or columns of any parity.
pub fn fullGrid(s: Size) Grid {
    return .{ .rows = s.height / cell, .columns = s.width / cell };
}

/// The reference's grid: the full grid cropped top-left to even sides (a 2 x 2 patch a row).
pub fn evenGrid(full: Grid) !Grid {
    const g: Grid = .{ .rows = full.rows / 2 * 2, .columns = full.columns / 2 * 2 };
    if (g.rows == 0 or g.columns == 0) return error.H3ReferenceTooSmall;
    return g;
}

/// The encoder's input from RGB bytes [height, width, 3]: cropped to 16s, x / 255, ImageNet-normalised, fp32 [h, w, 3].
pub fn input(rgb: []const u8, width: usize, height: usize, out: []f32) !Size {
    const s = try cropped(.{ .width = width, .height = height });
    if (rgb.len != try geometry.product(&.{ width, height, 3 }) or out.len != s.width * s.height * 3) return error.H3BadPictureSize;
    for (0..s.height) |y| for (0..s.width) |x| for (0..3) |c| {
        const v = @as(f32, @floatFromInt(rgb[(y * width + x) * 3 + c])) / 255.0;
        out[(y * s.width + x) * 3 + c] = (v - vae.pixel_mean[c]) / vae.pixel_std[c];
    };
    return s;
}

/// Index `i` of an axis of `n` reflected at its edges (-1 reads 1, n reads n - 2): reflect padding by one.
pub fn reflect(i: isize, n: usize) usize {
    const last: isize = @intCast(n - 1);
    return @intCast(if (i < 0) -i else if (i > last) 2 * last - i else i);
}

pub const levels = 6;
/// Levels 0-3 end with a stride-2 downsample: 16 pixels a cell.
pub const downsampled = 4;
pub const groups = 32;
/// The encoder's output: 24 means, then 24 log-variances.
pub const moments = 2 * vae.latent_channels;

/// The encoder's width a level: the released one's, or smaller ones for a self-test.
pub const Arch = struct { widths: [levels]usize };
pub const released: Arch = .{ .widths = .{ 128, 256, 256, 512, 512, 1024 } };

/// A layer: a 3x3x3 convolution run on its last time slice, a 1x1x1 one, or a GroupNorm.
pub const Kind = enum { conv3, conv1, norm };
/// One layer: its tensors' prefix, its kind, and its channels in and out (equal for a norm).
pub const Item = struct { name: []const u8, kind: Kind, cin: usize, cout: usize };

fn join(a: std.mem.Allocator, prefix: []const u8, part: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{s}{s}", .{ prefix, part });
}

/// Every layer in run order: conv_in, each level's blocks and downsample, norm_out, conv_out, quant_conv.
pub fn items(a: std.mem.Allocator, arch: Arch) ![]Item {
    var list: std.ArrayList(Item) = .empty;
    try list.append(a, .{ .name = "encoder.conv_in", .kind = .conv3, .cin = 3, .cout = arch.widths[0] });
    var cin = arch.widths[0];
    for (arch.widths, 0..) |w, l| {
        if (w == 0 or w % groups != 0) return error.H3BadEncoderWidths;
        for (0..2) |b| {
            const p = try std.fmt.allocPrint(a, "encoder.down.{d}.block.{d}.", .{ l, b });
            try list.appendSlice(a, &.{
                .{ .name = try join(a, p, "norm1"), .kind = .norm, .cin = cin, .cout = cin },
                .{ .name = try join(a, p, "conv1"), .kind = .conv3, .cin = cin, .cout = w },
                .{ .name = try join(a, p, "norm2"), .kind = .norm, .cin = w, .cout = w },
                .{ .name = try join(a, p, "conv2"), .kind = .conv3, .cin = w, .cout = w },
            });
            if (cin != w) try list.append(a, .{ .name = try join(a, p, "nin_shortcut"), .kind = .conv1, .cin = cin, .cout = w });
            cin = w;
        }
        if (l < downsampled) try list.append(a, .{ .name = try std.fmt.allocPrint(a, "encoder.down.{d}.downsample.conv", .{l}), .kind = .conv3, .cin = w, .cout = w });
    }
    try list.appendSlice(a, &.{
        .{ .name = "encoder.norm_out", .kind = .norm, .cin = cin, .cout = cin },
        .{ .name = "encoder.conv_out", .kind = .conv3, .cin = cin, .cout = moments },
        .{ .name = "quant_conv", .kind = .conv1, .cin = moments, .cout = moments },
    });
    return list.items;
}

/// A checkpoint shape of up to five dimensions.
pub const Shape = struct {
    dims: [5]usize,
    rank: usize,

    pub fn slice(s: *const Shape) []const usize {
        return s.dims[0..s.rank];
    }
};

/// The checkpoint's shape of an item's weight: [cout, cin, 3, 3, 3], [cout, cin, 1, 1, 1] or a norm's [c].
pub fn weightShape(it: Item) Shape {
    const k: usize = if (it.kind == .conv3) 3 else 1;
    return .{ .dims = .{ it.cout, it.cin, k, k, k }, .rank = if (it.kind == .norm) 1 else 5 };
}

/// Each layer's F32 weight and bias at its shape, no other `encoder.` or `quant_conv.` tensor; `culprit` names a miss.
pub fn checkHeader(a: std.mem.Allocator, header: *const st.Header, culprit: *[]const u8) !void {
    const list = try items(a, released);
    for (list) |it| for ([_][]const u8{ ".weight", ".bias" }) |part| {
        const name = try join(a, it.name, part);
        culprit.* = name;
        const e = header.get(name) orelse return error.H3MissingEncoderTensor;
        const want: Shape = if (part[1] == 'w') weightShape(it) else .{ .dims = .{ it.cout, 0, 0, 0, 0 }, .rank = 1 };
        if (e.dtype != .f32 or !std.mem.eql(usize, e.shape[0..e.rank], want.slice())) return error.H3BadEncoderTensor;
    };
    var count: usize = 0;
    for (header.keys()) |k| if (std.mem.startsWith(u8, k, "encoder.") or std.mem.startsWith(u8, k, "quant_conv.")) {
        culprit.* = k;
        count += 1;
    };
    if (count != 2 * list.len) return error.H3UnexpectedEncoderTensor;
    culprit.* = "";
}

/// A weight row's length on the GPU: cout padded to 64, the kernel's channel tile.
pub fn rowLength(cout: usize) usize {
    return std.mem.alignForward(usize, cout, 64);
}

/// Taps a layer's kernel has.
pub fn taps(it: Item) usize {
    return if (it.kind == .conv3) 9 else 1;
}

/// A convolution's weight as the GPU reads it, [taps, cin, rowLength] zero-padded; a 3x3x3 gives [:, :, 2].
pub fn pack(it: Item, weight: []align(1) const f32, out: []f32) !void {
    const t = taps(it);
    const ldw = rowLength(it.cout);
    const each: usize = if (t == 9) 27 else 1;
    if (it.kind == .norm or weight.len != it.cout * it.cin * each or out.len != t * it.cin * ldw)
        return error.H3BadEncoderTensor;
    @memset(out, 0);
    for (0..it.cout) |co| for (0..it.cin) |ci| for (0..t) |tap| {
        out[(tap * it.cin + ci) * ldw + co] = if (t == 9) weight[((co * it.cin + ci) * 3 + 2) * 9 + tap] else weight[co * it.cin + ci];
    };
}

/// `vae/config.json` must keep `vae_encoder_tiling` on: the tiles and their blend are part of the encoder's numbers.
pub fn checkTiling(a: std.mem.Allocator, json: []const u8) !void {
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return error.H3BadVaeConfig;
    if (doc != .object) return error.H3BadVaeConfig;
    const v = doc.object.get("vae_encoder_tiling") orelse return error.H3BadVaeConfig;
    if (v != .integer or v.integer != 1) return error.H3UnsupportedVae;
}

pub const Clean = struct { rows: []f32, grid: Grid };

/// Clean rows from moments [48, full grid]: sampled as fp16 with `noise`, normalised, cropped even, patchified.
pub fn clean(a: std.mem.Allocator, c: *const vae.Config, m: []const f32, full: Grid, noise: []const f32) !Clean {
    const n = try geometry.product(&.{ vae.latent_channels, full.rows, full.columns });
    if (m.len != 2 * n or noise.len != n) return error.H3BadLatentLength;
    const sample = try a.alloc(f32, n);
    defer a.free(sample);
    try vae.posterior(m[0..n], m[n..], noise, sample);
    try vae.normalize(c, sample);
    const g = try evenGrid(full);
    const latent = try a.alloc(f32, vae.latent_channels * g.rows * g.columns);
    defer a.free(latent);
    for (0..vae.latent_channels) |ch| for (0..g.rows) |y| {
        @memcpy(latent[(ch * g.rows + y) * g.columns ..][0..g.columns], sample[(ch * full.rows + y) * full.columns ..][0..g.columns]);
    };
    const rows = try a.alloc(f32, latent.len);
    try picture.patchify(latent, g, rows);
    return .{ .rows = rows, .grid = g };
}

/// The anchor rows: c x clean + (1 - c) x noise, c = f32(0.999) and 1 - c taken in fp32, every product in fp32.
pub fn anchors(clean_rows: []const f32, noise_rows: []const f32, out: []f32) !void {
    if (clean_rows.len != noise_rows.len or out.len != clean_rows.len) return error.H3BadLatentLength;
    for (out, clean_rows, noise_rows) |*o, x, z| o.* = anchor_clean * x + anchor_noise * z;
}
