//! MiniMax-H3-Image's image decoder on the host: the tensors it reads, a tile's sequence and rotary table, unpatchify.
const std = @import("std");
const st = @import("safetensors");
const rounding = @import("bf16_round");
const vae = @import("picture_vae.zig");
const geometry = @import("geometry.zig");

pub const width = vae.width;
pub const heads = vae.heads;
pub const head_dim = vae.head_dim;
pub const ffn = vae.ffn;
pub const channels = vae.latent_channels;
/// A token's output: 3 colours of a 16 x 16 pixel patch, channel-major.
pub const patch = 3 * vae.cell_pixels * vae.cell_pixels;
/// Tokens after a tile's cells: the 4 registers, then one zero token.
pub const suffix = vae.registers + 1;
/// The largest tile the attention kernel holds a side, in cells (256 pixels), and so its most tokens.
pub const most_cells = 16;
pub const most_tokens = most_cells * most_cells + suffix;
/// The attention scale, 1 / sqrt(64), and the angle scale 2 pi as fp32 holds it.
pub const scale: f32 = 0.125;
pub const two_pi: f32 = 2 * std.math.pi;

/// How the self-test's synthetic weights fill a tensor: a matrix by its fan-in, a bias, a gain near 1, others.
pub const Fill = enum { matrix, bias, gain, layer_scale, tokens };

/// A checkpoint tensor the decoder reads: its name (after the block prefix for a block's), F32 shape, synthetic fill.
pub const Part = struct { name: []const u8, shape: []const usize, fill: Fill };

pub const Head = enum { post_w, post_b, embed_w, embed_b, registers };
pub const Block = enum { norm1, qkv_w, qkv_b, out_w, out_b, scale1, norm2, w1_w, w1_b, w2_w, w2_b, scale2 };
pub const Tail = enum { norm_w, norm_b, proj_w, proj_b };

pub const head_parts = [_]Part{
    .{ .name = "post_quant_conv.weight", .shape = &.{ channels, channels, 1, 1 }, .fill = .matrix },
    .{ .name = "post_quant_conv.bias", .shape = &.{channels}, .fill = .bias },
    .{ .name = "decoder.x_embedder.weight", .shape = &.{ width, channels }, .fill = .matrix },
    .{ .name = "decoder.x_embedder.bias", .shape = &.{width}, .fill = .bias },
    .{ .name = "decoder.register_tokens", .shape = &.{ 1, vae.registers, width }, .fill = .tokens },
};

/// A block's tensors under `decoder.transformer_blocks.N.`; to_qkv's rows are each head's [q 64 | k 64 | v 64].
pub const block_parts = [_]Part{
    .{ .name = "norm1.weight", .shape = &.{width}, .fill = .gain },
    .{ .name = "attn.to_qkv.weight", .shape = &.{ 3 * width, width }, .fill = .matrix },
    .{ .name = "attn.to_qkv.bias", .shape = &.{3 * width}, .fill = .bias },
    .{ .name = "attn.to_out.weight", .shape = &.{ width, width }, .fill = .matrix },
    .{ .name = "attn.to_out.bias", .shape = &.{width}, .fill = .bias },
    .{ .name = "scale1", .shape = &.{width}, .fill = .layer_scale },
    .{ .name = "norm2.weight", .shape = &.{width}, .fill = .gain },
    .{ .name = "ff.w1.weight", .shape = &.{ 2 * ffn, width }, .fill = .matrix },
    .{ .name = "ff.w1.bias", .shape = &.{2 * ffn}, .fill = .bias },
    .{ .name = "ff.w2.weight", .shape = &.{ width, ffn }, .fill = .matrix },
    .{ .name = "ff.w2.bias", .shape = &.{width}, .fill = .bias },
    .{ .name = "scale2", .shape = &.{width}, .fill = .layer_scale },
};

pub const tail_parts = [_]Part{
    .{ .name = "decoder.norm_out.weight", .shape = &.{width}, .fill = .gain },
    .{ .name = "decoder.norm_out.bias", .shape = &.{width}, .fill = .bias },
    .{ .name = "decoder.proj_out.weight", .shape = &.{ patch, width }, .fill = .matrix },
    .{ .name = "decoder.proj_out.bias", .shape = &.{patch}, .fill = .bias },
};

/// The one decoder tensor in the checkpoint the decoder does not read (a training artefact).
pub const unused = "decoder.mask_token";
pub const block_prefix = "decoder.transformer_blocks.";

pub fn count(p: Part) usize {
    var n: usize = 1;
    for (p.shape) |d| n *= d;
    return n;
}

/// Each tensor starts on 128 bf16 values (256 bytes), so every kernel read of it is aligned.
const alignment = 128;

fn starts(comptime parts: []const Part) [parts.len + 1]usize {
    var out: [parts.len + 1]usize = undefined;
    out[0] = 0;
    for (parts, 0..) |p, i| out[i + 1] = out[i] + std.mem.alignForward(usize, count(p), alignment);
    return out;
}

const head_at = starts(&head_parts);
const block_at = starts(&block_parts);
const tail_at = starts(&tail_parts);

/// Where each tensor sits in the decoder's one bf16 buffer, in values: the head, `layers` blocks, the tail.
pub const Plan = struct {
    layers: usize,

    pub fn init(layers: usize) !Plan {
        if (layers == 0 or layers > 1024) return error.H3UnsupportedVae;
        return .{ .layers = layers };
    }

    pub fn head(_: Plan, part: Head) usize {
        return head_at[@backingInt(part)];
    }

    pub fn block(_: Plan, layer: usize, part: Block) usize {
        return head_at[head_parts.len] + layer * block_at[block_parts.len] + block_at[@backingInt(part)];
    }

    pub fn tail(p: Plan, part: Tail) usize {
        return head_at[head_parts.len] + p.layers * block_at[block_parts.len] + tail_at[@backingInt(part)];
    }

    pub fn total(p: Plan) usize {
        return p.tail(.norm_w) + tail_at[tail_parts.len];
    }

    /// Every tensor's checkpoint name, part and place, in buffer order.
    pub fn items(p: Plan, a: std.mem.Allocator) ![]Item {
        const out = try a.alloc(Item, head_parts.len + p.layers * block_parts.len + tail_parts.len);
        var at: usize = 0;
        for (head_parts, 0..) |part, i| {
            out[at] = .{ .name = part.name, .part = part, .offset = head_at[i] };
            at += 1;
        }
        for (0..p.layers) |layer| for (block_parts, 0..) |part, i| {
            const name = try std.fmt.allocPrint(a, "{s}{d}.{s}", .{ block_prefix, layer, part.name });
            out[at] = .{ .name = name, .part = part, .offset = p.block(layer, @fromBackingInt(@intCast(i))) };
            at += 1;
        };
        for (tail_parts, 0..) |part, i| {
            out[at] = .{ .name = part.name, .part = part, .offset = p.tail(@fromBackingInt(@intCast(i))) };
            at += 1;
        }
        return out;
    }
};

pub const Item = struct { name: []const u8, part: Part, offset: usize };

/// Each of `items` in `header`, F32 at its shape, and no other decoder tensor but `unused`; `culprit` names a miss.
pub fn check(a: std.mem.Allocator, header: *const st.Header, items: []const Item, culprit: *[]const u8) !void {
    var known: std.StringHashMapUnmanaged(void) = .empty;
    defer known.deinit(a);
    for (items) |it| {
        try known.put(a, it.name, {});
        culprit.* = it.name;
        const e = header.get(it.name) orelse return error.H3MissingVaeTensor;
        if (e.dtype != .f32 or e.rank != it.part.shape.len or !std.mem.eql(usize, e.shape[0..e.rank], it.part.shape))
            return error.H3BadVaeTensor;
    }
    for (header.keys()) |name| {
        culprit.* = name;
        const ours = std.mem.startsWith(u8, name, "decoder.") or std.mem.startsWith(u8, name, "post_quant_conv.");
        if (ours and !known.contains(name) and !std.mem.eql(u8, name, unused)) return error.H3UnexpectedVaeTensor;
    }
    culprit.* = "";
}

/// A tile of h x w cells: both sides from 1 to 16 cells; returns its token count, h * w + the suffix.
pub fn tokens(h: usize, w: usize) !usize {
    if (h == 0 or w == 0 or h > most_cells or w > most_cells) return error.H3UnsupportedVae;
    return h * w + suffix;
}

/// What token `t` of a tile of `cells` cells holds.
pub const Token = union(enum) { cell: usize, register: usize, zero };

pub fn token(t: usize, cells: usize) !Token {
    if (t < cells) return .{ .cell = t };
    if (t < cells + vae.registers) return .{ .register = t - cells };
    if (t == cells + vae.registers) return .zero;
    return error.H3BadTokenIndex;
}

/// Cell i's coordinate on an axis of n cells, 2 ((i + 0.5) / n) - 1: each step in fp32, then rounded to bf16.
pub fn axisId(i: usize, n: usize) f32 {
    const half = rounding.round32(@as(f32, @floatFromInt(i)) + 0.5);
    const unit = rounding.round32(half / @as(f32, @floatFromInt(n)));
    return rounding.round32(2 * unit - 1);
}

/// An axis's 8 inverse frequencies, 1 / theta^(j * 4 / 32): the power rounded to fp32, then its fp32 reciprocal.
pub fn inverseFrequencies(theta: f32) [vae.axis_freqs]f32 {
    var out: [vae.axis_freqs]f32 = undefined;
    for (&out, 0..) |*f, j| {
        const e = @as(f32, @floatFromInt(j * 4)) / @as(f32, vae.rotated);
        const p: f32 = @floatCast(std.math.pow(f64, theta, e));
        f.* = 1.0 / p;
    }
    return out;
}

/// A token's 32 angles, [h-axis 8 | w-axis 8] twice: 2 pi times the id rounded to bf16, times each frequency in fp32.
pub fn angles(row_id: f32, col_id: f32, inv: [vae.axis_freqs]f32) [vae.rotated]f32 {
    var out: [vae.rotated]f32 = undefined;
    const base = [2]f32{ rounding.round32(two_pi * row_id), rounding.round32(two_pi * col_id) };
    for (base, 0..) |b, ax| for (inv, 0..) |f, j| {
        out[ax * vae.axis_freqs + j] = b * f;
        out[vae.rotated / 2 + ax * vae.axis_freqs + j] = b * f;
    };
    return out;
}

/// fp32 cosine or sine of an fp32 angle, rounded to bf16 bits.
fn trig(v: f32, comptime cos: bool) u16 {
    const x: f64 = v;
    const r: f32 = @floatCast(if (cos) @cos(x) else @sin(x));
    return rounding.bits(r);
}

/// A tile's cos and sin tables [tokens, 32] in bf16: its cells row-major, then the suffix at (0, 0).
pub fn rotary(h: usize, w: usize, theta: f32, cos: []u16, sin: []u16) !void {
    const n = try tokens(h, w);
    if (cos.len != n * vae.rotated or sin.len != cos.len) return error.H3BadRotary;
    const inv = inverseFrequencies(theta);
    for (0..n) |t| {
        const ids: [2]f32 = switch (try token(t, h * w)) {
            .cell => |c| .{ axisId(c / w, h), axisId(c % w, w) },
            .register, .zero => .{ 0, 0 },
        };
        for (angles(ids[0], ids[1], inv), 0..) |v, i| {
            cos[t * vae.rotated + i] = trig(v, true);
            sin[t * vae.rotated + i] = trig(v, false);
        }
    }
}

/// proj_out's rows [h * w, 768] (bf16 bits) to a tile's pixels [3, 16 h, 16 w]: a row is colour, then row, then column.
pub fn unpatchify(rows: []const u16, h: usize, w: usize, out: []f32) !void {
    const side = vae.cell_pixels;
    if (rows.len != h * w * patch or out.len != rows.len) return error.H3BadLatentLength;
    const pw = w * side;
    const plane = h * side * pw;
    for (0..h) |r| for (0..w) |c| for (0..3) |ch| for (0..side) |y| for (0..side) |x| {
        const v = rows[(r * w + c) * patch + (ch * side + y) * side + x];
        out[ch * plane + (r * side + y) * pw + c * side + x] = rounding.value(v);
    };
}

/// A latent's tiles under `c`: both axes' spans and the cells a tile side, every tile the same size.
pub const Tiles = struct {
    rows: []vae.Span,
    cols: []vae.Span,
    h: usize,
    w: usize,

    pub fn total(t: Tiles) usize {
        return t.rows.len * t.cols.len;
    }

    /// Tile k in row-major order: its first latent cell (row, column) and its place in the assembly.
    pub fn at(t: Tiles, k: usize) struct { row: usize, col: usize, r: usize, c: usize } {
        const r = k / t.cols.len;
        const c = k % t.cols.len;
        return .{ .row = t.rows[r].start / vae.cell_pixels, .col = t.cols[c].start / vae.cell_pixels, .r = r, .c = c };
    }
};

/// A latent of `rows` x `columns` cells (at most 2^24) in tiles of at most 16 cells a side.
pub fn tiles(a: std.mem.Allocator, c: *const vae.Config, rows: usize, columns: usize) !Tiles {
    if (rows == 0 or columns == 0) return error.H3BadPictureGrid;
    if (try geometry.product(&.{ rows, columns }) > 1 << 24) return error.H3BadPictureGrid;
    const r = try vae.split(a, rows * vae.cell_pixels, c.tile, c.overlap);
    const k = try vae.split(a, columns * vae.cell_pixels, c.tile, c.overlap);
    const out: Tiles = .{ .rows = r, .cols = k, .h = r[0].len / vae.cell_pixels, .w = k[0].len / vae.cell_pixels };
    _ = try tokens(out.h, out.w);
    return out;
}

/// A latent file's float count: 24 x rows x columns.
pub fn latentLength(rows: usize, columns: usize) !usize {
    return geometry.product(&.{ channels, rows, columns });
}

pub const Size = struct { width: usize, height: usize };

/// "WIDTHxHEIGHT", each side from 1 to 65,536 pixels.
pub fn size(text: []const u8) !Size {
    const x = std.mem.indexOfScalar(u8, text, 'x') orelse return error.H3BadPictureSize;
    const w = std.fmt.parseInt(usize, text[0..x], 10) catch return error.H3BadPictureSize;
    const h = std.fmt.parseInt(usize, text[x + 1 ..], 10) catch return error.H3BadPictureSize;
    if (w == 0 or h == 0 or w > 65536 or h > 65536) return error.H3BadPictureSize;
    return .{ .width = w, .height = h };
}
