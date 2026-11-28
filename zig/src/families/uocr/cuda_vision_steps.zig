//! Launches of the image encoder's map and attention kernels (uocr_vision.cuh), shared by the engine and its contract.
const cuda = @import("cuda");
const ks = @import("cuda_kernels.zig");

/// A convolution's input rows from a square map: value (c, y, x) at c cstride + (y side + x) pstride.
pub const Cols = struct { channels: usize, in_side: usize, out_side: usize, span: usize, stride: usize, pad: usize, cstride: usize, pstride: usize };

/// SAM's windows: a side x side map padded to `padded`, cut into window x window squares of `channels`.
pub const Window = struct { side: usize, padded: usize, window: usize, channels: usize };

/// Relative positions of `groups` side x side groups of qkv rows `width` wide, heads of 64.
pub const Rel = struct { groups: usize, side: usize, heads: usize, width: usize };

/// Attention within `groups` groups of `len` qkv rows, heads of 64; side > 0 adds relPos's terms to the scores.
pub const Attn = struct { groups: usize, len: usize, heads: usize, width: usize, side: usize, scale: f32 };

/// Rows [out_side^2, channels span^2]: a block a row, 256 columns a block.
pub fn cols(k: *const ks.Kernels, s: cuda.Stream, x: u64, y: u64, p: Cols) !void {
    try k.run("im2col", s, .{ p.out_side * p.out_side, ks.blocks(p.channels * p.span * p.span), 1 }, 256, .{ x, y, p.channels, p.in_side, p.out_side, p.span, p.stride, p.pad, p.cstride, p.pstride });
}

/// The map's rows as the padded map's windows: a block a padded row, 256 channels a block.
pub fn window(k: *const ks.Kernels, s: cuda.Stream, x: u64, y: u64, p: Window) !void {
    try k.run("window", s, .{ p.padded * p.padded, ks.blocks(p.channels), 1 }, 256, .{ x, y, p.side, p.padded, p.window, p.channels });
}

/// The windows' rows added back onto the map in place: a block a map row, 256 channels a block.
pub fn unwindowAdd(k: *const ks.Kernels, s: cuda.Stream, x: u64, win: u64, p: Window) !void {
    try k.run("unwindow_add", s, .{ p.side * p.side, ks.blocks(p.channels), 1 }, 256, .{ x, win, p.side, p.padded, p.window, p.channels });
}

/// bf16 terms [heads, rows, side] for key rows (rel_h) and columns (rel_w): a thread a term, a grid row a head.
pub fn relPos(k: *const ks.Kernels, s: cuda.Stream, qkv: u64, rh: u64, rw: u64, out_h: u64, out_w: u64, p: Rel) !void {
    const rows = p.groups * p.side * p.side;
    try k.run("rel_pos", s, .{ ks.blocks(rows * 2 * p.side), p.heads, 1 }, 256, .{ qkv, rh, rw, out_h, out_w, p.groups, p.side, p.heads, p.width });
}

/// Outputs [groups len, width] bf16: 4 warps of 8 query rows a block, a grid row a head, a grid layer a group.
pub fn attention(k: *const ks.Kernels, s: cuda.Stream, qkv: u64, rel_h: u64, rel_w: u64, o: u64, p: Attn) !void {
    if (p.len > 4096) return error.AttentionRows; // tf_uocr_attention keeps 8 blocks of 512 keys a row
    try k.run("attention", s, .{ (p.len + 31) / 32, p.heads, p.groups }, 128, .{ qkv, rel_h, rel_w, o, p.groups, p.len, p.heads, p.width, p.side, p.scale });
}
