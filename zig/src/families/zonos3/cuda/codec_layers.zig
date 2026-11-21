//! The codec's causal transformer layer, shared by the decoder's and the encoder's stages.
const cuda = @import("cuda");
const kern = @import("codec_kernels.zig");
const wts = @import("codec_weights.zig");

/// A layer's scratch rows: the norm's output, a projection's output, attention's output, and the wide rows.
pub const Scratch = struct { n: u64, o: u64, att: u64, big: u64 };

/// A causal ConvNeXt block on x [rows][width]; hist [width][6] holds the six inputs before these rows.
pub fn convnext(k: *const kern.Kernels, s: cuda.Stream, sc: Scratch, b: wts.Block, hist: u64, x: u64, rows: usize) !void {
    try k.dwconv(s, x, rows, b.dw_w, b.dw_b, hist, sc.o, wts.width);
    try k.layernorm(s, sc.o, b.ln_w, b.ln_b, sc.n, wts.width, 1e-6, rows);
    try k.linear(s, b.pw1_w, b.pw1_b, sc.n, wts.width, wts.width, wts.hidden, sc.big, wts.hidden, rows);
    try k.gelu(s, sc.big, rows * wts.hidden);
    try k.linear(s, b.pw2_w, b.pw2_b, sc.big, wts.hidden, wts.hidden, wts.width, sc.o, wts.width, rows);
    try k.addScaled(s, x, sc.o, 0, wts.width, rows);
}

/// x [rows][d] at positions pos0 on: x += scale * attention(norm(x)), then x += scale * ffn(norm(x)).
pub fn layer(k: *const kern.Kernels, s: cuda.Stream, sc: Scratch, shape: wts.StageShape, L: wts.Layer, kv: [2]u64, x: u64, rows: usize, pos0: usize) !void {
    const d = shape.d;
    try k.layernorm(s, x, L.n1w, L.n1b, sc.n, d, 1e-5, rows);
    try k.linear(s, L.qkv, 0, sc.n, d, d, 3 * d, sc.big, 3 * d, rows);
    try k.attnPrep(s, sc.big, d, pos0, kv[0], kv[1], rows);
    try k.attention(s, sc.big, d, kv[0], kv[1], pos0, shape.ctx, sc.att, rows);
    try k.linear(s, L.attn_out, 0, sc.att, d, d, d, sc.o, d, rows);
    try k.addScaled(s, x, sc.o, L.attn_scale, d, rows);
    try k.layernorm(s, x, L.n2w, L.n2b, sc.n, d, 1e-5, rows);
    try k.linear(s, L.ffn0, 0, sc.n, d, d, shape.ff, sc.big, shape.ff, rows);
    try k.gelu(s, sc.big, rows * shape.ff);
    try k.linear(s, L.ffn1, 0, sc.big, shape.ff, shape.ff, d, sc.o, d, rows);
    try k.addScaled(s, x, sc.o, L.ffn_scale, d, rows);
}
