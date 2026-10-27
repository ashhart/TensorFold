//! GLM-5.3-Flash's KDA block on CUDA, a rank's share of the Python family's kda_block: projection, chain, o_proj.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const kda = @import("cuda_kda.zig");
const b16 = @import("cuda_b16.zig");
const prof = @import("cuda_prof.zig");
const projection = @import("cuda_proj.zig");

/// The input projection [q | k | v | f_a | g_a | b] [3 heads 128 + 256 + heads][dims]: one bf16 stack, or two parts.
pub const In = union(enum) {
    bf16: u64,
    split: struct { qkv: cuda.qmmf.Weight, rest: u64 }, // MXFP8 [q | k | v] for the lane (FP4-Local's), bf16 the rest
};

/// One KDA layer, `heads` heads: its input projection, f_b, g_b, o (bf16 or MXFP8), the chain's taps and norm.
pub const Layer = struct {
    proj: In,
    fb: u64,
    gb: u64,
    o: projection.Proj,
    conv_w: u64,
    a_log: u64,
    dt_bias: u64,
    norm_w: u64,
    eps: f32,
    lower: f32,
    heads: usize,
    dims: usize,

    pub fn width(l: Layer) usize {
        return 3 * l.heads * kda.dk + 2 * kda.dk + l.heads;
    }

    /// The width the input stack's rows actually carry: `width` plus any alignment pad (TF_GLM_KDA_PAD).
    pub fn matWidth(l: Layer) usize {
        return paddedWidth(l.width());
    }

    /// The input stack and its two views need glm_mm's eight-column output and row stride.
    pub fn tiledInput(l: Layer) bool {
        return l.matWidth() % 8 == 0;
    }
};

/// The fused input's row width, padded to eight columns when TF_GLM_KDA_PAD asks for it. glm_mm needs the output
/// width and row stride to be multiples of eight; the fused width is 3*heads*dk + 2*dk + heads, which at TP=3 is
/// 8,726 (22 heads) or 8,341 (21) - so without padding those tensors stay bf16 row-major and cost about 9 ms a call.
/// One rule, read by the loader (which pads the weights) and by the state (which sizes the rows), so they agree.
pub fn paddedWidth(raw: usize) usize {
    if (raw % 8 == 0) return raw;
    const v = std.c.getenv("TF_GLM_KDA_PAD") orelse return raw;
    const t = std.mem.trim(u8, std.mem.span(v), " ");
    if (t.len == 0 or std.mem.eql(u8, t, "0")) return raw;
    return std.mem.alignForward(usize, raw, 8);
}

/// The block's scratch for `rows` rows: projection, decay and gate rows, chain saves, B16 partials, a split's tmp.
pub const Scratch = struct { p: u64, ka: u64, kg: u64, saves: kda.Saves, wide: kda.Wide, mm_part: u64, tmp: u64 = 0 };

/// forward.kda_block: out [rows][dims] fp32 partial; `wide`: three-kernel chain; `prompt`: commits the conv window.
pub fn block(set: glue.Set, ks: *const kda.Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, conv_state: u64, state_in: u64, wide: bool, prompt: bool, out: u64) !void {
    try body(set, ks, stream, l, s, x, x_stride, rows, conv_state, state_in, wide, prompt);
    try fill(set, stream, l, s, 0, rows, out);
}

/// forward.out_proj's fill for the KDA block: o_proj of rows lo..hi (s.saves.out) into out [hi - lo][dims] fp32.
pub fn fill(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, lo: usize, hi: usize, out: u64) !void {
    const n = l.heads * kda.dk;
    try projection.matmul(set, stream, l.o, s.saves.out + lo * n * 2, n, out, s.mm_part, hi - lo, l.dims, n, true);
}

/// The block up to its out_proj: the projections, the chain, and a prompt chunk's conv window commit.
pub fn body(set: glue.Set, ks: *const kda.Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, conv_state: u64, state_in: u64, wide: bool, prompt: bool) !void {
    return bodyProfile(set, ks, stream, l, s, x, x_stride, rows, conv_state, state_in, wide, prompt, null);
}

pub fn bodyProfile(set: glue.Set, ks: *const kda.Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, conv_state: u64, state_in: u64, wide: bool, prompt: bool, profile: ?*prof.Prof) !void {
    var stamp = prof.span(profile, stream, .projections);
    try project(set, stream, l, s, x, x_stride, rows);
    stamp.end();
    stamp = prof.span(profile, stream, .kda);
    defer stamp.end();
    try chainRows(ks, stream, l, s, rows, conv_state, state_in, wide);
    if (prompt) try glue.convShift(set, stream, conv_state, s.p, rows, rows * l.width(), l.width(), 1, 3 * l.heads * kda.dk);
}

/// The projections of `rows` rows into s.p, s.ka (decay) and s.kg (gate), row by row so segments share one call.
pub fn project(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize) !void {
    var selected = set;
    if (!l.tiledInput()) selected.mm = null;
    const n = l.heads * kda.dk;
    const w = l.matWidth();
    const fa = 3 * n;
    switch (l.proj) {
        .bf16 => |p| try b16.matmul(selected, stream, x, x_stride, p, s.p, s.mm_part, rows, w, l.dims, false),
        .split => |sp| { // each part on its own kernel into its columns of s.p: one row straight there, more through tmp
            // cc-defer: two row copies a window; an output row stride on the lane and glm_mm if a profile flags them
            const one = rows == 1;
            try projection.matmul(set, stream, .{ .mxfp8 = sp.qkv }, x, x_stride, if (one) s.p else s.tmp, s.mm_part, rows, fa, l.dims, false);
            if (!one) try columns(set, stream, s.tmp, fa, s.p, w, rows);
            // the bf16 rest (2 * dk + heads wide) tiles with the stack, or stays row-major with it when uneven
            try b16.matmul(selected, stream, x, x_stride, sp.rest, if (one) s.p + fa * 2 else s.tmp, s.mm_part, rows, w - fa, l.dims, false);
            if (!one) try columns(set, stream, s.tmp, w - fa, s.p + fa * 2, w, rows);
        },
    }
    try b16.matmul(selected, stream, s.p + fa * 2, w, l.fb, s.ka, s.mm_part, rows, n, kda.dk, false);
    try b16.matmul(selected, stream, s.p + (fa + kda.dk) * 2, w, l.gb, s.kg, s.mm_part, rows, n, kda.dk, false);
}

/// `rows` rows of `cols` bf16 packed at `src` into `dst`, rows `stride` apart (tf_glm_copy_rows).
fn columns(set: glue.Set, stream: cuda.Stream, src: u64, cols: usize, dst: u64, stride: usize, rows: usize) !void {
    const f = set.copy_rows orelse return error.FormatNotLoaded;
    var a: cuda.Args = .{};
    a.add(src);
    a.add(@as(c_longlong, @intCast(cols * 2)));
    a.add(dst);
    a.add(@as(c_longlong, @intCast(stride * 2)));
    a.add(@as(c_int, @intCast(rows)));
    a.add(@as(c_int, @intCast(cols * 2)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((rows * cols * 2 + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
}

/// The chain over `rows` projected rows (s.p, s.ka, s.kg) on one sequence's conv window and recurrent state.
pub fn chainRows(ks: *const kda.Kernels, stream: cuda.Stream, l: Layer, s: Scratch, rows: usize, conv_state: u64, state_in: u64, wide: bool) !void {
    const n = l.heads * kda.dk;
    const fa = 3 * n;
    const c: kda.Chain = .{ .p = s.p, .p_stride = l.matWidth(), .b_off = fa + 2 * kda.dk, .a = s.ka, .a_stride = n, .g = s.kg, .g_stride = n, .conv_state = conv_state, .conv_w = l.conv_w, .state_in = state_in, .a_log = l.a_log, .dt_bias = l.dt_bias, .norm_w = l.norm_w, .eps = l.eps, .lower = l.lower, .heads = l.heads };
    try kda.chain(ks, stream, c, rows, s.saves, if (wide or rows >= kda.wide_rows) s.wide else null);
}

test {
    _ = @import("cuda_kda_tp3_test.zig");
}
