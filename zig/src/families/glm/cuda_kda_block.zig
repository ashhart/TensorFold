//! GLM-5.3-Flash's KDA block on CUDA, a rank's share of the Python family's kda_block: projection, chain, o_proj.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const kda = @import("cuda_kda.zig");
const b16 = @import("cuda_b16.zig");
const prof = @import("cuda_prof.zig");

/// One KDA layer, `heads` heads: proj bf16 [3 heads 128 + 256 + heads][dims], f_b, g_b, o, the chain's taps and norm.
pub const Layer = struct {
    proj: u64,
    fb: u64,
    gb: u64,
    o: u64,
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

    /// The input stack and its two views need glm_mm's eight-column output and row stride.
    pub fn tiledInput(l: Layer) bool {
        return l.width() % 8 == 0;
    }
};

/// The block's scratch for `rows` rows: projection, decay and gate rows, chain saves and scratch, B16 split-K partials.
pub const Scratch = struct { p: u64, ka: u64, kg: u64, saves: kda.Saves, wide: kda.Wide, mm_part: u64 };

/// forward.kda_block: out [rows][dims] fp32 partial; `wide`: three-kernel chain; `prompt`: commits the conv window.
pub fn block(set: glue.Set, ks: *const kda.Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, conv_state: u64, state_in: u64, wide: bool, prompt: bool, out: u64) !void {
    try body(set, ks, stream, l, s, x, x_stride, rows, conv_state, state_in, wide, prompt);
    try fill(set, stream, l, s, 0, rows, out);
}

/// forward.out_proj's fill for the KDA block: o_proj of rows lo..hi (s.saves.out) into out [hi - lo][dims] fp32.
pub fn fill(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, lo: usize, hi: usize, out: u64) !void {
    const n = l.heads * kda.dk;
    try b16.matmul(set, stream, s.saves.out + lo * n * 2, n, l.o, out, s.mm_part, hi - lo, l.dims, n, true);
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
    const w = l.width();
    const fa = 3 * n;
    try b16.matmul(selected, stream, x, x_stride, l.proj, s.p, s.mm_part, rows, w, l.dims, false);
    try b16.matmul(selected, stream, s.p + fa * 2, w, l.fb, s.ka, s.mm_part, rows, n, kda.dk, false);
    try b16.matmul(selected, stream, s.p + (fa + kda.dk) * 2, w, l.gb, s.kg, s.mm_part, rows, n, kda.dk, false);
}

/// The chain over `rows` projected rows (s.p, s.ka, s.kg) on one sequence's conv window and recurrent state.
pub fn chainRows(ks: *const kda.Kernels, stream: cuda.Stream, l: Layer, s: Scratch, rows: usize, conv_state: u64, state_in: u64, wide: bool) !void {
    const n = l.heads * kda.dk;
    const fa = 3 * n;
    const c: kda.Chain = .{ .p = s.p, .p_stride = l.width(), .b_off = fa + 2 * kda.dk, .a = s.ka, .a_stride = n, .g = s.kg, .g_stride = n, .conv_state = conv_state, .conv_w = l.conv_w, .state_in = state_in, .a_log = l.a_log, .dt_bias = l.dt_bias, .norm_w = l.norm_w, .eps = l.eps, .lower = l.lower, .heads = l.heads };
    try kda.chain(ks, stream, c, rows, s.saves, if (wide or rows >= kda.wide_rows) s.wide else null);
}

test {
    _ = @import("cuda_kda_tp3_test.zig");
}
