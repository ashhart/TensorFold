//! GLM-5.3-Flash's dense MLP (layers 0-2), a rank's share of the Python family's mlp_block: gate/up, SwiGLU, down.

const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const cuda_pdl = @import("cuda_pdl.zig");
const glue = @import("cuda_glue.zig");
const b16 = @import("cuda_b16.zig");
const projection = @import("cuda_proj.zig");

/// A dense layer's weights: NVFP4, bf16 as stored by the checkpoint, FP4-Local's MXFP8, or TFQ3.
pub const Weights = union(enum) {
    nvfp4: struct { gate: cuda.qmmf.Weight, up: cuda.qmmf.Weight, down: cuda.qmmf.Weight },
    bf16: struct { gu: u64, down: u64 }, // [gate | up] rows [2 width][dims], down [dims][width]
    mxfp8: struct { gu: cuda.qmmf.Weight, down: cuda.qmmf.Weight }, // the same two, packed for the lane
    tfq3: struct { gate: cuda.tfq3.DenseWeight, up: cuda.tfq3.DenseWeight, down: cuda.tfq3.DenseWeight },
};

/// How a dense layer stored as `stored` is held: bf16 (EXL3's, or any left unquantized), e4m3 as MXFP8, else NVFP4.
pub fn held(exl3: bool, stored: core.checkpoint.DType) std.meta.Tag(Weights) {
    if (exl3 or stored == .bf16) return .bf16;
    return if (stored == .f8_e4m3) .mxfp8 else .nvfp4;
}

test "a dense layer stored bf16 is held bf16 in any checkpoint; e4m3 is MXFP8 and FP4 codes are NVFP4" {
    const Kind = std.meta.Tag(Weights);
    try std.testing.expectEqual(Kind.bf16, held(false, .bf16));
    try std.testing.expectEqual(Kind.bf16, held(true, .bf16));
    try std.testing.expectEqual(Kind.mxfp8, held(false, .f8_e4m3));
    try std.testing.expectEqual(Kind.nvfp4, held(false, .u8));
}

/// One dense layer on a rank: gate and up [width][dims] and down [dims][width].
pub const Layer = struct { w: Weights, width: usize, dims: usize, limit: f32 };

/// The block's scratch for `rows` rows: a projection's rows, [gate | up], the activation, its sums, split-K partials.
pub const Scratch = struct { tmp: u64, gu: u64, act: u64, xs_act: u64, mm_part: u64 };

pub const Kernels = struct { set: glue.Set, linear: cuda.qlinear.Linear, copy_rows: cuda.Function };

/// Fp4Linear.__call__ (prefill on a prompt chunk): bf16 out, or fp32 with `wide`; the prompt GEMM needs `ldx` % 8 == 0.
fn mm(k: Kernels, stream: cuda.Stream, w: cuda.qmmf.Weight, x: u64, ldx: usize, rows: usize, out: u64, wide: bool, prompt: bool) !void {
    const q: cuda.qlinear.Weight = .{ .nvfp4 = w };
    const in: cuda.qlinear.Rows = .{ .x = x, .ldx = ldx };
    if (prompt) {
        std.debug.assert(x % 16 == 0 and (rows == 1 or ldx % 8 == 0));
        return if (wide) k.linear.promptF32(stream, q, in, out, rows) else k.linear.prompt(stream, q, in, out, rows);
    }
    return if (wide) k.linear.decodeF32(stream, q, in, out, rows) else k.linear.decode(stream, q, in, out, rows);
}

/// forward.mlp_block for `rows` normed rows (bf16, stride `x_stride`): out [rows][dims] fp32, this rank's partial.
pub fn block(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, prompt: bool, out: u64) !void {
    try body(k, stream, l, s, x, x_stride, rows, prompt);
    try fill(k, stream, l, s, 0, rows, prompt, out);
}

/// forward.out_proj's fill for the dense MLP: down of rows lo..hi (s.act) into out [hi - lo][dims] fp32.
pub fn fill(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, lo: usize, hi: usize, prompt: bool, out: u64) !void {
    switch (l.w) {
        .nvfp4 => |w| try mm(k, stream, w.down, s.act + lo * l.width * 2, l.width, hi - lo, out, true, prompt),
        .bf16 => |w| try b16.matmul(k.set, stream, s.act + lo * l.width * 2, l.width, w.down, out, s.mm_part, hi - lo, l.dims, l.width, true),
        .mxfp8 => |w| try projection.matmul(k.set, stream, .{ .mxfp8 = w.down }, s.act + lo * l.width * 2, l.width, out, s.mm_part, hi - lo, l.dims, l.width, true),
        .tfq3 => |w| try projection.matmul(k.set, stream, .{ .tfq3 = w.down }, s.act + lo * l.width * 2, l.width, out, s.mm_part, hi - lo, l.dims, l.width, true),
    }
}

/// The block up to its down projection: [gate | up] and SwiGLU into s.act.
pub fn body(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, prompt: bool) !void {
    const w = l.width;
    const q = switch (l.w) {
        .nvfp4 => |q| q,
        .bf16 => |b| { // [gate | up] in one matmul, as the shared expert's
            try b16.matmul(k.set, stream, x, x_stride, b.gu, s.gu, s.mm_part, rows, 2 * w, l.dims, false);
            return glue.swiglu(k.set, stream, s.gu, s.act, s.xs_act, l.limit, rows, w);
        },
        .mxfp8 => |m| { // [gate | up] stacked on the lane, its rows straight into s.gu
            try projection.matmul(k.set, stream, .{ .mxfp8 = m.gu }, x, x_stride, s.gu, s.mm_part, rows, 2 * w, l.dims, false);
            return glue.swiglu(k.set, stream, s.gu, s.act, s.xs_act, l.limit, rows, w);
        },
        .tfq3 => |t| {
            const tk = k.set.tfq3 orelse return error.FormatNotLoaded;
            if (prompt and rows > 16) {
                try tk.densePrompt(stream, t.gate, x, x_stride, s.gu, 2 * w, rows, false);
                try tk.densePrompt(stream, t.up, x, x_stride, s.gu + w * 2, 2 * w, rows, false);
            } else {
                try tk.dense(stream, t.gate, x, x_stride, s.gu, 2 * w, rows, false);
                try tk.dense(stream, t.up, x, x_stride, s.gu + w * 2, 2 * w, rows, false);
            }
            return glue.swiglu(k.set, stream, s.gu, s.act, s.xs_act, l.limit, rows, w);
        },
    };
    for ([_]cuda.qmmf.Weight{ q.gate, q.up }, 0..) |proj, j| {
        // Fp4Linear(out=a column slice): the rows computed contiguous, then copied into their half of each row
        try mm(k, stream, proj, x, x_stride, rows, s.tmp, false, prompt);
        var a: cuda.Args = .{};
        a.add(s.tmp);
        a.add(@as(c_longlong, @intCast(w * 2)));
        a.add(s.gu + j * w * 2);
        a.add(@as(c_longlong, @intCast(2 * w * 2)));
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(w * 2)));
        try cuda_pdl.launch(k.copy_rows, .{ .grid = .{ .x = @intCast((rows * w * 2 + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
    }
    try glue.swiglu(k.set, stream, s.gu, s.act, s.xs_act, l.limit, rows, w);
}
