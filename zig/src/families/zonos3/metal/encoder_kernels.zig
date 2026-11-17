//! The clip encoder's Metal launches: encoder.metal and teacher.metal by kernel, each with the threads of the launch of
//! that name in cuda/encoder.zig, cuda/quantizer.zig and cuda/teacher.zig.
const Gpu = @import("gpu.zig").Gpu;

fn i(x: usize) i32 {
    return @intCast(x);
}

fn l(x: usize) i64 {
    return @intCast(x);
}

/// 44.1 kHz `x` [total] to 44.16 kHz `out` [count].
pub fn resampleIn(g: *Gpu, x: u64, total: usize, kernel: u64, count: usize, out: u64) void {
    const A = extern struct { total: i64, count: i64 };
    g.go(.ze_resample_in, .{ (count + 255) / 256, 1, 1 }, 256, &.{ x, kernel, out }, A{ .total = l(total), .count = l(count) });
}

/// out [t / stride][c_out] from x [t][c_in]: `kernel` taps `dilation` apart, `pad` zeros before the first input.
pub fn conv(g: *Gpu, x: u64, c_in: usize, t: usize, w: u64, b: u64, kernel: usize, dilation: usize, stride: usize, pad: usize, out: u64, c_out: usize) void {
    const A = extern struct { Cin: i32, T: i32, K: i32, dilation: i32, stride: i32, pad: i32, Cout: i32 };
    const a: A = .{ .Cin = i(c_in), .T = i(t), .K = i(kernel), .dilation = i(dilation), .stride = i(stride), .pad = i(pad), .Cout = i(c_out) };
    g.goShared(.ze_conv, .{ t / stride, (c_out + 7) / 8, 1 }, 256, &.{ x, w, b, out }, a, kernel * c_in * 4);
}

pub fn snake(g: *Gpu, x: u64, alpha: u64, c: usize, n: usize, y: u64) void {
    const A = extern struct { C: i32, pad: i32 = 0, n: i64 };
    g.go(.ze_snake, .{ (n + 255) / 256, 1, 1 }, 256, &.{ x, alpha, y }, A{ .C = i(c), .n = l(n) });
}

/// x += y over n values.
pub fn add(g: *Gpu, x: u64, y: u64, n: usize) void {
    const A = extern struct { n: i64 };
    g.go(.ze_add, .{ (n + 255) / 256, 1, 1 }, 256, &.{ x, y }, A{ .n = l(n) });
}

/// x -= y over n values.
pub fn sub(g: *Gpu, x: u64, y: u64, n: usize) void {
    const A = extern struct { n: i64 };
    g.go(.ze_sub, .{ (n + 255) / 256, 1, 1 }, 256, &.{ x, y }, A{ .n = l(n) });
}

/// in [rows * h][c] to out [rows][c * h].
pub fn patchDown(g: *Gpu, in: u64, out: u64, c: usize, h: usize, rows: usize) void {
    const A = extern struct { C: i32, h: i32 };
    g.go(.ze_patch_down, .{ (c * h + 255) / 256, rows, 1 }, 256, &.{ in, out }, A{ .C = i(c), .h = i(h) });
}

/// codes[r][col] = the nearest of a codebook's 4096 unit rows to e[r] [8], for n rows.
pub fn nearest(g: *Gpu, e: u64, book: u64, codes: u64, col: usize, n: usize) void {
    const A = extern struct { col: i32 };
    g.go(.ze_nearest, .{ n, 1, 1 }, 256, &.{ e, book, codes }, A{ .col = i(col) });
}

/// residual[r] -= table[codes[r][col]] over d values a row, for n rows.
pub fn subCode(g: *Gpu, residual: u64, table: u64, codes: u64, col: usize, d: usize, n: usize) void {
    const A = extern struct { col: i32, D: i32 };
    g.go(.ze_sub_code, .{ (d + 255) / 256, n, 1 }, 256, &.{ residual, table, codes }, A{ .col = i(col), .D = i(d) });
}

pub fn swish(g: *Gpu, x: u64, n: usize) void {
    const A = extern struct { n: i64 };
    g.go(.zt_swish, .{ (n + 255) / 256, 1, 1 }, 256, &.{x}, A{ .n = l(n) });
}

/// out [rows][c] = in[.., :c] * sigmoid(in[.., c:]).
pub fn glu(g: *Gpu, in: u64, out: u64, c: usize, rows: usize) void {
    const A = extern struct { C: i32 };
    g.go(.zt_glu, .{ (c + 255) / 256, rows, 1 }, 256, &.{ in, out }, A{ .C = i(c) });
}

pub fn dwconv(g: *Gpu, x: u64, w: u64, taps: usize, c: usize, out: u64, rows: usize) void {
    const A = extern struct { K: i32, C: i32 };
    g.go(.zt_dwconv, .{ (c + 255) / 256, rows, 1 }, 256, &.{ x, w, out }, A{ .K = i(taps), .C = i(c) });
}

/// The teacher's attention over rows [q | k | v] of 3 d, heads of 64, distance table `e` [73][64].
pub fn attention(g: *Gpu, qkv: u64, d: usize, rows: usize, e: u64, out: u64) void {
    const A = extern struct { D: i32, T: i32 };
    g.go(.zt_attention, .{ rows, d / 64, 1 }, 128, &.{ qkv, e, out }, A{ .D = i(d), .T = i(rows) });
}
