//! Norm launches and the decode tails that fuse a norm.

const t = @import("types.zig");
const Ops = @import("ops.zig").Ops;
const Error = t.Error;
const Tensor = t.Tensor;
const p = t.p;
const f = t.f;
const int = t.int;

pub fn rms(o: Ops, x: Tensor, weight: ?u64, y: Tensor, rows: usize, width: usize, eps: f32) Error!void {
    if (x.kind != y.kind or width < 1 or width > 8192) return error.BadShape;
    if (rows == 0) return;
    try o.l.tf_rms(p(x.ptr), if (weight) |w| f(w) else null, p(y.ptr), @backingInt(x.kind), int(rows), int(width), eps, o.stream);
}

/// Two sets of fp32 rows (width at most 1024), each normed with its own weight, in one launch.
pub fn rms2(o: Ops, x0: u64, w0: u64, y0: u64, x1: u64, w1: u64, y1: u64, rows: usize, width: usize, eps: f32) Error!void {
    const z = o.l;
    try z.tf_rms2(f(x0), f(w0), f(y0), f(x1), f(w1), f(y1), int(rows), int(width), eps, o.stream);
}

/// The linear attention's gated norm: out = rms(y) * weight * round(silu(z)) in z's kind, `width` at most 1024.
pub fn gnormOut(o: Ops, y: u64, weight: u64, z: Tensor, out: Tensor, rows: usize, width: usize, eps: f32) Error!void {
    if (z.kind != out.kind or z.kind == .f32) return error.BadShape;
    const zig = o.l;
    try zig.tf_gnorm_out(f(y), f(weight), p(z.ptr), p(out.ptr), @backingInt(z.kind), int(rows), int(width), eps, o.stream);
}

/// x = round(x + y) and normed = rms(x) * weight (fp32) in one launch: a residual and the norm after it.
pub fn addRms(o: Ops, x: Tensor, y: Tensor, weight: u64, normed: Tensor, rows: usize, width: usize, eps: f32) Error!void {
    if (x.kind != y.kind or x.kind != normed.kind or x.kind == .f32) return error.BadShape;
    const z = o.l;
    try z.tf_tail(p(x.ptr), p(y.ptr), null, f(weight), p(normed.ptr), @backingInt(x.kind), int(rows), 0, int(width), eps, o.stream);
}

pub fn gnormSilu(o: Ops, y: u64, z: Tensor, out: Tensor, n: usize) Error!void {
    try o.l.tf_gnorm_silu(f(y), p(z.ptr), p(out.ptr), @backingInt(out.kind), @intCast(n), o.stream);
}
