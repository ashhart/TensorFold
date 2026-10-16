//! Casts and the elementwise launches.

const t = @import("types.zig");
const Ops = @import("ops.zig").Ops;
const Error = t.Error;
const Tensor = t.Tensor;
const p = t.p;
const int = t.int;

pub fn cast(o: Ops, src: Tensor, dst: Tensor, n: usize) Error!void {
    try o.l.tf_cast(p(src.ptr), @backingInt(src.kind), p(dst.ptr), @backingInt(dst.kind), @intCast(n), o.stream);
}

pub fn siluMul(o: Ops, gate: Tensor, up: Tensor, out: Tensor, n: usize) Error!void {
    if (gate.kind != up.kind or gate.kind != out.kind) return error.BadShape;
    try o.l.tf_silu_mul(p(gate.ptr), p(up.ptr), p(out.ptr), @backingInt(out.kind), @intCast(n), o.stream);
}

pub fn add(o: Ops, x: Tensor, y: Tensor, out: Tensor, n: usize) Error!void {
    if (x.kind != y.kind or x.kind != out.kind) return error.BadShape;
    try o.l.tf_add(p(x.ptr), p(y.ptr), p(out.ptr), @backingInt(out.kind), @intCast(n), o.stream);
}

pub fn copyCols(o: Ops, src: Tensor, stride: usize, offset: usize, dst: u64, rows: usize, cols: usize) Error!void {
    try o.l.tf_copy_cols(p(src.ptr), @intCast(stride), int(offset), p(dst), @backingInt(src.kind), int(rows), int(cols), o.stream);
}
