//! Buffer references shared by forward encoding and its differential baseline.
const mtl = @import("metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const attention = @import("attention.zig");
const Ref = @import("state.zig").Ref;

pub fn r(t: ck.Tensor) Ref {
    return .{ .buf = t.buffer, .off = t.offset };
}
pub fn at(buf: mtl.Buffer, off: usize) Ref {
    return .{ .buf = buf, .off = off };
}
pub fn row(buf: mtl.Buffer, i: usize, row_bytes: usize) Ref {
    return at(buf, i * row_bytes);
}
pub fn a(x_: Ref) attention.Ref {
    return .{ .buf = x_.buf, .off = x_.off };
}
