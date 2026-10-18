//! Validate every advertised axis and byte extent before fixed-layout readers use checkpoint storage.
const std = @import("std");
pub fn require(t: anytype, dtype: @TypeOf(t.dtype), dims: []const usize) error{UnexpectedStep5Tensor}!void {
    if (t.dtype != dtype or t.rank != dims.len or t.rank > t.shape.len) return error.UnexpectedStep5Tensor;
    if (!std.mem.eql(usize, t.shape[0..t.rank], dims)) return error.UnexpectedStep5Tensor;
    var bytes: usize = dtype.size();
    for (dims) |n| {
        if (n == 0) return error.UnexpectedStep5Tensor;
        bytes = std.math.mul(usize, bytes, n) catch return error.UnexpectedStep5Tensor;
    }
    if (t.bytes != bytes) return error.UnexpectedStep5Tensor;
}
