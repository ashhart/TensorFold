//! Validate every advertised axis and byte extent before fixed-layout readers use checkpoint storage.
const std = @import("std");
const mlx_affine = @import("../../core/quant/mlx_affine.zig");

/// `t` is `dtype` with exactly `dims`, its bytes those dims' extent: the core reader's check, under Step-5's error.
pub fn require(t: anytype, dtype: @TypeOf(t.dtype), dims: []const usize) error{UnexpectedStep5Tensor}!void {
    mlx_affine.require(t, dtype, dims) catch return error.UnexpectedStep5Tensor;
}
