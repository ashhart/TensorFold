//! Sixteen or twenty-four real heads through the unchanged 32-head prefill core; copies preserve the run's
//! group-major layout. A rank's heads are the leading heads of the 32 and the rest are zero.
const std = @import("std");
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const E = @import("block.zig").Emitter;
const Arg = calls.Arg;

/// The core's head count: a padded rank runs these, its own heads first.
pub const core_heads: i64 = 32;

/// A rank whose heads go through the padded core: four ranks (16 heads) and three ranks' 24-head shares.
pub fn padded(heads: i64) bool {
    return heads == 16 or heads == 24;
}

/// `real` of `full` as a whole ratio the copies accept: a half (16 of 32) or three quarters (24 of 32).
fn part(real: i64, full: i64) bool {
    return real > 0 and (2 * real == full or 4 * real == 3 * full);
}

pub fn query(e: *E, q: Arg, hd: i64) !Arg {
    const full = try e.buf("L.pf.q32", .{}, .bf16, &.{ e.n, core_heads * hd });
    try e.glue("pf_heads_q", &.{ q, full });
    return full;
}

pub fn sink(e: *E, real: Arg) !Arg {
    const full = try e.buf("L.pf.sink32", .{}, .f32, &.{core_heads});
    try e.glue("pf_heads_sink", &.{ real, full });
    return full;
}

pub fn extract(e: *E, source: Arg, dest: Arg) !void {
    try e.glue("pf_heads_out", &.{ source, dest });
}

/// Only these copy calls use this dispatcher; all storage belongs to the existing buffer plan and stream.
pub fn glue(r: anytype, c: *const calls.Call) !bool {
    if (!std.mem.startsWith(u8, c.name, "glue.pf_heads_")) return false;
    if (c.args.len != 2 or c.args[0].arg != .t or c.args[1].arg != .t) return error.BadGlue;
    const src = c.args[0].arg.t;
    const dst = c.args[1].arg.t;
    if (src.dt != dst.dt or src.shape.len != src.stride.len or dst.shape.len != dst.stride.len) return error.BadGlue;
    var width: usize = undefined;
    var height: usize = undefined;
    var sp: usize = undefined;
    var dp: usize = undefined;
    var zero: usize = 0;
    if (std.mem.eql(u8, c.name, "glue.pf_heads_q")) {
        if (src.dt != .bf16 or src.shape.len != 2 or dst.shape.len != 2 or src.shape[0] <= 0 or src.shape[1] <= 0 or
            dst.shape[0] != src.shape[0] or !part(src.shape[1], dst.shape[1]) or src.stride[1] != 1 or dst.stride[1] != 1 or
            src.stride[0] != src.shape[1] or dst.stride[0] != dst.shape[1]) return error.BadGlue;
        width = @intCast(2 * src.shape[1]);
        height = @intCast(src.shape[0]);
        sp = width;
        dp = @intCast(2 * dst.shape[1]);
        zero = height * dp;
    } else if (std.mem.eql(u8, c.name, "glue.pf_heads_sink")) {
        if (src.dt != .f32 or src.shape.len != 1 or dst.shape.len != 1 or !padded(src.shape[0]) or dst.shape[0] != core_heads or
            src.stride[0] != 1 or dst.stride[0] != 1) return error.BadGlue;
        width = @intCast(4 * src.shape[0]);
        height = 1;
        sp = width;
        dp = @intCast(4 * core_heads);
        zero = dp;
    } else if (std.mem.eql(u8, c.name, "glue.pf_heads_out")) {
        if (src.dt != .bf16 or src.shape.len != 3 or dst.shape.len != 3 or dst.shape[0] <= 0 or dst.shape[1] <= 0 or dst.shape[2] <= 0 or
            !part(dst.shape[0], src.shape[0]) or src.shape[1] != dst.shape[1] or src.shape[2] != dst.shape[2] or
            src.stride[2] != 1 or dst.stride[2] != 1 or src.stride[1] != src.shape[2] or dst.stride[1] != dst.shape[2] or
            src.stride[0] < src.shape[1] * src.shape[2] or dst.stride[0] < dst.shape[1] * dst.shape[2]) return error.BadGlue;
        width = @intCast(2 * dst.shape[1] * dst.shape[2]);
        height = @intCast(dst.shape[0]);
        sp = @intCast(2 * src.stride[0]);
        dp = @intCast(2 * dst.stride[0]);
    } else return error.BadGlue;
    const source = try r.tensorAddr(src);
    const dest = try r.tensorAddr(dst);
    if (zero != 0) try r.d.check(r.d.api.cuMemsetD8Async(dest, 0, zero, r.stream.handle), "cuMemsetD8Async");
    const copy: cuda.abi.Memcpy2D = .{ .src = source, .src_pitch = sp, .dst = dest, .dst_pitch = dp, .width = width, .height = height };
    try r.d.check(r.d.api.cuMemcpy2DAsync_v2(&copy, r.stream.handle), "cuMemcpy2DAsync");
    return true;
}
