//! Standalone input-group sums and buffer bounds for lane projections.
const mtl = @import("metal");

pub fn checkRef(r: anytype, need: usize, alignment: usize) !void {
    if (r.off % alignment != 0) return error.MisalignedBuffer;
    const len = r.buf.length();
    if (r.off > len or need > len - r.off) return error.BufferTooSmall;
}

/// tf_lane_sums of `rows` rows of x into the projection's scratch, and a barrier; return the padded row stride.
pub fn encode(p: anytype, e: mtl.ComputeEncoder, x: anytype, rows: u32) !u32 {
    const pipe = p.sums_pipe orelse return error.SumsNeedReg;
    if (rows == 0 or rows > p.max_rows) return error.InvalidRows;
    try checkRef(x, rows * p.k * 2, 2);
    const stride: u32 = 16 * ((rows + 15) / 16);
    e.setPipeline(pipe);
    e.setBuffer(x.buf, x.off, 0);
    e.setValue(@as(i32, @intCast(rows)), 1);
    e.setBuffer(p.sums_buffer.?, 0, 2);
    e.setValue(@as(i32, @intCast(stride)), 3);
    e.dispatchThreads(.{ .width = stride * p.group_count }, .{ .width = 256 });
    e.barrier();
    return stride;
}
