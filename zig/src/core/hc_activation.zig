//! BF16 HC activation with no accumulation; each element is independent of rows, prompt cuts and neighbours.
const std = @import("std");
const wo = @import("weight_only.zig");
pub const Range = wo.Range;
pub const Params = extern struct { rows: c_int, cols: c_int, streams: c_int, injection: c_int, repeat: c_int = 0 };
pub fn bytes(p: Params) !usize {
    if (p.rows < 1 or p.rows > 128 or p.cols < 1 or p.streams < 2 or p.streams > 8 or p.injection < 0 or p.injection > 1) return error.HcActivationShape;
    return std.math.mul(usize, try std.math.mul(usize, @intCast(p.rows), @intCast(p.cols)), 2);
}
pub fn validate(p: Params, x: wo.Range, out: wo.Range) !usize {
    const n = try bytes(p);
    for ([_]wo.Range{ x, out }) |range| {
        if (range.address == 0 or range.len < n) return error.HcActivationBuffer;
        if (range.address % 2 != 0) return error.HcActivationAlignment;
        _ = std.math.add(usize, range.address, range.len) catch return error.HcActivationBuffer;
    }
    if (x.address != out.address) {
        const overlap = if (x.address <= out.address) out.address - x.address < x.len else x.address - out.address < out.len;
        if (overlap) return error.HcActivationAlias;
    }
    return n / 2;
}
