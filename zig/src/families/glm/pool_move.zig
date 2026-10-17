//! Shared MLA prefixes moved after GPU completion, with memmove order for overlapping pool regions.
const std = @import("std");

pub fn prefix(c: anytype, base: anytype, from: u32, to: u32, at: u32) void {
    if (from == to) return;
    for (base) |m| {
        const refs = .{ m.keys, m.ik, m.ig, m.pool };
        const strides = [_]usize{ c.kv_lora * 2, c.i_dim * 2, c.i_dim * 2, c.i_dim * 2 };
        inline for (refs, strides, 0..) |r, stride, k| {
            const divisor: usize = if (k == 3) c.kpool else 1;
            const src = @as(usize, from) * 256 / divisor * stride;
            const dst = @as(usize, to) * 256 / divisor * stride;
            const n = (@as(usize, at) / divisor + @intFromBool(k == 3)) * stride;
            const bytes = r.addr();
            if (dst < src) std.mem.copyForwards(u8, bytes[dst..][0..n], bytes[src..][0..n]) else std.mem.copyBackwards(u8, bytes[dst..][0..n], bytes[src..][0..n]);
        }
    }
}
