//! A training reference borrows its owning allocation and validates every byte subview.
const std = @import("std");
pub fn Ref(comptime B: type) type {
    return struct {
        buffer: B.Buffer,
        offset: usize = 0,
        len: usize,
        pub fn plus(r: @This(), offset: usize, len: usize) !@This() {
            if (offset > r.len or len > r.len - offset) return error.GemmaTrainingRange;
            return .{ .buffer = r.buffer, .offset = try std.math.add(usize, r.offset, offset), .len = len };
        }
    };
}
