//! A borrowed Metal buffer region; element precision is carried by each operator's explicit descriptor.
const std = @import("std");
const mtl = @import("metal");

pub const View = struct {
    buffer: mtl.Buffer,
    offset: usize = 0,

    pub fn shifted(self: View, bytes: usize) View {
        return .{ .buffer = self.buffer, .offset = self.offset + bytes };
    }

    pub fn require(self: View, bytes: usize, alignment: usize) !void {
        if (self.offset % alignment != 0 or self.offset > self.buffer.length() or
            bytes > self.buffer.length() - self.offset) return error.InvalidTensorView;
    }

    /// Rows are strided elements; only the final row's used width is required.
    pub fn requireRows(self: View, rows: usize, stride: usize, width: usize, element_bytes: usize) !void {
        if (rows == 0 or width == 0 or stride < width) return error.InvalidTensorView;
        const preceding = try std.math.mul(usize, rows - 1, stride);
        const elements = try std.math.add(usize, preceding, width);
        try self.require(try std.math.mul(usize, elements, element_bytes), element_bytes);
    }
};
