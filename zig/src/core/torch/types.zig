const std = @import("std");
pub const DType = enum { f64, f32, f16, bf16, i64, i32, i16, i8, u8, bool, c64, c128 };

pub fn elementBytes(t: DType) usize {
    return switch (t) {
        .f64, .i64, .c64 => 8,
        .c128 => 16,
        .f32, .i32 => 4,
        .f16, .bf16, .i16 => 2,
        .i8, .u8, .bool => 1,
    };
}

/// torch's legacy storage class names, as torch.save pickles them.
pub fn storageType(name: []const u8) ?DType {
    const table = [_]struct { []const u8, DType }{
        .{ "DoubleStorage", .f64 },    .{ "FloatStorage", .f32 },        .{ "HalfStorage", .f16 },
        .{ "BFloat16Storage", .bf16 }, .{ "LongStorage", .i64 },         .{ "IntStorage", .i32 },
        .{ "ShortStorage", .i16 },     .{ "CharStorage", .i8 },          .{ "ByteStorage", .u8 },
        .{ "BoolStorage", .bool },     .{ "ComplexFloatStorage", .c64 }, .{ "ComplexDoubleStorage", .c128 },
    };
    for (table) |e| if (std.mem.eql(u8, e[0], name)) return e[1];
    return null;
}

/// One state-dict entry: a view into the storage member `data/<storage>`.
pub const Tensor = struct {
    dtype: DType,
    storage: []const u8,
    storage_numel: u64,
    offset: u64, // elements
    shape: []const u64,
    stride: []const u64,

    pub fn layout(self: Tensor) !struct { count: usize, contiguous: bool } {
        if (self.shape.len != self.stride.len or self.offset > self.storage_numel) return error.BadArchive;
        var count: u64 = 1;
        var contiguous = true;
        var axis = self.shape.len;
        while (axis > 0) {
            axis -= 1;
            if (self.shape[axis] != 1 and self.stride[axis] != count) contiguous = false;
            count = try std.math.mul(u64, count, self.shape[axis]);
        }
        if (count != 0) {
            var last = self.offset;
            for (self.shape, self.stride) |size, stride|
                last = try std.math.add(u64, last, try std.math.mul(u64, size - 1, stride));
            if (last >= self.storage_numel) return error.BadArchive;
        }
        return .{ .count = std.math.cast(usize, count) orelse return error.BadArchive, .contiguous = contiguous };
    }
};
