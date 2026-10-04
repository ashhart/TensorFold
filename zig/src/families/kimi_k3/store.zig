//! Named tensors on the GPU: a buffer slice with dtype and shape, from the checkpoint or made for tests.
const std = @import("std");
const mtl = @import("metal");
const Ref = @import("kernels.zig").Ref;

pub const DType = enum {
    bf16,
    f32,
    u8,

    pub fn size(d: DType) usize {
        return switch (d) {
            .bf16 => 2,
            .f32 => 4,
            .u8 => 1,
        };
    }

    pub fn parse(text: []const u8) !DType {
        if (std.mem.eql(u8, text, "BF16")) return .bf16;
        if (std.mem.eql(u8, text, "F32")) return .f32;
        if (std.mem.eql(u8, text, "U8")) return .u8;
        return error.UnsupportedDType;
    }
};

pub const Tensor = struct {
    ref: Ref,
    dtype: DType,
    shape: [3]u32 = .{ 1, 1, 1 },
    rank: u8 = 1,

    pub fn count(t: Tensor) usize {
        var n: usize = 1;
        for (t.shape[0..t.rank]) |d| n *= d;
        return n;
    }

    pub fn bytes(t: Tensor) usize {
        return t.count() * t.dtype.size();
    }

    /// Rows [lo, hi) of a 2-D tensor: a column-parallel shard, no copy.
    pub fn rows(t: Tensor, lo: u32, hi: u32) Tensor {
        var s = t;
        s.shape[0] = hi - lo;
        s.ref = t.ref.at(@as(usize, lo) * t.shape[1] * t.dtype.size());
        return s;
    }

    pub fn expect(t: Tensor, dtype: DType, shape: []const u32) !void {
        if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(u32, t.shape[0..t.rank], shape)) {
            std.log.err("tensor is {s} {any}, expected {s} {any}", .{ @tagName(t.dtype), t.shape[0..t.rank], @tagName(dtype), shape });
            return error.TensorShape;
        }
    }
};

/// Where tensors come from: the zero-copy checkpoint or synthetic test values.
pub const Source = struct {
    ptr: *anyopaque,
    getFn: *const fn (ptr: *anyopaque, name: []const u8, dtype: DType, shape: []const u32) anyerror!Tensor,

    /// The tensor `name`, checked against the dtype and shape the config implies.
    pub fn get(s: Source, name: []const u8, dtype: DType, shape: []const u32) !Tensor {
        return s.getFn(s.ptr, name, dtype, shape);
    }
};
