//! GGUF tensor dimensions are stored with the contiguous dimension first; blocks may not cross its row boundary.
const std = @import("std");
const meta = @import("gguf_value.zig");
pub const Kind = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    bf16 = 30,

    pub fn block(k: Kind) struct { values: usize, bytes: usize } {
        return switch (k) {
            .f32, .i32 => .{ .values = 1, .bytes = 4 },
            .f16, .bf16, .i16 => .{ .values = 1, .bytes = 2 },
            .i8 => .{ .values = 1, .bytes = 1 },
            .i64, .f64 => .{ .values = 1, .bytes = 8 },
            .q4_0 => .{ .values = 32, .bytes = 18 },
            .q4_1 => .{ .values = 32, .bytes = 20 },
            .q5_0 => .{ .values = 32, .bytes = 22 },
            .q5_1 => .{ .values = 32, .bytes = 24 },
            .q8_0 => .{ .values = 32, .bytes = 34 },
            .q8_1 => .{ .values = 32, .bytes = 36 },
            .q2_k => .{ .values = 256, .bytes = 84 },
            .q3_k => .{ .values = 256, .bytes = 110 },
            .q4_k => .{ .values = 256, .bytes = 144 },
            .q5_k => .{ .values = 256, .bytes = 176 },
            .q6_k => .{ .values = 256, .bytes = 210 },
            .q8_k => .{ .values = 256, .bytes = 292 },
        };
    }
};
pub const Tensor = struct {
    name: []const u8,
    kind: Kind,
    rank: u8,
    shape: [4]u64 = @splat(1),
    offset: usize,
    len: usize,
    bytes: []const u8 = &.{},

    pub fn read(c: *meta.Cursor) !Tensor {
        const name = try c.string();
        if (name.len == 0 or std.mem.indexOfScalar(u8, name, 0) != null) return error.InvalidTensorName;
        const rank = try c.int(u32);
        if (rank == 0 or rank > 4) return error.InvalidTensorRank;
        var out: Tensor = .{ .name = name, .kind = undefined, .rank = @intCast(rank), .offset = 0, .len = 0 };
        var n: u64 = 1;
        for (out.shape[0..rank]) |*dim| {
            dim.* = try c.int(u64);
            if (dim.* == 0) return error.ZeroTensorDimension;
            n = std.math.mul(u64, n, dim.*) catch return error.TensorShapeOverflow;
        }
        const kind = try c.int(u32);
        out.kind = std.enums.fromInt(Kind, kind) orelse {
            if (c.unsupported) |problem| problem.* = .{ .name = name, .kind = kind };
            return error.UnsupportedTensorType;
        };
        const layout = out.kind.block();
        if (out.shape[0] % layout.values != 0) return error.InvalidBlockShape;
        const bytes = std.math.mul(u64, n / layout.values, layout.bytes) catch return error.TensorShapeOverflow;
        out.len = std.math.cast(usize, bytes) orelse return error.TensorShapeOverflow;
        out.offset = std.math.cast(usize, try c.int(u64)) orelse return error.TensorOffsetOverflow;
        return out;
    }
};
