//! Inference scratch has one stable device allocation; slices are checked before a kernel can see them.
const std = @import("std");
pub const Kind = enum {
    bf16,
    f32,
    pub fn bytes(k: Kind) usize {
        return if (k == .bf16) 2 else 4;
    }
};
pub fn Arena(comptime B: type) type {
    return struct {
        const Self = @This();
        pub const Tensor = struct {
            buffer: B.Buffer,
            offset: usize,
            rows: usize,
            width: usize,
            kind: Kind,
            pub fn bytes(t: Tensor) !usize {
                return std.math.mul(usize, try std.math.mul(usize, t.rows, t.width), t.kind.bytes());
            }
            pub fn row(t: Tensor, index: usize) !Tensor {
                if (index >= t.rows) return error.Qwen4TensorRow;
                return .{ .buffer = t.buffer, .offset = try std.math.add(usize, t.offset, try std.math.mul(usize, index, try std.math.mul(usize, t.width, t.kind.bytes()))), .rows = 1, .width = t.width, .kind = t.kind };
            }
            pub fn subRows(t: Tensor, start: usize, count: usize) !Tensor {
                if (count == 0 or start > t.rows or count > t.rows - start) return error.Qwen4TensorRow;
                var out = try t.row(start);
                out.rows = count;
                return out;
            }
        };
        backend: *B,
        buffer: B.Buffer,
        capacity: usize,
        used: usize = 0,
        peak: usize = 0,
        pub fn open(b: *B, bytes: usize, budget: u64) !Self {
            if (bytes == 0) return error.Qwen4ScratchEmpty;
            if (try B.allocationBytes(bytes) > budget) return error.Qwen4ScratchBudget;
            return .{ .backend = b, .buffer = try b.allocate(bytes), .capacity = bytes };
        }
        pub fn close(a: *Self) !void {
            try a.backend.drain();
            a.backend.release(a.buffer);
            a.* = undefined;
        }
        pub fn take(a: *Self, rows: usize, width: usize, kind: Kind) !Tensor {
            if (rows == 0 or width == 0) return error.Qwen4TensorShape;
            const n = try std.math.mul(usize, try std.math.mul(usize, rows, width), kind.bytes());
            const start = (try std.math.add(usize, a.used, 15)) & ~@as(usize, 15);
            if (start > a.capacity or n > a.capacity - start) return error.Qwen4ScratchFull;
            a.used = start + n;
            a.peak = @max(a.peak, a.used);
            return .{ .buffer = a.buffer, .offset = start, .rows = rows, .width = width, .kind = kind };
        }
        /// All earlier users must already be encoded on this owner's queue before their ranges are reused.
        pub fn rewind(a: *Self, mark: usize) !void {
            if (mark > a.used) return error.Qwen4ScratchMark;
            a.used = mark;
        }
    };
}
