//! Plain or CSF-compressed ModelOpt scales and packed codes, materialized only for one tensor-parallel rectangle.

const std = @import("std");
const ckpt = @import("checkpoint.zig");
const csf = @import("nvfp4_csf.zig");
const Tensor = ckpt.Tensor;
const Span = @import("quant/slice.zig").Span;

pub const Scales = union(enum) { plain: Tensor, csf: struct { fixed: Tensor, exceptions: Tensor } };
pub const Source = struct { weight: Tensor, scales: Scales, global: Tensor, input: ?Tensor = null };
pub const Dims = struct { n: usize, k: usize };

/// Owned rank-local bytes: low-even FP4 nibbles, row-major E4M3 scales and the unchanged FP32 global scale.
pub const Linear = struct {
    n: usize,
    k: usize,
    codes: []u8,
    scales: []u8,
    global: f32,

    pub fn deinit(self: *Linear, gpa: std.mem.Allocator) void {
        gpa.free(self.codes);
        gpa.free(self.scales);
        self.* = undefined;
    }
};

/// Validate a projection's original geometry before allocating rank-local storage.
pub fn dimensions(source: Source) !Dims {
    const w = source.weight;
    if (w.rank != 2 or w.dtype != .u8 or w.shape[0] == 0 or w.shape[1] == 0) return error.BadNvfp4;
    const k = std.math.mul(usize, w.shape[1], 2) catch return error.BadNvfp4;
    const count = std.math.mul(usize, w.shape[0], w.shape[1]) catch return error.BadNvfp4;
    if (w.bytes.len != count or k % 16 != 0) return error.BadNvfp4;
    const d: Dims = .{ .n = w.shape[0], .k = k };
    if (!scalar(source.global)) return error.BadNvfp4;
    if (source.input) |a| if (!scalar(a)) return error.BadNvfp4;
    switch (source.scales) {
        .plain => |s| if (!s.is(.f8_e4m3, &.{ d.n, d.k / 16 }) or s.bytes.len != count / 8) return error.BadNvfp4,
        .csf => |s| {
            if (s.exceptions.dtype != .u32 or s.exceptions.rank != 1) return error.BadNvfp4;
            if (s.exceptions.shape[0] > std.math.maxInt(usize) / 4 or s.exceptions.bytes.len != s.exceptions.shape[0] * 4)
                return error.BadNvfp4;
            if (!s.fixed.is(.u8, &.{ d.n / 16, 16 + 8 * (d.k / 16) })) return error.BadNvfp4;
            _ = try csf.Plane.init(d.n, d.k / 16, s.fixed.bytes, s.exceptions.bytes);
        },
    }
    return d;
}

/// Rows and input channels are logical weight axes; input cuts must preserve complete groups of 16.
pub fn slice(gpa: std.mem.Allocator, source: Source, rows: Span, inputs: Span) !Linear {
    const d = try dimensions(source);
    if (rows.from >= rows.to or rows.to > d.n or inputs.from >= inputs.to or inputs.to > d.k)
        return error.BadNvfp4Slice;
    if (inputs.from % 16 != 0 or inputs.to % 16 != 0) return error.BadNvfp4Slice;
    const n = rows.to - rows.from;
    const k = inputs.to - inputs.from;
    const codes = try gpa.alloc(u8, n * (k / 2));
    errdefer gpa.free(codes);
    const scales = try gpa.alloc(u8, n * (k / 16));
    errdefer gpa.free(scales);
    for (0..n) |row| {
        const begin = (rows.from + row) * (d.k / 2) + inputs.from / 2;
        @memcpy(codes[row * (k / 2) ..][0 .. k / 2], source.weight.bytes[begin..][0 .. k / 2]);
    }
    switch (source.scales) {
        .plain => |s| for (0..n) |row| {
            const begin = (rows.from + row) * (d.k / 16) + inputs.from / 16;
            @memcpy(scales[row * (k / 16) ..][0 .. k / 16], s.bytes[begin..][0 .. k / 16]);
        },
        .csf => |s| {
            const plane = try csf.Plane.init(d.n, d.k / 16, s.fixed.bytes, s.exceptions.bytes);
            try plane.decode(scales, rows.from, n, inputs.from / 16, k / 16);
        },
    }
    return .{ .n = n, .k = k, .codes = codes, .scales = scales, .global = @bitCast(std.mem.readInt(u32, source.global.bytes[0..4], .little)) };
}

/// Read exact stored tensors, consuming optional calibration and refusing ambiguous plain/compressed scales.
pub fn read(ck: *ckpt.Checkpoint, gpa: std.mem.Allocator, module: []const u8, rows: Span, inputs: Span) !Linear {
    var buf: [512]u8 = undefined;
    const w = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{module}));
    const g = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight_scale_2", .{module}));
    const input_name = try std.fmt.bufPrint(&buf, "{s}.input_scale", .{module});
    const a: ?Tensor = if (ck.has(input_name)) try ck.get(input_name) else null;
    const fixed_name = try std.fmt.bufPrint(&buf, "{s}.weight_scale.nvfp4_csf_fixed", .{module});
    const fixed: ?Tensor = if (ck.has(fixed_name)) try ck.get(fixed_name) else null;
    const exception_name = try std.fmt.bufPrint(&buf, "{s}.weight_scale.nvfp4_csf_exceptions", .{module});
    const exceptions: ?Tensor = if (ck.has(exception_name)) try ck.get(exception_name) else null;
    if ((fixed != null) != (exceptions != null)) return error.BadNvfp4;
    const plain_name = try std.fmt.bufPrint(&buf, "{s}.weight_scale", .{module});
    const plain = ck.has(plain_name);
    if (plain == (fixed != null)) return error.BadNvfp4;
    const scales: Scales = if (plain) .{ .plain = try ck.get(plain_name) } else .{ .csf = .{ .fixed = fixed.?, .exceptions = exceptions.? } };
    return slice(gpa, .{ .weight = w, .scales = scales, .global = g, .input = a }, rows, inputs);
}

fn scalar(t: Tensor) bool {
    return t.dtype == .f32 and t.bytes.len == 4 and (t.rank == 0 or (t.rank == 1 and t.shape[0] == 1));
}

test {
    _ = @import("nvfp4_slice_test.zig");
}
