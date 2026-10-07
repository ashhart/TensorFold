//! The lane projection's path for chips without tensor units: row-layout 2-, 4- and 8-bit weights, one read a window, each row its own fp32 sums.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

/// Rows a weight read (the kernel's MB); a window takes batches of this, then one row at a time.
pub const batch = 2;

/// Output columns a simdgroup and simdgroups a threadgroup (the kernel's RPS and SG).
pub const columns = 4;
pub const groups = 2;

/// The code widths the kernels unpack.
pub const bit_widths = [_]u8{ 2, 4, 6, 8 };

/// A group's sum of inputs for the bias term: fp32 in order (the tensor path's xsum), or rounded to bf16 at each add (Nemotron's expert kernels).
pub const Sum = enum { f32, bf16 };

/// A matrix [n, k] as the checkpoint stores it: words [n][k*bits/32], scales and biases [n][k/group] bf16, at byte offsets.
pub const Weights = struct {
    w: mtl.Buffer,
    w_off: usize = 0,
    scales: mtl.Buffer,
    s_off: usize = 0,
    biases: mtl.Buffer,
    b_off: usize = 0,
    n: usize,
    k: usize,
    group: usize = 64,
    bits: u8 = 4,
    sum: Sum = .f32,

    pub fn validate(x: Weights) !void {
        if (x.n == 0 or x.n % (columns * groups) != 0 or x.k == 0 or x.k % 16 != 0 or x.group < 16 or x.group % 16 != 0 or x.k % x.group != 0 or 512 % x.group != 0) return error.UnsupportedRowShape;
        if (std.mem.indexOfScalar(u8, &bit_widths, x.bits) == null) return error.UnsupportedRowBits;
    }

    /// The pipeline index: by code width, then the sum mode.
    fn variant(x: Weights) usize {
        return std.mem.indexOfScalar(u8, &bit_widths, x.bits).? * 2 + @backingInt(x.sum);
    }
};

/// `experts` matrices of one shape back to back in W, S and B; slot i reads expert ids[i] on x row i / repeat.
pub const Experts = struct {
    one: Weights, // the first matrix; the rest follow at its stride
    experts: usize,
    repeat: usize = 1, // slots that share one x row (a token's top-k)

    pub fn validate(x: Experts) !void {
        try x.one.validate();
        if (x.experts == 0 or x.experts > std.math.maxInt(u32) or x.repeat == 0 or x.repeat > 16) return error.UnsupportedExpertShape;
    }
};

const variants = bit_widths.len * 2;

pub const Pipelines = struct {
    plain: [variants]mtl.Pipeline,
    relu2: [variants]mtl.Pipeline,
    indexed: [variants]mtl.Pipeline,
    indexed_relu2: [variants]mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lib = try mtl.Library.fromSource(device, sources.core_row_projection, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipelines = undefined;
        inline for (bit_widths, 0..) |bits, b| inline for (.{ "f32", "bf16" }, 0..) |sum, m| {
            const i = b * 2 + m;
            p.plain[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_row_projection_q{d}_{s}", .{ bits, sum }), false);
            p.relu2[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_row_projection_relu2_q{d}_{s}", .{ bits, sum }), false);
            p.indexed[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_row_projection_indexed_q{d}_{s}", .{ bits, sum }), false);
            p.indexed_relu2[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_row_projection_indexed_relu2_q{d}_{s}", .{ bits, sum }), false);
        };
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        for (p.plain) |x| x.deinit();
        for (p.relu2) |x| x.deinit();
        for (p.indexed) |x| x.deinit();
        for (p.indexed_relu2) |x| x.deinit();
    }
};

/// One dispatch: its pipeline, dims (K, N, group, rows or slots, experts, repeat) and the grid; buffers x, W, S, B, dims, out at 0-5, ids at 6.
pub const Call = struct { pipeline: mtl.Pipeline, dims: [6]i32, groups: usize, slots: usize = 1, threads: usize = 32 * groups };

/// y[rows, n] = x[rows, k] W^T (relu squared with `relu2`), x and y bf16 rows.
pub fn call(p: *const Pipelines, w: Weights, rows: usize, relu2: bool) !Call {
    try w.validate();
    if (rows == 0) return error.NoRows;
    const i = w.variant();
    return .{ .pipeline = if (relu2) p.relu2[i] else p.plain[i], .dims = .{ @intCast(w.k), @intCast(w.n), @intCast(w.group), @intCast(rows), 1, 1 }, .groups = w.n / (columns * groups) };
}

/// y[slots, n] = x[slot / repeat] W[ids[slot]]^T; the grid's y is the slot.
pub fn callIndexed(p: *const Pipelines, x: Experts, slots: usize, relu2: bool) !Call {
    try x.validate();
    if (slots == 0 or slots % x.repeat != 0 or slots > std.math.maxInt(u16)) return error.NoRows;
    const i = x.one.variant();
    return .{ .pipeline = if (relu2) p.indexed_relu2[i] else p.indexed[i], .dims = .{ @intCast(x.one.k), @intCast(x.one.n), @intCast(x.one.group), @intCast(slots), @intCast(x.experts), @intCast(x.repeat) }, .groups = x.one.n / (columns * groups), .slots = slots };
}

test "an indexed call takes whole top-k groups of slots" {
    const one = Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 64, .k = 128 };
    const x = Experts{ .one = one, .experts = 4, .repeat = 2 };
    try x.validate();
    try std.testing.expectError(error.UnsupportedExpertShape, (Experts{ .one = one, .experts = 0 }).validate());
    try std.testing.expectError(error.UnsupportedExpertShape, (Experts{ .one = one, .experts = 2, .repeat = 17 }).validate());
}

test "shapes the row kernels serve" {
    try (Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 10304, .k = 2688 }).validate();
    try (Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 2688, .k = 3712 }).validate();
    try std.testing.expectError(error.UnsupportedRowShape, (Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 2690, .k = 2688 }).validate());
    try std.testing.expectError(error.UnsupportedRowShape, (Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 2688, .k = 2688, .group = 48 }).validate());
    try (Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 256, .k = 2688, .bits = 8 }).validate();
    try (Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 256, .k = 2688, .bits = 2 }).validate();
    try (Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 256, .k = 2688, .bits = 6 }).validate();
    try std.testing.expectError(error.UnsupportedRowBits, (Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 256, .k = 2688, .bits = 3 }).validate());
}
