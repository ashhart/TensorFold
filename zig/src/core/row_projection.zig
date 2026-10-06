//! The lane projection's path for chips without tensor units: row-layout 4-bit weights, one read a window, each row its own sums.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

/// Rows a weight read (the kernel's MB); a window takes batches of this, then one row at a time.
pub const batch = 2;

/// Output columns a simdgroup and simdgroups a threadgroup (the kernel's RPS and SG).
pub const columns = 4;
pub const groups = 2;

/// A matrix [n, k] as the checkpoint stores it: words [n][k/8], scales and biases [n][k/group] bf16, at byte offsets.
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

    pub fn validate(x: Weights) !void {
        if (x.n == 0 or x.n % (columns * groups) != 0 or x.k == 0 or x.k % 16 != 0 or x.group < 16 or x.group % 16 != 0 or x.k % x.group != 0 or 512 % x.group != 0) return error.UnsupportedRowShape;
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

pub const Pipelines = struct {
    plain: mtl.Pipeline,
    relu2: mtl.Pipeline,
    indexed: mtl.Pipeline,
    indexed_relu2: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lib = try mtl.Library.fromSource(device, sources.core_row_projection, mtl.CompileOptions.mlx());
        defer lib.deinit();
        return .{
            .plain = try mtl.Pipeline.init(device, lib, "tf_row_projection", false),
            .relu2 = try mtl.Pipeline.init(device, lib, "tf_row_projection_relu2", false),
            .indexed = try mtl.Pipeline.init(device, lib, "tf_row_projection_indexed", false),
            .indexed_relu2 = try mtl.Pipeline.init(device, lib, "tf_row_projection_indexed_relu2", false),
        };
    }

    pub fn deinit(p: *Pipelines) void {
        p.plain.deinit();
        p.relu2.deinit();
        p.indexed.deinit();
        p.indexed_relu2.deinit();
    }
};

/// One dispatch: its pipeline, dims (K, N, group, rows or slots, experts, repeat) and the grid; buffers x, W, S, B, dims, out at 0-5, ids at 6.
pub const Call = struct { pipeline: mtl.Pipeline, dims: [6]i32, groups: usize, slots: usize = 1, threads: usize = 32 * groups };

/// y[rows, n] = x[rows, k] W^T (relu squared with `relu2`), x and y bf16 rows.
pub fn call(p: *const Pipelines, w: Weights, rows: usize, relu2: bool) !Call {
    try w.validate();
    if (rows == 0) return error.NoRows;
    return .{ .pipeline = if (relu2) p.relu2 else p.plain, .dims = .{ @intCast(w.k), @intCast(w.n), @intCast(w.group), @intCast(rows), 1, 1 }, .groups = w.n / (columns * groups) };
}

/// y[slots, n] = x[slot / repeat] W[ids[slot]]^T; the grid's y is the slot.
pub fn callIndexed(p: *const Pipelines, x: Experts, slots: usize, relu2: bool) !Call {
    try x.validate();
    if (slots == 0 or slots % x.repeat != 0 or slots > std.math.maxInt(u16)) return error.NoRows;
    return .{ .pipeline = if (relu2) p.indexed_relu2 else p.indexed, .dims = .{ @intCast(x.one.k), @intCast(x.one.n), @intCast(x.one.group), @intCast(slots), @intCast(x.experts), @intCast(x.repeat) }, .groups = x.one.n / (columns * groups), .slots = slots };
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
}
