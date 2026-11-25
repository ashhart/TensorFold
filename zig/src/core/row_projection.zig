//! The lane projection's path for chips without tensor units: row-layout 2- to 8-bit weights, one read a window.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

/// 4-bit groups of 64 with fp32 sums take simdgroup-matrix tiles instead (8 rows a weight read).
pub const simd = @import("simd_qmm.zig");
/// Groups of 64 at any width with fp32 sums: the same tiles, rows at a stride.
pub const simd_bits = @import("simd_qmm_bits.zig");

/// The narrow kernel's rows per weight read; wide windows use 8, 4, 2 and 1.
pub const batch = 2;

/// The cutover table and its chip key (row_cutover.zig, host-tested without Metal).
pub const cut = @import("row_cutover.zig");
pub const Tier = cut.Tier;
pub const Chip = cut.Chip;
pub const chipOf = cut.chipOf;
pub const never = cut.never;
pub const Cutover = cut.Cutover;
pub const cutovers = cut.cutovers;
pub const wideFrom = cut.wideFrom;
pub const keyLine = cut.keyLine;

/// Output columns a simdgroup and simdgroups a threadgroup (the kernel's RPS and SG).
pub const columns = 4;
pub const groups = 2;

/// The code widths the kernels unpack.
pub const bit_widths = cut.bit_widths;

/// A group's input sum for the bias: fp32 in order (tensor path's xsum), or bf16-rounded per add (Nemotron's experts).
pub const Sum = enum { f32, bf16 };

/// Matrix [n, k] as the checkpoint stores it: words [n][k*bits/32], scales, biases [n][k/group] bf16, at byte offsets.
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
        if (x.n == 0 or x.n % (columns * groups) != 0 or x.k == 0 or x.k % 16 != 0 or x.group < 16 or x.group % 16 != 0 or x.k % x.group != 0 or 512 % x.group != 0 or x.k * x.bits % 32 != 0) return error.UnsupportedRowShape;
        if (std.mem.indexOfScalar(u8, &bit_widths, x.bits) == null) return error.UnsupportedRowBits;
    }

    /// The pipeline index: by code width, then the sum mode.
    fn variant(x: Weights) usize {
        return std.mem.indexOfScalar(u8, &bit_widths, x.bits).? * 2 + @backingInt(x.sum);
    }
};

/// A checked MLX affine triple (core/quant/mlx_affine.zig) at its tensors' buffer offsets, as the row kernels read it.
pub fn affine(t: anytype) !Weights {
    const out: Weights = .{ .w = t.weight.buffer, .w_off = t.weight.offset, .scales = t.scales.buffer, .s_off = t.scales.offset, .biases = t.biases.buffer, .b_off = t.biases.offset, .n = t.n, .k = t.k, .group = t.group, .bits = t.bits };
    try out.validate();
    return out;
}

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
    strided: [variants]mtl.Pipeline,
    wide_plain: [variants]mtl.Pipeline,
    wide_relu2: [variants]mtl.Pipeline,
    wide_strided: [variants]mtl.Pipeline,
    wide: bool = false, // the 8-row kernels are built only on chips with tensor units (the M5 path)
    wide_from: [bit_widths.len]usize = @splat(never), // the narrowest window that takes them, a code width each

    /// Whether a window of `rows` of `w` takes the wide kernels.
    pub fn takesWide(p: *const Pipelines, w: Weights, rows: usize) bool {
        return p.wide and cut.takes(&p.wide_from, w.bits, rows);
    }

    pub fn load(device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const wide = device.tensorUnits();
        const lib = try mtl.Library.fromSource(device, source(wide), mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipelines = undefined;
        p.wide = wide;
        const chip = chipOf(device.generation(), std.mem.span(device.name()));
        for (&p.wide_from, bit_widths) |*from, bits| from.* = if (wide) wideFrom(chip, bits) orelse never else never;
        inline for (bit_widths, 0..) |bits, b| inline for (.{ "f32", "bf16" }, 0..) |sum, m| {
            const i = b * 2 + m;
            p.plain[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_row_projection_q{d}_{s}", .{ bits, sum }), false);
            p.relu2[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_row_projection_relu2_q{d}_{s}", .{ bits, sum }), false);
            p.indexed[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_row_projection_indexed_q{d}_{s}", .{ bits, sum }), false);
            p.indexed_relu2[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_row_projection_indexed_relu2_q{d}_{s}", .{ bits, sum }), false);
            p.strided[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_row_projection_strided_q{d}_{s}", .{ bits, sum }), false);
            if (wide) {
                p.wide_plain[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_row_projection_wide_q{d}_{s}", .{ bits, sum }), false);
                p.wide_relu2[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_row_projection_wide_relu2_q{d}_{s}", .{ bits, sum }), false);
                p.wide_strided[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_row_projection_wide_strided_q{d}_{s}", .{ bits, sum }), false);
            }
        };
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        for (p.plain) |x| x.deinit();
        for (p.relu2) |x| x.deinit();
        for (p.indexed) |x| x.deinit();
        for (p.indexed_relu2) |x| x.deinit();
        for (p.strided) |x| x.deinit();
        if (p.wide) {
            for (p.wide_plain) |x| x.deinit();
            for (p.wide_relu2) |x| x.deinit();
            for (p.wide_strided) |x| x.deinit();
        }
    }
};

/// Chips without tensor units compile the original narrow source, with no wide definitions or control edits.
pub fn source(wide: bool) []const u8 {
    return if (wide) sources.core_row_projection else sources.core_row_projection_legacy;
}

/// Dispatch: pipeline, dims (K, N, group, rows/slots, experts, repeat), grid; buffers x, W, S, B, dims, out 0-5, ids 6.
pub const Call = struct { pipeline: mtl.Pipeline, dims: [6]i32, groups: usize, slots: usize = 1, threads: usize = 32 * groups };

/// y[rows, n] = x[rows, k] W^T (relu squared with `relu2`), x and y bf16 rows.
pub fn call(p: *const Pipelines, w: Weights, rows: usize, relu2: bool) !Call {
    try w.validate();
    if (rows == 0) return error.NoRows;
    const i = w.variant();
    const pipeline = if (p.takesWide(w, rows)) (if (relu2) p.wide_relu2[i] else p.wide_plain[i]) else if (relu2) p.relu2[i] else p.plain[i];
    return .{ .pipeline = pipeline, .dims = .{ @intCast(w.k), @intCast(w.n), @intCast(w.group), @intCast(rows), 1, 1 }, .groups = w.n / (columns * groups) };
}

/// `call` with output row m at y + m * `stride` values: a fused stack member writes its columns of the rows.
pub fn callStrided(p: *const Pipelines, w: Weights, rows: usize, stride: usize) !Call {
    try w.validate();
    if (rows == 0) return error.NoRows;
    if (stride < w.n or stride > std.math.maxInt(i32)) return error.UnsupportedRowStride;
    return .{ .pipeline = if (p.takesWide(w, rows)) p.wide_strided[w.variant()] else p.strided[w.variant()], .dims = .{ @intCast(w.k), @intCast(w.n), @intCast(w.group), @intCast(rows), @intCast(stride), 1 }, .groups = w.n / (columns * groups) };
}

/// y[slots, n] = x[slot / repeat] W[ids[slot]]^T; the grid's y is the slot.
pub fn callIndexed(p: *const Pipelines, x: Experts, slots: usize, relu2: bool) !Call {
    try x.validate();
    if (slots == 0 or slots % x.repeat != 0 or slots > std.math.maxInt(u16)) return error.NoRows;
    const i = x.one.variant();
    return .{ .pipeline = if (relu2) p.indexed_relu2[i] else p.indexed[i], .dims = .{ @intCast(x.one.k), @intCast(x.one.n), @intCast(x.one.group), @intCast(slots), @intCast(x.experts), @intCast(x.repeat) }, .groups = x.one.n / (columns * groups), .slots = slots };
}

test {
    _ = simd;
    _ = simd_bits;
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
    try (Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 256, .k = 2688, .bits = 3 }).validate();
    try (Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 256, .k = 2688, .bits = 5 }).validate();
    try std.testing.expectError(error.UnsupportedRowBits, (Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 256, .k = 2688, .bits = 7 }).validate());
}

test "3- and 5-bit rows take whole words of codes" {
    for ([_]u8{ 3, 5 }) |bits| for ([_]usize{ 32, 64, 128 }) |group| {
        try (Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 256, .k = 2688, .group = group, .bits = bits }).validate();
        try (Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 5120, .k = 17408, .group = group, .bits = bits }).validate();
    };
    // 48 codes of 3 or 5 bits end inside a word: the checkpoint cannot hold that row
    try std.testing.expectError(error.UnsupportedRowShape, (Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 256, .k = 48, .group = 16, .bits = 3 }).validate());
    try std.testing.expectError(error.UnsupportedRowShape, (Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 256, .k = 48, .group = 16, .bits = 5 }).validate());
    try (Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 256, .k = 48, .group = 16, .bits = 6 }).validate();
}

test "every width has its own pipeline slot" {
    var seen: [variants]bool = @splat(false);
    for (bit_widths) |bits| for ([_]Sum{ .f32, .bf16 }) |sum| {
        const i = (Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 8, .k = 64, .bits = bits, .sum = sum }).variant();
        try std.testing.expect(!seen[i]);
        seen[i] = true;
    };
}

test "a strided call writes rows at least a matrix wide" {
    const p: Pipelines = undefined;
    const w = Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 64, .k = 128, .bits = 5 };
    try std.testing.expectError(error.UnsupportedRowStride, callStrided(&p, w, 2, 63));
    try std.testing.expectError(error.NoRows, callStrided(&p, w, 0, 64));
    const c = try callStrided(&p, w, 2, 192);
    try std.testing.expectEqual([6]i32{ 128, 64, 64, 2, 192, 1 }, c.dims);
}

const pipelines_have_ids = @import("builtin").os.tag == .macos and (@typeInfo(mtl.Pipeline) != .@"struct" or @hasField(mtl.Pipeline, "id"));

// Real Metal uses a pointer id. Recording pipelines can use an integer id or be integers.
fn testPipeline(id: usize) mtl.Pipeline {
    if (comptime @typeInfo(mtl.Pipeline) != .@"struct") return @intCast(id);
    return .{ .id = if (comptime @typeInfo(@FieldType(mtl.Pipeline, "id")) == .pointer) @ptrFromInt(id) else @intCast(id) };
}
fn testPipelineId(p: mtl.Pipeline) usize {
    if (comptime @typeInfo(mtl.Pipeline) != .@"struct") return @intCast(p);
    return if (comptime @typeInfo(@TypeOf(p.id)) == .pointer) @intFromPtr(p.id) else @intCast(p.id);
}

test "row dispatch takes the wide pipelines from the cutover at each width" {
    if (comptime !pipelines_have_ids) return error.SkipZigTest; // inert stand-ins carry no id to compare
    var p: Pipelines = undefined;
    p.wide = true;
    inline for (.{ "plain", "relu2", "strided", "wide_plain", "wide_relu2", "wide_strided" }, 1..) |field, tag| {
        for (&@field(p, field), 0..) |*pipeline, i| pipeline.* = testPipeline(0x1000 + tag * 0x100 + i);
    }
    const max = [bit_widths.len]usize{ 32, never, never, never, never, never }; // an M5 Max's: each width its own
    for ([_]?usize{ 4, 8, 16, 32, never, null }) |uniform| {
        p.wide_from = if (uniform) |u| @splat(u) else max;
        for (bit_widths) |bits| for ([_]Sum{ .f32, .bf16 }) |sum| {
            const w = Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 64, .k = 128, .bits = bits, .sum = sum };
            const i = w.variant();
            const from = uniform orelse max[i / 2];
            for (1..33) |rows| {
                for ([_]bool{ false, true }) |relu2| {
                    const expected = if (rows >= from) (if (relu2) p.wide_relu2[i] else p.wide_plain[i]) else if (relu2) p.relu2[i] else p.plain[i];
                    try std.testing.expectEqual(testPipelineId(expected), testPipelineId((try call(&p, w, rows, relu2)).pipeline));
                }
                const expected = if (rows >= from) p.wide_strided[i] else p.strided[i];
                try std.testing.expectEqual(testPipelineId(expected), testPipelineId((try callStrided(&p, w, rows, 128)).pipeline));
            }
        };
    }
}

test "row dispatch without tensor units keeps main's pipelines at every row count" {
    if (comptime !pipelines_have_ids) return error.SkipZigTest; // inert stand-ins carry no id to compare
    var p: Pipelines = undefined;
    p.wide = false;
    p.wide_from = @splat(1); // a stale cutover must not matter without the wide pipelines
    inline for (.{ "plain", "relu2", "strided", "wide_plain", "wide_relu2", "wide_strided" }, 1..) |field, tag| {
        for (&@field(p, field), 0..) |*pipeline, i| pipeline.* = testPipeline(0x1000 + tag * 0x100 + i);
    }
    for (bit_widths) |bits| for ([_]Sum{ .f32, .bf16 }) |sum| {
        const w = Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = 64, .k = 128, .bits = bits, .sum = sum };
        const i = w.variant();
        for (1..33) |rows| {
            for ([_]bool{ false, true }) |relu2| {
                const expected = if (relu2) p.relu2[i] else p.plain[i];
                try std.testing.expectEqual(testPipelineId(expected), testPipelineId((try call(&p, w, rows, relu2)).pipeline));
            }
            try std.testing.expectEqual(testPipelineId(p.strided[i]), testPipelineId((try callStrided(&p, w, rows, 128)).pipeline));
        }
    };
}

test "without tensor units the source has no wide grouping, tensor units retain the wide source" {
    const narrow = source(false);
    try std.testing.expect(std.mem.indexOf(u8, narrow, "tf_row_projection_wide") == null);
    try std.testing.expect(std.mem.indexOf(u8, narrow, "tf_row_projection_body") == null);
    try std.testing.expect(std.mem.indexOf(u8, narrow, "TF_RP_BAD_ROWS") == null);
    try std.testing.expect(std.mem.indexOf(u8, narrow, "tf_row_projection_indexed") != null);
    try std.testing.expect(std.mem.indexOf(u8, source(true), "tf_row_projection_wide") != null);
    try std.testing.expectEqualStrings(sources.core_row_projection, source(true));
}
