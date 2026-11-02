//! GLM-OCR pointwise operators with explicit checkpoint rounding nodes.
const std = @import("std");
const hip = @import("hip");
const kernels = @import("glm_ocr_kernels");

pub const Ops = struct {
    module: hip.Module,
    bias: hip.Function,
    bias_fill: hip.Function,
    cast: hip.Function,
    add_fn: hip.Function,
    silu_mul: hip.Function,
    gelu_fn: hip.Function,
    merge_layout: hip.Function,
    split: hip.Function,
    norm_fn: hip.Function,
    vision_rope: hip.Function,
    text_rope: hip.Function,

    pub fn load(r: *hip.Runtime) !Ops {
        var buf: [64]u8 = undefined;
        const full = try hip.Device.archName(r, 0, &buf);
        const arch = full[0 .. std.mem.indexOfScalar(u8, full, ':') orelse full.len];
        var module = try hip.Module.loadForArchitecture(r, &.{.{ .arch = kernels.arch, .bytes = kernels.bytes[0..] }}, arch);
        errdefer module.unload();
        return .{
            .module = module,
            .bias = try module.function("tf_glmocr_bias"),
            .bias_fill = try module.function("tf_glmocr_bias_fill"),
            .cast = try module.function("tf_glmocr_cast"),
            .add_fn = try module.function("tf_glmocr_add"),
            .silu_mul = try module.function("tf_glmocr_silu_mul"),
            .gelu_fn = try module.function("tf_glmocr_gelu"),
            .merge_layout = try module.function("tf_glmocr_merge_layout"),
            .split = try module.function("tf_glmocr_qkv_split"),
            .norm_fn = try module.function("tf_glmocr_norm"),
            .vision_rope = try module.function("tf_glmocr_vision_rope"),
            .text_rope = try module.function("tf_glmocr_text_rope"),
        };
    }

    pub fn unload(o: *Ops) void {
        o.module.unload();
        o.* = undefined;
    }

    /// In-place BF16 add of a repeated bias row, after the product's BF16 output rounding.
    pub fn addBias(o: *const Ops, stream: hip.Stream, x: u64, bias: u64, rows: usize, width: usize) !void {
        return o.biasGo(stream, x, bias, rows, width, false);
    }

    pub fn fillBias(o: *const Ops, stream: hip.Stream, x: u64, bias: u64, rows: usize, width: usize) !void {
        return o.biasGo(stream, x, bias, rows, width, true);
    }

    fn biasGo(o: *const Ops, stream: hip.Stream, x: u64, bias: u64, rows: usize, width: usize, fill: bool) !void {
        if (rows == 0 or width == 0 or width > 59392 or x == 0 or bias == 0 or x % 2 != 0 or bias % 2 != 0) return error.Invalid;
        const count = std.math.mul(u64, rows, width) catch return error.Invalid;
        const bytes = std.math.mul(u64, count, 2) catch return error.Invalid;
        const end = std.math.add(u64, x, bytes) catch return error.Invalid;
        const bias_end = std.math.add(u64, bias, 2 * width) catch return error.Invalid;
        if (x < bias_end and bias < end) return error.Invalid;
        const blocks = (count + 255) / 256;
        if (blocks > std.math.maxInt(u32) / 256) return error.Invalid;
        var args: hip.launch.Args = .{};
        try args.add(x);
        try args.add(bias);
        try args.add(count);
        try args.add(@as(u32, @intCast(width)));
        try hip.launch.launch(if (fill) o.bias_fill else o.bias, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = 256 } }, stream, &args);
    }

    fn flat(f: hip.Function, stream: hip.Stream, count: usize, values: anytype) !void {
        if (count == 0 or count > std.math.maxInt(u32) - 255) return error.Invalid;
        var args: hip.launch.Args = .{};
        inline for (values) |v| try args.add(v);
        try hip.launch.launch(f, .{ .grid = .{ .x = @intCast((count + 255) / 256) }, .block = .{ .x = 256 } }, stream, &args);
    }

    pub fn castBf16(o: *const Ops, stream: hip.Stream, src: u64, dst: u64, count: usize) !void {
        try disjoint(try region(src, count, 4), try region(dst, count, 2));
        try flat(o.cast, stream, count, .{ src, dst, @as(u64, count) });
    }

    pub fn add(o: *const Ops, stream: hip.Stream, x: u64, y: u64, out: u64, count: usize) !void {
        try pointwise(out, &.{ x, y }, count);
        try flat(o.add_fn, stream, count, .{ x, y, out, @as(u64, count) });
    }

    pub fn siluMul(o: *const Ops, stream: hip.Stream, gate: u64, up: u64, out: u64, count: usize) !void {
        try pointwise(out, &.{ gate, up }, count);
        try flat(o.silu_mul, stream, count, .{ gate, up, out, @as(u64, count) });
    }

    pub fn gelu(o: *const Ops, stream: hip.Stream, x: u64, count: usize) !void {
        _ = try region(x, count, 2);
        try flat(o.gelu_fn, stream, count, .{ x, @as(u64, count) });
    }

    pub fn merge(o: *const Ops, stream: hip.Stream, x: u64, out: u64, rows: usize, width: usize) !void {
        if (rows % 4 != 0 or width != 1024) return error.Invalid;
        const count = try product(rows, width);
        try disjoint(try region(x, count, 2), try region(out, count, 2));
        try flat(o.merge_layout, stream, count, .{ x, out, @as(u64, count), @as(u32, @intCast(width)), @as(c_int, 0) });
    }

    pub fn channelsLastFilter(o: *const Ops, stream: hip.Stream, x: u64, out: u64, channels: usize) !void {
        if (channels == 0 or channels > 1536) return error.Invalid;
        const count = channels * 1024 * 4;
        try disjoint(try region(x, count, 2), try region(out, count, 2));
        try flat(o.merge_layout, stream, count, .{ x, out, @as(u64, count), @as(u32, 1024), @as(c_int, 1) });
    }

    pub fn splitQkv(o: *const Ops, stream: hip.Stream, x: u64, q: u64, k: u64, v: u64, rows: usize, width: usize) !void {
        if (width != 1024) return error.Invalid;
        const count = try product(rows, width);
        const source = try region(x, try product(count, 3), 2);
        const destinations = [_]Region{ try region(q, count, 2), try region(k, count, 2), try region(v, count, 2) };
        for (destinations, 0..) |dst, i| {
            try disjoint(source, dst);
            for (destinations[0..i]) |previous| try disjoint(previous, dst);
        }
        try flat(o.split, stream, count, .{ x, q, k, v, @as(u64, count), @as(u32, @intCast(width)) });
    }

    pub fn norm(o: *const Ops, stream: hip.Stream, x: u64, weight: u64, bias: ?u64, out: u64, rows: usize, width: usize) !void {
        if (rows == 0 or rows > std.math.maxInt(u32) / 256 or
            (width != 64 and width != 1024 and width != 1536) or (bias != null and width != 1536)) return error.Invalid;
        const count = try product(rows, width);
        try pointwise(out, &.{x}, count);
        const output = try region(out, count, 2);
        try disjoint(output, try region(weight, width, 2));
        if (bias) |b| try disjoint(output, try region(b, width, 2));
        var args: hip.launch.Args = .{};
        inline for (.{ x, weight, bias orelse 0, out, @as(u32, @intCast(width)), @as(f32, 1e-5), @as(c_int, @intFromBool(bias != null)) }) |v| try args.add(v);
        try hip.launch.launch(o.norm_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &args);
    }

    pub fn visionRope(o: *const Ops, stream: hip.Stream, x: u64, inverse: u64, rows: usize, gw: usize) !void {
        if (gw == 0 or gw > std.math.maxInt(u32) or gw % 2 != 0 or rows % (try product(gw, 2)) != 0) return error.Invalid;
        const count = try product(rows, 16 * 32);
        try disjoint(try region(x, try product(count, 2), 2), try region(inverse, 16, 4));
        try flat(o.vision_rope, stream, count, .{ x, inverse, @as(u64, count), @as(u32, @intCast(gw)) });
    }

    pub fn textRope(o: *const Ops, stream: hip.Stream, x: u64, inverse: u64, pos: u64, rows: usize, heads: usize) !void {
        if (heads != 8 and heads != 16) return error.Invalid;
        const count = try product(rows, heads * 64);
        const input = try region(x, try product(count, 2), 2);
        try disjoint(input, try region(inverse, 64, 4));
        try disjoint(input, try region(pos, try product(rows, 3), 4));
        try flat(o.text_rope, stream, count, .{ x, inverse, pos, @as(u64, count), @as(u32, @intCast(heads)) });
    }
};

const Region = struct { start: u64, end: u64 };
fn product(x: usize, y: usize) error{Invalid}!usize {
    return std.math.mul(usize, x, y) catch error.Invalid;
}
fn region(p: u64, count: usize, size: usize) error{Invalid}!Region {
    if (p == 0 or p % size != 0 or count == 0) return error.Invalid;
    const bytes = try product(count, size);
    return .{ .start = p, .end = std.math.add(u64, p, bytes) catch return error.Invalid };
}
fn disjoint(x: Region, y: Region) error{Invalid}!void {
    if (x.start < y.end and y.start < x.end) return error.Invalid;
}
fn pointwise(out: u64, inputs: []const u64, count: usize) error{Invalid}!void {
    const dst = try region(out, count, 2);
    for (inputs) |input| {
        const src = try region(input, count, 2);
        if (input != out) try disjoint(src, dst);
    }
}

test "GLM-OCR HIP ops: bias shape and aliases are refused before device calls" {
    const o: Ops = undefined;
    const s: hip.Stream = undefined;
    try std.testing.expectError(error.Invalid, o.addBias(s, 2, 4, 0, 1));
    try std.testing.expectError(error.Invalid, o.addBias(s, 2, 4, 1, 0));
    try std.testing.expectError(error.Invalid, o.addBias(s, 2, 4, 1, 10));
    try std.testing.expectError(error.Invalid, o.addBias(s, 3, 1000, 1, 10));
    try std.testing.expectError(error.Invalid, o.addBias(s, 2, 1000, std.math.maxInt(usize), 10));
    try std.testing.expectError(error.Invalid, o.castBf16(s, 100, 102, 32));
    try std.testing.expectError(error.Invalid, o.add(s, 100, 1000, 102, 32));
    try std.testing.expectError(error.Invalid, o.siluMul(s, 100, 1000, 102, 32));
    try std.testing.expectError(error.Invalid, o.gelu(s, 0, 32));
    try std.testing.expectError(error.Invalid, o.merge(s, 100, 100, 4, 1024));
    try std.testing.expectError(error.Invalid, o.splitQkv(s, 100, 10000, 10000, 20000, 4, 1024));
    try std.testing.expectError(error.Invalid, o.norm(s, 100, 10000, null, 102, 4, 1024));
    try std.testing.expectError(error.Invalid, o.visionRope(s, 100, 102, 4, 2));
    try std.testing.expectError(error.Invalid, o.textRope(s, 100, 102, 10000, 4, 16));
}
