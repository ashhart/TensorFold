//! Shared vision operations: BF16 ops through the launcher (dots.mocr), dense kernels loaded by name (Nemotron Parse).
const std = @import("std");
const Ops = @import("ops.zig").Ops;
const t = @import("types.zig");
const raw = @import("../launches/vision.zig");
const count = @import("../launches/columns.zig").count;
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const Stream = @import("../stream.zig").Stream;
const launch = @import("../launch.zig");
const Tensor = t.Tensor;
pub const Error = t.Error;

const Span = struct { at: u64, bytes: usize };

fn disjoint(spans: []const Span) t.Error!void {
    for (spans, 0..) |a, i| {
        if (a.at == 0 or a.at % 2 != 0) return error.BadShape;
        const end = std.math.add(u64, a.at, a.bytes) catch return error.BadShape;
        for (spans[0..i]) |b| {
            const bend = std.math.add(u64, b.at, b.bytes) catch return error.BadShape;
            if (a.at < bend and b.at < end) return error.BadShape;
        }
    }
}

pub fn layerNorm(o: Ops, x: t.Tensor, w: u64, b: u64, out: t.Tensor, rows: usize, width: usize, eps: f32) t.Error!void {
    if (x.kind != .bf16 or out.kind != .bf16 or !std.math.isFinite(eps) or eps <= 0) return error.BadShape;
    const n = try count(rows, width);
    // Weight and bias may be the same read-only buffer; each must be disjoint from both activations.
    try disjoint(&.{ .{ .at = x.ptr, .bytes = n * 2 }, .{ .at = out.ptr, .bytes = n * 2 } });
    for ([_]u64{ w, b }) |p| try disjoint(&.{
        .{ .at = p, .bytes = width * 2 }, .{ .at = x.ptr, .bytes = n * 2 }, .{ .at = out.ptr, .bytes = n * 2 },
    });
    try raw.layerNorm(o.l, x.ptr, w, b, out.ptr, rows, width, eps, o.stream);
}

pub fn gelu(o: Ops, x: t.Tensor, out: t.Tensor, rows: usize, width: usize) t.Error!void {
    if (x.kind != .bf16 or out.kind != .bf16) return error.BadShape;
    const n = try count(rows, width);
    if (x.ptr == out.ptr) {
        if (x.ptr == 0 or x.ptr % 2 != 0 or x.ptr > std.math.maxInt(u64) - n * 2) return error.BadShape;
    } else try disjoint(&.{ .{ .at = x.ptr, .bytes = n * 2 }, .{ .at = out.ptr, .bytes = n * 2 } });
    try raw.gelu(o.l, x.ptr, out.ptr, n, o.stream);
}

fn geometry(rows: usize, heads: usize, d: usize) t.Error!usize {
    if (heads == 0 or heads > 256 or d == 0 or d > 256 or d % 64 != 0) return error.BadShape;
    return count(rows, heads * d);
}

pub fn rope(o: Ops, qkv: t.Tensor, positions: u64, q: u64, k: u64, v: u64, rows: usize, heads: usize, d: usize, theta: f32) t.Error!void {
    if (qkv.kind != .bf16 or !std.math.isFinite(theta) or theta <= 0 or positions % 4 != 0 or q % 4 != 0) return error.BadShape;
    const n = try geometry(rows, heads, d);
    try disjoint(&.{
        .{ .at = qkv.ptr, .bytes = n * 6 }, .{ .at = positions, .bytes = rows * 8 },
        .{ .at = q, .bytes = n * 4 },       .{ .at = k, .bytes = n * 2 },
        .{ .at = v, .bytes = n * 2 },
    });
    try raw.rope(o.l, qkv.ptr, positions, q, k, v, rows, heads, d, theta, o.stream);
}

pub fn attentionRows(o: Ops, x: u64, out: t.Tensor, rows: usize, heads: usize, d: usize) t.Error!void {
    if (out.kind != .bf16 or x % 4 != 0) return error.BadShape;
    const n = try geometry(rows, heads, d);
    try disjoint(&.{ .{ .at = x, .bytes = n * 4 }, .{ .at = out.ptr, .bytes = n * 2 } });
    try raw.attentionRows(o.l, x, out.ptr, rows, heads, d, o.stream);
}

/// One image per call. No cross-image visibility or causal mask is used.
pub fn fullAttention(o: Ops, q: u64, k: u64, v: u64, out: u64, rows: usize, heads: usize, d: usize) t.Error!void {
    const n = try geometry(rows, heads, d);
    if (q % 4 != 0 or out % 4 != 0) return error.BadShape;
    try disjoint(&.{
        .{ .at = q, .bytes = n * 4 }, .{ .at = k, .bytes = n * 2 },
        .{ .at = v, .bytes = n * 2 }, .{ .at = out, .bytes = n * 4 },
    });
    try raw.full(o.l, q, k, v, out, rows, heads, d, 1 / @sqrt(@as(f32, @floatFromInt(d))), o.stream);
}

pub const Kernels = struct {
    embed_fn: Function,
    linear_fn: Function,
    norm_fn: Function,
    point_fn: Function,
    layout_fn: Function,
    resize_fn: Function,
    position_fn: Function,

    pub fn load(module: Module) !Kernels {
        return .{
            .embed_fn = try module.function("tf_vl_embed"),
            .linear_fn = try module.function("tf_vl_linear"),
            .norm_fn = try module.function("tf_vl_layernorm"),
            .point_fn = try module.function("tf_vl_pointwise"),
            .layout_fn = try module.function("tf_vl_layout"),
            .resize_fn = try module.function("tf_vl_resize"),
            .position_fn = try module.function("tf_vl_position"),
        };
    }

    /// Device IDs must be in the table's admitted range; embedding scaling rounds in the output dtype.
    pub fn embed(k: Kernels, s: Stream, table: Tensor, ids: u64, out: Tensor, rows: usize, width: usize, scale: f32) Error!void {
        if (table.kind != out.kind or rows == 0 or rows > 16384 or width == 0 or width > 16384 or !std.math.isFinite(scale)) return error.BadShape;
        var args: launch.Args = .{};
        for ([_]u64{ table.ptr, ids, out.ptr }) |v| try args.add(v);
        try args.add(@as(c_int, @intCast(rows)));
        try args.add(@as(c_int, @intCast(width)));
        try args.add(@backingInt(out.kind));
        try args.add(scale);
        try flat(k.embed_fn, s, rows * width, &args);
    }

    /// Weights and bias have x's dtype; the sole output rounding is after the biased FP32 dot.
    pub fn linear(k: Kernels, s: Stream, x: Tensor, w: u64, bias: ?u64, out: Tensor, rows: usize, n: usize, width: usize) Error!void {
        if (rows == 0 or n == 0 or width == 0 or rows > 16384 or n > 1 << 20 or width > 16384) return error.BadShape;
        var args: launch.Args = .{};
        try args.add(x.ptr);
        try args.add(w);
        try args.add(bias orelse @as(u64, 0));
        try args.add(out.ptr);
        for ([_]usize{ rows, n, width }) |v| try args.add(@as(c_int, @intCast(v)));
        try args.add(@backingInt(x.kind));
        try args.add(@backingInt(out.kind));
        try launch.launch(k.linear_fn, .{ .grid = .{ .x = @intCast((n + 63) / 64), .y = @intCast((rows + 15) / 16) }, .block = .{ .x = 256 } }, s, &args);
    }

    /// The affine parameters have x's dtype, as torch LayerNorm after model.to(bfloat16).
    pub fn norm(k: Kernels, s: Stream, x: Tensor, weight: u64, bias: u64, out: Tensor, rows: usize, width: usize, eps: f32) Error!void {
        if (x.kind != out.kind or rows == 0 or rows > 16384 or width == 0 or width > 16384 or !(eps > 0) or !std.math.isFinite(eps)) return error.BadShape;
        var args: launch.Args = .{};
        for ([_]u64{ x.ptr, weight, bias, out.ptr }) |p| try args.add(p);
        try args.add(@as(c_int, @intCast(rows)));
        try args.add(@as(c_int, @intCast(width)));
        try args.add(@backingInt(x.kind));
        try args.add(eps);
        try launch.launch(k.norm_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &args);
    }

    pub const Point = enum(c_int) { gelu, add, scale };
    pub fn point(k: Kernels, s: Stream, x: Tensor, y: ?Tensor, out: Tensor, elements: usize, mode: Point, scale: f32) Error!void {
        if (x.kind != out.kind or elements == 0 or elements > std.math.maxInt(c_int)) return error.BadShape;
        if (mode == .add and (y == null or y.?.kind != x.kind)) return error.BadShape;
        var args: launch.Args = .{};
        try args.add(x.ptr);
        try args.add(if (y) |v| v.ptr else @as(u64, 0));
        try args.add(out.ptr);
        try args.add(@as(c_longlong, @intCast(elements)));
        try args.add(@backingInt(x.kind));
        try args.add(@backingInt(mode));
        try args.add(scale);
        try flat(k.point_fn, s, elements, &args);
    }

    /// One axis of antialiased bilinear resizing, or adaptive-average pooling of both axes.
    pub const Resize = enum(c_int) { horizontal, vertical, area };
    pub fn resize(k: Kernels, s: Stream, src: Tensor, out: u64, input: [2]usize, output: [2]usize, width: usize, mode: Resize) Error!void {
        for ([_][2]usize{ input, output }) |pair| for (pair) |v| {
            if (v == 0 or v > 128) return error.BadShape;
        };
        if (width == 0 or width > 4096) return error.BadShape;
        if ((mode == .horizontal and input[0] != output[0]) or
            (mode == .vertical and input[1] != output[1])) return error.BadShape;
        var args: launch.Args = .{};
        try args.add(src.ptr);
        try args.add(out);
        for ([_][2]usize{ input, output }) |pair| for (pair) |v| {
            try args.add(@as(c_int, @intCast(v)));
        };
        try args.add(@as(c_int, @intCast(width)));
        try args.add(@backingInt(src.kind));
        try args.add(@backingInt(mode));
        try flat(k.resize_fn, s, output[0] * output[1] * width, &args);
    }

    /// Prefix class rows and projected patches plus BF16-rounded, content-normalized positions.
    pub fn position(k: Kernels, s: Stream, patches: Tensor, cls: u64, field: u64, out: Tensor, grid: [2]usize, content: [2]usize, hidden: usize, prefix: usize) Error!void {
        if (patches.kind != out.kind or patches.kind == .f32 or hidden == 0 or hidden > 4096 or prefix > 32) return error.BadShape;
        for (grid, content) |g, c| if (g == 0 or g > 128 or c == 0 or c > g * 16) return error.BadShape;
        var args: launch.Args = .{};
        for ([_]u64{ patches.ptr, cls, field, out.ptr }) |v| try args.add(v);
        for ([_][2]usize{ grid, content, .{ hidden, prefix } }) |pair| for (pair) |v| {
            try args.add(@as(c_int, @intCast(v)));
        };
        try args.add(@backingInt(patches.kind));
        try flat(k.position_fn, s, (grid[0] * grid[1] + prefix) * hidden, &args);
    }

    pub const Layout = enum(c_int) { heads_to_rows, horizontal_four };
    pub fn layout(k: Kernels, s: Stream, x: Tensor, out: Tensor, rows: usize, heads: usize, dim: usize, mode: Layout) Error!void {
        if (rows == 0 or rows > 16384 or heads == 0 or heads > 256 or dim == 0 or dim > 16384) return error.BadShape;
        if (mode == .horizontal_four and (heads != 1 or rows % 4 != 0)) return error.BadShape;
        var args: launch.Args = .{};
        try args.add(x.ptr);
        try args.add(out.ptr);
        for ([_]usize{ rows, heads, dim }) |v| try args.add(@as(c_int, @intCast(v)));
        try args.add(@backingInt(x.kind));
        try args.add(@backingInt(out.kind));
        try args.add(@backingInt(mode));
        try flat(k.layout_fn, s, rows * heads * dim, &args);
    }
};

fn flat(f: Function, s: Stream, elements: usize, args: *launch.Args) Error!void {
    try launch.launch(f, .{ .grid = .{ .x = @intCast((elements + 255) / 256) }, .block = .{ .x = 256 } }, s, args);
}

test "dots HIP vision: malformed geometry and buffer spans fail before a launch" {
    var o: Ops = undefined;
    o.l = undefined;
    try std.testing.expectError(error.BadShape, geometry(1, 0, 128));
    try std.testing.expectError(error.BadShape, geometry(1, 12, 127));
    try std.testing.expectError(error.BadShape, disjoint(&.{ .{ .at = 100, .bytes = 100 }, .{ .at = 102, .bytes = 2 } }));
    try std.testing.expectError(error.BadShape, layerNorm(o, .{ .ptr = 100, .kind = .bf16 }, 200, 300, .{ .ptr = 100, .kind = .bf16 }, 1, 1, 1e-6));
    try std.testing.expectError(error.BadShape, gelu(o, .{ .ptr = 100, .kind = .f32 }, .{ .ptr = 200, .kind = .bf16 }, 1, 1));
}
