//! CogViT and the learned image connector, with per-request scratch and optional development taps.
const std = @import("std");
const hip = @import("hip");
const Weights = @import("hip_weights.zig").Weights;
const Ops = @import("hip_ops.zig").Ops;
const c = @import("config.zig");
const visual = @import("schema.zig").visual;
const attention = hip.attention_aotriton;

pub const Observer = struct {
    context: *anyopaque,
    capture: *const fn (*anyopaque, []const u8, u64, usize) anyerror!void,
    pub fn emit(o: ?Observer, name: []const u8, address: u64, values: usize) !void {
        if (o) |sink| try sink.capture(sink.context, name, address, values);
    }
};

pub const Vision = struct {
    stream: hip.Stream,
    weights: *const Weights,
    ops: *const Ops,
    blas: *const hip.blas.Blas,
    flash: *const attention.Library,
    scratch: hip.DeviceBuffer,
    slots: [11]u64,
    lse: u64,
    counter: u64,
    inverse: u64,
    down_filter: u64,
    rows: usize,
    gw: usize,
    convolution_budget: usize,

    /// Admission covers all retained scratch and the maximum temporary convolution workspace.
    pub fn init(stream: hip.Stream, weights: *const Weights, ops: *const Ops, blas: *const hip.blas.Blas, flash: *const attention.Library, rows: usize, gw: usize, budget: usize) !Vision {
        if (rows == 0 or rows > 24576 or gw == 0 or gw > 16384 / 14 or gw % 2 != 0 or rows % (gw * 2) != 0 or rows / gw > 16384 / 14) return error.Invalid;
        const widths = [_]usize{ 1024, 1024, 3072, 1024, 1024, 1024, 1024, 4096, 4096, 1024, 1024 };
        var offsets: [11]usize = undefined;
        var bytes: usize = 0;
        for (widths, &offsets) |width, *offset| {
            offset.* = bytes;
            bytes += rows * width * 2;
        }
        const lse_offset = bytes;
        bytes += rows * 16 * 4;
        const counter_offset = bytes;
        const inverse_offset = bytes + 256;
        bytes += 512;
        const filter_offset = bytes;
        bytes += 1536 * 1024 * 4 * 2;
        if (budget < bytes) return error.GlmOcrVisionDoesNotFit;
        var scratch = try hip.DeviceBuffer.alloc(stream.r, bytes);
        errdefer scratch.free();
        errdefer stream.synchronize() catch {};
        var inverse: [16]f32 = undefined;
        for (&inverse, 0..) |*v, i| {
            const power: f32 = @floatCast(std.math.pow(f64, 10000, @as(f64, @floatFromInt(i)) / 16));
            v.* = 1 / power;
        }
        try scratch.upload(inverse_offset, std.mem.asBytes(&inverse));
        var slots: [11]u64 = undefined;
        for (&slots, offsets) |*p, offset| p.* = try scratch.address(offset);
        const down_filter = try scratch.address(filter_offset);
        try ops.channelsLastFilter(stream, try weights.get(visual ++ "downsample.weight"), down_filter, 1536);
        return .{ .stream = stream, .weights = weights, .ops = ops, .blas = blas, .flash = flash, .scratch = scratch, .slots = slots, .lse = try scratch.address(lse_offset), .counter = try scratch.address(counter_offset), .inverse = try scratch.address(inverse_offset), .down_filter = down_filter, .rows = rows, .gw = gw, .convolution_budget = budget - bytes };
    }

    pub fn deinit(v: *Vision) void {
        v.stream.synchronize() catch {};
        v.scratch.free();
        v.* = undefined;
    }

    fn weight(v: *const Vision, stem: []const u8, name: []const u8, suffix: []const u8) !u64 {
        var buf: [256]u8 = undefined;
        return v.weights.get(try std.fmt.bufPrint(&buf, "{s}{s}.{s}", .{ stem, name, suffix }));
    }

    fn linear(v: *const Vision, stem: []const u8, name: []const u8, x: u64, out: u64, rows: usize, input: usize, output: usize, bias: bool) !void {
        const w = try v.weight(stem, name, "weight");
        if (bias) {
            try v.ops.fillBias(v.stream, out, try v.weight(stem, name, "bias"), rows, output);
            try v.blas.projectAdd(x, w, out, rows, output, input);
        } else try v.blas.project(x, w, out, rows, output, input);
    }

    fn norm(v: *const Vision, stem: []const u8, name: []const u8, x: u64, out: u64, rows: usize, width: usize) !void {
        try v.ops.norm(v.stream, x, try v.weight(stem, name, "weight"), null, out, rows, width);
    }

    fn convolution(v: *const Vision, name: []const u8, shape: hip.convolution.Shape, x: u64, out: u64, rows: usize, width: usize) !void {
        const w = try v.weight(visual, name, "weight");
        var conv = try hip.convolution.Convolution.init(v.stream, shape, x, w, out, v.convolution_budget);
        defer conv.deinit();
        try conv.run(x, w, out);
        try v.ops.addBias(v.stream, out, try v.weight(visual, name, "bias"), rows, width);
    }

    /// The input is BF16 processor patch rows; the output is [rows/4, 1536] in the caller's owned buffer.
    pub fn encode(v: *const Vision, pixels: u64, out: u64, observer: ?Observer) !void {
        const n = v.rows;
        const x, const normed, const qkv, const q, const k, const val, const attended, const gate, const up, const work, const merged = v.slots;
        _ = merged;
        const patch_shape = try hip.convolution.Shape.of(&.{ n, 3, 2, 14, 14 }, &.{ 1024, 3, 2, 14, 14 }, &.{ 2, 14, 14 });
        try v.convolution("patch_embed.proj", patch_shape, pixels, x, n, 1024);
        try Observer.emit(observer, "vision-patch", x, n * 1024);
        for (0..c.vision_layers) |layer| {
            var stem_buf: [96]u8 = undefined;
            const stem = try std.fmt.bufPrint(&stem_buf, visual ++ "blocks.{d}.", .{layer});
            try v.norm(stem, "norm1", x, normed, n, 1024);
            if (layer == 0) try Observer.emit(observer, "vision-block0-norm1", normed, n * 1024);
            try v.linear(stem, "attn.qkv", normed, qkv, n, 1024, 3072, true);
            if (layer == 0) try Observer.emit(observer, "vision-block0-qkv", qkv, n * 3072);
            try v.ops.splitQkv(v.stream, qkv, q, k, val, n, 1024);
            try v.norm(stem, "attn.q_norm", q, q, n * 16, 64);
            try v.norm(stem, "attn.k_norm", k, k, n * 16, 64);
            if (layer == 0) try Observer.emit(observer, "vision-block0-qnorm", q, n * 1024);
            try v.ops.visionRope(v.stream, q, v.inverse, n, v.gw);
            try v.ops.visionRope(v.stream, k, v.inverse, n, v.gw);
            if (layer == 0) try Observer.emit(observer, "vision-block0-qrope", q, n * 1024);
            try v.flash.forward(v.stream, attention.rows(q, 16, n, 64), attention.rows(k, 16, n, 64), attention.rows(val, 16, n, 64), attention.rows(attended, 16, n, 64), v.lse, v.counter, false);
            if (layer == 0) try Observer.emit(observer, "vision-block0-attended", attended, n * 1024);
            try v.linear(stem, "attn.proj", attended, work, n, 1024, 1024, true);
            if (layer == 0) try Observer.emit(observer, "vision-block0-proj", work, n * 1024);
            try v.ops.add(v.stream, x, work, x, n * 1024);
            try v.norm(stem, "norm2", x, normed, n, 1024);
            if (layer == 0) try Observer.emit(observer, "vision-block0-norm2", normed, n * 1024);
            try v.linear(stem, "mlp.gate_proj", normed, gate, n, 1024, 4096, true);
            try v.linear(stem, "mlp.up_proj", normed, up, n, 1024, 4096, true);
            try v.ops.siluMul(v.stream, gate, up, gate, n * 4096);
            try v.linear(stem, "mlp.down_proj", gate, work, n, 4096, 1024, true);
            try v.ops.add(v.stream, x, work, x, n * 1024);
            if (layer == 0) try Observer.emit(observer, "vision-block0", x, n * 1024);
            if (observer != null) {
                var name: [48]u8 = undefined;
                try Observer.emit(observer, try std.fmt.bufPrint(&name, "vision-layer-{d}", .{layer}), x, n * 1024);
            }
        }
        try Observer.emit(observer, "vision-last", x, n * 1024);
        try v.norm(visual, "post_layernorm", x, normed, n, 1024);
        try Observer.emit(observer, "vision-norm", normed, n * 1024);
        const down_shape = try (try hip.convolution.Shape.of(&.{ n / 4, 1024, 2, 2 }, &.{ 1536, 1024, 2, 2 }, &.{ 2, 2 })).channelsLast();
        {
            var conv = try hip.convolution.Convolution.init(v.stream, down_shape, normed, v.down_filter, x, v.convolution_budget);
            defer conv.deinit();
            try conv.run(normed, v.down_filter, x);
            try Observer.emit(observer, "vision-downsample-product", x, n / 4 * 1536);
            try v.ops.addBias(v.stream, x, try v.weight(visual, "downsample", "bias"), n / 4, 1536);
        }
        try Observer.emit(observer, "vision-downsample", x, n / 4 * 1536);
        try v.linear(visual, "merger.proj", x, normed, n / 4, 1536, 1536, false);
        try Observer.emit(observer, "vision-merger-proj", normed, n / 4 * 1536);
        try v.ops.norm(v.stream, normed, try v.weight(visual, "merger.post_projection_norm", "weight"), try v.weight(visual, "merger.post_projection_norm", "bias"), normed, n / 4, 1536);
        try Observer.emit(observer, "vision-merger-norm", normed, n / 4 * 1536);
        try v.ops.gelu(v.stream, normed, n / 4 * 1536);
        try Observer.emit(observer, "vision-merger-gelu", normed, n / 4 * 1536);
        try v.linear(visual, "merger.gate_proj", normed, gate, n / 4, 1536, 4608, false);
        try Observer.emit(observer, "vision-merger-gate", gate, n / 4 * 4608);
        try v.linear(visual, "merger.up_proj", normed, up, n / 4, 1536, 4608, false);
        try Observer.emit(observer, "vision-merger-up", up, n / 4 * 4608);
        try v.ops.siluMul(v.stream, gate, up, gate, n / 4 * 4608);
        try v.linear(visual, "merger.down_proj", gate, out, n / 4, 4608, 1536, false);
        try Observer.emit(observer, "vision-connector", out, n / 4 * 1536);
    }
};

test "GLM-OCR HIP vision admission: geometry and scratch budget are refused before device calls" {
    const r: hip.Runtime = undefined;
    const s: hip.Stream = .{ .r = &r, .handle = null };
    const w: Weights = undefined;
    const ops: Ops = undefined;
    const blas: hip.blas.Blas = undefined;
    const flash: attention.Library = undefined;
    try std.testing.expectError(error.Invalid, Vision.init(s, &w, &ops, &blas, &flash, 0, 2, 0));
    try std.testing.expectError(error.Invalid, Vision.init(s, &w, &ops, &blas, &flash, 24580, 2, 0));
    try std.testing.expectError(error.Invalid, Vision.init(s, &w, &ops, &blas, &flash, 4, 3, 0));
    try std.testing.expectError(error.Invalid, Vision.init(s, &w, &ops, &blas, &flash, 2344, 1172, 0));
    try std.testing.expectError(error.GlmOcrVisionDoesNotFit, Vision.init(s, &w, &ops, &blas, &flash, 2340, 1170, 0));
}
