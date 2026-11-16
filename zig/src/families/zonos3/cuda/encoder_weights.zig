//! The codec encoder's acoustic branch as published (generator.pt, fp32): resampler, conv front end, five stages.
const std = @import("std");
const cuda = @import("cuda");
const pth = @import("core").torch_pth;
const dec = @import("codec_weights.zig");

pub const resample_in = 735;
pub const resample_out = 736;
pub const resample_taps = 769;
pub const dilations = [3]usize{ 1, 3, 9 };

/// A front-end block: three residual units over `channels`, then a strided convolution to twice as many.
pub const BlockShape = struct { channels: usize, stride: usize };
pub const blocks = [2]BlockShape{ .{ .channels = 64, .stride = 2 }, .{ .channels = 128, .stride = 4 } };
pub const patch = 23; // front-end frames a 240 Hz row
pub const front_width = 256;

/// The stages after the patch: 240, 120, 60, 30 and 15 Hz; each but the last is followed by a patch down by 2.
pub const stages = [5]dec.StageShape{
    .{ .in = front_width * patch, .d = 768, .ff = 3072, .ctx = 480, .out = 384 },
    .{ .in = 768, .d = 768, .ff = 3072, .ctx = 480, .out = 384 },
    .{ .in = 768, .d = 768, .ff = 3072, .ctx = 480, .out = 384 },
    .{ .in = 768, .d = 768, .ff = 3072, .ctx = 300, .out = 640 },
    .{ .in = 1280, .d = 1280, .ff = 5120, .ctx = 150, .out = 1024 },
};

pub const Conv = struct { w: u64, b: u64 }; // w [out][kernel * in], the kernel's taps outermost
pub const Unit = struct { alpha0: u64, wide: Conv, alpha1: u64, point: Conv };
pub const Block = struct { units: [3]Unit, alpha: u64, down: Conv };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(cuda.DeviceBuffer),
    bytes: usize,
    resampler: u64, // [resample_out][resample_taps]
    first: Conv, // 1 -> 64, kernel 7
    blocks: [2]Block,
    stages: [5]dec.Stage,

    pub fn deinit(w: *Weights) void {
        for (w.bufs.items) |*b| b.free();
        w.bufs.deinit(w.gpa);
        for (w.stages) |st| w.gpa.free(st.layers);
    }
};

const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    a: *const pth.Archive,
    w: *Weights,

    fn device(l: *Loader, values: []const f32) !u64 {
        const bytes = std.mem.sliceAsBytes(values);
        var b = try cuda.DeviceBuffer.alloc(l.d, @max(bytes.len, 256));
        errdefer b.free();
        try b.upload(0, bytes);
        try l.w.bufs.append(l.gpa, b);
        l.w.bytes += bytes.len;
        return b.ptr;
    }

    fn one(l: *Loader, name: []const u8, shape: []const u64) !u64 {
        const v = try l.a.readF32(l.gpa, name, shape);
        defer l.gpa.free(v);
        return l.device(v);
    }

    /// A weight-normed convolution `prefix`conv: v * (g / |v|) over each output's inputs and taps, taps outermost.
    fn conv(l: *Loader, prefix: []const u8, out: usize, in: usize, kernel: usize) !Conv {
        var nb: [200]u8 = undefined;
        const v = try l.a.readF32(l.gpa, try std.fmt.bufPrint(&nb, "{s}conv.parametrizations.weight.original1", .{prefix}), &.{ out, in, kernel });
        defer l.gpa.free(v);
        const g = try l.a.readF32(l.gpa, try std.fmt.bufPrint(&nb, "{s}conv.parametrizations.weight.original0", .{prefix}), &.{ out, 1, 1 });
        defer l.gpa.free(g);
        const w = try l.gpa.alloc(f32, v.len);
        defer l.gpa.free(w);
        for (0..out) |o| {
            const row = v[o * in * kernel ..][0 .. in * kernel];
            var ss: f64 = 0;
            for (row) |x| ss += @as(f64, x) * x;
            const scale = g[o] / @as(f32, @floatCast(@sqrt(ss)));
            for (0..in) |c| for (0..kernel) |j| {
                w[(o * kernel + j) * in + c] = row[c * kernel + j] * scale;
            };
        }
        return .{ .w = try l.device(w), .b = try l.one(try std.fmt.bufPrint(&nb, "{s}conv.bias", .{prefix}), &.{out}) };
    }
};

fn fmt(buf: []u8, comptime f: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, f, args) catch unreachable;
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, generator: []const u8) !*Weights {
    var archive = try pth.Archive.openNested(gpa, io, generator);
    defer archive.close();
    const w = try gpa.create(Weights);
    errdefer gpa.destroy(w);
    w.* = undefined;
    w.gpa = gpa;
    w.bufs = .empty;
    w.bytes = 0;
    for (&w.stages) |*st| st.layers = &.{};
    errdefer w.deinit();
    var l: Loader = .{ .gpa = gpa, .d = d, .a = &archive, .w = w };
    var nb: [160]u8 = undefined;
    const e = "generator/encoder.";
    w.resampler = try l.one("generator/to_internal.kernel", &.{ resample_out, 1, resample_taps });
    w.first = try l.conv(e ++ "0.", blocks[0].channels, 1, 7);
    for (&w.blocks, blocks, 1..) |*b, shape, index| {
        const c = shape.channels;
        for (&b.units, 0..) |*u, ui| {
            const p = e ++ "{d}.module.block.{d}.module.block.";
            u.* = .{
                .alpha0 = try l.one(fmt(&nb, p ++ "0.alpha", .{ index, ui }), &.{ 1, c, 1 }),
                .wide = try l.conv(fmt(&nb, p ++ "1.", .{ index, ui }), c, c, 7),
                .alpha1 = try l.one(fmt(&nb, p ++ "2.alpha", .{ index, ui }), &.{ 1, c, 1 }),
                .point = try l.conv(fmt(&nb, p ++ "3.", .{ index, ui }), c, c, 1),
            };
        }
        b.alpha = try l.one(fmt(&nb, e ++ "{d}.module.block.3.alpha", .{index}), &.{ 1, c, 1 });
        b.down = try l.conv(fmt(&nb, e ++ "{d}.module.block.4.", .{index}), 2 * c, c, 2 * shape.stride);
    }
    for (&w.stages, stages, 0..) |*st, shape, si| {
        const s = 4 + 2 * si;
        var count: usize = 0;
        while (archive.tensors.contains(fmt(&nb, e ++ "{d}.layers.{d}.qkv.weight", .{ s, count }))) count += 1;
        if (count == 0) return error.MissingTensor;
        st.layers = try gpa.alloc(dec.Layer, count);
        st.in_proj = try l.one(fmt(&nb, e ++ "{d}.input_proj.weight", .{s}), &.{ shape.d, shape.in });
        st.out_proj = try l.one(fmt(&nb, e ++ "{d}.output_proj.weight", .{s}), &.{ shape.out, shape.d });
        for (st.layers, 0..) |*L, li| {
            const p = e ++ "{d}.layers.{d}.";
            L.* = .{
                .n1w = try l.one(fmt(&nb, p ++ "norm1.weight", .{ s, li }), &.{shape.d}),
                .n1b = try l.one(fmt(&nb, p ++ "norm1.bias", .{ s, li }), &.{shape.d}),
                .n2w = try l.one(fmt(&nb, p ++ "norm2.weight", .{ s, li }), &.{shape.d}),
                .n2b = try l.one(fmt(&nb, p ++ "norm2.bias", .{ s, li }), &.{shape.d}),
                .qkv = try l.one(fmt(&nb, p ++ "qkv.weight", .{ s, li }), &.{ 3 * shape.d, shape.d }),
                .attn_out = try l.one(fmt(&nb, p ++ "attention_out.weight", .{ s, li }), &.{ shape.d, shape.d }),
                .ffn0 = try l.one(fmt(&nb, p ++ "ffn.0.weight", .{ s, li }), &.{ shape.ff, shape.d }),
                .ffn1 = try l.one(fmt(&nb, p ++ "ffn.3.weight", .{ s, li }), &.{ shape.d, shape.ff }),
                .attn_scale = try l.one(fmt(&nb, p ++ "attention_scale", .{ s, li }), &.{shape.d}),
                .ffn_scale = try l.one(fmt(&nb, p ++ "ffn_scale", .{ s, li }), &.{shape.d}),
            };
        }
    }
    return w;
}
