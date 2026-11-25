//! ZONOS3's SSL teacher on one GPU: Wav2Vec2-BERT's 16 conformer layers over a lane's rows, fp32 over bf16 weights.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lm = @import("kernels.zig");
const codec = @import("codec_kernels.zig");

const Buf = cuda.DeviceBuffer;

pub const layers = 16;
pub const d = 1024;
pub const ffn = 4096;
pub const feature_width = 160;
pub const heads = 16;
pub const conv_taps = 31;
pub const distances = 73; // 64 to the left, the row itself, 8 to the right
pub const max_rows = 1000; // a 20 s lane is 999 rows
const eps: f32 = 1e-5;

/// A capture under the reference's dump names: device fp32 rows [rows, width].
pub const Probe = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, name: []const u8, ptr: u64, rows: usize, width: usize) anyerror!void,
};

const Norm = struct { w: u64, b: u64 }; // fp32
const Linear = struct { w: u64, b: u64 }; // bf16, b 0 without a bias
const Ffn = struct { in: Linear, out: Linear };
const Layer = struct {
    ffn1_norm: Norm,
    ffn1: Ffn,
    attn_norm: Norm,
    q: Linear,
    k: Linear,
    v: Linear,
    attn_out: Linear,
    distance: u64, // fp32 [73][64]
    conv_norm: Norm,
    point1: Linear,
    depthwise: u64, // fp32 [d][31]
    depthwise_norm: Norm,
    point2: Linear,
    ffn2_norm: Norm,
    ffn2: Ffn,
    final_norm: Norm,
};

const names = .{ "zt_swish", "zt_glu", "zt_dwconv", "zt_attention" };
const F = enum(u8) { swish, glu, dwconv, attention };

pub const Teacher = struct {
    gpa: std.mem.Allocator,
    dev: *const cuda.Driver,
    stream: cuda.Stream,
    lm: lm.Kernels,
    codec: codec.Kernels,
    mod: cuda.Module,
    fns: [names.len]cuda.Function,
    bufs: std.ArrayList(Buf),
    bytes: usize,
    in_norm: Norm,
    projection: Linear,
    layers: [layers]Layer,
    half: u64, // fp32 [d] of 0.5
    feats: u64, // [rows][160]
    x: u64, // [rows][d]
    h: u64,
    n: u64,
    att: u64,
    big: u64, // [rows][ffn]
    qkv: u64, // [rows][3 d]

    /// The teacher on a context the caller made current; `dir` holds its model.safetensors.
    pub fn create(gpa: std.mem.Allocator, io: std.Io, dev: *const cuda.Driver, stream: cuda.Stream, dir: []const u8) !*Teacher {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const t = try gpa.create(Teacher);
        errdefer gpa.destroy(t);
        t.* = undefined;
        t.gpa = gpa;
        t.dev = dev;
        t.stream = stream;
        t.bufs = .empty;
        t.bytes = 0;
        t.lm = try lm.Kernels.load(dev);
        errdefer t.lm.unload();
        t.codec = try codec.Kernels.load(dev);
        errdefer t.codec.unload();
        t.mod = try cuda.Module.load(dev, cuda.kernels.zonos3_teacher);
        errdefer t.mod.unload();
        inline for (names, 0..) |name, j| t.fns[j] = try t.mod.function(name);
        errdefer t.freeBuffers();
        try t.loadWeights(io, dir);
        const half: [d]f32 = @splat(0.5);
        t.half = try t.device(std.mem.sliceAsBytes(&half));
        t.feats = try t.alloc(max_rows * feature_width * 4);
        t.x = try t.alloc(max_rows * d * 4);
        t.h = try t.alloc(max_rows * d * 4);
        t.n = try t.alloc(max_rows * d * 4);
        t.att = try t.alloc(max_rows * d * 4);
        t.big = try t.alloc(max_rows * ffn * 4);
        t.qkv = try t.alloc(max_rows * 3 * d * 4);
        return t;
    }

    pub fn destroy(t: *Teacher) void {
        t.stream.synchronize() catch {};
        t.freeBuffers();
        t.mod.unload();
        t.codec.unload();
        t.lm.unload();
        t.gpa.destroy(t);
    }

    fn freeBuffers(t: *Teacher) void {
        for (t.bufs.items) |*b| b.free();
        t.bufs.deinit(t.gpa);
    }

    fn alloc(t: *Teacher, bytes: usize) !u64 {
        var b = try Buf.alloc(t.dev, @max(bytes, 256));
        errdefer b.free();
        try t.bufs.append(t.gpa, b);
        return b.ptr;
    }

    fn device(t: *Teacher, bytes: []const u8) !u64 {
        const ptr = try t.alloc(bytes.len);
        try t.buf(ptr).upload(0, bytes);
        t.bytes += bytes.len;
        return ptr;
    }

    fn buf(t: *Teacher, ptr: u64) Buf {
        return .{ .d = t.dev, .ptr = ptr, .len = std.math.maxInt(usize) / 2, .borrowed = true };
    }

    pub fn download(t: *Teacher, src: u64, out: []u8) !void {
        try t.buf(src).downloadAsync(0, out, t.stream.handle);
        try t.stream.synchronize();
    }

    fn loadWeights(t: *Teacher, io: std.Io, dir: []const u8) !void {
        const gpa = t.gpa;
        const path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors" });
        defer gpa.free(path);
        var file = try core.safetensors.File.open(gpa, io, path);
        defer file.close(io);
        var l: Loader = .{ .t = t, .file = &file };
        t.in_norm = try l.norm("feature_projection.layer_norm", .{}, feature_width);
        t.projection = try l.linear("feature_projection.projection", .{}, d, feature_width, true);
        for (&t.layers, 0..) |*L, i| {
            const p = "encoder.layers.{d}.";
            L.* = .{
                .ffn1_norm = try l.norm(p ++ "ffn1_layer_norm", .{i}, d),
                .ffn1 = .{ .in = try l.linear(p ++ "ffn1.intermediate_dense", .{i}, ffn, d, true), .out = try l.linear(p ++ "ffn1.output_dense", .{i}, d, ffn, true) },
                .attn_norm = try l.norm(p ++ "self_attn_layer_norm", .{i}, d),
                .q = try l.linear(p ++ "self_attn.linear_q", .{i}, d, d, true),
                .k = try l.linear(p ++ "self_attn.linear_k", .{i}, d, d, true),
                .v = try l.linear(p ++ "self_attn.linear_v", .{i}, d, d, true),
                .attn_out = try l.linear(p ++ "self_attn.linear_out", .{i}, d, d, true),
                .distance = try l.floats(p ++ "self_attn.distance_embedding.weight", .{i}, &.{ distances, d / heads }),
                .conv_norm = try l.norm(p ++ "conv_module.layer_norm", .{i}, d),
                .point1 = try l.conv1(p ++ "conv_module.pointwise_conv1", .{i}, 2 * d, d),
                .depthwise = try l.floats(p ++ "conv_module.depthwise_conv.weight", .{i}, &.{ d, 1, conv_taps }),
                .depthwise_norm = try l.norm(p ++ "conv_module.depthwise_layer_norm", .{i}, d),
                .point2 = try l.conv1(p ++ "conv_module.pointwise_conv2", .{i}, d, d),
                .ffn2_norm = try l.norm(p ++ "ffn2_layer_norm", .{i}, d),
                .ffn2 = .{ .in = try l.linear(p ++ "ffn2.intermediate_dense", .{i}, ffn, d, true), .out = try l.linear(p ++ "ffn2.output_dense", .{i}, d, ffn, true) },
                .final_norm = try l.norm(p ++ "final_layer_norm", .{i}, d),
            };
        }
    }

    const Loader = struct {
        t: *Teacher,
        file: *core.safetensors.File,
        nb: [160]u8 = undefined,

        fn bytesOf(l: *Loader, comptime f: []const u8, args: anytype, shape: []const usize) ![]const u8 {
            const name = try std.fmt.bufPrint(&l.nb, f, args);
            const tensor = l.file.get(name) orelse return error.MissingTensor;
            if (!tensor.is(.bf16, shape)) return error.ShapeMismatch;
            return tensor.bytes;
        }

        /// A bf16 tensor as stored.
        fn raw(l: *Loader, comptime f: []const u8, args: anytype, shape: []const usize) !u64 {
            return l.t.device(try l.bytesOf(f, args, shape));
        }

        /// A small bf16 tensor widened to fp32.
        fn floats(l: *Loader, comptime f: []const u8, args: anytype, shape: []const usize) !u64 {
            const bytes = try l.bytesOf(f, args, shape);
            const out = try l.t.gpa.alloc(f32, bytes.len / 2);
            defer l.t.gpa.free(out);
            for (out, 0..) |*o, i| o.* = @bitCast(@as(u32, std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)) << 16);
            return l.t.device(std.mem.sliceAsBytes(out));
        }

        fn norm(l: *Loader, comptime f: []const u8, args: anytype, n: usize) !Norm {
            return .{ .w = try l.floats(f ++ ".weight", args, &.{n}), .b = try l.floats(f ++ ".bias", args, &.{n}) };
        }

        fn linear(l: *Loader, comptime f: []const u8, args: anytype, out: usize, in: usize, bias: bool) !Linear {
            return .{ .w = try l.raw(f ++ ".weight", args, &.{ out, in }), .b = if (bias) try l.raw(f ++ ".bias", args, &.{out}) else 0 };
        }

        /// A pointwise convolution's weight [out][in][1], used as a linear without a bias.
        fn conv1(l: *Loader, comptime f: []const u8, args: anytype, out: usize, in: usize) !Linear {
            return .{ .w = try l.raw(f ++ ".weight", args, &.{ out, in, 1 }), .b = 0 };
        }
    };

    fn go(t: *Teacher, which: F, gx: usize, gy: usize, block: usize, a: *cuda.Args) !void {
        try cuda.launch.launch(t.fns[@backingInt(which)], .{ .grid = .{ .x = @intCast(gx), .y = @intCast(gy), .z = 1 }, .block = .{ .x = @intCast(block) } }, t.stream, a);
    }

    fn lin(t: *Teacher, w: Linear, x: u64, in: usize, out: usize, y: u64, ys: usize, rows: usize) !void {
        try t.lm.linear(t.stream, w.w, x, in, in, out, y, ys, rows, .{ .bias = w.b });
    }

    fn layernorm(t: *Teacher, x: u64, w: Norm, y: u64, n: usize, rows: usize) !void {
        try t.codec.layernorm(t.stream, x, w.w, w.b, y, n, eps, rows);
    }

    fn swish(t: *Teacher, x: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(n)));
        try t.go(.swish, (n + 255) / 256, 1, 256, &a);
    }

    /// x += half of ffn(norm(x)).
    fn feedForward(t: *Teacher, w: Norm, f: Ffn, rows: usize) !void {
        try t.layernorm(t.x, w, t.n, d, rows);
        try t.lin(f.in, t.n, d, ffn, t.big, ffn, rows);
        try t.swish(t.big, rows * ffn);
        try t.lin(f.out, t.big, ffn, d, t.h, d, rows);
        try t.codec.addScaled(t.stream, t.x, t.h, t.half, d, rows);
    }

    /// A lane's feature rows [rows][160] through the teacher; the device rows [rows][d] of its sixteenth layer.
    pub fn run(t: *Teacher, feats: []const f32, probe: ?Probe) !u64 {
        const rows = feats.len / feature_width;
        if (rows == 0 or rows > max_rows or feats.len % feature_width != 0) return error.BadFeatureRows;
        const s = t.stream;
        try t.buf(t.feats).uploadAsync(0, std.mem.sliceAsBytes(feats), s.handle);
        try t.layernorm(t.feats, t.in_norm, t.n, feature_width, rows);
        try t.lin(t.projection, t.n, feature_width, d, t.x, d, rows);
        if (probe) |p| try p.f(p.ctx, "lane.0.ssl.feature_projection", t.x, rows, d);
        for (t.layers, 0..) |L, li| {
            try t.feedForward(L.ffn1_norm, L.ffn1, rows);
            // attention over the whole lane, with the relative-key term
            try t.layernorm(t.x, L.attn_norm, t.n, d, rows);
            try t.lin(L.q, t.n, d, d, t.qkv, 3 * d, rows);
            try t.lin(L.k, t.n, d, d, t.qkv + d * 4, 3 * d, rows);
            try t.lin(L.v, t.n, d, d, t.qkv + 2 * d * 4, 3 * d, rows);
            {
                var a: cuda.Args = .{};
                a.add(t.qkv);
                a.add(@as(i32, d));
                a.add(@as(i32, @intCast(rows)));
                a.add(L.distance);
                a.add(t.att);
                try t.go(.attention, rows, heads, 128, &a);
            }
            try t.lin(L.attn_out, t.att, d, d, t.h, d, rows);
            try t.codec.addScaled(s, t.x, t.h, 0, d, rows);
            // the convolution module
            try t.layernorm(t.x, L.conv_norm, t.n, d, rows);
            try t.lin(L.point1, t.n, d, 2 * d, t.big, 2 * d, rows);
            {
                var a: cuda.Args = .{};
                a.add(t.big);
                a.add(t.h);
                a.add(@as(i32, d));
                try t.go(.glu, (d + 255) / 256, rows, 256, &a);
            }
            {
                var a: cuda.Args = .{};
                a.add(t.h);
                a.add(L.depthwise);
                a.add(@as(i32, conv_taps));
                a.add(@as(i32, d));
                a.add(t.n);
                try t.go(.dwconv, (d + 255) / 256, rows, 256, &a);
            }
            try t.layernorm(t.n, L.depthwise_norm, t.h, d, rows);
            try t.swish(t.h, rows * d);
            try t.lin(L.point2, t.h, d, d, t.n, d, rows);
            try t.codec.addScaled(s, t.x, t.n, 0, d, rows);
            try t.feedForward(L.ffn2_norm, L.ffn2, rows);
            try t.layernorm(t.x, L.final_norm, t.n, d, rows);
            std.mem.swap(u64, &t.x, &t.n);
            if (probe) |p| {
                var nb: [40]u8 = undefined;
                try p.f(p.ctx, try std.fmt.bufPrint(&nb, "lane.0.ssl.layer.{d}", .{li}), t.x, rows, d);
            }
        }
        if (probe) |p| try p.f(p.ctx, "lane.0.ssl", t.x, rows, d);
        return t.x;
    }
};
