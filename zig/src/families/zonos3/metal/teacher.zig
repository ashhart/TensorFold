//! ZONOS3's SSL teacher on Metal: Wav2Vec2-BERT's 16 conformer layers over a lane's rows, fp32 over bf16 weights.
const std = @import("std");
const core = @import("core");
const Gpu = @import("gpu.zig").Gpu;
const lm = @import("kernels.zig");
const codec = @import("codec_kernels.zig");
const ek = @import("encoder_kernels.zig");

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

pub const Teacher = struct {
    gpa: std.mem.Allocator,
    g: *Gpu,
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

    /// On `g` after all else made on it; `dir` holds model.safetensors. Destroy before anything earlier on `g`.
    pub fn create(gpa: std.mem.Allocator, io: std.Io, g: *Gpu, dir: []const u8) !*Teacher {
        const t = try gpa.create(Teacher);
        errdefer gpa.destroy(t);
        t.* = undefined;
        t.gpa = gpa;
        t.g = g;
        t.bytes = 0;
        const owner = g.own(t);
        defer _ = g.own(owner);
        errdefer g.release(t);
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
        t.g.release(t);
        t.gpa.destroy(t);
    }

    fn alloc(t: *Teacher, bytes: usize) !u64 {
        return t.g.alloc(bytes);
    }

    fn device(t: *Teacher, bytes: []const u8) !u64 {
        const ptr = try t.g.upload(bytes);
        t.bytes += bytes.len;
        return ptr;
    }

    pub fn download(t: *Teacher, src: u64, out: []u8) !void {
        try t.g.get(src, out);
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

    fn lin(t: *Teacher, w: Linear, x: u64, in: usize, out: usize, y: u64, ys: usize, rows: usize) void {
        lm.linear(t.g, w.w, x, in, in, out, y, ys, rows, .{ .bias = w.b });
    }

    fn layernorm(t: *Teacher, x: u64, w: Norm, y: u64, n: usize, rows: usize) void {
        codec.layernorm(t.g, x, w.w, w.b, y, n, eps, rows);
    }

    /// x += half of ffn(norm(x)).
    fn feedForward(t: *Teacher, w: Norm, f: Ffn, rows: usize) void {
        t.layernorm(t.x, w, t.n, d, rows);
        t.lin(f.in, t.n, d, ffn, t.big, ffn, rows);
        ek.swish(t.g, t.big, rows * ffn);
        t.lin(f.out, t.big, ffn, d, t.h, d, rows);
        codec.addScaled(t.g, t.x, t.h, t.half, d, rows);
    }

    /// A lane's feature rows [rows][160] through the teacher; the device rows [rows][d] of its sixteenth layer.
    pub fn run(t: *Teacher, feats: []const f32, probe: ?Probe) !u64 {
        const rows = feats.len / feature_width;
        if (rows == 0 or rows > max_rows or feats.len % feature_width != 0) return error.BadFeatureRows;
        const g = t.g;
        try g.put(t.feats, std.mem.sliceAsBytes(feats));
        t.layernorm(t.feats, t.in_norm, t.n, feature_width, rows);
        t.lin(t.projection, t.n, feature_width, d, t.x, d, rows);
        if (probe) |p| try p.f(p.ctx, "lane.0.ssl.feature_projection", t.x, rows, d);
        for (t.layers, 0..) |L, li| {
            t.feedForward(L.ffn1_norm, L.ffn1, rows);
            // attention over the whole lane, with the relative-key term
            t.layernorm(t.x, L.attn_norm, t.n, d, rows);
            t.lin(L.q, t.n, d, d, t.qkv, 3 * d, rows);
            t.lin(L.k, t.n, d, d, t.qkv + d * 4, 3 * d, rows);
            t.lin(L.v, t.n, d, d, t.qkv + 2 * d * 4, 3 * d, rows);
            ek.attention(g, t.qkv, d, rows, L.distance, t.att);
            t.lin(L.attn_out, t.att, d, d, t.h, d, rows);
            codec.addScaled(g, t.x, t.h, 0, d, rows);
            // the convolution module
            t.layernorm(t.x, L.conv_norm, t.n, d, rows);
            t.lin(L.point1, t.n, d, 2 * d, t.big, 2 * d, rows);
            ek.glu(g, t.big, t.h, d, rows);
            ek.dwconv(g, t.h, L.depthwise, conv_taps, d, t.n, rows);
            t.layernorm(t.n, L.depthwise_norm, t.h, d, rows);
            ek.swish(g, t.h, rows * d);
            t.lin(L.point2, t.h, d, d, t.n, d, rows);
            codec.addScaled(g, t.x, t.n, 0, d, rows);
            t.feedForward(L.ffn2_norm, L.ffn2, rows);
            t.layernorm(t.x, L.final_norm, t.n, d, rows);
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
