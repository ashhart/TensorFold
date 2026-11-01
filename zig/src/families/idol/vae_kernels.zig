//! Idol's image encoder pipelines (idol_vae.metal, fp32): convolutions, GroupNorm, attention and the condition rows.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
pub const Ref = @import("../../core/qwen_vl/weights.zig").Ref;
const cfg = @import("vae_config.zig");

/// Each pipeline by its function.
pub const names = [_][]const u8{ "iv_conv_in", "iv_conv", "iv_wino", "iv_norm", "iv_group_part", "iv_group_finish", "iv_attention", "iv_tokens" };

/// GroupNorm's groups and epsilon, the pixels a statistics part sums, and the mid block's attention width.
pub const groups = cfg.groups;
pub const eps = cfg.eps;
pub const chunk = 256;
pub const width = cfg.width;

pub const Kernels = struct {
    lib: mtl.Library,
    p: [names.len]mtl.Pipeline,

    pub fn load(device: mtl.Device) !Kernels {
        return loadSource(device, sources.idol_vae);
    }

    /// The pipelines of `text` (the served source, or a contract control's copy of it).
    pub fn loadSource(device: mtl.Device, text: []const u8) !Kernels {
        var k: Kernels = undefined;
        k.lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        errdefer k.lib.deinit();
        var made: usize = 0;
        errdefer for (k.p[0..made]) |p| p.deinit();
        for (names, 0..) |name, i| {
            k.p[i] = try mtl.Pipeline.init(device, k.lib, name, false);
            made += 1;
        }
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        for (k.p) |p| p.deinit();
        k.lib.deinit();
    }

    pub fn get(k: *const Kernels, comptime name: []const u8) mtl.Pipeline {
        inline for (names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return k.p[i];
        @compileError("no pipeline " ++ name);
    }
};

/// One kernel's buffers in order, then its parameters at the next index.
pub fn bind(e: mtl.ComputeEncoder, p: mtl.Pipeline, refs: []const Ref, params: anytype) void {
    e.setPipeline(p);
    for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(params, refs.len);
}

/// iv_conv_in: y [rows * cols, 128] = conv3x3(2 x - 1) + b of the picture x [rows * cols, 3], w [27, 128].
pub fn convIn(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, w: Ref, b: Ref, y: Ref, rows: usize, cols: usize) void {
    bind(e, p, &.{ x, w, b, y }, [2]u32{ @intCast(rows), @intCast(cols) });
    e.dispatchThreads(mtl.Size.of(128, rows * cols, 1), mtl.Size.of(128, 2, 1));
}

/// A convolution's size: the output picture, its input picture, channels, the weights' row length and taps (1 or 9).
pub const ConvShape = struct { rows: usize, cols: usize, in_rows: usize, in_cols: usize, cin: usize, cout: usize, ldw: usize, taps: usize = 1 };

fn convParams(s: ConvShape, add: bool) extern struct { rows: u32, cols: u32, in_rows: u32, in_cols: u32, cin: u32, cout: u32, ldw: u32, taps: u32, add: u32 } {
    return .{ .rows = @intCast(s.rows), .cols = @intCast(s.cols), .in_rows = @intCast(s.in_rows), .in_cols = @intCast(s.in_cols), .cin = @intCast(s.cin), .cout = @intCast(s.cout), .ldw = @intCast(s.ldw), .taps = @intCast(s.taps), .add = @intFromBool(add) };
}

/// iv_conv: y = conv(x) + b (+ r when given), a 1x1 or a stride-2 3x3; cin % 32 = 0, ldw % 64 = 0.
pub fn conv(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, w: Ref, b: Ref, r: ?Ref, y: Ref, s: ConvShape) void {
    std.debug.assert(s.cin % 32 == 0 and s.ldw % 64 == 0 and (s.taps == 1 or s.taps == 9));
    bind(e, p, &.{ x, w, b, r orelse y, y }, convParams(s, r != null));
    e.dispatchGroups(mtl.Size.of(s.ldw / 64, (s.rows * s.cols + 63) / 64, 1), mtl.Size.of(128, 1, 1));
}

/// iv_wino: y = conv3x3(x) + b (+ r when given) by Winograd F(2x2, 3x3), same size; cin % 16 = 0, ldw % 64 = 0.
pub fn winograd(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, u: Ref, b: Ref, r: ?Ref, y: Ref, s: ConvShape) void {
    std.debug.assert(s.cin % 16 == 0 and s.ldw % 64 == 0);
    bind(e, p, &.{ x, u, b, r orelse y, y }, convParams(s, r != null));
    e.dispatchGroups(mtl.Size.of(s.ldw / 64, (s.rows + 3) / 4, (s.cols + 15) / 16), mtl.Size.of(512, 1, 1));
}

/// The Winograd F(2x2, 3x3) form of one 3x3 kernel g: G g G^T in float64, each of its 16 values rounded once.
pub fn transform(g: *const [9]f32) [16]f32 {
    const G = [4][3]f64{ .{ 1, 0, 0 }, .{ 0.5, 0.5, 0.5 }, .{ 0.5, -0.5, 0.5 }, .{ 0, 0, 1 } };
    var out: [16]f32 = undefined;
    for (0..4) |i| for (0..4) |j| {
        var v: f64 = 0;
        for (0..3) |a| for (0..3) |b| {
            v += G[i][a] * @as(f64, g[a * 3 + b]) * G[j][b];
        };
        out[i * 4 + j] = @floatCast(v);
    };
    return out;
}

/// iv_norm: y = GroupNorm(x) by stats (32 means, then 32 rstd) with g and beta, then SiLU when `silu`.
pub fn normed(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, stats: Ref, g: Ref, beta: Ref, y: Ref, n: usize, channels: usize, silu: bool) void {
    const params = extern struct { pixels: u32, channels: u32, silu: u32 }{ .pixels = @intCast(n), .channels = @intCast(channels), .silu = @intFromBool(silu) };
    bind(e, p, &.{ x, stats, g, beta, y }, params);
    e.dispatchThreads(mtl.Size.of(channels, n, 1), mtl.Size.of(32, 8, 1));
}

/// GroupNorm's parts over `n` pixels: one sum a chunk of `chunk` pixels and a group.
pub fn chunks(n: usize) usize {
    return (n + chunk - 1) / chunk;
}

fn groupParams(n: usize, channels: usize, centered: bool) extern struct { pixels: u32, channels: u32, chunk: u32, chunks: u32, centered: u32, eps: f32 } {
    return .{ .pixels = @intCast(n), .channels = @intCast(channels), .chunk = chunk, .chunks = @intCast(chunks(n)), .centered = @intFromBool(centered), .eps = eps };
}

/// iv_group_part: part [32, chunks] = each chunk's sum of a group's values, or (centered) of (x - mean)^2.
pub fn groupPart(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, stats: Ref, part: Ref, n: usize, channels: usize, centered: bool) void {
    std.debug.assert(channels % groups == 0 and channels <= 512);
    bind(e, p, &.{ x, stats, part }, groupParams(n, channels, centered));
    e.dispatchGroups(mtl.Size.of(chunks(n), 1, 1), mtl.Size.of(channels, 1, 1));
}

/// iv_group_finish: stats' 32 means from part, or (centered) its 32 rstd.
pub fn groupFinish(e: mtl.ComputeEncoder, p: mtl.Pipeline, part: Ref, stats: Ref, n: usize, channels: usize, centered: bool) void {
    bind(e, p, &.{ part, stats }, groupParams(n, channels, centered));
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32 * groups, 1, 1));
}

/// iv_attention: o [rows, 512] = softmax(q k^T / sqrt(512)) v of qkv rows [rows, 1536], both padded to 32 rows.
pub fn attend(e: mtl.ComputeEncoder, p: mtl.Pipeline, qkv: Ref, o: Ref, rows: usize) void {
    const params = extern struct { rows: u32, scale: f32 }{ .rows = @intCast(rows), .scale = 1.0 / @sqrt(@as(f32, width)) };
    bind(e, p, &.{ qkv, o }, params);
    e.dispatchGroups(mtl.Size.of((rows + 7) / 8, 1, 1), mtl.Size.of(128, 1, 1));
}

/// iv_tokens: out [pad + rows * cols / 4, 128] from moments [rows, cols, 64], noise [rows, cols, 32], m and s [128].
pub fn tokens(e: mtl.ComputeEncoder, p: mtl.Pipeline, moments: Ref, noise: Ref, m: Ref, s: Ref, out: Ref, rows: usize, cols: usize, pad: usize) void {
    bind(e, p, &.{ moments, noise, m, s, out }, [3]u32{ @intCast(rows), @intCast(cols), @intCast(pad) });
    e.dispatchThreads(mtl.Size.of(128, pad + rows * cols / 4, 1), mtl.Size.of(128, 2, 1));
}

test "every image encoder pipeline compiles and loads on this GPU" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var k = try Kernels.load(device);
    k.deinit();
}

test "the Winograd transform of a delta kernel is G's outer products" {
    var g: [9]f32 = @splat(0);
    g[4] = 1;
    const u = transform(&g);
    const col = [4]f32{ 0, 0.5, -0.5, 0 };
    for (0..4) |i| for (0..4) |j| try std.testing.expectEqual(col[i] * col[j], u[i * 4 + j]);
}
