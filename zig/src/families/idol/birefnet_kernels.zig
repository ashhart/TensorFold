//! BiRefNet's pipelines (idol_birefnet.metal, fp32) and their launches.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
pub const Ref = @import("birefnet_weights.zig").Ref;

/// Each pipeline by its function.
pub const names = [_][]const u8{ "br_conv", "br_norm", "br_window", "br_resize", "br_tiles", "br_conv3", "br_deform", "br_pool", "br_gate", "br_head" };

pub const Kernels = struct {
    lib: mtl.Library,
    p: [names.len]mtl.Pipeline,

    pub fn load(device: mtl.Device) !Kernels {
        return loadSource(device, sources.idol_birefnet);
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

pub const flag_bn: u32 = 1;
pub const flag_relu: u32 = 2;
pub const flag_gelu: u32 = 4;
pub const flag_add: u32 = 8;
pub const flag_map: u32 = 16;

/// br_conv's launch: picture rows x cols, channels a tap (% 32), ldx, outputs, ldw (% 64), side, output, flags.
pub const ConvShape = struct {
    rows: usize,
    cols: usize = 1,
    cin: usize,
    ldx: usize,
    cout: usize,
    ldw: usize,
    k: usize = 1,
    ldy: usize,
    yoff: usize = 0,
    flags: u32 = 0,
    row0: usize = 0,
};

pub const ConvParams = extern struct { rows: u32, cols: u32, cin: u32, ldx: u32, cout: u32, ldw: u32, k: u32, ldy: u32, yoff: u32, flags: u32, row0: u32 };

pub fn convParams(s: ConvShape) ConvParams {
    return .{ .rows = @intCast(s.rows), .cols = @intCast(s.cols), .cin = @intCast(s.cin), .ldx = @intCast(s.ldx), .cout = @intCast(s.cout), .ldw = @intCast(s.ldw), .k = @intCast(s.k), .ldy = @intCast(s.ldy), .yoff = @intCast(s.yoff), .flags = s.flags, .row0 = @intCast(s.row0) };
}

/// br_conv: y = conv(x) + b and its epilogue; post, r and map are bound where the flags need them (else x).
pub fn conv(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, w: Ref, b: Ref, post: ?Ref, r: ?Ref, map: ?Ref, y: Ref, s: ConvShape) void {
    std.debug.assert(s.cin % 32 == 0 and s.ldw % 64 == 0 and s.ldx % 4 == 0 and s.cout <= s.ldw);
    bind(e, p, &.{ x, w, b, post orelse x, r orelse y, map orelse x, y }, convParams(s));
    e.dispatchGroups(mtl.Size.of(s.ldw / 64, (s.rows * s.cols + 63) / 64, 1), mtl.Size.of(128, 1, 1));
}

/// br_norm's modes: a row as it is, the row a map names (negative: zeros), PatchMerging's 2 x 2 patch.
pub const NormMode = enum(u32) { rows = 0, map = 1, merge = 2 };
pub const NormParams = extern struct { rows: u32, width: u32, mode: u32, src_w: u32, ldy: u32, yoff: u32, eps: f32 };

/// br_norm: y [rows, ldy] from yoff = LayerNorm of each row (`width` wide) with g and beta.
pub fn norm(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, map: ?Ref, g: Ref, beta: Ref, y: Ref, rows: usize, width: usize, mode: NormMode, src_w: usize, ldy: usize, yoff: usize, eps: f32) void {
    bind(e, p, &.{ x, map orelse x, g, beta, y }, NormParams{ .rows = @intCast(rows), .width = @intCast(width), .mode = @backingInt(mode), .src_w = @intCast(src_w), .ldy = @intCast(ldy), .yoff = @intCast(yoff), .eps = eps });
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

pub const WindowParams = extern struct { width: u32, heads: u32, rows: u32, cols: u32, shift: u32, scale: f32 };

/// br_window: o [windows * 144, width] = each window's attention of qkv [windows * 144, 3 * width], grid rows x cols.
pub fn window(e: mtl.ComputeEncoder, p: mtl.Pipeline, qkv: Ref, table: Ref, o: Ref, width: usize, heads: usize, rows: usize, cols: usize, shift: usize) void {
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(width / heads), -0.5));
    bind(e, p, &.{ qkv, table, o }, WindowParams{ .width = @intCast(width), .heads = @intCast(heads), .rows = @intCast(rows), .cols = @intCast(cols), .shift = @intCast(shift), .scale = scale });
    e.dispatchGroups(mtl.Size.of(3, heads, (rows / 12) * (cols / 12)), mtl.Size.of(192, 1, 1));
}

pub const Grid = struct { h: usize, w: usize };
pub const ResizeParams = extern struct { in_h: u32, in_w: u32, out_h: u32, out_w: u32, channels: u32, ldx: u32, xoff: u32, ldy: u32, yoff: u32 };

/// br_resize: bilinear with align_corners of `channels` columns of x (row length ldx, from xoff) into y's (ldy, yoff).
pub fn resize(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, y: Ref, in: Grid, out: Grid, channels: usize, ldx: usize, xoff: usize, ldy: usize, yoff: usize) void {
    bind(e, p, &.{ x, y }, ResizeParams{ .in_h = @intCast(in.h), .in_w = @intCast(in.w), .out_h = @intCast(out.h), .out_w = @intCast(out.w), .channels = @intCast(channels), .ldx = @intCast(ldx), .xoff = @intCast(xoff), .ldy = @intCast(ldy), .yoff = @intCast(yoff) });
    e.dispatchThreads(mtl.Size.of(channels, out.h * out.w, 1), mtl.Size.of(@min(channels, 64), @as(usize, 256) / @min(channels, 64), 1));
}

pub const TilesParams = extern struct { out_h: u32, out_w: u32, src_w: u32, channels: u32, a: u32, b: u32, mode: u32, cpad: u32 };

/// br_tiles: dst [out.h * out.w, cpad] from src [., src_w, channels], a x b patches (mode 0) or grid tiles (mode 1).
pub fn tiles(e: mtl.ComputeEncoder, p: mtl.Pipeline, src: Ref, dst: Ref, out: Grid, src_w: usize, channels: usize, a: usize, b: usize, grid: bool, cpad: usize) void {
    bind(e, p, &.{ src, dst }, TilesParams{ .out_h = @intCast(out.h), .out_w = @intCast(out.w), .src_w = @intCast(src_w), .channels = @intCast(channels), .a = @intCast(a), .b = @intCast(b), .mode = @intFromBool(grid), .cpad = @intCast(cpad) });
    e.dispatchThreads(mtl.Size.of(cpad, out.h * out.w, 1), mtl.Size.of(@min(cpad, 64), @as(usize, 256) / @min(cpad, 64), 1));
}

/// br_conv3: y [pixels, cout] = conv3x3(x) + b of a 3-channel picture, w [27, cout].
pub fn conv3(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, w: Ref, b: Ref, y: Ref, g: Grid, cout: usize) void {
    bind(e, p, &.{ x, w, b, y }, [3]u32{ @intCast(g.h), @intCast(g.w), @intCast(cout) });
    e.dispatchThreads(mtl.Size.of(cout, g.h * g.w, 1), mtl.Size.of(64, 4, 1));
}

pub const DeformParams = extern struct { rows: u32, cols: u32, k: u32, first: u32, count: u32, ldo: u32 };

/// br_deform: the columns [count, k * k * 64] of pixels first .. first + count, offsets and modulators in offs.
pub fn deform(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, offs: Ref, col: Ref, g: Grid, k: usize, first: usize, count: usize, ldo: usize) void {
    bind(e, p, &.{ x, offs, col }, DeformParams{ .rows = @intCast(g.h), .cols = @intCast(g.w), .k = @intCast(k), .first = @intCast(first), .count = @intCast(count), .ldo = @intCast(ldo) });
    const taps = k * k;
    const tx = @min(taps, 64);
    e.dispatchThreads(mtl.Size.of(taps, count, 1), mtl.Size.of(tx, @as(usize, 256) / tx, 1));
}

/// br_pool: mean [channels] over x's pixels.
pub fn pool(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, mean: Ref, pixels: usize, channels: usize) void {
    bind(e, p, &.{ x, mean }, [2]u32{ @intCast(pixels), @intCast(channels) });
    e.dispatchGroups(mtl.Size.of(channels, 1, 1), mtl.Size.of(256, 1, 1));
}

/// br_gate: x [pixels, channels] *= sigmoid(g [pixels, 16] . w + b).
pub fn gate(e: mtl.ComputeEncoder, p: mtl.Pipeline, g: Ref, w: Ref, b: Ref, x: Ref, pixels: usize, channels: usize) void {
    bind(e, p, &.{ g, w, b, x }, [2]u32{ @intCast(pixels), @intCast(channels) });
    e.dispatchThreads(mtl.Size.of(channels, pixels, 1), mtl.Size.of(64, 4, 1));
}

/// br_head: logits [out] = the head's 1x1 over [a resized from `in` | q].
pub fn head(e: mtl.ComputeEncoder, p: mtl.Pipeline, a: Ref, q: Ref, w: Ref, b: Ref, y: Ref, in: Grid, out: Grid, cp: usize, cq: usize) void {
    bind(e, p, &.{ a, q, w, b, y }, [6]u32{ @intCast(in.h), @intCast(in.w), @intCast(out.h), @intCast(out.w), @intCast(cp), @intCast(cq) });
    e.dispatchThreads(mtl.Size.of(out.h * out.w, 1, 1), mtl.Size.of(256, 1, 1));
}

test "every background remover pipeline compiles and loads on this GPU" {
    const pool_ = mtl.objc.Pool.push();
    defer pool_.pop();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var k = try Kernels.load(device);
    k.deinit();
}
