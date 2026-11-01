//! BiRefNet's CUDA kernels (idol_birefnet.cu, fp32) and their launches, as birefnet_kernels.zig's on Metal.
const std = @import("std");
const cuda = @import("cuda");

pub const flag_bn: u32 = 1;
pub const flag_relu: u32 = 2;
pub const flag_gelu: u32 = 4;
pub const flag_add: u32 = 8;
pub const flag_map: u32 = 16;

/// The kernels by function, each `tf_br_` and its name.
pub const names = [_][]const u8{ "conv", "norm", "window", "resize", "tiles", "conv3", "deform", "pool", "gate", "head" };

pub const Kernels = struct {
    mod: cuda.Module,
    conv: cuda.Function,
    norm: cuda.Function,
    window: cuda.Function,
    resize: cuda.Function,
    tiles: cuda.Function,
    conv3: cuda.Function,
    deform: cuda.Function,
    pool: cuda.Function,
    gate: cuda.Function,
    head: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.idol_birefnet);
        errdefer mod.unload();
        var k: Kernels = undefined;
        k.mod = mod;
        inline for (names) |n| @field(k, n) = try mod.function("tf_br_" ++ n);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }
};

fn int(x: usize) c_int {
    return @intCast(x);
}

fn ceil(n: usize, d: usize) u32 {
    return @intCast((n + d - 1) / d);
}

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, block: cuda.Dim3, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = block }, s, a);
}

fn addAll(a: *cuda.Args, ptrs: []const u64) void {
    for (ptrs) |p| a.add(p);
}

fn addInts(a: *cuda.Args, vals: []const usize) void {
    for (vals) |v| a.add(int(v));
}

/// A 2-D launch: `cols` (a channel, a tap) by `rows` (a pixel) on grid x, as Metal's (min(cols, 64), 256 / that).
pub fn grid2(cols: usize, rows: usize) struct { cuda.Dim3, cuda.Dim3 } {
    const tx: usize = @min(cols, 64);
    const ty = 256 / tx;
    return .{ .{ .x = ceil(rows, ty), .y = ceil(cols, tx) }, .{ .x = @intCast(tx), .y = @intCast(ty) } };
}

/// tf_br_conv's launch: picture rows x cols, channels a tap (% 32), ldx, outputs, ldw (% 64), side, output, flags.
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

/// The conv's grid: 64-pixel tiles on x, 64-output tiles on y.
pub fn convGrid(s: ConvShape) cuda.Dim3 {
    return .{ .x = ceil(s.rows * s.cols, 64), .y = @intCast(s.ldw / 64) };
}

/// tf_br_conv: y = conv(x) + b and its epilogue; post, r and map are read where the flags need them (else 0).
pub fn conv(k: *const Kernels, st: cuda.Stream, x: u64, w: u64, b: u64, post: u64, r: u64, map: u64, y: u64, s: ConvShape) !void {
    std.debug.assert(s.cin % 32 == 0 and s.ldw % 64 == 0 and s.ldx % 4 == 0 and s.cout <= s.ldw);
    var a: cuda.Args = .{};
    addAll(&a, &.{ x, w, b, post, r, map, y });
    addInts(&a, &.{ s.rows, s.cols, s.cin, s.ldx, s.cout, s.ldw, s.k, s.ldy, s.yoff });
    a.add(@as(c_int, @intCast(s.flags)));
    a.add(int(s.row0));
    try go(k.conv, st, convGrid(s), .{ .x = 256 }, &a);
}

/// tf_br_norm's modes: a row as it is, the row a map names (negative: zeros), PatchMerging's 2 x 2 patch.
pub const NormMode = enum(c_int) { rows = 0, map = 1, merge = 2 };

/// tf_br_norm: y [rows, ldy] from yoff = LayerNorm of each row (`width` wide) with g and beta.
pub fn norm(k: *const Kernels, st: cuda.Stream, x: u64, map: u64, g: u64, beta: u64, y: u64, rows: usize, width: usize, mode: NormMode, src_w: usize, ldy: usize, yoff: usize, eps: f32) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ x, map, g, beta, y });
    a.add(int(width));
    a.add(@backingInt(mode));
    addInts(&a, &.{ src_w, ldy, yoff });
    a.add(eps);
    try go(k.norm, st, .{ .x = @intCast(rows) }, .{ .x = 256 }, &a);
}

/// Queries a window_ block holds (tf_br_window's WQ), and its threads.
pub const window_queries = 24;
pub const window_threads = 192;

/// tf_br_window: o [windows * 144, width] = each window's attention of qkv [windows * 144, 3 * width].
pub fn window(k: *const Kernels, st: cuda.Stream, qkv: u64, table: u64, o: u64, width: usize, heads: usize, rows: usize, cols: usize, shift: usize) !void {
    std.debug.assert(width / heads == 32 and rows % 12 == 0 and cols % 12 == 0);
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(width / heads), -0.5));
    var a: cuda.Args = .{};
    addAll(&a, &.{ qkv, table, o });
    addInts(&a, &.{ width, heads, rows, cols, shift });
    a.add(scale);
    const grid: cuda.Dim3 = .{ .x = 144 / window_queries, .y = @intCast(heads), .z = @intCast((rows / 12) * (cols / 12)) };
    try go(k.window, st, grid, .{ .x = window_threads }, &a);
}

pub const Grid = struct { h: usize, w: usize };

/// tf_br_resize: align_corners bilinear of `channels` columns of x (ldx, xoff) into y's (ldy, yoff).
pub fn resize(k: *const Kernels, st: cuda.Stream, x: u64, y: u64, in: Grid, out: Grid, channels: usize, ldx: usize, xoff: usize, ldy: usize, yoff: usize) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ x, y });
    addInts(&a, &.{ in.h, in.w, out.h, out.w, channels, ldx, xoff, ldy, yoff });
    const g = grid2(channels, out.h * out.w);
    try go(k.resize, st, g[0], g[1], &a);
}

/// tf_br_tiles: dst [out.h * out.w, cpad] from src [., src_w, channels], a x b patches (mode 0) or grid tiles (mode 1).
pub fn tiles(k: *const Kernels, st: cuda.Stream, src: u64, dst: u64, out: Grid, src_w: usize, channels: usize, a_: usize, b_: usize, grid: bool, cpad: usize) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ src, dst });
    addInts(&a, &.{ out.h, out.w, src_w, channels, a_, b_, @intFromBool(grid), cpad });
    const g = grid2(cpad, out.h * out.w);
    try go(k.tiles, st, g[0], g[1], &a);
}

/// tf_br_conv3: y [pixels, cout] = conv3x3(x) + b of a 3-channel picture, w [27, cout].
pub fn conv3(k: *const Kernels, st: cuda.Stream, x: u64, w: u64, b: u64, y: u64, g_: Grid, cout: usize) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ x, w, b, y });
    addInts(&a, &.{ g_.h, g_.w, cout });
    const g = grid2(cout, g_.h * g_.w);
    try go(k.conv3, st, g[0], g[1], &a);
}

/// tf_br_deform: the columns [count, k * k * 64] of pixels first .. first + count, offsets and modulators in offs.
pub fn deform(k: *const Kernels, st: cuda.Stream, x: u64, offs: u64, col: u64, g_: Grid, side: usize, first: usize, count: usize, ldo: usize) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ x, offs, col });
    addInts(&a, &.{ g_.h, g_.w, side, first, count, ldo });
    const g = grid2(side * side, count);
    try go(k.deform, st, g[0], g[1], &a);
}

/// tf_br_pool: mean [channels] over x's pixels.
pub fn pool(k: *const Kernels, st: cuda.Stream, x: u64, mean: u64, pixels: usize, channels: usize) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ x, mean });
    addInts(&a, &.{ pixels, channels });
    try go(k.pool, st, .{ .x = @intCast(channels) }, .{ .x = 256 }, &a);
}

/// tf_br_gate: x [pixels, channels] *= sigmoid(g [pixels, 16] . w + b).
pub fn gate(k: *const Kernels, st: cuda.Stream, g_: u64, w: u64, b: u64, x: u64, pixels: usize, channels: usize) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ g_, w, b, x });
    addInts(&a, &.{ pixels, channels });
    const g = grid2(channels, pixels);
    try go(k.gate, st, g[0], g[1], &a);
}

/// tf_br_head: logits [out] = the head's 1x1 over [a resized from `in` | q].
pub fn head(k: *const Kernels, st: cuda.Stream, a_: u64, q: u64, w: u64, b: u64, y: u64, in: Grid, out: Grid, cp: usize, cq: usize) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ a_, q, w, b, y });
    addInts(&a, &.{ in.h, in.w, out.h, out.w, cp, cq });
    try go(k.head, st, .{ .x = ceil(out.h * out.w, 256) }, .{ .x = 256 }, &a);
}

test "launch geometry: pixels on grid x within its limit, channels or taps on y, the conv's tiles of 64" {
    // the largest launches: the head's 1024^2 pixels, ipt_blk1's conv3 at 64 channels, a 7x7 chunk of 8192 pixels
    const g = grid2(64, 1024 * 1024);
    try std.testing.expectEqual(@as(u32, 262144), g[0].x);
    try std.testing.expectEqual(@as(u32, 1), g[0].y);
    try std.testing.expectEqual(@as(u32, 256), g[1].x * g[1].y);
    const d = grid2(49, 8192);
    try std.testing.expectEqual(@as(u32, 49), d[1].x);
    try std.testing.expectEqual(@as(u32, 5), d[1].y);
    try std.testing.expectEqual(@as(u32, 1639), d[0].x);
    const three = grid2(3, 512 * 512);
    try std.testing.expectEqual(@as(u32, 85), three[1].y);
    const c = convGrid(.{ .rows = 32, .cols = 32, .cin = 5760, .ldx = 5760, .cout = 64, .ldw = 64, .k = 3, .ldy = 3456 });
    try std.testing.expectEqual(@as(u32, 16), c.x);
    try std.testing.expectEqual(@as(u32, 1), c.y);
    // the squeeze's qkv at stage 3: 4608 outputs in 72 tiles; a grid's y and z stay under 65,535
    try std.testing.expectEqual(@as(u32, 72), convGrid(.{ .rows = 576, .cin = 1536, .ldx = 1536, .cout = 4608, .ldw = 4608, .ldy = 4608 }).y);
    try std.testing.expect((264 / 12) * (264 / 12) < 65535);
}

test "the window kernel's shared tile and the conv's fit 48 KiB of static shared memory" {
    const window_smem = (144 * 33 + window_queries * 32 + window_queries * 145) * 4;
    try std.testing.expectEqual(@as(usize, 36000), window_smem);
    try std.testing.expect(window_smem <= 48 * 1024);
    try std.testing.expectEqual(@as(usize, 0), 144 % window_queries);
    try std.testing.expectEqual(@as(usize, window_threads), window_queries * 8);
    const conv_smem = (32 * 68 + 32 * 64) * 4;
    try std.testing.expect(conv_smem <= 48 * 1024);
}
