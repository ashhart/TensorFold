//! The DFlash2 drafter's glue on CUDA (ricursive_dflash2.cu): dflash2.metal's norm, prep, conv, silu, streams batched.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_kernels.zig");

const Stream = cuda.Stream;
const Function = cuda.Function;

/// Threads a block in every launch (ricursive.cuh's NT); prep runs a warp a (row, head).
pub const threads = 256;
const warps = threads / 32;
pub const head_dim = 128;
/// The widest row the add-norm holds (tf_rc's load_row).
pub const most_width = glue.most_width;

fn int(x: usize) c_int {
    return @intCast(x);
}

fn ceil(n: usize, d: usize) u32 {
    return @intCast((n + d - 1) / d);
}

pub const Kernels = struct {
    mod: cuda.Module,
    add_norm: Function,
    prep: Function,
    conv: Function,
    silu: Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.ricursive_dflash2);
        errdefer mod.unload();
        var k: Kernels = undefined;
        k.mod = mod;
        inline for (@typeInfo(Kernels).@"struct".field_names, @typeInfo(Kernels).@"struct".field_types) |name, T| if (T == Function) {
            const sym = "tf_rcd_" ++ name;
            @field(k, name) = try mod.function(sym);
        };
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }
};

fn go(f: Function, s: Stream, dims: cuda.Dim3, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = dims, .block = .{ .x = threads }, .shared = 0 }, s, a);
}

/// An elementwise launch's grid: 256 of a row's values a block, a grid row a row (at most 65,535).
fn plane(rows: usize, width: usize) !cuda.Dim3 {
    if (rows == 0 or rows > 65535 or width == 0 or width > std.math.maxInt(i32)) return error.UnsupportedShape;
    return .{ .x = ceil(width, threads), .y = @intCast(rows) };
}

/// The block's residual rows R = E[ids] unscaled: tf_rc_embed at scale 1 (exact); `scratch` takes its second copy.
pub fn embed(rk: *const glue.Kernels, s: Stream, ids: u64, table: u64, resid: u64, scratch: u64, rows: usize, width: usize) !void {
    return embedBy(rk.embed, s, ids, table, resid, scratch, rows, width);
}

/// `embed` through `f`: tf_rc_embed or a control of it.
pub fn embedBy(f: Function, s: Stream, ids: u64, table: u64, resid: u64, scratch: u64, rows: usize, width: usize) !void {
    return glue.embedBy(f, s, ids, table, resid, scratch, rows, width, 1.0);
}

/// One add-norm launch: `branch` 0 is norm16's form (X = norm(R), R only read).
pub const Norm = struct {
    resid: u64, // R: the residual (in place with the add), or norm16's bf16 rows
    branch: u64 = 0, // Y: added raw, bf16(R + Y)
    w: u64,
    x: u64,
    rows: usize,
    width: usize,
    eps: f32,
};

/// The add-norm's layout: 8-value chunks, at most most_width a row, every pointer 16-byte aligned.
pub fn normLayout(p: Norm) !void {
    if (p.width == 0 or p.width % 8 != 0 or p.width > most_width) return error.UnsupportedWidth;
    if (p.rows == 0 or p.rows > std.math.maxInt(i32)) return error.UnsupportedShape;
    for ([_]u64{ p.resid, p.branch, p.w, p.x }) |v| if (v % 16 != 0) return error.Misaligned;
}

/// R = bf16(R + Y) with Y, then X = bf16(w bf16(R / rms)): dflash2.metal's rc_df_norm (rc_df_norm16 without Y).
pub fn norm(k: *const Kernels, s: Stream, p: Norm) !void {
    return normBy(k.add_norm, s, p);
}

/// `norm` through `f`: the served kernel or a control of it.
pub fn normBy(f: Function, s: Stream, p: Norm) !void {
    try normLayout(p);
    var a: cuda.Args = .{};
    for ([_]u64{ p.resid, p.branch, p.w, p.x }) |v| a.add(v);
    a.add(int(p.rows));
    a.add(int(p.width));
    a.add(p.eps);
    try go(f, s, .{ .x = @intCast(p.rows) }, &a);
}

/// One stream's rows in a prep launch (the kernel's Segment): its K and V rings, first row and its position; row order.
pub const Segment = extern struct { k: u64, v: u64, begin: i64, pos: i64 };

/// Where a row's heads sit in X: a block's q | k | v projections, or a context's k | v (no query heads).
pub const Layout = struct {
    heads: usize,
    stride: usize,
    q_off: usize,
    k_off: usize,
    v_off: usize,

    /// dflash2.zig's block rows: q (heads), k and v (kv_heads each), 128 wide a head.
    pub fn block(heads: usize, kv_heads: usize) Layout {
        const qw = heads * head_dim;
        const kvw = kv_heads * head_dim;
        return .{ .heads = heads, .stride = qw + 2 * kvw, .q_off = 0, .k_off = qw, .v_off = qw + kvw };
    }

    /// dflash2.zig's absorb rows: each layer's k and v of the normed context.
    pub fn context(kv_heads: usize) Layout {
        const kvw = kv_heads * head_dim;
        return .{ .heads = 0, .stride = 2 * kvw, .q_off = 0, .k_off = 0, .v_off = kvw };
    }
};

/// One prep launch: segs on the device; inv the 64 fp32 inverse frequencies; q (0 for a context); rope false: NoPE.
pub const Prep = struct { x: u64, qw: u64, kw: u64, inv: u64, segs: u64, streams: usize, q: u64, rows: usize, kv_heads: usize, at: Layout, ring: usize, rope: bool, eps: f32 };

/// The launch's own limits (each stream's rows against its ring: checkSegments, before the table's upload).
pub fn admit(p: Prep) !void {
    if (p.rows == 0 or p.rows > 65535 or p.streams == 0 or p.streams > p.rows or p.kv_heads == 0) return error.PrepShape;
    if (p.ring == 0 or p.ring > std.math.maxInt(i32) or p.at.stride > std.math.maxInt(i32)) return error.PrepShape;
    if ((p.at.heads == 0) != (p.q == 0)) return error.MissingOperand;
}

/// A block of 8 heads across, a row down.
pub fn grid(p: Prep) cuda.Dim3 {
    return .{ .x = ceil(p.at.heads + p.kv_heads, warps), .y = @intCast(p.rows), .z = 1 };
}

/// Heads normed and rotated (not on the NoPE layer), keys and values into each stream's ring: rc_df_prep, batched.
pub fn prep(k: *const Kernels, s: Stream, p: Prep) !void {
    return prepBy(k.prep, s, p);
}

/// `prep` through `f`: the served kernel or a control of it.
pub fn prepBy(f: Function, s: Stream, p: Prep) !void {
    try admit(p);
    var a: cuda.Args = .{};
    for ([_]u64{ p.x, p.qw, p.kw, p.inv, p.segs }) |v| a.add(v);
    a.add(int(p.streams));
    a.add(p.q);
    for ([_]usize{ p.rows, p.at.heads, p.kv_heads, p.at.stride, p.at.q_off, p.at.k_off, p.at.v_off, p.ring }) |v| a.add(int(v));
    a.add(@as(c_int, @intFromBool(p.rope)));
    a.add(p.eps);
    try go(f, s, grid(p), &a);
}

/// A launch's segments: the first at row 0, in row order, positions whole, and no stream past its ring's slots.
pub fn checkSegments(segs: []const Segment, rows: usize, ring: usize) !void {
    if (segs.len == 0 or segs[0].begin != 0) return error.BadSegments;
    for (segs, 0..) |sg, i| {
        const end: i64 = if (i + 1 < segs.len) segs[i + 1].begin else @intCast(rows);
        if (sg.pos < 0 or end < sg.begin or end > rows or end - sg.begin > ring) return error.BadSegments;
        if (end > sg.begin and (sg.k == 0 or sg.v == 0)) return error.BadSegments;
    }
}

/// RoPE's 64 inverse frequencies as dflash2.metal computes them: 1 / fp32(theta^(2i / 128)), in fp32.
pub fn inverseFrequencies(theta: f32) [head_dim / 2]f32 {
    var out: [head_dim / 2]f32 = undefined;
    for (&out, 0..) |*f, i| {
        const e: f64 = @as(f64, @floatFromInt(2 * i)) / head_dim;
        const p: f32 = @floatCast(std.math.pow(f64, theta, e));
        f.* = 1.0 / p;
    }
    return out;
}

/// One conv launch: `dyn` rows [2][taps][groups], `base` [2][taps][width]; rows in whole blocks of `block` a stream.
pub const Conv = struct { x: u64, dyn: u64, base: u64, y: u64, rows: usize, width: usize, block: usize, groups: usize, side: usize, taps: usize };

pub fn convShape(p: Conv) !void {
    if (p.groups == 0 or p.width % p.groups != 0 or p.block == 0 or p.side > 1 or p.taps == 0) return error.UnsupportedShape;
    if (p.groups > std.math.maxInt(i32) or p.taps > p.block) return error.UnsupportedShape;
}

/// Y = the dynamic causal conv of X inside each row's block, dflash2.metal's rc_df_conv rounding.
pub fn conv(k: *const Kernels, s: Stream, p: Conv) !void {
    return convBy(k.conv, s, p);
}

/// `conv` through `f`: the served kernel or a control of it.
pub fn convBy(f: Function, s: Stream, p: Conv) !void {
    try convShape(p);
    const g = try plane(p.rows, p.width);
    var a: cuda.Args = .{};
    for ([_]u64{ p.x, p.dyn, p.base, p.y }) |v| a.add(v);
    for ([_]usize{ p.rows, p.width, p.block, p.groups, p.side, p.taps }) |v| a.add(int(v));
    try go(f, s, g, &a);
}

/// A = bf16(bf16(g sigmoid(g)) * u) from stacked gate | up rows, 2 x `width` each (no clamp).
pub fn silu(k: *const Kernels, s: Stream, gu: u64, act: u64, rows: usize, width: usize) !void {
    return siluBy(k.silu, s, gu, act, rows, width);
}

/// `silu` through `f`: the served kernel or a control of it.
pub fn siluBy(f: Function, s: Stream, gu: u64, act: u64, rows: usize, width: usize) !void {
    const g = try plane(rows, width);
    var a: cuda.Args = .{};
    a.add(gu);
    a.add(act);
    a.add(int(rows));
    a.add(int(width));
    try go(f, s, g, &a);
}

test "dflash2 glue geometry: grids, layouts and the norm's limits" {
    const t = std.testing;
    try t.expectEqual(@as(u32, 20), (try plane(16, 5120)).x);
    try t.expectEqual(@as(u32, 160), (try plane(128, 40960)).x);
    try t.expectError(error.UnsupportedShape, plane(65536, 5120));
    const b = Layout.block(80, 8);
    try t.expectEqual(@as(usize, 12288), b.stride);
    try t.expectEqual(@as(usize, 10240), b.k_off);
    try t.expectEqual(@as(usize, 11264), b.v_off);
    const cx = Layout.context(8);
    try t.expectEqual(@as(usize, 2048), cx.stride);
    try t.expectEqual(@as(usize, 1024), cx.v_off);
    var p: Prep = .{ .x = 16, .qw = 32, .kw = 48, .inv = 64, .segs = 80, .streams = 8, .q = 96, .rows = 128, .kv_heads = 8, .at = b, .ring = 4224, .rope = true, .eps = 1e-6 };
    try admit(p);
    try t.expectEqual(@as(u32, 11), grid(p).x);
    try t.expectEqual(@as(u32, 128), grid(p).y);
    p.at = cx;
    try t.expectError(error.MissingOperand, admit(p));
    p.q = 0;
    try admit(p);
    try t.expectEqual(@as(u32, 1), grid(p).x);
    const n: Norm = .{ .resid = 256, .w = 512, .x = 768, .rows = 2048, .width = 5120, .eps = 1e-6 };
    try normLayout(n);
    var bad = n;
    bad.width = 8200;
    try t.expectError(error.UnsupportedWidth, normLayout(bad));
    bad = n;
    bad.branch = 1032;
    try t.expectError(error.Misaligned, normLayout(bad));
    const cv: Conv = .{ .x = 0, .dyn = 0, .base = 0, .y = 0, .rows = 128, .width = 5120, .block = 16, .groups = 320, .side = 1, .taps = 4 };
    try convShape(cv);
    var bc = cv;
    bc.side = 2;
    try t.expectError(error.UnsupportedShape, convShape(bc));
    bc = cv;
    bc.groups = 300;
    try t.expectError(error.UnsupportedShape, convShape(bc));
}

test "dflash2 prep segments: row order, rows within a ring, the kernel's layout" {
    const t = std.testing;
    try t.expectEqual(@as(usize, 32), @sizeOf(Segment));
    try t.expectEqual(@as(usize, 16), @offsetOf(Segment, "begin"));
    const two = [_]Segment{ .{ .k = 256, .v = 512, .begin = 0, .pos = 4221 }, .{ .k = 1024, .v = 2048, .begin = 16, .pos = 90 } };
    try checkSegments(&two, 32, 4224);
    try t.expectError(error.BadSegments, checkSegments(&two, 15, 4224)); // the second stream begins past the rows
    try t.expectError(error.BadSegments, checkSegments(&two, 32, 15)); // 16 rows overrun a 15-slot ring
    var late = two;
    late[0].begin = 1;
    try t.expectError(error.BadSegments, checkSegments(&late, 32, 4224));
    var back = two;
    back[1].pos = -1;
    try t.expectError(error.BadSegments, checkSegments(&back, 32, 4224));
}

test "dflash2 RoPE frequencies: dflash2.metal's 1 / pow(theta, 2i / 128) in fp32" {
    const f = inverseFrequencies(10000);
    try std.testing.expectEqual(@as(f32, 1), f[0]);
    try std.testing.expectEqual(@as(f32, 1.0 / @as(f32, @floatCast(std.math.pow(f64, 10000, 0.5)))), f[32]);
    for (f[1..], f[0..63]) |next, prev| try std.testing.expect(next < prev);
}
