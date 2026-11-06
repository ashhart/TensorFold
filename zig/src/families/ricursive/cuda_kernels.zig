//! Ripple's glue kernels on CUDA (ricursive.cu): the Metal family's glue.metal launches, one by one.

const std = @import("std");
const cuda = @import("cuda");

const Stream = cuda.Stream;
const Function = cuda.Function;

fn int(x: usize) c_int {
    return @intCast(x);
}

fn ceil(n: usize, d: usize) u32 {
    return @intCast((n + d - 1) / d);
}

/// Threads a block in every glue launch (ricursive.cuh's NT).
pub const threads = 256;
/// The widest row tf_rc_add_norm holds: each thread four 8-value chunks (MOST).
pub const most_width = 8 * threads * 4;

pub const Kernels = struct {
    mod: cuda.Module,
    embed: Function,
    add_norm: Function,
    gate: Function,
    silu_clamp: Function,
    tap: Function, // a drafter's tap: residual rows into a slot's ring (dflash2.metal's rc_tap)
    // plug-in point: the prep and attention launches, a Function each, tf_rc_<field> in ricursive.cu

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.ricursive);
        errdefer mod.unload();
        var k: Kernels = undefined;
        k.mod = mod;
        inline for (@typeInfo(Kernels).@"struct".field_names, @typeInfo(Kernels).@"struct".field_types) |name, T| if (T == Function) {
            const sym = "tf_rc_" ++ name;
            @field(k, name) = try mod.function(sym);
        };
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }
};

fn go(f: Function, s: Stream, grid: cuda.Dim3, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads }, .shared = 0 }, s, a);
}

/// An elementwise launch's grid: 256 of a row's values a block, a grid row a row (at most 65,535).
fn plane(rows: usize, width: usize) !cuda.Dim3 {
    if (rows == 0 or rows > 65535 or width == 0) return error.UnsupportedShape;
    return .{ .x = ceil(width, threads), .y = @intCast(rows) };
}

/// R and V0 = bf16(E[ids] * scale), the fp32 scale: the residual's rows and the value residual's source (u32 ids).
pub fn embed(k: *const Kernels, s: Stream, ids: u64, table: u64, resid: u64, v0: u64, rows: usize, width: usize, scale: f32) !void {
    return embedBy(k.embed, s, ids, table, resid, v0, rows, width, scale);
}

/// `embed` through `f`: the served kernel or a control of it.
pub fn embedBy(f: Function, s: Stream, ids: u64, table: u64, resid: u64, v0: u64, rows: usize, width: usize, scale: f32) !void {
    const grid = try plane(rows, width);
    var a: cuda.Args = .{};
    for ([_]u64{ ids, table, resid, v0 }) |v| a.add(v);
    a.add(int(rows));
    a.add(int(width));
    a.add(scale);
    try go(f, s, grid, &a);
}

/// One tf_rc_add_norm launch: `branch` 0 skips the add (and `post`), `v0` 0 skips the lerp (and `vx`).
pub const NormRows = struct {
    resid: u64, // R, read and written in place
    branch: u64 = 0, // Y: the attention's or the FFN's output rows
    post: u64 = 0, // w_post, Y's norm weight
    next: u64, // w_next, the next norm's weight
    v0: u64 = 0, // the value residual's source rows
    x: u64, // X: the next norm's rows
    vx: u64 = 0, // lerp(V0, X, blend): v's input
    rows: usize,
    width: usize,
    eps: f32,
    blend: f32 = 0,
};

/// tf_rc_add_norm's layout: 8-value chunks, at most most_width a row, every pointer 16-byte aligned.
pub fn normLayout(p: NormRows) !void {
    if (p.width == 0 or p.width % 8 != 0 or p.width > most_width) return error.UnsupportedWidth;
    if (p.rows == 0 or p.rows > std.math.maxInt(i32)) return error.UnsupportedShape;
    if ((p.branch == 0) != (p.post == 0) or (p.v0 == 0) != (p.vx == 0)) return error.MissingOperand;
    for ([_]u64{ p.resid, p.branch, p.post, p.next, p.v0, p.x, p.vx }) |v| if (v % 16 != 0) return error.Misaligned;
}

/// R = bf16(R + norm(Y, w_post)) with Y, then X = norm(R, w_next), and VX = lerp(V0, X, blend) with V0: a row a block.
pub fn addNorm(k: *const Kernels, s: Stream, p: NormRows) !void {
    return addNormBy(k.add_norm, s, p);
}

/// `addNorm` through `f`: the served kernel or a control of it.
pub fn addNormBy(f: Function, s: Stream, p: NormRows) !void {
    try normLayout(p);
    var a: cuda.Args = .{};
    for ([_]u64{ p.resid, p.branch, p.post, p.next, p.v0, p.x, p.vx }) |v| a.add(v);
    a.add(int(p.rows));
    a.add(int(p.width));
    a.add(p.eps);
    a.add(p.blend);
    try go(f, s, .{ .x = @intCast(p.rows) }, &a);
}

/// The attention's output gate in place: A = bf16(A * bf16(sigmoid(G))) over rows of `width` (query heads x 128).
pub fn gate(k: *const Kernels, s: Stream, act: u64, g: u64, rows: usize, width: usize) !void {
    return gateBy(k.gate, s, act, g, rows, width);
}

/// `gate` through `f`: the served kernel or a control of it.
pub fn gateBy(f: Function, s: Stream, act: u64, g: u64, rows: usize, width: usize) !void {
    const grid = try plane(rows, width);
    var a: cuda.Args = .{};
    a.add(act);
    a.add(g);
    a.add(int(rows));
    a.add(int(width));
    try go(f, s, grid, &a);
}

/// A = bf16(bf16(silu(min(g, limit))) * clamp(u, -limit, limit)) from stacked gate | up rows, 2 x `width` each.
pub fn siluClamp(k: *const Kernels, s: Stream, gu: u64, act: u64, rows: usize, width: usize, limit: f32) !void {
    return siluClampBy(k.silu_clamp, s, gu, act, rows, width, limit);
}

/// `siluClamp` through `f`: the served kernel or a control of it.
pub fn siluClampBy(f: Function, s: Stream, gu: u64, act: u64, rows: usize, width: usize, limit: f32) !void {
    const grid = try plane(rows, width);
    var a: cuda.Args = .{};
    a.add(gu);
    a.add(act);
    a.add(int(rows));
    a.add(int(width));
    a.add(limit);
    try go(f, s, grid, &a);
}

/// One tf_rc_tap launch: rows into tap column col at ring row pos % ring_rows; chunk: p0 + r of s0; round: pos, slot.
pub const TapRows = struct {
    resid: u64, // R [rows, width] bf16
    ring: u64, // the rings from slot 0: [slots][ring_rows][taps][width] bf16
    pos: u64 = 0, // int32 [rows]: a round's positions
    slot: u64 = 0, // int32 [rows]: a round's slots
    p0: usize = 0,
    s0: usize = 0,
    rows: usize,
    width: usize,
    taps: usize,
    col: usize,
    ring_rows: usize,
};

/// tf_rc_tap's layout: 16-byte rows (8 values a thread), a column of the taps, both or neither of pos and slot.
pub fn tapLayout(p: TapRows) !void {
    if (p.width == 0 or p.width % 8 != 0) return error.UnsupportedWidth;
    if (p.rows == 0 or p.rows > 65535 or p.ring_rows == 0 or p.taps == 0 or p.col >= p.taps) return error.UnsupportedShape;
    if ((p.pos == 0) != (p.slot == 0)) return error.MissingOperand;
    if (p.pos == 0 and (p.p0 + p.rows > std.math.maxInt(i32) or p.s0 > std.math.maxInt(i32))) return error.UnsupportedShape;
    for ([_]u64{ p.resid, p.ring }) |v| if (v % 16 != 0) return error.Misaligned;
}

/// T[slot][pos % ring_rows][col] = R[r] for each row r: a bf16 copy, no rounding.
pub fn tap(k: *const Kernels, s: Stream, p: TapRows) !void {
    return tapBy(k.tap, s, p);
}

/// `tap` through `f`: the served kernel or a control of it.
pub fn tapBy(f: Function, s: Stream, p: TapRows) !void {
    try tapLayout(p);
    var a: cuda.Args = .{};
    for ([_]u64{ p.resid, p.ring, p.pos, p.slot }) |v| a.add(v);
    for ([_]usize{ p.p0, p.s0, p.rows, p.width, p.taps, p.col, p.ring_rows }) |v| a.add(int(v));
    try go(f, s, .{ .x = ceil(p.width / 8, threads), .y = @intCast(p.rows) }, &a);
}

test "launch geometry: ceilings and the add-norm's layout" {
    try std.testing.expectEqual(@as(u32, 20), ceil(5120, threads));
    try std.testing.expectEqual(@as(u32, 40), ceil(10240, threads));
    try std.testing.expectEqual(@as(u32, 160), ceil(40960, threads));
    try std.testing.expectEqual(@as(usize, 8192), most_width);
    try std.testing.expectError(error.UnsupportedShape, plane(65536, 5120));
    try std.testing.expectError(error.UnsupportedShape, plane(1, 0));
    const p: NormRows = .{ .resid = 256, .next = 512, .x = 768, .rows = 2048, .width = 5120, .eps = 1e-6 };
    try normLayout(p);
    for ([_]usize{ 5124, 8200, 0 }) |w| {
        var bad = p;
        bad.width = w;
        try std.testing.expectError(error.UnsupportedWidth, normLayout(bad));
    }
    var bad = p;
    bad.branch = 1024;
    try std.testing.expectError(error.MissingOperand, normLayout(bad));
    bad = p;
    bad.v0 = 1024;
    try std.testing.expectError(error.MissingOperand, normLayout(bad));
    bad = p;
    bad.x = 776;
    try std.testing.expectError(error.Misaligned, normLayout(bad));
}

test "launch geometry: the tap's layout" {
    const p: TapRows = .{ .resid = 256, .ring = 4096, .rows = 2048, .width = 5120, .taps = 5, .col = 4, .ring_rows = 2064 };
    try tapLayout(p);
    try std.testing.expectEqual(@as(u32, 3), ceil(p.width / 8, threads)); // 640 chunks of 8
    var bad = p;
    bad.width = 5124;
    try std.testing.expectError(error.UnsupportedWidth, tapLayout(bad));
    bad = p;
    bad.col = 5;
    try std.testing.expectError(error.UnsupportedShape, tapLayout(bad));
    bad = p;
    bad.rows = 65536;
    try std.testing.expectError(error.UnsupportedShape, tapLayout(bad));
    bad = p;
    bad.pos = 512;
    try std.testing.expectError(error.MissingOperand, tapLayout(bad));
    bad.slot = 768;
    try tapLayout(bad);
    bad = p;
    bad.ring = 4104;
    try std.testing.expectError(error.Misaligned, tapLayout(bad));
}
