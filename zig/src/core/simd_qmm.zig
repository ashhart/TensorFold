//! Row-exact 4-bit projections before the M5: 8-row simdgroup-matrix tiles, and a scalar twin for 1-2 rows.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

/// K chunks by output width alone, so a row's reduction tree never depends on the window's row count.
pub fn splits(n: usize) u32 {
    return if (n <= 64) 32 else if (n <= 6144) 16 else 8;
}

/// Output tiles a simdgroup within the 16 KB reduction buffer; tiles never change a row's bits.
pub fn tiles(n: usize, rt: u32, s: u32) u32 {
    var nt: u32 = if (n % 32 == 0) 4 else if (n % 16 == 0) 2 else 1;
    while (nt > 1 and !fits(s, nt, rt)) nt /= 2;
    return nt;
}

/// Groups of inputs the scalar kernel stages at 1 or 2 rows (0: it can't take them).
pub fn scalarBlock(rows: usize, s: u32) u32 {
    if (rows < 1 or rows > max_scalar_rows) return 0;
    const xb: u32 = if (rows == 1) 32 else @max(s, 16);
    return if (xb % s == 0 and rows * xb * 76 * 4 <= 20480) xb else 0;
}

pub const max_scalar_rows = 2;
const s_values = [_]u32{ 8, 16, 32 };
const nt_values = [_]u32{ 1, 2, 4 };

pub const Args = extern struct { k: i32, n: i32, r: i32, one: f32 = 1.0 };

/// Whether a tile shape's reduction buffer fits 16 KB (the only shapes `tiles` picks, and the only ones built).
fn fits(s: u32, nt: u32, rt: u32) bool {
    return s * rt * nt * 64 * 4 <= 16384;
}

pub const Pipelines = struct {
    mma: [s_values.len][nt_values.len][2]?mtl.Pipeline,
    scalar: [s_values.len][2][max_scalar_rows]mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lib = try mtl.Library.fromSource(device, sources.core_simd_qmm, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipelines = undefined;
        inline for (s_values, 0..) |s, si| {
            inline for (nt_values, 0..) |nt, ti| inline for (.{ 1, 2 }, 0..) |rt, ri| {
                p.mma[si][ti][ri] = if (comptime fits(s, nt, rt)) try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_sq_mma_{d}_{d}_{d}", .{ s, nt, rt }), false) else null;
            };
            inline for (.{ 1, 2 }, 0..) |nr, ni| inline for (.{ 1, 2 }, 0..) |rs, ri| {
                p.scalar[si][ni][ri] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_sq_scalar_{d}_{d}_{d}", .{ s, nr, rs }), false);
            };
        }
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        for (p.mma) |a| for (a) |b| for (b) |x| if (x) |y| y.deinit();
        for (p.scalar) |a| for (a) |b| for (b) |x| x.deinit();
    }
};

/// One dispatch: buffers x, W, S, B, args, out at 0-5; `groups` threadgroups of `threads`.
pub const Call = struct { pipeline: mtl.Pipeline, args: Args, groups: mtl.Size, threads: usize };

/// y = x W^T for 4-bit groups of 64: the scalar twin at 1-2 rows when `scalar` (checked equal), else MMA tiles.
pub fn call(p: *const Pipelines, n: usize, k: usize, rows: usize, scalar: bool) !Call {
    if (n == 0 or n % 8 != 0 or n > std.math.maxInt(i32) or k == 0 or k % 64 != 0 or k > std.math.maxInt(i32)) return error.UnsupportedSimdShape;
    if (rows == 0 or rows > std.math.maxInt(i32)) return error.NoRows;
    const s = splits(n);
    const si = std.mem.indexOfScalar(u32, &s_values, s).?;
    const args = Args{ .k = @intCast(k), .n = @intCast(n), .r = @intCast(rows) };
    if (scalar and scalarBlock(rows, s) != 0) {
        const nr: usize = if (n > 2048) 2 else 1;
        const sgs: usize = if (n > 2048) @max(1, 16 / ((32 / s) * nr)) else 8;
        const per = sgs * (32 / s) * nr;
        const pipeline = p.scalar[si][nr - 1][rows - 1];
        if (pipeline.maxThreads() < sgs * 32) return error.SimdThreads;
        return .{ .pipeline = pipeline, .args = args, .groups = .{ .width = (n + per - 1) / per }, .threads = sgs * 32 };
    }
    const rt: u32 = @min(2, @as(u32, @intCast((rows + 7) / 8)));
    const nt = tiles(n, rt, s);
    const pipeline = p.mma[si][std.mem.indexOfScalar(u32, &nt_values, nt).?][rt - 1] orelse return error.UnsupportedSimdShape;
    const sgs: usize = @min(s, 16, pipeline.maxThreads() / 32);
    if (sgs == 0) return error.SimdThreads;
    return .{ .pipeline = pipeline, .args = args, .groups = .{ .width = (n + 8 * nt - 1) / (8 * nt), .height = (rows + 8 * rt - 1) / (8 * rt) }, .threads = sgs * 32 };
}

/// A checkpoint's [n, k] matrix: words [n][k/8], scales and biases [n][k/64] bf16, at byte offsets.
pub const Matrix = struct { w: mtl.Buffer, w_off: usize = 0, scales: mtl.Buffer, s_off: usize = 0, biases: mtl.Buffer, b_off: usize = 0, n: usize, k: usize };

pub fn encode(e: mtl.ComputeEncoder, c: Call, m: Matrix, x: mtl.Buffer, x_off: usize, y: mtl.Buffer, y_off: usize) void {
    e.setPipeline(c.pipeline);
    e.setBuffer(x, x_off, 0);
    e.setBuffer(m.w, m.w_off, 1);
    e.setBuffer(m.scales, m.s_off, 2);
    e.setBuffer(m.biases, m.b_off, 3);
    e.setValue(c.args, 4);
    e.setBuffer(y, y_off, 5);
    e.dispatchGroups(c.groups, .{ .width = c.threads });
}

/// Whether the scalar twin's 1-2 row results equal the MMA tiles' for `m`; where not, every width takes tiles.
pub fn twinExact(p: *const Pipelines, device: mtl.Device, m: Matrix) !bool {
    const rows = 8;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const x = try device.buffer(rows * m.k * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    var prng = std.Random.DefaultPrng.init(0x51d_0001);
    for (x.slice(u16, rows * m.k)) |*v| v.* = @truncate(@as(u32, @bitCast(prng.random().float(f32) * 2 - 1)) >> 16);
    const tiled = try device.buffer(rows * m.n * 2, mtl.ResourceOptions.shared);
    defer tiled.deinit();
    const twin = try device.buffer(rows * m.n * 2, mtl.ResourceOptions.shared);
    defer twin.deinit();
    const pairs = try device.buffer(4 * m.n * 2, mtl.ResourceOptions.shared);
    defer pairs.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    encode(e, try call(p, m.n, m.k, rows, false), m, x, 0, tiled, 0);
    for (0..rows) |r| encode(e, try call(p, m.n, m.k, 1, true), m, x, r * m.k * 2, twin, r * m.n * 2);
    for ([_]usize{ 0, rows - 2 }, 0..) |r, i| encode(e, try call(p, m.n, m.k, 2, true), m, x, r * m.k * 2, pairs, i * 2 * m.n * 2);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.log.err("simd_qmm check: {s}", .{why});
        return error.SimdCheckFailed;
    }
    const all = tiled.slice(u16, rows * m.n);
    if (!std.mem.eql(u16, all, twin.slice(u16, rows * m.n))) return false;
    const two = pairs.slice(u16, 4 * m.n);
    return std.mem.eql(u16, all[0 .. 2 * m.n], two[0 .. 2 * m.n]) and std.mem.eql(u16, all[(rows - 2) * m.n ..], two[2 * m.n ..]);
}

test "chunks follow output width only and tiles fit the reduction buffer" {
    try std.testing.expectEqual(@as(u32, 32), splits(48));
    try std.testing.expectEqual(@as(u32, 16), splits(5120));
    try std.testing.expectEqual(@as(u32, 8), splits(34816));
    for (s_values) |s| for ([_]u32{ 1, 2 }) |rt| {
        const nt = tiles(34816, rt, s);
        try std.testing.expect(s * rt * nt * 64 * 4 <= 16384);
        try std.testing.expect(std.mem.indexOfScalar(u32, &nt_values, nt) != null);
    };
    try std.testing.expectEqual(@as(u32, 1), tiles(40, 1, 8));
    try std.testing.expectEqual(@as(u32, 2), tiles(48, 1, 8));
}

test "the scalar twin stages its groups within 20 KB at one and two rows only" {
    for (s_values) |s| {
        try std.testing.expectEqual(@as(u32, 32), scalarBlock(1, s));
        try std.testing.expectEqual(@max(s, 16), scalarBlock(2, s));
        try std.testing.expectEqual(@as(u32, 0), scalarBlock(3, s));
        try std.testing.expectEqual(@as(u32, 0), scalarBlock(0, s));
    }
}
