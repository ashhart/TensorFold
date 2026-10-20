//! Row-exact 2- to 8-bit projections before the M5 on simd_qmm's tiles, K chunks and scalar twin, rows at a stride.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const simd = @import("simd_qmm.zig");

/// The code widths the tiles unpack (groups of 64); 4-bit gives simd_qmm's MMA bits.
pub const bit_widths = [_]u8{ 2, 3, 4, 5, 6, 8 };
const s_values = [_]u32{ 8, 16, 32 };
const nt_values = [_]u32{ 1, 2, 4 };

pub const Args = extern struct { k: i32, n: i32, r: i32, ld: i32, one: f32 = 1.0 };

/// Pipelines built on first use from one library.
pub const Pipelines = struct {
    device: mtl.Device,
    lib: mtl.Library,
    built: [bit_widths.len][s_values.len][nt_values.len][2]?mtl.Pipeline = @splat(@splat(@splat(@splat(null)))),
    /// Simdgroups a 16-row tile of a grid of at least 512 groups takes at most: 4 on the M1, else 16.
    wide: usize = 16,
    scalar: [bit_widths.len][s_values.len][2][simd.max_scalar_rows]?mtl.Pipeline = @splat(@splat(@splat(@splat(null)))),

    pub fn load(device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const wide: usize = if (device.generation() == 13) 4 else 16;
        return .{ .device = device, .lib = try mtl.Library.fromSource(device, sources.core_simd_qmm_bits, mtl.CompileOptions.mlx()), .wide = wide };
    }

    pub fn deinit(p: *Pipelines) void {
        for (p.built) |a| for (a) |b| for (b) |c| for (c) |x| if (x) |y| y.deinit();
        for (p.scalar) |a| for (a) |b| for (b) |c| for (c) |x| if (x) |y| y.deinit();
        p.lib.deinit();
    }

    fn get(p: *Pipelines, bits: u8, s: u32, nt: u32, rt: u32) !mtl.Pipeline {
        const bi = std.mem.indexOfScalar(u8, &bit_widths, bits) orelse return error.UnsupportedSimdBits;
        const si = std.mem.indexOfScalar(u32, &s_values, s).?;
        const ti = std.mem.indexOfScalar(u32, &nt_values, nt).?;
        const slot = &p.built[bi][si][ti][rt - 1];
        if (slot.*) |x| return x;
        var name: [48]u8 = undefined;
        const z = try std.fmt.bufPrint(&name, "tf_sqw_mma_q{d}_{d}_{d}_{d}\x00", .{ bits, s, nt, rt });
        slot.* = try mtl.Pipeline.init(p.device, p.lib, name[0 .. z.len - 1 :0], false);
        return slot.*.?;
    }

    fn getScalar(p: *Pipelines, bits: u8, s: u32, nr: usize, rows: usize) !mtl.Pipeline {
        const bi = std.mem.indexOfScalar(u8, &bit_widths, bits) orelse return error.UnsupportedSimdBits;
        const si = std.mem.indexOfScalar(u32, &s_values, s).?;
        const slot = &p.scalar[bi][si][nr - 1][rows - 1];
        if (slot.*) |x| return x;
        var name: [48]u8 = undefined;
        const z = try std.fmt.bufPrint(&name, "tf_sqw_scalar_q{d}_{d}_{d}_{d}\x00", .{ bits, s, nr, rows });
        slot.* = try mtl.Pipeline.init(p.device, p.lib, name[0 .. z.len - 1 :0], false);
        return slot.*.?;
    }
};

/// One dispatch: buffers x, W, S, B, args, out at 0-5; `groups` threadgroups of `threads`.
pub const Call = struct { pipeline: mtl.Pipeline, args: Args, groups: mtl.Size, threads: usize };

/// y[m * ld + j] = x W^T, g64 codes of `bits`: the twin at 1-2 rows when `scalar` (checked equal), else tiles.
pub fn call(p: *Pipelines, bits: u8, n: usize, k: usize, rows: usize, ld: usize, scalar: bool) !Call {
    if (n == 0 or n % 8 != 0 or n > std.math.maxInt(i32) or k == 0 or k % 64 != 0 or k > std.math.maxInt(i32)) return error.UnsupportedSimdShape;
    if (ld < n or ld > std.math.maxInt(i32)) return error.UnsupportedSimdStride;
    if (rows == 0 or rows > std.math.maxInt(i32)) return error.NoRows;
    const s = simd.splits(n);
    const args = Args{ .k = @intCast(k), .n = @intCast(n), .r = @intCast(rows), .ld = @intCast(ld) };
    if (scalar and simd.scalarBlock(rows, s) != 0) {
        const nr: usize = if (n > 2048) 2 else 1;
        const sgs: usize = if (n > 2048) @max(1, 16 / ((32 / s) * nr)) else 8;
        const per = sgs * (32 / s) * nr;
        const pipeline = try p.getScalar(bits, s, nr, rows);
        if (pipeline.maxThreads() < sgs * 32) return error.SimdThreads;
        return .{ .pipeline = pipeline, .args = args, .groups = .{ .width = (n + per - 1) / per }, .threads = sgs * 32 };
    }
    const rt: u32 = @min(2, @as(u32, @intCast((rows + 7) / 8)));
    const nt = simd.tiles(n, rt, s);
    const pipeline = try p.get(bits, s, nt, rt);
    const groups = mtl.Size{ .width = (n + 8 * nt - 1) / (8 * nt), .height = (rows + 8 * rt - 1) / (8 * rt) };
    const many = rt == 2 and s <= 16 and groups.width * groups.height >= 512;
    const sgs: usize = @min(s, if (many) p.wide else 16, pipeline.maxThreads() / 32);
    if (sgs == 0) return error.SimdThreads;
    return .{ .pipeline = pipeline, .args = args, .groups = groups, .threads = sgs * 32 };
}

/// A checkpoint's [n, k] matrix of `bits` codes: words [n][k*bits/32], scales and biases [n][k/64] bf16.
pub const Matrix = struct { w: mtl.Buffer, w_off: usize = 0, scales: mtl.Buffer, s_off: usize = 0, biases: mtl.Buffer, b_off: usize = 0 };

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

/// Whether the scalar twin's 1-2 row results equal the tiles' for an [n, k] matrix of `bits` codes.
pub fn twinExact(p: *Pipelines, device: mtl.Device, bits: u8, m: Matrix, n: usize, k: usize) !bool {
    const rows = 8;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const x = try device.buffer(rows * k * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    var prng = std.Random.DefaultPrng.init(0x51d_0002);
    for (x.slice(u16, rows * k)) |*v| v.* = @truncate(@as(u32, @bitCast(prng.random().float(f32) * 2 - 1)) >> 16);
    const tiled = try device.buffer(rows * n * 2, mtl.ResourceOptions.shared);
    defer tiled.deinit();
    const twin = try device.buffer(rows * n * 2, mtl.ResourceOptions.shared);
    defer twin.deinit();
    const pairs = try device.buffer(4 * n * 2, mtl.ResourceOptions.shared);
    defer pairs.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    encode(e, try call(p, bits, n, k, rows, n, false), m, x, 0, tiled, 0);
    for (0..rows) |r| encode(e, try call(p, bits, n, k, 1, n, true), m, x, r * k * 2, twin, r * n * 2);
    for ([_]usize{ 0, rows - 2 }, 0..) |r, i| encode(e, try call(p, bits, n, k, 2, n, true), m, x, r * k * 2, pairs, i * 2 * n * 2);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.log.err("simd_qmm_bits check: {s}", .{why});
        return error.SimdCheckFailed;
    }
    const all = tiled.slice(u16, rows * n);
    if (!std.mem.eql(u16, all, twin.slice(u16, rows * n))) return false;
    const two = pairs.slice(u16, 4 * n);
    return std.mem.eql(u16, all[0 .. 2 * n], two[0 .. 2 * n]) and std.mem.eql(u16, all[(rows - 2) * n ..], two[2 * n ..]);
}

test "every tile shape simd_qmm picks is built at every width" {
    var count: usize = 0;
    for (s_values) |s| for (nt_values) |nt| for ([_]u32{ 1, 2 }) |rt| {
        if (s * rt * nt * 64 * 4 <= 16384) count += 1;
    };
    try std.testing.expectEqual(@as(usize, 14), count);
    for ([_]usize{ 48, 1024, 5120, 6144, 10240, 12288, 34816, 248320 }) |n| for ([_]u32{ 1, 2 }) |rt| {
        const s = simd.splits(n);
        const nt = simd.tiles(n, rt, s);
        try std.testing.expect(s * rt * nt * 64 * 4 <= 16384);
    };
}
