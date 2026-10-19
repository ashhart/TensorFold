//! dense_rows.cu against core/dense_ref.zig's float64 rows, and every row's bits at every row count.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const dump = @import("byte_dump.zig");

const counts = [_]usize{ 1, 3, 17, 40, 300 };
const most = counts[counts.len - 1];

fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn case(gpu: check.Gpu, a: std.mem.Allocator, s: cuda.Stream, g: *const cuda.dense_rows.DenseRows, rng: std.Random, n: usize, k: usize, worst: *f64) !void {
    const xs = try a.alloc(u16, most * k);
    const xv = try a.alloc(f64, most * k);
    for (xs, xv, 0..) |*h, *v, i| {
        h.* = bf16(rng.floatNorm(f32) * (if (i % 61 == 0) @as(f32, 12) else 1));
        v.* = @as(f32, @bitCast(@as(u32, h.*) << 16));
    }
    const ws = try a.alloc(u16, n * k);
    const wv = try a.alloc(f64, n * k);
    for (ws, wv) |*h, *v| {
        h.* = bf16(rng.floatNorm(f32) * 0.02);
        v.* = @as(f32, @bitCast(@as(u32, h.*) << 16));
    }
    var bufs: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(gpu.gpa);
    }
    const dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xs));
    try bufs.append(gpu.gpa, dx);
    const dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ws));
    try bufs.append(gpu.gpa, dw);
    const outs = try a.alloc(cuda.DeviceBuffer, counts.len);
    for (counts, outs) |m, *o| {
        o.* = try cuda.DeviceBuffer.alloc(gpu.d, m * n * 4);
        try bufs.append(gpu.gpa, o.*);
        try o.fill8(0xff, s.handle);
        try g.rows(s, dx.ptr, k, dw.ptr, o.ptr, n, m, n, k);
    }
    try s.synchronize();
    const wide = try a.alloc(u8, most * n * 4);
    try outs[counts.len - 1].download(0, wide);
    try dump.output(a, "dense-rows", "product", "rows", &.{ most, n }, .f32, .{ .k = k }, wide);
    const want = try a.alloc(f64, n);
    const tol = try a.alloc(f64, n);
    var q_most: f64 = 0;
    for (0..most) |r| {
        core.dense_ref.row(xv[r * k ..][0..k], wv, k, want, tol);
        for (want, tol, 0..) |w, t, j| {
            const got: f32 = @bitCast(std.mem.readInt(u32, wide[4 * (r * n + j) ..][0..4], .little));
            q_most = @max(q_most, @abs(got - w) / t);
        }
    }
    worst.* = @max(worst.*, q_most);
    try check.expect(q_most <= 1, "dense rows [{d}, {d}]: within float64 tolerance (worst {d:.3})", .{ n, k, q_most });
    for (counts[0 .. counts.len - 1], outs[0 .. counts.len - 1]) |m, o| {
        const got = try a.alloc(u8, m * n * 4);
        try o.download(0, got);
        try dump.output(a, "dense-rows", "product", "rows", &.{ m, n }, .f32, .{ .k = k }, got);
        try check.sameBytes("dense rows: a row count's bits equal the widest call's", got, wide[0 .. m * n * 4]);
    }
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try cuda.dense_rows.DenseRows.load(gpu.d);
    defer g.unload();
    var prng = std.Random.DefaultPrng.init(29);
    var worst: f64 = 0;
    try case(gpu, arena.allocator(), s, &g, prng.random(), 384, 2560, &worst); // Kolibri 1's router
    try case(gpu, arena.allocator(), s, &g, prng.random(), 4103, 2560, &worst); // a head's shape, odd columns
    try case(gpu, arena.allocator(), s, &g, prng.random(), 257, 7168, &worst);
    check.pass("dense rows: within float64 tolerance (worst {d:.3}), every row count's bits equal", .{worst});
}
