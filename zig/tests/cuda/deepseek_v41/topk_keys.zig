//! topk_keys.cu against a host sort (value descending, ties by lower column) on fp32 and bf16-valued logits.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");

const Gpu = check.Gpu;

/// A rank's vocabulary columns at world 4; rows: fp32 noise, apart below bf16's bits, bf16 ties, flat bands.
const C: usize = 32384;
const rows: usize = 8;

fn fill(rnd: std.Random, out: []f32) void {
    for (0..rows) |r| for (out[r * C ..][0..C], 0..) |*v, c| {
        const x: f32 = switch (r % 4) {
            0 => @floatCast(rnd.floatNorm(f64) * 3.0),
            1 => 20.0 + @as(f32, @floatFromInt((c * 7919) % 4096)) * 0x1p-18,
            2 => @floatCast(@floor(rnd.floatNorm(f64) * 4.0) / 2.0),
            else => if (c % 3 == 0) 1.5 else -2.0,
        };
        v.* = if (r % 4 >= 2) @bitCast(@as(u32, @bitCast(x)) & 0xFFFF_0000) else x;
    };
}

const Pick = struct { v: f32, c: u32 };

fn before(_: void, a: Pick, b: Pick) bool {
    return a.v > b.v or (a.v == b.v and a.c < b.c);
}

pub fn run(gpu: Gpu, k: *const dsv41.Kernels) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const o = k.others(stream);
    const a = gpu.gpa;
    var prng = std.Random.DefaultPrng.init(0x70b6_4e75);
    const lg = try a.alloc(f32, rows * C);
    defer a.free(lg);
    fill(prng.random(), lg);
    var d_lg = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(lg));
    defer d_lg.free();
    const sorted = try a.alloc(Pick, C);
    defer a.free(sorted);
    for ([_]usize{ 1, 50, 1024 }) |kk| {
        var d_vals = try cuda.DeviceBuffer.alloc(gpu.d, rows * kk * 4);
        defer d_vals.free();
        var d_cols = try cuda.DeviceBuffer.alloc(gpu.d, rows * kk * 8);
        defer d_cols.free();
        try o.topKeys(d_lg.ptr, C, rows, C, kk, d_vals.ptr, d_cols.ptr);
        try stream.synchronize();
        const vals = try check.download(gpu, d_vals);
        defer a.free(vals);
        const cols = try check.download(gpu, d_cols);
        defer a.free(cols);
        const gv = std.mem.bytesAsSlice(f32, vals);
        const gc = std.mem.bytesAsSlice(i64, cols);
        for (0..rows) |r| {
            for (sorted, lg[r * C ..][0..C], 0..) |*p, v, c| p.* = .{ .v = v, .c = @intCast(c) };
            std.sort.pdq(Pick, sorted, {}, before);
            for (0..kk) |i| {
                const want = sorted[i];
                try check.expect(gc[r * kk + i] == want.c and @as(u32, @bitCast(gv[r * kk + i])) == @as(u32, @bitCast(want.v)), "topk_keys k {d} row {d} (kind {d}) place {d}: column {d} value {d}, want column {d} value {d}", .{ kk, r, r % 4, i, gc[r * kk + i], gv[r * kk + i], want.c, want.v });
            }
        }
        check.pass("topk_keys k {d}: {d} rows of {d} columns (fp32, sub-bf16 near ties, bf16 ties, flat bands) equal the host sort", .{ kk, rows, C });
    }
}
