//! Native MXFP8: lossless packing, independent FP64 products, strided rows and row-count invariance.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");

fn bf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}

fn tobf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

pub fn run(gpu: check.Gpu) !void {
    try scaleRange(gpu);
    const a = gpu.gpa;
    const n = 70; // a partial tile and padded rows
    const k = 1024; // split-K, both cluster and explicit reduction
    const m = 256; // fused prompt reduction
    const ldx = k + 64;
    const npad = cuda.mxfp8.padded(n);
    var prng = std.Random.DefaultPrng.init(0x8e80_5eed);
    const rng = prng.random();
    const codes = try a.alloc(u8, n * k);
    defer a.free(codes);
    for (codes) |*v| {
        v.* = rng.int(u8);
        if (v.* & 0x7f == 0x7f) v.* ^= 1;
    }
    const scales = try a.alloc(u8, n * (k / 32));
    defer a.free(scales);
    for (scales) |*v| v.* = 121 + rng.uintLessThan(u8, 8);
    // A dedicated row makes UE8M0 byte 0 observable in a normal BF16 result.
    @memset(codes[0..k], 0x38); // E4M3 1
    @memset(scales[0 .. k / 32], 0);
    const packed_codes = try a.alloc(u8, npad * k);
    defer a.free(packed_codes);
    cuda.mxfp8.packCodes(packed_codes, codes, n, k);
    const packed_scales = try a.alloc(u8, npad * (k / 32));
    defer a.free(packed_scales);
    cuda.mxfp8.packScales(packed_scales, scales, n, k);
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, packed_codes);
    defer dw.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, packed_scales);
    defer ds.free();
    const w = cuda.mxfp8.weight(dw.ptr, ds.ptr, n, k);
    const x = try a.alloc(u16, m * ldx);
    defer a.free(x);
    @memset(x, 0x7fc0); // stride padding is poison, not another activation
    for (0..m) |row| {
        for (x[row * ldx ..][0..k]) |*v| v.* = tobf(rng.float(f32) * 2);
    }
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
    defer dx.free();
    var dy = try cuda.DeviceBuffer.alloc(gpu.d, m * n * 2);
    defer dy.free();
    var part = try cuda.DeviceBuffer.alloc(gpu.d, 8 * m * n * 4);
    defer part.free();
    var lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var baseline: ?[]u8 = null;
    defer if (baseline) |b| a.free(b);
    for ([_]bool{ true, false }) |clusters| {
        lane.clusters = clusters and (try gpu.ctx.capability()) >= 90;
        const lin: cuda.qlinear.Linear = .{ .lane = &lane };
        for ([_]usize{ 256, 1, 7, 17, 33 }) |rows| {
            for ([_]bool{ false, true }) |prompt| {
                try dy.fill8(0xff, s.handle);
                const in: cuda.qlinear.Rows = .{ .x = dx.ptr, .ldx = ldx, .part = part.ptr };
                if (prompt) try lin.prompt(s, .{ .mxfp8 = w }, in, dy.ptr, rows) else try lin.decode(s, .{ .mxfp8 = w }, in, dy.ptr, rows);
                try s.synchronize();
                const got = try a.alloc(u8, rows * n * 2);
                defer a.free(got);
                try dy.download(0, got);
                if (baseline) |b| {
                    try check.sameBytes("MXFP8 row-count/dispatch/cluster invariance", got, b[0..got.len]);
                } else baseline = try a.dupe(u8, got);
                for (0..rows) |row| for (0..n) |col| {
                    var ref: f64 = 0;
                    var magnitude: f64 = 0;
                    for (0..k) |i| {
                        const scale = std.math.ldexp(@as(f64, 1), @as(i32, scales[col * (k / 32) + i / 32]) - 127);
                        const v = bf(x[row * ldx + i]) * core.quant.fp8.e4m3(codes[col * k + i]) * scale;
                        ref += v;
                        magnitude += @abs(v);
                    }
                    const actual = bf(std.mem.readInt(u16, got[(row * n + col) * 2 ..][0..2], .little));
                    const tolerance = @abs(ref) / 128 + magnitude / 4096 + 0x1p-133;
                    try check.expect(@abs(actual - ref) <= tolerance, "MXFP8 FP64 row {d}, column {d}: {e} versus {e}, tolerance {e}", .{ row, col, actual, ref, tolerance });
                };
            }
        }
    }
    check.pass("MXFP8: strided rows 1,7,17,33,256 equal across dispatch and reduction paths; FP64 including UE8M0 byte zero", .{});
}

/// Every UE8M0 encoding, through the production matmul: each dot product is exactly one times its scale.
fn scaleRange(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    const n = 256;
    const k = 64;
    const codes = try a.alloc(u8, n * k);
    defer a.free(codes);
    @memset(codes, 0x38);
    var scales: [n * 2]u8 = undefined;
    for (0..n) |r| @memset(scales[r * 2 ..][0..2], @intCast(r));
    const tiled_codes = try a.alloc(u8, n * k);
    defer a.free(tiled_codes);
    cuda.mxfp8.packCodes(tiled_codes, codes, n, k);
    var tiles: [n * 2]u8 = undefined;
    cuda.mxfp8.packScales(&tiles, &scales, n, k);
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, tiled_codes);
    defer dw.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, &tiles);
    defer ds.free();
    var x: [k]u16 = @splat(0x3c80); // 1/64: 64 equal terms sum to one
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&x));
    defer dx.free();
    var dy = try cuda.DeviceBuffer.alloc(gpu.d, n * 2);
    defer dy.free();
    var lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    try lane.matmul(s, dx.ptr, k, 1, cuda.mxfp8.weight(dw.ptr, ds.ptr, n, k), dy.ptr, 0);
    try s.synchronize();
    var got: [n * 2]u8 = undefined;
    try dy.download(0, &got);
    for (0..n) |r| {
        const bits = std.mem.readInt(u16, got[r * 2 ..][0..2], .little);
        if (r == 255) {
            try check.expect(std.math.isNan(bf(bits)), "UE8M0 reserved byte produces NaN", .{});
        } else {
            const want: u16 = if (r == 0) 0x40 else @intCast(r << 7);
            try check.expect(bits == want, "UE8M0 {d}: BF16 bits {x} versus {x}", .{ r, bits, want });
        }
    }
    check.pass("MXFP8: all 256 UE8M0 encodings preserve the specified value, including subnormal and NaN", .{});
}
