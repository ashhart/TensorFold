//! Device checks for the native BF16 projection path against exact independent integer dots.
const std = @import("std");
const hip = @import("hip");

fn bf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

fn value(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}

test "HIP BLAS device: bias is added before BF16 output rounding" {
    var r = try selectedRuntime();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try hip.Stream.init(&r);
    defer stream.deinit();
    var b = try hip.blas.Blas.open(stream);
    defer b.close();
    const x = [_]u16{ bf16(256), bf16(1) };
    const w = [_]u16{ bf16(1), bf16(1) };
    const bias = [_]u16{bf16(-256)};
    var dx = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(&x));
    defer dx.free();
    var dw = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(&w));
    defer dw.free();
    var dy = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(&bias));
    defer dy.free();
    try b.projectAdd(dx.base(), dw.base(), dy.base(), 1, 1, 2);
    try stream.synchronize();
    var got: [1]u16 = undefined;
    try dy.download(0, std.mem.sliceAsBytes(&got));
    try std.testing.expectEqual(bf16(1), got[0]);
    try b.project(dx.base(), dw.base(), dy.base(), 1, 1, 2);
    try stream.synchronize();
    try dy.download(0, std.mem.sliceAsBytes(&got));
    try std.testing.expectEqual(bf16(256), got[0]);
}

test "HIP BLAS device: cancellation dots distinguish FP32 accumulation from FP16 and BF16" {
    const a = std.testing.allocator;
    const k = 1536;
    var r = try selectedRuntime();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try hip.Stream.init(&r);
    defer stream.deinit();
    var b = try hip.blas.Blas.open(stream);
    defer b.close();
    const x = try a.alloc(u16, k);
    defer a.free(x);
    const w = try a.alloc(u16, k);
    defer a.free(w);
    @memset(x, bf16(1));
    var half_sum: f16 = 0;
    var brain_sum: f32 = 0;
    for (w, 0..) |*v, j| {
        const f: f32 = if (j < k / 2) 128 else -127;
        v.* = bf16(f);
        half_sum = @floatCast(@as(f32, half_sum) + f);
        brain_sum = @floatCast(value(bf16(brain_sum + f)));
    }
    try std.testing.expect(!std.math.isFinite(half_sum));
    try std.testing.expect(brain_sum != 768);
    var dx = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(x));
    defer dx.free();
    var dw = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(w));
    defer dw.free();
    var dy = try hip.DeviceBuffer.alloc(&r, 2);
    defer dy.free();
    for (0..5) |_| {
        try b.project(dx.base(), dw.base(), dy.base(), 1, 1, k);
        try stream.synchronize();
        var got: [1]u16 = undefined;
        try dy.download(0, std.mem.sliceAsBytes(&got));
        try std.testing.expectEqual(bf16(768), got[0]);
    }
}

test "HIP BLAS device: BF16 projection preserves inputs, guards and exact FP32 sums at served widths" {
    const a = std.testing.allocator;
    var r = try selectedRuntime();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try hip.Stream.init(&r);
    defer stream.deinit();
    var b = try hip.blas.Blas.open(stream);
    defer b.close();
    const cases = [_][3]usize{ .{ 1, 1536, 1536 }, .{ 4, 1024, 1176 }, .{ 7, 9216, 1536 }, .{ 2, 1536, 4608 }, .{ 17, 2048, 1536 }, .{ 4, 1536, 2048 }, .{ 4, 1024, 1536 }, .{ 2, 59392, 1536 } };
    var dimensions: [cases.len]hip.blas_fixed.Dimensions = undefined;
    for (cases, &dimensions) |shape, *d| d.* = .{ .n = shape[1], .k = shape[2] };
    var fixed = try hip.blas_fixed.Fixed.open(stream, &dimensions);
    defer fixed.close();
    for (cases) |shape| {
        const m, const n, const k = shape;
        if ((try ctx.memInfo()).free < (m * k + n * k + m * n + 32) * 2 + (128 << 20)) return error.NoDeviceBudget;
        const x = try a.alloc(u16, m * k);
        defer a.free(x);
        const w = try a.alloc(u16, n * k);
        defer a.free(w);
        const out = try a.alloc(u16, m * n + 32);
        defer a.free(out);
        const want = try a.alloc(u16, m * n);
        defer a.free(want);
        for (x, 0..) |*v, i| v.* = bf16(@floatFromInt(@as(i32, @intCast((i * 3 + i / k) % 7)) - 3));
        for (w, 0..) |*v, i| v.* = bf16(@floatFromInt(@as(i32, @intCast((i * 5 + i / k) % 11)) - 5));
        for (0..m) |row| for (0..n) |col| {
            var sum: f64 = 0;
            for (0..k) |j| sum += value(x[row * k + j]) * value(w[col * k + j]);
            want[row * n + col] = bf16(@floatCast(sum));
        };
        var dx = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(x));
        defer dx.free();
        var dw = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(w));
        defer dw.free();
        var dy = try hip.DeviceBuffer.alloc(&r, out.len * 2);
        defer dy.free();
        try dy.fill8(0xa5);
        for (0..10) |repeat| {
            if (repeat < 5) try b.project(dx.base(), dw.base(), dy.base() + 32, m, n, k) else try fixed.project(dx.base(), dw.base(), dy.base() + 32, m, n, k);
            try stream.synchronize();
            try dy.download(0, std.mem.sliceAsBytes(out));
            try std.testing.expectEqualSlices(u16, want, out[16..][0 .. m * n]);
            for (out[0..16]) |v| try std.testing.expectEqual(@as(u16, 0xa5a5), v);
            for (out[16 + m * n ..]) |v| try std.testing.expectEqual(@as(u16, 0xa5a5), v);
        }
        const actual_x = try a.alloc(u16, x.len);
        defer a.free(actual_x);
        const actual_w = try a.alloc(u16, w.len);
        defer a.free(actual_w);
        try dx.download(0, std.mem.sliceAsBytes(actual_x));
        try dw.download(0, std.mem.sliceAsBytes(actual_w));
        try std.testing.expectEqualSlices(u16, x, actual_x);
        try std.testing.expectEqualSlices(u16, w, actual_w);
    }
}

fn selectedRuntime() !hip.Runtime {
    var r = try hip.Runtime.open();
    errdefer r.close();
    if (try r.deviceCount() != 1) return error.SelectOneHipDevice;
    return r;
}
