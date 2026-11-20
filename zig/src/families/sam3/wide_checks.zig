const std = @import("std");
const gpu = @import("tensor_metal");
const dense_device = @import("dense_device.zig");

// Single products x * w through the wide layouts: correctly rounded when normal, else within 2^-126 of the product.
pub fn checkFlushedProducts(g: *gpu.Gpu) !void {
    @setFloatMode(.strict);
    const exponents = [_]i32{ 0, -20, -36, -40, -50, -60, -70, -80, -90, -100, -110, -120, -126, -130, -140, -149 };
    const scales = [_]i32{ 0, -6, -10, -20, -30, -36, -40, -50 };
    const rows = 32;
    const cols = 64;
    const depth = 32;
    const x = try g.alloc(rows * depth, .f32);
    defer x.deinit();
    const w = try g.alloc(cols * depth, .f32);
    defer w.deinit();
    const bias = try g.alloc(cols, .f32);
    defer bias.deinit();
    @memset(x.slice(f32, x.count), 0);
    @memset(w.slice(f32, w.count), 0);
    @memset(bias.slice(f32, cols), 0);
    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();
    for (0..rows) |r| {
        const mantissa: f32 = if (r < 16) 1 + @as(f32, @floatFromInt(random.int(u23))) * 0x1p-23 else 1;
        x.slice(f32, x.count)[r * depth] = std.math.ldexp(mantissa, exponents[r % 16]);
    }
    for (0..cols) |c| {
        const mantissa = 1 + @as(f32, @floatFromInt(random.int(u23))) * 0x1p-23;
        w.slice(f32, w.count)[c * depth] = std.math.ldexp(mantissa, scales[c / 8]);
    }
    var flushed: usize = 0;
    for ([_]dense_device.Layout{ .wide, .wide_compensated }) |layout| {
        try g.beginBatch();
        errdefer if (g.batch != null) g.endBatch() catch {};
        const out = try dense_device.run(g, x, w, bias, rows, cols, depth, layout);
        defer out.deinit();
        try g.endBatch();
        for (0..rows) |r| for (0..cols) |c| {
            const exact = @as(f64, x.get(r * depth)) * w.get(c * depth);
            const got = out.get(r * cols + c);
            const normal = @abs(exact) >= 0x1p-126;
            if (normal and got != @as(f32, @floatCast(exact))) return error.SamWideProduct;
            if (!normal and !(@abs(got - exact) < 0x1p-126)) return error.SamWideProduct;
            flushed += @intFromBool(!normal and got != @as(f32, @floatCast(exact)));
        };
    }
    std.debug.print("SAM wide products: correctly rounded down to 2^-126, {d} subnormal products flushed within 2^-126\n", .{flushed});
}
