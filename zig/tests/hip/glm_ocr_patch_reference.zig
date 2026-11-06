//! Every patch dot in FP64, with the served FP32 sum bound and both BF16 rounding nodes.
const std = @import("std");
const core = @import("core");
const c = @import("glm_ocr").config;
const bf = core.bf16_round;
const a = std.testing.allocator;

fn value(bits: u16) f64 {
    return bf.value(bits);
}

fn finish(dot: f64, bias: f64) f64 {
    const product = bf.round32(@floatCast(dot));
    return bf.round32(product + @as(f32, @floatCast(bias)));
}

pub fn check(input: []const u16, weight: []align(1) const u16, bias: []align(1) const u16, native: []align(1) const u16, reference: []align(1) const u16, control: []align(1) const u16) !void {
    const columns = c.vision_hidden;
    const rows = input.len / c.patch_values;
    const worst_native = try a.alloc(f64, columns);
    defer a.free(worst_native);
    const worst_reference = try a.alloc(f64, columns);
    defer a.free(worst_reference);
    @memset(worst_native, 0);
    @memset(worst_reference, 0);
    var control_failures: usize = 0;
    var ranged: usize = 0;
    const u: f64 = 0x1p-24;
    const gamma = (c.patch_values * u) / (1 - c.patch_values * u);
    for (0..rows) |row| for (0..columns) |col| {
        var dot: f64 = 0;
        var mag: f64 = 0;
        for (0..c.patch_values) |k| {
            const product = value(input[row * c.patch_values + k]) * value(weight[col * c.patch_values + k]);
            dot += product;
            mag += @abs(product);
        }
        const b = value(bias[col]);
        const low = finish(dot - gamma * mag, b);
        const high = finish(dot + gamma * mag, b);
        const at = row * columns + col;
        const got = value(native[at]);
        const want = value(reference[at]);
        const wrong = value(control[at]);
        if (low != high) ranged += 1;
        if (wrong < low or wrong > high) control_failures += 1;
        try std.testing.expect(std.math.isFinite(got) and got >= low and got <= high);
        try std.testing.expect(std.math.isFinite(want) and want >= low and want <= high);
        worst_native[col] = @max(worst_native[col], @abs(got - (dot + b)));
        worst_reference[col] = @max(worst_reference[col], @abs(want - (dot + b)));
    };
    for (worst_native, worst_reference) |got, want| try std.testing.expect(got <= want);
    try std.testing.expect(control_failures > 0);
    std.debug.print("GLM-OCR HIP patch FP64: {d} outputs, {d} ranged; no worse on all {d} columns; fused-bias control fails {d}\n", .{ native.len, ranged, columns, control_failures });
}
