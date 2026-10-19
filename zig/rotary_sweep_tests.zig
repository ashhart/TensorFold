//! Exhaustive domain coverage and source/deployment refusal without a Metal compiler or GPU.
const std = @import("std");
const f = @import("src/families/flashnext/vision_contract.zig");
const accuracy = @import("src/families/flashnext/rotary_accuracy.zig");
const d = accuracy.domain;

test "rotary sweep: Cartesian enumeration covers every position, frequency, image axis and pooled endpoint" {
    var total: u64 = 0;
    for (std.enums.values(d.Role)) |role| {
        const n = d.count(role);
        try std.testing.expectEqual(@as(u64, if (d.stride(role) == 4) 512 else 2048), n % d.chunk_cases);
        try std.testing.expectEqual(@as(u64, d.maximum / d.stride(role)), n / 32);
        var linear: u64 = 0;
        var axes: [3]usize = @splat(0);
        for (0..32) |i| axes[d.point(role, i).axis] += 1;
        if (role == .mrope) try std.testing.expectEqualSlices(usize, &.{ 11, 11, 10 }, &axes);
        while (linear < n) : (linear += 32) {
            const first = d.point(role, linear);
            const last = d.point(role, linear + 31);
            try std.testing.expectEqual(@as(u32, 0), first.index);
            try std.testing.expectEqual(@as(u32, 31), last.index);
            try std.testing.expectEqual(first.position, last.position);
            try std.testing.expectEqual(linear / 32, first.position / d.stride(role));
        }
        try std.testing.expectEqual(@as(u32, @intCast(d.maximum - d.stride(role))), d.point(role, n - 1).position);
        total += 2 * n;
    }
    try std.testing.expectEqual(@as(u64, 234895360), total);
}

test "rotary sweep: each real function and diagnostic use the identical scalar helper" {
    const helper = @embedFile("kernels/metal/prefill/fn_rotary_scalar.metal");
    for ([_]f.Which{ .mrope, .target, .mtp, .pooled, .absolute }) |role| {
        const text = f.source(role);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, helper));
        try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, text, "tf_rotary_scalar("));
    }
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, @embedFile("kernels/metal/prefill/fn_rotary_sweep.metal"), "tf_rotary_scalar("));
}

test "rotary sweep: every helper call binds its table in the constant address space" {
    const helper = @embedFile("kernels/metal/prefill/fn_rotary_scalar.metal");
    try std.testing.expect(std.mem.containsAtLeast(u8, helper, 1, "const constant float* LOG2BASE)"));
    for ([_]f.Which{ .mrope, .target, .mtp, .pooled, .absolute }) |role| {
        const source = f.source(role);
        const start = std.mem.indexOf(u8, source, f.symbol(role)).?;
        const end = start + std.mem.indexOfScalar(u8, source[start..], '{').?;
        const slot: []const u8 = if (role == .pooled or role == .absolute) "4" else "6";
        var buffer: [80]u8 = undefined;
        const declaration = try std.fmt.bufPrint(&buffer, "const constant float* LOG2BASE [[buffer({s})]]", .{slot});
        try std.testing.expect(std.mem.containsAtLeast(u8, source[start..end], 1, declaration));
    }
    const sweep = @embedFile("kernels/metal/prefill/fn_rotary_sweep.metal");
    try std.testing.expect(std.mem.startsWith(u8, sweep, "[[kernel]] void tf_rotary_sweep(const constant float* table [[buffer(0)]]"));
    try std.testing.expect(std.mem.containsAtLeast(u8, sweep, 1, "tf_rotary_scalar(i, position, table)"));
}

test "rotary sweep: exact rotary table identity rejects a one-ulp mutation" {
    for (std.enums.values(d.Mode)) |mode| {
        var table = d.table(mode);
        try std.testing.expectEqual(mode, try d.modeOf(&table));
        const bits: u32 = @bitCast(table[12]);
        table[12] = @bitCast(bits + 1);
        try std.testing.expectError(error.UnrecordedRotaryTable, d.modeOf(&table));
    }
}

test "rotary sweep: ULP error uses the reference binade and records distinct extrema" {
    try std.testing.expectEqual(@as(f64, 0x1p-24), accuracy.ulp(std.math.nextAfter(f64, 1, 0)));
    try std.testing.expectEqual(@as(f64, 0x1p-23), accuracy.ulp(1));
    try std.testing.expectEqual(@as(f64, 0x1p-149), accuracy.ulp(0x1p-126));
    try std.testing.expectEqual(@as(f64, 0x1p-149), accuracy.ulp(0));
    var maximum: accuracy.Maximum = .{};
    try maximum.observe(7, 1 + 0x1p-23, 1);
    try std.testing.expectEqual(@as(f64, 1), maximum.ulps);
    try maximum.observe(8, 0x1p-30 + 0x1p-52, 0x1p-30);
    try std.testing.expectEqual(@as(f64, 2), maximum.ulps);
    try std.testing.expectEqual(@as(u64, 7), maximum.absolute_at.linear);
    try std.testing.expectEqual(@as(u64, 8), maximum.ulps_at.linear);
    try std.testing.expectError(error.NonfiniteRotarySweep, maximum.observe(0, std.math.nan(f32), 1));
}
