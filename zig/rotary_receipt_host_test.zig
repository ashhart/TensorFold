//! Synthetic tables test the receipt protocol and interval math; none is a GPU accuracy receipt.
const std = @import("std");
const f = @import("src/families/flashnext/vision_contract.zig");
const a = f.rotary_accuracy;
const fixture = @import("src/families/flashnext/rotary_fixture.zig");
const Synthetic = struct {
    norms: [5][51]a.Norm,
    rows: [10]a.Row,
    fn init(s: *Synthetic) void {
        for (std.enums.values(a.Role), 0..) |role, ri| {
            for (0..fixture.kinds(role)) |kind| for (0..17) |phase| {
                const x = fixture.make(role, kind, phase);
                const n = &s.norms[ri][kind * 17 + phase];
                n.* = .{ .width = x.width, .weight = @bitCast(x.weight), .epsilon = @bitCast(@as(f32, 1e-5)), .input = x.input };
                const weights: [256]f32 = @splat(x.weight);
                const rotary = a.domain.table(.normal);
                f.normalizedRotation(x.input[0..x.width], @splat(0), weights[0..x.width], 1e-5, &rotary, .f64, .sequential, n.output[0..x.width], null);
            };
            for (std.enums.values(a.Mode), 0..) |mode, mi| {
                const row = &s.rows[ri * 2 + mi];
                row.* = .{ .role = role, .mode = mode, .cases = a.domain.count(role), .normalization = s.norms[ri][0 .. fixture.kinds(role) * 17], .cosine = .{ .absolute = 0x1p-22, .ulps = 4 }, .sine = .{ .absolute = 0x1p-22, .ulps = 4 } };
                row.exp2_cases = if (mode == .normal) row.cases else 0;
                row.axis_cases = if (role == .mrope) .{ a.domain.maximum * 11, a.domain.maximum * 11, a.domain.maximum * 10 } else .{ row.cases, 0, 0 };
                const table = a.domain.table(mode);
                for (&row.table_bits, table) |*bits, value| bits.* = @bitCast(value);
                for (&row.frequency_bits, 0..) |*bits, i| {
                    const exponent: f32 = -(@as(f32, @floatFromInt(i)) / 32) * table[0];
                    const frequency: f32 = if (mode == .yarn) table[3 + i] else @floatCast(@exp2(@as(f64, exponent)));
                    bits.* = @bitCast(frequency);
                }
            }
        }
    }
    fn receipt(s: *Synthetic) a.Receipt {
        return .{ .complete = true, .rows = &s.rows, .identity = .{ .compiler_service = "synthetic", .compiler_framework = "synthetic", .compiler_deployment = @splat('0'), .os_version = "test", .os_build = "test", .device_name = "synthetic GPU", .device_generation = 15, .language = 4, .math_mode = 0, .math_functions = 0, .sources = @splat(@splat('0')), .oracle_sources = @splat('0') } };
    }
};

test "rotary sweep: unrecorded compiler OS source and partial tables refuse before numerical use" {
    var synthetic: Synthetic = undefined;
    synthetic.init();
    var r = synthetic.receipt();
    try r.validate(r.identity);
    var changed = r.identity;
    changed.compiler_service = "new compiler";
    try std.testing.expectError(error.UnrecordedRotaryCompiler, r.validate(changed));
    changed = r.identity;
    changed.os_build = "new build";
    try std.testing.expectError(error.UnrecordedRotaryCompiler, r.validate(changed));
    changed = r.identity;
    changed.sources[0][0] = '1';
    try std.testing.expectError(error.UnrecordedRotaryCompiler, r.validate(changed));
    changed = r.identity;
    changed.oracle_sources[0] = '1';
    try std.testing.expectError(error.UnrecordedRotaryCompiler, r.validate(changed));
    changed = r.identity;
    changed.math_functions = 1;
    try std.testing.expectError(error.UnrecordedRotaryCompiler, r.validate(changed));
    r.complete = false;
    try std.testing.expectError(error.IncompleteRotarySweep, r.validate(r.identity));
    r.complete = true;
    synthetic.rows[0].cases -= 1;
    try std.testing.expectError(error.IncompleteRotarySweep, r.validate(r.identity));
    synthetic.rows[0].cases += 1;
    const original = synthetic.norms[0][1];
    synthetic.norms[0][1] = synthetic.norms[0][0];
    try std.testing.expectError(error.IncompleteRotarySweep, r.validate(r.identity));
    synthetic.norms[0][1] = original;
    try r.validate(r.identity);
}

test "rotary sweep: measured pointwise intervals cover FP32 phase and each contraction while rejecting sign flips" {
    var synthetic: Synthetic = undefined;
    synthetic.init();
    var rejected_signs: usize = 0;
    for (&synthetic.rows) |*row| {
        const x = fixture.make(row.role, 0, 5);
        const weights: [256]f32 = @splat(x.weight);
        var table = a.domain.table(row.mode);
        var intervals: [256]f.rotary_interval.Interval = undefined;
        try f.rotary_interval.normalized(row, x.input[0..x.width], @splat(1048576), weights[0..x.width], 1e-5, &table, intervals[0..x.width]);
        const n = &synthetic.norms[@backingInt(row.role)][5];
        for (0..64) |d| {
            const i = d % 32;
            const frequency: f32 = @bitCast(row.frequency_bits[i]);
            const angle: f32 = 1048576 * frequency;
            const scale: f32 = if (row.mode == .yarn) table[2] else 1;
            const co = @as(f32, @floatCast(@cos(@as(f64, angle)))) * scale;
            const si = @as(f32, @floatCast(@sin(@as(f64, angle)))) * scale;
            const left = fixture.from(n.output[d]);
            const right = fixture.from(n.output[if (d < 32) d + 32 else d - 32]);
            const point = f.rotary_interval.rotation(left, right, @splat(co), @splat(si), d >= 32);
            try std.testing.expect(point.low >= intervals[d].low and point.high <= intervals[d].high);
            const flipped = -point.low;
            rejected_signs += @intFromBool(flipped < intervals[d].low or flipped > intervals[d].high);
        }
        table[12] = @bitCast(@as(u32, @bitCast(table[12])) + 1);
        try std.testing.expectError(error.UnrecordedRotaryTable, f.rotary_interval.normalized(row, x.input[0..x.width], @splat(1048576), weights[0..x.width], 1e-5, &table, intervals[0..x.width]));
    }
    try std.testing.expect(rejected_signs > 0);
}
