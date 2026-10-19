//! Real probe callbacks without pipeline construction, GPU commands or readback substitution.
const std = @import("std");
const f = @import("tensorfold").flashnext_vision_contract;
const c = @import("contract").spec;
const prep = @import("tests/contract/metal_flashnext.zig");
const pool = @import("tests/contract/metal_flashnext_pool.zig");
const long = @import("tests/contract/flashnext_positions.zig");

test "rotary probe: the sweep accumulator refuses skipped duplicated and incorrectly rounded cases" {
    const cpu = @import("tests/contract/rotary_sweep_cpu.zig");
    const accuracy = f.rotary_accuracy;
    var row: accuracy.Row = .{ .role = .target, .mode = .normal };
    var sample: cpu.Sample = .{ .frequency = 1, .angle = 0, .cosine = 1, .sine = 0, .co = 1, .si = 0 };
    const table = accuracy.domain.table(.normal);
    try std.testing.expectError(error.NoncontiguousRotarySweep, cpu.observe(&row, 1, sample, &table));
    sample.angle = 1;
    try std.testing.expectError(error.UnrecordedPhaseRounding, cpu.observe(&row, 0, sample, &table));
    sample.angle = 0;
    sample.co = 0;
    try std.testing.expectError(error.UnrecordedCoefficientRounding, cpu.observe(&row, 0, sample, &table));
    sample.co = 1;
    try cpu.observe(&row, 0, sample, &table);
    try std.testing.expectEqual(@as(u64, 1), row.cases);
    try std.testing.expectError(error.NoncontiguousRotarySweep, cpu.observe(&row, 0, sample, &table));
}

test "rotary probe: real callbacks refuse an unrecorded deployment before buffer or device access" {
    var low: [1]f64 = undefined;
    var high: [1]f64 = undefined;
    var p: prep.Probe = .{ .a = std.testing.allocator, .rig = undefined, .pipe = undefined, .which = .target };
    try std.testing.expectError(error.UnrecordedRotaryCompiler, p.intervals(&low, &high));
    var q: pool.Probe = .{ .a = std.testing.allocator, .rig = undefined, .pipe = undefined, .which = .pooled };
    try std.testing.expectError(error.UnrecordedRotaryCompiler, q.intervals(&low, &high));
}

test "rotary probe: a one-ulp mutation of the actual bound table refuses before receipt lookup" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const metal = @import("metal");
    metal.allocator = allocator;
    defer metal.allocator = undefined;
    var receipt: f.rotary_accuracy.Receipt = .{ .identity = undefined, .rows = &.{} };
    var p: prep.Probe = .{ .a = allocator, .rig = .{ .a = allocator, .device = .{}, .queue = undefined }, .pipe = undefined, .which = .target, .accuracy = &receipt };
    _ = try p.prepare(.{ .rows = 1, .n = f.output_width, .k = 256 }, .random, 7);
    const word = p.rig.bytes(p.base, f.table_words * 4)[48..][0..4];
    std.mem.writeInt(u32, word, std.mem.readInt(u32, word, .little) + 1, .little);
    var low: [f.output_width]f64 = undefined;
    var high: [f.output_width]f64 = undefined;
    try std.testing.expectError(error.UnrecordedRotaryTable, p.intervals(&low, &high));
    var q: pool.Probe = .{ .a = allocator, .rig = .{ .a = allocator, .device = .{}, .queue = undefined }, .pipe = undefined, .which = .pooled, .accuracy = &receipt };
    _ = try q.prepare(.{ .rows = 1, .n = 128, .k = 128 }, .random, 7);
    const other = q.rig.bytes(q.base, f.table_words * 4)[48..][0..4];
    std.mem.writeInt(u32, other, std.mem.readInt(u32, other, .little) + 1, .little);
    try std.testing.expectError(error.UnrecordedRotaryTable, q.intervals(low[0..128], high[0..128]));
}

test "rotary probe: long positions reach the full margin and pooling stays in the real allocation" {
    for (0..8) |row| {
        for (0..3) |axis| try std.testing.expect(long.token(row, axis) >= 0 and long.token(row, axis) < f.maximum);
    }
    try std.testing.expectEqual(@as(i32, 1048576), long.token(6, 0));
    try std.testing.expectEqual(@as(i32, f.maximum - 1), long.token(7, 0));
    for ([_]usize{ 1, 7, 16, 128 }) |rows| for ([_]f.Which{ .pooled, .absolute }) |which| {
        const p: pool.Probe = .{ .a = std.testing.allocator, .rig = undefined, .pipe = undefined, .which = which, .long_positions = true, .shape = .{ .rows = rows, .n = 128, .k = 128 } };
        try std.testing.expectEqual(f.maximum, 4 * (p.firstBlock() + rows));
        if (rows == 16) try std.testing.expectEqual(@as(usize, 1048576), 4 * p.firstBlock());
    };
}
test "rotary probe: Q K and indexer allowances cover every block with the witness row in its actual layout" {
    const a = std.testing.allocator;
    for ([_]f.Which{ .mrope, .target, .mtp }) |which| for ([_]usize{ 1, 7, 16, 128 }) |rows| {
        if (which == .mtp and rows > 16) continue;
        const host = try a.alloc(u16, rows * f.packed_width);
        defer a.free(host);
        for (host, 0..) |*h, i| h.* = c.toBf16(@as(f32, @floatFromInt((i + 2) % 17)) / 8 - 1);
        const positions = try a.alloc(i32, rows * 3);
        defer a.free(positions);
        for (0..rows) |r| for (0..3) |axis| {
            positions[r * 3 + axis] = @intCast(r * 17);
        };
        var p: prep.Probe = .{ .a = a, .rig = undefined, .pipe = undefined, .which = which, .host = host, .positions = positions, .shape = .{ .rows = rows, .n = f.output_width, .k = 256 } };
        if (which != .mrope) for (0..rows) |r| {
            positions[r] = @intCast(r * 17);
        };
        const allow = try a.alloc(f64, rows * f.output_width);
        defer a.free(allow);
        @memset(allow, std.math.nan(f64));
        try p.conditionalBounds(&.{}, &.{}, allow);
        for (allow) |v| try std.testing.expect(std.math.isFinite(v) and v >= 0);
        if (rows == 128) try std.testing.expect(allow[50 * 24 * 256 + 33] >= 0.000011444091796875);
    };
}
test "rotary probe: both pooled callbacks include the absolute block prefix without widening its guard" {
    const a = std.testing.allocator;
    for ([_]f.Which{ .pooled, .absolute }) |which| for ([_]f64{ 0, 4 }) |factor| {
        const host = try a.alloc(u16, (7 + 128) * 4 * 128);
        defer a.free(host);
        for (host, 0..) |*h, i| h.* = c.toBf16(@as(f32, @floatFromInt((i + 11) % 17)) / 8 - 1);
        var p: pool.Probe = .{ .a = a, .rig = undefined, .pipe = undefined, .which = which, .factor = factor, .host = host, .shape = .{ .rows = 128, .n = 128, .k = 128 } };
        var allow: [128 * 128]f64 = @splat(std.math.nan(f64));
        try p.conditionalBounds(&.{}, &.{}, &allow);
        for (allow) |v| try std.testing.expect(std.math.isFinite(v) and v >= 0);
    };
}
