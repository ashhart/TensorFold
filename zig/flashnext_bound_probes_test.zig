//! Real probe callbacks without pipeline construction, GPU commands or readback substitution.
const std = @import("std");
const f = @import("tensorfold").flashnext_vision_contract;
const c = @import("contract").spec;
const prep = @import("tests/contract/metal_flashnext.zig");
const pool = @import("tests/contract/metal_flashnext_pool.zig");
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
        try p.bounds(&.{}, &.{}, allow);
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
        try p.bounds(&.{}, &.{}, &allow);
        for (allow) |v| try std.testing.expect(std.math.isFinite(v) and v >= 0);
    };
}
