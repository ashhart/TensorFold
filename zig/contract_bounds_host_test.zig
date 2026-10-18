//! The actual shape and sum gates must apply a probe's derived absolute allowance.
const std = @import("std");
const con = @import("contract");
const g = con.gates;
const c = con.spec;
const P = struct {
    output_bytes: [4]u8 = @bitCast(@as(f32, 0.125)),
    allowance: f64 = 0.125,
    fn prepare(_: *anyopaque, _: c.Shape, _: g.Fill, _: u64) !bool {
        return true;
    }
    fn run(_: *anyopaque) !void {}
    fn output(ctx: *anyopaque) ![]const u8 {
        const p: *P = @ptrCast(@alignCast(ctx));
        return &p.output_bytes;
    }
    fn inputs(_: *anyopaque) ![]const u8 {
        return &.{};
    }
    fn intact(_: *anyopaque) !bool {
        return true;
    }
    fn overlap(_: *anyopaque) bool {
        return false;
    }
    fn count(_: *anyopaque) usize {
        return 1;
    }
    fn reference(_: *anyopaque, want: []f64, mag: []f64) !void {
        want[0] = 0;
        mag[0] = 0;
    }
    fn simulate(_: *anyopaque, _: c.Accum, _: g.Order, got: []f64) !void {
        got[0] = 0.125;
    }
    fn bounds(ctx: *anyopaque, _: []const f64, _: []const f64, allow: []f64) !void {
        const p: *P = @ptrCast(@alignCast(ctx));
        allow[0] = p.allowance;
    }
    fn depth(_: c.Shape) usize {
        return 1;
    }
    fn kernel(p: *P, v: *const g.VTable) g.Kernel {
        return .{ .ctx = p, .vt = v, .spec = .{ .name = "absolute interval", .accum = .f32, .depth = depth, .out = .f32, .served = &.{.{ .rows = 1, .n = 1, .k = 1 }}, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "an interval hook control" } } };
    }
};
fn table() g.VTable {
    var v: g.VTable = .{ .prepare = P.prepare, .run = P.run, .output = P.output, .inputs = P.inputs, .intact = P.intact, .overlap = P.overlap, .count = P.count, .reference = P.reference, .simulate = P.simulate };
    if (@hasField(g.VTable, "bounds")) v.bounds = P.bounds;
    return v;
}
test "rotary bound: both production gates use the absolute hook and reject finite corruption" {
    g.quiet = true;
    defer g.quiet = false;
    var p: P = .{};
    const v = table();
    try std.testing.expectEqual(g.Verdict.pass, try g.shapes(std.testing.allocator, p.kernel(&v), 0));
    try std.testing.expectEqual(g.Verdict.pass, try g.sums(std.testing.allocator, p.kernel(&v), 0));
    p.output_bytes = @bitCast(std.math.nan(f32));
    try std.testing.expectEqual(g.Verdict.fail, try g.shapes(std.testing.allocator, p.kernel(&v), 0));
    try std.testing.expectEqual(g.Verdict.fail, try g.sums(std.testing.allocator, p.kernel(&v), 0));
    p.output_bytes = @bitCast(@as(f32, 0.25));
    try std.testing.expectEqual(g.Verdict.fail, try g.shapes(std.testing.allocator, p.kernel(&v), 0));
    try std.testing.expectEqual(g.Verdict.fail, try g.sums(std.testing.allocator, p.kernel(&v), 0));
}
test "rotary bound: invalid or nonfinite allowances fail closed" {
    g.quiet = true;
    defer g.quiet = false;
    if (!@hasField(g.VTable, "bounds")) return;
    var p: P = .{};
    const v = table();
    for ([_]f64{ -1, std.math.nan(f64), std.math.inf(f64) }) |bad| {
        p.allowance = bad;
        try std.testing.expectError(error.InvalidErrorBound, g.shapes(std.testing.allocator, p.kernel(&v), 0));
        try std.testing.expectError(error.InvalidErrorBound, g.sums(std.testing.allocator, p.kernel(&v), 0));
    }
}
