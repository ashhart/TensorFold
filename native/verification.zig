const std = @import("std");
const mx = @import("mlx.zig");
const model = @import("model.zig");
fn equal(s: *mx.Scope, a: mx.Array, b: mx.Array) !void {
    var out = mx.c.mlx_array_new();
    const rc = mx.c.mlx_array_equal(&out, a, b, false, mx.stream);
    out = try s.result(rc, out);
    try mx.eval(out);
    var value: bool = false;
    try mx.check(mx.c.mlx_array_item_bool(&value, out));
    if (!value) return error.NotBitIdentical;
}
fn prefill(m: *model.Model) !void {
    var tokens: [128]i32 = undefined;
    var parents: [128]i32 = undefined;
    var rows: [128]i32 = undefined;
    for (0..128) |i| {
        tokens[i] = @intCast(i % 31 + 1);
        parents[i] = @as(i32, @intCast(i)) - 1;
        rows[i] = @intCast(i);
    }
    // Cross a 512-key attention chunk and a 64-key tile boundary.
    for (0..4) |_| {
        var p = try m.forward(&tokens, &parents);
        defer p.deinit();
        try m.commit(&p, &rows);
    }
    var p = try m.forward(&.{17}, &.{-1});
    defer p.deinit();
    try m.commit(&p, &.{0});
}
pub fn check(m: *model.Model) !void {
    const tokens = [_]i32{ 21, 22, 23, 24, 25, 26, 27, 28 };
    const parents = [_]i32{ -1, 0, 0, 1, 1, 2, 3, 5 };
    const path = [_]i32{ 0, 2, 5, 7 };
    try prefill(m);
    var tree = try m.forward(&tokens, &parents);
    defer tree.deinit();
    try m.commit(&tree, &path);
    var next = try m.forward(&.{29}, &.{-1});
    defer next.deinit();
    try m.commit(&next, &.{0});
    var cached: [64]model.Cache = @splat(.{});
    defer for (&cached) |*c| c.deinit();
    for (m.cache, 0..) |c, i| {
        cached[i].a = try mx.retain(c.a);
        cached[i].b = try mx.retain(c.b);
    }
    m.reset();
    try prefill(m);
    for (path) |row| {
        var serial = try m.forward(&.{tokens[@intCast(row)]}, &.{-1});
        defer serial.deinit();
        const expected = try serial.scope.slice(tree.logits, 1, row, row + 1);
        try equal(&serial.scope, expected, serial.logits);
        try m.commit(&serial, &.{0});
    }
    var serial = try m.forward(&.{29}, &.{-1});
    defer serial.deinit();
    try equal(&serial.scope, next.logits, serial.logits);
    try m.commit(&serial, &.{0});
    for (m.cache, 0..) |c, i| {
        try equal(&serial.scope, cached[i].a, c.a);
        try equal(&serial.scope, cached[i].b, c.b);
    }
    std.debug.print("PASS: tree rows, partial commit, continuation logits and all 64 caches equal serial execution bit for bit at 513-token context.\n", .{});
}
