const std = @import("std");
const mx = @import("mlx.zig");

pub const Grid = struct {
    height: i32,
    width: i32,
    pub fn count(g: Grid) !usize {
        if (g.height <= 0 or g.width <= 0 or @mod(g.height, 2) != 0 or @mod(g.width, 2) != 0) return error.InvalidImageGrid;
        return std.math.mul(usize, @intCast(@divExact(g.height, 2)), @intCast(@divExact(g.width, 2)));
    }
};
pub const Tokens = struct { image: i32, start: i32, end: i32, video: i32 };
pub const Span = struct { begin: usize, end: usize };
pub const Positions = struct {
    axes: []i32,
    spans: []Span,
    delta: i32,
    pub fn deinit(p: Positions, allocator: std.mem.Allocator) void {
        allocator.free(p.axes);
        allocator.free(p.spans);
    }
    pub fn init(allocator: std.mem.Allocator, tokens: []const i32, grids: []const Grid, ids: Tokens) !Positions {
        if (tokens.len > 262144) return error.ContextLimitExceeded;
        if (std.mem.indexOfScalar(i32, tokens, ids.video) != null) return error.VideoNotSupported;
        const axes = try allocator.alloc(i32, 3 * tokens.len);
        errdefer allocator.free(axes);
        const spans = try allocator.alloc(Span, grids.len);
        errdefer allocator.free(spans);
        var cursor: usize = 0;
        var next: i32 = 0;
        for (grids, spans) |grid, *span| {
            const count = try grid.count();
            const begin = std.mem.indexOfScalarPos(i32, tokens, cursor, ids.image) orelse return error.MissingImageTokens;
            const end = try std.math.add(usize, begin, count);
            if (begin == 0 or tokens[begin - 1] != ids.start or end >= tokens.len or tokens[end] != ids.end) return error.InvalidImageSpan;
            for (tokens[begin..end]) |token| if (token != ids.image) return error.InvalidImageSpan;
            for (cursor..begin) |i| for (0..3) |axis| {
                axes[axis * tokens.len + i] = next + @as(i32, @intCast(i - cursor));
            };
            const base = next + @as(i32, @intCast(begin - cursor));
            const width: usize = @intCast(@divExact(grid.width, 2));
            for (begin..end) |i| {
                axes[i] = base;
                axes[tokens.len + i] = base + @as(i32, @intCast((i - begin) / width));
                axes[2 * tokens.len + i] = base + @as(i32, @intCast((i - begin) % width));
            }
            next = base + @divExact(@max(grid.height, grid.width), 2);
            cursor = end;
            span.* = .{ .begin = begin, .end = end };
        }
        if (std.mem.indexOfScalar(i32, tokens[cursor..], ids.image) != null) return error.UnmatchedImageTokens;
        for (cursor..tokens.len) |i| for (0..3) |axis| {
            axes[axis * tokens.len + i] = next + @as(i32, @intCast(i - cursor));
        };
        return .{ .axes = axes, .spans = spans, .delta = next - @as(i32, @intCast(cursor)) };
    }
    pub fn chunk(p: Positions, s: *mx.Scope, begin: usize, end: usize) !mx.Array {
        const n = p.axes.len / 3;
        if (begin >= end or end > n) return error.InvalidImagePositions;
        var parts: [3]mx.Array = undefined;
        for (0..3) |axis| parts[axis] = try s.ints(p.axes[axis * n + begin .. axis * n + end]);
        return s.stack(&parts, 0);
    }
};

// x is [1, heads, tokens, width], positions is [3, tokens].
pub fn rope(s: *mx.Scope, x: mx.Array, positions: mx.Array) !mx.Array {
    if (mx.shape(x).len != 4 or mx.dim(x, 0) != 1 or mx.dim(x, 3) < 64 or mx.shape(positions).len != 2 or mx.dim(positions, 0) != 3 or mx.dim(positions, 1) != mx.dim(x, 2)) return error.InvalidImagePositions;
    const rows = try s.transpose(x, &.{ 2, 1, 0, 3 });
    var rotated: [3]mx.Array = undefined;
    for (0..3) |axis| rotated[axis] = try s.rope(rows, try s.reshape(try s.slice(positions, 0, @intCast(axis), @intCast(axis + 1)), &.{mx.dim(x, 2)}), 64);
    var axes: [64]i32 = undefined;
    for (0..64) |j| {
        const i = j % 32;
        axes[j] = if (i % 3 == 1 and i < 33) 1 else if (i % 3 == 2 and i < 30) 2 else 0;
    }
    const ids = try s.ints(&axes);
    const one = try s.binary(mx.c.mlx_equal, ids, try s.ints(&.{1}));
    const two = try s.binary(mx.c.mlx_equal, ids, try s.ints(&.{2}));
    var inner = mx.c.mlx_array_new();
    const rc = mx.c.mlx_where(&inner, two, try s.slice(rotated[2], 3, 0, 64), try s.slice(rotated[0], 3, 0, 64), mx.stream);
    _ = try s.result(rc, inner);
    var prefix = mx.c.mlx_array_new();
    const rc2 = mx.c.mlx_where(&prefix, one, try s.slice(rotated[1], 3, 0, 64), inner, mx.stream);
    _ = try s.result(rc2, prefix);
    return s.transpose(try s.cat(&.{ prefix, try s.slice(rows, 3, 64, mx.dim(rows, 3)) }, 3), &.{ 2, 1, 0, 3 });
}

test "rectangular images carry compressed rotary positions into continuation" {
    const ids = Tokens{ .image = 10, .start = 11, .end = 12, .video = 13 };
    const p = try Positions.init(std.testing.allocator, &.{ 5, 11, 10, 10, 10, 10, 10, 10, 12, 6, 11, 10, 12, 7 }, &.{ .{ .height = 4, .width = 6 }, .{ .height = 2, .width = 2 } }, ids);
    defer p.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(i32, -3), p.delta);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 7, 8, 9, 10 }, p.axes[0..14]);
    try std.testing.expectEqualSlices(i32, &.{ 2, 2, 2, 3, 3, 3 }, p.axes[16..22]);
    try std.testing.expectEqualSlices(i32, &.{ 2, 3, 4, 2, 3, 4 }, p.axes[30..36]);
    try std.testing.expectError(error.InvalidImageSpan, Positions.init(std.testing.allocator, &.{ 11, 10, 12 }, &.{.{ .height = 4, .width = 4 }}, ids));
    try std.testing.expectError(error.UnmatchedImageTokens, Positions.init(std.testing.allocator, &.{ 11, 10, 12 }, &.{}, ids));
    try std.testing.expectError(error.InvalidImageGrid, (Grid{ .height = 3, .width = 4 }).count());
    try std.testing.expectError(error.VideoNotSupported, Positions.init(std.testing.allocator, &.{13}, &.{}, ids));
}
