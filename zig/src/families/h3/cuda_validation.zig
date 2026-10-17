//! Scalar and byte-range admission shared by device and host entry points, before any CUDA call or caller read.
const std = @import("std");
pub const Plan = struct {
    count: usize,
    prefix: usize,
    keep: usize,
    own0: usize,
    own1: usize,
    input: [8]usize,
    buffers: [17]usize,
};
pub fn product(ns: []const usize) !usize {
    var n: usize = 1;
    for (ns) |v| n = std.math.mul(usize, n, v) catch return error.BadH3Product;
    if (n > std.math.maxInt(i64)) return error.BadH3Product;
    return n;
}
fn address(ptr: u64, bytes: usize, alignment: usize) !void {
    if (bytes == 0) return;
    if (ptr == 0 or ptr % alignment != 0) return error.BadH3Address;
    _ = std.math.add(u64, ptr, bytes) catch return error.BadH3Address;
}
pub fn read(m: anytype, in: anytype, own0: usize, own1: usize) !Plan {
    const rows: usize = in.rows;
    const tiles: usize = in.tiles;
    if (rows == 0 or tiles == 0 or in.lines == 0) return error.BadH3Dimensions;
    if (m.hidden == 0 or m.inner == 0 or m.heads == 0 or m.mlp == 0 or m.blocks.len == 0) return error.BadH3Dimensions;
    if (try product(&.{ m.heads, 128 }) != m.inner) return error.BadH3Dimensions;
    for ([_]usize{ rows, tiles, in.lines, m.hidden, try product(&.{ 4, m.inner }), try product(&.{ 2, m.mlp }), try product(&.{ tiles, 64 }), m.blocks.len }) |v| {
        if (v > std.math.maxInt(i32)) return error.BadH3Scalar;
    }
    if (m.heads > 65535 or tiles > 65535) return error.BadH3Scalar;
    if (own0 >= own1 or own1 > tiles or rows > try product(&.{ own1 - own0, 64 }) or rows < own1 - own0) return error.BadH3Dimensions;
    if (in.prefix_tiles > tiles or (in.keep != 0 and in.keep > tiles - in.prefix_tiles)) return error.BadH3Keep;
    // CUDA's rotary argument is a half-width within a 128-wide head.
    if (in.rot > 64) return error.BadH3Rotary;
    const count: usize = if (in.blocks == 0) m.blocks.len else in.blocks;
    if (count > m.blocks.len) return error.BadH3Blocks;
    const prefix: usize = if (in.keep == 0) tiles else in.prefix_tiles;
    const keep: usize = in.keep;
    const h = m.hidden;
    const i = m.inner;
    const heads = m.heads;
    const mlp = m.mlp;
    const input = [8]usize{
        try product(&.{ rows, h, 2 }),      try product(&.{ count, in.lines, 6, h, 4 }),                     try product(&.{ rows, 4 }),
        try product(&.{ rows, in.rot, 4 }), try product(&.{ rows, in.rot, 4 }),                              try product(&.{ rows, 4 }),
        try product(&.{ tiles, 4 }),        if (in.attention_out == 0) 0 else try product(&.{ rows, i, 2 }),
    };
    const pointers = [8]u64{ in.x, in.tables, in.line, in.cos, in.sin, in.slot, in.sizes, in.attention_out };
    for (pointers, input, [_]usize{ 2, 4, 4, 4, 4, 4, 4, 2 }) |ptr, n, a| try address(ptr, n, a);
    const padded = try product(&.{ heads, tiles, 64, 128, 2 });
    const means = try product(&.{ heads, tiles, 128, 4 });
    return .{ .count = count, .prefix = prefix, .keep = keep, .own0 = own0, .own1 = own1, .input = input, .buffers = .{
        input[0],                                  input[0],                                                            try product(&.{ rows, 4, i, 2 }), padded,                        padded,                             padded,                          means,                           means,    means,
        try product(&.{ heads, tiles, tiles, 4 }), try product(&.{ heads, @max(tiles - prefix, 1), prefix + keep, 4 }), means,                            try product(&.{ rows, i, 2 }), try product(&.{ rows, 2, mlp, 2 }), try product(&.{ rows, mlp, 2 }), try product(&.{ tiles, 64, 4 }), input[6],
    } };
}
