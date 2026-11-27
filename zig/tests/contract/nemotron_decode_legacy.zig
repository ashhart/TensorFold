//! Diagnostic baseline: the original whole-window Q pack, partial_min(rows,16), merge and output pack.
const std = @import("std");
fn bind(e: anytype, r: anytype, index: usize) void {
    e.buf(r.buf, r.off, index);
}
pub fn encode(e: anytype, p: anytype, c: anytype) void {
    const hd = c.head_dim;
    const group = c.heads / c.kv_heads;
    const rows = c.rows;
    std.debug.assert(group == 16 and hd == 128 and rows > 0 and c.len + rows <= c.capacity);
    const tiles: usize = @min(rows, 16);
    const keys = c.len + rows;
    const chunks = (keys + 511) / 512;
    const dims = [5]i32{ @intCast(keys), @intCast(chunks), @intCast(rows), 1, @intCast(rows) };
    const strides = [4]i64{ @intCast(c.kv_heads * c.capacity * hd), @intCast(c.capacity * hd), @intCast(hd), 1 };
    e.pipe(p.q);
    e.buf(c.qkv.buf, c.qkv.off + c.first * c.qkv_n * 2, 0);
    bind(e, c.qp, 1);
    e.bytes([2]u32{ @intCast(c.qkv_n), @intCast(group) }, 2);
    e.run(.{ hd, group * rows, c.kv_heads }, .{ hd, 1, 1 });
    e.pipe(p.partial);
    bind(e, c.qp, 0);
    bind(e, c.k, 1);
    e.bytes(strides, 2);
    bind(e, c.v, 3);
    e.bytes(strides, 4);
    e.bytes(@as(f32, @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(hd))))), 5);
    e.bytes(dims, 6);
    bind(e, c.po, 7);
    bind(e, c.pm, 8);
    bind(e, c.pl, 9);
    if (c.table) |table| {
        bind(e, table, 10);
        e.bytes(c.page_bytes, 11);
    }
    e.run(.{ c.kv_heads * 32 * tiles, chunks, (rows + tiles - 1) / tiles }, .{ 32 * tiles, 1, 1 });
    e.pipe(p.merge);
    bind(e, c.po, 0);
    bind(e, c.pm, 1);
    bind(e, c.pl, 2);
    e.bytes(dims, 3);
    bind(e, c.att, 4);
    e.run(.{ c.kv_heads * 32, group * rows, 1 }, .{ 32, 1, 1 });
    e.pipe(p.out);
    bind(e, c.att, 0);
    e.buf(c.out.buf, c.out.off + c.first * c.heads * hd * 2, 1);
    e.run(.{ hd, c.heads, rows }, .{ hd, 1, 1 });
}
