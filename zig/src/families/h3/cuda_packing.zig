//! A packing's inverse map is refreshed on every call, including equal dimensions and reused slot addresses.
const std = @import("std");
pub fn read(m: anytype, in: anytype, p: anytype) ![]i32 {
    const map = try m.gpa.alloc(i32, @as(usize, in.tiles) * 64);
    errdefer m.gpa.free(map);
    @memset(map, -1);
    const slots = try m.gpa.alloc(i32, in.rows);
    defer m.gpa.free(slots);
    const sizes = try m.gpa.alloc(i32, in.tiles);
    defer m.gpa.free(sizes);
    const lines = try m.gpa.alloc(i32, in.rows);
    defer m.gpa.free(lines);
    try m.driver.check(m.driver.api.cuMemcpyDtoH_v2(slots.ptr, in.slot, p.input[5]), "cuMemcpyDtoH slots");
    try m.driver.check(m.driver.api.cuMemcpyDtoH_v2(sizes.ptr, in.sizes, p.input[6]), "cuMemcpyDtoH sizes");
    try m.driver.check(m.driver.api.cuMemcpyDtoH_v2(lines.ptr, in.line, p.input[2]), "cuMemcpyDtoH lines");
    for (sizes) |n| if (n <= 0 or n > 64) return error.BadH3Packing;
    var total: usize = 0;
    for (sizes[p.own0..p.own1]) |n| total += @intCast(n);
    if (total != in.rows) return error.BadH3Packing;
    for (slots, lines, 0..) |at, line, row| {
        if (line < 0 or line >= in.lines) return error.BadH3Line;
        if (at < 0 or at >= map.len) return error.BadH3Packing;
        const slot: usize = @intCast(at);
        const tile = slot / 64;
        if (tile < p.own0 or tile >= p.own1 or slot % 64 >= sizes[tile] or map[slot] != -1) return error.BadH3Packing;
        map[slot] = @intCast(row);
    }
    return map;
}
