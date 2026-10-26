const std = @import("std");
const sage = @import("sage.zig");

const alloc = std.heap.c_allocator;

fn differ(a: []const u8, b: []const u8) usize {
    var n: usize = 0;
    for (a, b) |x, y| {
        if (x != y) n += 1;
    }
    return n;
}

fn takeRows(r: *sage.Runner, ntok: usize) ![]u8 {
    const n = r.z.s.hidden * 2;
    const buf = try alloc.alloc(u8, ntok * n);
    var ids: [17]i32 = undefined;
    for (0..ntok) |i| ids[i] = @intCast(100 + i);
    try r.use(0);
    try r.reset();
    try r.prompt(ids[0..ntok]);
    for (0..ntok) |i| try r.copyHidden(@intCast(i), buf[i * n ..][0..n]);
    return buf;
}

fn streamOrder(r: *sage.Runner) !void {
    const n = r.z.s.hidden * 2;
    const a0 = try alloc.alloc(u8, n);
    defer alloc.free(a0);
    const a1 = try alloc.alloc(u8, n);
    defer alloc.free(a1);
    const b0 = try alloc.alloc(u8, n);
    defer alloc.free(b0);
    const b1 = try alloc.alloc(u8, n);
    defer alloc.free(b1);
    try r.use(0);
    try r.reset();
    try r.prompt(&.{100});
    try r.use(1);
    try r.reset();
    try r.prompt(&.{200});
    var out: [2]i32 = undefined;
    try r.stepMulti(&.{ 0, 1 }, &.{ 101, 201 }, &out);
    try r.copyHidden(0, a0);
    try r.copyHidden(1, a1);
    try r.use(0);
    try r.reset();
    try r.prompt(&.{100});
    try r.use(1);
    try r.reset();
    try r.prompt(&.{200});
    try r.stepMulti(&.{ 1, 0 }, &.{ 201, 101 }, &out);
    try r.copyHidden(0, b0);
    try r.copyHidden(1, b1);
    const d0 = differ(a0, b1);
    const d1 = differ(a1, b0);
    std.debug.print("stream order: slot 0 {d} bytes differ, slot 1 {d} bytes differ\n", .{ d0, d1 });
    if (d0 != 0 or d1 != 0) return error.RowBytes;
}

pub fn run(r: *sage.Runner) !void {
    r.hold_hidden = true;
    const n = r.z.s.hidden * 2;
    const wide = try takeRows(r, 16);
    defer alloc.free(wide);
    const two = try takeRows(r, 2);
    defer alloc.free(two);
    const two_again = try takeRows(r, 2);
    defer alloc.free(two_again);
    const one = try takeRows(r, 1);
    defer alloc.free(one);
    const chunk = try takeRows(r, 17);
    defer alloc.free(chunk);
    const same = differ(two, two_again);
    const head = differ(two, wide[0 .. 2 * n]);
    std.debug.print("repeat 2 rows: {d} bytes differ\n", .{same});
    std.debug.print("2 rows vs first 2 of 16: {d} bytes differ\n", .{head});
    const one_line = "1 row vs row 0 of 16: {d} bytes differ (gdn_one and down_combine)\n";
    std.debug.print(one_line, .{differ(one, wide[0..n])});
    if (same != 0 or head != 0) return error.RowBytes;
    var ids: [17]i32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(100 + i);
    try r.use(0);
    try r.reset();
    try r.prompt(ids[0..8]);
    const piece = try alloc.alloc(u8, 8 * n);
    defer alloc.free(piece);
    for (0..8) |i| try r.copyHidden(@intCast(i), piece[i * n ..][0..n]);
    try r.prompt(ids[8..17]);
    const tail = try alloc.alloc(u8, 9 * n);
    defer alloc.free(tail);
    for (0..9) |i| try r.copyHidden(@intCast(i), tail[i * n ..][0..n]);
    const piece_line = "prompt 8 vs first 8 of 17: {d} bytes differ (grouped vs tiled)\n";
    const tail_line = "prompt 9 after 8 vs last 9 of 17: {d} bytes differ (grouped vs tiled)\n";
    std.debug.print(piece_line, .{differ(piece, chunk[0 .. 8 * n])});
    std.debug.print(tail_line, .{differ(tail, chunk[8 * n .. 17 * n])});
    try streamOrder(r);
    std.debug.print("row check done\n", .{});
}
