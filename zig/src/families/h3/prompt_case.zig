//! The prompt stage's host case file: the command-line request and the picture case's tensors, tf-h3-image's layout.
const std = @import("std");
const picture = @import("picture.zig");
const rules = @import("picture_prompt.zig");
const fixture = @import("core").checkpoint_fixture;

/// A text-to-image request: the size asked for, the seed and the smallest area drawn.
pub const Request = struct { size: picture.Size, seed: u64, min_area: u64 };

/// `WIDTHxHEIGHT` on the command line.
pub fn size(text: []const u8) !picture.Size {
    const x = std.mem.indexOfScalar(u8, text, 'x') orelse return error.BadSize;
    return .{ .width = try std.fmt.parseInt(usize, text[0..x], 10), .height = try std.fmt.parseInt(usize, text[x + 1 ..], 10) };
}

/// A presentation as a case file: the request, size and seed rows, `extra` tensors, ids, text rows; data at 16 bytes.
pub fn write(a: std.mem.Allocator, hidden: usize, p: rules.Presentation, text: []const u16, request: ?Request, draw: *const fn (u64, []f32) void, extra: []const fixture.Entry) ![]u8 {
    const T = p.ids.len;
    const ids = try a.alloc(i32, T);
    for (ids, p.ids) |*o, id| o.* = @intCast(id);
    var entries: std.ArrayList(fixture.Entry) = .empty;
    if (request) |r| {
        // the canvas and the seed's starting rows, ahead of the two-byte text so every tensor stays aligned
        const canvas = try picture.canvas(r.size.width, r.size.height, r.min_area);
        const grid = try picture.gridOf(canvas);
        const start = try picture.startRows(a, grid, r.seed, draw);
        const dims = try a.dupe(i32, &.{ @intCast(canvas.width), @intCast(canvas.height), @intCast(r.size.width), @intCast(r.size.height) });
        try entries.appendSlice(a, &.{
            .{ .name = "canvas", .dtype = "I32", .shape = &.{2}, .bytes = std.mem.sliceAsBytes(dims[0..2]) },
            .{ .name = "requested", .dtype = "I32", .shape = &.{2}, .bytes = std.mem.sliceAsBytes(dims[2..4]) },
            .{ .name = "noise", .dtype = "F32", .shape = try a.dupe(usize, &.{ start.len / picture.row_width, picture.row_width }), .bytes = std.mem.sliceAsBytes(start) },
        });
    }
    try entries.appendSlice(a, extra);
    try entries.appendSlice(a, &.{
        .{ .name = "text_tags", .dtype = "I32", .shape = &.{T}, .bytes = std.mem.sliceAsBytes(p.tags) },
        .{ .name = "ids", .dtype = "I32", .shape = &.{T}, .bytes = std.mem.sliceAsBytes(ids) },
        .{ .name = "text", .dtype = "BF16", .shape = &.{ T, hidden }, .bytes = std.mem.sliceAsBytes(text) },
    });
    return fixture.aligned(a, entries.items);
}
