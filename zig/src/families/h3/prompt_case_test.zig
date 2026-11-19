//! The prompt stage's case writer against the layout tf-h3-image-prompt wrote before it moved to the host.
const std = @import("std");
const fixture = @import("checkpoint_fixture");
const noise = @import("moss_noise");
const picture = @import("picture.zig");
const rules = @import("picture_prompt.zig");
const case = @import("prompt_case.zig");

/// One id per byte, below the vision marks.
const Bytes = struct {
    pub fn encode(_: Bytes, a: std.mem.Allocator, text: []const u8) ![]u32 {
        const out = try a.alloc(u32, text.len);
        for (out, text) |*o, c| o.* = c;
        return out;
    }
};

/// The case the Metal tool built before the writer moved: the same tensors in the same order, the fixture's own image.
fn was(a: std.mem.Allocator, hidden: usize, p: rules.Presentation, text: []const u16, request: case.Request, extra: []const fixture.Entry) ![]u8 {
    const canvas = try picture.canvas(request.size.width, request.size.height, request.min_area);
    const start = try picture.startRows(a, try picture.gridOf(canvas), request.seed, &noise.randn);
    const dims = try a.dupe(i32, &.{ @intCast(canvas.width), @intCast(canvas.height), @intCast(request.size.width), @intCast(request.size.height) });
    const ids = try a.alloc(i32, p.ids.len);
    for (ids, p.ids) |*o, id| o.* = @intCast(id);
    const T = p.ids.len;
    var entries: std.ArrayList(fixture.Entry) = .empty;
    try entries.appendSlice(a, &.{
        .{ .name = "canvas", .dtype = "I32", .shape = &.{2}, .bytes = std.mem.sliceAsBytes(dims[0..2]) },
        .{ .name = "requested", .dtype = "I32", .shape = &.{2}, .bytes = std.mem.sliceAsBytes(dims[2..4]) },
        .{ .name = "noise", .dtype = "F32", .shape = try a.dupe(usize, &.{ start.len / picture.row_width, picture.row_width }), .bytes = std.mem.sliceAsBytes(start) },
    });
    try entries.appendSlice(a, extra);
    try entries.appendSlice(a, &.{
        .{ .name = "text_tags", .dtype = "I32", .shape = &.{T}, .bytes = std.mem.sliceAsBytes(p.tags) },
        .{ .name = "ids", .dtype = "I32", .shape = &.{T}, .bytes = std.mem.sliceAsBytes(ids) },
        .{ .name = "text", .dtype = "BF16", .shape = &.{ T, hidden }, .bytes = std.mem.sliceAsBytes(text) },
    });
    const plain = try fixture.image(a, entries.items);
    const header: usize = @intCast(std.mem.readInt(u64, plain[0..8], .little));
    const pad = (16 - (8 + header) % 16) % 16;
    const out = try a.alloc(u8, plain.len + pad);
    std.mem.writeInt(u64, out[0..8], header + pad, .little);
    @memcpy(out[8..][0..header], plain[8..][0..header]);
    @memset(out[8 + header ..][0..pad], ' ');
    @memcpy(out[8 + header + pad ..], plain[8 + header ..]);
    return out;
}

test "the case file is byte for byte the Metal tool's: canvas, noise, anchors, tags, ids, text" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const marks: rules.Marks = .{ .start = 90, .image = 91, .end = 92 };
    const p = try rules.present(a, Bytes{}, marks, &.{}, "a tiny prompt");
    const hidden = 8;
    const text = try a.alloc(u16, p.ids.len * hidden);
    for (text, 0..) |*t, i| t.* = @intCast(0x3800 + i % 5);
    const request: case.Request = .{ .size = .{ .width = 64, .height = 64 }, .seed = 42, .min_area = 64 * 64 };
    const grids = [_]i32{ 4, 4 };
    const extra = [_]fixture.Entry{.{ .name = "anchors", .dtype = "I32", .shape = &.{ 1, 2 }, .bytes = std.mem.sliceAsBytes(&grids) }};
    try std.testing.expectEqualSlices(u8, try was(a, hidden, p, text, request, &extra), try case.write(a, hidden, p, text, request, &noise.randn, &extra));
}

test "a WIDTHxHEIGHT size parses; anything else is refused" {
    try std.testing.expectEqual(picture.Size{ .width = 2048, .height = 1536 }, try case.size("2048x1536"));
    try std.testing.expectError(error.BadSize, case.size("2048"));
    try std.testing.expectError(error.InvalidCharacter, case.size("axb"));
}
