//! A round's rows under M-RoPE: each window's rope positions from its slot's images, and the rows image rows replace.
const std = @import("std");
const mrope = @import("core").qwen_vl.rope;
const Ref = @import("projection.zig").Ref;
const Round = @import("round_plan.zig").Round;
const Frame = @import("gpu_frame.zig").Frame;

/// One image's rows in a prompt pass: prompt rows [at, at + n) take rows [row, row + n) of the pass's image rows.
pub const Span = struct { at: u32, n: u32, row: u32 };

/// A prompt pass's images: the stream in `slot` takes `rows` (bf16, the model's width) at its spans.
pub const Images = struct { slot: u32, rows: Ref, spans: []const Span };

/// What a round's forward reads beyond today's: M-RoPE's pair bounds (0: plain), and image rows to take.
pub const Placed = struct { h_end: u32 = 0, w_end: u32 = 0, images: ?Ref = null };

/// The frame's rope planes and take rows for `round`; a round of text-only slots keeps plane 0 as setRows wrote it.
pub fn place(f: *Frame, rope: []const mrope.Layout, images: ?Images, sections: [3]usize, round: *const Round) Placed {
    const cap = f.capacity;
    return placeRows(f.get(.positions).buffer.slice(i32, 3 * cap), f.get(.takes).buffer.slice(i32, cap), cap, rope, images, sections, round);
}

/// `place` on the planes (`cap` rows apart) and takes as slices.
fn placeRows(pos: []i32, takes: []i32, cap: usize, rope: []const mrope.Layout, images: ?Images, sections: [3]usize, round: *const Round) Placed {
    var any = false;
    for (round.slots) |slot| any = any or rope[slot].count > 0;
    if (!any) return .{};
    var grid = false;
    var took = false;
    for (round.slots, round.windows, round.firsts) |slot, w, first| {
        const l = &rope[slot];
        for (0..w.parents.len) |i| {
            const row = first + i;
            const p: u32 = @intCast(round.feed_positions[row]);
            const x = l.position(p);
            pos[row] = x[0];
            pos[cap + row] = x[1];
            pos[2 * cap + row] = x[2];
            grid = grid or x[1] != x[0] or x[2] != x[0];
            takes[row] = -1;
            if (images) |im| if (im.slot == slot) for (im.spans) |s| if (p >= s.at and p < s.at + s.n) {
                takes[row] = @intCast(s.row + (p - s.at));
                took = true;
            };
        }
    }
    var out: Placed = .{ .images = if (took) images.?.rows else null };
    if (grid) out = .{ .h_end = @intCast(3 * sections[1]), .w_end = @intCast(3 * sections[2]), .images = out.images };
    return out;
}

test "a text round keeps its positions; an image chunk gets planes, shifted text and its takes" {
    const ids: mrope.Ids = .{ .start = 50, .image = 51, .end = 52 };
    const prompt = [_]u32{ 7, 50, 51, 51, 51, 51, 51, 51, 52, 9 };
    const Img = struct { at: u32, gh: u32, gw: u32, tokens: u32 };
    var rope: [8]mrope.Layout = @splat(.{});
    rope[1] = try mrope.Layout.init(&prompt, &[_]Img{.{ .at = 2, .gh = 4, .gw = 6, .tokens = 6 }}, ids, 2);
    var f = try fakeFrame(std.testing.allocator);
    defer std.testing.allocator.free(f.pos);
    defer std.testing.allocator.free(f.takes);
    // slot 0 (text) alone: nothing placed, plane 0 stays setRows'
    var text = try Round.init(std.testing.allocator, &.{.{ .slot = 0, .start = 5, .capacity = 64, .ids = &.{ 1, 2 } }}, 4, 2, 100);
    defer text.deinit();
    try std.testing.expectEqual(Placed{}, placeOn(&f, &rope, null, &text));
    // slot 1's prompt chunk [0, 10) with its image rows from row 30 of the pass's rows
    var chunk = try Round.init(std.testing.allocator, &.{.{ .slot = 1, .start = 0, .capacity = 64, .ids = &prompt }}, 4, 2, 100);
    defer chunk.deinit();
    const images: Images = .{ .slot = 1, .rows = undefined, .spans = &.{.{ .at = 2, .n = 6, .row = 30 }} };
    const placed = placeOn(&f, &rope, images, &chunk);
    try std.testing.expectEqual(@as(u32, 33), placed.h_end);
    try std.testing.expectEqual(@as(u32, 30), placed.w_end);
    try std.testing.expect(placed.images != null);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 2, 2, 2, 2, 2, 5, 6 }, f.pos[0..10]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 2, 2, 3, 3, 3, 5, 6 }, f.pos[128..138]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 4, 2, 3, 4, 5, 6 }, f.pos[256..266]);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, 30, 31, 32, 33, 34, 35, -1, -1 }, f.takes[0..10]);
    // a decode round after it: plain rope at the shifted position, nothing taken
    var step = try Round.init(std.testing.allocator, &.{.{ .slot = 1, .start = 12, .capacity = 64, .ids = &.{3} }}, 4, 2, 100);
    defer step.deinit();
    try std.testing.expectEqual(Placed{}, placeOn(&f, &rope, null, &step));
    try std.testing.expectEqual(@as(i32, 9), f.pos[0]);
}

/// Planes and takes in host memory, as a 128-row frame lays them out.
const Fake = struct { pos: []i32, takes: []i32 };

fn fakeFrame(a: std.mem.Allocator) !Fake {
    return .{ .pos = try a.alloc(i32, 3 * 128), .takes = try a.alloc(i32, 128) };
}

/// `place` over host slices: setRows' plane 0, then the same arithmetic.
fn placeOn(f: *Fake, rope: []const mrope.Layout, images: ?Images, round: *const Round) Placed {
    for (round.feed_positions, 0..) |p, i| f.pos[i] = p;
    return placeRows(f.pos, f.takes, 128, rope, images, .{ 11, 11, 10 }, round);
}
