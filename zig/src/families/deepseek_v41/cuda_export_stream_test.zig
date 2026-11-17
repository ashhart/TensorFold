//! A streamed export: parts converted from windows of the pool one by one equal the whole export's.
const std = @import("std");
const g = @import("handoff/geometry.zig");
const plan = @import("handoff/plan.zig");
const ex = @import("handoff/export.zig");
const s = @import("handoff/synth.zig");
const cx = @import("cuda_export.zig");
const cxt = @import("cuda_export_test.zig");

const gpa = std.testing.allocator;

/// Part `i` of `b` converted from only the pages it covers, as the stream's thread reads them back.
fn windowed(b: *cx.Built, i: usize, snap: cxt.Snap, index: ex.IndexFormat) !void {
    const shape = b.shape;
    const p = &b.parts[i];
    const si = shape.geo.sourceIndex(p.spec.layer).?;
    const total = shape.pages(si);
    const end = @min(total, @as(u64, p.spec.first_page) + shape.span(si, p.spec.kind));
    const whole = if (p.spec.kind == .comp) snap.pages[si].comp else snap.pages[si].index;
    const pb = whole.len / total;
    const view: ex.Paged = .{ .split = false, .ranks = &.{whole[p.spec.first_page * pb .. end * pb]}, .first = p.spec.first_page };
    const src: ex.SourceState = .{ .layer = p.spec.layer, .comp = view, .index = view };
    try ex.convert(p, shape, .{ .index_format = index, .sources = &.{src}, .swa = &.{}, .carries = &.{} });
}

fn streamCase(upto: u64, index: ex.IndexFormat) !void {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const ds: u32 = @intCast(@min(upto, g.release.window));
    const w = try s.make(arena, .{ .upto = upto, .index = index, .ds_rows = ds, .seed = upto * 3 + 1 });
    const snap = try cxt.snapshotOf(arena, w);
    const tokens = try arena.alloc(u32, upto);
    for (tokens, 0..) |*x, i| x.* = @intCast(i * 5 + 3);
    const id: [16]u8 = @splat(0x6d);
    const state: cx.State = .{ .pos = upto, .ds_rows = ds, .index_format = index, .pieces = snap.pieces, .pages = snap.pages };
    const whole = try cx.build(gpa, id, tokens, state);
    defer whole.deinit(gpa);

    const b = try cx.skeleton(gpa, id, tokens, cx.shapeOf(upto, ds));
    defer b.deinit(gpa);
    const progress = b.exp.progress.?;
    try std.testing.expectEqual(whole.exp.layers.len, b.exp.layers.len);
    // every other paged part is streamed; the cut finds the rest in the pool read at the cut and the bounded pieces
    for (b.parts, 0..) |p, i| if ((p.spec.kind == .comp or p.spec.kind == .ixk) and i % 2 == 0) {
        try windowed(b, i, snap, index);
        progress.mark(i);
    };
    try std.testing.expect(!progress.whole.load(.acquire));
    try cx.complete(b, state);
    try std.testing.expect(progress.whole.load(.acquire));
    for (whole.exp.sources, b.exp.sources) |a, c| for (a.parts, c.parts) |x, y| try std.testing.expectEqualSlices(u8, x, y);
    for (0..b.parts.len) |i| try std.testing.expect(progress.isReady(i));

    // when every paged part was streamed, the cut needs no pages
    const b2 = try cx.skeleton(gpa, id, tokens, cx.shapeOf(upto, ds));
    defer b2.deinit(gpa);
    for (b2.parts, 0..) |p, i| if (p.spec.kind == .comp or p.spec.kind == .ixk) {
        try windowed(b2, i, snap, index);
        b2.exp.progress.?.mark(i);
    };
    var bare = state;
    bare.pages = &.{};
    try cx.complete(b2, bare);
    for (whole.exp.sources, b2.exp.sources) |a, c| for (a.parts, c.parts) |x, y| try std.testing.expectEqualSlices(u8, x, y);

    // a manifest announced for other rows than the cut holds is refused
    const b3 = try cx.skeleton(gpa, id, tokens, cx.shapeOf(upto, if (ds > 1) ds - 1 else 0));
    defer b3.deinit(gpa);
    try std.testing.expectError(error.BadSnapshot, cx.complete(b3, state));
}

test "streamed parts and the cut's equal the whole export" {
    try streamCase(300, .fp8);
    try streamCase(5000, .fp8);
    try streamCase(4500, .bf16);
}
