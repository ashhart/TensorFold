//! Packed DSpark context through overflow, snapshot resume and release of an in-flight hold.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.lfm2_moe;
const c = q.config;

fn tapped(s: *tf.lanes.Stream, taps: mtl.Buffer, width: usize, start: usize, rows: usize) tf.lanes.drafter.Absorb {
    return .{
        .stream = s,
        .features = .{ .buffer = @intFromPtr(taps.id), .offset = start * width, .rows = @intCast(rows), .row_bytes = @intCast(width), .space = .device, .dtype = .bf16 },
        .start = start,
        .follow = &.{},
        .prompt = true,
    };
}

fn direct(d: *q.dspark.Drafter, l: *q.dspark.Lane, taps: mtl.Buffer, width: usize, start: usize, rows: usize) !void {
    var at: usize = 0;
    while (at < rows) {
        const n = @min(q.dspark.absorb_rows, rows - at);
        try q.dspark_block.absorbNow(d, &.{.{ .lane = l, .src = .{ .buf = taps, .off = (start + at) * width }, .rows = n, .p0 = start + at }});
        at += n;
    }
}

fn equalContext(a: *const q.dspark.Lane, b: *const q.dspark.Lane, layers: usize, rows: usize) bool {
    const bytes = rows * c.head_dim * 2;
    for (0..layers) |i| for ([_]bool{ false, true }) |values| {
        const x = a.kv(i, values);
        const y = b.kv(i, values);
        for (0..c.kv_heads) |h| {
            const xoff = x.off + h * a.cap * c.head_dim * 2;
            const yoff = y.off + h * b.cap * c.head_dim * 2;
            if (!std.mem.eql(u8, (x.buf.contents() + xoff)[0..bytes], (y.buf.contents() + yoff)[0..bytes])) return false;
        }
    };
    return true;
}

pub fn run(gpa: std.mem.Allocator, d: *q.dspark.Drafter, taps: mtl.Buffer, ids: []const u32, ctx: usize) !void {
    if (ctx < 2) return;
    const grid = d.holds;
    const grid_len = d.holds_len;
    const spent = d.hold_spent_ms;
    const step_ms = d.step_ms;
    const hold_ms = d.hold_ms;
    defer {
        d.holds = grid;
        d.holds_len = grid_len;
        d.hold_spent_ms = spent;
        d.step_ms = step_ms;
        d.hold_ms = hold_ms;
    }
    var first = try tf.lanes.Stream.init(gpa, .{ .id = "context-first", .prompt = ids, .max_new = 1 });
    defer first.deinit(gpa);
    var resumed = try tf.lanes.Stream.init(gpa, .{ .id = "context-resumed", .prompt = ids, .max_new = 1 });
    defer resumed.deinit(gpa);
    const dr = d.drafter();
    try dr.open(&first);
    var first_open = true;
    defer if (first_open) dr.release(&first);
    try dr.open(&resumed);
    defer dr.release(&resumed);
    const original = d.lanes.get(&first) orelse return error.LaneNotOpened;
    const restored = d.lanes.get(&resumed) orelse return error.LaneNotOpened;
    const reference = try d.newLane(ctx);
    defer d.dropLane(reference);
    const reference2 = try d.newLane(ctx);
    defer d.dropLane(reference2);
    const width = d.tapBytes();
    // On long contexts a full stage followed by one row exercises the overflow path before the snapshot.
    const prefix = if (ctx > q.dspark.stage_rows + 1) q.dspark.stage_rows + 1 else @max(1, ctx / 2);
    const first_part = @min(prefix, q.dspark.stage_rows);
    try dr.absorb(&.{tapped(&first, taps, width, 0, first_part)});
    if (prefix > first_part) try dr.absorb(&.{tapped(&first, taps, width, first_part, prefix - first_part)});
    try direct(d, reference, taps, width, 0, prefix);
    try direct(d, reference2, taps, width, 0, prefix);
    var snap: q.snapshot.Snap = .{ .at = @intCast(prefix), .bytes = try gpa.alloc(u8, 0) };
    defer gpa.free(snap.bytes);
    defer if (snap.draft) |bytes| gpa.free(bytes);
    try d.keepLane(&first, &snap);
    if (snap.draft == null or original.end != prefix or original.staged != 0) return error.SnapshotMissesContext;
    dr.release(&first);
    first_open = false;
    if (d.lanes.contains(&first)) return error.SnapshotOwnerNotReleased;
    try dr.open(&first);
    first_open = true;
    first.cached = @intCast(prefix);
    first.reuse.saved = @ptrCast(&snap);
    resumed.cached = @intCast(prefix);
    resumed.reuse.saved = @ptrCast(&snap);
    try dr.absorb(&.{tapped(&first, taps, width, prefix, ctx - prefix)});
    try dr.absorb(&.{tapped(&resumed, taps, width, prefix, ctx - prefix)});
    try direct(d, reference, taps, width, prefix, ctx - prefix);
    try direct(d, reference2, taps, width, prefix, ctx - prefix);
    reference.end = ctx;
    reference2.end = ctx;
    var completed: q.snapshot.Snap = .{ .at = @intCast(ctx), .bytes = try gpa.alloc(u8, 0) };
    defer gpa.free(completed.bytes);
    defer if (completed.draft) |bytes| gpa.free(bytes);
    const other = d.lanes.get(&first) orelse return error.LaneNotOpened;
    const other_end = other.end;
    const other_staged = other.staged;
    try d.keepLane(&resumed, &completed);
    if (completed.draft == null or restored.end != ctx or restored.staged != 0) return error.ResumedSnapshotMissesContext;
    if (other.end != other_end or other.staged != other_staged) return error.SnapshotAbsorbedAnotherLane;
    if (!equalContext(reference2, restored, d.config.layers, ctx)) return error.SnapshotResumeContextDiffers;
    const depth = d.config.block;
    _ = try q.dspark_block.hold(d, &.{
        .{ .lane = reference, .pending = ids[ctx], .anchor = ctx, .depth = depth },
        .{ .lane = reference2, .pending = ids[ctx], .anchor = ctx, .depth = depth },
    }, null);
    var expected: [16]u32 = undefined;
    @memcpy(expected[0..depth], reference2.held[0..depth]);
    try dr.hold(&.{
        .{ .stream = &first, .pending = ids[ctx], .position = ctx + 1, .depth = depth },
        .{ .stream = &resumed, .pending = ids[ctx], .position = ctx + 1, .depth = depth },
    });
    if (d.pending == null) return error.LifecycleHoldNotPending;
    dr.release(&first); // settles the shared hold before releasing lane storage
    first_open = false;
    if (d.pending != null or d.pass_len != 0 or d.lanes.contains(&first)) return error.CancelledLaneStillHeld;
    var got: [16]u32 = undefined;
    try dr.held(&.{&resumed}, &.{got[0..depth]});
    if (!std.mem.eql(u32, expected[0..depth], got[0..depth])) return error.SnapshotResumePicksDiffer;
    if (!equalContext(reference, restored, d.config.layers, ctx)) return error.SnapshotResumeContextDiffers;
}
