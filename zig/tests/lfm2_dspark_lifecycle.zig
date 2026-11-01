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

/// Binds a stream's stage, writes rows there as a forward's kernels do, and absorbs them in place.
fn capture(d: *q.dspark.Drafter, s: *tf.lanes.Stream, history: mtl.Buffer, width: usize, start: usize, rows: usize) !q.backend.Stage {
    const src = d.stageSource();
    const stage = (try src.bind(src.ptr, s, start, rows)) orelse return error.StageRefused;
    @memcpy((stage.place.buf.contents() + stage.place.off)[0 .. rows * width], (history.contents() + start * width)[0 .. rows * width]);
    const f: tf.lanes.backend.Features = .{ .buffer = @intFromPtr(stage.place.buf.id), .offset = stage.place.off, .rows = @intCast(rows), .row_bytes = @intCast(width), .space = .device, .dtype = .bf16 };
    try d.drafter().absorb(&.{.{ .stream = s, .features = f, .start = start, .follow = &.{}, .prompt = false }});
    return stage;
}

/// Captured rows stay in the stage without GPU work; a full stage flushes only the rows before a reused anchor.
fn stagedCapture(gpa: std.mem.Allocator, d: *q.dspark.Drafter, taps: mtl.Buffer, ids: []const u32, ctx: usize) !void {
    const rows = q.dspark.stage_rows;
    if (d.options.capacity < rows + 256) return;
    const width = d.tapBytes();
    const history = try d.target.device.buffer((rows + 8) * width, mtl.ResourceOptions.shared);
    defer history.deinit();
    for (0..rows + 8) |row| @memcpy((history.contents() + row * width)[0..width], (taps.contents() + (row % ctx) * width)[0..width]);
    var s = try tf.lanes.Stream.init(gpa, .{ .id = "stage-capture", .prompt = &.{1}, .max_new = rows + 64 });
    defer s.deinit(gpa);
    const dr = d.drafter();
    try dr.open(&s);
    var open = true;
    defer if (open) dr.release(&s);
    const l = d.lanes.get(&s) orelse return error.LaneNotOpened;
    const spent = d.hold_spent_ms;
    var at: usize = 0;
    while (at < rows - 2) : (at += 16) _ = try capture(d, &s, history, width, at, @min(16, rows - 2 - at));
    if (l.end != 0 or l.staged != rows - 2 or d.hold_spent_ms != spent) return error.CaptureNotOwnedInPlace;
    // the next forward starts at the last kept row: only the rows before it are absorbed, the stage then starts over
    const anchor = rows - 3;
    const stage = try capture(d, &s, history, width, anchor, 6);
    if (l.end != anchor or l.staged != 6 or stage.place.off != l.stage_at) return error.OverflowFlushedPastTheAnchor;
    if (!std.mem.eql(u8, (l.mem.contents() + l.stage_at)[0 .. 6 * width], (history.contents() + anchor * width)[0 .. 6 * width])) return error.OverflowLostCapturedRows;
    const reference = try d.newLane(rows + 16);
    defer d.dropLane(reference);
    try direct(d, reference, history, width, 0, anchor);
    if (!equalContext(reference, l, d.config.layers, anchor)) return error.OverflowContextDiffers;
    // a capture after a signalled hold follows it
    try dr.hold(&.{.{ .stream = &s, .pending = ids[0], .position = anchor + 7, .depth = d.config.block }});
    if (d.pending == null) return error.StageHoldNotPending;
    const next = (try d.stageSource().bind(d.stageSource().ptr, &s, anchor + 6, 2)) orelse return error.StageRefused;
    const wait = next.wait orelse return error.CaptureNotAfterPendingHold;
    if (wait.value != d.signalled or l.end != anchor + 6 or l.staged != 0) return error.CaptureWaitsWrongHold;
    // release leaves no destination to the target
    dr.release(&s);
    open = false;
    const src = d.stageSource();
    if (src.buffer(src.ptr, &s) != null or (try src.bind(src.ptr, &s, 0, 1)) != null) return error.StageOutlivesLane;
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
    try stagedCapture(gpa, d, taps, ids, ctx);
}
