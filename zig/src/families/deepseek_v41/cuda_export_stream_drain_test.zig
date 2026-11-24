//! The cut's wait on a failing stream: it never waits on a segment no worker takes, nor converts beside one.
const std = @import("std");
const fabric = @import("fabric");
const stream = @import("cuda_export_stream.zig");

/// A worker holding a part beside the cut, and the cut's drain: the overlap a queue failure must not open.
const Overlap = struct {
    s: *stream.Stream,
    converting: std.atomic.Value(bool) = .init(false), // the worker holds a part's bytes
    release: std.atomic.Value(bool) = .init(false), // the test lets the worker finish
    entered: std.atomic.Value(bool) = .init(false), // the cut thread is at its drain
    done: std.atomic.Value(bool) = .init(false), // the cut's drain returned
    beside: std.atomic.Value(bool) = .init(false), // it returned while the worker still held the parts
    whole: std.atomic.Value(bool) = .init(false), // what the drain answered

    fn convert(o: *Overlap) void {
        o.converting.store(true, .release);
        while (!o.release.load(.acquire)) std.atomic.spinLoopHint();
        o.converting.store(false, .release); // clear the marker before publishing the last segment as done
        o.s.queued.store(0, .release);
    }

    fn cut(o: *Overlap) void {
        o.entered.store(true, .release);
        o.whole.store(o.s.drain(), .release);
        o.beside.store(o.converting.load(.acquire), .release);
        o.done.store(true, .release);
    }
};

/// A worker that fails and ends between the engine's failure check and its append: the race a queue must refuse.
const Ended = struct {
    s: *stream.Stream,
    checked: std.atomic.Value(bool) = .init(false), // the engine passed `progress`'s failure check
    gone: std.atomic.Value(bool) = .init(false), // the worker failed and its cleanup finished

    fn fail(e: *Ended) void {
        while (!e.checked.load(.acquire)) std.atomic.spinLoopHint();
        e.s.broken.store(true, .release); // its conversion failed
        e.s.retire(); // and it ended as `run` ends
        e.gone.store(true, .release);
    }
};

test "a segment queued after its worker failed and ended is refused, so the cut never waits for it" {
    const gpa = std.testing.allocator;
    var st: stream.Stream = .{ .gpa = gpa, .io = std.testing.io, .built = undefined, .kx = undefined, .converted = &.{} };
    defer {
        for (st.jobs.items) |j| gpa.free(j.pages);
        st.jobs.deinit(gpa);
    }
    var e: Ended = .{ .s = &st };
    const worker = try std.Thread.spawn(.{}, Ended.fail, .{&e});
    const passed = !st.broken.load(.acquire); // `progress`'s check, while the worker still runs
    e.checked.store(true, .release);
    while (!e.gone.load(.acquire)) std.atomic.spinLoopHint(); // the worker fails and ends before the append
    worker.join();
    try std.testing.expect(passed);
    const pages = try gpa.dupe(u32, &.{ 0, 1 });
    const queued = st.enqueue(.{ .fence = 0, .pos = 0, .pages = pages });
    if (queued) |_| {} else |_| gpa.free(pages);
    try std.testing.expectError(error.StreamStopped, queued);
    try std.testing.expectEqual(0, st.queued.load(.acquire)); // no job is counted for a worker that has gone
    try std.testing.expect(!st.drain()); // so the cut answers instead of waiting forever
}

test "the cut waits for the segment its worker is converting when a later queue fails" {
    var st: stream.Stream = .{ .gpa = std.testing.allocator, .io = std.testing.io, .built = undefined, .kx = undefined, .converted = &.{} };
    st.queued.store(1, .release); // an earlier segment is in the worker's hands
    st.broken.store(true, .release); // and queueing the next one failed
    var o: Overlap = .{ .s = &st };
    const worker = try std.Thread.spawn(.{}, Overlap.convert, .{&o});
    const cut = try std.Thread.spawn(.{}, Overlap.cut, .{&o});
    while (!o.converting.load(.acquire) or !o.entered.load(.acquire)) std.atomic.spinLoopHint();
    const began = fabric.words.nowNs();
    while (!o.done.load(.acquire) and fabric.words.nowNs() - began < 250 * std.time.ns_per_ms) std.atomic.spinLoopHint();
    const early = o.done.load(.acquire);
    o.release.store(true, .release);
    worker.join();
    cut.join();
    try std.testing.expect(!early); // the cut was still waiting while the worker held the parts
    try std.testing.expect(o.done.load(.acquire));
    try std.testing.expect(!o.beside.load(.acquire)); // and it never returned beside the worker
    try std.testing.expect(!o.whole.load(.acquire)); // a failure: the cut reads the pages the worker left
}
