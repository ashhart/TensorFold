//! The cut's wait on a failing stream: a segment's failed queue must not let the cut convert beside its worker.
const std = @import("std");
const fabric = @import("fabric");
const stream = @import("cuda_export_stream.zig");

/// A worker holding a part beside the cut, and the cut's drain: the overlap a queue failure must not open.
const Overlap = struct {
    s: *stream.Stream,
    converting: std.atomic.Value(bool) = .init(false), // the worker holds a part's bytes
    release: std.atomic.Value(bool) = .init(false), // the test lets the worker finish
    done: std.atomic.Value(bool) = .init(false), // the cut's drain returned
    beside: std.atomic.Value(bool) = .init(false), // it returned while the worker still held the parts
    whole: std.atomic.Value(bool) = .init(false), // what the drain answered

    fn convert(o: *Overlap) void {
        o.converting.store(true, .release);
        while (!o.release.load(.acquire)) std.atomic.spinLoopHint();
        o.s.queued.store(0, .release); // its last segment done
        o.converting.store(false, .release);
    }

    fn cut(o: *Overlap) void {
        o.whole.store(o.s.drain(), .release);
        o.beside.store(o.converting.load(.acquire), .release);
        o.done.store(true, .release);
    }
};

test "the cut waits for the segment its worker is converting when a later queue fails" {
    var st: stream.Stream = .{ .gpa = std.testing.allocator, .io = std.testing.io, .built = undefined, .kx = undefined, .converted = &.{} };
    st.queued.store(1, .release); // an earlier segment is in the worker's hands
    st.broken.store(true, .release); // and queueing the next one failed
    var o: Overlap = .{ .s = &st };
    const worker = try std.Thread.spawn(.{}, Overlap.convert, .{&o});
    const cut = try std.Thread.spawn(.{}, Overlap.cut, .{&o});
    while (!o.converting.load(.acquire)) std.atomic.spinLoopHint();
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
