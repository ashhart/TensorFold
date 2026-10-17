const std = @import("std");
const t = std.testing;
const source = @import("copy_source.zig");
const graph = @import("graph.zig");
const Stream = @import("stream.zig").Stream;
const Fake = @import("copy_source_extra_fake.zig").Fake;
test "synchronous temporary sources refuse capture through real graph wrappers" {
    var d = Fake.init();
    source.setCurrent(@ptrFromInt(51));
    const s: Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    try graph.beginCapture(s, .thread_local);
    Fake.fail = false;
    const got: ?anyerror = if (source.synchronousCopy(&d, 256, &.{3}, t.allocator)) |_| null else |e| e;
    var g = try graph.endCapture(s);
    g.deinit();
    try source.synchronizeContext(&d);
    try t.expectEqual(@as(?anyerror, error.UnownedCaptureSource), got);
}
test "ending a second context capture cannot unguard the first capture" {
    var d = Fake.init();
    const a: Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    const b: Stream = .{ .d = &d, .handle = @ptrFromInt(2) };
    source.setCurrent(@ptrFromInt(51));
    try graph.beginCapture(a, .thread_local);
    source.setCurrent(@ptrFromInt(52));
    try graph.beginCapture(b, .thread_local);
    var gb = try graph.endCapture(b);
    gb.deinit();
    source.setCurrent(@ptrFromInt(51));
    const got: ?anyerror = if (source.enqueue(&d, 256, &.{4}, a)) |_| null else |e| e;
    var ga = try graph.endCapture(a);
    ga.deinit();
    Fake.fail = false;
    try source.synchronizeContext(&d);
    try t.expectEqual(@as(?anyerror, error.UnownedCaptureSource), got);
}
test "stable Buffer survives capture, two marked replays, graph disposal and failed completion" {
    for (0..3) |failure| {
        var d = Fake.init();
        source.setCurrent(@ptrFromInt(53));
        const s: Stream = .{ .d = &d, .handle = @ptrFromInt(3) };
        var b = try source.Buffer.alloc(&d, 16);
        @memset(b.bytes, 0x41);
        try graph.beginCapture(s, .thread_local);
        Fake.fail_event = failure == 1;
        Fake.fail_record = failure == 2;
        if (failure == 0) try b.copyTo(.{ .d = &d, .ptr = 256, .len = 16 }, 0, b.bytes, s) else try t.expectError(error.NotReady, b.copyTo(.{ .d = &d, .ptr = 256, .len = 16 }, 0, b.bytes, s));
        var g = try graph.endCapture(s);
        var exec = try g.instantiate();
        try b.mark(s);
        try exec.launchOn(s);
        Fake.fail = false;
        try b.wait();
        try t.expectEqual(@as(u8, 0x41), Fake.result[0]);
        @memset(b.bytes, 0x42);
        try b.mark(s);
        try exec.launchOn(s);
        Fake.fail = true;
        try t.expectError(error.NotReady, b.wait());
        try t.expect(Fake.pending.?.ptr == b.bytes.ptr);
        exec.deinit();
        g.deinit();
        b.free();
        try t.expectEqual(@as(usize, 1), source.retained());
        try t.expectEqual(@as(usize, 0), Fake.early);
        Fake.fail = false;
        try source.synchronizeContext(&d);
        try t.expectEqual(@as(u8, 0x42), Fake.result[0]);
        try t.expectEqual(@as(usize, 2), Fake.replays);
        try t.expectEqual(@as(usize, 0), source.retained());
    }
}

test "capture start rollback and failed end preserve the temporary-source guard" {
    var d = Fake.init();
    source.setCurrent(@ptrFromInt(54));
    const s: Stream = .{ .d = &d, .handle = @ptrFromInt(4) };
    Fake.fail_begin = true;
    try t.expectError(error.NotReady, graph.beginCapture(s, .thread_local));
    Fake.fail = false;
    try source.synchronousCopy(&d, 256, &.{1}, t.allocator);
    Fake.fail_begin = false;
    try graph.beginCapture(s, .thread_local);
    Fake.fail_end = true;
    try t.expectError(error.NotReady, graph.endCapture(s));
    try t.expectError(error.UnownedCaptureSource, source.synchronousCopy(&d, 256, &.{2}, t.allocator));
    Fake.fail_end = false;
    var g = try graph.endCapture(s);
    g.deinit();
    try source.synchronousCopy(&d, 256, &.{3}, t.allocator);
}
const OtherThread = struct {
    const Driver = @import("driver.zig").Driver;
    var result: ?anyerror = null;
    fn run(arg: ?*anyopaque) callconv(.c) ?*anyopaque {
        const d: *Driver = @ptrCast(@alignCast(arg.?));
        source.setCurrent(@ptrFromInt(55));
        result = if (source.synchronousCopy(d, 256, &.{1}, t.allocator)) |_| null else |e| e;
        return null;
    }
};
test "a registered capture guards temporary uploads on another submitting thread" {
    var d = Fake.init();
    source.setCurrent(@ptrFromInt(55));
    const s: Stream = .{ .d = &d, .handle = @ptrFromInt(5) };
    try graph.beginCapture(s, .thread_local);
    var worker: std.c.pthread_t = undefined;
    try t.expectEqual(.SUCCESS, std.c.pthread_create(&worker, null, OtherThread.run, &d));
    try t.expectEqual(.SUCCESS, std.c.pthread_join(worker, null));
    var g = try graph.endCapture(s);
    g.deinit();
    try t.expectEqual(@as(?anyerror, error.UnownedCaptureSource), OtherThread.result);
}
