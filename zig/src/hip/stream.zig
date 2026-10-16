//! HIP nonblocking streams and events; callers synchronize before releasing in-flight buffers.
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const std = @import("std");

pub const Stream = struct {
    r: *const runtime.Runtime,
    handle: abi.Stream,

    pub fn init(r: *const runtime.Runtime) runtime.Error!Stream {
        var handle: abi.Stream = null;
        try runtime.check(r.api.hipStreamCreateWithFlags(&handle, 1));
        if (handle == null) return error.Invalid;
        return .{ .r = r, .handle = handle };
    }

    pub fn synchronize(self: Stream) runtime.Error!void {
        try runtime.check(self.r.api.hipStreamSynchronize(self.handle));
    }

    /// True once every queued item has finished.
    pub fn done(self: Stream) runtime.Error!bool {
        return finished(self.r.api.hipStreamQuery(self.handle));
    }

    /// Later work on this stream starts only after `event`'s recorded work completes.
    pub fn wait(self: Stream, event: Event) runtime.Error!void {
        if (self.r != event.r) return error.Invalid;
        try runtime.check(self.r.api.hipStreamWaitEvent(self.handle, event.handle, 0));
    }

    pub fn deinit(self: *Stream) void {
        _ = self.r.api.hipStreamDestroy(self.handle);
        self.* = undefined;
    }
};

pub const Event = struct {
    r: *const runtime.Runtime,
    handle: abi.Event,

    /// Timing events cost a little more to record; ordering-only events skip the timestamp.
    pub fn init(r: *const runtime.Runtime, timing: bool) runtime.Error!Event {
        var handle: abi.Event = null;
        try runtime.check(r.api.hipEventCreateWithFlags(&handle, if (timing) 0 else abi.event_disable_timing));
        if (handle == null) return error.Invalid;
        return .{ .r = r, .handle = handle };
    }

    pub fn record(self: Event, stream: Stream) runtime.Error!void {
        if (self.r != stream.r) return error.Invalid;
        try runtime.check(self.r.api.hipEventRecord(self.handle, stream.handle));
    }

    pub fn synchronize(self: Event) runtime.Error!void {
        try runtime.check(self.r.api.hipEventSynchronize(self.handle));
    }

    pub fn done(self: Event) runtime.Error!bool {
        return finished(self.r.api.hipEventQuery(self.handle));
    }

    /// Milliseconds between two recorded timing events.
    pub fn elapsedMs(start: Event, end: Event) runtime.Error!f32 {
        if (start.r != end.r) return error.Invalid;
        var ms: f32 = 0;
        try runtime.check(start.r.api.hipEventElapsedTime(&ms, start.handle, end.handle));
        return ms;
    }

    pub fn deinit(self: *Event) void {
        _ = self.r.api.hipEventDestroy(self.handle);
        self.* = undefined;
    }
};

fn finished(result: abi.Result) runtime.Error!bool {
    if (result == abi.not_ready) return false;
    try runtime.check(result);
    return true;
}

test "queries report not-ready as unfinished and keep other failures" {
    try std.testing.expect(try finished(0));
    try std.testing.expect(!try finished(abi.not_ready));
    try std.testing.expectError(error.HipFailed, finished(1));
    try std.testing.expectError(error.HipFailed, finished(-1));
}

test "event flags, waits and timing reach HIP; foreign owners are refused first" {
    const Mock = struct {
        var flags: c_uint = 99;
        var wait_flags: c_uint = 99;
        var elapsed_result: c_int = 0;
        fn create(out: *abi.Event, f: c_uint) callconv(.c) c_int {
            flags = f;
            out.* = @ptrFromInt(64);
            return 0;
        }
        fn waitEvent(_: abi.Stream, _: abi.Event, f: c_uint) callconv(.c) c_int {
            wait_flags = f;
            return 0;
        }
        fn elapsed(ms: *f32, _: abi.Event, _: abi.Event) callconv(.c) c_int {
            ms.* = 1.5;
            return elapsed_result;
        }
    };
    var r: runtime.Runtime = undefined;
    r.api.hipEventCreateWithFlags = Mock.create;
    r.api.hipStreamWaitEvent = Mock.waitEvent;
    r.api.hipEventElapsedTime = Mock.elapsed;
    const timing = try Event.init(&r, true);
    try std.testing.expectEqual(@as(c_uint, 0), Mock.flags);
    const ordering = try Event.init(&r, false);
    try std.testing.expectEqual(abi.event_disable_timing, Mock.flags);
    const stream = Stream{ .r = &r, .handle = null };
    try stream.wait(ordering);
    try std.testing.expectEqual(@as(c_uint, 0), Mock.wait_flags);
    try std.testing.expectEqual(@as(f32, 1.5), try Event.elapsedMs(timing, ordering));
    Mock.elapsed_result = 1;
    try std.testing.expectError(error.HipFailed, Event.elapsedMs(timing, ordering));
    const foreign = Event{ .r = @ptrFromInt(32), .handle = null };
    try std.testing.expectError(error.Invalid, stream.wait(foreign));
    try std.testing.expectError(error.Invalid, foreign.record(stream));
    try std.testing.expectError(error.Invalid, Event.elapsedMs(timing, foreign));
}
