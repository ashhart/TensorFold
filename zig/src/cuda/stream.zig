//! Streams and events: ordering, waits and GPU-side timing.

const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

pub const Stream = struct {
    d: *const Driver,
    handle: abi.Stream,

    /// A non-blocking stream never waits on the legacy default stream (what every engine stream should be).
    pub fn init(d: *const Driver, non_blocking: bool) Error!Stream {
        var s: abi.Stream = null;
        try d.check(d.api.cuStreamCreate(&s, if (non_blocking) abi.stream_non_blocking else 0), "cuStreamCreate");
        return .{ .d = d, .handle = s };
    }

    pub fn deinit(self: *Stream) void {
        _ = self.d.api.cuStreamDestroy_v2(self.handle);
        self.* = undefined;
    }

    pub fn synchronize(self: Stream) Error!void {
        try self.d.check(self.d.api.cuStreamSynchronize(self.handle), "cuStreamSynchronize");
    }

    /// True once every queued item has finished.
    pub fn done(self: Stream) Error!bool {
        self.d.check(self.d.api.cuStreamQuery(self.handle), "cuStreamQuery") catch |e| switch (e) {
            error.NotReady => return false,
            else => return e,
        };
        return true;
    }

    pub fn wait(self: Stream, event: Event) Error!void {
        try self.d.check(self.d.api.cuStreamWaitEvent(self.handle, event.handle, 0), "cuStreamWaitEvent");
    }
};

pub const Event = struct {
    d: *const Driver,
    handle: abi.Event,

    /// Timing events cost a little more to record; ordering-only events skip the timestamp.
    pub fn init(d: *const Driver, timing: bool) Error!Event {
        var e: abi.Event = null;
        try d.check(d.api.cuEventCreate(&e, if (timing) 0 else abi.event_disable_timing), "cuEventCreate");
        return .{ .d = d, .handle = e };
    }

    pub fn deinit(self: *Event) void {
        _ = self.d.api.cuEventDestroy_v2(self.handle);
        self.* = undefined;
    }

    pub fn record(self: Event, stream: Stream) Error!void {
        try self.d.check(self.d.api.cuEventRecord(self.handle, stream.handle), "cuEventRecord");
    }

    pub fn synchronize(self: Event) Error!void {
        try self.d.check(self.d.api.cuEventSynchronize(self.handle), "cuEventSynchronize");
    }

    pub fn done(self: Event) Error!bool {
        self.d.check(self.d.api.cuEventQuery(self.handle), "cuEventQuery") catch |e| switch (e) {
            error.NotReady => return false,
            else => return e,
        };
        return true;
    }

    /// Milliseconds between two recorded timing events (about 0.5 us resolution).
    pub fn elapsedMs(start: Event, end: Event) Error!f32 {
        var ms: f32 = 0;
        try start.d.check(start.d.api.cuEventElapsedTime(&ms, start.handle, end.handle), "cuEventElapsedTime");
        return ms;
    }
};
