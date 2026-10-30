//! A second HIP stream beside a step's own: `fork` gives it the step's work so far, `join` gives its work back.
const std = @import("std");
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const util = @import("launches/util.zig");

/// Ordering events forks and joins take in turn: 256 gives each record of one captured forward its own event.
pub const ring_len = 256;

/// A slot is recorded and waited on in one call and a wait takes the record queued before it, so the ring may wrap.
pub const Ring = struct {
    r: *const runtime.Runtime,
    events: [ring_len]abi.Event = @splat(null),
    next: usize = 0,

    pub fn init(r: *const runtime.Runtime) runtime.Error!Ring {
        var ring: Ring = .{ .r = r };
        errdefer ring.deinit();
        for (&ring.events) |*e| {
            try runtime.check(r.api.hipEventCreateWithFlags(e, abi.event_disable_timing));
            if (e.* == null) return error.Invalid;
        }
        return ring;
    }

    pub fn deinit(ring: *Ring) void {
        for (ring.events) |e| if (e != null) {
            _ = ring.r.api.hipEventDestroy(e);
        };
        ring.* = undefined;
    }

    fn take(ring: *Ring) abi.Event {
        const e = ring.events[ring.next];
        ring.next = (ring.next + 1) % ring_len;
        return e;
    }
};

/// The side stream and its events, as `Ops` carries them (both null: no side stream, `fork` and `join` do nothing).
pub const Side = struct {
    stream: abi.Stream = null,
    ring: ?*Ring = null,

    fn on(s: Side, main: abi.Stream) bool {
        return s.stream != null and s.ring != null and main != util.counting;
    }

    /// Later work on the side stream starts after `main`'s work so far.
    pub fn fork(s: Side, main: abi.Stream) runtime.Error!void {
        if (!s.on(main)) return;
        const ring = s.ring.?;
        const e = ring.take();
        try runtime.check(ring.r.api.hipEventRecord(e, main));
        try runtime.check(ring.r.api.hipStreamWaitEvent(s.stream, e, 0));
    }

    /// Later work on `main` starts after the side stream's work so far.
    pub fn join(s: Side, main: abi.Stream) runtime.Error!void {
        if (!s.on(main)) return;
        const ring = s.ring.?;
        const e = ring.take();
        try runtime.check(ring.r.api.hipEventRecord(e, s.stream));
        try runtime.check(ring.r.api.hipStreamWaitEvent(main, e, 0));
    }
};

test "fork records on the main stream and the side waits; join the other way; off without a side or when counting" {
    const Mock = struct {
        var log: [8]struct { what: u8, stream: usize, event: usize } = undefined;
        var n: usize = 0;
        fn record(e: abi.Event, s: abi.Stream) callconv(.c) c_int {
            log[n] = .{ .what = 'r', .stream = @intFromPtr(s), .event = @intFromPtr(e) };
            n += 1;
            return 0;
        }
        fn wait(s: abi.Stream, e: abi.Event, _: c_uint) callconv(.c) c_int {
            log[n] = .{ .what = 'w', .stream = @intFromPtr(s), .event = @intFromPtr(e) };
            n += 1;
            return 0;
        }
    };
    var r: runtime.Runtime = undefined;
    r.api.hipEventRecord = Mock.record;
    r.api.hipStreamWaitEvent = Mock.wait;
    var ring: Ring = .{ .r = &r };
    for (&ring.events, 0..) |*e, i| e.* = @ptrFromInt(0x1000 + i);
    const main: abi.Stream = @ptrFromInt(0x10);
    const side: Side = .{ .stream = @ptrFromInt(0x20), .ring = &ring };
    try side.fork(main);
    try side.join(main);
    try std.testing.expectEqual(@as(usize, 4), Mock.n);
    try std.testing.expectEqual(@as(u8, 'r'), Mock.log[0].what);
    try std.testing.expectEqual(@as(usize, 0x10), Mock.log[0].stream);
    try std.testing.expectEqual(@as(usize, 0x20), Mock.log[1].stream);
    try std.testing.expectEqual(Mock.log[0].event, Mock.log[1].event);
    try std.testing.expectEqual(@as(usize, 0x20), Mock.log[2].stream);
    try std.testing.expectEqual(@as(usize, 0x10), Mock.log[3].stream);
    try std.testing.expect(Mock.log[2].event != Mock.log[0].event);
    try (Side{}).fork(main);
    try side.join(util.counting);
    try std.testing.expectEqual(@as(usize, 4), Mock.n);
}

test "more forks than the ring before one join: each wait follows its own record at once, the slots wrap in order" {
    const Mock = struct {
        var last_record: usize = 0;
        var records: usize = 0;
        var waits: usize = 0;
        var paired: bool = true;
        var first: [3]usize = .{ 0, 0, 0 };
        fn record(e: abi.Event, _: abi.Stream) callconv(.c) c_int {
            if (records == waits + 1) paired = false; // a record while another one's wait is outstanding
            last_record = @intFromPtr(e);
            if (records < 3) first[records] = last_record;
            records += 1;
            return 0;
        }
        fn wait(_: abi.Stream, e: abi.Event, _: c_uint) callconv(.c) c_int {
            if (@intFromPtr(e) != last_record or waits + 1 != records) paired = false;
            waits += 1;
            return 0;
        }
    };
    var r: runtime.Runtime = undefined;
    r.api.hipEventRecord = Mock.record;
    r.api.hipStreamWaitEvent = Mock.wait;
    var ring: Ring = .{ .r = &r };
    for (&ring.events, 0..) |*e, i| e.* = @ptrFromInt(0x1000 + i);
    const side: Side = .{ .stream = @ptrFromInt(0x20), .ring = &ring };
    const forks = 2 * ring_len + 3;
    for (0..forks) |_| try side.fork(@ptrFromInt(0x10));
    try side.join(@ptrFromInt(0x10));
    try std.testing.expect(Mock.paired);
    try std.testing.expectEqual(@as(usize, forks + 1), Mock.records);
    try std.testing.expectEqual(@as(usize, forks + 1), Mock.waits);
    // the join took slot (forks mod ring_len): the ring wrapped twice in order
    try std.testing.expectEqual(@as(usize, 0x1000 + forks % ring_len), Mock.last_record);
    try std.testing.expectEqual(@as(usize, 0x1001), Mock.first[1]);
}
