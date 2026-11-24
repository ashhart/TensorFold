//! A request's cancellation hook at a tower's checkpoints, on the lanes' own CancelCheck.
const std = @import("std");
const CancelCheck = @import("lanes").stream.CancelCheck;

/// error.Cancelled when the hook says the request left; a null hook never cancels.
pub fn poll(hook: ?CancelCheck) error{Cancelled}!void {
    if (hook) |h| if (h.check(h.ptr)) return error.Cancelled;
}

/// With a hook: finish the stream's queued work, then poll, so a cancelled request frees idle buffers; null: neither.
pub fn drain(stream: anytype, hook: ?CancelCheck) !void {
    if (hook == null) return;
    try stream.synchronize();
    try poll(hook);
}

/// A long prompt chunk's layer edges: polled with the next layer queued behind the running one, not a drain an edge.
pub fn Layers(comptime Stream: type, comptime Event: type) type {
    return struct {
        stream: Stream,
        events: ?[2]Event = null, // null: each edge drains the stream, then polls
        edges: usize = 0,

        /// After a layer's launches: wait for the layer before it, then poll; a cancel drains the stream, then returns.
        pub fn edge(l: *@This(), ctx: *anyopaque, left: *const fn (*anyopaque) bool) !void {
            const ev = l.events orelse {
                try l.stream.synchronize();
                if (left(ctx)) return error.Cancelled;
                return;
            };
            try ev[l.edges % 2].record(l.stream);
            if (l.edges > 0) try ev[(l.edges - 1) % 2].synchronize();
            l.edges += 1;
            if (!left(ctx)) return;
            try l.stream.synchronize(); // the queued layer finishes, so the pass unwinds over idle buffers
            return error.Cancelled;
        }
    };
}

const Mock = struct {
    queued: bool = true,
    synchronizations: usize = 0,
    polls: usize = 0,
    polled_before_drain: bool = false,
    fail_sync: bool = false,
    left: bool = true,

    pub fn synchronize(m: *Mock) !void {
        if (m.fail_sync) return error.HipFailed;
        m.synchronizations += 1;
        m.queued = false;
    }

    fn check(ptr: *anyopaque) bool {
        const m: *Mock = @ptrCast(@alignCast(ptr));
        m.polls += 1;
        m.polled_before_drain = m.polled_before_drain or m.queued;
        return m.left;
    }

    fn hook(m: *Mock) CancelCheck {
        return .{ .ptr = m, .check = check };
    }
};

test "a null hook neither synchronizes nor polls, so a tower without one stays asynchronous" {
    var m: Mock = .{};
    try drain(&m, null);
    try poll(null);
    try std.testing.expectEqual(@as(usize, 0), m.synchronizations);
    try std.testing.expect(m.queued);
}

test "a hook drains the stream before it polls, and a left request cancels" {
    var m: Mock = .{};
    try std.testing.expectError(error.Cancelled, drain(&m, m.hook()));
    try std.testing.expectEqual(@as(usize, 1), m.synchronizations);
    try std.testing.expectEqual(@as(usize, 1), m.polls);
    try std.testing.expect(!m.queued and !m.polled_before_drain);
    var stays: Mock = .{ .left = false };
    try drain(&stays, stays.hook());
    try poll(stays.hook());
    try std.testing.expectEqual(@as(usize, 2), stays.polls);
}

test "a failed synchronize is the error, and the hook is not polled" {
    var m: Mock = .{ .fail_sync = true };
    try std.testing.expectError(error.HipFailed, drain(&m, m.hook()));
    try std.testing.expectEqual(@as(usize, 0), m.polls);
}

/// A layer-edge stream and its two events, counted: the drains, records and waits `Layers.edge` makes.
const Edges = struct {
    drains: usize = 0,
    records: usize = 0,
    waits: usize = 0,
    recorded: usize = 9,
    waited: usize = 9,
    polls: usize = 0,
    left_at: usize = std.math.maxInt(usize), // the poll that first finds the request gone

    pub fn synchronize(e: *Edges) !void {
        e.drains += 1;
    }

    fn left(ptr: *anyopaque) bool {
        const e: *Edges = @ptrCast(@alignCast(ptr));
        e.polls += 1;
        return e.polls >= e.left_at;
    }

    const Event = struct {
        e: *Edges,
        id: usize,
        pub fn record(v: Event, _: *Edges) !void {
            v.e.records += 1;
            v.e.recorded = v.id;
        }
        pub fn synchronize(v: Event) !void {
            v.e.waits += 1;
            v.e.waited = v.id;
        }
    };
};

test "layer edges keep one layer queued: each waits for the layer before it, and only a cancel drains" {
    var e: Edges = .{};
    var l: Layers(*Edges, Edges.Event) = .{ .stream = &e, .events = .{ .{ .e = &e, .id = 0 }, .{ .e = &e, .id = 1 } } };
    try l.edge(&e, Edges.left);
    try std.testing.expectEqual(@as(usize, 0), e.waits); // the first layer has none before it
    for (1..5) |i| {
        try l.edge(&e, Edges.left);
        try std.testing.expectEqual(i % 2, e.recorded);
        try std.testing.expectEqual((i - 1) % 2, e.waited);
    }
    try std.testing.expectEqual(@as(usize, 5), e.records);
    try std.testing.expectEqual(@as(usize, 4), e.waits);
    try std.testing.expectEqual(@as(usize, 0), e.drains);
    e.left_at = e.polls + 1;
    try std.testing.expectError(error.Cancelled, l.edge(&e, Edges.left));
    try std.testing.expectEqual(@as(usize, 1), e.drains);
    try std.testing.expectEqual(@as(usize, 6), e.records);
}

test "layer edges without events drain the stream at each edge, then poll" {
    var e: Edges = .{ .left_at = 3 };
    var l: Layers(*Edges, Edges.Event) = .{ .stream = &e };
    try l.edge(&e, Edges.left);
    try l.edge(&e, Edges.left);
    try std.testing.expectError(error.Cancelled, l.edge(&e, Edges.left));
    try std.testing.expectEqual(@as(usize, 3), e.drains);
    try std.testing.expectEqual(@as(usize, 3), e.polls);
    try std.testing.expectEqual(@as(usize, 0), e.records);
}
