//! Opt-in non-final prompt completion and two pinned staging banks with independent upload fences.
const std = @import("std");
const cuda = @import("cuda");

pub fn enabled() !bool {
    // The startup scope rejects malformed values before a model or scratch allocation.
    return @import("sys").environment.read(bool, "TF_FLASHNEXT_PROMPT_ASYNC", false);
}

/// Preserve the default drain after a no-logits pass; the opt-in leaves device work queued in the same order.
pub fn afterChunk(owner: anytype, logits_none: bool, async_enabled: bool) !void {
    if (logits_none and !async_enabled) try owner.complete();
}

/// Host logits are read only after the same completion call used by the synchronous path.
pub fn beforeLogits(owner: anytype) !void {
    try owner.complete();
}

/// The compute stream as a pass's pinned staging uses it: copies from it, an event behind them, and a drain.
pub const Uploads = struct {
    stream: cuda.Stream,
    event: cuda.Event,

    pub fn copy(u: Uploads, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(u.stream.d, dst, bytes, u.stream);
    }
    pub fn record(u: Uploads) !void {
        try u.event.record(u.stream);
    }
    pub fn wait(u: Uploads) !void {
        try u.event.synchronize();
    }
    pub fn drain(u: Uploads) !void {
        try u.stream.synchronize();
    }
};

/// Pinned bytes' uploads and the fence that says when the bytes may change (`Q`: `Uploads`, or a test's stand-in).
pub fn Staging(comptime Q: type) type {
    return struct {
        const Self = @This();
        q: Q,
        /// An upload was queued after the event's last record (a copy or the record failed): only a drain covers it.
        open: bool = false,

        /// Queues a copy from the staged bytes; until `fence` records behind it, `ready` drains the stream.
        pub fn upload(g: *Self, dst: u64, bytes: []const u8) !void {
            g.open = true;
            try g.q.copy(dst, bytes);
        }

        /// Records the event behind every upload queued so far.
        pub fn fence(g: *Self) !void {
            try g.q.record();
            g.open = false;
        }

        /// Returns once no queued upload reads the staged bytes: the event's wait, or a drain if one came after it.
        pub fn ready(g: *Self) !void {
            if (!g.open) return g.q.wait();
            try g.q.drain();
            g.open = false;
        }
    };
}

/// Two pinned banks. `select` alternates only with the opt-in; `ready` fences the selected bank before host writes.
pub fn Bank(comptime Q: type, comptime H: type) type {
    return struct {
        const Self = @This();
        enabled: bool,
        active: usize = 0,
        hosts: [2]?H = .{ null, null },
        queues: [2]Staging(Q),

        pub fn select(b: *Self) bool {
            if (!b.enabled) return false;
            b.active = 1 - b.active;
            return true;
        }
        pub fn ready(b: *Self) !void {
            try b.queues[b.active].ready();
        }
        pub fn upload(b: *Self, dst: u64, src: []const u8) !void {
            try b.queues[b.active].upload(dst, src);
        }
        pub fn fence(b: *Self) !void {
            try b.queues[b.active].fence();
        }
        pub fn bytes(b: *const Self) []u8 {
            return b.hosts[b.active].?.bytes;
        }
        pub fn drain(b: *Self) void {
            const n: usize = if (b.enabled) 2 else 1;
            for (b.queues[0..n]) |*q| q.ready() catch {};
        }
    };
}

pub const Pair = Bank(Uploads, cuda.HostBuffer);

/// Allocate the second pinned bank and event only for the opt-in; every partial allocation is rolled back.
pub fn allocPair(d: *const cuda.Driver, stream: cuda.Stream, len: usize) !Pair {
    const on = try enabled();
    var p: Pair = .{ .enabled = on, .queues = .{
        .{ .q = .{ .stream = stream, .event = undefined } },
        .{ .q = .{ .stream = stream, .event = undefined } },
    } };
    var events: usize = 0;
    errdefer {
        for (p.queues[0..events]) |*q| q.q.event.deinit();
        for (&p.hosts) |*h| if (h.*) |*buf| buf.free();
    }
    p.hosts[0] = try cuda.HostBuffer.alloc(d, len);
    p.queues[0].q.event = try cuda.Event.init(d, false);
    events = 1;
    if (on) {
        p.hosts[1] = try cuda.HostBuffer.alloc(d, len);
        p.queues[1].q.event = try cuda.Event.init(d, false);
        events = 2;
    }
    return p;
}

/// Both banks can have uploads in flight, so both fences precede their event and host destruction.
pub fn freePair(p: *Pair) void {
    const n: usize = if (p.enabled) 2 else 1;
    p.drain();
    for (p.queues[0..n]) |*q| q.q.event.deinit();
    for (&p.hosts) |*h| if (h.*) |*buf| buf.free();
    p.* = undefined;
}
