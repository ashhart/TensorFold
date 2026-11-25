//! Pinned staging bytes' uploads and the fence that says when they may change.
const cuda = @import("cuda");

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
