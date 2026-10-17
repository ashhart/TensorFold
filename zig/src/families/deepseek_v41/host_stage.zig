//! Pinned host staging a prompt segment rewrites without draining the stream (waits only on its last copy's event).

const std = @import("std");
const cuda = @import("cuda");

/// forward.CHUNK: windows of more rows stage through pinned memory without a stream sync
pub const chunk: usize = 128;

pub const Stage = struct {
    buf: ?cuda.HostBuffer = null,
    ev: ?cuda.Event = null,
    /// a copy out of `buf` was issued and `ev` recorded after it
    pending: bool = false,

    /// `bytes` of the buffer to write: the copy that last read it has finished; grown when too small.
    pub fn take(s: *Stage, d: *const cuda.Driver, bytes: usize) ![]u8 {
        if (s.pending) {
            try s.ev.?.synchronize();
            s.pending = false;
        }
        if (s.buf == null or s.buf.?.bytes.len < bytes) {
            if (s.buf) |*b| b.free();
            s.buf = null;
            s.buf = try cuda.HostBuffer.alloc(d, @max(bytes, 4096));
        }
        return s.buf.?.bytes[0..bytes];
    }

    /// The copy out of the buffer is on `stream`: the next `take` waits for it alone.
    pub fn issued(s: *Stage, d: *const cuda.Driver, stream: cuda.Stream) !void {
        if (s.ev == null) s.ev = try cuda.Event.init(d, false);
        try s.ev.?.record(stream);
        s.pending = true;
    }

    pub fn deinit(s: *Stage) void {
        if (s.ev) |*e| e.deinit();
        if (s.buf) |*b| b.free();
        s.* = .{};
    }
};
