//! Prompt states kept inside a prompt chunk: the chunk writes their Mamba states as it runs, their KV rows follow it.
const cuda = @import("cuda");

/// Marks a chunk holds at most, the scan's own limit; a pass keeps a handful (the history, the rewind, shared cuts).
pub const max = @import("cuda_kernels.zig").Snaps.max;

/// Spare Mamba buffers the prompt cache sets aside at most: a first turn keeps up to four rows in one chunk.
pub const spares_max = 4;

/// A mark inside the chunk: its row, where its state's SSM and conv parts go, and its kept copy (cuda_reuse.zig).
pub const Mark = struct { at: usize, ssm: u64, conv: u64, copy: *anyopaque };

/// The marks of the chunk about to run, rows [open[0], open[1]): a save strictly inside is armed instead of copied.
pub const Armed = struct {
    open: ?[2]usize = null,
    list: [max]Mark = undefined,
    n: usize = 0,

    /// The armed marks, in row order (saves arrive in the order of the pass's ascending marks).
    pub fn marks(a: *const Armed) []const Mark {
        return a.list[0..a.n];
    }

    /// A kept copy dropped before its chunk ran: no capture may write it.
    pub fn forget(a: *Armed, copy: *anyopaque) void {
        for (a.list[0..a.n], 0..) |m, i| if (m.copy == copy) {
            for (i..a.n - 1) |j| a.list[j] = a.list[j + 1];
            a.n -= 1;
            return;
        };
    }
};

/// Free buffers for the next kept states' Mamba parts, inside the cache's budget: a keep before a chunk allocates none.
pub const Spares = struct {
    list: [spares_max]cuda.DeviceBuffer = undefined,
    n: usize = 0,
    want: usize = 0, // what the cache set aside (cuda_reuse.zig `ready`); 0: no spares, every keep allocates

    pub fn take(s: *Spares) ?cuda.DeviceBuffer {
        if (s.n == 0) return null;
        s.n -= 1;
        return s.list[s.n];
    }

    /// Keeps `b` while fewer than `want` wait; false: the caller frees it.
    pub fn give(s: *Spares, b: cuda.DeviceBuffer) bool {
        if (s.n >= s.want) return false;
        s.list[s.n] = b;
        s.n += 1;
        return true;
    }

    /// New spares up to `want`; called once a chunk is queued, so the host allocates while the GPU runs it.
    pub fn fill(s: *Spares, d: *const cuda.Driver, len: usize) void {
        while (s.n < s.want) {
            s.list[s.n] = cuda.DeviceBuffer.alloc(d, len) catch return; // none now: a keep allocates its own
            s.n += 1;
        }
    }

    pub fn deinit(s: *Spares) void {
        for (s.list[0..s.n]) |*b| b.free();
        s.n = 0;
    }
};
