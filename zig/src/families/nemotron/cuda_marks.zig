//! Prompt states kept inside a prompt chunk: the chunk writes their Mamba states as it runs, their KV rows follow it.

/// Marks a chunk holds at most, the scan's own limit; a pass keeps a handful (the history, the rewind, shared cuts).
pub const max = @import("cuda_kernels.zig").Snaps.max;

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
