//! Prompt states kept inside a prompt chunk: the chunk writes their Mamba states as it runs, their KV rows follow it.
const std = @import("std");
const cuda = @import("cuda");

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

/// The prompt cache's snapshot slots, made at load in one device buffer: a kept state's Mamba states take one.
pub const Slots = struct {
    buf: ?cuda.DeviceBuffer = null,
    bytes: usize = 0, // one slot's
    free: [most]u32 = undefined,
    n: usize = 0, // free slots
    count: usize = 0,

    /// Slots the store keeps at most (core kv_pool.most_snaps).
    pub const most = 64;

    pub fn init(s: *Slots, d: *const cuda.Driver, count: usize, bytes: usize) !void {
        if (s.buf != null or count > most) return error.SlotsMade;
        if (count == 0) return;
        s.buf = try cuda.DeviceBuffer.alloc(d, count * bytes);
        s.bytes = bytes;
        s.count = count;
        for (0..count) |i| s.free[i] = @intCast(count - 1 - i);
        s.n = count;
    }

    pub fn take(s: *Slots) ?u32 {
        if (s.n == 0) return null;
        s.n -= 1;
        return s.free[s.n];
    }

    pub fn give(s: *Slots, slot: u32) void {
        std.debug.assert(s.n < s.count);
        s.free[s.n] = slot;
        s.n += 1;
    }

    /// Slot `slot`'s address.
    pub fn at(s: *const Slots, slot: u32) u64 {
        return s.buf.?.ptr + @as(u64, slot) * s.bytes;
    }

    pub fn deinit(s: *Slots) void {
        if (s.buf) |*b| b.free();
        s.* = .{};
    }
};
