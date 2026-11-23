//! Qwen3.5 CUDA kept states' freed pinned buffers, kept whole for the next save or disk recall of the same size.
const std = @import("std");
const cuda = @import("cuda");
const reuse = @import("cuda_reuse.zig");

/// Freed pinned buffers, oldest first; one comes back only once no copy into or out of it can still be queued.
pub const Spares = struct {
    bufs: [cap]cuda.HostBuffer = undefined,
    n: usize = 0,

    /// The most buffers kept; the prompt cache's budget trims them further (`trim`).
    pub const cap = 2;

    /// The newest spare of exactly `len` bytes.
    fn find(s: *const Spares, len: usize) ?usize {
        var i = s.n;
        while (i > 0) {
            i -= 1;
            if (s.bufs[i].bytes.len == len) return i;
        }
        return null;
    }

    fn remove(s: *Spares, i: usize) cuda.HostBuffer {
        const host = s.bufs[i];
        for (i..s.n - 1) |j| s.bufs[j] = s.bufs[j + 1];
        s.n -= 1;
        return host;
    }

    /// Whether a spare of exactly `len` bytes waits.
    pub fn fits(s: *const Spares, len: usize) bool {
        return s.find(len) != null;
    }

    /// A buffer of `len` bytes: a spare of that size, else newly pinned memory.
    pub fn take(s: *Spares, d: *const cuda.Driver, len: usize) !cuda.HostBuffer {
        const i = s.find(len) orelse return cuda.HostBuffer.alloc(d, len);
        return s.remove(i);
    }

    /// A drained buffer back (no copy queued into or out of it); a full list frees its oldest.
    pub fn give(s: *Spares, host: cuda.HostBuffer) void {
        if (s.n == cap) {
            var old = s.remove(0);
            old.free();
        }
        s.bufs[s.n] = host;
        s.n += 1;
    }

    /// The spares' bytes.
    pub fn bytes(s: *const Spares) u64 {
        var n: u64 = 0;
        for (s.bufs[0..s.n]) |b| n += b.bytes.len;
        return n;
    }

    /// Spares freed, the largest first, until at most `room` bytes stay.
    pub fn trim(s: *Spares, room: u64) void {
        while (s.bytes() > room) {
            var big: usize = 0;
            for (s.bufs[1..s.n], 1..) |b, i| if (b.bytes.len > s.bufs[big].bytes.len) {
                big = i;
            };
            var old = s.remove(big);
            old.free();
        }
    }

    /// Every spare freed: with the engine, after the prompt cache dropped its states.
    pub fn deinit(s: *Spares) void {
        s.trim(0);
    }
};

/// A host copy's `len` bytes: from `spares` when the copy has a list, else newly pinned.
pub fn pin(spares: ?*Spares, d: *const cuda.Driver, len: usize) !cuda.HostBuffer {
    return if (spares) |s| s.take(d, len) else cuda.HostBuffer.alloc(d, len);
}

/// A drained host copy's buffer: back to its list, else freed.
pub fn release(spares: ?*Spares, host: cuda.HostBuffer) void {
    if (spares) |s| return s.give(host);
    var h = host;
    h.free();
}

fn targetOf(ptr: *anyopaque) *reuse.Target {
    return @ptrCast(@alignCast(ptr));
}

/// The prompt cache's spare bytes (cuda_snaps.zig): counted inside its budget.
pub fn spare(ptr: *anyopaque) u64 {
    return targetOf(ptr).spares.bytes();
}

/// The prompt cache's trim: spares freed until at most `room` bytes stay.
pub fn trim(ptr: *anyopaque, room: u64) void {
    targetOf(ptr).spares.trim(room);
}

/// Whether a save at `at` takes a spare: a host save, with a spare of its exact size.
pub fn reuses(ptr: *anyopaque, at: u32) bool {
    const t = targetOf(ptr);
    return !t.onDevice() and t.spares.fits(@intCast(reuse.bytes(ptr, at)));
}

test {
    _ = @import("cuda_spares_test.zig");
}
