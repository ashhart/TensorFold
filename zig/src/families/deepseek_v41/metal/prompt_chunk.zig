//! An encoder chunk's command buffers on one queue, and the wait that reports a failed buffer.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");

/// An encoder chunk's command buffers, committed in order on one queue; its open encoder is `e`.
pub const Chunk = struct {
    cbs: [2 + host.engram_host.max_layers]mtl.CommandBuffer = undefined,
    /// each buffer's commit on the host clock Metal's GPU times use
    at: [2 + host.engram_host.max_layers]f64 = undefined,
    n: usize = 0,
    e: mtl.ComputeEncoder = undefined,
    live: bool = false,

    pub fn open(ch: *Chunk, q: mtl.Queue) void {
        ch.cbs[ch.n] = q.commandBuffer();
        ch.e = ch.cbs[ch.n].compute(.serial);
        ch.n += 1;
        ch.live = true;
    }

    pub fn close(ch: *Chunk) void {
        ch.e.end();
        ch.at[ch.n - 1] = mtl.clock.seconds();
        ch.cbs[ch.n - 1].commit();
        ch.live = false;
    }

    /// An encode error: the open buffer is dropped, the committed ones run out before the caller's buffers go.
    pub fn abandon(ch: *Chunk) void {
        if (ch.live) {
            ch.e.end();
            ch.n -= 1;
            ch.live = false;
        }
        for (ch.cbs[0..ch.n]) |cb| cb.wait();
    }

    /// Waits for every buffer, then reports the first failure (none still runs when the caller reuses the set).
    pub fn wait(ch: *const Chunk) !void {
        for (ch.cbs[0..ch.n]) |cb| cb.wait();
        for (ch.cbs[0..ch.n]) |cb| try submitted(cb);
    }

    pub fn gpuSeconds(ch: *const Chunk) f64 {
        var t: f64 = 0;
        for (ch.cbs[0..ch.n]) |cb| t += cb.gpuSeconds();
        return t;
    }

    /// Seconds the buffers waited to start after both their commit and the previous buffer's end.
    pub fn lateSeconds(ch: *const Chunk) f64 {
        var t: f64 = 0;
        for (ch.cbs[0..ch.n], ch.at[0..ch.n], 0..) |cb, at, k| {
            const ready = if (k == 0) at else @max(at, ch.cbs[k - 1].gpuEnd());
            t += @max(0, cb.gpuStart() - ready);
        }
        return t;
    }
};

pub fn submit(cb: mtl.CommandBuffer) !void {
    cb.commit();
    return submitted(cb);
}

/// Waits for a committed command buffer and reports its failure.
fn submitted(cb: mtl.CommandBuffer) !void {
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("prompt command buffer failed: {s}\n", .{msg});
        return error.CommandBufferFailed;
    }
}
