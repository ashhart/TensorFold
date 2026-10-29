//! Command ownership for cold-storage cuts. Every submitted consumer drains before its storage is reused.
const std = @import("std");

pub fn Commands(comptime M: type) type {
    return struct {
        const Self = @This();
        active: ?M.CommandBuffer = null,
        pending: ?M.CommandBuffer = null,
        mode: M.DispatchType = .serial,
        seconds: f64 = 0,
        waits: usize = 0,

        pub fn begin(c: *Self, cb: M.CommandBuffer, mode: M.DispatchType) M.ComputeEncoder {
            std.debug.assert(c.active == null);
            c.active = cb;
            c.mode = mode;
            return cb.compute(mode);
        }

        fn wait(c: *Self, cb: M.CommandBuffer) !void {
            cb.wait();
            c.waits += 1;
            c.seconds += cb.gpuSeconds();
            if (cb.failure() != null) return error.GpuFailed;
        }

        fn join(c: *Self) !void {
            if (c.pending) |cb| {
                c.pending = null;
                defer cb.release();
                try c.wait(cb);
            }
        }

        /// Joining both buffers detects an earlier asynchronous failure even when the last buffer succeeded.
        pub fn pause(c: *Self, queue: M.Queue, enc: *M.ComputeEncoder) !void {
            const cb = c.active orelse return error.NoActiveCommandBuffer;
            c.active = null;
            enc.end();
            cb.commit();
            var failure: ?anyerror = null;
            c.join() catch |err| {
                failure = err;
            };
            c.wait(cb) catch |err| {
                failure = failure orelse err;
            };
            if (failure) |err| return err;
            enc.* = c.begin(queue.commandBuffer(), c.mode);
        }

        /// Submit current expert compute before starting a forecast reader in disjoint, unleased slots.
        pub fn overlap(c: *Self, queue: M.Queue, enc: *M.ComputeEncoder, fence: anytype) !void {
            if (c.pending != null) return error.ForecastStillInFlight;
            const cb = c.active orelse return error.NoActiveCommandBuffer;
            c.active = null;
            fence.update(enc.*);
            enc.end();
            c.pending = cb.retain();
            cb.commit();
            const next = queue.commandBuffer();
            enc.* = c.begin(next, c.mode);
            fence.wait(enc.*);
        }

        /// The caller submits this actual final buffer, which can differ from the one it opened before a cut.
        pub fn end(c: *Self, enc: M.ComputeEncoder) !M.CommandBuffer {
            const cb = c.active orelse return error.NoActiveCommandBuffer;
            c.active = null;
            enc.end();
            return cb;
        }

        /// Drop unsubmitted commands and join all submitted readers' consumers before returning an error.
        pub fn abort(c: *Self, enc: M.ComputeEncoder) void {
            if (c.active != null) {
                c.active = null;
                enc.end();
            }
            c.join() catch {};
        }

        pub fn completed(c: *Self) !f64 {
            try c.join();
            const seconds = c.seconds;
            c.seconds = 0;
            return seconds;
        }
    };
}
