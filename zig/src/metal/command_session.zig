//! One queue's compute/blit order and retained command buffers, with nonblocking retirement and sticky device faults.
const std = @import("std");
pub fn Session(comptime M: type) type {
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        queue: M.Queue,
        active: ?M.CommandBuffer = null,
        encoding: ?M.ComputeEncoder = null,
        pool: ?M.objc.Pool = null,
        pending: std.ArrayList(M.CommandBuffer) = .empty,
        failed: bool = false,
        pub fn init(a: std.mem.Allocator, queue: M.Queue) Self {
            return .{ .a = a, .queue = queue };
        }
        pub fn begin(s: *Self) !void {
            if (s.active != null) return error.MetalCommandActive;
            try s.poll();
            if (s.failed) return error.MetalDeviceFailed;
            s.pool = M.objc.Pool.push();
            s.active = s.queue.commandBuffer().retain();
        }
        fn endCompute(s: *Self) void {
            if (s.encoding) |e| {
                e.end();
                s.encoding = null;
            }
        }
        pub fn encoder(s: *Self) !M.ComputeEncoder {
            if (s.failed) return error.MetalDeviceFailed;
            const cb = s.active orelse return error.MetalCommandMissing;
            if (s.encoding == null) s.encoding = cb.compute(.serial);
            return s.encoding.?;
        }
        /// Transfer the retained command buffer after list admission; failures remain owned.
        pub fn submit(s: *Self) !void {
            if (s.failed) return error.MetalDeviceFailed;
            const cb = s.active orelse return error.MetalCommandMissing;
            s.endCompute();
            try s.pending.ensureUnusedCapacity(s.a, 1);
            s.pending.appendAssumeCapacity(cb);
            cb.commit();
            s.active = null;
            s.pool.?.pop();
            s.pool = null;
        }
        pub fn discard(s: *Self) void {
            s.endCompute();
            if (s.active) |cb| cb.release();
            s.active = null;
            if (s.pool) |pool| pool.pop();
            s.pool = null;
        }
        pub fn copy(s: *Self, to: M.Buffer, to_offset: usize, from: M.Buffer, from_offset: usize, n: usize) !void {
            if (s.failed) return error.MetalDeviceFailed;
            if (n == 0) return;
            const own = s.active == null;
            if (own) try s.begin();
            errdefer if (own) s.discard();
            s.endCompute();
            const e = s.active.?.blit();
            e.copy(from, from_offset, to, to_offset, n);
            e.end();
            if (own) try s.submit();
        }
        pub fn zero(s: *Self, b: M.Buffer, offset: usize, n: usize) !void {
            if (s.failed) return error.MetalDeviceFailed;
            if (n == 0) return;
            const own = s.active == null;
            if (own) try s.begin();
            errdefer if (own) s.discard();
            s.endCompute();
            const e = s.active.?.blit();
            e.fill(b, offset, n, 0);
            e.end();
            if (own) try s.submit();
        }
        fn retire(s: *Self, wait: bool) !void {
            var bad = false;
            while (s.pending.items.len > 0) {
                const cb = s.pending.items[0];
                if (wait) cb.wait();
                const status = cb.status();
                if (status != .completed and status != .@"error") break;
                bad = bad or cb.failure() != null;
                cb.release();
                _ = s.pending.orderedRemove(0);
            }
            s.failed = s.failed or bad;
            if (bad) return error.MetalDeviceFailed;
        }
        /// A failed completion poisons later submissions. Poll never waits.
        pub fn poll(s: *Self) !void {
            return s.retire(false);
        }
        pub fn drain(s: *Self) !void {
            if (s.active != null) return error.MetalCommandUnsubmitted;
            try s.retire(true);
            if (s.pending.items.len != 0) return error.MetalCommandIncomplete;
        }
        /// A reported completed fault permits teardown on retry; failed completion retains every owner.
        pub fn deinit(s: *Self) !void {
            try s.drain();
            s.pending.deinit(s.a);
            s.* = undefined;
        }
    };
}
