//! While a paired request runs, a second queue keeps the residency set in use: the engine's queue waits on the peer.
const std = @import("std");
pub const PairWarm = For(@import("metal"));

/// The served keepalive's interval: macOS unwires an unused set after 1-2 s.
pub const interval_ns: u64 = 400 * std.time.ns_per_ms;

pub fn For(comptime Metal: type) type {
    return struct {
        const Self = @This();
        target: Metal.keepalive.Target, // the owner's sets and hold on this queue and scratch, never flushing
        active: *const std.atomic.Value(bool), // the request gate's flag: probes run only while it is set
        io: std.Io,
        interval_ns: u64 = interval_ns,
        stopped: std.atomic.Value(bool) = .init(false),
        thread: ?std.Thread = null,

        /// A second queue and its scratch; the scratch belongs in the owner's set before `track`.
        pub fn init(device: Metal.Device, io: std.Io, active: *const std.atomic.Value(bool)) !Self {
            const queue = try device.queue();
            errdefer queue.deinit();
            const scratch = try device.buffer(64, Metal.ResourceOptions.shared);
            return .{ .target = .{ .queue = queue, .scratch = scratch }, .active = active, .io = io };
        }

        /// The owner's sets and lock; its flush stays out, since set changes never commit inside a request.
        pub fn track(w: *Self, owner: Metal.keepalive.Target) void {
            w.target.sets = owner.sets;
            w.target.hold = owner.hold;
        }

        /// One committed probe under the owner's lock while a request is active, not waited for.
        pub fn step(w: *Self) bool {
            if (!w.active.load(.acquire)) return false;
            Metal.keepalive.Target.busy(&w.target);
            return true;
        }

        pub fn start(w: *Self) !void {
            w.stopped.store(false, .release);
            w.thread = try std.Thread.spawn(.{}, loop, .{w});
        }

        fn loop(w: *Self) void {
            while (!w.stopped.load(.acquire)) {
                std.Io.sleep(w.io, .fromNanoseconds(w.interval_ns), .awake) catch {};
                if (w.stopped.load(.acquire)) break;
                _ = w.step();
            }
        }

        pub fn stop(w: *Self) void {
            w.stopped.store(true, .release);
            if (w.thread) |t| t.join();
            w.thread = null;
        }

        /// Stops the thread, then releases the scratch and queue; the owner's set must not hold the scratch.
        pub fn deinit(w: *Self) void {
            w.stop();
            w.target.scratch.?.deinit();
            w.target.queue.deinit();
        }
    };
}
