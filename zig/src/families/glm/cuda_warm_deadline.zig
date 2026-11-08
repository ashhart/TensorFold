//! One deadline for the peer exchange, including blocking NCCL calls and stream cleanup.
const std = @import("std");

pub const timeout_ms: u64 = 300_000;

/// The test supplies a short limit and a fake clock through the same boundary as the production warm-up.
pub fn run(clock: anytype, limit_ms: u64, action: anytype, args: anytype, comptime work: anytype) anyerror!void {
    const Guard = struct {
        const State = enum(u8) { waiting, finished, expired };
        state: std.atomic.Value(State) = .init(.waiting),
        clock: @TypeOf(clock),
        action: @TypeOf(action),
        since: u64,
        limit: u64,

        fn expire(g: *@This()) void {
            if (g.state.cmpxchgStrong(.waiting, .expired, .acq_rel, .acquire) == null) g.action.expired();
        }
        fn watch(g: *@This()) void {
            while (g.state.load(.acquire) == .waiting) {
                if (g.clock.now() -| g.since >= g.limit) {
                    g.expire();
                    return;
                }
                g.clock.pause();
            }
        }
        fn finish(g: *@This()) bool {
            if (g.clock.now() -| g.since >= g.limit) g.expire();
            return g.state.cmpxchgStrong(.waiting, .finished, .acq_rel, .acquire) != null;
        }
    };
    var guard: Guard = .{ .clock = clock, .action = action, .since = clock.now(), .limit = limit_ms };
    const watcher = try std.Thread.spawn(.{}, Guard.watch, .{&guard});
    const result = work(args);
    const expired = guard.finish();
    watcher.join();
    if (expired) return error.PeerWarmupTimedOut;
    return result;
}
