//! An engine thread that fails initialization closes admission and ends all queued work.
pub fn start(h: anytype) bool {
    if (h.on_thread) |t| t.run(t.ctx) catch |e| {
        h.lock();
        h.closing = true;
        h.halt_reason = @errorName(e);
        h.wake.broadcast(h.io);
        h.unlock();
        @import("lane_halt.zig").haltAll(h, @errorName(e));
        return false;
    };
    return true;
}
