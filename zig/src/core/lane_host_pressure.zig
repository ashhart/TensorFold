//! The lane host under memory pressure: admission held to one stream, no new kept states, the cache shed, 503s.
const std = @import("std");
const LaneHost = @import("lane_host.zig").LaneHost;
const shed = @import("prompt_cache_shed.zig").shed;

/// On the engine thread once a loop: a reading at most every `every_ns`, a change logged once, the cache acted on.
pub fn step(h: *LaneHost) void {
    if (h.pressure == null) return;
    const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
    h.lock();
    const g = &h.pressure.?;
    const change = g.poll(now);
    const level = g.level;
    h.unlock();
    const store = h.cache;
    const mark: ?u64 = if (store) |s| g.mark(s.budget) else null;
    if (change) |c| g.log.line(g.log.ctx, c, mark);
    const s = store orelse return;
    s.hold = level != .normal;
    if (s.hold) _ = shed(s, mark.?);
}

/// Whether a new request is refused: the governed level is critical (lock held).
pub fn refuses(h: *const LaneHost) bool {
    const g = h.pressure orelse return false;
    return g.level == .critical;
}

/// Whether admission holds: under pressure a stream already runs, so no other starts beside it (lock held).
pub fn holds(h: *const LaneHost) bool {
    const g = h.pressure orelse return false;
    return g.level != .normal and h.admitted.items.len > 0;
}
