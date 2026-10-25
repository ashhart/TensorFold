//! A shared round's kept DeltaNet states copied back into each session's first state rows.
const std = @import("std");
const mtl = @import("metal");

/// Pending rounds keep their rows; true when a session's state rows still need their copy.
pub fn keep(b: anytype) !bool {
    var needed = false;
    for (b.sessions) |*s| if (s.used) {
        if (s.round.pending) try s.keep(s.round.rows);
        needed = needed or s.m.state != 0 or s.m.state_row != 0;
    };
    return needed;
}

/// The copies at the head of `cb`, fenced before the next compute encoder; encode before `outputs` repoints cs[1].
pub fn encode(b: anytype, cb: mtl.CommandBuffer, fence: mtl.counters.Fence) void {
    const blit = cb.blit();
    for (b.sessions) |*s| if (s.used) s.normalizeGpu(blit);
    fence.updateBlit(blit);
    blit.end();
    reset(b);
}

/// The copies in their own command buffer, waited on.
pub fn now(b: anytype) !void {
    if (!try keep(b)) return;
    const cb = b.e.r.queue.commandBuffer();
    const blit = cb.blit();
    for (b.sessions) |*s| if (s.used) s.normalizeGpu(blit);
    blit.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |message| {
        std.log.err("session state copy failed: {s}", .{message});
        return error.GpuFailed;
    }
    reset(b);
}

fn reset(b: anytype) void {
    for (b.sessions) |*s| if (s.used) {
        s.m.state = 0;
        s.m.state_row = 0;
    };
}
