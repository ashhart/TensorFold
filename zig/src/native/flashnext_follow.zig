//! A terminal follower failure stops the native pair process without freeing live GPU aliases.
const std = @import("std");

pub fn stop(h: anytype) void {
    if (h.follower) |thread| {
        h.eng.stopFollowing();
        thread.join();
        h.follower = null;
    }
}

pub fn failed(eng: anytype, err: anyerror) noreturn {
    std.log.err("speed-up mode: following rank 0 ended: {s}", .{@errorName(err)});
    if (eng.r.tp) |tp| {
        tp.handoff.fail(tp.seq, err);
        tp.stopService();
    }
    std.process.exit(1);
}
