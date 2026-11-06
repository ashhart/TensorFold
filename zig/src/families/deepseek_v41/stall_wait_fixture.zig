//! A host-owned CUDA wait blocks until the fixture releases it; no device call or kernel executes.
const std = @import("std");
const cuda = @import("cuda");
const t = std.testing;
pub const Wait = struct {
    entered: std.atomic.Value(bool) = .init(false),
    release: std.atomic.Value(bool) = .init(false),
    result: cuda.abi.Result = cuda.abi.error_not_ready,
    var current: ?*Wait = null;
    pub fn install(w: *Wait, d: *cuda.Driver) void {
        current = w;
        d.api.cuStreamSynchronize = sync;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        const w = current.?;
        w.entered.store(true, .release);
        while (!w.release.load(.acquire)) std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        return w.result;
    }
};
pub fn wait(flag: *const std.atomic.Value(bool)) !void {
    const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (!flag.load(.acquire)) {
        if (std.Io.Clock.awake.now(t.io).toNanoseconds() >= until) return error.StallFixtureTimeout;
        try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
}
pub const Failure = struct {
    done: std.atomic.Value(bool) = .init(false),
    err: ?anyerror = null,
    pub fn healthy(_: *anyopaque) bool {
        return true;
    }
    pub fn fail(ctx: *anyopaque, err: anyerror) void {
        const f: *Failure = @ptrCast(@alignCast(ctx));
        f.err = err;
        f.done.store(true, .release);
    }
};
