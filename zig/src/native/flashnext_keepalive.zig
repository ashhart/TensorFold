//! Pair idle probes finish outside request work; single-Mac keepalive retains its existing target.
const Host = @import("flashnext_host.zig").Host;
const mtl = @import("metal");

pub fn tick(ctx: *anyopaque) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    const tp = h.eng.r.tp.?;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    _ = tp.request_gate.keepWarm(&h.warm) catch |err| {
        tp.handoff.fail(tp.seq, err);
        return;
    };
}
