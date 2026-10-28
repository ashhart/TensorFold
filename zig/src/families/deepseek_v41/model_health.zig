//! Readiness and process failure share the same owned model supervisor on every rank.
const std = @import("std");
const Model = @import("cuda_model.zig").Model;

pub fn serving(ctx: *anyopaque) bool {
    const m: *Model = @ptrCast(@alignCast(ctx));
    return m.ctx.healthy() and (if (m.supervisor) |*s| !s.failed.load(.acquire) else true);
}
pub fn context(ctx: *anyopaque) bool {
    const m: *Model = @ptrCast(@alignCast(ctx));
    return m.ctx.healthy();
}
pub fn failed(ctx: *anyopaque, err: anyerror) void {
    const m: *Model = @ptrCast(@alignCast(ctx));
    m.session.fatal(if (err == error.CudaContextLost) "CUDA context" else if (err == error.StalledRound) "round progress" else "execution", err);
    std.c._exit(70);
}
