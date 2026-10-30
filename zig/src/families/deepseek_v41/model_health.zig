//! Readiness and process failure share the same owned model supervisor on every rank.
const std = @import("std");
const api = @import("engine_api");
const Model = @import("cuda_model.zig").Model;
const Progress = @import("round_progress.zig").Progress;

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
    if (m.session.fate) |*f| f.fatal(if (err == error.CudaContextLost) "CUDA context" else if (err == error.StalledRound) "round progress" else "execution", err);
    std.c._exit(70);
}

/// The stall watchdog's scopes reach the runner and follower only when it is on; off, a launch costs one null check.
pub fn watch(m: *Model) void {
    const on = if (m.supervisor) |s| s.stall_ns > 0 else false;
    m.runner.progress = if (on) &m.progress else null;
    m.f.progress = m.runner.progress;
}
/// The lane host's work scopes follow the watchdog; its fatal round errors reach the supervisor only under fail-fast.
pub fn attach(m: *Model, h: *api.LaneHost) void {
    h.activity = if (m.runner.progress != null) .{ .ctx = m, .begin = beginActivity, .end = endActivity } else null;
    h.failure = if (m.session.fate != null) .{ .ctx = m, .call = executionFailed } else null;
}
fn beginActivity(ctx: *anyopaque) void {
    const m: *Model = @ptrCast(@alignCast(ctx));
    _ = m.progress.begin(.round);
}
fn endActivity(ctx: *anyopaque) void {
    const m: *Model = @ptrCast(@alignCast(ctx));
    var scope: Progress.Scope = .{ .owner = &m.progress };
    scope.end();
}
fn executionFailed(ctx: *anyopaque, err: anyerror) void {
    const m: *Model = @ptrCast(@alignCast(ctx));
    if (@import("lockstep.zig").isFatal(err)) if (m.supervisor) |*s| s.report(err);
}
