//! A bounded asynchronous scripted-engine turn, used only by the request ownership regressions.
const std = @import("std");
const api = @import("engine_api");
const base = @import("cortheon_test.zig");
const Conn = @import("../http_conn.zig").Conn;
const io = std.testing.io;

pub const Engine = struct {
    original: *base.Eng,
    at: ?usize = null,
    delay_ms: ?u32 = null,
    disconnect: ?*Conn = null,
    fail: bool = false,
    submissions: usize = 0,
    cancellations: usize = 0,
    active: std.atomic.Value(bool) = .init(false),
    mutex: std.Io.Mutex = .init,
    held: ?Held = null,
    worker: ?std.Thread = null,
    const Held = struct { id: api.Id, request: *const api.Request, sink: api.Sink };

    pub fn engine(e: *Engine) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        const b = e.original.engine();
        return b.vtable.info(b.ctx);
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        e.original.sample();
        const index = e.submissions;
        e.submissions += 1;
        if (e.at != index) {
            const b = e.original.engine();
            return b.vtable.submit(b.ctx, id, request, sink);
        }
        if (e.fail) {
            sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .failed, .message = "recorded stream failure" } });
            return;
        }
        e.held = .{ .id = id, .request = request, .sink = sink };
        e.active.store(true, .release);
        if (e.disconnect) |conn| conn.broken = true;
        e.worker = std.Thread.spawn(.{}, watch, .{e}) catch {
            e.held = null;
            e.active.store(false, .release);
            return error.Busy;
        };
    }
    fn take(e: *Engine) ?Held {
        e.mutex.lockUncancelable(io);
        defer e.mutex.unlock(io);
        const h = e.held;
        e.held = null;
        e.active.store(false, .release);
        return h;
    }
    fn watch(e: *Engine) void {
        const bound = e.delay_ms orelse 1000;
        var elapsed: u32 = 0;
        while (elapsed < bound and e.active.load(.acquire)) : (elapsed += 1)
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        const h = e.take() orelse return;
        if (e.delay_ms != null) {
            const b = e.original.engine();
            b.vtable.submit(b.ctx, h.id, h.request, h.sink) catch
                h.sink.event(h.sink.ctx, h.id, &.{ .finished = .{ .reason = .failed, .message = "recorded delayed refusal" } });
        } else h.sink.event(h.sink.ctx, h.id, &.{ .finished = .{ .reason = .failed, .message = "bounded fixture watchdog" } });
    }
    fn cancel(ctx: *anyopaque, id: api.Id) void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        e.original.sample();
        e.cancellations += 1;
        if (e.take()) |h| h.sink.event(h.sink.ctx, id, &.{ .finished = .{ .reason = .cancelled } });
    }
    fn status(ctx: *anyopaque, out: *api.Status, _: []u32) void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        out.* = .{ .running = @intFromBool(e.active.load(.acquire)) };
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    pub fn join(e: *Engine) void {
        if (e.worker) |thread| thread.join();
        e.worker = null;
    }
};
