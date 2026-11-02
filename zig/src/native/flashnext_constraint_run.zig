//! One constrained job on the ordinary shared lane host, joined before its borrowed backend or request can leave.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");

pub const Hooks = struct {
    sink: api.Sink,
    cancelled: *const fn (*anyopaque) bool,
};

pub const Finished = struct {
    reason: api.Reason = .failed,
    stats: api.Stats = .{},
    message: [1024]u8 = undefined,
    len: usize = 0,
    pub fn text(f: *const Finished) []const u8 {
        return f.message[0..f.len];
    }
};

pub fn run(gpa: std.mem.Allocator, io: std.Io, target: lanes.backend.Backend, facts: lanes.Model, info: api.Info, cache: ?*api.prompt_cache.Store, id: api.Id, request: *const api.Request, hooks: Hooks) !Finished {
    if (request.structure == null) return error.UnconstrainedRequest;
    if (hooks.cancelled(hooks.sink.ctx)) return .{ .reason = .cancelled };
    var cfg = try lanes.Config.init(gpa, facts, 16, 15);
    defer cfg.deinit(gpa);
    var clock: lanes.backend.WallClock = .{ .io = io };
    var core = lanes.Engine.init(gpa, &cfg, target, clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, io, &core, info);
    host.cache = cache;
    try host.start();
    defer host.stop();
    const Completion = struct {
        hooks: Hooks,
        done: std.atomic.Value(bool) = .init(false),
        finished: Finished = .{},
        fn event(ptr: *anyopaque, job: api.Id, e: *const api.Event) void {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            if (e.* == .finished) {
                const f = e.finished;
                c.finished.reason = f.reason;
                c.finished.stats = f.stats;
                c.finished.len = @min(c.finished.message.len, f.message.len);
                @memcpy(c.finished.message[0..c.finished.len], f.message[0..c.finished.len]);
                c.done.store(true, .release);
            } else c.hooks.sink.event(c.hooks.sink.ctx, job, e);
        }
    };
    var completion: Completion = .{ .hooks = hooks };
    try host.engine().submit(id, request, .{ .ctx = &completion, .event = Completion.event });
    var cancelled = false;
    while (!completion.done.load(.acquire)) {
        if (!cancelled and hooks.cancelled(hooks.sink.ctx)) {
            host.engine().cancel(id);
            cancelled = true;
        }
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return completion.finished;
}
