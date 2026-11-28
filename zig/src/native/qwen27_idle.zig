//! Qwen's prepared extents leave and rejoin one residency set without changing buffer identity.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;

fn Adapter(comptime H: type) type {
    return struct {
        fn before(ctx: ?*anyopaque, buffers: []const mtl.Buffer) !void {
            const h: H = @ptrCast(@alignCast(ctx.?));
            if (h.resident) |*r| r.removeNow(buffers);
        }

        fn after(ctx: ?*anyopaque, buffers: []const mtl.Buffer) !void {
            const h: H = @ptrCast(@alignCast(ctx.?));
            if (h.resident) |*r| {
                if (!r.tryAddNow(buffers)) return error.ResidencyRejoinFailed;
            }
        }
    };
}

fn restore(ctx: *anyopaque) !void {
    const owner: *q.model.IdleRelease = @ptrCast(@alignCast(ctx));
    try owner.restore(std.Io.Clock.awake.now(owner.io).toNanoseconds());
}

fn idle(ctx: *anyopaque) void {
    const owner: *q.model.IdleRelease = @ptrCast(@alignCast(ctx));
    const was_released = owner.isReleased();
    owner.idle(std.Io.Clock.awake.now(owner.io).toNanoseconds());
    if (!was_released and owner.isReleased()) std.log.info("qwen27: target prepared extents purgeable_bytes={d}", .{owner.releasedBytes()});
}

/// Attach the same model owner to its stable served host; submission and the engine thread serialize calls.
pub fn attach(h: anytype, owner: *q.model.IdleRelease) void {
    const callbacks = Adapter(@TypeOf(h));
    owner.setEvents(.{ .ctx = h, .beforeRelease = callbacks.before, .afterRestore = callbacks.after });
    h.host.weight_release = .{ .ctx = owner, .restore = restore, .idle = idle };
}
