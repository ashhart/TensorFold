//! Native CUDA readiness reads family state and context faults without binding CUDA or waiting on a device.
const api = @import("engine_api");

pub fn source(h: anytype) api.HealthSource {
    return .{ .ctx = h, .read = struct {
        fn read(ctx: *anyopaque) bool {
            const host: @TypeOf(h) = @ptrCast(@alignCast(ctx));
            if (host.gpu) |gpu| if (!gpu.ctx.healthy()) return false;
            return if (host.health) |family| family.read(family.ctx) else true;
        }
    }.read };
}
