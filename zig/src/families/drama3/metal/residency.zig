//! The speech host holds every registered Metal address, including codec and stream scratch, on one set.
const std = @import("std");
const mtl = @import("metal");
const Gpu = @import("gpu.zig").Gpu;
pub const Warm = struct {
    resident: ?mtl.resident.Resident = null,
    target: mtl.keepalive.Target = undefined,
    pub fn hold(w: *Warm, g: *Gpu, io: std.Io) !void {
        w.resident = mtl.resident.Resident.hold(g.gpa, io, g.device, g.queue, g.bufs.items[1..], "drama3") catch |err| blk: {
            std.log.warn("drama3: residency skipped ({s})", .{@errorName(err)});
            break :blk null;
        };
        if (w.resident) |*r| g.setResident(r.follow());
        w.target = if (w.resident) |*r| r.target() else .{ .queue = g.queue };
    }
    pub fn drop(w: *Warm, g: *Gpu) void {
        g.setResident(null);
        if (w.resident) |*r| r.deinit();
        w.resident = null;
        w.target = .{ .queue = g.queue };
    }
};
