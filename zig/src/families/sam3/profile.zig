const std = @import("std");
const gpu = @import("tensor_metal");

pub const Timing = struct {
    wall_seconds: f64 = 0,
    gpu_seconds: f64 = 0,
    wait_seconds: f64 = 0,
    allocation_seconds: f64 = 0,
    submissions: usize = 0,
};

pub const Frame = struct {
    preprocess: Timing = .{},
    trunk: Timing = .{},
    pyramid: Timing = .{},
    tracker: Timing = .{},
    memory: Timing = .{},
    readback: Timing = .{},
    state: Timing = .{},
};

pub const Stamp = struct {
    wall: f64 = 0,
    gpu_seconds: f64 = 0,
    wait_seconds: f64 = 0,
    allocation_seconds: f64 = 0,
    submissions: usize = 0,

    pub fn take(g: *const gpu.Gpu, enabled: bool) Stamp {
        if (!enabled) return .{};
        std.debug.assert(g.batch == null);
        return .{
            .wall = gpu.mtl.clock.seconds(),
            .gpu_seconds = g.gpu_seconds,
            .wait_seconds = g.wait_seconds,
            .allocation_seconds = g.allocation_seconds,
            .submissions = g.submissions,
        };
    }

    pub fn record(self: Stamp, g: *const gpu.Gpu, enabled: bool, timing: *Timing) void {
        if (!enabled) return;
        const end = take(g, true);
        timing.wall_seconds += end.wall - self.wall;
        timing.gpu_seconds += end.gpu_seconds - self.gpu_seconds;
        timing.wait_seconds += end.wait_seconds - self.wait_seconds;
        timing.allocation_seconds += end.allocation_seconds - self.allocation_seconds;
        timing.submissions += end.submissions - self.submissions;
    }
};
