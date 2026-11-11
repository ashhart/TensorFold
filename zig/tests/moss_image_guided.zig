//! The guided denoise loop both MOSS-Image drivers share: two branches a step, the negative one skipping a block.
const std = @import("std");

/// One step's seconds, the two branches summed.
pub const Timing = struct { wall: f64 = 0, gpu: f64 = 0, host: f64 = 0 };

/// `x += (uncond + guidance * (cond - uncond)) * (sigmas[s + 1] - sigmas[s])` per step, both branches.
pub fn denoise(
    comptime B: type,
    a: std.mem.Allocator,
    model: *B.mi.Model,
    branches: *[2]B.mi.Prompt,
    x: []f32,
    timesteps: []const f32,
    sigmas: []const f32,
    guidance: f32,
    negative_skip: usize,
    timings: []Timing,
) !f64 {
    const count = x.len;
    const cond = try a.alloc(f32, count);
    const uncond = try a.alloc(f32, count);
    var total: f64 = 0;
    for (0..@min(timings.len, timesteps.len)) |s| {
        const c = try model.forward(&branches[0], x, timesteps[s], null, cond);
        const u = try model.forward(&branches[1], x, timesteps[s], negative_skip, uncond);
        const by = sigmas[s + 1] - sigmas[s];
        for (x, cond, uncond) |*o, pc, pu| o.* += (pu + guidance * (pc - pu)) * by;
        const wall = c.wall + u.wall;
        total += wall;
        timings[s] = .{ .wall = wall, .gpu = c.gpu + u.gpu, .host = c.host + u.host };
    }
    return total;
}
