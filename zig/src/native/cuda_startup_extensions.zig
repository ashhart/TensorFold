//! Scoped qualification precedes unchanged upstream startup and shares its actual guards with injected controls.
const std = @import("std");
const api = @import("engine_api");

pub fn beforeStandard(a: std.mem.Allocator, o: api.Open, problem: *[]const u8, env: anytype) !bool {
    const settings = @import("sys").environment;
    if (try settings.refusal(a, env, .{ .cuda = true, .qwen = std.mem.eql(u8, o.model_type, "qwen3_5") or std.mem.eql(u8, o.model_type, "qwen3_5_moe"), .slide = o.slide, .device = o.device, .segments = o.segments })) |why| {
        problem.* = why;
        return true;
    }
    if (try @import("sys").cuda_qualification.refusal(o.model_type, @import("qwen_prefill_config").max_rows, env)) |why| {
        problem.* = why.message;
        return true;
    }
    return false;
}

pub fn startupWith(a: std.mem.Allocator, o: api.Open, problem: *[]const u8, env: anytype) !bool {
    if (try beforeStandard(a, o, problem, env)) return true;
    if (o.fsrRefusal("cuda")) |why| {
        problem.* = why;
        return true;
    }
    return false;
}
