//! ZONOS3 Mini's open on CUDA: the flags it serves, then the host.
const std = @import("std");
const api = @import("engine_api");
const zonos_host = @import("zonos_host.zig");
const Allocator = std.mem.Allocator;
const setting = @import("cuda_device.zig").setting;
const memory = @import("cuda_family_memory.zig");
const cuda = @import("cuda");

/// The pool as every CUDA family reads it (the reserve, TENSORFOLD_CUDA_MEMORY_LIMIT_GB), less what this process holds.
const Room = struct {
    var gpa: Allocator = undefined;
    var io: std.Io = undefined;

    fn room(_: *anyopaque, ctx: *const cuda.Context) anyerror!u64 {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        var problem: []const u8 = "";
        const pool = try memory.pool(arena.allocator(), io, ctx, &problem) orelse return error.UnreadableMemoryPool;
        return pool.room(cuda.usage(false).device);
    }
};

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const settings = @import("sys").environment;
    if (try settings.refusal(a, settings.Process{}, .{ .cuda = true, .slide = o.slide, .device = o.device, .segments = o.segments })) |why| {
        problem.* = why;
        return null;
    }
    if (try o.slideRefusal(a, "CUDA")) |why| {
        problem.* = why;
        return null;
    }
    if (o.cfg_enabled or o.cfg_batched) {
        problem.* = "ZONOS3 Mini is served without guidance: serve without --cfg";
        return null;
    }
    if (o.tp != 1 or o.rank != 0 or (o.lanes_fixed and o.lanes != 1)) {
        problem.* = "ZONOS3 Mini serves one request at a time on one CUDA device: use --parallel 1, without tensor parallelism";
        return null;
    }
    if (o.silence.enabled()) {
        problem.* = "ZONOS3 Mini has no silence guard: serve without the silence flags";
        return null;
    }
    const window = o.context orelse (zonos_host.Options{}).context;
    _ = zonos_host.budget(std.math.cast(u32, window) orelse 0, o.max_frames) catch |e| {
        problem.* = switch (e) {
            error.BadZonosContext => "ZONOS3 Mini supports --context 1025 through 32768",
            error.BadZonosFrames => "ZONOS3 Mini's --max-frames must be 12 through 8192 and below --context",
        };
        return null;
    };
    if (o.vision != null or o.precision != null or o.record != null or o.recall or o.learn != null or
        o.learn_conversations or o.speed_up != null or o.slide or o.drafter != null or o.mtp_drafts != null or o.mtp_confidence != null or
        (o.prompt_cache_gib != null and o.prompt_cache_gib.? != 0) or
        (o.keep != null and o.keep.? != 0) or o.policy.len != 0)
    {
        problem.* = "ZONOS3 Mini does not support learned weights, speculative drafts, retained prompt caches, or text-engine policies";
        return null;
    }
    const device = try setting(a, o.device, "TF_CUDA_DEVICE", 0, problem) orelse return null;
    const segments = try setting(a, o.segments, "TF_CUDA_SEGMENTS", 1, problem) orelse return null;
    if (segments != 1) {
        problem.* = "ZONOS3 Mini uses one prompt segment at a time: use --segments 1";
        return null;
    }
    Room.gpa = gpa;
    Room.io = io;
    var refusal: zonos_host.Refusal = .{};
    const admit: zonos_host.Admit = .{ .ptr = &Room.gpa, .room = Room.room };
    var held = try @import("cuda_schedule.zig").hold(@intCast(device)); // TF_CUDA_SCHED, before the engine's context starts
    defer if (held) |*d| d.close();
    return zonos_host.open(gpa, io, o.dir, .{ .context = @intCast(window), .device = device, .max_frames = o.max_frames, .admit = admit, .refusal = &refusal }) catch |e| {
        if (e == error.InsufficientDeviceMemory) {
            const text = "ZONOS3 Mini needs {d:.2} GiB of device memory at --context {d}; {d:.2} GiB is free for it after the reserve";
            problem.* = try std.fmt.allocPrint(a, text, .{ memory.toGib(refusal.need), window, memory.toGib(refusal.room) });
            return null;
        }
        problem.* = try std.fmt.allocPrint(a, "the native ZONOS3 Mini CUDA engine could not load the checkpoint ({s})", .{@errorName(e)});
        return null;
    };
}
