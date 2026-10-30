const std = @import("std");
const api = @import("engine_api");
const drama_host = @import("drama_host.zig");
const Allocator = std.mem.Allocator;
const setting = @import("cuda_device.zig").setting;
const getenv = @import("cuda_device.zig").getenv;

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (o.cfg_enabled or o.cfg_batched) {
        problem.* = "this engine revision implements Drama CFG on Metal only; use the pinned CUDA quant engine on Linux";
        return null;
    }
    if (o.tp != 1 or o.rank != 0 or (o.lanes_fixed and (o.lanes == 0 or o.lanes > 8))) {
        problem.* = "Drama 3 serves 1 through 8 frame lanes on one CUDA device; tensor parallelism is unsupported";
        return null;
    }
    const window = o.context orelse 8192;
    if (window < 1025 or window > 8192) {
        problem.* = "Drama 3 supports --context 1025 through 8192";
        return null;
    }
    if (frameProblem(@intCast(window), o.max_frames)) |why| {
        problem.* = why;
        return null;
    }
    if (o.vision != null or o.precision != null or o.record != null or o.recall or o.learn != null or
        o.learn_conversations or o.speed_up != null or o.slide or o.drafter != null or o.mtp_drafts != null or o.mtp_confidence != null or
        (o.prompt_cache_gib != null and o.prompt_cache_gib.? != 0) or
        (o.keep != null and o.keep.? != 0) or o.policy.len != 0)
    {
        problem.* = "Drama 3 does not support learned weights, speculative drafts, retained prompt caches, or text-engine policies";
        return null;
    }
    const device = try setting(a, o.device, "TF_CUDA_DEVICE", 0, problem) orelse return null;
    const prefill_rows = try setting(a, null, "TF_DRAMA_PREFILL_ROWS", (drama_host.Options{}).prefill_rows, problem) orelse return null;
    if (prefill_rows == 0 or prefill_rows > 1024) {
        problem.* = "TF_DRAMA_PREFILL_ROWS must be 1 through 1024";
        return null;
    }
    const segments = try setting(a, o.segments, "TF_CUDA_SEGMENTS", 1, problem) orelse return null;
    if (segments != 1) {
        problem.* = "Drama 3 uses one prompt segment at a time: use --segments 1";
        return null;
    }
    if (getenv("TENSORFOLD_CUDA_KERNELS")) |path| if (path.len > 0 and !std.mem.eql(u8, path, "native")) {
        problem.* = "Drama 3 uses embedded native kernels, not captured kernel sets: unset TENSORFOLD_CUDA_KERNELS or set it to native";
        return null;
    };
    return drama_host.open(gpa, io, o.dir, "", .{
        .context = @intCast(window),
        .device = device,
        .lanes = if (o.lanes_fixed) o.lanes else 2,
        .silence = o.silence,
        .max_frames = o.max_frames,
        .prefill_rows = prefill_rows,
    }) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the native Drama 3 CUDA engine could not load the checkpoint ({s})", .{@errorName(e)});
        return null;
    };
}

pub fn openMetal(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const window = o.context orelse 8192;
    if (window < 1025 or window > 8192 or o.tp != 1 or o.rank != 0 or
        (o.lanes_fixed and (o.lanes == 0 or o.lanes > 8)) or
        o.vision != null or o.precision != null or o.record != null or o.recall or o.learn != null or
        o.learn_conversations or o.speed_up != null or o.slide or o.drafter != null or
        o.mtp_drafts != null or o.mtp_confidence != null or o.policy.len != 0 or
        (o.prompt_cache_gib != null and o.prompt_cache_gib.? != 0) or (o.keep != null and o.keep.? != 0) or
        (o.device != null and o.device.? != 0) or (o.segments != null and o.segments.? != 1))
    {
        problem.* = "Drama 3 needs context 1025..8192, 1..8 frame lanes and no text, vision, TP or cache modifiers";
        return null;
    }
    if (frameProblem(@intCast(window), o.max_frames)) |why| {
        problem.* = why;
        return null;
    }
    return drama_host.open(gpa, io, o.dir, "", .{
        .context = @intCast(window),
        .lanes = if (o.lanes_fixed) o.lanes else 2,
        .silence = o.silence,
        .max_frames = o.max_frames,
        .cfg_enabled = o.cfg_enabled,
        .cfg_batched = o.cfg_batched,
    }) catch |err| {
        problem.* = if (err == error.OverMemoryLimit)
            try std.fmt.allocPrint(a, "Drama 3 needs more memory than this Mac's load limit allows (70% of RAM, or DRAMA_LOAD_LIMIT_GB); the log names the sizes", .{})
        else
            try std.fmt.allocPrint(a, "Drama 3 Metal checkpoint load failed ({s})", .{@errorName(err)});
        return null;
    };
}

fn frameProblem(context: u32, frames: ?u32) ?[]const u8 {
    _ = @import("drama_budget.zig").budget(context, frames) catch |err| return switch (err) {
        error.BadDramaMaxFrames => "Drama 3 supports --max-frames 1 through 8190",
        error.NoDramaPromptRoom => "Drama 3 --max-frames must leave prompt room: use --max-frames < --context - 1",
        error.BadDramaContext => "Drama 3 supports --context 1025 through 8192",
    };
    return null;
}
