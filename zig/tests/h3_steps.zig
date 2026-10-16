//! tf-h3-dit's schedule made here: the checkpoint's contract, N rungs, the tables, and the case's tables compared.
const std = @import("std");
const h3 = @import("h3");

fn read(init: std.process.Init, dir: []const u8, comptime rest: []const u8) ?[]u8 {
    const a = init.arena.allocator();
    const path = std.fmt.allocPrint(a, "{s}/../" ++ rest, .{dir}) catch return null;
    return std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(1 << 20)) catch null;
}

/// The run's Euler pairs (seen, ratio) for each step, video and audio, flat.
pub const Made = struct { video: []f32, audio: []f32 };

/// The released model's audio sigma shift.
const audio_shift: f32 = 3.0;

/// Re-voicing: the picture held, the audio rows down the released model's grid of n + 1 points.
pub fn revoice(init: std.process.Init, model: *h3.Model, case: *const h3.Tensors, n: usize) !Made {
    const a = init.arena.allocator();
    const audio = try h3.schedule.Schedule.fromPoints(a, n + 1, audio_shift);
    const steps = try h3.modulation.reschedule(model, a, audio, audio, true);
    std.debug.print("re-voice schedule made here: {d} audio steps, {d} timesteps\n", .{ audio.timesteps.len, model.lines / h3.schedule.modalities });
    compare(model, case);
    const k = audio.timesteps.len;
    return .{ .video = @as([*]f32, @ptrCast(steps.video.ptr))[0 .. 2 * k], .audio = @as([*]f32, @ptrCast(steps.audio.ptr))[0 .. 2 * k] };
}

/// The released model's video sigma shift.
const video_shift: f32 = 12.0;

pub fn reschedule(init: std.process.Init, model: *h3.Model, case: *const h3.Tensors, dir: []const u8, n: usize) !Made {
    const a = init.arena.allocator();
    // the released weights (FL2VA, Ref2VA) sample their own grid of n + 1 points
    if (model.base) {
        const video = try h3.schedule.Schedule.fromPoints(a, n + 1, video_shift);
        const audio = try h3.schedule.Schedule.fromPoints(a, n + 1, audio_shift);
        const steps = try h3.modulation.reschedule(model, a, video, audio, false);
        std.debug.print("released-model schedule made here: {d} steps, {d} timesteps\n", .{ n, model.lines / h3.schedule.modalities });
        compare(model, case);
        const k = video.timesteps.len;
        return .{ .video = @as([*]f32, @ptrCast(steps.video.ptr))[0 .. 2 * k], .audio = @as([*]f32, @ptrCast(steps.audio.ptr))[0 .. 2 * k] };
    }
    const json = read(init, dir, "fastvideo_inference.json") orelse return error.NoContract;
    const c = try h3.schedule.contract(a, json, read(init, dir, "scheduler/scheduler_config.json"), read(init, dir, "audio_scheduler/scheduler_config.json"));
    const nodes = try h3.schedule.spread(a, c.nodes, n);
    const video = try h3.schedule.Schedule.fromNodes(a, nodes, c.video_shift);
    const audio = try h3.schedule.Schedule.fromNodes(a, nodes, c.audio_shift);
    const started = std.Io.Clock.awake.now(init.io);
    const steps = try h3.modulation.reschedule(model, a, video, audio, false);
    const ms = @divTrunc(started.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds(), std.time.ns_per_ms);
    std.debug.print("schedule made here: {d} steps, shifts {d}/{d}, {d} timesteps, tables in {d} ms\n", .{ n, c.video_shift, c.audio_shift, model.lines / h3.schedule.modalities, ms });
    compare(model, case);
    return .{ .video = @as([*]f32, @ptrCast(steps.video.ptr))[0 .. 2 * n], .audio = @as([*]f32, @ptrCast(steps.audio.ptr))[0 .. 2 * n] };
}

/// When the case was exported for the same schedule: its tables against these, value for value.
fn compare(model: *h3.Model, case: *const h3.Tensors) void {
    const lines = (case.names.get("tables.0") orelse return).shape[1];
    if (lines != model.lines) return;
    var same: usize = 0;
    var total: usize = 0;
    var worst: f32 = 0;
    var largest: f32 = 0;
    for (model.blocks, 0..) |b, i| {
        var name: [32]u8 = undefined;
        const key = std.fmt.bufPrint(&name, "tables.{d}", .{i}) catch return;
        const n = 6 * lines * model.hidden;
        const theirs = case.at(key, .bf16, n) catch return;
        const want = case.values(u16, theirs, n);
        const got = b.tables.buffer.slice(u16, n);
        for (want, got) |x, y| {
            same += @intFromBool(x == y);
            const p: f32 = @bitCast(@as(u32, x) << 16);
            const q: f32 = @bitCast(@as(u32, y) << 16);
            worst = @max(worst, @abs(p - q));
            largest = @max(largest, @abs(p));
        }
        total += n;
    }
    std.debug.print("block tables against the case's: {d} of {d} values bit-identical, largest difference {e:.3} (largest value {e:.3})\n", .{ same, total, worst, largest });
}
