//! FastH3's sampling schedule: the checkpoint's rungs or N spread from its first, the sigma shifts and the step plan.
const std = @import("std");
const geometry = @import("geometry.zig");

/// Keyframe rows are held at this timestep for every step.
pub const keyframe_noise: f32 = 0.999;
/// One modulation line per (timestep, modality): 0 video, 1 text, 2 audio.
pub const modalities = 3;

/// How a FastH3 checkpoint was trained to be sampled: its unshifted rungs and the two sigma shifts.
pub const Contract = struct { nodes: []f64, video_shift: f64, audio_shift: f64 };

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadContract,
    };
}

/// fastvideo_inference.json; a preview checkpoint leaves a shift to its scheduler folder, given as that JSON.
pub fn contract(a: std.mem.Allocator, json: []const u8, video_folder: ?[]const u8, audio_folder: ?[]const u8) !Contract {
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, json, .{});
    if (doc != .object) return error.BadContract;
    const field = doc.object.get("dmd_denoising_steps") orelse return error.BadContract;
    if (field != .array) return error.BadContract;
    if (field.array.items.len == 0) return error.H3BadSchedule;
    const steps = field.array.items;
    _ = try geometry.product(&.{ steps.len, @sizeOf(f64) });
    const nodes = try a.alloc(f64, steps.len);
    for (nodes, steps) |*n, s| n.* = try number(s) / 1000.0;
    try validNodes(nodes);
    return .{ .nodes = nodes, .video_shift = try shift(a, doc, "video_scheduler_shift", video_folder), .audio_shift = try shift(a, doc, "audio_scheduler_shift", audio_folder) };
}

fn shift(a: std.mem.Allocator, doc: std.json.Value, key: []const u8, folder: ?[]const u8) !f64 {
    if (doc.object.get(key)) |v| return number(v);
    const text = folder orelse return error.NoShift;
    const config = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    if (config != .object) return error.BadContract;
    return number(config.object.get("shift") orelse return error.NoShift);
}

/// `n` rungs spread evenly from the first of `nodes` towards 0, as the exporter's --steps does.
pub fn spread(a: std.mem.Allocator, nodes: []const f64, n: usize) ![]f64 {
    try validNodes(nodes);
    try geometry.dimension(n);
    _ = try geometry.product(&.{ n, @sizeOf(f64) });
    if (n == nodes.len) return a.dupe(f64, nodes);
    const out = try a.alloc(f64, n);
    for (out, 0..) |*o, i| o.* = nodes[0] * (1.0 - @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n)));
    return out;
}

/// Sigmas (n + 1, the last 0) and the timesteps the model sees (n), in float32.
pub const Schedule = struct {
    sigmas: []f32,
    timesteps: []f32,

    /// A distilled model's rungs: shifted in float64 and cast once, as FastVideo does.
    pub fn fromNodes(a: std.mem.Allocator, nodes: []const f64, s: f64) !Schedule {
        try validNodes(nodes);
        try validShift(s);
        const points = try geometry.sum(&.{ nodes.len, 1 });
        _ = try geometry.product(&.{ points, 4 });
        const sigmas = try a.alloc(f32, points);
        for (sigmas, 0..) |*o, i| {
            const raw = if (i < nodes.len) nodes[i] else 0.0;
            o.* = @floatCast(s * raw / (1.0 + (s - 1.0) * raw));
        }
        const timesteps = try a.alloc(f32, nodes.len);
        for (timesteps, sigmas[0..nodes.len]) |*t, sg| t.* = 1.0 - sg;
        return .{ .sigmas = sigmas, .timesteps = timesteps };
    }

    /// The released model's grid: torch's float32 linspace(1, 0, points), shifted in float32, repeats dropped.
    pub fn fromPoints(a: std.mem.Allocator, points: usize, s: f32) !Schedule {
        if (points < 2) return error.H3BadSchedule;
        try geometry.dimension(points);
        try validShift(s);
        _ = try geometry.product(&.{ points, 4 });
        const step: f32 = -1.0 / @as(f32, @floatFromInt(points - 1));
        const half = points / 2;
        var values: std.ArrayList(f32) = .empty;
        for (0..points) |i| {
            const fi: f64 = @floatFromInt(i);
            const base: f32 = @floatCast(if (i < half) 1.0 + @as(f64, step) * fi else -@as(f64, step) * @as(f64, @floatFromInt(points - 1 - i)));
            const shifted = (s * base) / (1.0 + (s - 1.0) * base);
            if (values.items.len == 0 or values.items[values.items.len - 1] != shifted) try values.append(a, shifted);
        }
        const sigmas = try values.toOwnedSlice(a);
        const timesteps = try a.alloc(f32, sigmas.len - 1);
        for (timesteps, sigmas[0..timesteps.len]) |*t, sg| t.* = 1.0 - sg;
        return .{ .sigmas = sigmas, .timesteps = timesteps };
    }

    /// Step `i`'s Euler pair: the sigma the model saw and the grid's ratio.
    pub fn euler(s: Schedule, i: usize) [2]f32 {
        return .{ 1.0 - s.timesteps[i], s.sigmas[i + 1] / s.sigmas[i] };
    }
};

/// Rows in the engine's order (text, keyframe, audio with held audio first, video); `hold` keeps all but audio still.
pub const Rows = struct { text: usize, condition: usize = 0, audio: usize, held_audio: usize = 0, video: usize, hold: bool = false };

/// The distinct timesteps of a run, sorted, and for every step each row's index into them.
pub const Plan = struct { table: []f32, index: []i32 };

/// Held reference audio rows are clean and pinned at this timestep.
pub const audio_condition: f32 = 1.0;

fn rowTime(r: Rows, row: usize, video_t: f32, audio_t: f32) f32 {
    const audio_at = r.text + r.condition;
    const audio_rows = row >= audio_at and row < audio_at + r.audio;
    if (r.hold and !audio_rows) return keyframe_noise;
    if (row < r.text) return video_t;
    if (row < audio_at) return @max(video_t, keyframe_noise);
    if (row < audio_at + r.held_audio) return audio_condition;
    if (row < audio_at + r.audio) return audio_t;
    return video_t;
}

pub fn plan(a: std.mem.Allocator, r: Rows, video: []const f32, audio: []const f32) !Plan {
    if (r.held_audio > r.audio) return error.H3HeldAudioExceedsTotal;
    const rows = try geometry.sum(&.{ r.text, r.condition, r.audio, r.video });
    try geometry.dimension(rows);
    if (video.len == 0 or video.len != audio.len) return error.H3BadSchedule;
    const lines = try geometry.product(&.{ video.len, 4, modalities });
    try geometry.dimension(lines);
    _ = try geometry.product(&.{ video.len, rows, 4 });
    for (video, audio) |vt, at| if (!std.math.isFinite(vt) or !std.math.isFinite(at) or
        vt < 0 or vt > 1 or at < 0 or at > 1) return error.H3BadSchedule;
    var seen: std.ArrayList(f32) = .empty;
    for (video, audio) |vt, at| for ([_]f32{ vt, if (r.hold) keyframe_noise else @max(vt, keyframe_noise), at, audio_condition }, 0..) |t, k| {
        // only the kinds of row this run has
        if ((k == 1 and r.condition == 0 and !r.hold) or (k == 3 and r.held_audio == 0) or (k == 0 and r.hold)) continue;
        if (std.mem.indexOfScalar(f32, seen.items, t) == null) try seen.append(a, t);
    };
    std.mem.sort(f32, seen.items, {}, std.sort.asc(f32));
    const index = try a.alloc(i32, video.len * rows);
    for (video, audio, 0..) |vt, at, step| for (0..rows) |row| {
        index[step * rows + row] = @intCast(std.mem.indexOfScalar(f32, seen.items, rowTime(r, row, vt, at)).?);
    };
    return .{ .table = try seen.toOwnedSlice(a), .index = index };
}

test "FastH3 8-Step V2's contract gives the exporter's Euler pairs, and 4 spread rungs fall evenly" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const json = "{\"dmd_denoising_steps\": [999, 874, 749, 624, 500, 375, 250, 125], \"video_scheduler_shift\": 10.0, \"audio_scheduler_shift\": 3.0}";
    const c = try contract(a, json, null, null);
    const v = try Schedule.fromNodes(a, c.nodes, c.video_shift);
    try std.testing.expectEqual(@as(usize, 8), v.timesteps.len);
    try std.testing.expectEqual(@as(f32, 0.0), v.sigmas[8]);
    // sigma 0.999 shifted by 10: 9.99 / 9.991
    try std.testing.expectEqual(@as(f32, @floatCast(9.99 / 9.991)), v.sigmas[0]);
    try std.testing.expectEqual(@as(f32, 1.0) - v.sigmas[0], v.timesteps[0]);
    // the exporter's (seen, ratio) pairs for this checkpoint, video then audio, as its case file holds them
    const want_video = [_][2]f32{ .{ 0.9998999238014221, 0.9858870506286621 }, .{ 0.9857884049415588, 0.9815242886543274 }, .{ 0.9675752520561218, 0.974774956703186 }, .{ 0.9431681036949158, 0.9638694524765015 }, .{ 0.9090909361839294, 0.9428571462631226 }, .{ 0.8571428656578064, 0.8974359035491943 }, .{ 0.7692307829856873, 0.7647058963775635 }, .{ 0.5882353186607361, 0.0 } };
    const want_audio = [_][2]f32{ .{ 0.9996664524078369, 0.9544668197631836 }, .{ 0.9541484713554382, 0.9427459836006165 }, .{ 0.8995196223258972, 0.9257609844207764 }, .{ 0.8327401876449585, 0.9006410241127014 }, .{ 0.75, 0.8571428656578064 }, .{ 0.6428571343421936, 0.7777777910232544 }, .{ 0.5, 0.6000000238418579 }, .{ 0.30000001192092896, 0.0 } };
    const au = try Schedule.fromNodes(a, c.nodes, c.audio_shift);
    for (want_video, want_audio, 0..) |wv, wa, i| {
        try std.testing.expectEqual(wv, v.euler(i));
        try std.testing.expectEqual(wa, au.euler(i));
    }
    const four = try spread(a, c.nodes, 4);
    try std.testing.expectEqualSlices(f64, &.{ 0.999, 0.999 * 0.75, 0.999 * 0.5, 0.999 * 0.25 }, four);
}

test "the released model's grid equals the Python schedule for 9 and 31 points" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const nine = try Schedule.fromPoints(arena.allocator(), 9, 3.0);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 0.9545454382896423, 0.8999999761581421, 0.8333333134651184, 0.75 }, nine.sigmas[0..5]);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 0.30000001192092896, 0.0 }, nine.sigmas[6..]);
    try std.testing.expectEqualSlices(f32, &.{ 0.0, 0.045454561710357666, 0.10000002384185791 }, nine.timesteps[0..3]);
    const many = try Schedule.fromPoints(arena.allocator(), 31, 12.0);
    try std.testing.expectEqual(@as(usize, 31), many.sigmas.len);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 0.9971346259117126, 0.9940828680992126, 0.9908256530761719, 0.9873417019844055 }, many.sigmas[0..5]);
    try std.testing.expectEqualSlices(f32, &.{ 0.46153849363327026, 0.292682945728302, 0.0 }, many.sigmas[28..]);
}

test "a held picture: every row but the generated audio at the keyframe level" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const r = Rows{ .text = 1, .audio = 2, .video = 1, .hold = true };
    const p = try plan(arena.allocator(), r, &.{ 0.5, 0.75 }, &.{ 0.25, 0.5 });
    try std.testing.expectEqualSlices(f32, &.{ 0.25, 0.5, 0.999 }, p.table);
    try std.testing.expectEqualSlices(i32, &.{ 2, 0, 0, 2, 2, 1, 1, 2 }, p.index);
}

test "a preview contract takes its shifts from the scheduler folders" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const c = try contract(arena.allocator(), "{\"dmd_denoising_steps\": [999, 500]}", "{\"shift\": 12.0}", "{\"shift\": 3}");
    try std.testing.expectEqual(@as(f64, 12.0), c.video_shift);
    try std.testing.expectEqual(@as(f64, 3.0), c.audio_shift);
    try std.testing.expectError(error.NoShift, contract(arena.allocator(), "{\"dmd_denoising_steps\": [999]}", null, null));
}

test "the plan: keyframe rows held at 0.999, held audio at 1, one sorted table of distinct timesteps" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const r = Rows{ .text = 1, .condition = 1, .audio = 2, .held_audio = 1, .video = 1 };
    const p = try plan(arena.allocator(), r, &.{ 0.5, 0.75 }, &.{ 0.25, 0.5 });
    try std.testing.expectEqualSlices(f32, &.{ 0.25, 0.5, 0.75, 0.999, 1.0 }, p.table);
    // step 0: text 0.5, keyframe 0.999, held audio 1, audio 0.25, video 0.5
    try std.testing.expectEqualSlices(i32, &.{ 1, 3, 4, 0, 1 }, p.index[0..5]);
    try std.testing.expectEqualSlices(i32, &.{ 2, 3, 4, 1, 2 }, p.index[5..10]);
}

fn validNodes(nodes: []const f64) !void {
    if (nodes.len == 0) return error.H3BadSchedule;
    try geometry.dimension(nodes.len);
    for (nodes) |n| if (!std.math.isFinite(n) or n <= 0 or n > 1) return error.H3BadSchedule;
}

fn validShift(s: f64) !void {
    if (!std.math.isFinite(s) or s <= 0) return error.H3BadSchedule;
}

pub fn validate(s: Schedule) !void {
    if (s.timesteps.len == 0 or s.sigmas.len == 0 or s.sigmas.len - 1 != s.timesteps.len)
        return error.H3BadSchedule;
    for (s.timesteps, 0..) |t, i| {
        if (!std.math.isFinite(t) or t < 0 or t >= 1 or !std.math.isFinite(s.sigmas[i]) or s.sigmas[i] <= 0 or
            !std.math.isFinite(s.sigmas[i + 1]) or s.sigmas[i + 1] < 0) return error.H3BadSchedule;
    }
}

pub fn pointsForSteps(n: usize) !usize {
    try geometry.dimension(n);
    return geometry.sum(&.{ n, 1 });
}
