//! One request's frames: the prompt, then a frame a step until the end-of-audio countdown or the frame budget.
const std = @import("std");
const cfg = @import("config.zig");
const sampler = @import("sampler.zig");

const C = cfg.codebooks;
const W = cfg.frame_width;

pub const Result = struct {
    /// Generated rows [n][11] (codes, then the text placeholder), the finishing row included; caller frees.
    rows: [][W]i32,
    /// The first end-of-audio frame among the generated rows, aligned across the delayed codebooks.
    eos_frame: ?usize,
};

/// Codebooks whose delayed stream has not started are padding, unless the prompt already ends in audio.
fn coldStart(row: *[C]i32, generated: usize, has_audio_prefix: bool) void {
    if (has_audio_prefix or generated >= C - 1) return;
    for (row[generated + 1 ..]) |*code| code.* = cfg.audio_pad_id;
}

/// `engine` has run(rows, replaced, probe) and lastLogits(out): the backbone, or a stand-in under test.
pub fn generate(gpa: std.mem.Allocator, engine: anytype, prompt: []const [W]i32, replaced: anytype, sampling: sampler.Sampling, max_frames: usize) !Result {
    return generateWatched(gpa, engine, prompt, replaced, sampling, max_frames, null);
}

/// As generate; `watch`, when given, has row(rows so far, eos frame, finished) and sees every generated row.
pub fn generateWatched(gpa: std.mem.Allocator, engine: anytype, prompt: []const [W]i32, replaced: anytype, sampling: sampler.Sampling, max_frames: usize, watch: anytype) !Result {
    return generateGuided(gpa, engine, prompt, replaced, &.{}, replaced, 1, sampling, max_frames, watch);
}

/// speaker_cfg.guided_logits: unconditional + scale * (conditional - unconditional), in fp32 before the sampler.
pub fn guide(conditional: *[cfg.head_rows]f32, unconditional: *const [cfg.head_rows]f32, scale: f32) void {
    for (conditional, unconditional) |*c, u| c.* = if (scale == 0) u else u + scale * (c.* - u);
}

/// As generateWatched, each frame sampled from its logits mixed with those of `twin` on the engine's second lane.
pub fn generateGuided(gpa: std.mem.Allocator, engine: anytype, prompt: []const [W]i32, replaced: anytype, twin: []const [W]i32, twin_replaced: anytype, scale: f32, sampling: sampler.Sampling, max_frames: usize, watch: anytype) !Result {
    if (prompt.len == 0 or max_frames == 0) return error.NoRows;
    const logits = try gpa.create([C][cfg.audio_vocab]f32);
    defer gpa.destroy(logits);
    const guided = twin.len != 0;
    const unconditional: ?*[cfg.head_rows]f32 = if (guided) try gpa.create([cfg.head_rows]f32) else null;
    defer if (unconditional) |u| gpa.destroy(u);
    var rows: std.ArrayList([W]i32) = .empty;
    errdefer rows.deinit(gpa);
    var codes: std.ArrayList([C]i32) = .empty;
    defer codes.deinit(gpa);
    var has_audio_prefix = false;
    for (prompt[prompt.len - 1][0..C]) |code| has_audio_prefix = has_audio_prefix or code != cfg.audio_pad_id;
    try engine.run(prompt, replaced, null);
    try engine.lastLogits(@ptrCast(logits));
    if (unconditional) |u| {
        engine.swap();
        defer engine.swap();
        try engine.run(twin, twin_replaced, null);
        try engine.lastLogits(u);
    }
    var eos_frame: ?usize = null;
    var countdown: usize = 0;
    while (true) {
        if (unconditional) |u| guide(@ptrCast(logits), u, scale);
        const generated = rows.items.len;
        var row = try sampler.sample(gpa, logits, codes.items, generated, cfg.eoa_id, sampling);
        coldStart(&row, generated, has_audio_prefix);
        try codes.append(gpa, row);
        var full: [W]i32 = undefined;
        full[0..C].* = row;
        full[C] = cfg.text_vocab;
        try rows.append(gpa, full);
        var finished = rows.items.len >= max_frames;
        if (eos_frame == null) {
            var last: ?usize = null;
            for (row, 0..) |code, cb| if (code == cfg.eoa_id) {
                last = cb;
            };
            if (last) |cb| {
                eos_frame = generated -| cb;
                countdown = C + 1;
            }
        }
        if (countdown > 0) {
            countdown -= 1;
            if (countdown == 0) finished = true;
        }
        if (@TypeOf(watch) != @TypeOf(null)) try watch.row(rows.items, eos_frame, finished);
        if (finished) break;
        try engine.run(rows.items[generated..], &.{}, null);
        try engine.lastLogits(@ptrCast(logits));
        if (unconditional) |u| {
            engine.swap();
            defer engine.swap();
            try engine.run(rows.items[generated..], &.{}, null);
            try engine.lastLogits(u);
        }
    }
    return .{ .rows = try rows.toOwnedSlice(gpa), .eos_frame = eos_frame };
}

test "generation: an end-of-audio code in codebook 0 ends the request eleven rows later" {
    const t = std.testing;
    const Fake = struct {
        calls: usize = 0,
        pub fn run(f: *@This(), rows: []const [W]i32, _: anytype, _: anytype) !void {
            f.calls += rows.len;
        }
        pub fn swap(_: *@This()) void {}
        pub fn lastLogits(f: *@This(), out: *[cfg.head_rows]f32) !void {
            @memset(out, 0);
            // row 2 on: codebook 0 says end of audio; the others say code 7
            for (0..C) |cb| {
                const best: usize = if (cb == 0 and f.calls >= 3) cfg.eoa_id else 7;
                out[cb * cfg.audio_vocab + best] = 5;
            }
        }
    };
    var fake: Fake = .{};
    var prompt = [_][W]i32{@splat(cfg.audio_pad_id)};
    prompt[0][C] = 2;
    const r = try generate(t.allocator, &fake, &prompt, @as([]const u8, &.{}), .{ .temperature = 0, .repetition_penalty = 1 }, 100);
    defer t.allocator.free(r.rows);
    try t.expectEqual(@as(?usize, 2), r.eos_frame);
    try t.expectEqual(@as(usize, 2 + C + 1), r.rows.len);
    // the cold start pads the codebooks whose delayed stream has not begun
    try t.expectEqual(@as(i32, 7), r.rows[0][0]);
    try t.expectEqual(@as(i32, cfg.audio_pad_id), r.rows[0][1]);
    try t.expectEqual(@as(i32, 7), r.rows[1][1]);
}

test "generation: a guided request runs each row on both lanes and samples the mixed logits" {
    const t = std.testing;
    // lane 0 prefers code 7, lane 1 code 9 more strongly: scale 1 keeps 7, scale 0 takes 9, scale 3 leans further to 7
    const Fake = struct {
        lane: usize = 0,
        rows: [2]usize = .{ 0, 0 },
        pub fn run(f: *@This(), rows: []const [W]i32, _: anytype, _: anytype) !void {
            f.rows[f.lane] += rows.len;
        }
        pub fn swap(f: *@This()) void {
            f.lane = 1 - f.lane;
        }
        pub fn lastLogits(f: *@This(), out: *[cfg.head_rows]f32) !void {
            @memset(out, 0);
            for (0..C) |cb| {
                out[cb * cfg.audio_vocab + 7] = if (f.lane == 0) 2 else 1;
                out[cb * cfg.audio_vocab + 9] = if (f.lane == 0) 1 else 4;
            }
        }
    };
    var prompt = [_][W]i32{ @splat(cfg.audio_pad_id), @splat(cfg.audio_pad_id) };
    prompt[0][C] = 2;
    prompt[1][C] = 3;
    for ([_]f32{ 0, 0.5, 3 }, [_]i32{ 9, 9, 7 }) |scale, want| {
        var fake: Fake = .{};
        const r = try generateGuided(t.allocator, &fake, &prompt, @as([]const u8, &.{}), prompt[1..], @as([]const u8, &.{}), scale, .{ .temperature = 0, .repetition_penalty = 1 }, 3, null);
        defer t.allocator.free(r.rows);
        try t.expectEqual(want, r.rows[0][0]);
        // the prompt on each lane, then every generated row but the last on both
        try t.expectEqual([2]usize{ 2 + 2, 1 + 2 }, fake.rows);
        try t.expectEqual(@as(usize, 0), fake.lane);
    }
    var mixed: [cfg.head_rows]f32 = @splat(2);
    const un: [cfg.head_rows]f32 = @splat(0.5);
    guide(&mixed, &un, 1.5);
    try t.expectEqual(@as(f32, 0.5 + 1.5 * 1.5), mixed[0]);
}
