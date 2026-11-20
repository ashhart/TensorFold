//! One request's frames: the prompt, then a frame a step until the end-of-audio countdown or the frame budget.
const std = @import("std");
const cfg = @import("config.zig");
const sampler = @import("sampler.zig");
const Steer = @import("steering.zig").Steer;

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
    var run = try Run.begin(gpa, engine, prompt, replaced, twin, twin_replaced, scale, sampling, max_frames);
    defer run.deinit();
    while (true) {
        const finished = try run.sample();
        if (@TypeOf(watch) != @TypeOf(null)) try watch.row(run.rows.items, run.eos_frame, finished);
        if (finished) break;
        try run.advance(engine);
    }
    return run.result();
}

/// A request in progress, a frame a turn; the caller puts the request's lane in use around `sample` and `advance`.
pub const Run = struct {
    gpa: std.mem.Allocator,
    logits: *[C][cfg.audio_vocab]f32,
    /// The twin's logits of a guided request: the prompt without its voice, on the lane engine.swap() exchanges.
    unconditional: ?*[cfg.head_rows]f32,
    scale: f32,
    sampling: sampler.Sampling,
    max_frames: usize,
    has_audio_prefix: bool,
    rows: std.ArrayList([W]i32) = .empty,
    codes: std.ArrayList([C]i32) = .empty,
    eos_frame: ?usize = null,
    countdown: usize = 0,
    /// A steered request's deltas by generated frame, and whether its twin takes them too (a speaker twin does).
    steer: ?Steering = null,

    pub const Steering = struct {
        ctx: *anyopaque,
        /// The deltas of the forward that takes generated row `frame` (1 on); none: that frame is not steered.
        at: *const fn (ctx: *anyopaque, frame: usize) []const Steer,
        twin: bool,
    };

    /// Runs the prompt (and the twin's, when it has rows) and holds the first frame's logits.
    pub fn begin(gpa: std.mem.Allocator, engine: anytype, prompt: []const [W]i32, replaced: anytype, twin: []const [W]i32, twin_replaced: anytype, scale: f32, sampling: sampler.Sampling, max_frames: usize) !Run {
        if (prompt.len == 0 or max_frames == 0) return error.NoRows;
        const logits = try gpa.create([C][cfg.audio_vocab]f32);
        errdefer gpa.destroy(logits);
        const unconditional: ?*[cfg.head_rows]f32 = if (twin.len != 0) try gpa.create([cfg.head_rows]f32) else null;
        errdefer if (unconditional) |u| gpa.destroy(u);
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
        return .{ .gpa = gpa, .logits = logits, .unconditional = unconditional, .scale = scale, .sampling = sampling, .max_frames = max_frames, .has_audio_prefix = has_audio_prefix };
    }

    /// A regeneration (LACI): `kept` rows of the prompt's end were generated before, and count as the sampler's past.
    pub fn keep(r: *Run, kept: []const [W]i32) !void {
        for (kept) |row| try r.codes.append(r.gpa, row[0..C].*);
    }

    pub fn deinit(r: *Run) void {
        r.rows.deinit(r.gpa);
        r.codes.deinit(r.gpa);
        if (r.unconditional) |u| r.gpa.destroy(u);
        r.gpa.destroy(r.logits);
    }

    /// The next row, appended to `rows`; true when it is the request's last.
    pub fn sample(r: *Run) !bool {
        if (r.unconditional) |u| guide(@ptrCast(r.logits), u, r.scale);
        const generated = r.rows.items.len;
        var row = try sampler.sample(r.gpa, r.logits, r.codes.items, r.codes.items.len, cfg.eoa_id, r.sampling);
        coldStart(&row, generated, r.has_audio_prefix);
        try r.codes.append(r.gpa, row);
        var full: [W]i32 = undefined;
        full[0..C].* = row;
        full[C] = cfg.text_vocab;
        try r.rows.append(r.gpa, full);
        var finished = r.rows.items.len >= r.max_frames;
        if (r.eos_frame == null) {
            var last: ?usize = null;
            for (row, 0..) |code, cb| if (code == cfg.eoa_id) {
                last = cb;
            };
            if (last) |cb| {
                r.eos_frame = generated -| cb;
                r.countdown = C + 1;
            }
        }
        if (r.countdown > 0) {
            r.countdown -= 1;
            if (r.countdown == 0) finished = true;
        }
        return finished;
    }

    /// Runs the row `sample` appended, on the request's lane and its twin's, for the next frame's logits.
    pub fn advance(r: *Run, engine: anytype) !void {
        const last = r.rows.items[r.rows.items.len - 1 ..];
        const steer: []const Steer = if (r.steer) |s| s.at(s.ctx, r.rows.items.len) else &.{};
        try step(engine, last, steer);
        try engine.lastLogits(@ptrCast(r.logits));
        if (r.unconditional) |u| {
            engine.swap();
            defer engine.swap();
            try step(engine, last, if (r.steer != null and r.steer.?.twin) steer else &.{});
            try engine.lastLogits(u);
        }
    }

    fn step(engine: anytype, last: []const [W]i32, steer: []const Steer) !void {
        const T = @TypeOf(engine);
        const steers = @hasDecl(if (@typeInfo(T) == .pointer) @typeInfo(T).pointer.child else T, "runSteered");
        if (steers and steer.len != 0) return engine.runSteered(last[0], steer);
        if (steer.len != 0) return error.EngineDoesNotSteer;
        try engine.run(last, &.{}, null);
    }

    /// The row `sample` appended: what a pass over several requests' lanes runs for this one.
    pub fn lastRow(r: *const Run) [W]i32 {
        return r.rows.items[r.rows.items.len - 1];
    }

    /// The generated rows (caller frees) and the end-of-audio frame; the run is spent.
    pub fn result(r: *Run) !Result {
        return .{ .rows = try r.rows.toOwnedSlice(r.gpa), .eos_frame = r.eos_frame };
    }
};

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

test "generation: two requests taking turns on one engine each draw what they draw alone" {
    const t = std.testing;
    // an engine whose logits depend on the lane in use and on how many rows that lane has run
    const Fake = struct {
        lane: usize = 0,
        rows: [2]usize = .{ 0, 0 },
        pub fn run(f: *@This(), rows: []const [W]i32, _: anytype, _: anytype) !void {
            f.rows[f.lane] += rows.len;
        }
        pub fn swap(_: *@This()) void {}
        pub fn lastLogits(f: *@This(), out: *[cfg.head_rows]f32) !void {
            @memset(out, 0);
            for (0..C) |cb| out[cb * cfg.audio_vocab + 100 + 10 * f.lane + (f.rows[f.lane] + cb) % 7] = 3;
        }
    };
    const sampling: sampler.Sampling = .{ .temperature = 0, .repetition_penalty = 1 };
    var prompt = [_][W]i32{ @splat(cfg.audio_pad_id), @splat(cfg.audio_pad_id), @splat(cfg.audio_pad_id) };
    for (&prompt, 2..) |*row, text| row[C] = @intCast(text);
    var alone: [2][][W]i32 = undefined;
    for (&alone, 0..) |*rows, lane| {
        var fake: Fake = .{ .lane = lane };
        rows.* = (try generate(t.allocator, &fake, prompt[0 .. 2 + lane], @as([]const u8, &.{}), sampling, 5 + lane)).rows;
    }
    defer for (alone) |rows| t.allocator.free(rows);
    var fake: Fake = .{};
    var runs: [2]Run = undefined;
    for (&runs, 0..) |*r, lane| {
        fake.lane = lane;
        r.* = try Run.begin(t.allocator, &fake, prompt[0 .. 2 + lane], @as([]const u8, &.{}), &.{}, @as([]const u8, &.{}), 1, sampling, 5 + lane);
    }
    defer for (&runs) |*r| r.deinit();
    var done = [2]bool{ false, false };
    while (!done[0] or !done[1]) for (&runs, 0..) |*r, lane| {
        if (done[lane]) continue;
        fake.lane = lane;
        done[lane] = try r.sample();
        if (!done[lane]) try r.advance(&fake);
    };
    for (runs, alone) |r, rows| try t.expectEqualSlices([W]i32, rows, r.rows.items);
}
