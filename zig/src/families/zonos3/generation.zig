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
    if (prompt.len == 0 or max_frames == 0) return error.NoRows;
    const logits = try gpa.create([C][cfg.audio_vocab]f32);
    defer gpa.destroy(logits);
    var rows: std.ArrayList([W]i32) = .empty;
    errdefer rows.deinit(gpa);
    var codes: std.ArrayList([C]i32) = .empty;
    defer codes.deinit(gpa);
    var has_audio_prefix = false;
    for (prompt[prompt.len - 1][0..C]) |code| has_audio_prefix = has_audio_prefix or code != cfg.audio_pad_id;
    try engine.run(prompt, replaced, null);
    var eos_frame: ?usize = null;
    var countdown: usize = 0;
    while (true) {
        try engine.lastLogits(@ptrCast(logits));
        const generated = rows.items.len;
        var row = sampler.sample(logits, codes.items, generated, cfg.eoa_id, sampling);
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
        if (finished) break;
        try engine.run(rows.items[generated..], &.{}, null);
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
