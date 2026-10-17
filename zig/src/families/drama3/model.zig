//! The logical frame contract is shared; each backend owns its physical KV and recurrent buffers.
const std = @import("std");
const cfg = @import("config.zig");
pub const Frame = [cfg.frame_width]i32;
pub const max_reference_samples = 30 * 44100;

pub fn reference(samples: []const f32, rate: u32) !void {
    if (rate == 0 or rate > 192000 or samples.len == 0 or samples.len > @as(usize, rate) * 30)
        return error.BadReference;
    for (samples) |s| if (!std.math.isFinite(s)) return error.BadReference;
}

pub fn finishFrames(gpa: std.mem.Allocator, host: []Frame, made: usize) ![]Frame {
    if (made > host.len) return error.BadFrameCount;
    return gpa.realloc(host, made);
}

test "early EOS returns an owned short allocation and references stay bounded" {
    const gpa = std.testing.allocator;
    const frames = try gpa.alloc(Frame, 8);
    const out = try finishFrames(gpa, frames, 1);
    defer gpa.free(out);
    try std.testing.expectEqual(@as(usize, 1), out.len);
    try reference(&.{ 0, 0.5 }, 44100);
    try std.testing.expectError(error.BadReference, reference(&.{std.math.nan(f32)}, 44100));
    try std.testing.expectError(error.BadReference, reference(&.{0}, 0));
}

pub fn validate(rows: []const Frame, ids: cfg.Ids, max_ctx: usize, max_frames: usize) !void {
    if (rows.len < 2 or rows.len > max_ctx or max_frames == 0 or max_frames > max_ctx - rows.len -| 1)
        return error.BadPromptSize;
    for (rows) |row| {
        if (row[0] < 0 or row[0] >= cfg.vocab) return error.BadPromptToken;
        const semantic = row[0] >= ids.sem_begin and row[0] <= ids.sem_end;
        for (row[1..], 0..) |code, i| {
            const limit: i32 = if (i == 0) 4096 else 1024;
            if (code < 0 or code >= limit or (!semantic and code != 0)) return error.BadCode;
        }
        if (semantic and row[0] - ids.sem_begin != row[1]) return error.BadCode;
    }
}

test "typed reference rows are bounded and agree with their semantic code" {
    const ids: cfg.Ids = .{ .eos = 1, .pad = 2, .sem_begin = 100, .sem_end = 4195, .audio_pad = 3 };
    var rows: [2]Frame = @splat(@splat(0));
    rows[1][0] = 104;
    rows[1][1] = 4;
    try validate(&rows, ids, 16, 8);
    rows[1][10] = 1024;
    try std.testing.expectError(error.BadCode, validate(&rows, ids, 16, 8));
    rows[1][10] = 0;
    rows[1][1] = 3;
    try std.testing.expectError(error.BadCode, validate(&rows, ids, 16, 8));
    try std.testing.expectError(error.BadPromptSize, validate(&rows, ids, 2, 1));
}
