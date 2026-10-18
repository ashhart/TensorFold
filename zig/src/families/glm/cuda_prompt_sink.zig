//! Prompt logprob rows owned by the requesting stream.
const std = @import("std");
const lanes = @import("lanes");
const prompt_lp = @import("cuda_prompt_logprobs.zig");
/// A stream's prompt rows as rank 0 reads them back, in prompt order.
pub const Sink = struct {
    gpa: std.mem.Allocator,
    s: *lanes.Stream,

    pub fn prepare(p: *Sink) !void {
        if (p.s.prompt_logprobs != null) try p.s.prompt_rows.ensureTotalCapacity(p.gpa, p.s.prompt_len -| 1);
    }

    pub fn want(p: *Sink) ?prompt_lp.Want {
        const k = p.s.prompt_logprobs orelse return null;
        return .{ .k = k, .reader = .{ .ctx = p, .call = read } };
    }

    pub fn read(ctx: *anyopaque, words: []const u32, picks: []const u32) anyerror!void {
        const p: *Sink = @ptrCast(@alignCast(ctx));
        const w = prompt_lp.row_words;
        for (picks, 0..) |pick, i| p.s.prompt_rows.appendAssumeCapacity(lanes.LogprobRow.fromWords(words[i * w ..][0..w], pick, p.s.prompt_logprobs.?));
    }
};
