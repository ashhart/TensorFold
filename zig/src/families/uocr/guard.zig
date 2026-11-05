//! The publisher's PageCountGuard: EOS only after every page's marker, 20,000 tokens a page.
const std = @import("std");
const cfg = @import("config.zig");
const prompt_mod = @import("prompt.zig");
const resize = @import("resize.zig");

/// The guard's count of the reply so far.
pub const Guard = struct {
    expected: u32,
    marks: u32 = 0,
    page_len: u32 = 0,

    pub fn init(p: prompt_mod.Prompt) Guard {
        var n: u32 = 0;
        for (p.ids) |id| n += @intFromBool(id == cfg.page_token);
        return .{ .expected = n };
    }

    /// The one token the guard forces, or null when it only bars EOS or lets every token through.
    fn forced(g: Guard) ?u32 {
        if (g.marks > g.expected) return cfg.eos;
        if (g.page_len >= cfg.max_page_tokens) return if (g.marks < g.expected) cfg.page_token else cfg.eos;
        return null;
    }

    /// The token the guard forces, if any (a backend that picks on the GPU asks first).
    pub fn forcedToken(g: Guard) ?u32 {
        return g.forced();
    }

    /// Whether the pick skips EOS: the reply has fewer <PAGE> markers than the prompt.
    pub fn barsEos(g: Guard) bool {
        return g.marks < g.expected;
    }

    /// The greedy pick under the guard from bf16 logits (the first of equal maxima, as torch's argmax gives).
    pub fn pick(g: Guard, logits: []const u16) u32 {
        if (g.forced()) |only| return only;
        const bar_eos = g.marks < g.expected;
        var best: u32 = 0;
        var best_v = -std.math.inf(f32);
        for (logits, 0..) |h, i| {
            if (bar_eos and i == cfg.eos) continue;
            const v = resize.widen(h);
            if (v > best_v) {
                best_v = v;
                best = @intCast(i);
            }
        }
        return best;
    }

    pub fn take(g: *Guard, id: u32) void {
        if (id == cfg.page_token) {
            g.marks += 1;
            g.page_len = 0;
        } else g.page_len += 1;
    }
};

test "uocr: the guard bars EOS until the pages are done, then forces it" {
    var logits: [8]u16 = @splat(resize.bf16(0));
    logits[cfg.eos] = resize.bf16(5);
    logits[3] = resize.bf16(1);
    var g: Guard = .{ .expected = 1 };
    try std.testing.expectEqual(@as(u32, 3), g.pick(&logits));
    g.take(cfg.page_token);
    try std.testing.expectEqual(@as(u32, cfg.eos), g.pick(&logits));
    g.marks = 2;
    logits[cfg.eos] = resize.bf16(-5);
    try std.testing.expectEqual(@as(u32, cfg.eos), g.pick(&logits));
}
