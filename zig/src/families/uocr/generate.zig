//! The publisher's generation loop: PageCountGuard over the logits (EOS barred until the reply has as many <PAGE>
//! markers as the prompt, then forced; a page held to 20,000 tokens), greedy argmax, EOS not emitted.
const std = @import("std");
const cfg = @import("config.zig");
const prompt_mod = @import("prompt.zig");
const decoder_mod = @import("decoder.zig");
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

/// Greedy tokens after a prefill, up to `max_new` or EOS (not included), into `out`; `on_token` sees each one.
pub fn greedy(d: *decoder_mod.Decoder, p: prompt_mod.Prompt, max_new: usize, out: *std.ArrayList(u32), a: std.mem.Allocator, ctx: anytype, comptime on_token: fn (@TypeOf(ctx), u32) void) !void {
    var guard = Guard.init(p);
    var st = prompt_mod.Step.init(p);
    var lg = try d.logits();
    while (out.items.len < max_new and d.len + 1 < cfg.max_context) {
        const id = guard.pick(lg);
        if (id == cfg.eos) break;
        try out.append(a, id);
        on_token(ctx, id);
        guard.take(id);
        if (out.items.len == max_new) break;
        const c = st.next(id);
        try d.step(id, c.tok, c.page, c.qpage);
        lg = try d.logits();
    }
}

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
