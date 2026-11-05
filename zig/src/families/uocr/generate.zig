//! The publisher's generation: greedy under PageCountGuard (EOS only after every page's marker, 20,000 tokens a page).
const std = @import("std");
const cfg = @import("config.zig");
const prompt_mod = @import("prompt.zig");
const decoder_mod = @import("decoder.zig");
const resize = @import("resize.zig");

pub const Guard = @import("guard.zig").Guard;

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
