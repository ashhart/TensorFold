//! GLM-5.3-Flash's prefill_many: fresh streams' whole prompts in one prompt pass, each left as its own pass leaves it.

const std = @import("std");
const fw = @import("cuda_forward.zig");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;

/// Whether `parts` go in one pass: 1 to max_together fresh prompts, prompt_rows in all, each under the capacity.
pub fn fits(parts: []const engine.Prompt, capacity: usize) bool {
    if (parts.len == 0 or parts.len > engine.max_together) return false;
    var rows: usize = 0;
    for (parts, 0..) |p, i| {
        if (p.ids.len == 0 or p.ids.len + fw.window_rows >= capacity) return false;
        for (parts[0..i]) |q| if (q.slot == p.slot) return false;
        rows += p.ids.len;
    }
    return rows <= fw.prefill_rows;
}

/// The prompts' pass on their slots (each begun, at position 0); `firsts[k]`: prompt k's first token.
pub fn prefillMany(e: *Engine, parts: []const engine.Prompt, firsts: []u32) !void {
    if (!fits(parts, e.capacity) or firsts.len < parts.len) return error.BadPrompts;
    var segs: [engine.max_together]fw.Seg = undefined;
    var ids: [fw.prefill_rows]u32 = undefined;
    var rows: usize = 0;
    for (parts, 0..) |p, k| {
        try e.bind(p.slot);
        if (e.cur.st.pos != 0 or e.cur.st.mtp_len != 0) return error.BadPrompts;
        segs[k] = .{ .st = &e.cur.st, .row0 = rows, .rows = p.ids.len };
        @memcpy(ids[rows..][0..p.ids.len], p.ids);
        rows += p.ids.len;
    }
    try e.gate(null);
    var one: [1]fw.Rank = .{.{ .w = &e.w, .st = segs[0].st, .b = &e.bp, .segs = segs[0..parts.len] }};
    var f: fw.Forward = .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = &one, .exchange = e.exchange(), .pieces = &e.pieces, .split = e.settings.split, .kda_wide = e.settings.kda_wide };
    try f.run(@ptrCast(ids[0..rows]));
    const D = e.c.hidden;
    for (parts, segs[0..parts.len], 0..) |p, g, k| {
        try e.bind(p.slot);
        try e.draw(e.bp.logits + k * e.share.vocab.len() * 2, 1, p.ids.len, firsts[k..][0..1]);
        if (!e.mtp) continue;
        try e.copy(e.hold.ptr + k * D * 2, e.bp.fnormed + (g.row0 + g.rows - 1) * D * 2, D * 2);
        e.cur.held_at = e.hold.ptr + k * D * 2;
    }
    f.pieces = null;
    for (parts, segs[0..parts.len]) |p, g| {
        one[0] = .{ .w = &e.w, .st = g.st, .b = &e.bp };
        // decode._absorb_rows; the head's pass writes final-norm row 0, which the hold rows above already copied
        if (e.mtp and g.rows > 1) {
            try f.mtp(@ptrCast(p.ids[1..]), &.{e.bp.fnormed + g.row0 * D * 2});
            try g.st.setMtpLen(e.stream, g.st.mtp_len + g.rows - 1);
        }
        try f.commit(g.rows);
    }
}
