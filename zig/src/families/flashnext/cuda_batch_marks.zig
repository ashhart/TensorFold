//! Prompt cache marks inside a pass (TF_FLASHNEXT_KEEP_INPASS): their kept states filled on the way, no cut pass.
const wt = @import("cuda_prompt_wt.zig");
const gdn = @import("cuda_gdn.zig");
const gdn_fused = @import("cuda_gdn_fused.zig");
const parts = @import("cuda_batch_state.zig");
const snap_at = @import("cuda_snap.zig").layout;

/// A mark's rows into this lone pass when it falls inside it (after its first row, up to its last).
fn cutOf(p: anytype, m: parts.Mark) ?usize {
    if (m.base == 0 or p.lanes.len > 0 or m.at <= p.pos0 or m.at > p.pos0 + p.rows) return null;
    return m.at - p.pos0;
}

/// Whether a mark falls inside this lone pass: only the fused DeltaNet kernel keeps a mark's state on the way.
pub fn inPass(p: anytype) bool {
    for (p.hooks.marks) |m| if (cutOf(p, m) != null) return true;
    return false;
}

/// Each mark's n-gram tail: the window before the pass moved on by the mark's rows.
pub fn tails(p: anytype) !void {
    for (p.hooks.marks) |m| if (cutOf(p, m)) |cut| {
        const dst = m.base + snap_at.tail(m.at);
        try p.copy(dst, p.seq.tail.ptr, snap_at.tail_size);
        try p.shiftWindow(dst, p.s.ple_nrow, wt.wide, cut, wt.ple_tail_n, wt.wide);
    };
}

/// Each mark's DeltaNet layer `li`: its conv window (the one before the pass moved on), its state from the fused chain.
pub fn cuts(p: anytype, li: usize, fused: bool) ![parts.max_marks]gdn_fused.Cut {
    var c: [parts.max_marks]gdn_fused.Cut = .{ .{}, .{} };
    for (p.hooks.marks, 0..) |m, i| if (cutOf(p, m)) |cut| {
        if (!fused) return error.UnexpectedTensor;
        c[i] = .{ .row = @intCast(cut), .state = m.base + snap_at.state(li) };
        const dst = m.base + snap_at.conv(li);
        try p.copy(dst, p.seq.conv[li].ptr, snap_at.conv_size);
        try p.shiftWindow(dst, p.s.proj, gdn.proj_width, cut, gdn.taps - 1, gdn.conv_dim);
    };
    return c;
}
