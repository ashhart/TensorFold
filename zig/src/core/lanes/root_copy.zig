//! Copy-only rounds still absorb every kept target row so neural drafting can resume at any rejection.
const Engine = @import("engine.zig").Engine;
const be = @import("backend.zig");
const win = @import("windows.zig");

pub fn prepare(e: *Engine, requests: []const be.DraftRequest) ![]const be.DraftRequest {
    if (!e.cfg.root_copies) return requests;
    var out: ?[]be.DraftRequest = null;
    for (requests, 0..) |r, i| {
        const s = r.stream;
        if (r.depth == 0 or s.finished or !s.drafts or s.root_copy_miss or s.force.items.len > 0) continue;
        const copies = if (r.rows == null and s.pending == null) blk: {
            const p = s.proposer orelse continue;
            if (!e.cfg.lone_copies and e.alone) continue;
            const pending = try e.readFeed(r.first orelse continue);
            const most = @min(@as(i64, e.cfg.max_copy), s.draftRoom() - 1);
            const got = p.proposeAfter(s.context.items, &.{pending}, most) catch continue;
            const least: usize = if (e.alone and e.cfg.context_copies) 1 else 2;
            break :blk got.len >= least and p.lastMatch() >= e.cfg.enter_match;
        } else (try win.copyProposal(e, s)).tokens.len > 0;
        if (!copies) continue;
        if (out == null) out = e.arena.allocator().dupe(be.DraftRequest, requests) catch return requests;
        out.?[i].depth = 0;
        out.?[i].lanes = null;
        out.?[i].ranks = false;
    }
    return out orelse requests;
}
