//! The shared MTP lane adapter; the lane core still owns round membership and row budgets.
const lanes = @import("lanes");
const fw = @import("cuda_forward.zig");
const engine = @import("cuda_engine.zig");

/// One stream's MTP absorb and draft request in a shared round.
pub const DraftPart = struct {
    slot: u32,
    kept: usize = 0, // pending verify rows to commit on `slot` before absorbing `follow` (0: already committed)
    from_prompt: bool = false,
    follow: []const u32,
    depth: usize,
    out: []u32,
    chances: ?[]f64 = null,
};

/// Lanes MTP dispatch: one-stream rounds keep `Leader.draft`; shared rounds batch into one `Leader.draftShared` step.
pub fn draftLanes(self: anytype, requests: []const lanes.backend.DraftRequest) !void {
    if (!self.x.e.mtp) return error.NoDraftHead;
    if (requests.len <= 1 or self.alone) {
        for (requests) |r| {
            const l = try self.bind(r.stream, if (r.rows) |rows| rows.len else null);
            if (r.position != self.x.e.pos() + 1) return error.PositionMismatch;
            if (r.lanes != null) return error.TreesNotBuilt;
            const depth = @min(r.depth, engine.max_drafts);
            const chances: ?[]f64 = if (self.mtp_confidence != null) l.chances[0..depth] else null;
            if (r.rows) |rows| {
                for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                try self.x.draft(false, r.follow[0..rows.len], depth, &l.held, chances);
            } else {
                const first = self.value(r.first orelse return error.NoFirstToken);
                try self.x.draft(true, &.{first}, depth, &l.held, chances);
            }
            l.held_n = depth;
        }
        return;
    }
    if (requests.len > engine.max_shared) return error.WindowTooWide;
    const LanePtr = @TypeOf(self.lanes.getPtr(requests[0].stream).?);
    var parts: [engine.max_shared]DraftPart = undefined;
    var firsts: [engine.max_shared][1]u32 = undefined;
    var held: [engine.max_shared]LanePtr = undefined;
    var absorbed: usize = 0;
    for (requests, 0..) |r, k| {
        const l = self.lanes.getPtr(r.stream) orelse return error.UnknownStream;
        if (l.draft_failed) return error.SharedDraftFailed;
        for (parts[0..k]) |q| if (q.slot == l.slot) return error.SlotTwice;
        if (r.lanes != null) return error.TreesNotBuilt;
        var kept: usize = 0;
        var follow_ids: []const u32 = undefined;
        var from_prompt = false;
        if (r.rows) |rows| {
            if (rows.len == 0 or rows.len > fw.window_rows) return error.WindowTooWide;
            if (r.follow.len < rows.len) return error.WindowTooWide;
            for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
            if (l.pending_rows) |pending| {
                if (rows.len > pending) return error.BadCommit;
                kept = rows.len;
            }
            follow_ids = r.follow[0..rows.len];
        } else {
            if (l.pending_rows) |pending| kept = pending;
            firsts[k] = .{self.value(r.first orelse return error.NoFirstToken)};
            follow_ids = &firsts[k];
            from_prompt = true;
        }
        absorbed += follow_ids.len;
        if (absorbed > engine.shared_rows) return error.WindowTooWide;
        try self.x.e.bind(l.slot);
        if (r.position != self.x.e.pos() + kept + 1) return error.PositionMismatch;
        const depth = @min(r.depth, engine.max_drafts);
        parts[k] = .{
            .slot = l.slot,
            .kept = kept,
            .from_prompt = from_prompt,
            .follow = follow_ids,
            .depth = depth,
            .out = l.held[0..depth],
            .chances = if (self.mtp_confidence != null) l.chances[0..depth] else null,
        };
        held[k] = l;
    }
    // Dispatch errors are fatal to every member until release; an already kept prefix must never be retried.
    self.x.draftShared(parts[0..requests.len]) catch |err| {
        for (held[0..requests.len]) |l| l.draft_failed = true;
        return err;
    };
    for (parts[0..requests.len], held[0..requests.len]) |p, l| {
        l.pending_rows = null;
        l.held_n = p.depth;
    }
}
