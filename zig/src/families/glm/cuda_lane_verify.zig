//! Target verification windows and their request-owned repetition rows.
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const adapter = @import("cuda_lanes.zig");
const Lanes = adapter.Lanes;
const Lane = adapter.Lane;
const be = lanes.backend;
const max_rows = adapter.max_rows;

pub fn run(self: *Lanes, windows: []const be.Window, out: []be.Verified) anyerror!void {
    if (windows.len > 1) return shared(self, windows, out);
    for (windows, out) |w, *o| {
        if (w.parents != null) return error.TreesNotBuilt;
        const l = try self.bind(w.stream, null);
        const rows = w.rows();
        if (rows > max_rows) return error.WindowTooWide;
        if (w.held > l.held_n) return error.NoHeldDrafts;
        const pos = self.x.e.pos();
        for (w.positions, 0..) |p, r| if (p != pos + 1 + r) return error.PositionMismatch;
        var ids: [max_rows]u32 = undefined;
        ids[0] = w.pending;
        const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
        @memcpy(ids[1..][0..drafts.len], drafts);
        const stage = try self.resetProcessors();
        if (w.stream.cache_len > w.stream.context.items.len) return error.PositionMismatch;
        try self.addProcessor(w.stream, pos + 1, w.stream.context.items[0..@intCast(w.stream.cache_len)], ids[0..rows]);
        try self.x.processors(stage);
        var sampled: [max_rows]u32 = undefined;
        const bits = try self.putBits(0, rows, w.masks); // before the step goes out: a refusal after it strands the ranks
        self.x.e.structured = bits != 0;
        defer self.x.e.structured = false;
        try self.x.verify(ids[0..rows], sampled[0..rows], bits);
        @memcpy(o.sampled[0..rows], sampled[0..rows]);
        @memcpy(o.drafts[0..drafts.len], drafts);
        l.pending_rows = rows;
    }
}

/// A round's windows in one forward (Engine.verifyShared), each stream's rows on its own sequence.
fn shared(self: *Lanes, windows: []const be.Window, out: []be.Verified) !void {
    if (windows.len > engine.max_shared) return error.WindowTooWide;
    var parts: [engine.max_shared]engine.Part = undefined;
    var ids: [engine.max_shared][max_rows]u32 = undefined;
    var sampled: [engine.max_shared][max_rows]u32 = undefined;
    var lane: [engine.max_shared]*Lane = undefined;
    var row: usize = 0; // the windows' rows so far: each one's grammar rows start there in the bits
    var structured = false;
    const stage = try self.resetProcessors();
    for (windows, 0..) |w, k| {
        if (w.parents != null) return error.TreesNotBuilt;
        const l = try self.bind(w.stream, null);
        const rows = w.rows();
        if (rows > max_rows) return error.WindowTooWide;
        if (w.held > l.held_n) return error.NoHeldDrafts;
        const pos = self.x.e.pos();
        for (w.positions, 0..) |p, r| if (p != pos + 1 + r) return error.PositionMismatch;
        ids[k][0] = w.pending;
        const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
        @memcpy(ids[k][1..][0..drafts.len], drafts);
        if (w.stream.cache_len > w.stream.context.items.len) return error.PositionMismatch;
        try self.addProcessor(w.stream, pos + 1, w.stream.context.items[0..@intCast(w.stream.cache_len)], ids[k][0..rows]);
        const bits = try self.putBits(row, rows, w.masks);
        structured = structured or bits != 0;
        parts[k] = .{ .slot = l.slot, .ids = ids[k][0..rows], .out = sampled[k][0..rows], .bits = bits };
        lane[k] = l;
        row += rows;
    }
    try self.x.processors(stage);
    self.x.e.structured = structured; // one window under a grammar: the whole step draws from whole rows
    defer self.x.e.structured = false;
    try self.x.verifyShared(parts[0..windows.len]);
    self.shared += 1;
    for (out[0..windows.len], parts[0..windows.len], lane[0..windows.len]) |*o, p, l| {
        const rows = p.ids.len;
        @memcpy(o.sampled[0..rows], p.out);
        @memcpy(o.drafts[0 .. rows - 1], p.ids[1..rows]);
        l.pending_rows = rows;
    }
}
