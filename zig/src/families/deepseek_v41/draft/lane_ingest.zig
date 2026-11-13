//! The drafter's committed rows for the lanes: a window's kept rows, after any rows that were left for later.
const iface = @import("iface.zig");
const branches = @import("branches.zig");
const ln = @import("lanes.zig");

/// A round's kept rows into the drafter: first the rows a skipped absorb left out (`catchUp`), then these.
pub fn ingest(x: *ln.Lanes, st: *ln.St, start: u64, rows: []const u32) !void {
    try catchUp(x, st, start, start + rows.len);
    st.ring_end = start + rows.len;
    const span = iface.ingestSpan(rows.len, x.opts.shape.window);
    if (span.count == 0) return;
    const taps = x.target.taps(st.slot);
    const m = taps.map orelse return x.pass.ingest(st.slot, start + span.skip, taps, rows[span.skip..]);
    // a tree resolved as chains: the path's rows in the taps of the chain that holds it
    var mapped: [branches.max_rows]u32 = undefined;
    if (span.count > mapped.len) return error.TreeRows;
    for (rows[span.skip..], mapped[0..span.count]) |r, *o| {
        if (r >= m.len or m[r] == branches.unrun) return error.TreeTaps;
        o.* = m[r];
    }
    try x.pass.ingest(st.slot, start + span.skip, taps, mapped[0..span.count]);
}

/// Absorb skipped: the rows stay in the target's history, and the next ingest takes them from there.
pub fn deferred(x: *const ln.Lanes) bool {
    return x.target.keepsHistory();
}

/// The committed rows between the drafter's end and `start`, the last `window` before `end` only (all it reads).
fn catchUp(x: *ln.Lanes, st: *ln.St, start: u64, end: u64) !void {
    const from = st.ring_end orelse return;
    const lo = @max(from, end -| x.opts.shape.window);
    if (lo >= start) return;
    const n: u32 = @intCast(start - lo);
    const taps = x.target.history(st.slot, lo, n) orelse return error.HistoryLost;
    const rows = try x.scratch.allocator().alloc(u32, n);
    for (rows, 0..) |*r, i| r.* = @intCast(i);
    try x.pass.ingest(st.slot, lo, taps, rows);
}
