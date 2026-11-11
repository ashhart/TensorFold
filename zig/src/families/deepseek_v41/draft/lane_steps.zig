//! Backend.prefill_step over a target's begin / piece / finish (Options.steps): the lane host's rounds between steps.
const lanes = @import("lanes");
const ln = @import("lanes.zig");
const Stream = lanes.Stream;

/// A prompt step: begun on the first call, a piece of the step's rows while more are left, else `finish` on the rest.
pub fn prefillStep(ptr: *anyopaque, s: *Stream, max_rows: u32) anyerror!bool {
    const x: *ln.Lanes = @ptrCast(@alignCast(ptr));
    const st = x.state(s) orelse try begin(x, s);
    // a failed step frees the slot here: the target abandons the prompt, the lane goes back to the free list
    errdefer ln.Lanes.release(x, s);
    if (s.isCancelled()) return error.Cancelled;
    const ids = s.prompt();
    const rows: u64 = @max(1, if (max_rows > 0) max_rows else x.opts.step_rows);
    try x.settle();
    if (ids.len - st.len > rows) {
        try x.target.piece(st.slot, ids, st.len, st.len + rows, false);
        st.len += rows;
        s.fill_at.store(@intCast(st.len), .monotonic);
        return false;
    }
    try x.finished(st, ids, try x.target.finish(st.slot, ids, s.sampling, ids.len));
    s.fill_at.store(@intCast(ids.len), .monotonic);
    return true;
}

/// The stream's slot and target prompt begun: its rows held so far are the kept ones the target restored.
fn begin(x: *ln.Lanes, s: *Stream) !*ln.St {
    const b = try ln.Lanes.begin(x, s);
    s.cached = @intCast(b.cached);
    const st = x.state(s).?;
    st.len = b.cached;
    s.fill_at.store(@intCast(b.cached), .monotonic);
    return st;
}
