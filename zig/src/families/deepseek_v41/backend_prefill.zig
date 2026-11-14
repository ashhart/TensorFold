//! The V4.1 prompt pass a chunk at a time: the lane host runs the other streams' rounds between two chunks.
const std = @import("std");
const ds = @import("model.zig");
const lanes = @import("lanes");
const Stream = lanes.Stream;
const Backend = @import("backend.zig").Backend;

/// The stream's slot, taken, reset and (for a kept prompt state) restored on the first chunk; the pass's cursor.
fn begin(b: *Backend, s: *Stream, prompt: []const u32) !u32 {
    const i = b.free() orelse return error.NoFreeSlot;
    const state = &b.model.caches[i];
    if (prompt.len + s.max_new + 1 > state.plan.capacity) return error.ContextFull;
    try b.by.put(b.gpa, s, i);
    try b.model.resetSlot(i);
    s.cached = 0;
    // A KEPT PROMPT STATE RESUMES THE PASS WHERE IT STOOD.
    if (s.reuse.saved) |saved| {
        const snap: *ds.Model.Snapshot = @ptrCast(@alignCast(saved));
        const want = s.reuse.at;
        if (want > 0 and want < prompt.len) blk: {
            b.model.snapshotRestore(i, snap) catch |err| {
                s.reuse_failed = true;
                s.reuse_error = err;
                break :blk;
            };
            s.cached = want;
        }
    }
    s.fill_at.store(s.cached, .monotonic);
    return i;
}

/// One chunk of at most `max_rows` rows (0: the workspace's); true once the prompt is in and its first token drawn.
pub fn step(ptr: *anyopaque, s: *Stream, max_rows: u32) anyerror!bool {
    const b = Backend.self(ptr);
    const prompt = s.prompt();
    if (prompt.len == 0) return error.EmptyPrompt;
    const i = b.by.get(s) orelse try begin(b, s, prompt);
    if (s.isCancelled()) return error.Cancelled;
    const at: u32 = b.model.caches[i].position;
    const plan_rows = b.model.workspace.plan.rows;
    const rows = if (max_rows != 0) @min(plan_rows, max_rows) else plan_rows;
    // THE TAIL MARK IS SNAPSHOTTED INSIDE THE CHUNK THAT RUNS PAST IT, not cut into a second forward.
    const merge_cold = std.c.getenv("TF_DSV41_SNAP_COLD") == null;
    const snap_at: u32 = if (std.c.getenv("TF_DSV41_NO_TAIL_SNAP") == null and b.model.snapTailAllowed() and (merge_cold or s.reuse.at > 0)) s.reuse.snap_at else 0;
    var end = @min(at + rows, @as(u32, @intCast(prompt.len)));
    // The store's marks always end a chunk: the pass stands exactly there when it calls the hook.
    for (s.reuse.marks) |m| if (m > at and m < end and m != snap_at) {
        end = m;
        break;
    };
    const last = end == prompt.len;
    // The next chunk's Engram rows are read while this chunk's GPU work runs.
    if (!last) b.model.lookahead(i, prompt[end..@min(prompt.len, end + plan_rows)]);
    var merge: u32 = if (snap_at > at and snap_at < end and end - at >= 64) snap_at else 0;
    if (merge != 0 and !b.model.beginSnapTail(i, prompt[@max(0, @as(usize, merge) - 4)..merge], merge, at)) merge = 0;
    if (last) {
        _ = try b.model.forwardOn(i, prompt[at..end], .last);
        b.first[i] = try b.draw(i, s, @intCast(prompt.len));
    } else {
        _ = try b.model.forwardOn(i, prompt[at..end], .none);
    }
    if (merge != 0) {
        // the state at the mark, copied from the stash the chunk's own publishes wrote
        if (s.reuse.hook) |hook| hook.at(hook.ptr, s, merge);
        b.model.endSnapTail();
    }
    // The prompt cache's keep hook, at the chunk end it asked to see.
    if (s.reuse.hook) |hook| if (std.mem.indexOfScalar(u32, s.reuse.marks, end) != null) hook.at(hook.ptr, s, end);
    s.fill_at.store(end, .monotonic);
    return last;
}

/// The whole prompt pass in one call, as the lane host runs it when it has nothing to interleave.
pub fn whole(ptr: *anyopaque, s: *Stream) anyerror!void {
    while (!try step(ptr, s, 0)) {}
}

/// TF_DSV41_PREFILL_INTERLEAVE: 0 runs each prompt whole (other streams stop for it); else by chunks.
pub fn interleaveOn() bool {
    return @import("sys").environment.read(bool, "TF_DSV41_PREFILL_INTERLEAVE", true) catch unreachable;
}
