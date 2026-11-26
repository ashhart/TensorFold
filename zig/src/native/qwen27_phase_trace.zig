//! Qwen DFlash2's opt-in shared verify timestamps; the existing command-buffer waits supply GPU times.
const std = @import("std");
const mtl = @import("metal");
pub const name = "TF_Q27_PHASE_TRACE";
pub fn create(a: std.mem.Allocator, draft: bool, queue: mtl.Queue) !?*mtl.phase_trace.Trace {
    const on = try @import("sys").environment.read(bool, name, false);
    if (!on or !draft) return null;
    const t = try a.create(mtl.phase_trace.Trace);
    t.* = .{ .queue = queue.traceKey(), .now = mtl.clock.seconds };
    return t;
}
pub fn destroy(a: std.mem.Allocator, t: ?*mtl.phase_trace.Trace) void {
    if (t) |p| {
        p.flush();
        a.destroy(p);
    }
}
pub fn begin(t: *mtl.phase_trace.Trace, windows: anytype, growth: usize) void {
    var meta: mtl.phase_trace.Meta = .{ .streams = windows.len, .eligible = windows.len <= 128, .growth = growth };
    for (windows, 0..) |w, i| {
        if (i >= meta.keys.len) break;
        meta.grants[i] = @intCast(w.rows());
        meta.keys[i] = @intFromPtr(w.stream);
        meta.turns[i] = w.stream.rounds;
        meta.rows += w.rows();
        meta.eligible = meta.eligible and w.rows() > 1 and w.stream.sampling == null and w.masks.len == 0 and !w.stream.penalties.active();
    }
    t.begin(meta);
}
