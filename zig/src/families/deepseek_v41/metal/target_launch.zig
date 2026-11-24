//! Decode launch shared by ordinary slot windows and tree shadow chains.
const mtl = @import("metal");
const host = @import("dsv41");
const kv = @import("kv.zig");
const forward = @import("forward.zig");
const scratch = @import("scratch.zig");
const ctx_ = @import("ctx.zig");
const pipelines = @import("pipelines.zig");
const Target = @import("target.zig").Target;
const iface = host.iface;
const R = pipelines.max_rows;

/// One chain of a launch: the state it runs on, its ids from position `start`, the Engram ids before them.
pub const Run = struct { kv: *const kv.Slot, ids: []const u32, start: u32, lookback: []const u32 };

/// One launch of the decode forward over the runs' rows, in order (at most R rows).
pub fn run(t: *Target, runs: []const Run, group: []const iface.Segment) !void {
    const e = t.env;
    const c = e.c;
    var pieces: [R]forward.Piece = undefined;
    var spans: [R]scratch.Span = undefined;
    var segs: [R]ctx_.Seg = undefined;
    var r0: u32 = 0;
    for (runs, 0..) |g, i| {
        const n: u32 = @intCast(g.ids.len);
        pieces[i] = .{ .ids = g.ids, .lookback = g.lookback };
        spans[i] = .{ .p0 = g.start, .n = n };
        segs[i] = .{ .slot = g.kv, .p0 = g.start, .r0 = r0, .n = n };
        r0 += n;
    }
    // the Engram rows' reads start now; each layer's rows land as its command buffer comes up (Hook.rows)
    const look_up = try forward.prepareIds(&t.b, c, pieces[0..runs.len], e.lookup);
    if (look_up) |l| forward.prefetchRows(pieces[0..runs.len], l);
    try e.s.writeSpans(c, spans[0..runs.len]);
    const x: ctx_.Ctx = .{ .c = c, .p = e.p, .s = e.s, .slot = segs[0].slot, .tables = e.tables, .p0 = segs[0].p0, .rows = r0, .segs = segs[0..runs.len] };
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var hook: Hook = .{ .t = t, .group = group, .pieces = pieces[0..runs.len], .look_up = look_up };
    try forward.runWith(&x, e.queue, e.model, &t.b, .{ .ctx = &hook, .rows = if (look_up != null) Hook.rows else null, .then = if (t.behind != null and group.len > 0) Hook.then else null });
}

/// A launch's hooks: each Engram layer's rows before the command buffer that reads them, the drafter's chain after.
const Hook = struct {
    t: *Target,
    group: []const iface.Segment,
    pieces: []const forward.Piece,
    look_up: ?forward.Lookup,

    fn rows(ctx: *anyopaque, li: usize) anyerror!void {
        const h: *Hook = @ptrCast(@alignCast(ctx));
        try forward.fillRows(&h.t.b, h.t.env.c, h.pieces, h.look_up.?, li);
    }

    fn then(ctx: *anyopaque) void {
        const h: *Hook = @ptrCast(@alignCast(ctx));
        const b = h.t.behind.?;
        var taps: [R]mtl.Buffer = undefined;
        if (h.group.len > R) return;
        for (h.group, 0..) |g, i| taps[i] = h.t.slots[g.slot].kv.taps orelse return;
        b.queue(b.ctx, h.group, h.t.b.logits.buf, h.t.b.logits.off, taps[0..h.group.len]);
    }
};
