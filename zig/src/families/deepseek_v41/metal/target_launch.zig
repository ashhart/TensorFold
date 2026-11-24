//! Decode launch shared by ordinary slot windows and tree shadow chains.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const kv = @import("kv.zig");
const forward = @import("forward.zig");
const scratch = @import("scratch.zig");
const round_prof = @import("round_prof.zig");
const ctx_ = @import("ctx.zig");
const pipelines = @import("pipelines.zig");
const Target = @import("target.zig").Target;
const iface = host.iface;
const lanes = host.lanes;
const R = pipelines.max_rows;

/// One chain of a launch: the state it runs on, its ids from position `start`, the Engram ids before them.
pub const Run = struct { kv: *const kv.Slot, ids: []const u32, start: u32, lookback: []const u32 };

/// One launch of the decode forward over the runs' rows, in order (at most R rows).
pub fn run(t: *Target, runs: []const Run, group: []const iface.Segment, rec: ?*round_prof.Round) !void {
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
    if (rec) |r| r.lap(.prepare);
    const x: ctx_.Ctx = .{ .c = c, .p = e.p, .s = e.s, .slot = segs[0].slot, .tables = e.tables, .p0 = segs[0].p0, .rows = r0, .segs = segs[0..runs.len] };
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var hook: Hook = .{ .t = t, .group = group, .pieces = pieces[0..runs.len], .look_up = look_up };
    try forward.runTimed(&x, e.queue, e.model, &t.b, .{ .ctx = &hook, .rows = if (look_up != null) Hook.rows else null, .then = if (t.behind != null and group.len > 0) Hook.then else null }, rec);
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
        const tb = h.t.slots[h.group[0].slot].kv.taps orelse return;
        b.queue(b.ctx, h.group, h.t.b.logits.buf, h.t.b.logits.off, tb);
    }
};

/// Whether the next segment would repeat a slot already in this launch.
pub fn duplicateSlot(done: []const iface.Segment, slot: u32) bool {
    for (done) |o| if (o.slot == slot) return true;
    return false;
}

/// Starts reading the Engram rows of the slot's next pending row (a guess: a cut commit only wastes the reads).
pub fn prefetchNext(t: *const Target, l: forward.Lookup, g: iface.Segment, out: []const u32) void {
    if (t.env.c.engram_layers.len == 0) return;
    var acc: usize = 0;
    while (acc + 1 < g.tokens.len and g.tokens[acc + 1] == out[acc]) acc += 1;
    const hist = t.slots[g.slot].hist.items;
    const at: usize = @intCast(g.start + acc + 1);
    const look = t.env.c.engram_max_ngram - 1;
    forward.prefetchRows(&.{.{ .ids = out[acc..][0..1], .lookback = hist[at -| look..at] }}, l);
}

/// The serial reference without lanes: the prompt, then one row a step, each drawn at its own position.
pub fn serial(t: *Target, slot: u32, ids: []const u32, n: usize, smp: ?lanes.Sampling) ![]u32 {
    const tg = t.target();
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(t.a);
    var tok = try tg.prefill(slot, ids, smp, ids.len);
    while (out.items.len < n) {
        try out.append(t.a, tok);
        const at: u64 = ids.len + out.items.len - 1;
        var ch: [1]u32 = undefined;
        var cs = [_][]u32{&ch};
        try tg.window(&.{.{ .slot = slot, .start = at, .tokens = &.{tok}, .parents = null, .draws = &.{at + 1}, .sampling = smp }}, &cs);
        tok = ch[0];
    }
    tg.release(slot);
    return out.toOwnedSlice(t.a);
}
