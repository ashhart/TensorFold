//! A lane round on the engine: its device plan, its forward (replayed, captured or eager), its draws and its keep.

const std = @import("std");
const hip = @import("hip");
const state = @import("../forward/state.zig");
const win = @import("../forward/window.zig");
const fwd = @import("../forward/forward.zig");
const plan = @import("../forward/plan.zig");
const view = @import("../model/view.zig");
const draw = @import("draw.zig");
const round_graphs = @import("round_graphs.zig");
const Engine = @import("engine.zig").Engine;

/// What a round does with its shape's graph: run eagerly, replay it, or capture it (every rank does the same).
pub const Pick = enum(u32) { eager, replay, capture };

/// What a verify hands back: the round's final rows, padding included.
pub const Verified = struct { hidden: hip.ops.Tensor };

/// One stream's rows of a round: tokens (the pending one, then drafts) from slot `pos` over its caches.
pub const Rows = struct { caches: *state.Caches, pos: usize, tokens: []const u32 };

const Chosen = struct { pick: Pick, entry: ?*round_graphs.Entry, shape: plan.Shape };

/// The slots a keep names and the plan they index.
const Last = struct { plan: hip.plan_ops.Plan, keep_at: usize, hidden: u64 };

/// Keep lists in flight: a flush's pinned words stay until its copy has run, which a later round's sync guarantees.
const keep_ring = 4;

pub const State = struct {
    buffer: plan.Buffer,
    /// The caches the padding rows run in.
    scratch: state.Caches,
    /// Each linear layer's snapshots of the round in flight.
    snaps: [][2]u64,
    wins: []plan.Window,
    /// The rows each slot keeps at the next flush (negative: not listed), and the pinned words they go up in.
    staging: []i32,
    keeps: hip.HostBuffer,
    ring: usize = 0,
    /// For checks: a layer-by-layer trace of the forward (graphs off), and a least bucket of rows.
    trace: ?fwd.Trace = null,
    pad_to: usize = 0,
    chosen: ?Chosen = null,
    last: ?Last = null,
    /// The layout of the plan in flight, for the copy a round starts with.
    layout: plan.Layout = undefined,

    pub fn init(e: *Engine) !State {
        const m = e.model();
        if (!win.supported(e.ops(&e.rounds), m)) return error.RoundsUnsupported;
        const rows = e.o.batch_rows;
        var buffer = try plan.Buffer.init(&e.driver, rows, m.spec.n_layers, e.pool.count);
        errdefer buffer.deinit();
        var scratch = try state.Caches.initFull(e.gpa, &e.pool, m, rows);
        errdefer scratch.deinit(e.gpa);
        const snaps = try e.gpa.alloc([2]u64, m.spec.n_layers);
        errdefer e.gpa.free(snaps);
        const wins = try e.gpa.alloc(plan.Window, rows);
        errdefer e.gpa.free(wins);
        const staging = try e.gpa.alloc(i32, rows + 1);
        errdefer e.gpa.free(staging);
        @memset(staging, -1);
        return .{ .buffer = buffer, .scratch = scratch, .snaps = snaps, .wins = wins, .staging = staging, .keeps = try hip.HostBuffer.alloc(&e.driver, keep_ring * 4 * (rows + 1)) };
    }

    pub fn deinit(s: *State, e: *Engine) void {
        s.keeps.free();
        e.gpa.free(s.staging);
        e.gpa.free(s.wins);
        e.gpa.free(s.snaps);
        s.scratch.deinit(e.gpa);
        s.buffer.deinit();
    }
};

/// The shape of a round of `rows`: its bucket of rows, its slots and the keys its walk covers.
pub fn shapeOf(e: *Engine, rows: []const Rows) !plan.Shape {
    var total: usize = 0;
    var visible: usize = 0;
    for (rows) |r| {
        total += r.tokens.len;
        visible = @max(visible, r.pos + r.tokens.len);
    }
    if (rows.len == 0 or total == 0 or total > e.o.batch_rows) return error.WindowTooWide;
    const cap = std.mem.alignForward(usize, e.o.capacity, plan.min_span);
    const bucket = plan.bucketOf(@max(total, e.round.pad_to), e.o.batch_rows);
    const slots = plan.bucketOf(rows.len + 1, e.o.batch_rows + 1);
    return .{ .rows = @intCast(bucket), .slots = @intCast(slots), .span = @intCast(@min(plan.spanOf(visible), cap)) };
}

/// The round's graph choice: from the shape's history, or `forced` (rank 0's pick, which a follower obeys).
pub fn choose(e: *Engine, rows: []const Rows, forced: ?Pick) !Pick {
    const st = &e.round;
    const shape = try shapeOf(e, rows);
    st.chosen = .{ .pick = .eager, .entry = null, .shape = shape };
    if (!e.o.graphs) {
        if ((forced orelse .eager) != .eager) return error.GraphsDisagree;
        return .eager;
    }
    const entry = try e.graphs.find(shape);
    const mine: Pick = switch (entry.state) {
        .failed => .eager,
        .ready => .replay,
        .seen => .capture,
    };
    const pick = forced orelse mine;
    if (pick == .replay and entry.state != .ready) return error.GraphsDisagree;
    st.chosen = .{ .pick = pick, .entry = entry, .shape = shape };
    return pick;
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Every window in one forward: `out` the token of every row per `reqs`; snapshots live until the next round's `keep`.
pub fn verify(e: *Engine, rows: []const Rows, reqs: []const draw.Request, out: []u32) !Verified {
    const st = &e.round;
    const m = e.model();
    const began = nowNs();
    if (st.chosen == null) _ = try choose(e, rows, null);
    const c = st.chosen.?;
    defer st.chosen = null;
    var total: usize = 0;
    for (rows, 0..) |r, i| {
        if (r.pos + r.tokens.len > r.caches.covered()) return error.ContextFull;
        st.wins[i] = .{ .desc = r.caches.desc.base(), .pos = r.pos, .tokens = r.tokens };
        total += r.tokens.len;
    }
    e.rounds.reset();
    const o = e.ops(&e.rounds);
    try win.kept(o, m, c.shape.rows, st.snaps);
    const l = plan.Layout.of(c.shape.rows, c.shape.slots, m.spec.n_layers);
    st.buffer.fill(l, c.shape, st.wins[0..rows.len], st.scratch.desc.base(), st.snaps);
    st.layout = l;
    const p: hip.plan_ops.Plan = .{ .args = st.buffer.args(l), .rows = c.shape.rows, .slots = c.shape.slots };
    const r: win.Round = .{ .plan = p, .tokens = st.buffer.dev.base() + 4 * l.tokens, .span = c.shape.span, .inputs = st.snaps };
    const done = try run(e, c, r);
    e.graphs.rounds += 1;
    e.graphs.submit_ns += nowNs() - began;
    try e.drawRows(o, done.y, total, reqs, out, m.tp == null);
    st.last = .{ .plan = p, .keep_at = 4 * l.keep, .hidden = done.hidden.ptr };
    return .{ .hidden = done.hidden };
}

/// The forward and its logits projection, and the greedy draw of every row when this rank holds the whole head.
fn body(e: *Engine, r: win.Round) !round_graphs.Out {
    // the plan goes up first, inside the graph
    try e.round.buffer.send(e.stream.handle, e.round.layout);
    return forwardOn(e.ops(&e.rounds), e.model(), r, e.round.trace, e.drawer.argmaxAt());
}

/// `body` after the plan's copy, on `o`: on the counting stream it measures what a round takes.
pub fn forwardOn(ops: hip.ops.Ops, m: *const view.Model, r: win.Round, trace: ?fwd.Trace, argmax_at: u64) !round_graphs.Out {
    var o = ops;
    // a round's rows keep their decode kernels however many share it
    o.window = true;
    const hidden = try win.forward(o, m, r, trace);
    const y = try o.project(hidden, m.head, r.plan.rows, false);
    if (m.tp == null) try o.argmaxRows(y, r.plan.rows, m.head.n, argmax_at);
    return .{ .hidden = hidden, .y = y, .used = o.arena.used };
}

/// One round's forward as `choose` picked: replayed from the shape's graph, captured, or eager.
fn run(e: *Engine, c: Chosen, r: win.Round) !round_graphs.Out {
    const entry = c.entry orelse return body(e, r);
    switch (c.pick) {
        .eager => return body(e, r),
        .replay => {
            e.rounds.used = entry.out.used;
            try entry.exec.?.launchOn(e.stream);
            e.graphs.replayed += 1;
            return entry.out;
        },
        .capture => return capture(e, entry, r),
    }
}

const Recording = struct { e: *Engine, r: win.Round };

fn record(ctx: Recording) anyerror!round_graphs.Out {
    return body(ctx.e, ctx.r);
}

/// Records the round into a graph and launches it; a capture that fails on any rank runs eagerly on every rank.
fn capture(e: *Engine, entry: *round_graphs.Entry, r: win.Round) !round_graphs.Out {
    var kept = false;
    var out: round_graphs.Out = undefined;
    const at = e.rounds.mark();
    const began = nowNs();
    if (try round_graphs.Graphs.record(e.stream, Recording{ .e = e, .r = r }, record)) |rec| {
        out = rec.out;
        kept = if (e.graphs.keep(entry, rec.graph, e.stream, rec.out)) true else |_| false;
    }
    // the policy's `graph_fail` makes that rank's capture fail, to check that every rank falls back
    if (e.o.policy.graph_fail >= 0 and @as(usize, @intCast(e.o.policy.graph_fail)) == e.o.rank) kept = false;
    e.graphs.capture_ns += nowNs() - began;
    const all = if (e.o.world > 1) try agreed(e, kept) else kept;
    if (!all) {
        e.graphs.revoke(entry);
        e.rounds.release(at);
        return body(e, r);
    }
    try entry.exec.?.launchOn(e.stream);
    return out;
}

/// Whether every rank says yes (an all-gather of one word each).
fn agreed(e: *Engine, yes: bool) !bool {
    const world = e.o.world;
    var host = try hip.HostBuffer.alloc(&e.driver, 8 * (1 + world));
    defer host.free();
    var send = try hip.DeviceBuffer.alloc(&e.driver, 8);
    defer send.free();
    var recv = try hip.DeviceBuffer.alloc(&e.driver, 8 * world);
    defer recv.free();
    host.slice(u64)[0] = @intFromBool(yes);
    try hip.raw.upload(send, 0, host.bytes[0..8], e.stream.handle);
    try e.comm.allGather(send.base(), recv.base(), 1, .i64, e.stream.handle);
    try hip.raw.download(recv, 0, host.bytes[8 .. 8 * (1 + world)], e.stream.handle);
    try e.stream.synchronize();
    for (host.slice(u64)[1 .. 1 + world]) |v| if (v == 0) return false;
    return true;
}

/// Marks slot `slot` of the last verify to keep its first `rows` rows at the next `flush`.
pub fn mark(e: *Engine, slot: usize, rows: usize) void {
    e.round.staging[slot] = @intCast(rows);
}

/// Keeps every marked slot of the last verify in one launch: linear states after the kept rows, last kept final rows.
pub fn flush(e: *Engine) !void {
    const st = &e.round;
    const last = st.last orelse return;
    const slots = last.plan.slots;
    var any = false;
    for (st.staging[0..slots]) |k| any = any or k >= 0;
    if (!any) return;
    const per = e.o.batch_rows + 1;
    const from = 4 * st.ring * per;
    const words = st.keeps.slice(i32)[st.ring * per ..][0..slots];
    st.ring = (st.ring + 1) % keep_ring;
    @memcpy(words, st.staging[0..slots]);
    @memset(st.staging[0..slots], -1);
    try st.buffer.dev.uploadAsync(last.keep_at, st.keeps.view(from, from + 4 * slots), e.stream);
    try win.keep(e.ops(&e.rounds), e.model(), last.plan, st.buffer.dev.base() + last.keep_at, last.hidden);
}
