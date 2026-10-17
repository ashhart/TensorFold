//! A lone drafted stream decoded as `tensorfold run` decodes: the confidence depth rule, copies, the graphs.

const std = @import("std");
const lanes = @import("lanes");
const Cuda = @import("cuda_lanes.zig").Cuda;
const drafts = @import("cuda_drafts.zig");
const state = @import("cuda_state.zig");
const LogRow = lanes.LogprobRow;
const row_words = lanes.logprob.words;

/// The server's engine thread between rounds: told when tokens land, asked whether to hand the stream to the lane core.
pub const Hooks = struct {
    ctx: *anyopaque,
    committed: *const fn (ctx: *anyopaque) void,
    yield: *const fn (ctx: *anyopaque) bool,
    resumed: bool = false,
};

/// Thinking budgets a lone stream takes: the budget's cut must lie past its first round.
pub const min_budget: u32 = 2 * state.max_rows;

/// Prefill (or resume) `s` and decode it until it finishes (false) or `hooks.yield` hands it to the lane core (true).
pub fn run(gpa: std.mem.Allocator, b: *Cuda, s: *lanes.Stream, hooks: Hooks) !bool {
    const h = b.head orelse return error.NoDraftHead;
    if (!s.drafts) return error.DraftedOnly;
    const costs = b.measured orelse return error.CostsNotMeasured;
    const be = b.backend();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var kept_rows: ?usize = null;
    if (hooks.resumed) {
        kept_rows = try b.reclaim(s) orelse return true; // mid-round in the lane core: it keeps the stream
    } else if (s.logprobs != null or s.grammar != null) { // as the lane core opens a stream: masked, with the prompt's last row
        errdefer be.release(s);
        try be.prefill(s);
        const mask = if (s.grammar) |g| try g.next(arena.allocator()) else null;
        const token = try be.read(if (mask) |m| try be.vtable.first_masked.?(be.ptr, s, s.prompt_len, m) else try be.first(s, s.prompt_len));
        const row = if (s.logprobs != null) [1]LogRow{(try be.firstRow(s)).forToken(token)} else undefined;
        _ = try s.commit(gpa, &.{token}, if (s.logprobs != null) &row else &.{});
    } else _ = try be.opening(gpa, s); // the prompt pass, its first token committed; a cancel or failure releases the stream
    var handed = false;
    defer if (!handed) be.release(s);
    if (kept_rows == null) hooks.committed(hooks.ctx);
    if (s.finished) return false;
    var rule = drafts.depthRule(costs);
    b.draftControls().configure(&rule);
    var r = if (kept_rows) |k|
        try drafts.Rounds.resumed(gpa, b.e, h, &rule, s.prompt(), s.emitted(), k, s.recall)
    else
        try drafts.Rounds.init(gpa, b.e, h, &rule, s.prompt(), s.emitted(), null, s.recall);
    defer r.deinit();
    var held: Held = .{ .b = b, .arena = &arena };
    while (true) {
        held.c = s.grammar;
        const v = try r.verify(s.context.items[s.context.items.len - 1], if (s.grammar != null) held.gate() else null);
        const kept = v.accepted + 1;
        var rows: [state.max_rows]LogRow = undefined;
        if (s.logprobs) |k| try windowRows(b, k, v.sampled[0..kept], rows[0..kept]); // the window's logits stay put
        const landed = try s.commit(gpa, v.sampled[0..kept], if (s.logprobs != null) rows[0..kept] else &.{});
        s.rounds += 1;
        s.drafted += r.proposal.len;
        s.accepted += v.accepted;
        hooks.committed(hooks.ctx);
        if (s.finished) return false;
        // a loop guard's close or a thinking budget's cut is the lane core's to force: the stream goes there first
        const forced = s.force.items.len > 0 or s.budgetWithin(state.max_rows);
        const handing = forced or hooks.yield(hooks.ctx);
        try r.keep(v.sampled, landed, !handing);
        if (!handing) continue;
        try b.handOver(s, landed);
        handed = true;
        return true;
    }
}

/// A structured stream's grammar over each window, as the lane core's windows.constrain keeps a lone stream's chain.
const Held = struct {
    b: *Cuda,
    arena: *std.heap.ArenaAllocator,
    c: ?*lanes.grammar.Constraint = null,

    fn gate(x: *Held) drafts.Gate {
        return .{ .ptr = x, .window = window };
    }

    fn window(ptr: *anyopaque, ids: []u32, planned: usize) anyerror!drafts.Gated {
        const x: *Held = @ptrCast(@alignCast(ptr));
        _ = x.arena.reset(.retain_capacity);
        const a = x.arena.allocator();
        const parents = try a.alloc(i32, planned);
        for (parents, 0..) |*p, i| p.* = @as(i32, @intCast(i)) - 1;
        const w = try x.c.?.window(a, ids[0..planned], parents, false, ids.len);
        const rows = w.rows.len + w.extra.len;
        @memcpy(ids[w.rows.len..rows], w.extra);
        return .{ .rows = rows, .mask = try x.b.putBits(0, w.masks) };
    }
};

/// The kept rows of the window just verified, as the lane core's windows read them (Cuda.verifyFn).
fn windowRows(b: *Cuda, k: u8, picks: []const u32, out: []LogRow) !void {
    const e = b.e;
    try e.ops().logprobRows(e.b.logits, e.c.vocab, e.b.sampled, k, false, b.rows_dev, picks.len);
    try e.stream.synchronize();
    const words = b.rows_out.slice(u32);
    for (out, picks, 0..) |*r, pick, i| r.* = LogRow.fromWords(words[i * row_words ..][0..row_words], pick, k);
}
