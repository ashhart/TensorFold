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
};

/// Prefill `s` and decode it until it finishes (false) or `hooks.yield` hands it to the lane core (true).
pub fn run(gpa: std.mem.Allocator, b: *Cuda, s: *lanes.Stream, hooks: Hooks) !bool {
    const h = b.head orelse return error.NoDraftHead;
    if (!s.drafts) return error.DraftedOnly;
    const costs = b.measured orelse return error.CostsNotMeasured;
    const be = b.backend();
    if (s.logprobs != null) { // as the lane core opens a stream: the first token with the prompt's last row
        errdefer be.release(s);
        try be.prefill(s);
        const token = try be.read(try be.first(s, s.prompt_len));
        const row = [1]LogRow{(try be.firstRow(s)).forToken(token)};
        _ = try s.commit(gpa, &.{token}, &row);
    } else _ = try be.opening(gpa, s); // the prompt pass, its first token committed; a cancel or failure releases the stream
    var handed = false;
    defer if (!handed) be.release(s);
    hooks.committed(hooks.ctx);
    if (s.finished) return false;
    var rule = drafts.depthRule(costs);
    var r = try drafts.Rounds.init(gpa, b.e, h, &rule, s.prompt(), s.emitted(), null);
    defer r.deinit();
    while (true) {
        const v = try r.verify(s.context.items[s.context.items.len - 1]);
        const kept = v.accepted + 1;
        var rows: [state.max_rows]LogRow = undefined;
        if (s.logprobs) |k| try windowRows(b, k, v.sampled[0..kept], rows[0..kept]); // the window's logits stay put
        _ = try s.commit(gpa, v.sampled[0..kept], if (s.logprobs != null) rows[0..kept] else &.{}); // all `kept` unless it finishes
        s.rounds += 1;
        s.drafted += r.proposal.len;
        s.accepted += v.accepted;
        hooks.committed(hooks.ctx);
        if (s.finished) return false;
        const handing = hooks.yield(hooks.ctx);
        try r.keep(v.sampled, kept, !handing);
        if (!handing) continue;
        try b.handOver(s, kept);
        handed = true;
        return true;
    }
}

/// The kept rows of the window just verified, as the lane core's windows read them (Cuda.verifyFn).
fn windowRows(b: *Cuda, k: u8, picks: []const u32, out: []LogRow) !void {
    const e = b.e;
    try e.ops().logprobRows(e.b.logits, e.c.vocab, e.b.sampled, k, false, b.rows_dev, picks.len);
    try e.stream.synchronize();
    const words = b.rows_out.slice(u32);
    for (out, picks, 0..) |*r, pick, i| r.* = LogRow.fromWords(words[i * row_words ..][0..row_words], pick, k);
}
