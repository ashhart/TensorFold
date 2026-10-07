//! A lone drafted stream decoded as `tensorfold run` decodes: the confidence depth rule, copies, the graphs.

const std = @import("std");
const lanes = @import("lanes");
const Cuda = @import("cuda_lanes.zig").Cuda;
const drafts = @import("cuda_drafts.zig");

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
    _ = try be.opening(gpa, s); // the prompt pass, its first token committed; a cancel or failure releases the stream
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
        _ = try s.commit(gpa, v.sampled[0..kept]); // lands all `kept` unless the stream finishes inside them
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
