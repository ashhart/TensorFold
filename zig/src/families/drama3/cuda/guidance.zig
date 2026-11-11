//! Semantic-only CFG on CUDA: an independent backbone state over the conditional engine's weights.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("../config.zig");
const Engine = @import("engine.zig").Engine;
const fwd = @import("forward.zig");
const H = cfg.hidden;
const shared = @import("../guidance.zig");
pub const Guide = shared.Guide(Engine);
pub const promptRound = shared.promptRound;

/// Paired: each stream waits on the other's event with no host round trip; separate: the host waits on each side.
pub fn frame(g: *Guide, conditional: *Engine, sampling: @import("../common.zig").Sampling) !void {
    if (!g.state.active) return conditional.frameStep(sampling, true, null);
    const u = g.e;
    const s = u.stream;
    try u.k.gemv(s, u.w.cand_head, u.hlast, H, u.w.n_cand, 0, u.cand_logits);
    try handoff(g, s, conditional.stream);
    const guidance: ?fwd.Guidance = if (g.state.mixing()) |gamma| .{ .unconditioned = u.cand_logits, .gamma = gamma } else null;
    try fwd.enqueueFrameGuided(conditional, sampling, null, null, guidance);
    try handoff(g, conditional.stream, s);
    // Feed the actual sampled conditional frame to the unconditional backbone, never its codec tower.
    try u.k.nextInput(s, conditional.frame, u.bb_input, u.pos, u.pos_bb);
    try u.k.embedBackbone(s, u.bb_input, u.w.embed, u.w.cb_emb, u.x, 1, H, u.c.ids.sem_begin, u.c.ids.sem_end, cfg.codebook_size);
    for (0..cfg.layers) |li| try fwd.backboneLayer(u, fwd.laneBB(u, u.pos_bb), li, 1, 0);
    try u.k.rmsnorm(s, u.x, H, u.w.final_norm, u.hlast, H, 1, H, cfg.eps, true);
    try u.k.copyI32(s, u.pos_bb, u.pos, 1);
}

fn handoff(g: *Guide, from: cuda.Stream, to: cuda.Stream) !void {
    if (!g.batched) return from.synchronize();
    try g.e.ev_fork.record(from);
    try to.wait(g.e.ev_fork);
}

/// A lane's drain: waits the codec and conditional streams, then the guide's stream; the first failure returns.
pub fn complete(tts: anytype, guide: anytype) !void {
    const conditional = tts.complete();
    const unconditional = if (guide) |g| g.e.stream.synchronize() else {};
    try conditional;
    try unconditional;
}

test "drama guidance: a lane's drain fails when only the guide's stream failed, and waits it after a codec failure" {
    const Wait = struct {
        fail: bool = false,
        waits: usize = 0,
        fn complete(w: *@This()) !void {
            w.waits += 1;
            if (w.fail) return error.LaunchFailed;
        }
        fn synchronize(w: *@This()) !void {
            return w.complete();
        }
    };
    const Backbone = struct { stream: *Wait };
    const FakeGuide = struct { e: *const Backbone };
    var tts: Wait = .{};
    var stream: Wait = .{ .fail = true };
    const guide: ?FakeGuide = .{ .e = &.{ .stream = &stream } };
    try std.testing.expectError(error.LaunchFailed, complete(&tts, guide));
    tts.fail = true;
    stream.fail = false;
    try std.testing.expectError(error.LaunchFailed, complete(&tts, guide));
    try std.testing.expectEqual(2, stream.waits);
    tts.fail = false;
    try complete(&tts, @as(?FakeGuide, null)); // a lane without guidance waits only its own streams
    try std.testing.expectEqual(2, stream.waits);
}
