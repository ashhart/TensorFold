//! GLM-5.3-Flash's plain greedy rounds in generate: the next round committed before this one's pick is read.
const std = @import("std");
const mtl = @import("metal");
const fwd = @import("forward.zig");
const Fence = @import("../../core/fence.zig").Fence;
const Ref = @import("weights.zig").Ref;
const eng = @import("engine.zig");
const Engine = eng.Engine;

/// A committed plain round: its command buffer and when its encoding began.
const Pending = struct { cb: mtl.CommandBuffer, t_enc: u64, committed: u64 };

/// One plain round at `pos`, its token from `from` (null: the ids as set), its pick into `to`; after a round it waits on the fence, not the event.
fn plainRound(e: *Engine, x: *fwd.Ctx, from: ?Ref, to: Ref, pos: u32) Pending {
    const t_enc = std.c.mach_absolute_time();
    const fence = e.round_fence.?;
    const cb = e.queue.commandBuffer();
    if (from == null and e.ev > 0) cb.waitFor(e.event, e.ev);
    const b = .{ .cb = cb, .enc = cb.compute(.serial) };
    if (from != null) fence.wait(b.enc);
    if (from) |f| {
        b.enc.setPipeline(x.k.copy_u32);
        fwd.bind(b.enc, 0, .{ f, e.sc.ids });
        b.enc.setValue(@as(u32, 1), 2);
        b.enc.dispatchThreads(fwd.size(1, 1, 1), fwd.size(1, 1, 1));
    }
    fwd.backbone(x, b.enc, e.sc.ids, 1, pos);
    fwd.head(x, b.enc, e.sc.hidden, e.sc.logits, to, 1);
    fence.update(b.enc);
    b.enc.end();
    e.ev += 1;
    b.cb.signal(e.event, e.ev);
    b.cb.commit();
    e.committed = std.c.mach_absolute_time();
    return .{ .cb = b.cb, .t_enc = t_enc, .committed = e.committed };
}

fn waitRound(e: *Engine, p: Pending) !void {
    p.cb.wait();
    e.gpu = .{ p.cb.gpuStart(), p.cb.gpuEnd() };
    if (p.cb.failure()) |msg| {
        e.event.set(e.ev);
        std.log.err("glm: command buffer failed: {s}", .{msg});
        return error.GpuFailed;
    }
}

/// Plain rounds, each committed before the last one's pick is read; one past the reply's end is waited out, its KDA slots flipped back.
pub fn run(e: *Engine, x: *fwd.Ctx, first: u32, max_tokens: usize, eos: []const u32, out: eng.Out, res: *eng.Result) !void {
    if (e.round_fence == null) e.round_fence = try Fence.init(e.device);
    Engine.u32s(e.sc.ids, 1)[0] = first;
    var emitted: usize = 1;
    var round: usize = 0;
    var last_end: f64 = 0;
    var cur = plainRound(e, x, null, e.pick_ring, e.s.pos);
    while (true) {
        const ahead = emitted + 1 < max_tokens and e.s.pos + 3 <= e.s.cap;
        if (ahead) fwd.flipKda(x); // the next round reads this one's states, as the serial loop's next round would
        const next: ?Pending = if (ahead) plainRound(e, x, e.pick_ring.at((round % 2) * 4), e.pick_ring.at(((round + 1) % 2) * 4), e.s.pos + 1) else null;
        try waitRound(e, cur);
        res.encode_seconds += @as(f64, @floatFromInt(cur.committed - cur.t_enc)) / 24e6;
        res.gpu_seconds += e.gpu[1] - e.gpu[0];
        if (last_end > 0) res.gap_seconds += @max(e.gpu[0] - last_end, 0);
        last_end = e.gpu[1];
        const tok = Engine.u32s(e.pick_ring, 2)[round % 2];
        res.rounds += 1;
        var stop = std.mem.indexOfScalar(u32, eos, tok) != null;
        if (out.tokens(out.ctx, &.{tok})) stop = true;
        emitted += 1;
        res.generated = emitted;
        e.s.pos += 1;
        const quit: ?eng.Reason = if (stop) .stop else if (emitted >= max_tokens) .length else if (out.cancelled(out.ctx)) .cancelled else if (e.s.pos + 2 > e.s.cap) .length else null;
        if (quit) |q| {
            if (next) |n| {
                try waitRound(e, n);
                fwd.flipKda(x);
            }
            res.reason = q;
            return;
        }
        cur = next orelse return error.PipelineStalled;
        round += 1;
    }
}
