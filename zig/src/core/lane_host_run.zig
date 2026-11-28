//! The lane host's engine thread: one loop of admission, prompt fills, rounds and delivery until the host closes.
const std = @import("std");
const LaneHost = @import("lane_host.zig").LaneHost;
const status = @import("lane_host_status.zig");
const learn = @import("lane_host_learn.zig");
const lone = @import("lane_host_lone.zig");
const fill = @import("lane_host_fill.zig");
const ends = @import("lane_host_end.zig");
const rows_host = @import("lane_host_rows.zig");
const pressure = @import("lane_host_pressure.zig");
const rounds = @import("lane_rounds.zig");
const scoring = @import("lane_host_score.zig");
const memory_host = @import("lane_host_memory.zig");
const lane_trace = @import("lane_trace.zig");
const admit = @import("lane_host_admit.zig");

pub fn run(h: *LaneHost) void {
    defer scoring.close(h);
    if (!@import("lane_thread.zig").start(h)) return;
    // a family planner: prompt pieces and decode rows in one pass; else whole or `prefill_step` chunked prompts
    if (h.rounds) |rd| return rounds.run(h, rd);
    // learning off (no bank: folded_memory is set before the host starts) runs main's loop, memory ops compiled out
    if (h.folded_memory == null) return loop(false, h);
    return loop(true, h);
}

/// The engine thread's loop; `memory` adds the folded-memory ops (a poll between rounds, idle answers, idle wake).
fn loop(comptime memory: bool, h: *LaneHost) void {
    while (true) {
        @import("lane_host_backlog.zig").step(h); // holds precede activity: waits are idle
        var activity = @import("lane_activity.zig").Activity.enter(h.activity);
        defer activity.end();
        ends.takeCancels(h);
        pressure.step(h);
        rows_host.answer(h);
        scoring.answer(h);
        learn.refuse(h);
        if (memory) memory_host.answerOneInFlight(h);
        while (h.admitOne()) {}
        fill.run(h);
        h.core.sweep() catch |err| {
            ends.failRound(h, err);
            continue;
        };
        ends.deliverPending(h);
        status.noteLive(h);
        h.lock();
        if (h.closing.load(.acquire)) {
            const left = h.queued.items.len + h.admitted.items.len;
            h.unlock();
            if (h.lesson_open) {
                h.learner.?.abort(h.learner.?.ctx);
                h.lesson_open = false;
            }
            if (left == 0) return rows_host.flush(h);
            ends.closeAll(h);
            continue;
        }
        if (h.core.activeCount() == 0) {
            if (h.filling.items.len > 0) {
                h.unlock();
                continue;
            }
            if (memory and memory_host.ready(h)) {
                h.unlock();
                memory_host.answer(h);
                continue;
            }
            if (learn.ready(h)) {
                h.unlock();
                learn.step(h);
                continue;
            }
            if (@import("lane_activity.zig").idleRows(h, &activity)) continue;
            if (h.cancels.items.len == 0 and (!memory or h.mem_ops.asks.items.len == 0) and (h.queued.items.len == 0 or h.admitted.items.len >= h.info_.lanes or admit.holds(h)))
                h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
            h.unlock();
            continue;
        }
        h.unlock();
        const owed = h.share.debt_ns > 0; // a round is timed only while it pays a prompt step's debt down
        const timed = owed or h.srpt.on; // srpt samples every round's time and tokens
        const r0 = if (timed) std.Io.Clock.awake.now(h.io) else std.Io.Timestamp.zero;
        const before = if (h.srpt.on) @import("lane_host_srpt.zig").emittedTotal(h.core.live.items) else 0;
        var round = if (h.trace) |t| lane_trace.Event.round(t, h.core.live.items.len) else undefined;
        h.core.step() catch |e| {
            if (h.trace) |t| t.endFailed(&round);
            ends.failRound(h, e);
            continue;
        };
        if (h.trace) |t| t.end(&round);
        if (timed) {
            const ns: i64 = @intCast(r0.durationTo(std.Io.Clock.awake.now(h.io)).toNanoseconds());
            if (owed) h.share.rounded(ns);
            if (h.srpt.on) @import("lane_host_srpt.zig").round(&h.srpt, ns, h.core.live.items, before);
        }
        ends.deliverPending(h);
        lone.reclaim(h);
    }
}
