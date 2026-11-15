//! Service each posted job completely before releasing the GPU to consume or reuse its window slots.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const handoff = @import("tp_handoff.zig");

pub fn service(t: anytype) void {
    serviceFrom(t, 1);
}

pub fn serviceFrom(t: anytype, first: u64) void {
    _ = platform.c.pthread_set_qos_class_self_np(.USER_INTERACTIVE, 0);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var seq: u64 = first;
    while (!t.stop.load(.acquire) and !t.failed.load(.acquire)) : (seq += 1) {
        var began: ?u64 = null;
        while (t.handoff.posted.value() < seq or t.queued.load(.acquire) < seq) {
            if (t.stop.load(.acquire) or t.failed.load(.acquire)) return;
            if (t.submitted.load(.acquire) < seq) {
                handoff.sleep(20_000);
            } else {
                const now = platform.c.mach_absolute_time();
                if (began == null) began = now;
                if (now -% began.? > 30 * 24_000_000) return t.handoff.fail(seq, error.GpuPostTimeout);
                _ = t.handoff.posted.wait(seq, 1);
            }
        }
        const job = t.jobs[seq % t.jobs.len];
        const seen = if (t.stats) platform.c.mach_absolute_time() else 0;
        if (t.trace) std.debug.print("TP rank{d} seq {d} {s} flag {d} value {d}\n", .{ t.rank, seq, @tagName(job.kind), job.flag, job.value });
        const until = t.handoff.arm(seq, handoff.deadlines.send_budget_ns, error.TpTransferTimeout) orelse return;
        if (!t.local or job.kind == .send or job.kind == .wait) switch (job.kind) {
            .exchange, .reduce, .pick => {
                t.ep.writeSignalFromUntil(t.peer, job.local, job.remote, job.bytes, job.flag, job.value, until) catch |err| return t.handoff.fail(seq, err);
                t.ep.flushPeerUntil(t.peer, until) catch |err| return t.handoff.fail(seq, err);
            },
            .send => {
                t.rd.setDeadline(until);
                defer t.rd.setDeadline(0);
                t.sendNow(job.writes[0..job.n], job.flag, job.value) catch |err| return t.handoff.fail(seq, err);
                t.ep.flushPeerUntil(t.peer, until) catch |err| return t.handoff.fail(seq, err);
            },
            .wait => {},
        };
        if (job.kind != .send and !t.local) {
            // arm checks the send deadline under the publication lock before replacing it with this one.
            _ = t.handoff.arm(seq, handoff.deadlines.arrival_budget_ns, error.PeerSilent) orelse return;
            while (@as(i32, @bitCast(@as(u32, @truncate(@atomicLoad(u64, t.word64(job.flag), .acquire))) -% @as(u32, @truncate(job.value)))) < 0) {
                if (t.stop.load(.acquire) or t.failed.load(.acquire)) return;
                std.atomic.spinLoopHint();
            }
        }
        if (!t.handoff.publish(seq)) return;
        t.recordStats(seq, job, seen);
    }
}
