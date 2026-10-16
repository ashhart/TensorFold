//! A family's round planner under the lane host: prompts in pieces between the other streams' decode rounds.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const host = @import("lane_host.zig");
const lane_stream = @import("lane_stream.zig");
const learn = @import("lane_host_learn.zig");
const status = @import("lane_host_status.zig");
const LaneHost = host.LaneHost;
const Job = LaneHost.Job;
const Allocator = std.mem.Allocator;

/// A family's round planner: admissions, then `admitted`, begins, prompt pieces, a decode round, finals, `after`.
pub const Rounds = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const Verdict = enum { admit, wait, skip, refuse };
    pub const Piece = struct { key: usize, start: u64, end: u64, save: bool, run: bool = true };
    pub const Ask = struct { key: usize, prompt: []const u32, max_tokens: u32, submitted_s: f64 };

    pub const VTable = struct {
        /// An admission pass starts (a lane is free and requests wait).
        begin_admit: *const fn (ctx: *anyopaque) void,
        /// The next queued request: start it, wait (it and those behind it), skip past it, or refuse it (`words`).
        admit: *const fn (ctx: *anyopaque, ask: Ask, words: *[]const u8) Verdict,
        /// The pass is decided: what runs before the admitted streams begin.
        admitted: *const fn (ctx: *anyopaque) anyerror!void,
        /// Around an admitted request's stream begin (`damaged`: its saved prefix could not be restored).
        arm: *const fn (ctx: *anyopaque, key: usize) void,
        began: *const fn (ctx: *anyopaque, key: usize, damaged: bool) void,
        /// This round's pieces, in order, and the requests whose prompts are in.
        plan: *const fn (ctx: *anyopaque, pieces: *std.ArrayList(Piece), finals: *std.ArrayList(usize)) anyerror!void,
        /// A request left: finished, cancelled, failed or refused, started or not.
        left: *const fn (ctx: *anyopaque, key: usize) void,
        /// After the round: the pieces' seconds and the decode round's (finals included).
        after: *const fn (ctx: *anyopaque, piece_s: f64, decode_s: f64) void,
    };
};

/// Makes and frees a request's copy proposer (`prompt` and `eos` outlive it).
pub const Proposers = struct {
    ctx: *anyopaque,
    make: *const fn (ctx: *anyopaque, gpa: Allocator, prompt: []const u32, eos: []const u32) anyerror!lanes.proposer.Proposer,
    free: *const fn (ctx: *anyopaque, p: lanes.proposer.Proposer) void,
};

/// One round under the family's planner; false when it did nothing (nothing to admit, prefill or decode).
fn round(h: *LaneHost, rd: Rounds) bool {
    var did = false;
    // the admission pass, decided whole before any of it runs
    var admits: std.ArrayList(*Job) = .empty;
    defer admits.deinit(h.gpa);
    var refused: std.ArrayList(struct { job: *Job, why: []const u8 }) = .empty;
    defer refused.deinit(h.gpa);
    h.lock();
    if (h.queued.items.len > 0 and h.admitted.items.len < h.info_.lanes) {
        rd.vtable.begin_admit(rd.ctx);
        var free = h.info_.lanes - h.admitted.items.len;
        var i: usize = 0;
        while (i < h.queued.items.len and free > 0) {
            const job = h.queued.items[i];
            var why: []const u8 = "";
            const v = rd.vtable.admit(rd.ctx, .{ .key = @intFromPtr(job), .prompt = job.request.prompt, .max_tokens = job.request.max_tokens, .submitted_s = seconds(job.submitted) }, &why);
            switch (v) {
                .admit => {
                    if (h.admitted.append(h.gpa, job)) |_| {} else |_| break;
                    if (admits.append(h.gpa, job)) |_| {} else |_| {
                        _ = h.admitted.pop();
                        break;
                    }
                    _ = h.queued.orderedRemove(i);
                    free -= 1;
                },
                .wait => break,
                .skip => i += 1,
                .refuse => {
                    if (refused.append(h.gpa, .{ .job = job, .why = why })) |_| {} else |_| break;
                    _ = h.queued.orderedRemove(i);
                },
            }
        }
    }
    h.unlock();
    for (refused.items) |x| h.finish(x.job, .failed, x.why);
    rd.vtable.admitted(rd.ctx) catch |e| {
        h.failAll(h.words(e));
        return true;
    };
    for (admits.items) |job| {
        did = true;
        if (!lane_stream.open(h, job)) continue;
        rd.vtable.arm(rd.ctx, @intFromPtr(job));
        const b = h.core.beginStream(&job.stream) catch |e| {
            _ = h.drop(job, h.words(e));
            continue;
        };
        rd.vtable.began(rd.ctx, @intFromPtr(job), b.damaged);
    }
    // this round's pieces, a decode round of the streams decoding, then the prompts whose rows are in
    rd.vtable.plan(rd.ctx, &h.pieces, &h.finals) catch |e| {
        h.failAll(h.words(e));
        return true;
    };
    const t0 = std.Io.Clock.awake.now(h.io).toNanoseconds();
    if (h.core.cfg.piece_runs and h.core.backend.vtable.pieces != null) {
        // bounded calls for the round's pieces; a failed batch drops each affected job once
        var list: [16]lanes.backend.Backend.Piece = undefined;
        var jobs: [16]*Job = undefined;
        var at: usize = 0;
        while (at < h.pieces.items.len) {
            var n: usize = 0;
            while (at < h.pieces.items.len and n < list.len) : (at += 1) {
                const piece = h.pieces.items[at];
                if (!piece.run) continue;
                did = true;
                const job: *Job = @ptrFromInt(piece.key);
                if (job.stream.finished) continue;
                list[n] = .{ .stream = &job.stream, .start = piece.start, .end = piece.end, .save = piece.save };
                jobs[n] = job;
                n += 1;
            }
            if (n > 0) h.core.pieces(list[0..n]) catch |e| {
                for (jobs[0..n], 0..) |job, i| {
                    const seen = for (jobs[0..i]) |previous| {
                        if (previous == job) break true;
                    } else false;
                    if (!seen) dropPiece(h, job, h.words(e));
                }
            };
        }
    } else for (h.pieces.items) |piece| {
        if (!piece.run) continue;
        did = true;
        const job: *Job = @ptrFromInt(piece.key);
        if (job.stream.finished) continue;
        h.core.piece(&job.stream, piece.start, piece.end, piece.save) catch |e| {
            dropPiece(h, job, h.words(e));
        };
    }
    const t1 = std.Io.Clock.awake.now(h.io).toNanoseconds();
    // Config.join_tail: a joined prompt's last token decodes in this round's window; else its own runs after
    const join = h.core.cfg.join_tail;
    if (join and h.finals.items.len > 0) {
        did = true;
        runFinals(h);
    }
    if (h.core.activeCount() > 0) {
        did = true;
        h.core.step() catch |e| {
            h.failAll(@errorName(e));
            return true;
        };
    }
    if (!join and h.finals.items.len > 0) {
        did = true;
        runFinals(h);
    }
    const t2 = std.Io.Clock.awake.now(h.io).toNanoseconds();
    var i: usize = 0;
    while (i < h.admitted.items.len) {
        const job = h.admitted.items[i];
        if (job.started and h.deliver(job)) {
            h.lock();
            _ = h.admitted.orderedRemove(i);
            h.unlock();
        } else i += 1;
    }
    rd.vtable.after(rd.ctx, seconds(t1 - t0), seconds(t2 - t1));
    return did;
}

/// A job whose piece failed: dropped and taken out of this round's finals (`drop` frees the job).
fn dropPiece(h: *LaneHost, job: *Job, message: []const u8) void {
    const key = @intFromPtr(job);
    for (h.pieces.items) |*piece| if (piece.key == key) {
        piece.run = false;
    };
    var i: usize = 0;
    while (i < h.finals.items.len) {
        if (h.finals.items[i] == key) _ = h.finals.orderedRemove(i) else i += 1;
    }
    _ = h.drop(job, message);
}

/// The prompts whose rows are in: each stream's last row (Engine.finishStream), then it decodes with the rounds.
fn runFinals(h: *LaneHost) void {
    if (h.core.cfg.replay_runs) {
        // the finals' shared work first (one forward for all); a failure fails each
        var ss: [16]*lanes.Stream = undefined;
        var jobs: [16]*Job = undefined;
        var n: usize = 0;
        for (h.finals.items) |key| {
            const job: *Job = @ptrFromInt(key);
            if (job.stream.finished or n == ss.len) continue;
            ss[n] = &job.stream;
            jobs[n] = job;
            n += 1;
        }
        h.core.prepareFinals(ss[0..n]) catch |e| {
            for (jobs[0..n]) |job| _ = h.drop(job, h.words(e));
            return;
        };
    }
    if (h.core.cfg.join_tail and h.core.cfg.join_drafts) {
        // the round's finals together: the joined prompts' first drafts in one pass (Engine.finishStreams)
        var ss: [16]*lanes.Stream = undefined;
        var jobs: [16]*Job = undefined;
        var errs: [16]?anyerror = undefined;
        var at: usize = 0;
        while (at < h.finals.items.len) {
            var n: usize = 0;
            while (at < h.finals.items.len and n < ss.len) : (at += 1) {
                const job: *Job = @ptrFromInt(h.finals.items[at]);
                if (job.stream.finished) continue;
                ss[n] = &job.stream;
                jobs[n] = job;
                n += 1;
            }
            const joined = h.core.finishStreams(ss[0..n], errs[0..n]) catch |e| blk: {
                for (errs[0..n]) |*x| x.* = e;
                break :blk 0;
            };
            // a round's line when its finished prompts' first drafts joined in one pass
            if (joined > 0) std.log.info("join drafts: {d} of {d} finished prompts drafted in one pass", .{ joined, n });
            for (jobs[0..n], errs[0..n]) |job, x| {
                if (x) |e| {
                    _ = if (e == error.Cancelled) h.cancel(job) else h.drop(job, h.words(e));
                    continue;
                }
                h.prefilled(job, job.began);
            }
        }
        return;
    }
    for (h.finals.items) |key| {
        const job: *Job = @ptrFromInt(key);
        if (job.stream.finished) continue;
        h.core.finishStream(&job.stream) catch |e| {
            _ = if (e == error.Cancelled) h.cancel(job) else h.drop(job, h.words(e));
            continue;
        };
        h.prefilled(job, job.began);
    }
}

/// The engine thread under a round planner; an idle round waits for an arrival, a cancel or a short timeout.
pub fn run(h: *LaneHost, rd: Rounds) void {
    while (true) {
        h.takeCancels();
        h.lock();
        if (h.closing) {
            const left = h.queued.items.len + h.admitted.items.len;
            h.unlock();
            if (h.lesson_open) {
                h.learner.?.abort(h.learner.?.ctx);
                h.lesson_open = false;
            }
            if (left == 0) return;
            h.closeAll();
            continue;
        }
        const work = h.admitted.items.len > 0 or (h.queued.items.len > 0 and h.admitted.items.len < h.info_.lanes);
        if (!work and learn.ready(h)) {
            h.unlock();
            learn.step(h);
            continue;
        }
        if (!work) {
            if (h.cancels.items.len == 0) h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
            h.unlock();
            continue;
        }
        h.unlock();
        const did = round(h, rd);
        status.noteLive(h);
        if (!did) {
            h.lock();
            if (h.cancels.items.len == 0) h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(20), .clock = .awake } }) catch {};
            h.unlock();
        }
    }
}

fn seconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}
