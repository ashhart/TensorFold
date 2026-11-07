//! The 27B's MTP head on Metal as the lane core's drafter (lanes.drafted), as qwen3_5/cuda_mtp.zig is on CUDA.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const dr = lanes.drafter;
const Stream = lanes.Stream;
const Ref = q.projection.Ref;
const Round = q.round_plan.Round;
const lb = @import("qwen27_lanes.zig");

const flags = @import("qwen27_mtp_flags.zig");
pub const max_drafts = flags.max_drafts;
pub const refusal = flags.refusal;
/// A pass's logits rows at most: every stream of a shared round, or the kept rows of a few that each want a pick.
const logit_rows = 32;
/// Hazard-tracked: a pass's staged rows and each stream's saved row move between blits and kernels.
const opts = mtl.ResourceOptions.shared;

/// A stream's head: its rows, its newest row's normed output, and the chain it holds.
const Ctx = struct {
    out: mtl.Buffer, // the newest absorbed row's normed output: a chain's first row
    len: u64 = 0, // head rows, by cache position
    first: ?q.mtp.Pick = null, // the head's pick after its newest row (null: none was asked for)
    held: [max_drafts]u32 = undefined,
    chances: [max_drafts]f64 = undefined, // each held draft's chance, its earlier drafts' included
    held_n: usize = 0,
};

pub const Mtp = struct {
    gpa: std.mem.Allocator,
    back: *lb.Metal,
    scratch: q.mtp.Scratch,
    stage: mtl.Buffer, // a pass's hidden rows, in round order
    index: mtl.Buffer, // the feature rows the stage gathers
    controls: lanes.mtp_policy.Controls,
    tap: [1]u32,
    step_ms: f64 = 0, // one chained step, timed at load
    streams: std.AutoHashMapUnmanaged(*Stream, *Ctx) = .empty,

    /// The head of the target's model, its taps the last layer's rows (the residual the final norm reads).
    pub fn init(gpa: std.mem.Allocator, back: *lb.Metal, controls: lanes.mtp_policy.Controls) !*Mtp {
        const m = back.runner.model;
        if (m.weights.mtp == null) return error.NoMtpHead;
        const cap = m.frame.capacity;
        var scratch = try q.mtp.Scratch.init(m.device, cap, m.config.hidden, m.config.vocab, logit_rows);
        errdefer scratch.deinit();
        const stage = try m.device.buffer(@as(usize, cap) * m.config.hidden * 2, opts);
        errdefer stage.deinit();
        const index = try m.device.buffer(@as(usize, cap) * 4, opts);
        errdefer index.deinit();
        const x = try gpa.create(Mtp);
        x.* = .{ .gpa = gpa, .back = back, .scratch = scratch, .stage = stage, .index = index, .controls = controls, .tap = .{@intCast(m.config.layers - 1)} };
        x.step_ms = try x.time();
        return x;
    }

    pub fn deinit(x: *Mtp) void {
        var it = x.streams.valueIterator();
        while (it.next()) |c| drop(x, c.*);
        x.streams.deinit(x.gpa);
        x.scratch.deinit();
        x.stage.deinit();
        x.index.deinit();
        x.gpa.destroy(x);
    }

    fn drop(x: *Mtp, c: *Ctx) void {
        c.out.deinit();
        x.gpa.destroy(c);
    }

    /// Drafts a round at most: the cap asked for, within a chain's.
    pub fn depth(x: *const Mtp) u32 {
        return @intCast(x.controls.limit(max_drafts));
    }

    /// One chained step's ms at load, the fastest of a few (its host pick included): slot 0's first row is rewritten.
    fn time(x: *Mtp) !f64 {
        const r = x.back.runner;
        var best: f64 = std.math.inf(f64);
        for (0..4) |_| {
            var round = try x.plan(&.{.{ .slot = 0, .start = 0, .capacity = r.capacity, .ids = &.{0} }});
            defer round.deinit();
            const t0 = mtl.clock.seconds();
            _ = try r.headPass(&round, &x.scratch, .{ .buffer = x.stage }, .last);
            _ = try x.pickRow(0);
            best = @min(best, (mtl.clock.seconds() - t0) * 1000);
        }
        return best;
    }

    fn plan(x: *const Mtp, inputs: []const q.round_plan.Input) !Round {
        const m = x.back.runner.model;
        return Round.init(x.gpa, inputs, @intCast(m.config.conv_kernel), x.back.runner.slots, @intCast(m.shard.global.vocab));
    }

    fn pickRow(x: *Mtp, row: usize) !q.mtp.Pick {
        const vocab = x.back.runner.model.config.vocab;
        return q.mtp.pick(x.scratch.logits.slice(u16, row * vocab + vocab)[row * vocab ..]);
    }

    fn ctxOf(x: *Mtp, s: *Stream) !*Ctx {
        return x.streams.get(s) orelse error.UnknownStream;
    }

    pub fn drafter(x: *Mtp) dr.Drafter {
        return .{ .kind = "MTP", .ptr = x, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn, .chances = chancesFn } };
    }

    fn cast(p: *anyopaque) *Mtp {
        return @ptrCast(@alignCast(p));
    }

    fn tapsFn(p: *anyopaque) []const u32 {
        return &cast(p).tap;
    }

    /// A shared round's streams absorb and chain in one pass a level; each draft priced by its chance.
    fn factsFn(p: *anyopaque) dr.Facts {
        const x = cast(p);
        const m = x.back.runner.model;
        // the head's keys and values are the target's (one more attention layer a slot): a stream adds its saved row
        return .{ .depth = x.depth(), .step_ms = x.step_ms, .plain_guard = m.config.moe(), .batched = true, .stream_bytes = m.config.hidden * 2, .chances = true, .confidence_floor = x.controls.floor };
    }

    fn openFn(p: *anyopaque, s: *Stream) anyerror!void {
        const x = cast(p);
        if (x.streams.contains(s)) return;
        try x.streams.ensureUnusedCapacity(x.gpa, 1);
        const c = try x.gpa.create(Ctx);
        errdefer x.gpa.destroy(c);
        c.* = .{ .out = try x.back.runner.model.device.buffer(x.back.runner.model.config.hidden * 2, opts) };
        x.streams.putAssumeCapacity(s, c);
    }

    fn releaseFn(p: *anyopaque, s: *Stream) void {
        const x = cast(p);
        if (x.streams.fetchRemove(s)) |kv| drop(x, kv.value);
    }

    fn heldFn(p: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const x = cast(p);
        for (streams, out) |s, o| {
            const c = try x.ctxOf(s);
            if (o.len > c.held_n) return error.TooFewDrafts;
            @memcpy(o, c.held[0..o.len]);
        }
    }

    fn chancesFn(p: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
        const c = try cast(p).ctxOf(s);
        if (out.len > c.held_n) return error.TooFewDrafts;
        @memcpy(out, c.chances[0..out.len]);
        return true;
    }

    fn absorbFn(p: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const x = cast(p);
        var rows: usize = 0;
        for (items) |it| rows += count(it);
        // a shared round's streams in one pass; a prompt chunk alone, and so are passes too wide for the logits rows
        if (items.len > 1 and (rows > x.back.runner.model.frame.capacity or rows > logit_rows)) {
            for (items) |it| try x.absorb(&.{it});
            return;
        }
        try x.absorb(items);
    }

    fn count(it: dr.Absorb) usize {
        return if (it.rows) |r| r.len else it.features.rows;
    }

    /// Kept rows j beside the token after each, at cache rows start + j; a pick after a stream's newest row if asked.
    fn absorb(x: *Mtp, items: []const dr.Absorb) !void {
        const r = x.back.runner;
        const m = r.model;
        const row = m.config.hidden * 2;
        var inputs: [lb.max_streams]q.round_plan.Input = undefined;
        var ctxs: [lb.max_streams]*Ctx = undefined;
        var wants: [lb.max_streams]bool = undefined;
        if (items.len > lb.max_streams) return error.BadMtpRows;
        for (items, inputs[0..items.len], ctxs[0..items.len], wants[0..items.len]) |it, *in, *c, *want| {
            const f = it.features;
            const n = count(it);
            if (f.space != .device or f.dtype != .bf16 or f.row_bytes != row or n == 0 or it.follow.len < n) return error.BadMtpRows;
            c.* = try x.ctxOf(it.stream);
            // a prompt's rows but its last want no pick: the first token's row comes next
            want.* = it.propose and it.start + n >= it.stream.prompt_len;
            in.* = .{ .slot = try x.back.slotFor(it.stream), .start = @intCast(it.start), .capacity = r.capacity, .ids = it.follow[0..n] };
        }
        var rd = try x.plan(inputs[0..items.len]);
        defer rd.deinit();
        const total = rd.ids.len;
        var picks: usize = 0;
        for (wants[0..items.len]) |w| picks += @intFromBool(w);
        const lone_last = picks == 1 and wants[items.len - 1] and rd.firsts[items.len - 1] + count(items[items.len - 1]) == total;
        const head: q.forward.Head = if (picks == 0) .none else if (lone_last) .last else .all;
        if (head == .all and total > logit_rows) return error.BadMtpRows;
        {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = m.queue.commandBuffer();
            const e = cb.compute(.serial);
            errdefer e.end(); // an encoder left open aborts the process when it is released
            const index = x.index.slice(u32, total);
            for (items, rd.firsts) |it, first| {
                const n = count(it);
                for (index[first..][0..n], 0..) |*v, j| v.* = if (it.rows) |rs| rs[j] else @intCast(j);
                const f = it.features;
                const src = Ref{ .buffer = .{ .id = @ptrFromInt(f.buffer) }, .offset = f.offset };
                try m.glue.tapRows(e, src, .{ .buffer = x.index, .offset = first * 4 }, .{ .buffer = x.stage, .offset = first * row }, .{ .rows = @intCast(n), .width = @intCast(m.config.hidden), .capacity = f.rows, .planes = 1 });
            }
            e.end();
            cb.commit();
            cb.wait();
            if (q.decode_round.gpuFailed(cb, "the MTP head's rows")) return error.RoundGpuFailure;
        }
        const out = try r.headPass(&rd, &x.scratch, .{ .buffer = x.stage }, head);
        var lasts: [lb.max_streams]usize = undefined;
        for (items, rd.firsts, lasts[0..items.len]) |it, first, *last| last.* = first + count(it) - 1;
        try x.save(out, lasts[0..items.len], ctxs[0..items.len]);
        for (items, ctxs[0..items.len], wants[0..items.len], lasts[0..items.len]) |it, c, want, last| {
            // from the target's start: a state the head did not follow (a resumed prompt) has the target's rows below
            c.len = it.start + count(it);
            c.held_n = 0;
            c.first = if (want) try x.pickRow(if (head == .last) 0 else last) else null;
        }
    }

    /// Each stream's newest normed row into its own buffer: the pass's rows are the next pass's scratch.
    fn save(x: *Mtp, out: Ref, rows: []const usize, ctxs: []const *Ctx) !void {
        const row = x.back.runner.model.config.hidden * 2;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = x.back.runner.model.queue.commandBuffer();
        const bl = cb.blit();
        for (rows, ctxs) |at, c| bl.copy(out.buffer, out.offset + at * row, c.out, 0, row);
        bl.end();
        cb.commit();
        cb.wait();
        if (q.decode_round.gpuFailed(cb, "the MTP head's row")) return error.RoundGpuFailure;
    }

    /// Each stream's chain: its first pick, then a pass a level over every stream still going, from the normed rows.
    fn holdFn(p: *anyopaque, items: []const dr.Hold) anyerror!void {
        const x = cast(p);
        const r = x.back.runner;
        const row = r.model.config.hidden * 2;
        if (items.len > lb.max_streams) return error.BadMtpRows;
        var ctxs: [lb.max_streams]*Ctx = undefined;
        var got: [lb.max_streams]usize = @splat(0);
        var src: [lb.max_streams]Ref = undefined; // where each stream's newest normed row is
        var deepest: usize = 0;
        for (items, ctxs[0..items.len], got[0..items.len], src[0..items.len]) |it, *c, *n, *at| {
            c.* = try x.ctxOf(it.stream);
            const want = @min(it.depth, max_drafts);
            c.*.held_n = want;
            deepest = @max(deepest, want);
            at.* = .{ .buffer = c.*.out };
            // no pick, or a head that does not stand where the pending token goes: fillers the verify rejects
            const first = c.*.first orelse continue;
            if (want == 0 or c.*.len + 1 != it.position) continue;
            c.*.held[0] = first.token;
            c.*.chances[0] = first.chance;
            n.* = 1;
        }
        var level: usize = 1;
        while (level < deepest) : (level += 1) {
            var inputs: [lb.max_streams]q.round_plan.Input = undefined;
            var who: [lb.max_streams]usize = undefined;
            var n: usize = 0;
            for (items, ctxs[0..items.len], got[0..items.len], 0..) |it, c, k, i| {
                // a later draft below the floor is not verified: the chain stops before running the head for it
                if (k != level or it.depth <= level) continue;
                if (x.controls.floor) |floor| if (!(c.chances[level - 1] >= floor)) continue;
                inputs[n] = .{ .slot = try x.back.slotFor(it.stream), .start = @intCast(c.len + level - 1), .capacity = r.capacity, .ids = c.held[level - 1 ..][0..1] };
                who[n] = i;
                n += 1;
            }
            if (n == 0) break;
            var rd = try x.plan(inputs[0..n]);
            defer rd.deinit();
            const hidden = if (n == 1) src[who[0]] else try x.stageRows(src[0..items.len], who[0..n], rd.firsts);
            const out = try r.headPass(&rd, &x.scratch, hidden, if (n == 1) .last else .all);
            for (who[0..n], rd.firsts) |i, first| {
                const c = ctxs[i];
                const pick = try x.pickRow(if (n == 1) 0 else first);
                c.held[level] = pick.token;
                c.chances[level] = c.chances[level - 1] * pick.chance;
                got[i] = level + 1;
                src[i] = .{ .buffer = out.buffer, .offset = out.offset + first * row };
            }
        }
        // past a stream's chain: its last draft again, at no chance (the pending token when it has none)
        for (items, ctxs[0..items.len], got[0..items.len]) |it, c, k| {
            const fill = if (k > 0) c.held[k - 1] else it.pending;
            for (c.held[k..c.held_n], c.chances[k..c.held_n]) |*t, *ch| {
                t.* = fill;
                ch.* = 0;
            }
        }
    }

    /// The streams' newest rows side by side in the stage, at their rows in the next pass.
    fn stageRows(x: *Mtp, src: []const Ref, who: []const usize, firsts: []const u32) !Ref {
        const row = x.back.runner.model.config.hidden * 2;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = x.back.runner.model.queue.commandBuffer();
        const bl = cb.blit();
        for (who, firsts) |i, first| bl.copy(src[i].buffer, src[i].offset, x.stage, first * row, row);
        bl.end();
        cb.commit();
        cb.wait();
        if (q.decode_round.gpuFailed(cb, "the MTP head's chain rows")) return error.RoundGpuFailure;
        return .{ .buffer = x.stage };
    }
};
