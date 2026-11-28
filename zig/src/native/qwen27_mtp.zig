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
const topk = tf.bf16_topk_gpu;

/// TF_Q27_MTP_GPU_PICK: the pick's kernel, its rows, the rows a fused save reads, and the chance's terms.
const Fast = struct {
    ops: topk.Ops,
    rows: mtl.Buffer,
    saved: mtl.Buffer,
    memo: q.mtp_pick.Memo,
    held: u32 = 0, // logits rows of the last pass with a GPU pick (0: the CPU picks)

    fn init(gpa: std.mem.Allocator, device: mtl.Device) !Fast {
        const ops = try topk.Ops.init(device);
        errdefer ops.deinit();
        const rows = try device.buffer(q.mtp_pick.max_rows * @sizeOf(topk.contract.Row), opts);
        errdefer rows.deinit();
        const saved = try device.buffer(lb.max_streams * 4, opts);
        errdefer saved.deinit();
        return .{ .ops = ops, .rows = rows, .saved = saved, .memo = try q.mtp_pick.Memo.init(gpa) };
    }

    fn deinit(f: *Fast, gpa: std.mem.Allocator) void {
        f.ops.deinit();
        f.rows.deinit();
        f.saved.deinit();
        f.memo.deinit(gpa);
    }

    /// The top-1 of each logits row when one launch holds them; a wider pass keeps the CPU pick.
    fn encode(f: *Fast, e: mtl.ComputeEncoder, logits: mtl.Buffer, rows: u32, vocab: u32) !void {
        f.held = 0;
        if (rows == 0 or rows > q.mtp_pick.max_rows) return;
        try f.ops.encode(e, .{ .buf = logits }, .{ .buf = f.rows }, q.mtp_pick.params(rows, vocab));
        f.held = rows;
    }
};

/// Picks on the head's command buffer, for a chained step.
const PickOnly = struct {
    x: *Mtp,
    rows: u32,
    pub fn before(_: PickOnly, _: mtl.ComputeEncoder) !void {}
    pub fn after(p: PickOnly, e: mtl.ComputeEncoder, _: Ref) !void {
        try p.x.fast.?.encode(e, p.x.scratch.logits, p.rows, @intCast(p.x.back.runner.model.config.vocab));
    }
};

/// One absorb on one command buffer: the kept rows, the head, its picks, each stream's newest row.
const Fused = struct {
    x: *Mtp,
    items: []const dr.Absorb,
    firsts: []const u32,
    lasts: []const usize,
    ctxs: []const *Ctx,
    total: usize,
    picks: u32,
    pub fn before(a: Fused, e: mtl.ComputeEncoder) !void {
        try a.x.gatherRows(e, a.items, a.firsts, a.total);
        e.barrier();
    }
    pub fn after(a: Fused, e: mtl.ComputeEncoder, out: Ref) !void {
        const x = a.x;
        const m = x.back.runner.model;
        const f = &x.fast.?;
        try f.encode(e, x.scratch.logits, a.picks, @intCast(m.config.vocab));
        const at = f.saved.slice(u32, a.lasts.len);
        for (a.lasts, at, a.ctxs, 0..) |last, *v, c, i| {
            v.* = @intCast(last);
            try m.glue.tapRows(e, out, .{ .buffer = f.saved, .offset = i * 4 }, .{ .buffer = c.out }, .{ .rows = 1, .width = @intCast(m.config.hidden), .capacity = @intCast(a.total), .planes = 1 });
        }
    }
};

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
    fast: ?Fast = null, // TF_Q27_MTP_GPU_PICK=1
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
        errdefer gpa.destroy(x);
        if (try @import("sys").environment.read(bool, "TF_Q27_MTP_GPU_PICK", false)) x.fast = try Fast.init(gpa, m.device);
        errdefer if (x.fast) |*f| f.deinit(gpa);
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
        if (x.fast) |*f| f.deinit(x.gpa);
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
            _ = try x.chainPass(&round, .{ .buffer = x.stage }, .last, 1);
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
        const values = x.scratch.logits.slice(u16, row * vocab + vocab)[row * vocab ..];
        if (x.fast) |*f| if (row < f.held) return q.mtp_pick.fromRow(&f.rows.slice(topk.contract.Row, f.held)[row], values, &f.memo);
        return q.mtp.pick(values);
    }

    /// A chained head pass, its picks on the same command buffer when the GPU pick is on.
    fn chainPass(x: *Mtp, round: *const Round, hidden: Ref, head: q.forward.Head, rows: u32) !Ref {
        const r = x.back.runner;
        if (x.fast == null) return r.headPass(round, &x.scratch, hidden, head);
        return q.mtp.passAround(r, round, &x.scratch, hidden, head, PickOnly{ .x = x, .rows = rows });
    }

    /// Encodes each item's kept rows into the stage, at its first row in the pass.
    fn gatherRows(x: *Mtp, e: mtl.ComputeEncoder, items: []const dr.Absorb, firsts: []const u32, total: usize) !void {
        const m = x.back.runner.model;
        const row = m.config.hidden * 2;
        const index = x.index.slice(u32, total);
        for (items, firsts) |it, first| {
            const n = count(it);
            for (index[first..][0..n], 0..) |*v, j| v.* = if (it.rows) |rs| rs[j] else @intCast(j);
            const f = it.features;
            const src = Ref{ .buffer = .{ .id = @ptrFromInt(f.buffer) }, .offset = f.offset };
            try m.glue.tapRows(e, src, .{ .buffer = x.index, .offset = first * 4 }, .{ .buffer = x.stage, .offset = first * row }, .{ .rows = @intCast(n), .width = @intCast(m.config.hidden), .capacity = f.rows, .planes = 1 });
        }
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
        var lasts: [lb.max_streams]usize = undefined;
        for (items, rd.firsts, lasts[0..items.len]) |it, first, *last| last.* = first + count(it) - 1;
        if (x.fast != null) {
            const logits: u32 = switch (head) {
                .none => 0,
                .last => 1,
                .all => @intCast(total),
            };
            _ = try q.mtp.passAround(r, &rd, &x.scratch, .{ .buffer = x.stage }, head, Fused{ .x = x, .items = items, .firsts = rd.firsts, .lasts = lasts[0..items.len], .ctxs = ctxs[0..items.len], .total = total, .picks = logits });
        } else {
            {
                const pool = mtl.objc.Pool.push();
                defer pool.pop();
                const cb = m.queue.commandBuffer();
                const e = cb.compute(.serial);
                errdefer e.end(); // an encoder left open aborts the process when it is released
                try x.gatherRows(e, items, rd.firsts, total);
                e.end();
                cb.commit();
                cb.wait();
                if (q.decode_round.gpuFailed(cb, "the MTP head's rows")) return error.RoundGpuFailure;
            }
            const out = try r.headPass(&rd, &x.scratch, .{ .buffer = x.stage }, head);
            try x.save(out, lasts[0..items.len], ctxs[0..items.len]);
        }
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
            const out = try x.chainPass(&rd, hidden, if (n == 1) .last else .all, @intCast(if (n == 1) 1 else rd.ids.len));
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

test "qwen27 MTP: its own block does not reapply the target's last-layer direction" {
    // A control vector that covers the target's last block (63 of 64), as the format allows.
    var dirs = [_]mtl.Buffer{.{ .id = @ptrFromInt(0x5150) }};
    var ids = [_]u32{63};
    var cover: [64]?usize = @splat(null);
    cover[63] = 0;
    var plan = q.steer.Plan{ .allocator = std.testing.allocator, .alpha = 0.5, .hidden = 1, .ids = &ids, .dirs = &dirs, .cover = &cover };
    var m: q.model.Model = undefined;
    m.steer = &plan;
    const fake = Ref{ .buffer = .{ .id = @ptrFromInt(0x5160) } };
    const s = q.mtp.headStages(&m, 1, undefined, .all, null, fake, fake, fake);
    // The target's own stages read the direction; the head's must not, or layer 63 is added twice.
    var target = s;
    target.steer = true;
    try std.testing.expect(target.direction(63) != null);
    try std.testing.expect(s.direction(63) == null);
}
