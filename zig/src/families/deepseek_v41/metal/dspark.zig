//! The DSpark drafter on Metal: per-slot key rings, main rows absorbed from the taps, blocks of drafts proposed.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const ctx_ = @import("ctx.zig");
const kv = @import("kv.zig");
const rope = @import("rope.zig");
const scratch = @import("scratch.zig");
const weights = @import("weights.zig");
const pipelines = @import("pipelines.zig");
const layer = @import("layer.zig");
const forward = @import("forward.zig");
const plan_ = @import("dspark_plan.zig");
const dw = @import("dspark_weights.zig");
const st = @import("dspark_stage.zig");
const undo_ = @import("dspark_undo.zig");
const mkv = @import("markov.zig");
const picks_ = @import("early_picks.zig");
const round_prof = @import("round_prof.zig");

const Config = host.config.Config;
const Params = host.dspark.Params;
const Ctx = ctx_.Ctx;
const Ref = ctx_.Ref;
const R = pipelines.max_rows;

pub const Weights = dw.Weights;
pub const loadWeights = dw.load;
pub const Run = plan_.Run;

pub const Error = error{ Slots, NotAPrefix, DuplicateSlot, BadOutput, PastLimit, NoDspark };

/// A slot's committed rows from the target: bf16 [rows][targets * hidden] at `off` of `buf` (the slot's taps).
pub const Taps = struct { buf: mtl.Buffer, off: usize = 0, rows: u32 };

/// One slot of a pass: its anchor (the pending token) at `start` and the keyed draft parameters.
pub const Ask = struct { slot: u32, anchor: u32, start: u64, params: Params };

/// One slot's results: `block` drafts and confidences; `cand` and `base` (candidate ids, base logits) for trees.
pub const Proposal = struct { drafts: []u32, conf: []f32, cand: ?[]i32 = null, base: ?[]f32 = null };

/// A launch prepared for a group of asks: its rows (`n` an ask) and their positions; the records in the plan buffer.
pub const Launch = struct {
    rows: u32,
    n: u32,
    asks: []const Ask,
    /// asks whose chain runs on the GPU behind the head
    gpu: bool = false,
    /// whether this launch needs top-K GPU Markov filtering (greedy/raced asks stay on shared-w2 full-vocab)
    use_topk: bool = false,
    at: [R]u32 = undefined,
};

pub const Options = struct {
    /// slots with rings
    slots: u32,
    /// candidates a draft row keeps from its base logits (the Markov bias is added to these only)
    candidates: u32 = 64,
    /// the drafts of a pass for a slot whose last window verified fewer (0: every pass makes the config's block)
    narrow: u32 = 0,
    /// chains on the GPU over the whole vocabulary (the release's rule) in place of the candidates' chain
    gpu_chain: bool = false,
    /// top-K candidates per row for the resident single-TG GPU Markov chain (0: full-vocab only; 1..256: top-K)
    markov_topk: u32 = 0,
};

/// The drafter's state and working buffers; the config, kernels, scratch, rope tables and heads are borrowed.
pub const Pass = struct {
    a: std.mem.Allocator,
    c: *const Config,
    p: *const pipelines.Pipelines,
    s: *const scratch.Scratch,
    tables: *const rope.Tables,
    top: *const weights.Top,
    w: *const Weights,
    device: mtl.Device,
    queue: mtl.Queue,
    opt: Options,
    /// TF_DSV41_ROUND_PROF: where the passes book their times (null: no clock reads)
    prof: ?*round_prof.RoundProf = null,
    b: forward.Bufs,
    plan: plan_.Plan,
    /// act_quant of the taps rows [16][targets * hidden]
    quant: mtl.Buffer,
    /// concurrent ingest scratch (up to window rows x all stages) + 4-slot ring plan
    ib: st.IngestBuf,
    ingest_ring: u32 = 0,
    /// one ring buffer a stage: [slots][window + block][head_dim] bf16
    rings: []mtl.Buffer,
    runs: []Run,
    /// per-slot confidence-guided wide/narrow block state
    adapt: st.Adaptive,
    /// each slot's ring rows its last ingest overwrote (put back when a commit cuts that ingest's window)
    undo: undo_.Undo,
    /// the context-free slot the layer encoders' Ctx names (the drafter reads its own rings)
    noslot: kv.Slot,
    /// the drafter's weights and buffers held resident on the queue (null: no residency sets on this OS)
    resident: ?mtl.ResidencySet = null,
    /// command buffers committed and not yet waited for (`settle` waits), in commit order
    queued: [4]?mtl.CommandBuffer = @splat(null),
    timing: [4]?Timing = @splat(null),
    /// a pass queued by `begin` that `collect` finishes
    begun: ?Begun = null,
    /// chains on the GPU (Options.gpu_chain, with a confidence head of a rank the kernels take)
    mk: ?mkv.Chain = null,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, c: *const Config, p: *const pipelines.Pipelines, s: *const scratch.Scratch, tables: *const rope.Tables, top: *const weights.Top, w: *const Weights, opt: Options) !*Pass {
        if (c.mtp_layers == 0 or p.route_dspark == null or w.stages.len != c.mtp_layers) return error.NoDspark;
        if (opt.slots == 0 or c.dspark_block == 0 or c.dspark_block > R or opt.candidates == 0) return error.Slots;
        const x = try a.create(Pass);
        errdefer a.destroy(x);
        x.* = .{ .a = a, .c = c, .p = p, .s = s, .tables = tables, .top = top, .w = w, .device = device, .queue = queue, .opt = opt, .b = undefined, .plan = undefined, .quant = undefined, .ib = undefined, .rings = undefined, .runs = undefined, .adapt = undefined, .undo = undefined, .noslot = .{ .regions = &.{}, .limit = 0, .a = a } };
        x.b = try forward.Bufs.init(device, c);
        errdefer x.b.deinit();
        x.plan = try plan_.Plan.init(device);
        errdefer x.plan.deinit();
        x.quant = try device.buffer(@as(usize, R) * c.dspark_targets.len * c.hidden * 2, mtl.ResourceOptions.shared);
        errdefer x.quant.deinit();
        x.ib = try st.IngestBuf.init(device, c);
        errdefer x.ib.deinit();
        x.rings = try a.alloc(mtl.Buffer, c.mtp_layers);
        errdefer a.free(x.rings);
        var made: usize = 0;
        errdefer for (x.rings[0..made]) |r| r.deinit();
        const bytes = @as(usize, opt.slots) * plan_.poolRows(c) * c.head_dim * 2;
        for (x.rings) |*r| {
            r.* = try device.buffer(bytes, mtl.ResourceOptions.shared);
            @memset(r.contents()[0..bytes], 0);
            made += 1;
        }
        x.runs = try a.alloc(Run, opt.slots);
        errdefer a.free(x.runs);
        @memset(x.runs, .{});
        x.adapt = try st.Adaptive.init(a, opt.slots);
        errdefer x.adapt.deinit(a);
        x.undo = try undo_.Undo.init(a, device, c, x.rings, opt.slots);
        errdefer x.undo.deinit();
        if (opt.gpu_chain and w.conf != null and mkv.takes(c.dspark_markov_rank)) {
            const mk_topk: u32 = if (opt.markov_topk > 0) @min(@min(opt.markov_topk, mkv.max_topk), c.vocab) else 0;
            x.mk = try mkv.Chain.init(device, w.markov(c), mk_topk);
        }
        x.resident = x.holdResident();
        return x;
    }

    /// Every buffer a pass binds, resident in one set on the queue (outside one, an ingest waited up to 0.6 s).
    fn holdResident(x: *const Pass) ?mtl.ResidencySet {
        const set = x.device.residencySet(64) catch return null;
        for (x.rings) |r| set.add(r);
        for ([_]mtl.Buffer{ x.quant, x.undo.saved, x.ib.buf, x.ib.plan_buf, x.plan.buf, x.b.buf, x.w.arena }) |b| set.add(b);
        for (x.w.stages) |*l| {
            set.add(l.arena);
            set.add(l.table);
            for (l.backing()) |b| set.add(b);
        }
        if (x.w.head) |h| set.add(h.buf);
        if (x.mk) |*m| for ([_]mtl.Buffer{ m.t.buf, m.o.buf }) |b| set.add(b);
        set.commit();
        set.requestResidency();
        x.queue.addResidencySet(set);
        std.log.info("dsv41 drafter: {d:.1} GiB kept resident in its own set", .{mtl.ResidencySet.gibs(set.allocatedSize())});
        return set;
    }

    pub fn deinit(x: *Pass) void {
        x.settle() catch {};
        if (x.resident) |set| {
            x.queue.removeResidencySet(set);
            set.deinit();
        }
        x.undo.deinit();
        if (x.mk) |*m| m.deinit();
        for (x.rings) |r| r.deinit();
        x.a.free(x.rings);
        x.a.free(x.runs);
        x.adapt.deinit(x.a);
        x.ib.deinit();
        x.quant.deinit();
        x.plan.deinit();
        x.b.deinit();
        x.a.destroy(x);
    }

    /// The launch context of `rows` rows (the drafter reads its own rings, so no slot and no first position).
    pub fn ctx(x: *const Pass, rows: u32) Ctx {
        return .{ .c = x.c, .p = x.p, .s = x.s, .slot = &x.noslot, .tables = x.tables, .p0 = 0, .rows = rows };
    }

    /// The heads the family's chain and trees read.
    pub fn markov(x: *const Pass) host.dspark.Markov {
        return x.w.markov(x.c);
    }

    /// After a slot's verify window of `rows`: whether its next passes make the whole block or the narrow one.
    pub fn note(x: *Pass, slot: u32, rows: u32) void {
        x.adapt.note(slot, rows, x.opt.narrow, x.c.dspark_block);
    }

    /// The drafts a pass makes for a slot: the whole block after a wide window, else the narrow one.
    pub fn blockOf(x: *const Pass, slot: u32) u32 {
        return x.adapt.blockOf(slot, x.opt.narrow, x.c.dspark_block);
    }

    /// The drafts a pass makes for these asks: the least of their slots'.
    pub fn blockFor(x: *const Pass, asks: []const Ask) u32 {
        var n = x.c.dspark_block;
        for (asks) |a| n = @min(n, x.blockOf(a.slot));
        return n;
    }

    /// A new request on the slot: its context restarts at position 0.
    pub fn reset(x: *Pass, slot: u32) void {
        if (slot < x.runs.len) x.runs[slot] = .{};
        x.adapt.reset(slot);
        x.undo.forget(slot);
    }

    /// The slot's context restarts after `at` (rows that are not on the device).
    pub fn restart(x: *Pass, slot: u32, at: u64) void {
        if (slot < x.runs.len) x.runs[slot] = .{ .first = at, .end = at };
        x.undo.forget(slot);
    }

    /// Timing attached to a queued command buffer until its existing wait.
    const Timing = struct { start: u64, events: [2]round_prof.Event };

    /// The buffer is committed and kept for `settle` (the queue runs buffers in order).
    fn queue1(x: *Pass, comptime F: type, f: F, kind: mtl.DispatchType, events: [2]round_prof.Event) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const t0 = if (x.prof) |q| q.clock() else 0;
        const cb = x.queue.commandBuffer();
        const e = cb.compute(kind);
        f.encode(e) catch |err| {
            e.end();
            return err;
        };
        e.end();
        try x.hold(cb, if (x.prof != null) .{ .start = t0, .events = events } else null);
    }

    /// Commits `cb` and keeps it until `settle` (a full list is settled first).
    fn hold(x: *Pass, cb: mtl.CommandBuffer, timing: ?Timing) !void {
        if (x.queued[x.queued.len - 1] != null) try x.settle();
        cb.commit();
        for (&x.queued, &x.timing) |*q, *tm| if (q.* == null) {
            tm.* = timing;
            q.* = cb.retain();
            return;
        };
        unreachable;
    }

    /// Waits for every queued command buffer (the last one: the queue runs them in order) and checks each.
    pub fn settle(x: *Pass) !void {
        var last: ?mtl.CommandBuffer = null;
        for (x.queued) |q| if (q) |cb| {
            last = cb;
        };
        if (last) |cb| cb.wait();
        var failed = false;
        for (&x.queued, &x.timing) |*q, *tm| if (q.*) |cb| {
            if (tm.*) |t| {
                const p = x.prof.?;
                p.event(t.events[0], p.clock() -| t.start);
                p.event(t.events[1], round_prof.secondsNs(cb.gpuSeconds()));
            }
            if (cb.failure()) |msg| {
                std.debug.print("dspark command buffer failed: {s}\n", .{msg});
                failed = true;
            }
            cb.release();
            q.* = null;
            tm.* = null;
        };
        if (failed) return error.CommandBufferFailed;
    }

    // ingest -------------------------------------------------------------------------------------------------------

    /// Committed rows into the slot's rings: `taps` rows `rows[i]` (a contiguous run) land at positions start + i.
    pub fn ingest(x: *Pass, slot: u32, start: u64, taps: Taps, rows: []const u32) !void {
        try x.ingestQueued(slot, start, taps, rows);
        try x.settle();
    }

    /// Whether any drafter command buffer is still queued (rings may be in flight on the GPU).
    pub fn hasQueued(x: *const Pass) bool {
        return x.queued[0] != null;
    }

    /// Restore overwritten ring rows for `slot` from `end` on (CPU when rings idle, else queued GPU blit).
    pub fn restoreUndo(x: *Pass, slot: u32, end: u64) !void {
        if (!x.undo.pending(slot, end)) return;
        if (!x.hasQueued()) {
            x.undo.settle(slot, end);
            return;
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = x.queue.commandBuffer();
        _ = x.undo.encodeSettle(cb, slot, end);
        try x.hold(cb, null);
    }

    /// `ingest` whose last launch is left queued (`settle` waits); the taps must stay as they are until then.
    pub fn ingestQueued(x: *Pass, slot: u32, start: u64, taps: Taps, rows: []const u32) !void {
        if (slot >= x.runs.len) return error.Slots;
        if (rows.len == 0) return;
        for (rows, 0..) |r, i| if (r != rows[0] + i) return error.NotAPrefix;
        if (rows[rows.len - 1] >= taps.rows) return error.NotAPrefix;
        const sp = ingestSpan(rows.len, x.c.window);
        const keep = sp.count;
        const skip = sp.skip;
        const kn: u32 = @intCast(keep);
        if (start + skip + kn > x.tables.positions) return error.PastLimit;
        if (x.queued[x.queued.len - 1] != null) try x.settle();
        const refs = try x.ib.writePlan(x.c, x.ingest_ring, slot, start + skip, kn);
        x.ingest_ring = (x.ingest_ring + 1) % st.IngestBuf.ring_slots;
        const tap_ref: Ref = .{ .buf = taps.buf, .off = taps.off + @as(usize, rows[skip]) * x.c.dspark_targets.len * x.c.hidden * 2 };
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const t0 = if (x.prof) |q| q.clock() else 0;
        const cb = x.queue.commandBuffer();
        const cx = x.ctx(kn);
        try st.encodeIngestCb(cb, &cx, x.w, &x.ib, &x.undo, x.hasQueued(), slot, start + rows.len, start + skip, tap_ref, x.rings, refs.pos, refs.dest);
        try x.hold(cb, if (x.prof != null) .{ .start = t0, .events = .{ .ingest_wall, .ingest_gpu } } else null);
        x.runs[slot].land(start + skip, keep);
    }

    /// The ingest's encoders for `n` rows of `taps` using plan ingest rows `[r0 .. r0 + n)`.
    pub fn encodeIngestAt(x: *const Pass, e: mtl.ComputeEncoder, taps: Taps, r0: u32, n: u32) !void {
        const cx = x.ctx(n);
        const refs = x.plan.ingestRefs();
        const off = @as(usize, r0) * 4;
        const p: Ref = .{ .buf = refs.pos.buf, .off = refs.pos.off + off };
        const d: Ref = .{ .buf = refs.dest.buf, .off = refs.dest.off + off };
        return st.ingestConcurrent(&cx, e, x.w, &x.ib, .{ .buf = taps.buf, .off = taps.off }, x.rings, p, d);
    }

    /// The ingest's encoders for `n` rows of `taps` (from its first row) with the plan's ingest rows.
    pub fn encodeIngest(x: *const Pass, e: mtl.ComputeEncoder, taps: Taps, n: u32) !void {
        return x.encodeIngestAt(e, taps, 0, n);
    }

    // propose ------------------------------------------------------------------------------------------------------

    /// Whether an ask needs top-K filtering (top_k/top_p/min_p) rather than full-vocab greedy/raced GPU Markov.
    pub fn needsTopK(x: *const Pass, p: Params) bool {
        return p.sampled and !picks_.raced(p.sampling().?, x.c.vocab);
    }

    /// Whether GPU chain draws this ask (top-K handles all sampling; full-vocab handles greedy and raced).
    pub fn drawsOnGpu(x: *const Pass, p: Params) bool {
        const m = x.mk orelse return false;
        return m.topk > 0 or !x.needsTopK(p);
    }

    /// The most slots a launch carries (rows are limited to 16).
    pub fn group(x: *const Pass) u32 {
        return @max(1, R / x.c.dspark_block);
    }

    /// The plan and the block ids of a launch for these asks (at most `group`, distinct slots).
    pub fn prepare(x: *Pass, asks: []const Ask) !Launch {
        const n = x.blockFor(asks);
        if (asks.len == 0 or asks.len > x.group()) return error.Slots;
        var l: Launch = .{ .rows = @intCast(asks.len * n), .n = n, .asks = asks, .gpu = x.mk != null };
        var any_topk = false;
        for (asks) |a| {
            l.gpu = l.gpu and x.drawsOnGpu(a.params);
            any_topk = any_topk or x.needsTopK(a.params);
        }
        l.use_topk = l.gpu and x.mk != null and x.mk.?.topk > 0 and any_topk;
        if (l.gpu) for (asks, 0..) |a, j| x.mk.?.writeAsk(j, a.params);
        if (x.hasQueued()) {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            if (x.undo.encodeSettleAsks(x.queue, asks)) |cb| try x.hold(cb, null);
        } else {
            for (asks) |a| if (a.slot < x.runs.len) x.undo.settle(a.slot, a.start);
        }
        const ids: []u32 = @alignCast(std.mem.bytesAsSlice(u32, x.b.bytes(x.b.ids, 4 * l.rows)));
        for (asks, 0..) |a, j| {
            if (a.slot >= x.runs.len) return error.Slots;
            x.runs[a.slot].cut(a.start);
            for (asks[0..j]) |o| if (o.slot == a.slot) return error.DuplicateSlot;
            if (a.start + n > x.tables.positions or a.anchor >= x.c.vocab) return error.PastLimit;
            try x.plan.writeBlock(x.c, @intCast(j), a.slot, a.start, x.runs[a.slot], n);
            host.dspark.blockIds(.{ .block = n, .noise = x.c.dspark_noise_token }, a.anchor, ids[j * n ..][0..n]);
            for (0..n) |i| l.at[j * n + i] = @intCast(a.start + i);
        }
        return l;
    }

    pub fn rowsOf(x: *const Pass, l: *const Launch) st.Rows {
        const refs = x.plan.refs();
        return .{ .pos = refs.pos, .dest = refs.dest, .meta = refs.meta, .sel = refs.sel, .asks = refs.asks, .n_asks = l.rows / l.n, .block = l.n };
    }

    /// The stage's input and output streams: stage i reads streams[i % 2], writes the other.
    pub fn io(x: *const Pass, i: usize) layer.Io {
        const a = i % 2;
        return .{ .x = x.b.streams[a], .pre = x.b.pres[a], .x_out = x.b.streams[1 - a], .pre_out = x.b.pres[1 - a] };
    }

    pub fn encodeEmbed(x: *const Pass, e: mtl.ComputeEncoder, l: *const Launch) !void {
        const cx = x.ctx(l.rows);
        try st.embed(&cx, e, x.top, &x.b);
    }

    pub fn encodeStage(x: *const Pass, e: mtl.ComputeEncoder, l: *const Launch, i: usize) !void {
        const cx = x.ctx(l.rows);
        try st.stage(&cx, e, &x.w.stages[i], x.io(i), x.rings[i], x.rowsOf(l));
    }

    pub fn encodeHead(x: *const Pass, e: mtl.ComputeEncoder, l: *const Launch) !void {
        const cx = x.ctx(l.rows);
        try st.head(&cx, e, x.top, x.w, &x.b);
    }

    /// The whole launch: embed, every stage, the head, a barrier wherever a step reads the one before (concurrent).
    pub fn encodeAll(x: *const Pass, e: mtl.ComputeEncoder, l: *const Launch) !void {
        try x.encodeEmbed(e, l);
        e.barrier();
        for (0..x.w.stages.len) |i| try x.encodeStage(e, l, i);
        try x.encodeHead(e, l);
        if (!l.gpu) return;
        const m = &(x.mk orelse return error.NoGpuChain);
        const n = l.n;
        e.barrier();
        if (l.use_topk and m.topk > 0) {
            mkv.encodeTopK(&m.p, e, &m.t, &m.o, ctx_.cv(mkv.Ref, x.b.logits), ctx_.cv(mkv.Ref, x.b.hf), ctx_.cv(mkv.Ref, x.b.ids), x.c.vocab, x.c.hidden, l.rows / n, n, m.topk);
        } else {
            mkv.encode(&m.p, e, &m.t, &m.o, ctx_.cv(mkv.Ref, x.b.logits), ctx_.cv(mkv.Ref, x.b.hf), ctx_.cv(mkv.Ref, x.b.ids), x.c.vocab, x.c.hidden, l.rows / n, n);
        }
    }

    /// Candidates and the Markov chain over a launch's head outputs, each ask's results into `out` (its order).
    pub fn finish(x: *Pass, l: *const Launch, out: []Proposal) !void {
        const start = if (x.prof) |p| p.clock() else 0;
        defer if (x.prof) |p| p.event(.finish, p.clock() -| start);
        const n: usize = l.n;
        const B: usize = x.c.dspark_block;
        const k: usize = @min(x.opt.candidates, x.c.vocab);
        const D: usize = x.c.hidden;
        const V: usize = x.c.vocab;
        if (out.len != l.asks.len) return error.BadOutput;
        if (l.gpu) return x.finishGpu(l, out, k);
        var arena = std.heap.ArenaAllocator.init(x.a);
        defer arena.deinit();
        const sa = arena.allocator();
        const logits = x.b.logitsOf(x.c, l.rows);
        const hf: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, x.b.bytes(x.b.hf, 2 * l.rows * D)));
        const cand = try sa.alloc(i32, l.rows * k);
        const base = try sa.alloc(f32, l.rows * k);
        const hid = try sa.alloc(f32, n * D);
        const mk = x.markov();
        try host.dspark.candidateRows(logits, V, k, cand, base);
        for (l.asks, out, 0..) |a, *o, j| {
            if (o.drafts.len != B or o.conf.len != B) return error.BadOutput;
            const cd = cand[j * n * k ..][0 .. n * k];
            const cv = base[j * n * k ..][0 .. n * k];
            for (hid[0 .. n * D], hf[j * n * D ..][0 .. n * D]) |*h, v| h.* = host.dspark.bf16(v);
            try host.dspark.chain(sa, mk, cd, cv, k, a.anchor, a.params, if (mk.conf != null) hid[0 .. n * D] else null, o.drafts[0..n], o.conf[0..n]);
            x.adapt.record(a.slot, l.n, x.opt.narrow, x.c.dspark_block, o.conf[0..n]);
            @memset(o.drafts[n..], o.drafts[n - 1]);
            @memset(o.conf[n..], 0);
            try copyCands(o, B, k, cd, cv);
        }
    }

    /// A GPU chain's results: drafts and confidences as left on GPU; top-K GPU candidates copied directly for trees.
    fn finishGpu(x: *Pass, l: *const Launch, out: []Proposal, k: usize) !void {
        const n: usize = l.n;
        const B: usize = x.c.dspark_block;
        const m = &x.mk.?;
        const drafts = m.o.drafts(l.rows);
        const conf = m.o.conf(l.rows);
        const gpu_cands = l.use_topk and m.topk > 0 and k == m.topk;
        var cpu_cand: ?[]i32 = null;
        var cpu_base: ?[]f32 = null;
        defer if (cpu_cand) |b| x.a.free(b);
        defer if (cpu_base) |b| x.a.free(b);
        for (l.asks, out, 0..) |a, *o, j| {
            if (o.drafts.len != B or o.conf.len != B) return error.BadOutput;
            @memcpy(o.drafts[0..n], drafts[j * n ..][0..n]);
            @memcpy(o.conf[0..n], conf[j * n ..][0..n]);
            x.adapt.record(a.slot, l.n, x.opt.narrow, x.c.dspark_block, o.conf[0..n]);
            @memset(o.drafts[n..], o.drafts[n - 1]);
            @memset(o.conf[n..], 0);
            if (o.cand == null and o.base == null) continue;
            if (gpu_cands) {
                try copyCands(o, B, k, m.o.cand(l.rows, k)[j * n * k ..][0 .. n * k], m.o.base(l.rows, k)[j * n * k ..][0 .. n * k]);
                continue;
            }
            if (cpu_cand == null) {
                cpu_cand = try x.a.alloc(i32, n * k);
                cpu_base = try x.a.alloc(f32, n * k);
            }
            const logits = x.b.logitsOf(x.c, l.rows);
            try host.dspark.candidateRows(logits[j * n * x.c.vocab ..][0 .. n * x.c.vocab], x.c.vocab, k, cpu_cand.?, cpu_base.?);
            try copyCands(o, B, k, cpu_cand.?, cpu_base.?);
        }
    }

    fn copyCands(o: *Proposal, B: usize, k: usize, cd: []const i32, cv: []const f32) !void {
        if (o.cand) |t| {
            if (t.len != B * k) return error.BadOutput;
            @memcpy(t[0..cd.len], cd);
            @memset(t[cd.len..], -1);
        }
        if (o.base) |t| {
            if (t.len != B * k) return error.BadOutput;
            @memcpy(t[0..cv.len], cv);
            @memset(t[cv.len..], -std.math.inf(f32));
        }
    }

    const AllJob = struct {
        x: *const Pass,
        l: *const Launch,
        fn encode(j: @This(), e: mtl.ComputeEncoder) !void {
            try j.x.encodeAll(e, j.l);
        }
    };

    /// Drafts for every ask: groups of at most `group` slots, one launch each; `out[i]` answers `asks[i]`.
    pub fn propose(x: *Pass, asks: []const Ask, out: []Proposal) !void {
        if (out.len != asks.len) return error.BadOutput;
        const g = x.group();
        var at: usize = 0;
        while (at < asks.len) : (at += g) {
            const m = @min(g, asks.len - at);
            const l = try x.prepare(asks[at..][0..m]);
            try x.queue1(AllJob, .{ .x = x, .l = &l }, .concurrent, .{ .propose_wall, .propose_gpu });
            try x.settle();
            try x.finish(&l, out[at..][0..m]);
        }
    }

    /// A pass of one launch queued (`collect` finishes it); more asks than a launch holds: false, nothing queued.
    pub fn begin(x: *Pass, asks: []const Ask) !bool {
        if (x.begun != null) return error.PassBegun;
        if (asks.len == 0 or asks.len > x.group()) return false;
        x.begun = .{ .n = asks.len, .l = undefined };
        const b = &x.begun.?;
        @memcpy(b.asks[0..asks.len], asks);
        b.l = x.prepare(b.asks[0..b.n]) catch |e| {
            x.begun = null;
            return e;
        };
        x.queue1(AllJob, .{ .x = x, .l = &b.l }, .concurrent, .{ .propose_wall, .propose_gpu }) catch |e| {
            x.begun = null;
            return e;
        };
        return true;
    }

    /// The begun pass's results, in its asks' order.
    pub fn collect(x: *Pass, out: []Proposal) !void {
        if (x.begun == null) return error.NothingBegun;
        defer x.begun = null;
        try x.settle();
        try x.finish(&x.begun.?.l, out);
    }

    /// One slot's context for a block starting at `start`, as the attention reads it.
    pub fn context(x: *const Pass, slot: u32, start: u64) plan_.Context {
        return x.runs[slot].context(x.c.window, start);
    }
};

/// An ask's draws as the GPU chain reads them (the host's keyed rule: the seed, T floored at 1e-6, pos0 + i).
pub fn askOf(p: Params) mkv.Ask {
    return .{ .seed = p.seed, .pos0 = p.pos0, .t = @floatCast(@max(p.temperature, 1e-6)), .sampled = @intFromBool(p.sampled) };
}

/// A queued pass: its asks (the launch's own copy) and launch.
pub const Begun = struct { asks: [R]Ask = undefined, n: usize, l: Launch };

/// Which committed rows an ingest writes: only the last `window` (older ones are never read).
pub fn ingestSpan(n: usize, window: usize) struct { skip: usize, count: usize } {
    const skip = n -| window;
    return .{ .skip = skip, .count = n - skip };
}

test "an ingest keeps the last window of its rows" {
    const s = ingestSpan(300, 128);
    try std.testing.expectEqual(@as(usize, 172), s.skip);
    try std.testing.expectEqual(@as(usize, 128), s.count);
    try std.testing.expectEqual(@as(usize, 0), ingestSpan(5, 128).skip);
}
