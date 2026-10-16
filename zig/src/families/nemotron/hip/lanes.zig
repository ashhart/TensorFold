//! The lane core's HIP backend for Nemotron-H: every stream its own pages and Mamba states, every round one forward.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const serve = @import("hip_serve");
const engine = @import("engine.zig");
const forward = @import("forward.zig");
const round = @import("round.zig");
const pages = @import("pages.zig");
const head = @import("head.zig");
const prefix = @import("prefix.zig");
const costs = @import("costs.zig");

const be = lanes.backend;
const draw = serve.draw;
const glue = serve.backend;
const Engine = engine.Engine;
const Tensor = hip.ops.Tensor;

/// Rows a window holds at most: the pending token and the drafts (every width reproduces one-token steps).
pub const max_window = round.max_window;

pub const Lane = struct {
    s: pages.Stream,
    /// Each row's Mamba states of the last verify (slot 0 the state before it), for a keep of fewer rows.
    snaps: hip.DeviceBuffer,
    /// The last verify's final-normed rows (a prompt's last row in row 0): the draft head's inputs.
    hidden: hip.DeviceBuffer,
    /// Drafts the head holds for the next window.
    held: [head.max_depth]u32 = undefined,
    held_n: usize = 0,
    /// The prompt pass in progress, a chunk a call (null once it is in).
    fill: ?glue.Fill = null,
    /// The last verify's rows: kept whole unless keep drops some first.
    pending: ?glue.Pending = null,
    /// A structured reply's prompt row, kept for its masked first draw.
    first_row: ?[]u16 = null,
    /// The prompt pass resumed from a kept state: the head still owes the pair of its last row and the next token.
    resumed: bool = false,
    /// The stream's id on every rank.
    id: u32 = 0,
    /// Pages of the table the other ranks hold already.
    known: usize = 0,
};

/// TF_NEMOTRON_TRACE=1 logs every window, keep and draft (a check of the lane protocol against serial runs).
fn tracing() bool {
    const v = std.c.getenv("TF_NEMOTRON_TRACE") orelse return false;
    return v[0] == '1';
}

/// A round's scratch on the host: each window's lane, tokens, rows and draws, sized for the engine's biggest round.
const Scratch = struct {
    lanes: []*Lane,
    tokens: [][max_window]u32,
    rows: []round.Rows,
    jobs: []head.Job,
    firsts: []u32,
    reqs: []draw.Request,
    drawn: []u32,

    fn init(gpa: std.mem.Allocator, streams: usize, rows: usize) !Scratch {
        return .{
            .lanes = try gpa.alloc(*Lane, streams),
            .tokens = try gpa.alloc([max_window]u32, streams),
            .rows = try gpa.alloc(round.Rows, streams),
            .jobs = try gpa.alloc(head.Job, streams),
            .firsts = try gpa.alloc(u32, streams),
            .reqs = try gpa.alloc(draw.Request, rows),
            .drawn = try gpa.alloc(u32, rows),
        };
    }

    fn deinit(sc: *Scratch, gpa: std.mem.Allocator) void {
        gpa.free(sc.drawn);
        gpa.free(sc.reqs);
        gpa.free(sc.firsts);
        gpa.free(sc.jobs);
        gpa.free(sc.rows);
        gpa.free(sc.tokens);
        gpa.free(sc.lanes);
    }
};

pub const Hip = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    streams: glue.Streams(Lane),
    head: ?*head.Head,
    sampler: draw.Drawer,
    prefix: *prefix.Prefix,
    side: prefix.Side,
    sc: Scratch,
    /// Bytes of one stream's Mamba state: a snapshot slot.
    state_bytes: usize,
    /// Tensor parallelism: the other ranks, which get every step before this rank runs it (`worker.follow`).
    tp: serve.protocol.Sender,
    /// The forwards and the head timed at load (none: the rule drafts by acceptance alone).
    costs: costs.Costs = .{},
    /// Lanes of finished requests, their buffers kept: a new request takes one, so its rounds' graphs come back.
    spare: std.ArrayList(*Lane) = .empty,

    const Shared = glue.Backend(Hip);
    pub const backend = Shared.backend;

    pub fn init(gpa: std.mem.Allocator, e: *Engine) !*Hip {
        const h = try gpa.create(Hip);
        errdefer gpa.destroy(h);
        const m = e.model_();
        const dtype: serve.sample.Dtype = if (e.act == .f16) .f16 else .bf16;
        var sampler = try draw.Drawer.init(gpa, &e.driver, dtype, m.dims.vocab, e.plan.rows);
        errdefer sampler.deinit();
        const mt = try head.Head.init(gpa, e);
        errdefer if (mt) |x| x.deinit();
        var sc = try Scratch.init(gpa, e.plan.slots, e.plan.rows);
        errdefer sc.deinit(gpa);
        const state_bytes = pages.Stream.stateBytes(m.dims, e.act, e.cfg.count(.mamba));
        h.* = .{
            .gpa = gpa,
            .e = e,
            .streams = .init(gpa),
            .head = mt,
            .sampler = sampler,
            .prefix = undefined,
            .side = .{ .e = e, .state_bytes = state_bytes, .row_bytes = m.dims.hidden * e.act.size() },
            .sc = sc,
            .state_bytes = state_bytes,
            .tp = .{ .gpa = gpa, .stopped = &e.stopped },
        };
        h.prefix = try prefix.init(gpa, &h.side, &h.tp);
        return h;
    }

    pub fn deinit(h: *Hip) void {
        if (h.tp.on()) h.tp.stop() catch {};
        h.e.stream.synchronize() catch {};
        var it = h.streams.map.valueIterator();
        while (it.next()) |l| h.free(l.*);
        for (h.spare.items) |l| h.drop(l);
        h.spare.deinit(h.gpa);
        h.streams.deinit();
        h.prefix.deinit();
        if (h.head) |x| x.deinit();
        h.sampler.deinit();
        h.sc.deinit(h.gpa);
        h.tp.deinit();
        h.gpa.destroy(h);
    }

    /// Rank 0 of a tensor-parallel group: every step also goes to the ranks behind `link`.
    pub fn withLink(h: *Hip, link: hip.collective.Link) void {
        h.tp.link = link;
    }

    /// The pages the lane's table grew by since the other ranks last heard, sent to them.
    fn syncPages(h: *Hip, l: *Lane) !void {
        const n = l.s.ids.items.len;
        if (h.tp.on() and n > l.known) try h.tp.pages(l.id, l.known, l.s.ids.items[l.known..]);
        l.known = n;
    }

    /// Device bytes one lane holds beside its pages: the Mamba state, its window's snapshots, the head's rows.
    pub fn laneBytes(e: *const Engine) usize {
        const m = e.model_();
        const state = pages.Stream.stateBytes(m.dims, e.act, e.cfg.count(.mamba));
        return state * (max_window + 2) + max_window * m.dims.hidden * e.act.size() + 4 * pages.pagesFor(e.o.capacity);
    }

    /// The facts the round loop reads at setup: exact windows, every stream's window in one round, batched drafts.
    pub fn facts(h: *const Hip) lanes.Model {
        const drafting = h.head != null and h.e.o.policy.mtp.drafts > 0;
        // plain rounds compete with drafted ones once the forwards are timed
        const timed = drafting and h.costs.windows > 0;
        return .{
            .exact_width = max_window,
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .drafts = @min(head.max_depth, h.e.o.policy.mtp.drafts),
            .hidden_rows = true,
            .batch_rows = @intCast(h.e.plan.rows),
            .max_streams = @intCast(h.e.plan.slots),
            .draft_streams = drafting,
            .plain_guard = timed,
            .window_costs = h.costs.window[0..h.costs.windows],
            .shared_costs = h.costs.shared[0..h.costs.shareds],
            .mtp_step_ms = h.costs.step_ms,
        };
    }

    pub fn drawer(h: *Hip) *draw.Drawer {
        return &h.sampler;
    }

    /// A lane's pages back to the pool and its buffers kept for the next request; rank 0 tells the other ranks.
    pub fn free(h: *Hip, l: *Lane) void {
        h.tp.release(l.id) catch {};
        h.e.stream.synchronize() catch {};
        if (l.first_row) |row| h.gpa.free(row);
        l.first_row = null;
        l.s.empty(&h.e.pool);
        if (h.spare.items.len < h.e.plan.slots) {
            h.spare.append(h.gpa, l) catch return h.drop(l);
            return;
        }
        h.drop(l);
    }

    /// A lane's device side freed (its pages are back already).
    fn drop(h: *Hip, l: *Lane) void {
        l.s.deinit(&h.e.pool);
        l.snaps.free();
        l.hidden.free();
        h.gpa.destroy(l);
    }

    /// A fresh lane: a stream at the engine's capacity, its window's snapshot slots and the head's rows.
    pub fn make(h: *Hip) !*Lane {
        const e = h.e;
        if (h.spare.pop()) |l| {
            // a finished request's lane: its Mamba state zeroed, as a new stream's is
            errdefer h.drop(l);
            try l.s.clear(e.stream);
            const kept: Lane = .{ .s = l.s, .snaps = l.snaps, .hidden = l.hidden };
            l.* = kept;
            return l;
        }
        const m = e.model_();
        const l = try h.gpa.create(Lane);
        errdefer h.gpa.destroy(l);
        var st = try e.openStream();
        errdefer st.deinit(&e.pool);
        var snaps = try hip.DeviceBuffer.alloc(&e.driver, (max_window + 1) * h.state_bytes);
        errdefer snaps.free();
        var hidden = try hip.DeviceBuffer.alloc(&e.driver, max_window * m.dims.hidden * e.act.size());
        errdefer hidden.free();
        l.* = .{ .s = st, .snaps = snaps, .hidden = hidden };
        return l;
    }

    /// The last verify's first `rows` rows kept: the Mamba states back to the row's when fewer than all.
    pub fn keepRows(h: *Hip, l: *Lane, rows: usize) !void {
        const n = (l.pending orelse return).rows;
        if (rows > n) return error.PositionMismatch;
        try h.tp.send(.keep, &.{&.{ l.id, @intCast(rows) }});
        if (rows < n) try hip.raw.copy(
            l.s.mem,
            0,
            l.snaps.base() + rows * h.state_bytes,
            h.state_bytes,
            h.e.stream.handle,
        );
        if (tracing()) std.log.info("trace keep {d} of {d} at {d}", .{ rows, n, l.s.len });
        l.s.len += rows;
        l.pending = null;
    }

    /// The keeps land as they are marked: nothing waits for a flush.
    pub fn flush(_: *Hip) !void {}

    /// The stream's lane and prompt pass begun: its pages, the resume from the prompt cache, the cuts.
    pub fn begin(h: *Hip, s: *lanes.Stream) !*Lane {
        const e = h.e;
        const prompt = s.prompt();
        if (prompt.len == 0 or prompt.len + s.max_new + max_window + 1 > e.o.capacity) return error.PromptTooLong;
        const gop = try h.streams.map.getOrPut(h.gpa, s);
        if (gop.found_existing) h.free(gop.value_ptr.*);
        errdefer h.streams.map.removeByPtr(gop.key_ptr);
        const l = try h.make();
        errdefer h.free(l);
        gop.value_ptr.* = l;
        l.id = try h.streams.idOf(s);
        // pages for the whole request are promised now, so no round runs out of them
        const total = prompt.len + s.max_new + max_window + 1;
        const need = pages.pagesFor(total);
        if (!h.prefix.store.reclaim(need)) return error.OutOfPages;
        l.s.promised = need;
        e.pool.ids.reserved += need;
        // a request resumes from the deepest state the prompt cache holds and keeps its own marks
        h.prefix.restored = serve.protocol.no_snapshot;
        var adopt: std.ArrayList(u32) = .empty;
        defer adopt.deinit(h.gpa);
        const plan = try h.prefix.begin(prompt, s.history_len, s.shared_prefixes, l, &adopt);
        defer h.gpa.free(plan.marks);
        if (adopt.items.len > 0) {
            // the tree held them once more for this stream already
            errdefer for (adopt.items) |id| e.pool.release(id);
            try l.s.adopt(&e.pool, adopt.items, false);
        }
        if (plan.from > 0) {
            l.s.len = plan.from;
            l.s.mtp_len = plan.from - 1;
            l.resumed = true;
        }
        s.cached = @intCast(plan.from);
        l.fill = glue.Fill.init(plan.from, plan.marks);
        // the other ranks get the stream's pages, the slot it resumed from and the prompt
        const held = l.s.ids.items;
        try h.tp.send(
            .prefill,
            &.{
                &.{ l.id, @intCast(plan.from), h.prefix.restored, @intCast(held.len) },
                held,
                &.{@intCast(prompt.len)},
                prompt,
            },
        );
        if (h.tp.on()) l.known = held.len;
        return l;
    }

    /// The prompt's rows [from, to) through the model (the head absorbing them): the last row's logits at the end.
    pub const chunk = @import("prompt.zig").chunk;

    pub fn runPrompt(h: *Hip, o: hip.ops.Ops, ctx: forward.Ctx, ids: u64, rows: usize, pos: usize) !Tensor {
        return forward.run(o, h.e.model_(), h.e.place[0..h.e.cfg.layers], ctx, ids, rows, pos);
    }

    /// Where the chunk from `at` ends: the engine's chunk rows, or the cut; its pages ready on every rank.
    pub fn reach(h: *Hip, l: *Lane, at: usize, limit: usize) !usize {
        const to = @min(limit, (at / h.e.o.chunk + 1) * h.e.o.chunk);
        try l.s.cover(&h.e.pool, to);
        try h.syncPages(l);
        return to;
    }

    pub fn cancelOf(_: anytype, s: *lanes.Stream) ?Engine.Cancel {
        const c = s.cancel_check orelse return null;
        return .{ .ctx = c.ptr, .check = c.check };
    }

    pub fn advance(h: anytype, l: *Lane, prompt: []const u32, from: usize, to: usize, cancel: ?Engine.Cancel) !void {
        _ = try h.chunk(l, prompt, from, to, cancel);
    }

    /// The prompt's last chunk; its last row's token drawn as `req` says.
    pub fn finish(h: anytype, l: *Lane, prompt: []const u32, from: usize, req: draw.Request, cancel: ?Engine.Cancel) !u32 {
        const e = h.e;
        const y = try h.chunk(l, prompt, from, prompt.len, cancel);
        var token: [1]u32 = undefined;
        try h.sampler.draw(e.ops(&e.arena), e.stream, y.?, &.{req}, &token, false);
        return token[0];
    }

    pub fn keepCut(h: *Hip, l: *Lane, prompt: []const u32, at: usize) !void {
        try h.prefix.keep(l, prompt, at);
    }

    /// The lanes' windows (each the pending token, then its drafts) in one round with per-row snapshots: the logits.
    pub fn windows(h: *Hip, ls: []const *Lane, tokens: []const []const u32) !Tensor {
        const e = h.e;
        const m = e.model_();
        const width = m.dims.hidden * e.act.size();
        var total: usize = 0;
        for (ls, tokens, h.sc.rows[0..ls.len]) |l, t, *r| {
            r.* = .{ .s = &l.s, .tokens = t, .snaps = .{ .buf = l.snaps, .stride = h.state_bytes } };
            total += t.len;
        }
        const hidden = try e.round(h.sc.rows[0..ls.len], null);
        const o = e.ops(&e.arena);
        var at: usize = 0;
        for (ls, tokens) |l, t| {
            try hip.raw.copy(l.hidden, 0, hidden.ptr + at * width, t.len * width, o.stream);
            l.held_n = 0;
            l.pending = .{ .rows = t.len };
            at += t.len;
        }
        return e.roundLogits(o, hidden, total);
    }

    /// Each stream's window from its kept length: the pending token, then the drafts; one forward for all of them.
    pub fn verify(h: *Hip, ws: []const be.Window, out: []be.Verified) !void {
        const e = h.e;
        if (ws.len > h.sc.lanes.len) return error.WindowTooWide;
        var toks: [64][]const u32 = undefined;
        if (ws.len > toks.len) return error.WindowTooWide;
        var total: usize = 0;
        for (ws, 0..) |w, i| {
            const l = try h.streams.get(w.stream);
            try glue.settle(h, l);
            const n = try glue.windowTokens(w, l.held[0..l.held_n], l.s.len, &h.sc.tokens[i]);
            try l.s.cover(&e.pool, l.s.len + n);
            try h.syncPages(l);
            h.sc.lanes[i] = l;
            toks[i] = h.sc.tokens[i][0..n];
            if (total + n > h.sc.reqs.len) return error.WindowTooWide;
            glue.requests(w, h.sc.reqs[total..]);
            total += n;
        }
        try h.tp.begin(.verify, &.{@intCast(ws.len)});
        for (h.sc.lanes[0..ws.len], toks[0..ws.len]) |l, t| {
            try h.tp.add(&.{ l.id, @intCast(t.len) });
            try h.tp.add(t);
        }
        try h.tp.finish();
        const y = try h.windows(h.sc.lanes[0..ws.len], toks[0..ws.len]);
        try h.sampler.draw(e.ops(&e.arena), e.stream, y, h.sc.reqs[0..total], h.sc.drawn[0..total], false);
        var at: usize = 0;
        for (ws, out, toks[0..ws.len]) |w, *o, t| {
            @memcpy(o.sampled, h.sc.drawn[at..][0..t.len]);
            @memcpy(o.drafts, t[1..]);
            if (tracing()) std.log.info(
                "trace {s} verify at {d}: rows {any} -> {any}",
                .{ w.stream.id, h.streams.map.get(w.stream).?.s.len, t, o.sampled },
            );
            at += t.len;
        }
    }

    /// The head absorbs each stream's kept rows and chains `depth` greedy drafts from its pending token, all at once.
    pub fn draft(h: *Hip, requests: []const be.DraftRequest) !void {
        const e = h.e;
        const hd = h.head orelse return error.NoDraftHead;
        if (requests.len > h.sc.jobs.len) return error.WindowTooWide;
        for (requests, 0..) |r, i| {
            const l = try h.streams.get(r.stream);
            try glue.keepFor(h, l, r);
            if (r.position != l.s.len + 1) {
                std.log.err("draft for {s} at {d}, the stream holds {d} slots", .{ r.stream.id, r.position, l.s.len });
                return error.PositionMismatch;
            }
            l.held_n = 0;
            // the pairs: each kept row with the token after it; a prompt's last row with the first drawn token
            const next: []const u32 = if (r.rows != null) r.follow else blk: {
                h.sc.firsts[i] = try glue.firstToken(&h.streams.drawn, r);
                break :blk h.sc.firsts[i..][0..1];
            };
            if (next.len == 0 or next.len > max_window) return error.PositionMismatch;
            // the chained steps write the head's keys past the kept rows: their pages must be the stream's own
            try l.s.cover(&e.pool, l.s.len + head.max_depth);
            try h.syncPages(l);
            if (tracing()) std.log.info(
                "trace {s} draft at {d}: pairs {any}, depth {d}",
                .{ r.stream.id, l.s.len, next, r.depth },
            );
            h.sc.lanes[i] = l;
            h.sc.jobs[i] = .{
                .s = &l.s,
                .hidden = l.hidden.base(),
                .next = next,
                .out = l.held[0..@min(r.depth, head.max_depth)],
            };
        }
        try e.fresh();
        try hd.chains(e.ops(&e.arena), h.sc.jobs[0..requests.len]);
        for (requests, h.sc.lanes[0..requests.len], h.sc.jobs[0..requests.len]) |r, l, j| {
            l.held_n = j.got;
            if (l.s.mtp_len != l.s.len) {
                std.log.err("the head holds {d} rows of {s}, the stream {d}", .{ l.s.mtp_len, r.stream.id, l.s.len });
                return error.HeadBehind;
            }
        }
    }

    /// The stream's lane freed (the other ranks hear it in `free`).
    pub fn release(h: *Hip, l: *Lane) void {
        h.free(l);
    }
};
