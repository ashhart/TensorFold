//! The lane core's HIP backend for Nemotron-H: every stream its own pages and Mamba states, a window a forward.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const serve = @import("hip_serve");
const engine = @import("engine.zig");
const forward = @import("forward.zig");
const pages = @import("pages.zig");
const head = @import("head.zig");
const prefix = @import("prefix.zig");

const be = lanes.backend;
const draw = serve.draw;
const glue = serve.backend;
const Engine = engine.Engine;
const Tensor = hip.ops.Tensor;

/// Rows a window holds at most: the pending token and the drafts (every width reproduces one-token steps).
pub const max_window = 4;

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

pub const Hip = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    streams: glue.Streams(Lane),
    head: ?*head.Head,
    sampler: draw.Drawer,
    prefix: *prefix.Prefix,
    side: prefix.Side,
    /// Bytes of one stream's Mamba state: a snapshot slot.
    state_bytes: usize,
    /// Tensor parallelism: the other ranks, which get every step before this rank runs it (`worker.follow`).
    tp: serve.protocol.Sender,

    const Shared = glue.Backend(Hip);
    pub const backend = Shared.backend;

    pub fn init(gpa: std.mem.Allocator, e: *Engine) !*Hip {
        const h = try gpa.create(Hip);
        errdefer gpa.destroy(h);
        const m = e.model_();
        const dtype: serve.sample.Dtype = if (e.act == .f16) .f16 else .bf16;
        var sampler = try draw.Drawer.init(gpa, &e.driver, dtype, m.dims.vocab, max_window);
        errdefer sampler.deinit();
        const mt = try head.Head.init(gpa, e);
        errdefer if (mt) |x| x.deinit();
        const state_bytes = pages.Stream.stateBytes(m.dims, e.act, e.cfg.count(.mamba));
        h.* = .{
            .gpa = gpa,
            .e = e,
            .streams = .init(gpa),
            .head = mt,
            .sampler = sampler,
            .prefix = undefined,
            .side = .{ .e = e, .state_bytes = state_bytes, .row_bytes = m.dims.hidden * e.act.size() },
            .state_bytes = state_bytes,
            .tp = .{ .gpa = gpa },
        };
        h.prefix = try prefix.init(gpa, &h.side, &h.tp);
        return h;
    }

    pub fn deinit(h: *Hip) void {
        if (h.tp.on()) h.tp.stop() catch {};
        h.e.stream.synchronize() catch {};
        var it = h.streams.map.valueIterator();
        while (it.next()) |l| h.free(l.*);
        h.streams.deinit();
        h.prefix.deinit();
        if (h.head) |x| x.deinit();
        h.sampler.deinit();
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

    /// The facts the round loop reads at setup: exact windows, one stream a forward, the head's chained drafts.
    pub fn facts(h: *const Hip) lanes.Model {
        const drafting = h.head != null and h.e.o.policy.mtp.drafts > 0;
        return .{
            .exact_width = max_window,
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .drafts = @min(head.max_depth, h.e.o.policy.mtp.drafts),
            .hidden_rows = false,
            .batch_rows = max_window,
            .max_streams = 1,
            .draft_streams = false,
            .plain_guard = false,
        };
    }

    pub fn drawer(h: *Hip) *draw.Drawer {
        return &h.sampler;
    }

    /// A lane's device side, freed; rank 0 tells the other ranks.
    pub fn free(h: *Hip, l: *Lane) void {
        h.tp.release(l.id) catch {};
        h.e.stream.synchronize() catch {};
        l.s.deinit(&h.e.pool);
        l.snaps.free();
        l.hidden.free();
        if (l.first_row) |row| h.gpa.free(row);
        h.gpa.destroy(l);
    }

    /// A fresh lane: a stream at the engine's capacity, its window's snapshot slots and the head's rows.
    pub fn make(h: *Hip) !*Lane {
        const e = h.e;
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
        return forward.run(o, h.e.model_(), h.e.place[0..h.e.cfg.layers], ctx, ids, rows, pos, true);
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

    /// The window's `tokens` (the pending token, then the drafts) through the model with per-row snapshots: logits.
    pub fn window(h: *Hip, l: *Lane, tokens: []const u32) !Tensor {
        const e = h.e;
        const m = e.model_();
        const n = tokens.len;
        try e.fresh();
        const ops = e.ops(&e.arena);
        const dev = try e.ids(ops, tokens);
        const ctx: forward.Ctx = .{
            .pool = &e.pool,
            .stream = &l.s,
            .snaps = .{ .buf = l.snaps, .stride = h.state_bytes },
        };
        const hidden = try forward.run(ops, m, e.place[0..e.cfg.layers], ctx, dev, n, l.s.len, false);
        try hip.raw.copy(l.hidden, 0, hidden.ptr, n * m.dims.hidden * e.act.size(), ops.stream);
        l.held_n = 0;
        l.pending = .{ .rows = n };
        return e.logits(ops, hidden, n);
    }

    /// Each stream's window from its kept length: the pending token, then the drafts; one forward a window.
    pub fn verify(h: *Hip, windows: []const be.Window, out: []be.Verified) !void {
        const e = h.e;
        for (windows, out) |w, *o| {
            const l = try h.streams.get(w.stream);
            try glue.settle(h, l);
            var tokens: [max_window]u32 = undefined;
            const n = try glue.windowTokens(w, l.held[0..l.held_n], l.s.len, &tokens);
            try l.s.cover(&e.pool, l.s.len + n);
            try h.syncPages(l);
            try h.tp.send(.verify, &.{ &.{ l.id, @intCast(n) }, tokens[0..n] });
            const y = try h.window(l, tokens[0..n]);
            var reqs: [max_window]draw.Request = undefined;
            glue.requests(w, &reqs);
            try h.sampler.draw(e.ops(&e.arena), e.stream, y, reqs[0..n], o.sampled[0..n], false);
            if (tracing()) std.log.info(
                "trace {s} verify at {d}: rows {any} -> {any}",
                .{ w.stream.id, l.s.len, tokens[0..n], o.sampled[0..n] },
            );
            @memcpy(o.drafts, tokens[1..n]);
        }
    }

    /// The head absorbs each stream's kept rows and chains `depth` greedy drafts from its pending token.
    pub fn draft(h: *Hip, requests: []const be.DraftRequest) !void {
        const e = h.e;
        const hd = h.head orelse return error.NoDraftHead;
        for (requests) |r| {
            const l = try h.streams.get(r.stream);
            try glue.keepFor(h, l, r);
            if (r.position != l.s.len + 1) {
                std.log.err("draft for {s} at {d}, the stream holds {d} slots", .{ r.stream.id, r.position, l.s.len });
                return error.PositionMismatch;
            }
            l.held_n = 0;
            // the pairs: each kept row with the token after it; a prompt's last row with the first drawn token
            var first: [1]u32 = undefined;
            const next: []const u32 = if (r.rows != null) r.follow else blk: {
                first[0] = try glue.firstToken(&h.streams.drawn, r);
                break :blk &first;
            };
            if (next.len == 0 or next.len > max_window) return error.PositionMismatch;
            // the chained steps write the head's keys past the kept rows: their pages must be the stream's own
            try l.s.cover(&e.pool, l.s.len + head.max_depth);
            try h.syncPages(l);
            try e.fresh();
            const o = e.ops(&e.arena);
            const hidden: Tensor = .{ .ptr = l.hidden.base(), .kind = e.act };
            if (tracing()) std.log.info(
                "trace {s} draft at {d}: pairs {any}, depth {d}",
                .{ r.stream.id, l.s.len, next, r.depth },
            );
            if (r.depth == 0) {
                _ = try hd.absorb(o, &l.s, hidden, next, false);
            } else l.held_n = try hd.chain(o, &l.s, hidden, next, l.held[0..@min(r.depth, head.max_depth)], false);
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
