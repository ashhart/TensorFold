//! The lane core's HIP backend: every stream its own caches, every round's windows verified in one forward.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const serve = @import("hip_serve");
const Engine = @import("engine.zig").Engine;
const state = @import("../forward/state.zig");
const mtp = @import("mtp.zig");
const costs = @import("hip_costs.zig");
const prefix = @import("prefix.zig");
const hip_prefix = @import("hip_prefix.zig");
const memory = @import("memory.zig");
const pages = @import("../forward/pages.zig");

const be = lanes.backend;
const draw = serve.draw;
const glue = serve.backend;

/// Most chains one draft request runs (the head batches as many as the engine has rows, up to this).
const max_jobs = mtp.max_chains;

pub const Lane = struct {
    caches: state.Caches,
    /// The last kept row's final hidden row: the draft head's input.
    hidden: hip.DeviceBuffer,
    /// Drafts the head holds for the next window.
    held: [mtp.max_depth]u32 = undefined,
    held_n: usize = 0,
    /// Slots written and kept: the next window starts here.
    len: usize,
    /// The stream's id on every rank.
    id: u32 = 0,
    /// The prompt pass in progress, a chunk a call (null once it is in).
    fill: ?glue.Fill = null,
    /// The last verify's slot and rows: kept whole unless keep drops some first.
    pending: ?glue.Pending = null,
    /// A structured reply's prompt row, kept for its masked first draw (null for the others).
    first_row: ?[]u16 = null,
};

pub const Hip = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    streams: glue.Streams(Lane),
    head: ?*mtp.Head = null,
    /// The streams (and rows) marked kept since the last flush, as the other ranks are told.
    marked: std.ArrayList([2]u32) = .empty,
    order: []*const lanes.Stream,
    /// The pages and linear snapshots kept from prompts for later turns (nothing until `keepPrompts`).
    prefix: *hip_prefix.Prefix,
    side: hip_prefix.Side,
    /// Scratch: the pages a match took, and the pages a round copied before writing.
    matched: std.ArrayList(u32) = .empty,
    copied: std.ArrayList([3]u32) = .empty,
    /// Tensor parallelism: the other ranks, which get every step before this rank runs it (`worker.follow`).
    tp: serve.protocol.Sender,
    /// This engine's forward and head timings, filled by `measure` (none: the rule drafts the deepest it may).
    costs: costs.Costs = .{},

    const Shared = glue.Backend(Hip);
    pub const backend = Shared.backend;

    pub fn init(gpa: std.mem.Allocator, e: *Engine) !*Hip {
        const h = try gpa.create(Hip);
        errdefer gpa.destroy(h);
        const rows = e.o.batch_rows;
        h.* = .{
            .gpa = gpa,
            .e = e,
            .streams = .init(gpa),
            .order = undefined,
            .prefix = undefined,
            .side = .{ .gpa = gpa, .e = e },
            .tp = .{ .gpa = gpa, .stopped = &e.stopped },
        };
        h.prefix = try hip_prefix.init(gpa, &h.side, &h.tp);
        errdefer h.prefix.deinit();
        h.order = try gpa.alloc(*const lanes.Stream, rows);
        errdefer gpa.free(h.order);
        // under tensor parallelism only rank 0 holds the head (whole): drafts only choose the rows every rank verifies
        h.head = try mtp.Head.init(gpa, &e.driver, &e.lib, &e.weights, e.model(), @min(rows, max_jobs));
        return h;
    }

    /// Time the forwards and the head on scratch caches, for the depth rule; before `facts`, on a lone rank.
    pub fn measure(h: *Hip) void {
        costs.measure(h.gpa, h.e, h.head, &h.costs) catch |err| {
            std.log.warn("lane costs not timed: {s}", .{@errorName(err)});
            h.costs = .{};
        };
    }

    /// Keep up to `slots` snapshots and the pages the rest of `budget` bytes buys in the prefix tree (none: off).
    pub fn keepPrompts(h: *Hip, slots: usize, budget: usize) void {
        const m = h.e.model();
        const room = memory.cache(m.spec, m.act.size(), budget, slots);
        h.keepPages(@min(room.pages, h.e.pool.count), room.snaps);
    }

    /// The prefix tree may hold `max_pages` pages and `snaps` snapshots.
    pub fn keepPages(h: *Hip, max_pages: usize, snaps: usize) void {
        h.prefix.keepPages(max_pages, snaps);
    }

    /// Rank 0 of a tensor-parallel group: every step also goes to the ranks behind `link`.
    pub fn withLink(h: *Hip, link: hip.collective.Link) void {
        h.tp.link = link;
    }

    pub fn deinit(h: *Hip) void {
        if (h.tp.on()) h.tp.stop() catch {};
        h.marked.deinit(h.gpa);
        h.tp.deinit();
        h.e.stream.synchronize() catch {};
        h.matched.deinit(h.gpa);
        h.copied.deinit(h.gpa);
        var it = h.streams.map.valueIterator();
        while (it.next()) |l| h.free(l.*);
        if (h.head) |hd| {
            if (hd.graphs.captured > 0) std.log.info("head graphs: {d} captured, {d} of {d} batches replayed", .{ hd.graphs.captured, hd.graphs.replayed, hd.graphs.rounds });
            hd.deinit();
        }
        h.streams.deinit();
        h.prefix.deinit();
        h.gpa.free(h.order);
        h.gpa.destroy(h);
    }

    /// Rows a window holds: every width keeps a row's bits (the window forward), drafts come from the core.
    pub const max_window = 16;

    /// The facts the round loop reads at setup: shared rounds of exact windows, and the MTP head's batched chains.
    pub fn facts(h: *const Hip) lanes.Model {
        const drafting = h.head != null and h.e.o.policy.mtp.drafts > 0;
        // plain rounds compete with drafted ones once the forwards are timed
        const plain_guard = drafting and h.costs.windows > 0;
        return .{
            .exact_width = max_window,
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .drafts = @min(mtp.max_depth, h.e.o.policy.mtp.drafts),
            .hidden_rows = true,
            .batch_rows = @intCast(h.e.o.batch_rows),
            .max_streams = @intCast(h.e.o.batch_rows),
            .draft_streams = drafting,
            .plain_guard = plain_guard,
            .window_costs = h.costs.window[0..h.costs.windows],
            .shared_costs = h.costs.shared[0..h.costs.shareds],
            .mtp_step_ms = h.costs.step_ms,
        };
    }

    pub fn drawer(h: *Hip) *draw.Drawer {
        return &h.e.drawer;
    }

    fn free(h: *Hip, l: *Lane) void {
        h.e.drain();
        l.caches.deinit(h.gpa);
        l.hidden.free();
        if (l.first_row) |row| h.gpa.free(row);
        h.gpa.destroy(l);
    }

    /// Whether a stream past its prompt holds a lane: a prompt's chunk then holds its round back.
    fn decoding(h: *const Hip) bool {
        var it = h.streams.map.valueIterator();
        while (it.next()) |l| if (l.*.fill == null) return true;
        return false;
    }

    /// Marks the pending window's first `rows` rows kept (the stream's length moves on); `flush` lands them.
    pub fn keepRows(h: *Hip, lane: *Lane, rows: usize) !void {
        const p = lane.pending.?;
        try h.marked.append(h.gpa, .{ lane.id, @intCast(rows) });
        h.e.keep(p.slot, rows);
        lane.len += rows;
        lane.pending = null;
    }

    /// Lands the marked keeps in one launch: linear states after the kept rows, the last kept final row for the head.
    pub fn flush(h: *Hip) !void {
        if (h.marked.items.len == 0) return;
        try h.tp.begin(.keep, &.{@intCast(h.marked.items.len)});
        for (h.marked.items) |k| try h.tp.add(&k);
        try h.tp.finish();
        h.marked.clearRetainingCapacity();
        try h.e.flush();
    }

    /// Pages for the stream's positions up to `upto`, each its own to write: the other ranks are told what changed.
    fn ensure(h: *Hip, lane: *Lane, upto: usize) !void {
        const before = lane.caches.table.items.len;
        const grown = try lane.caches.grow(h.gpa, upto);
        if (grown.len > 0) try h.tp.pages(lane.id, before, grown);
        h.copied.clearRetainingCapacity();
        try lane.caches.writable(&h.copied, h.gpa, lane.len, upto, h.e.stream.handle);
        for (h.copied.items) |c| {
            try h.tp.copy(c[1], c[2]);
            try h.tp.pages(lane.id, c[0], &.{c[2]});
        }
    }

    /// The stream's lane and prompt pass begun: its pages, the resume from a snapshot, the cuts, the other ranks told.
    pub fn begin(h: *Hip, s: *lanes.Stream) !*Lane {
        const prompt = s.prompt();
        if (prompt.len == 0 or prompt.len + s.max_new + 1 > h.e.o.capacity) return error.PromptTooLong;
        const gop = try h.streams.map.getOrPut(h.gpa, s);
        if (gop.found_existing) h.free(gop.value_ptr.*);
        errdefer h.streams.map.removeByPtr(gop.key_ptr);
        const total = prompt.len + s.max_new + max_window + 1;
        const lane = blk: {
            var caches = try h.e.emptyCaches(total);
            errdefer caches.deinit(h.gpa);
            var hidden = try hip.DeviceBuffer.alloc(&h.e.driver, h.e.model().spec.hidden * h.e.model().act.size());
            errdefer hidden.free();
            try caches.setHidden(hidden.base());
            const l = try h.gpa.create(Lane);
            l.* = .{ .caches = caches, .hidden = hidden, .len = prompt.len };
            break :blk l;
        };
        errdefer h.free(lane);
        gop.value_ptr.* = lane;
        lane.id = try h.streams.idOf(s);
        // a request resumes from the deepest shared snapshot and keeps its own marks, drafted or serial alike
        var owner: hip_prefix.Side.Owner = .{ .caches = &lane.caches, .id = lane.id };
        h.matched.clearRetainingCapacity();
        h.prefix.restored = serve.protocol.no_snapshot;
        const plan = try h.prefix.begin(prompt, s.history_len, s.shared_prefixes, &owner, &h.matched);
        defer h.gpa.free(plan.marks);
        errdefer for (h.matched.items) |id| h.e.pool.release(id);
        const at: usize = plan.from;
        const adopted = h.matched.items.len;
        try lane.caches.set(h.gpa, 0, h.matched.items);
        h.matched.clearRetainingCapacity();
        // pages for the whole request are promised now, so no round runs out of them
        const need = pages.pagesFor(lane.caches.total) - adopted;
        if (!h.prefix.store.reclaim(need)) return error.OutOfPages;
        lane.caches.promised = need;
        h.e.pool.ids.reserved += need;
        _ = try lane.caches.grow(h.gpa, prompt.len);
        s.cached = @intCast(at);
        const fill = glue.Fill.init(at, plan.marks);
        lane.fill = fill;
        // the other ranks get the stream's pages and where it resumed and cuts
        const held = lane.caches.table.items;
        try h.tp.send(
            .prefill,
            &.{
                &.{
                    lane.id,
                    @intCast(lane.caches.total),
                    @intCast(prompt.len),
                    @intCast(at),
                    @intCast(fill.count),
                    h.prefix.restored,
                    @intCast(held.len),
                },
                held,
                fill.stops[0..fill.count],
                prompt,
            },
        );
        return lane;
    }

    /// Where the chunk from `at` ends: a cut anywhere on 64 rows gives the same bits; decoding streams wait less.
    pub fn reach(h: *Hip, _: *Lane, at: usize, limit: usize) !usize {
        return prefix.chunkEnd(at, limit, if (h.decoding()) h.e.o.policy.fill_rows else prefix.chunk);
    }

    pub fn advance(h: *Hip, lane: *Lane, prompt: []const u32, from: usize, to: usize) !void {
        try h.e.advance(&lane.caches, prompt, from, to, null);
    }

    /// The prompt's last chunk, its final row kept for the head: the first token.
    pub fn finish(h: *Hip, lane: *Lane, prompt: []const u32, from: usize, req: draw.Request) !u32 {
        return h.e.prefill(&lane.caches, prompt, from, lane.hidden, req, null);
    }

    pub fn keepCut(h: *Hip, lane: *Lane, prompt: []const u32, at: usize) !void {
        var owner: hip_prefix.Side.Owner = .{ .caches = &lane.caches, .id = lane.id };
        try h.prefix.keep(&owner, prompt, at);
    }

    /// Each stream's window from its kept length: the pending token, then the host's drafts; one forward for all.
    pub fn verify(h: *Hip, windows: []const be.Window, out: []be.Verified) !void {
        if (windows.len > h.order.len) return error.WindowTooWide;
        // windows the round loop did not trim keep every row before this round's plan replaces theirs
        for (windows) |w| try glue.settle(h, try h.streams.get(w.stream));
        try h.flush();
        for (windows) |w| {
            const lane = h.streams.map.get(w.stream).?;
            try h.ensure(lane, lane.len + w.rows());
        }
        var rows: [64]Engine.Rows = undefined;
        var tokens: [64][16]u32 = undefined;
        if (windows.len > rows.len) return error.WindowTooWide;
        for (windows, 0..) |w, i| {
            const lane = try h.streams.get(w.stream);
            const n = try glue.windowTokens(w, lane.held[0..lane.held_n], lane.len, &tokens[i]);
            rows[i] = .{ .caches = &lane.caches, .pos = lane.len, .tokens = tokens[i][0..n] };
            h.order[i] = w.stream;
        }
        var reqs: [256]draw.Request = undefined;
        var drawn: [256]u32 = undefined;
        var total: usize = 0;
        for (windows) |w| {
            if (total + w.positions.len > reqs.len) return error.WindowTooWide;
            glue.requests(w, reqs[total..]);
            total += w.positions.len;
        }
        // the graph choice is rank 0's, and goes with the round
        const pick = try h.e.choose(rows[0..windows.len], null);
        try h.tp.begin(.verify, &.{ @backingInt(pick), @intCast(windows.len) });
        for (windows, rows[0..windows.len]) |w, r| {
            try h.tp.add(&.{ h.streams.map.get(w.stream).?.id, @intCast(r.tokens.len) });
            try h.tp.add(r.tokens);
        }
        try h.tp.finish();
        _ = try h.e.verify(rows[0..windows.len], reqs[0..total], &drawn);
        var at: usize = 0;
        for (windows, out, 0..) |w, *o, i| {
            @memcpy(o.sampled, drawn[at..][0..o.sampled.len]);
            const lane = h.streams.map.get(w.stream).?;
            @memcpy(o.drafts, if (w.held > 0) lane.held[0..w.held] else w.tokens);
            lane.held_n = 0;
            lane.pending = .{ .slot = i, .rows = w.rows() };
            at += w.rows();
        }
    }

    /// The head drafts `depth` from each stream's last kept row and its pending token, one head forward a step for all.
    pub fn draft(h: *Hip, requests: []const be.DraftRequest) !void {
        const head = h.head orelse return error.NoDraftHead;
        const m = h.e.model();
        var jobs: [max_jobs]mtp.Job = undefined;
        var chained: [max_jobs]*Lane = undefined;
        var n: usize = 0;
        // a shared round asks for drafts before its keep: the request names the kept rows
        for (requests) |r| try glue.keepFor(h, try h.streams.get(r.stream), r);
        try h.flush();
        for (requests) |r| {
            const lane = try h.streams.get(r.stream);
            if (r.position != lane.len + 1) {
                std.log.err("draft for {s} at {d}, the stream holds {d} slots (rows {any}, depth {d})", .{ r.stream.id, r.position, lane.len, r.rows, r.depth });
                return error.PositionMismatch;
            }
            const token: u32 = if (r.rows != null) r.follow[r.follow.len - 1] else try glue.firstToken(
                &h.streams.drawn,
                r,
            );
            lane.held_n = 0;
            if (r.depth == 0) continue;
            if (n == max_jobs) return error.WindowTooWide;
            // a chain ends after a draft the head gives under the confidence; the first drafts run whole
            const stop_under: f64 = if (r.rows != null) h.e.o.policy.mtp.confidence else 0.0;
            jobs[n] = .{
                .hidden = lane.hidden.base(),
                .token = token,
                .position = lane.len,
                .depth = r.depth,
                .sampling = glue.sampling(r.stream),
                .stop_under = stop_under,
                .out = &lane.held,
            };
            chained[n] = lane;
            n += 1;
        }
        try head.chains(&h.e.lib, h.e.stream, &h.e.drawer, m, jobs[0..n], h.e.headGraphs());
        var k: usize = 0;
        for (requests) |r| {
            if (r.depth == 0) continue;
            const lane = chained[k];
            lane.held_n = jobs[k].kept;
            k += 1;
            if (lane.held_n > r.depth or (lane.held_n < r.depth and r.rows == null)) return error.ShortChain;
        }
    }

    /// The stream's lane freed; the other ranks free theirs.
    pub fn release(h: *Hip, lane: *Lane) void {
        h.tp.release(lane.id) catch {};
        h.e.stream.synchronize() catch {};
        h.free(lane);
    }
};
