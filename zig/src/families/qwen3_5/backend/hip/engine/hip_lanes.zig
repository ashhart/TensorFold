//! The lane core's HIP backend: every stream its own caches, every round's windows verified in one forward.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const Engine = @import("engine.zig").Engine;
const state = @import("../forward/state.zig");
const draw = @import("draw.zig");
const mtp = @import("mtp.zig");
const costs = @import("hip_costs.zig");
const prefix = @import("prefix.zig");
const hip_prefix = @import("hip_prefix.zig");
const radix = @import("engine_api").prompt_radix;
const pages = @import("../forward/pages.zig");
const worker = @import("worker.zig");

const be = lanes.backend;

/// Drawn tokens a handle names, newest last.
const ring = 1024;

/// Where a prompt pass stands: the row it has reached, and the cuts it still keeps a state at.
const Fill = struct { at: usize, stops: [16]u32, count: usize, next: usize };

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
    fill: ?Fill = null,
    /// The last verify's slot and rows: kept whole unless keep drops some first.
    pending: ?struct { window: usize, rows: usize } = null,
};

/// The engine this thread's HIP calls go to.
threadlocal var bound: ?*Engine = null;

pub const Hip = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, *Lane) = .empty,
    head: ?*mtp.Head = null,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    /// The streams (and rows) marked kept since the last flush, as the other ranks are told.
    marked: std.ArrayList([2]u32) = .empty,
    order: []*const lanes.Stream,
    /// The pages and linear snapshots kept from prompts for later turns (nothing until `keepPrompts`).
    prefix: *hip_prefix.Prefix,
    /// Scratch: the pages a match took, and the pages a round copied before writing.
    matched: std.ArrayList(u32) = .empty,
    copied: std.ArrayList([3]u32) = .empty,
    /// Tensor parallelism: the other ranks, which get every step before this rank runs it (`worker.follow`).
    link: ?*const hip.link.Link = null,
    msg: std.ArrayList(u32) = .empty,
    /// This engine's forward and head timings, filled by `measure` (none: the rule drafts the deepest it may).
    costs: costs.Costs = .{},
    /// Each stream's id on every rank.
    ids: std.AutoHashMapUnmanaged(*const lanes.Stream, u32) = .empty,
    /// Nanoseconds and calls the backend's verifies, keeps and drafts took; the core's own work is the rest.
    spent: [3]u64 = @splat(0),
    calls: [3]u64 = @splat(0),
    next_id: u32 = 0,

    pub fn init(gpa: std.mem.Allocator, e: *Engine) !*Hip {
        const h = try gpa.create(Hip);
        errdefer gpa.destroy(h);
        const rows = e.o.batch_rows;
        h.* = .{ .gpa = gpa, .e = e, .order = undefined, .prefix = try hip_prefix.Prefix.init(gpa, e) };
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
        h.prefix.keepPrompts(slots, budget);
    }

    /// The prefix tree may hold `max_pages` pages and `snaps` snapshots.
    pub fn keepPages(h: *Hip, max_pages: usize, snaps: usize) void {
        h.prefix.keepPages(max_pages, snaps);
    }

    /// Rank 0 of a tensor-parallel group: every step also goes to the ranks behind `link`.
    pub fn withLink(h: *Hip, link: *const hip.link.Link) void {
        h.link = link;
        h.prefix.link = link;
    }

    fn send(h: *Hip, words: []const u32) !void {
        if (h.link) |l| try l.send(words);
    }

    /// A stream's id on every rank.
    fn idOf(h: *Hip, s: *const lanes.Stream) !u32 {
        const gop = try h.ids.getOrPut(h.gpa, s);
        if (!gop.found_existing) {
            gop.value_ptr.* = h.next_id;
            h.next_id += 1;
        }
        return gop.value_ptr.*;
    }

    fn timed(h: *Hip, which: usize, began: u64) void {
        h.spent[which] += costs.nowNs() - began;
        h.calls[which] += 1;
    }

    pub fn deinit(h: *Hip) void {
        const names = [_][]const u8{ "verify", "keep", "draft" };
        for (names, h.spent, h.calls) |name, ns, n| if (n > 0) std.log.info("backend {s}: {d} calls, {d:.0} us each", .{ name, n, @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(n)) / 1e3 });
        if (h.link != null) h.send(&.{@backingInt(worker.Op.stop)}) catch {};
        h.ids.deinit(h.gpa);
        h.marked.deinit(h.gpa);
        h.msg.deinit(h.gpa);
        h.e.stream.synchronize() catch {};
        h.matched.deinit(h.gpa);
        h.copied.deinit(h.gpa);
        var it = h.lanes.valueIterator();
        while (it.next()) |l| h.free(l.*);
        if (h.head) |hd| {
            if (hd.graphs.captured > 0) std.log.info("head graphs: {d} captured, {d} of {d} batches replayed", .{ hd.graphs.captured, hd.graphs.replayed, hd.graphs.rounds });
            hd.deinit();
        }
        h.lanes.deinit(h.gpa);
        h.prefix.deinit();
        h.gpa.free(h.order);
        h.gpa.destroy(h);
    }

    pub fn backend(h: *Hip) be.Backend {
        return .{ .ptr = h, .vtable = &.{
            .prefill = prefillFn,
            .prefill_step = prefillStepFn,
            .first = firstFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .tree = heldFn,
            .release = releaseFn,
        } };
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

    fn free(h: *Hip, l: *Lane) void {
        h.e.drain();
        l.caches.deinit(h.gpa);
        l.hidden.free();
        h.gpa.destroy(l);
    }

    /// The backend behind `ptr`, the calling thread bound to its GPU (the lane thread did not open it).
    fn of(ptr: *anyopaque) *Hip {
        const h: *Hip = @ptrCast(@alignCast(ptr));
        if (bound != h.e) {
            h.e.ctx.makeCurrent() catch |err| std.log.err("hipSetDevice on the lane thread: {s}", .{@errorName(err)});
            bound = h.e;
        }
        return h;
    }

    fn take(h: *Hip, token: u32) u64 {
        const at = h.next;
        h.drawn[at % ring] = token;
        h.next += 1;
        return at;
    }

    /// A verify the round loop did not trim keeps every row (the core keeps only on drops).
    fn settle(h: *Hip, lane: *Lane) !void {
        const p = lane.pending orelse return;
        try h.mark(lane, p.rows);
    }

    /// Marks the pending window's first `rows` rows kept (the stream's length moves on); `flush` lands them.
    fn mark(h: *Hip, lane: *Lane, rows: usize) !void {
        const p = lane.pending.?;
        try h.marked.append(h.gpa, .{ lane.id, @intCast(rows) });
        h.e.keep(p.window, rows);
        lane.len += rows;
        lane.pending = null;
    }

    /// Lands the marked keeps in one launch: linear states after the kept rows, the last kept final row for the head.
    fn flush(h: *Hip) !void {
        if (h.marked.items.len == 0) return;
        if (h.link != null) {
            h.msg.clearRetainingCapacity();
            try h.msg.appendSlice(h.gpa, &.{ @backingInt(worker.Op.keep), @intCast(h.marked.items.len) });
            for (h.marked.items) |k| try h.msg.appendSlice(h.gpa, &k);
            try h.send(h.msg.items);
        }
        h.marked.clearRetainingCapacity();
        try h.e.flush();
    }

    fn sampling(s: *const lanes.Stream) ?lanes.Sampling {
        return s.sampling;
    }

    /// Pages for the stream's positions up to `upto`, each its own to write: the other ranks are told what changed.
    fn ensure(h: *Hip, lane: *Lane, upto: usize) !void {
        const before = lane.caches.table.items.len;
        const grown = try lane.caches.grow(h.gpa, upto);
        if (grown.len > 0) try h.prefix.sendPages(lane.id, before, grown);
        h.copied.clearRetainingCapacity();
        try lane.caches.writable(&h.copied, h.gpa, lane.len, upto, h.e.stream.handle);
        for (h.copied.items) |c| {
            try h.send(&.{ @backingInt(worker.Op.copy), c[1], c[2] });
            try h.prefix.sendPages(lane.id, c[0], &.{c[2]});
        }
    }

    /// The whole prompt pass, chunk by chunk (a driver of its own, with no rounds between).
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        while (!try prefillStepFn(ptr, s)) {}
    }

    /// The stream's lane and prompt pass begun: its pages, the resume from a snapshot, the cuts, the other ranks told.
    fn begin(h: *Hip, s: *lanes.Stream) !*Lane {
        const prompt = s.prompt();
        if (prompt.len == 0 or prompt.len + s.max_new + 1 > h.e.o.capacity) return error.PromptTooLong;
        const gop = try h.lanes.getOrPut(h.gpa, s);
        if (gop.found_existing) h.free(gop.value_ptr.*);
        errdefer h.lanes.removeByPtr(gop.key_ptr);
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
        lane.id = try h.idOf(s);
        // a drafted request resumes from the deepest shared snapshot and keeps its own marks; a serial one does neither
        var owner: hip_prefix.Owner = .{ .caches = &lane.caches, .id = lane.id };
        h.matched.clearRetainingCapacity();
        var plan: radix.Plan = .{};
        h.prefix.restored = worker.no_snapshot;
        if (s.drafts) plan = try h.prefix.begin(prompt, s.history_len, s.shared_prefixes, &owner, &h.matched);
        defer if (s.drafts) h.gpa.free(plan.marks);
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
        var fill: Fill = .{ .at = at, .stops = undefined, .count = @min(plan.marks.len, 16), .next = 0 };
        @memcpy(fill.stops[0..fill.count], plan.marks[0..fill.count]);
        lane.fill = fill;
        if (h.link != null) {
            // the other ranks get the stream's pages and where it resumed and cuts
            h.msg.clearRetainingCapacity();
            const held = lane.caches.table.items;
            try h.msg.appendSlice(h.gpa, &.{ @backingInt(worker.Op.prefill), lane.id, @intCast(lane.caches.total), @intCast(prompt.len), @intCast(at), @intCast(fill.count), h.prefix.restored, @intCast(held.len) });
            try h.msg.appendSlice(h.gpa, held);
            try h.msg.appendSlice(h.gpa, fill.stops[0..fill.count]);
            try h.msg.appendSlice(h.gpa, prompt);
            try h.send(h.msg.items);
        }
        return lane;
    }

    /// One chunk of the prompt pass, ending on the next cut or the prompt's end; the last draws the first token.
    fn prefillStepFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {
        const h = of(ptr);
        if (s.isCancelled()) return error.Cancelled;
        const lane = if (h.lanes.get(s)) |l| (if (l.fill != null) l else try h.begin(s)) else try h.begin(s);
        const prompt = s.prompt();
        const f = &lane.fill.?;
        const limit: usize = if (f.next < f.count) f.stops[f.next] else prompt.len;
        const to = prefix.chunkEnd(f.at, limit);
        if (h.link != null) try h.send(&.{ @backingInt(worker.Op.fill), lane.id, @intCast(to) });
        if (to == prompt.len) {
            _ = h.take(try h.e.prefill(&lane.caches, prompt, f.at, lane.hidden, .{ .sampling = sampling(s), .position = prompt.len }, null));
            lane.fill = null;
            return true;
        }
        try h.e.advance(&lane.caches, prompt, f.at, to, null);
        f.at = to;
        if (f.next < f.count and to == f.stops[f.next]) {
            var owner: hip_prefix.Owner = .{ .caches = &lane.caches, .id = lane.id };
            h.prefix.keep(&owner, prompt, to) catch |err| std.log.warn("prefix not kept: {s}", .{@errorName(err)});
            f.next += 1;
        }
        return false;
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const h = of(ptr);
        if (position != s.prompt_len) {
            std.log.err("first draw at {d}, the prompt has {d} tokens", .{ position, s.prompt_len });
            return error.PositionMismatch;
        }
        return h.next - 1;
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        _ = ptr;
        _ = s;
        _ = feed;
        _ = position;
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const h = of(ptr);
        if (handle >= h.next or h.next - handle > ring) return error.NoSuchToken;
        return h.drawn[handle % ring];
    }

    /// Each stream's window from its kept length: the pending token, then the host's drafts; one forward for all.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const h = of(ptr);
        const began = costs.nowNs();
        defer h.timed(0, began);
        if (windows.len > h.order.len) return error.WindowTooWide;
        // windows the round loop did not trim keep every row before this round's plan replaces theirs
        for (windows) |w| try h.settle(h.lanes.get(w.stream) orelse return error.UnknownStream);
        try h.flush();
        for (windows) |w| {
            const lane = h.lanes.get(w.stream).?;
            try h.ensure(lane, lane.len + w.rows());
        }
        var rows: [64]Engine.Rows = undefined;
        var tokens: [64][16]u32 = undefined;
        if (windows.len > rows.len) return error.WindowTooWide;
        for (windows, 0..) |w, i| {
            if (w.parents != null) return error.TreesNotBuilt;
            const lane = h.lanes.get(w.stream) orelse return error.UnknownStream;
            if (w.held > lane.held_n or (w.held > 0 and w.tokens.len > 0)) return error.NoHeldDrafts;
            const n = w.rows();
            if (n > tokens[i].len) return error.WindowTooWide;
            for (w.positions, 0..) |p, r| if (p != lane.len + 1 + r) {
                std.log.err("row {d} keyed at {d}, the stream holds {d} slots", .{ r, p, lane.len });
                return error.PositionMismatch;
            };
            tokens[i][0] = w.pending;
            if (w.held > 0) @memcpy(tokens[i][1..n], lane.held[0..w.held]) else @memcpy(tokens[i][1..n], w.tokens);
            rows[i] = .{ .caches = &lane.caches, .pos = lane.len, .tokens = tokens[i][0..n] };
            h.order[i] = w.stream;
        }
        var reqs: [256]draw.Request = undefined;
        var drawn: [256]u32 = undefined;
        var total: usize = 0;
        for (windows) |w| {
            if (total + w.positions.len > reqs.len) return error.WindowTooWide;
            for (w.positions) |p| {
                reqs[total] = .{ .sampling = sampling(w.stream), .position = p };
                total += 1;
            }
        }
        // the graph choice is rank 0's, and goes with the round
        const pick = try h.e.choose(rows[0..windows.len], null);
        if (h.link != null) {
            h.msg.clearRetainingCapacity();
            try h.msg.appendSlice(h.gpa, &.{ @backingInt(worker.Op.verify), @backingInt(pick), @intCast(windows.len) });
            for (windows, rows[0..windows.len]) |w, r| {
                try h.msg.appendSlice(h.gpa, &.{ h.lanes.get(w.stream).?.id, @intCast(r.tokens.len) });
                try h.msg.appendSlice(h.gpa, r.tokens);
            }
            try h.send(h.msg.items);
        }
        _ = try h.e.verify(rows[0..windows.len], reqs[0..total], &drawn);
        var at: usize = 0;
        for (windows, out, 0..) |w, *o, i| {
            @memcpy(o.sampled, drawn[at..][0..o.sampled.len]);
            const lane = h.lanes.get(w.stream).?;
            @memcpy(o.drafts, if (w.held > 0) lane.held[0..w.held] else w.tokens);
            lane.held_n = 0;
            lane.pending = .{ .window = i, .rows = w.rows() };
            at += w.rows();
        }
    }

    /// Keep each stream's accepted prefix: lengths, and the linear states of its last kept row.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const h = of(ptr);
        const began = costs.nowNs();
        defer h.timed(1, began);
        for (windows, paths) |w, path| {
            const lane = h.lanes.get(w.stream) orelse return error.UnknownStream;
            for (path, 0..) |r, j| if (r != j) return error.TreesNotBuilt;
            // the draft request of the same round may have kept these rows already
            if (lane.pending == null) continue;
            try h.mark(lane, path.len);
        }
        try h.flush();
    }

    /// The head drafts `depth` from each stream's last kept row and its pending token, one head forward a step for all.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const h = of(ptr);
        const began = costs.nowNs();
        defer h.timed(2, began);
        const head = h.head orelse return error.NoDraftHead;
        const m = h.e.model();
        var jobs: [max_jobs]mtp.Job = undefined;
        var chained: [max_jobs]*Lane = undefined;
        var n: usize = 0;
        // a shared round asks for drafts before its keep: the request names the kept rows
        for (requests) |r| {
            const lane = h.lanes.get(r.stream) orelse return error.UnknownStream;
            if (lane.pending != null) {
                if (r.rows) |kept| try h.mark(lane, kept.len) else try h.settle(lane);
            }
        }
        try h.flush();
        for (requests) |r| {
            const lane = h.lanes.get(r.stream) orelse return error.UnknownStream;
            if (r.position != lane.len + 1) {
                std.log.err("draft for {s} at {d}, the stream holds {d} slots (rows {any}, depth {d})", .{ r.stream.id, r.position, lane.len, r.rows, r.depth });
                return error.PositionMismatch;
            }
            const token: u32 = if (r.rows != null) r.follow[r.follow.len - 1] else switch (r.first orelse return error.NoFirstToken) {
                .handle => |at| h.drawn[at % ring],
                .value => |v| v,
            };
            lane.held_n = 0;
            if (r.depth == 0) continue;
            if (n == max_jobs) return error.WindowTooWide;
            // a chain ends after a draft the head gives under the confidence; the first drafts run whole
            const stop_under: f64 = if (r.rows != null) h.e.o.policy.mtp.confidence else 0.0;
            jobs[n] = .{ .hidden = lane.hidden.base(), .token = token, .position = lane.len, .depth = r.depth, .sampling = sampling(r.stream), .stop_under = stop_under, .out = &lane.held };
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

    /// The drafts the head holds for the stream after its last chain: fewer than asked when a draft cut it.
    fn heldFn(ptr: *anyopaque, s: *lanes.Stream, _: std.mem.Allocator) anyerror!?lanes.stream.Held {
        const lane = of(ptr).lanes.get(s) orelse return error.UnknownStream;
        return .{ .count = @intCast(lane.held_n) };
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const h = of(ptr);
        const kv = h.lanes.fetchRemove(s) orelse return;
        if (h.link != null) h.send(&.{ @backingInt(worker.Op.release), kv.value.id }) catch {};
        _ = h.ids.remove(s);
        h.e.stream.synchronize() catch {};
        h.free(kv.value);
    }
};
