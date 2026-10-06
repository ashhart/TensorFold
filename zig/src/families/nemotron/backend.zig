//! The lane core's Metal backend for Nemotron: streams' rows in one forward a round, drawn tokens in GPU slots.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const Fence = @import("../../core/fence.zig").Fence;
const fwd = @import("forward.zig");
const st = @import("state.zig");
const mtp = @import("mtp.zig");
const timing = @import("timing.zig");
const tree = @import("tree.zig");
const head_tree = @import("head_tree.zig");
const prefill = @import("prefill.zig");
const head_block = @import("head_block.zig");
const Model = @import("model.zig").Model;
const nemotron_config = @import("config.zig");

const be = lanes.backend;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const ring = 1 << 16;

/// A lone stream's window rows at most, and a shared round's.
pub const max_window = 16;

/// A stream's tree window's rows at most (its lanes).
pub const max_lanes = 64;

/// Prompt chunks this short take the decode kernels, longer ones the prefill kernels (Python's fused_rows).
pub const fused_rows = 16;

pub const Kind = timing.Kind;

const Flight = struct { end: u64, cb: mtl.CommandBuffer, kind: Kind };

/// A window's host rows in the token ring: the pending token at `stage`, then `n` tokens after its `held` drafts.
const Fill = struct { stage: u64, held: usize, n: usize };

pub const Options = struct {
    capacity: usize, // a stream's prompt + reply + a window, in rows
    chunk: usize = 2048, // prompt rows a forward (the lane engine's PrefillPlan step)
    drafts: bool = true, // load the MTP head
    streams: usize = 8, // streams at once
    batch_rows: usize = 32, // a shared round's rows at most
    block: ?head_block.Weights = null, // the head drafts greedy chains in one pass with these placeholder rows
};

pub const Metal = struct {
    gpa: std.mem.Allocator,
    m: *Model,
    o: Options,
    pool: st.Pool,
    scratch: st.Scratch,
    wide: ?prefill.Prefill = null, // prompt chunks over fused_rows
    head: ?mtp.Head = null,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *st.Cache) = .empty,
    tokens: mtl.Buffer, // u32 [ring]: drawn tokens and host-fed ones
    prompt: mtl.Buffer, // u32 [capacity]
    fence: Fence,
    next: u64 = 0,
    landed: u64 = 0, // slots below this are written and visible to the host
    flights: std.ArrayList(Flight) = .empty,
    timing: timing.Timing = .{},
    costs: timing.Costs = .{},
    geometry: usize = 0, // routed experts' kernel geometry (bit-identical)
    members: usize = 2, // routed experts' member rows a pass (bit-identical; one at a time for one-row forwards)
    ranks_always: bool = false, // the head keeps every chained level's best tokens, asked or not (shared rounds' grafts)
    concurrent: bool = true, // concurrent encoders with barriers on data flow (false: serial encoders)
    gpu_ms: f64 = 0, // GPU ms of every command buffer landed so far
    last_ms: f64 = 0, // and of the last one

    pub fn init(gpa: std.mem.Allocator, m: *Model, o: Options) !*Metal {
        const b = try gpa.create(Metal);
        errdefer gpa.destroy(b);
        const rows = @max(o.batch_rows, max_lanes);
        b.* = .{
            .gpa = gpa,
            .m = m,
            .o = o,
            .pool = try st.Pool.init(gpa, m.device, m.config, rows + o.streams + 1, o.streams + 4),
            .scratch = try st.Scratch.init(m.device, m.config, rows, o.capacity),
            .tokens = try m.device.buffer(ring * 4, opts),
            .prompt = try m.device.buffer(@max(o.capacity, 16) * 4, opts),
            .fence = try Fence.init(m.device),
        };
        if (o.chunk > fused_rows) b.wide = try prefill.Prefill.init(m, @min(o.chunk, o.capacity));
        if (o.drafts and m.weights.mtp != null) b.head = try mtp.Head.init(m.device, m.config, &m.kernels, &m.weights, &b.pool, max_lanes, o.capacity);
        if (b.head) |*h| if (b.o.block) |*bw| {
            h.block = bw;
        };
        return b;
    }

    pub fn deinit(self: *Metal) void {
        self.drain() catch {};
        self.flights.deinit(self.gpa);
        var it = self.caches.valueIterator();
        while (it.next()) |c| {
            c.*.deinit(&self.pool);
            self.gpa.destroy(c.*);
        }
        self.caches.deinit(self.gpa);
        if (self.head) |*h| h.deinit();
        if (self.wide) |*w| w.deinit();
        self.fence.deinit();
        self.prompt.deinit();
        self.tokens.deinit();
        self.scratch.deinit();
        self.pool.deinit();
        self.gpa.destroy(self);
    }

    pub fn backend(self: *Metal) be.Backend {
        return .{ .ptr = self, .vtable = &.{
            .prefill = prefillFn,
            .first = firstFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .release = releaseFn,
            .alternatives = alternativesFn,
        } };
    }

    /// The facts the round loop reads at setup (Python's NemotronH attributes), with this engine's own timings.
    pub fn facts(self: *const Metal) lanes.Model {
        const drafting = self.head != null;
        return .{
            .exact_width = if (drafting) max_window else 1,
            .gpu_tokens = true,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .draft_prior = &nemotron_config.draft_prior,
            .drafts = 4,
            .window_costs = self.costs.window[0..@min(self.costs.windows, max_window)],
            .mtp_step_ms = self.costs.head_ms,
            .hidden_rows = true,
            .batch_rows = @intCast(self.o.batch_rows),
            .max_streams = @intCast(self.o.streams),
            .shared_costs = self.costs.shared[0..self.costs.shareds],
            .head_trees = drafting,
            .lane_costs = self.costs.window[0..self.costs.windows],
        };
    }

    pub fn forward(self: *Metal) fwd.Forward {
        return .{ .c = self.m.config, .k = &self.m.kernels, .w = &self.m.weights, .s = &self.scratch, .pool = &self.pool, .geometry = self.geometry, .members = self.members };
    }

    pub fn cacheOf(self: *Metal, s: *lanes.Stream) !*st.Cache {
        return self.caches.get(s) orelse error.UnknownStream;
    }

    /// One command buffer after the fence; `end`: the token slots below it are written when it completes.
    pub fn submit(self: *Metal, kind: Kind, end: u64, body: anytype) !void {
        const t0 = mtl.clock.seconds();
        defer self.timing.encode_ms[@backingInt(kind)] += (mtl.clock.seconds() - t0) * 1e3;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = self.m.queue.commandBufferUnretained();
        const enc = cb.compute(if (self.concurrent) .concurrent else .serial);
        self.fence.wait(enc);
        var e = fwd.Enc{ .e = enc, .concurrent = self.concurrent };
        try body.encode(self, &e);
        self.fence.update(enc);
        enc.end();
        cb.commit();
        self.timing.dispatches += e.dispatches;
        try self.flights.append(self.gpa, .{ .end = end, .cb = cb.retain(), .kind = kind });
    }

    /// Wait for the oldest command buffer in flight and record its GPU time.
    pub fn land(self: *Metal) !void {
        const f = self.flights.orderedRemove(0);
        defer f.cb.release();
        f.cb.wait();
        if (f.cb.failure()) |text| {
            std.log.err("command buffer failed: {s}", .{text});
            return error.GpuFailed;
        }
        self.landed = @max(self.landed, f.end);
        self.last_ms = f.cb.gpuSeconds() * 1e3;
        self.gpu_ms += self.last_ms;
        self.timing.ms[@backingInt(f.kind)] += self.last_ms;
        self.timing.count[@backingInt(f.kind)] += 1;
    }

    pub fn drain(self: *Metal) !void {
        while (self.flights.items.len > 0) try self.land();
    }

    fn wait(self: *Metal, handle: u64) !void {
        while (handle >= self.landed) {
            if (self.flights.items.len == 0) return error.NoSuchToken;
            try self.land();
        }
    }

    pub fn slot(h: u64) usize {
        return @intCast(h % ring * 4);
    }

    /// `n` consecutive token slots (never wrapping the ring).
    pub fn take(self: *Metal, n: usize) u64 {
        if (self.next % ring + n > ring) self.next += ring - self.next % ring;
        const h = self.next;
        self.next += n;
        return h;
    }

    /// Commit a verify whose rows all stayed (the round loop calls keep() only when it drops rows).
    fn settle(self: *Metal, c: *st.Cache) void {
        if (c.pending) |p| c.commit(&self.pool, p.rows);
    }

    /// A stream's draw at `position`: argmax, or its keyed sampler (rows after it at the next positions).
    fn drawOf(s: *const lanes.Stream, position: u64) fwd.Draw {
        return if (s.sampling) |t| .{ .sampled = .{ .sampling = t, .position = position } } else .greedy;
    }

    /// The fresh state slot a segment stores to: always after all its rows, after its replayed rows if it has any.
    pub fn slotFor(self: *Metal, c: *const st.Cache, store: fwd.Store) !i32 {
        return if (store == .full or c.replay > 0) @intCast(try self.pool.take()) else -1;
    }

    /// Hold a forward's Mamba outcome for the round loop's keep (its rows' inputs went to the other parity).
    pub fn hold(c: *st.Cache, rows: usize, store: fwd.Store, state: i32) void {
        c.pending = .{ .rows = rows, .slot = if (state >= 0) @intCast(state) else null, .parity = 1 - c.parity, .full = store == .full };
    }

    // -- the vtable ---------------------------------------------------------------------------------------------

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self: *Metal = @ptrCast(@alignCast(ptr));
        const ids = s.prompt();
        if (ids.len == 0 or ids.len + s.max_new + max_window > self.o.capacity) return error.PromptTooLong;
        if (s.isCancelled()) return error.Cancelled;
        const rows_needed = @min(self.o.capacity, ids.len + s.max_new + max_lanes + 1); // the reply and a widest window past it
        try self.drain();
        if (self.caches.get(s)) |old| {
            old.deinit(&self.pool);
            self.gpa.destroy(old);
            _ = self.caches.remove(s);
        }
        const c = try self.gpa.create(st.Cache);
        c.* = try st.Cache.init(self.m.device, self.m.config, rows_needed, self.head != null);
        c.rid = try self.pool.takeRid();
        try self.caches.put(self.gpa, s, c);
        @memcpy(self.prompt.slice(u32, ids.len), ids);
        const Chunks = struct {
            n: usize,
            c: *st.Cache,
            at: usize,
            rows: usize,
            fn encode(j: @This(), b: *Metal, e: *fwd.Enc) !void {
                const d = b.m.config.hidden * 2;
                const rows = j.rows;
                const fresh = try b.pool.take();
                var out = b.scratch.x; // the chunk's final-normed rows
                if (rows <= fused_rows) {
                    const segs = [_]fwd.Seg{.{ .rows = rows, .cache = j.c, .store = .full, .slot = @intCast(fresh) }};
                    b.forward().body(e, &segs, b.prompt, j.at * 4);
                } else {
                    const w = &b.wide.?;
                    w.chunk(e, b.forward(), j.c, b.prompt, j.at * 4, rows, fresh);
                    out = w.s.x;
                }
                j.c.advance(&b.pool, rows, fresh);
                // the head's cache takes each row whose next token the prompt holds, as many rows a call as a window
                const absorbed = @min(j.at + rows, j.n - 1) - j.at;
                if (b.head) |*h| {
                    var r: usize = 0;
                    while (r < absorbed) : (r += max_window) h.absorb(e, j.c, @min(max_window, absorbed - r), out, r * d, b.prompt, (j.at + 1 + r) * 4);
                }
                j.c.start = 0;
                j.c.rows = rows;
                if (rows > fused_rows) {
                    // the last row where the first draw and the head read it
                    e.pipe(b.m.kernels.get("tf_copy_rows"));
                    e.buf(out, (rows - 1) * d, 0);
                    e.buf(b.scratch.x, 0, 1);
                    e.run(.{ d / 2, 1, 1 }, .{ 256, 1, 1 });
                    j.c.rows = 1;
                }
            }
        };
        // a command buffer a chunk, each committed before the one ahead of it is waited on, so the GPU keeps a chunk queued
        var at: usize = 0;
        var k: usize = 0;
        while (at < ids.len) {
            while (k < s.chunks.len and s.chunks[k] <= at) k += 1;
            const end = if (k < s.chunks.len) @min(s.chunks[k], ids.len) else ids.len;
            const rows = @min(self.o.chunk, end - at);
            if (s.isCancelled()) return error.Cancelled; // release() waits for the chunk still in flight
            try self.submit(.prefill, self.next, Chunks{ .n = ids.len, .c = c, .at = at, .rows = rows });
            while (self.flights.items.len > 1) try self.land();
            at += rows;
        }
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self: *Metal = @ptrCast(@alignCast(ptr));
        const c = try self.cacheOf(s);
        const h = self.take(1);
        const Head = struct {
            h: u64,
            c: *st.Cache,
            how: fwd.Draw,
            fn encode(j: @This(), b: *Metal, e: *fwd.Enc) !void {
                const segs = [_]fwd.Seg{.{ .rows = j.c.rows, .cache = j.c, .draw = j.how }};
                b.forward().draw(e, &segs, .last, b.tokens, slot(j.h));
            }
        };
        try self.submit(.prefill, h + 1, Head{ .h = h, .c = c, .how = drawOf(s, position) });
        return h;
    }

    /// A host token into a fresh ring slot (no command in flight reads a fresh slot).
    fn feedSlot(self: *Metal, feed: be.Feed) u64 {
        return switch (feed) {
            .handle => |h| h,
            .value => |v| blk: {
                const h = self.take(1);
                self.tokens.slice(u32, ring)[h % ring] = v;
                break :blk h;
            },
        };
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const self: *Metal = @ptrCast(@alignCast(ptr));
        const c = try self.cacheOf(s);
        self.settle(c);
        const in = self.feedSlot(feed);
        const out = self.take(1);
        if (c.len + 1 > c.kv[0].capacity) return error.ContextFull;
        const state = try self.slotFor(c, .full);
        const Step = struct {
            in: u64,
            out: u64,
            c: *st.Cache,
            state: i32,
            how: fwd.Draw,
            fn encode(j: @This(), b: *Metal, e: *fwd.Enc) !void {
                const segs = [_]fwd.Seg{.{ .rows = 1, .cache = j.c, .store = .full, .slot = j.state, .draw = j.how }};
                const f = b.forward();
                compactNow(f, e, j.c);
                f.body(e, &segs, b.tokens, slot(j.in));
                f.draw(e, &segs, .all, b.tokens, slot(j.out));
            }
        };
        try self.submit(.step, out + 1, Step{ .in = in, .out = out, .c = c, .state = state, .how = drawOf(s, position) });
        hold(c, 1, .full, state);
        c.commit(&self.pool, 1);
        c.start = 0;
        c.rows = 1;
        return out;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const self: *Metal = @ptrCast(@alignCast(ptr));
        try self.wait(handle);
        return self.tokens.slice(u32, ring)[handle % ring];
    }

    /// Every stream's window (pending token, then drafts) in one forward, each row drawn at its own position.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self: *Metal = @ptrCast(@alignCast(ptr));
        var segs: [st.max_rows]fwd.Seg = undefined;
        var fills: [st.max_rows]Fill = undefined;
        var total: usize = 0;
        for (windows, 0..) |w, i| {
            const c = try self.cacheOf(w.stream);
            self.settle(c);
            const rows = w.rows();
            if (rows > max_lanes or c.len + rows > c.kv[0].capacity) return error.WindowTooWide;
            if (w.parents == null) {
                for (w.positions, 0..) |p, r| if (p != w.positions[0] + r) return error.PositionsNotAChain;
            } else if (windows.len > 1) return error.SharedTreesNotBuilt else if (w.stream.sampling != null) return error.SampledTreesNotBuilt;
            // the host's rows go to fresh ring slots and the verify copies them in: a draft still in flight writes this window
            const stage = self.take(1 + w.tokens.len);
            const staged = self.tokens.slice(u32, ring);
            staged[stage % ring] = w.pending;
            for (w.tokens, 0..) |t, r| staged[(stage + 1 + r) % ring] = t;
            fills[i] = .{ .stage = stage, .held = w.held, .n = w.tokens.len };
            segs[i] = .{ .rows = rows, .cache = c, .store = .lag, .slot = try self.slotFor(c, .lag), .parents = w.parents, .draw = drawOf(w.stream, w.positions[0]) };
            c.start = total;
            c.rows = rows;
            total += rows;
        }
        if (total > self.scratch.rows) return error.WindowTooWide;
        var replayed: usize = 0;
        for (segs[0..windows.len]) |g| replayed += g.cache.replay;
        if (total + replayed > 2 * self.scratch.rows) return error.WindowTooWide;
        const first = self.take(total);
        const Win = struct {
            segs: []const fwd.Seg,
            fills: []const Fill,
            first: u64,
            fn encode(j: @This(), b: *Metal, e: *fwd.Enc) !void {
                const f = b.forward();
                for (j.segs, j.fills) |g, x| {
                    const win = g.cache.windows[g.cache.wcur];
                    f.copyIds(e, b.tokens, slot(x.stage), win, 0, 1);
                    if (x.n > 0) f.copyIds(e, b.tokens, slot(x.stage + 1), win, (1 + x.held) * 4, x.n);
                }
                for (j.segs) |g| compactNow(f, e, g.cache);
                if (j.segs.len == 1) {
                    const c = j.segs[0].cache;
                    f.body(e, j.segs, c.windows[c.wcur], 0);
                } else {
                    for (j.segs) |g| f.copyIds(e, g.cache.windows[g.cache.wcur], 0, b.scratch.ids, g.cache.start * 4, g.rows);
                    f.body(e, j.segs, b.scratch.ids, 0);
                }
                f.draw(e, j.segs, .all, b.tokens, slot(j.first));
            }
        };
        try self.submit(.verify, first + total, Win{ .segs = segs[0..windows.len], .fills = fills[0..windows.len], .first = first });
        try self.wait(first + total - 1);
        const drawn = self.tokens.slice(u32, ring);
        for (windows, out, segs[0..windows.len]) |w, *o, g| {
            const c = g.cache;
            const ids = c.windows[c.wcur].slice(u32, st.max_rows);
            for (o.sampled, 0..) |*t, r| t.* = drawn[(first + c.start + r) % ring];
            for (o.drafts, 0..) |*t, r| t.* = ids[1 + r];
            hold(c, w.rows(), .lag, g.slot);
            c.wcur = 1 - c.wcur;
        }
    }

    /// Keep each stream's accepted rows: a prefix, or a tree's path (its cache rows move with the next command buffer).
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const self: *Metal = @ptrCast(@alignCast(ptr));
        for (windows, paths) |w, path| {
            const c = try self.cacheOf(w.stream);
            if (c.pending == null or path.len == 0 or path[0] != 0) return error.NothingToKeep;
            var prefix = true;
            for (path, 0..) |r, i| prefix = prefix and r == i;
            if (!prefix) {
                c.compact = .{ .len = c.len, .n = path.len, .path = undefined };
                @memcpy(c.compact.?.path[0..path.len], path);
            }
            c.commit(&self.pool, path.len);
            for (path, 0..) |r, i| c.replay_map[i] = @intCast(r);
        }
    }

    /// Move a kept tree path's key and value rows into place before anything reads the stream's caches.
    fn compactNow(f: fwd.Forward, e: *fwd.Enc, c: *st.Cache) void {
        const k = c.compact orelse return;
        for (0..c.attentions) |a| {
            if (a > 0) e.alongside();
            tree.compact(f, e, c.kv[a], k.len, k.path[0..k.n]);
        }
        c.compact = null;
    }

    /// Absorb each stream's kept rows into its head cache, then hold `depth` drafts in its next window.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self: *Metal = @ptrCast(@alignCast(ptr));
        if (self.head == null) return error.NoDraftHead;
        const Draft = struct {
            c: *st.Cache,
            kept: usize, // kept rows of the stream's last window (0: the prompt's last row)
            depth: usize,
            how: fwd.Draw,
            token: u64, // the ring slot of the first token, after the prompt's last row
            path: []const u32 = &.{}, // the kept rows (a tree's path: not a prefix)
            lanes: ?*const lanes.shape.Shape = null, // a tree of drafts to hold
            ranks: bool = false, // keep each chained level's best tokens
        };
        var jobs: [st.max_rows]Draft = undefined;
        for (requests, 0..) |r, i| {
            const c = try self.cacheOf(r.stream);
            if (r.lanes == null and r.depth + 1 > max_window) return error.WindowTooWide;
            if (r.lanes) |l| if (l.parents.len + 1 > max_lanes or r.stream.sampling != null) return error.WindowTooWide;
            jobs[i] = .{ .c = c, .kept = 0, .depth = r.depth, .how = drawOf(r.stream, r.position), .token = 0, .lanes = r.lanes, .ranks = r.ranks or self.ranks_always };
            if (r.rows) |rows| {
                jobs[i].kept = rows.len;
                jobs[i].path = rows;
                @memcpy(c.follow.slice(u32, rows.len), r.follow[0..rows.len]);
            } else {
                jobs[i].token = self.feedSlot(r.first orelse return error.NoFirstToken);
            }
        }
        const All = struct {
            jobs: []const Draft,
            fn encode(a: @This(), b: *Metal, e: *fwd.Enc) !void {
                const h = &b.head.?;
                const d = b.m.config.hidden * 2;
                for (a.jobs) |j| {
                    const c = j.c;
                    const window = c.windows[c.wcur];
                    compactNow(b.forward(), e, c);
                    // a tree's kept rows, gathered into path order
                    var prefix = true;
                    for (j.path, 0..) |r, i| prefix = prefix and r == i;
                    var hidden = b.scratch.x;
                    var start = c.start;
                    if (!prefix) {
                        tree.gather(b.forward(), e, b.scratch.x, c.start, j.path, b.scratch.kept);
                        hidden = b.scratch.kept;
                        start = 0;
                    }
                    if (j.kept > 1) h.absorb(e, c, j.kept - 1, hidden, start * d, c.follow, 0);
                    const last = start + (if (j.kept > 0) j.kept else c.rows) - 1;
                    const tok: mtl.Buffer = if (j.kept > 0) c.follow else b.tokens;
                    const tok_off = if (j.kept > 0) (j.kept - 1) * 4 else slot(j.token);
                    if (j.lanes) |l| {
                        head_tree.draft(h, e, c, l.*, hidden, last * d, tok, tok_off, window, 4);
                    } else {
                        h.topk = j.ranks;
                        h.chain(e, c, j.depth, hidden, last * d, tok, tok_off, j.how, window, 4);
                    }
                }
            }
        };
        try self.submit(.draft, self.next, All{ .jobs = jobs[0..requests.len] });
    }

    /// The head's best 4 tokens and probabilities at each level of the stream's held drafts (after they land).
    fn alternativesFn(ptr: *anyopaque, s: *lanes.Stream, out: []be.Alternative) anyerror!usize {
        const self: *Metal = @ptrCast(@alignCast(ptr));
        const c = try self.cacheOf(s);
        try self.drain();
        const n = @min(c.levels, out.len);
        const ids = c.topk.slice(u32, st.max_levels * 4);
        const probs = c.topk.slice(f32, 2 * st.max_levels * 4)[st.max_levels * 4 ..];
        for (out[0..n], c.trunk[0..n]) |*a, l| for (0..4) |r| {
            a.tokens[r] = ids[@as(usize, l) * 4 + r];
            a.probs[r] = probs[@as(usize, l) * 4 + r];
        };
        return n;
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self: *Metal = @ptrCast(@alignCast(ptr));
        self.drain() catch {};
        if (self.caches.fetchRemove(s)) |kv| {
            kv.value.deinit(&self.pool);
            self.gpa.destroy(kv.value);
        }
    }
};
