//! The lane core's backend for Qwen27: a stream a runner slot, shared rounds across slots, DFlash2 trees for slot 0's stream.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const be = lanes.backend;
pub const Draft = @import("qwen27_draft.zig").Draft;
const Round = q.round_plan.Round;
const Tree = q.dflash.selector.Tree;
const Ring = q.tap_ring.Ring;
const contract = tf.tree_round;

/// A window's rows at most: the pending row and a DFlash2 tree.
pub const max_rows = 16;
/// A shared round's rows at most: the frame's.
pub const batch_rows = 128;
const ring = 64;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// Called before a finished stream leaves its slot, its state still standing there (the reply's prompt-cache keep).
pub const Release = struct { ctx: *anyopaque, kept: *const fn (ctx: *anyopaque, s: *lanes.Stream, slot: u32) void };

/// A slot's memory taken when a stream is first bound to it: `room` says whether `bytes` more fit, `added` gets the
/// new buffers (the server's residency set). error.LaneMemory when they do not fit: the request waits for a slot.
pub const Grow = struct {
    ctx: *anyopaque,
    room: *const fn (ctx: *anyopaque, bytes: usize) bool,
    added: *const fn (ctx: *anyopaque, buffers: []const mtl.Buffer) void,
};

/// A slot's own copy of the drafter's taps ring (slots past 0, when a drafter is attached).
const SlotRing = struct {
    buf: mtl.Buffer,
    end: u64 = 0,
};

pub const Metal = struct {
    gpa: std.mem.Allocator,
    runner: *q.decode_round.Runner,
    draft: ?*Draft,
    chunk: usize = 128, // prompt rows a pass
    nodes: u32 = max_rows - 1, // a tree's drafts at most
    owners: [8]?*lanes.Stream = @splat(null), // the stream each runner slot holds
    rings: [8]?SlotRing = @splat(null), // slots past 0: their taps, so a kept state carries them
    release_hook: ?Release = null,
    grow: ?Grow = null,
    drawn: [ring]u32 = undefined, // drawn tokens by handle
    next: u64 = 0,
    round: ?Round = null, // the last verify, until its rows are kept
    windows: [8]u32 = undefined, // the last verify's slots, in window order
    widths: [8]u32 = undefined, // and their rows
    count: usize = 0, // the last verify's windows
    early: usize = 0, // slot 0's kept rows its drafter absorbed before a shared round's keep
    ids: [batch_rows]u32 = undefined,
    parents: [batch_rows]i32 = undefined,
    rows: usize = 0,
    tree: ?Tree = null, // the drafts held for slot 0's next round
    ops: tf.tree_round_gpu.Ops,
    picks: mtl.Buffer,
    kept: mtl.Buffer, // a keep's gathered rows, by round row
    gathered_taps: ?mtl.Buffer = null, // and their taps, in path order (with a drafter)
    costs: [max_rows]lanes.config.Cost = undefined, // a window's ms by rows, timed by measure()
    timed: usize = 0,
    from_tree: usize = 0, // the last window's drafts that were the held tree's first nodes
    sampled: [max_rows]u32 = undefined, // the last lone window's draws
    landing: [16][2]f64 = @splat(.{ 0, 0 }), // by a node's own log-probability bucket: offered with its parent kept, landed

    pub fn init(gpa: std.mem.Allocator, runner: *q.decode_round.Runner, draft: ?*Draft) !*Metal {
        if (runner.slots == 0 or runner.slots > 8 or runner.model.frame.capacity < batch_rows) return error.TargetBinding;
        const device = runner.model.device;
        const ops = try tf.tree_round_gpu.Ops.init(device);
        errdefer ops.deinit();
        const picks = try device.buffer(batch_rows * @sizeOf(contract.Pick), opts);
        errdefer picks.deinit();
        const kept = try device.buffer(batch_rows * 4, opts);
        errdefer kept.deinit();
        const b = try gpa.create(Metal);
        errdefer gpa.destroy(b);
        b.* = .{ .gpa = gpa, .runner = runner, .draft = draft, .ops = ops, .picks = picks, .kept = kept };
        errdefer b.freeRings();
        if (draft) |d| {
            const g = d.model.ring();
            b.gathered_taps = try device.buffer(batch_rows * g.stride, opts);
            for (b.rings[1..runner.ready]) |*r| r.* = .{ .buf = try device.buffer(@as(usize, g.window) * g.stride, opts) };
        }
        return b;
    }

    /// A slot's memory past the runner's ready ones: its K/V rows and taps ring, taken the first time it is used.
    fn ready(b: *Metal, slot: u32) !void {
        const kv = slot >= b.runner.ready;
        const ring_missing = b.draft != null and slot > 0 and b.rings[slot] == null;
        if (!kv and !ring_missing) return;
        const ring_bytes: usize = if (b.draft) |d| @as(usize, d.model.ring().window) * d.model.ring().stride else 0;
        const need = (if (kv) b.runner.slotBytes() else 0) + (if (ring_missing) ring_bytes else 0);
        if (b.grow) |g| if (!g.room(g.ctx, need)) return error.LaneMemory;
        var added: std.ArrayList(mtl.Buffer) = .empty;
        defer added.deinit(b.gpa);
        try added.appendSlice(b.gpa, try b.runner.ensure(slot));
        if (ring_missing) {
            const buf = try b.runner.model.device.buffer(ring_bytes, opts);
            b.rings[slot] = .{ .buf = buf };
            try added.append(b.gpa, buf);
        }
        if (b.grow) |g| g.added(g.ctx, added.items);
    }

    fn freeRings(b: *Metal) void {
        for (&b.rings) |*r| if (r.*) |x| {
            x.buf.deinit();
            r.* = null;
        };
        if (b.gathered_taps) |t| t.deinit();
        b.gathered_taps = null;
    }

    pub fn deinit(b: *Metal) void {
        b.dropRound();
        b.dropTree();
        b.freeRings();
        b.kept.deinit();
        b.picks.deinit();
        b.ops.deinit();
        b.gpa.destroy(b);
    }

    pub fn backend(b: *Metal) be.Backend {
        return .{ .ptr = b, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .probabilities = probabilitiesFn, .tree = treeFn, .release = releaseFn } };
    }

    /// What the round loop reads at setup: slot 0's stream drafts trees; every stream verifies copies.
    pub fn facts(b: *const Metal) lanes.Model {
        const drafting = b.draft != null;
        const shared = b.runner.slots > 1;
        return .{ .exact_width = if (drafting or shared) max_rows else 1, .mtp = drafting, .speculate = drafting, .speculate_early = false, .drafts = b.nodes, .draft_probabilities = true, .window_costs = b.costs[0..b.timed], .hidden_rows = shared, .streams_exact = true, .batch_rows = if (shared) batch_rows else max_rows, .max_streams = b.runner.slots };
    }

    /// The slot holding `s`.
    pub fn slotOf(b: *const Metal, s: *const lanes.Stream) ?u32 {
        for (b.owners[0..b.runner.slots], 0..) |o, i| if (o == s) return @intCast(i);
        return null;
    }

    /// The taps ring of `slot`'s stream: the drafter's for slot 0, the slot's own past it; null without a drafter.
    pub fn ringOf(b: *Metal, slot: u32) ?Ring {
        const d = b.draft orelse return null;
        if (slot == 0) return d.model.ring();
        const g = d.model.ring();
        const r = &(b.rings[slot] orelse return null);
        return .{ .buf = r.buf, .end = &r.end, .window = g.window, .stride = g.stride };
    }

    /// Take the stream whose prompt another driver prefilled into slot 0 (the serial host).
    pub fn attach(b: *Metal, s: *lanes.Stream) !void {
        if (b.owners[0] != null) return error.StreamBusy;
        if (b.runner.offsets[0] != s.prompt_len) return error.CachePositionDiffers;
        b.dropRound();
        b.dropTree();
        b.owners[0] = s;
    }

    /// Each window width's ms (verify, draws, keep) on a scratch context in slot 0; resets it. Kept per build.
    pub fn measure(b: *Metal, io: std.Io) !void {
        const r = b.runner;
        if (b.owners[0] != null or r.active != null) return error.StreamBusy;
        const c = r.model.config;
        var shape: [128]u8 = undefined;
        const parts = [_][]const u8{ "qwen27-lanes", std.mem.span(r.model.device.name()), try std.fmt.bufPrint(&shape, "{d}/{d}/{d}/{d}/{d}/{d}", .{ c.layers, c.hidden, c.intermediate, c.vocab, max_rows, @intFromBool(b.draft != null) }) };
        const k = lanes.cost_cache.key(b.gpa, io, &parts) catch null;
        if (k) |key| if (lanes.cost_cache.load([max_rows]lanes.config.Cost, b.gpa, io, key)) |kept| {
            b.costs = kept;
            b.timed = max_rows;
            return;
        };
        if (b.draft) |d| try d.reset() else try r.reset(0);
        defer {
            if (b.draft) |d| d.reset() catch {} else r.reset(0) catch {};
        }
        var ids: [192]u32 = undefined;
        for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % r.model.config.vocab);
        if (b.draft) |d| try d.generation.prefill(&ids, 128) else try (q.session.Session{ .runner = r }).prefill(&ids, 128);
        var s = try lanes.Stream.init(b.gpa, .{ .id = "timing", .prompt = &ids, .max_new = 1 });
        defer s.deinit(b.gpa);
        const timer: WindowTimer = .{ .b = b, .s = &s };
        for (0..5) |_| _ = try b.timeWindow(&s, max_rows); // the widest window first, while the GPU's clocks ramp up
        var ms: [max_rows]f64 = undefined;
        for (&ms, 0..) |*m, i| m.* = try lanes.cost_rule.fastest(timer, i);
        try lanes.cost_rule.smooth(timer, &ms);
        const ref = lanes.cost_cache.referenceKey(&parts);
        if (lanes.cost_cache.load([max_rows]f64, b.gpa, io, ref)) |reference| if (lanes.cost_rule.drifted(&ms, &reference)) try lanes.cost_rule.again(timer, &ms);
        for (&b.costs, ms, 0..) |*cost, m, i| cost.* = .{ .width = @intCast(i + 1), .ms = m };
        b.timed = max_rows;
        if (k) |key| lanes.cost_cache.save([max_rows]lanes.config.Cost, b.gpa, io, key, b.costs);
        lanes.cost_cache.save([max_rows]f64, b.gpa, io, ref, ms);
    }

    /// Window entry `i` is `i + 1` rows, timed once.
    const WindowTimer = struct {
        b: *Metal,
        s: *lanes.Stream,
        pub fn time(t: WindowTimer, i: usize) !f64 {
            return t.b.timeWindow(t.s, i + 1);
        }
    };

    fn timeWindow(b: *Metal, s: *lanes.Stream, rows: usize) !f64 {
        const t0 = mtl.clock.seconds();
        const positions: [max_rows]u64 = @splat(0);
        const drafts: [max_rows]u32 = @splat(1);
        var sampled: [max_rows]u32 = undefined;
        var echoed: [max_rows]u32 = undefined;
        b.owners[0] = s;
        defer b.owners[0] = null;
        const w = be.Window{ .stream = s, .pending = 0, .held = 0, .tokens = drafts[0 .. rows - 1], .parents = null, .positions = positions[0..rows] };
        var out = [_]be.Verified{.{ .sampled = sampled[0..rows], .drafts = echoed[0 .. rows - 1] }};
        try verifyFn(b, &.{w}, &out);
        var path = [_]u32{0};
        try b.keepPaths(&.{&path});
        return (mtl.clock.seconds() - t0) * 1e3;
    }

    fn self(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }

    fn slotFor(b: *const Metal, s: *const lanes.Stream) !u32 {
        return b.slotOf(s) orelse error.UnknownStream;
    }

    fn push(b: *Metal, token: u32) u64 {
        b.drawn[b.next % ring] = token;
        b.next += 1;
        return b.next - 1;
    }

    fn value(b: *const Metal, feed: be.Feed) !u32 {
        return switch (feed) {
            .value => |v| v,
            .handle => |h| if (h < b.next and b.next - h <= ring) b.drawn[h % ring] else error.NoSuchToken,
        };
    }

    fn dropRound(b: *Metal) void {
        if (b.round) |*r| r.deinit();
        b.round = null;
        b.count = 0;
        b.early = 0;
    }

    fn dropTree(b: *Metal) void {
        if (b.tree) |*t| t.deinit();
        b.tree = null;
    }

    /// Slot 0's stream drafts with the DFlash2 head once a drafter is attached.
    fn drafter(b: *const Metal, slot: u32) ?*Draft {
        return if (slot == 0) b.draft else null;
    }

    /// Logits rows [first, first + rows) drawn at their positions: argmax on the GPU, or the stream's keyed sampler.
    fn draw(b: *Metal, s: *const lanes.Stream, first: usize, rows: usize, positions: []const u64, out: []u32) !void {
        const logits = b.runner.model.frame.get(.logits);
        const vocab: u32 = @intCast(b.runner.model.config.vocab);
        if (s.sampling) |sampling| {
            const words: [*]const u16 = @ptrCast(@alignCast(logits.buffer.contents() + logits.offset));
            // the 2B's draw: the top-k candidates first, then the shared keyed sampler (the same token as the whole row)
            for (out[0..rows], positions[0..rows], first..) |*t, p, r| t.* = try tf.qwen35.backend.draw(b.gpa, words[r * vocab ..][0..vocab], sampling, p);
            return;
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = b.runner.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        try b.ops.argmax(e, .{ .buf = logits.buffer, .off = logits.offset + first * vocab * 2 }, .{ .buf = b.picks }, .{ .rows = @intCast(rows), .vocab = vocab, .stride = vocab });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.DrawGpuFailure;
        const picks: [*]const contract.Pick = @ptrCast(@alignCast(b.picks.contents()));
        for (out[0..rows], picks[0..rows]) |*t, p| {
            if (p.token == contract.invalid_token or p.nonfinite != 0) return error.NonfiniteLogits;
            t.* = p.token;
        }
    }

    /// Keep the last verify's rows on `paths` (one a window). Slot 0 alone with a drafter keeps on the GPU with its
    /// taps and head; otherwise the runner keeps every window, then each slot's ring takes its kept rows' taps.
    fn keepPaths(b: *Metal, paths: []const []const u32) !void {
        defer b.dropRound();
        if (paths.len != b.count) return error.NothingToKeep;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        if (b.count == 1 and b.windows[0] == 0) {
            b.observe(paths[0]);
            if (b.draft) |d| return b.commitDevice(d, paths[0]);
        }
        try b.runner.keep(&b.round.?, paths);
        if (b.draft != null) try b.gatherTaps(paths);
        if (b.count != 1) return;
        // a lone window's last kept row's logits become row 0, where a one-token step leaves them
        const last = paths[0][paths[0].len - 1];
        if (last == 0) return;
        const logits = b.runner.model.frame.get(.logits);
        const bytes = b.runner.model.config.vocab * 2;
        const base = logits.buffer.contents() + logits.offset;
        @memcpy(base[0..bytes], base[@as(usize, last) * bytes ..][0..bytes]);
    }

    /// Every window's kept rows' taps in one gather, then into its slot's ring.
    fn gatherTaps(b: *Metal, paths: []const []const u32) !void {
        const r = b.runner;
        const round = &b.round.?;
        var n: usize = 0;
        const rows = b.kept.slice(u32, batch_rows);
        for (paths, round.firsts, b.windows[0..b.count]) |path, first, slot| {
            if (slot == 0 and b.early > 0) continue;
            for (path) |row| {
                rows[n] = first + row;
                n += 1;
            }
        }
        const out = b.gathered_taps.?;
        const cb = r.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        try r.taps.accepted(r.model.glue, e, rows[0..n], .{ .buffer = b.kept }, .{ .buffer = out });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.DraftTapGpuFailure;
        var at: usize = 0;
        for (paths, b.windows[0..b.count]) |path, slot| {
            const g = b.ringOf(slot).?;
            if (slot != 0 or b.early == 0) {
                try g.absorb(out, at * g.stride, @intCast(path.len));
                at += path.len;
            }
            if (g.end.* != r.offsets[slot]) return error.BadDraftPosition;
        }
    }

    /// One command buffer keeps slot 0's path rows, gathers their taps for the drafter and moves its last logits to row 0.
    fn commitDevice(b: *Metal, d: *Draft, path: []const u32) !void {
        const g = &d.generation;
        const r = b.runner;
        const result: *contract.Result = @ptrCast(@alignCast(g.matcher.buffers[4].contents()));
        result.* = .{ .status = 0, .stop = 0, .nonfinite = 0, .consumed_count = @intCast(path.len), .emitted_count = @intCast(path.len), .matched_count = 0, .bonus_emitted = 0, .pending_valid = 0, .pending_token = 0, .reserved = 0, .path = @splat(0), .tokens = @splat(0) };
        @memcpy(result.path[0..path.len], path);
        try r.taps.complete();
        const ref = tf.tree_commit_gpu.Ref{ .buf = g.matcher.buffers[4] };
        const logits = r.model.frame.get(.logits);
        const vocab: u32 = @intCast(r.model.config.vocab);
        const rows: u32 = @intCast(b.rows);
        const cb = r.model.queue.commandBuffer();
        const e = cb.compute(.concurrent);
        var ended = false;
        errdefer if (!ended) e.end();
        try r.keepDevice(&b.round.?, e, g.commit, ref);
        try g.commit.taps(e, ref, .{ .buf = r.taps.buffer }, .{ .buf = g.taps }, .{ .rows = rows, .width = r.taps.width, .capacity = r.taps.capacity, .planes = 5 });
        try g.commit.head(e, ref, .{ .buf = logits.buffer, .off = logits.offset }, .{ .rows = rows, .width = vocab, .stride = vocab });
        e.end();
        ended = true;
        cb.commit();
        cb.wait();
        if (cb.failure() != null) {
            r.failed = true;
            return error.KeepGpuFailure;
        }
        try r.completeDeviceKeep(&b.round.?, path);
        try g.draft.absorb(.{ .buf = g.taps }, @intCast(path.len));
        if (g.draft.committed_end != r.offsets[0]) return error.BadDraftPosition;
    }

    /// A verify whose rows all stayed keeps them before anything else (keep is called only to drop rows).
    fn settle(b: *Metal) !void {
        if (b.round == null) return;
        var rows: [batch_rows]u32 = undefined;
        for (&rows, 0..) |*p, i| p.* = @intCast(i);
        var paths: [8][]const u32 = undefined;
        for (paths[0..b.count], b.widths[0..b.count]) |*p, w| p.* = rows[0..w];
        try b.keepPaths(paths[0..b.count]);
    }

    /// A stream's prompt into `slot`: from its kept state when it has one, in chunks cut where a fresh pass cuts,
    /// decoded spans with decode arithmetic; slot 0 feeds the drafter, other slots their rings.
    fn prefillSlot(b: *Metal, s: *lanes.Stream, slot: u32) !void {
        const ids = s.prompt();
        if (ids.len == 0 or ids.len + s.max_new + max_rows > b.runner.capacity) return error.PromptTooLong;
        const d = b.drafter(slot);
        if (d) |x| try x.reset() else try b.runner.reset(slot);
        if (b.ringOf(slot)) |g| g.end.* = 0;
        s.cached = 0;
        var at: usize = 0;
        if (s.reuse.saved) |saved| { // a kept state of this prompt's prefix: the pass starts there
            const st: *const q.snapshot.State = @ptrCast(@alignCast(saved));
            if (st.at < ids.len) {
                if (b.restoreSlot(slot, st)) {
                    at = st.at;
                    s.cached = st.at;
                } else |_| {
                    s.reuse_failed = true;
                    if (d) |x| try x.reset() else try b.runner.reset(slot);
                    if (b.ringOf(slot)) |g| g.end.* = 0;
                }
            } else s.reuse_failed = true;
        }
        while (at < ids.len) {
            if (s.isCancelled()) return error.Cancelled;
            var end = @min(ids.len, at + b.chunk);
            for (s.chunks) |cut| if (cut > at and cut < end) {
                end = cut;
            };
            for (s.reuse.marks) |mark| if (mark > at and mark < end) {
                end = mark;
            };
            var decoded = false;
            for (s.decode_spans) |span| {
                for (span) |cut| if (cut > at and cut < end) {
                    end = cut;
                };
                decoded = decoded or (at >= span[0] and at < span[1]);
            }
            const part = ids[at..end];
            const last = end == ids.len;
            if (d) |x| {
                if (decoded) try x.decodeChunk(part, last) else try x.promptChunk(part, last);
            } else {
                const session = q.session.Session{ .runner = b.runner, .slot = slot };
                if (decoded) try session.decodeChunk(part, last) else try session.promptChunk(part, last);
                if (b.ringOf(slot)) |g| try b.ringChunk(g, part.len);
            }
            at = end;
            if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(at)) != null) if (s.reuse.hook) |k| k.at(k.ptr, s, @intCast(at));
        }
    }

    /// A prompt chunk's `rows` taps into a slot's ring.
    fn ringChunk(b: *Metal, g: Ring, rows: usize) !void {
        const r = b.runner;
        const list = b.kept.slice(u32, rows);
        for (list, 0..) |*x, i| x.* = @intCast(i);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = r.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        try r.taps.accepted(r.model.glue, e, list, .{ .buffer = b.kept }, .{ .buffer = b.gathered_taps.? });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.DraftTapGpuFailure;
        try g.absorb(b.gathered_taps.?, 0, @intCast(rows));
    }

    /// `slot` becomes a kept state: the drafter (slot 0) rebuilds its context from the restored taps.
    fn restoreSlot(b: *Metal, slot: u32, st: *const q.snapshot.State) !void {
        if (b.drafter(slot)) |d| try d.model.reset();
        try q.snapshot.restoreSlot(b.gpa, b.runner, slot, b.ringOf(slot), st);
    }

    /// A free slot for a new stream, slot 0 (the drafter's) first.
    fn freeSlot(b: *const Metal) ?u32 {
        for (b.owners[0..b.runner.slots], 0..) |o, i| if (o == null) return @intCast(i);
        return null;
    }

    /// The drafter absorbs nothing more for a stream that is gone; its slot is free once its state is kept.
    fn unbind(b: *Metal, s: *lanes.Stream) void {
        const slot = b.slotOf(s) orelse return;
        b.settle() catch {};
        b.dropRound();
        if (slot == 0) b.dropTree();
        if (b.release_hook) |hook| if (s.finished and s.reason != .cancelled and s.reason != .@"error") hook.kept(hook.ctx, s, slot);
        b.owners[slot] = null;
        if (slot == 0) b.promote();
    }

    /// Slot 0 freed while other streams decode: the lowest one moves there and drafts from its next round on.
    fn promote(b: *Metal) void {
        const d = b.draft orelse return;
        if (b.owners[0] != null or b.round != null) return;
        for (1..b.runner.slots) |k| {
            const s = b.owners[k] orelse continue;
            if (s.finished) continue;
            const from: u32 = @intCast(k);
            d.model.reset() catch return;
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            q.snapshot.moveSlot(b.gpa, b.runner, from, b.ringOf(from), 0, d.model.ring()) catch |err| {
                std.log.warn("qwen27: a stream stays in slot {d}, undrafted ({s})", .{ k, @errorName(err) });
                return;
            };
            b.owners[0] = s;
            b.owners[k] = null;
            b.rings[k].?.end = 0;
            return;
        }
    }

    // -- the lone driver ----------------------------------------------------------------------------------------

    /// A lone greedy stream on the DFlash2 serial loop in slot 0, until it finishes (false) or another request
    /// arrives (true: the lane core adopts it where it stands).
    pub fn lone(b: *Metal, s: *lanes.Stream, hooks: api.LoneHooks) !bool {
        const d = b.draft orelse return error.DraftNotAttached;
        if (s.sampling != null) return error.GreedyDraftsOnly;
        if (b.owners[0] != null) return error.StreamBusy;
        b.owners[0] = s;
        var handed = false;
        defer if (!handed) b.unbind(s);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try b.prefillSlot(s, 0);
        if (s.isCancelled()) return error.Cancelled;
        const g = &d.generation;
        _ = try s.commit(b.gpa, &.{try (q.session.Session{ .runner = b.runner }).greedy()}, &.{});
        hooks.committed(hooks.ctx);
        while (!s.finished) {
            if (hooks.yield(hooks.ctx)) {
                s.cache_len = b.runner.offsets[0];
                s.pending = s.context.items[s.context.items.len - 1];
                handed = true;
                return true;
            }
            const room: u32 = @intCast(s.max_new - s.emitted().len);
            const result = try g.runRound(room, b.nodes, s.eos, true);
            defer b.gpa.free(result.tokens);
            if (result.tokens.len == 0) break;
            s.rounds += @intCast(result.rounds);
            s.drafted += @intCast(result.verified - result.rounds);
            s.accepted += @intCast(result.matched);
            if (result.min_rows > 0) s.min_rows = if (s.min_rows == 0) result.min_rows else @min(s.min_rows, result.min_rows);
            _ = try s.commit(b.gpa, result.tokens, &.{});
            hooks.committed(hooks.ctx);
        }
        if (!s.finished) {
            s.finished = true;
            s.reason = .stop;
        }
        return false;
    }

    // -- the vtable ---------------------------------------------------------------------------------------------

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const b = self(ptr);
        b.settle() catch {};
        b.dropRound();
        const slot = b.slotOf(s) orelse (b.freeSlot() orelse return error.NoFreeSlot);
        try b.ready(slot);
        if (slot == 0) b.dropTree();
        b.owners[slot] = s;
        errdefer b.owners[slot] = null;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try b.prefillSlot(s, slot);
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const b = self(ptr);
        _ = try b.slotFor(s);
        var token: [1]u32 = undefined;
        try b.draw(s, 0, 1, &.{position}, &token);
        return b.push(token[0]);
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const b = self(ptr);
        const slot = try b.slotFor(s);
        try b.settle();
        const token = try b.value(feed);
        if (b.drafter(slot)) |d| try d.advance(token) else {
            try (q.session.Session{ .runner = b.runner, .slot = slot }).step(token);
            if (b.ringOf(slot)) |g| try b.ringChunk(g, 1);
        }
        var next: [1]u32 = undefined;
        try b.draw(s, 0, 1, &.{position}, &next);
        return b.push(next[0]);
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return self(ptr).value(.{ .handle = handle });
    }

    /// Every stream's window (pending token, then its drafts as a chain or tree) in one forward, every row drawn.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = self(ptr);
        if (windows.len == 0 or windows.len > b.runner.slots or windows.len != out.len) return error.BadWindows;
        try b.settle();
        const r = b.runner;
        var inputs: [8]q.round_plan.Input = undefined;
        var total: usize = 0;
        b.from_tree = 0;
        for (windows, 0..) |w, i| {
            const slot = try b.slotFor(w.stream);
            if (w.held > 0 and w.tokens.len > 0) return error.HeldAndHostDrafts;
            if (w.held > 0 and slot != 0) return error.HeldWithoutDrafter;
            const rows = w.rows();
            if (rows > max_rows or total + rows > batch_rows) return error.WindowTooWide;
            const ids = b.ids[total..][0..rows];
            ids[0] = w.pending;
            if (w.held > 0) b.heldChain(ids[1..], w.pending) else @memcpy(ids[1..], w.tokens);
            if (slot == 0 and windows.len == 1) b.from_tree = b.treeRows(w);
            const parents = b.parents[total..][0..rows];
            if (w.parents) |p| @memcpy(parents, p) else for (parents, 0..) |*p, row| {
                p.* = @as(i32, @intCast(row)) - 1;
            }
            inputs[i] = .{ .slot = slot, .start = r.offsets[slot], .capacity = r.capacity, .ids = ids, .parents = parents };
            b.windows[i] = slot;
            b.widths[i] = @intCast(rows);
            total += rows;
        }
        // a tree held for slot 0 is spent by this round, alone or shared
        if (windows.len > 1) b.dropTree();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        b.round = try Round.init(b.gpa, inputs[0..windows.len], @intCast(r.model.config.conv_kernel), r.slots, @intCast(r.model.config.vocab));
        b.count = windows.len;
        errdefer b.dropRound();
        try r.verifyHead(&b.round.?, .all);
        b.rows = total;
        for (windows, out, b.round.?.firsts) |w, o, first| {
            const rows = w.rows();
            try b.draw(w.stream, first, rows, w.positions, o.sampled);
            @memcpy(o.drafts, b.ids[first + 1 ..][0 .. rows - 1]);
        }
        if (windows.len == 1) @memcpy(b.sampled[0..b.rows], out[0].sampled);
    }

    /// How many drafts are the held tree's first nodes with their own parents (0: copies, fills or a chain).
    fn treeRows(b: *const Metal, w: be.Window) usize {
        const t = b.tree orelse return 0;
        const n = w.tokens.len;
        if (w.held > 0 or n == 0 or n > t.tokens.len or !std.mem.eql(u32, w.tokens, t.tokens[0..n])) return 0;
        const parents = w.parents orelse return if (isChain(t.parents[0..n])) n else 0;
        for (parents[1..], t.parents[0..n]) |row, node| if (row != node + 1) return 0;
        return n;
    }

    fn isChain(parents: []const i32) bool {
        for (parents, 0..) |p, i| if (p != @as(i32, @intCast(i)) - 1) return false;
        return true;
    }

    /// A node's own log-probability under the drafter (its path score less its parent's).
    fn local(t: Tree, i: usize) f64 {
        const parent = t.parents[i];
        return t.scores[i] - (if (parent >= 0) t.scores[@intCast(parent)] else 0);
    }

    fn bucket(own: f64) usize {
        return @min(15, @as(usize, @intFromFloat(@max(0, @floor(-own / 0.5)))));
    }

    /// Count every proposed node whose parent row was kept, and those the target drew, by own log-probability.
    fn observe(b: *Metal, path: []const u32) void {
        const t = b.tree orelse return;
        if (b.from_tree == 0) return;
        var kept: [max_rows]bool = @splat(false);
        for (path) |r| kept[r] = true;
        for (t.tokens, t.parents, 0..) |token, parent, i| {
            const parent_row: usize = @intCast(parent + 1);
            if (parent_row > b.from_tree or !kept[parent_row]) continue;
            const x = &b.landing[bucket(local(t, i))];
            x[0] += 1;
            if (token == b.sampled[parent_row]) x[1] += 1;
        }
    }

    /// Held drafts the core reads as a chain (a first round): the tree's first-child path, last token repeated.
    fn heldChain(b: *const Metal, out: []u32, pending: u32) void {
        var n: usize = 0;
        if (b.tree) |t| {
            var parent: i32 = -1;
            for (t.tokens, t.parents, 0..) |token, p, i| {
                if (n == out.len) break;
                if (p != parent) continue;
                out[n] = token;
                n += 1;
                parent = @intCast(i);
            }
        }
        const fill = if (n > 0) out[n - 1] else pending;
        for (out[n..]) |*x| x.* = fill;
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = self(ptr);
        if (b.round == null or windows.len != b.count) return error.NothingToKeep;
        for (windows, paths, b.windows[0..b.count]) |w, path, slot| if (path.len == 0 or path[0] != 0 or b.slotOf(w.stream) != slot) return error.NothingToKeep;
        try b.keepPaths(paths);
    }

    /// Slot 0's stream: the drafter proposes the next round's tree from its pending token. In a shared round the
    /// drafts come before the keep, so its kept rows' taps reach the drafter here, from the verify's taps.
    /// Every other stream drafts nothing; its rows reach its ring at the keep.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const b = self(ptr);
        for (requests) |r| {
            const slot = try b.slotFor(r.stream);
            const d = b.drafter(slot) orelse continue;
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            if (b.count > 1) try b.absorbEarly(d, r.rows orelse return error.NoKeptRows) else try b.settle();
            b.dropTree();
            const g = &d.generation;
            const stands = b.runner.offsets[0] + b.early;
            if (g.draft.committed_end != stands) return error.BadDraftPosition;
            const room = b.runner.capacity - stands;
            if (r.depth == 0 or room < 2) continue;
            const pending = if (r.rows != null) r.follow[r.follow.len - 1] else try b.value(r.first orelse return error.NoFirstToken);
            const nodes: u32 = @intCast(@min(@min(r.depth, b.nodes), room - 1));
            b.tree = if (g.proposer) |p| try p.call(p.ptr, pending, nodes) else try g.draft.propose(pending, nodes);
        }
    }

    /// Slot 0's kept rows of a shared round's verify, gathered from its taps for the drafter before the keep.
    fn absorbEarly(b: *Metal, d: *Draft, path: []const u32) !void {
        if (b.early > 0) return error.AbsorbedTwice;
        const r = b.runner;
        const w = std.mem.indexOfScalar(u32, b.windows[0..b.count], 0) orelse return error.UnknownStream;
        const first = b.round.?.firsts[w];
        const rows = b.kept.slice(u32, path.len);
        for (rows, path) |*x, row| x.* = first + row;
        const cb = r.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        try r.taps.accepted(r.model.glue, e, rows, .{ .buffer = b.kept }, .{ .buffer = b.gathered_taps.? });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.DraftTapGpuFailure;
        try d.generation.draft.absorb(.{ .buf = b.gathered_taps.? }, @intCast(path.len));
        b.early = path.len;
    }

    /// Each held node's chance of landing: its own given its parent (e^own as the prior) times its parent's.
    fn probabilitiesFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
        const b = self(ptr);
        if (b.slotOf(s) != 0) return false;
        const t = b.tree orelse return false;
        if (out.len > t.scores.len) return false;
        for (out, t.parents[0..out.len], 0..) |*p, parent, i| {
            const own = local(t, i);
            const x = b.landing[bucket(own)];
            p.* = (x[1] + 2 * @exp(own)) / (x[0] + 2) * (if (parent >= 0) out[@intCast(parent)] else 1);
        }
        return true;
    }

    /// Slot 0's held tree as host tokens; any other stream holds no drafts (its windows verify copies or one row).
    fn treeFn(ptr: *anyopaque, s: *lanes.Stream, gpa: std.mem.Allocator) anyerror!?lanes.stream.Held {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return null;
        const t = (if (slot == 0) b.tree else null) orelse return .{ .count = 0, .tokens = try gpa.alloc(u32, 0) };
        const tokens = try gpa.dupe(u32, t.tokens);
        errdefer gpa.free(tokens);
        return .{ .count = @intCast(t.tokens.len), .tokens = tokens, .parents = try gpa.dupe(i32, t.parents) };
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        self(ptr).unbind(s);
    }
};
