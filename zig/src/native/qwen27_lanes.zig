//! The 27B's lane core backend: a stream a runner slot, shared rounds across slots, DFlash2 trees for slot 0.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const be = lanes.backend;
pub const Draft = @import("qwen27_draft.zig").Draft;
const slots = @import("qwen27_slots.zig");
const draft_fns = @import("qwen27_drafts.zig");
const Round = q.round_plan.Round;
const Tree = q.dflash.selector.Tree;
const Ring = q.tap_ring.Ring;
const Context = q.dflash.runtime_model.Context;
const Landing = @import("qwen27_landing.zig").Landing;
const contract = tf.tree_round;

/// A window's rows at most: the pending row and a DFlash2 tree.
pub const max_rows = 16;
/// A shared round's rows at most: the frame's.
pub const batch_rows = 128;
/// Streams a runner holds at most (the attention plan's).
pub const max_streams = q.attention_plan.max_streams;
const ring = 64;
pub const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// Called before a finished stream leaves its slot, its state still standing there (the reply's prompt-cache keep).
pub const Release = struct { ctx: *anyopaque, kept: *const fn (ctx: *anyopaque, s: *lanes.Stream, slot: u32) void };

/// A slot's memory on first use: `room` says whether `bytes` more fit for stream `s`, `added` takes the new buffers.
pub const Grow = struct {
    ctx: *anyopaque,
    room: *const fn (ctx: *anyopaque, bytes: usize, s: *const lanes.Stream) bool,
    added: *const fn (ctx: *anyopaque, buffers: []const mtl.Buffer) void,
};

pub const Metal = struct {
    residency: ?mtl.ResidencySet.Follow = null,
    gpa: std.mem.Allocator,
    runner: *q.decode_round.Runner,
    draft: ?*Draft,
    chunk: usize = 128, // prompt rows a pass
    nodes: u32 = max_rows - 1, // a tree's drafts at most
    owners: [max_streams]?*lanes.Stream = @splat(null), // the stream each runner slot holds
    contexts: [max_streams]?*Context = @splat(null), // slots past 0: their taps (a kept state carries them) and drafter state
    draft_slots: u32 = 1, // slots below this draft their own trees while no more streams are live (TF_Q27_DRAFT_SLOTS)
    release_hook: ?Release = null,
    grow: ?Grow = null,
    drawn: [ring]u32 = undefined, // drawn tokens by handle
    next: u64 = 0,
    round: ?Round = null, // the last verify, until its rows are kept
    windows: [max_streams]u32 = undefined, // the last verify's slots, in window order
    widths: [max_streams]u32 = undefined, // and their rows
    count: usize = 0, // the last verify's windows
    early: [max_streams]usize = @splat(0), // a slot's kept rows its drafter absorbed before a shared round's keep
    ids: [batch_rows]u32 = undefined,
    parents: [batch_rows]i32 = undefined,
    rows: usize = 0,
    trees: [max_streams]?Tree = @splat(null), // the drafts held for each slot's next round
    ops: tf.tree_round_gpu.Ops,
    picks: mtl.Buffer,
    kept: mtl.Buffer, // a keep's gathered rows, by round row
    gathered_taps: ?mtl.Buffer = null, // and their taps, in path order (with a drafter)
    costs: [max_rows]lanes.config.Cost = undefined, // a window's ms by rows, timed by measure()
    timed: usize = 0,
    from_tree: usize = 0, // the last window's drafts that were the held tree's first nodes
    advanced: bool = false, // the round's rows were kept by its verify (one undrafted row a window): no keep to run
    sampled: [max_rows]u32 = undefined, // the last lone window's draws
    landing: Landing = .{}, // drafted nodes offered with their parent kept, and landed, by own log-probability
    vision: ?*@import("qwen27_images.zig").Vision = null, // --vision: the image tower and a prompt pass's image rows
    fills: std.AutoHashMapUnmanaged(*lanes.Stream, slots.Fill) = .empty, // prompt passes the lane host runs by chunks
    image_fill: ?*const lanes.Stream = null, // the pass whose images the vision rows hold, until it closes

    pub fn init(gpa: std.mem.Allocator, runner: *q.decode_round.Runner, draft: ?*Draft) !*Metal {
        if (runner.slots == 0 or runner.slots > max_streams or runner.model.frame.capacity < batch_rows) return error.TargetBinding;
        const device = runner.model.device;
        const ops = try tf.tree_round_gpu.Ops.init(device);
        errdefer ops.deinit();
        const picks = try device.buffer(batch_rows * @sizeOf(contract.Pick), opts);
        errdefer picks.deinit();
        const kept = try device.buffer(batch_rows * 4, opts);
        errdefer kept.deinit();
        const b = try gpa.create(Metal);
        errdefer gpa.destroy(b);
        b.* = .{ .gpa = gpa, .runner = runner, .draft = draft, .ops = ops, .picks = picks, .kept = kept, .draft_slots = draftSlots() };
        errdefer b.freeContexts();
        if (draft) |d| {
            const g = d.model.ring();
            b.gathered_taps = try device.buffer(batch_rows * g.stride, opts);
            for (b.contexts[1..runner.ready], 1..) |*x, slot| x.* = try Context.init(gpa, d.model, slot < b.draft_slots);
        }
        return b;
    }

    fn freeContexts(b: *Metal) void {
        for (&b.contexts) |*c| if (c.*) |x| {
            x.deinit(b.gpa);
            c.* = null;
        };
        if (b.gathered_taps) |t| t.deinit();
        b.gathered_taps = null;
    }

    pub fn deinit(b: *Metal) void {
        q.residency.setBack(b, null);
        b.dropRound();
        b.dropTree();
        b.freeContexts();
        b.fills.deinit(b.gpa);
        b.kept.deinit();
        b.picks.deinit();
        b.ops.deinit();
        b.gpa.destroy(b);
    }

    pub const lone = slots.lone;
    const prefillSlot = slots.prefill;
    const freeSlot = slots.free;
    const ringChunk = slots.ringChunk;
    const unbind = slots.unbind;
    const ready = slots.ready;
    const refill = slots.refill;

    pub fn backend(b: *Metal) be.Backend {
        return .{ .ptr = b, .vtable = &.{ .prefill = prefillFn, .prefill_step = slots.prefillStep, .cancel_prefill = slots.cancelPrefill, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draft_fns.draftFn, .probabilities = draft_fns.probabilitiesFn, .tree = draft_fns.treeFn, .release = releaseFn, .reports_fill = true } };
    }

    /// What the round loop reads at setup: slot 0's stream drafts trees; every stream verifies copies.
    pub fn facts(b: *const Metal) lanes.Model {
        const drafting = b.draft != null;
        const shared = b.runner.slots > 1;
        // a lone stream drafts DFlash2 trees, as in 1.0.5: copies land less often there (TF_QWEN27_COPIES=1 opts in)
        const lone_copies = if (platform.c.getenv("TF_QWEN27_COPIES")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        return .{ .exact_width = if (drafting or shared) max_rows else 1, .mtp = drafting, .speculate = drafting, .speculate_early = false, .drafts = b.nodes, .draft_probabilities = true, .window_costs = b.costs[0..b.timed], .hidden_rows = shared, .streams_exact = true, .batch_rows = if (shared) batch_rows else max_rows, .max_streams = b.runner.slots, .lone_copies = lone_copies, .draft_streams = drafting and b.draft_slots > 1 };
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
        const x = b.contexts[slot] orelse return null;
        return x.ring(d.model.graph.config);
    }

    /// Whether `slot`'s stream drafts its own trees.
    pub fn draftsOwn(b: *const Metal, slot: u32) bool {
        if (b.draft == null) return false;
        if (slot == 0) return true;
        const x = b.contexts[slot] orelse return false;
        var live: u32 = 0;
        for (b.owners[0..b.runner.slots]) |o| live += @intFromBool(o != null);
        return slot < b.draft_slots and live <= b.draft_slots and x.caches.len > 0;
    }

    /// Each window width's ms (verify, draws, keep) on a scratch context in slot 0; resets it. Kept per build.
    pub fn measure(b: *Metal, io: std.Io) !void {
        const r = b.runner;
        if (b.owners[0] != null or r.active != null) return error.StreamBusy;
        var shape: [128]u8 = undefined;
        const parts = [_][]const u8{ "qwen27-lanes", std.mem.span(r.model.device.name()), try q.widths.costShape(r.model.config, r.model.widths_tag, &shape, max_rows, b.draft != null) };
        const k = lanes.cost_cache.key(b.gpa, io, &parts) catch null;
        if (k) |key| if (lanes.cost_cache.load([max_rows]lanes.config.Cost, b.gpa, io, key)) |kept| {
            b.costs = kept;
            b.timed = max_rows;
            return;
        };
        if (r.capacity <= max_rows) return error.ContextFull;
        defer {
            if (b.draft) |d| d.reset() catch {} else r.reset(0) catch {};
        }
        var ids: [192]u32 = undefined;
        for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % r.model.config.vocab);
        const prompt = ids[0..@min(ids.len, r.capacity - max_rows)]; // a small --context still holds a window past it
        try b.refill(prompt);
        var s = try lanes.Stream.init(b.gpa, .{ .id = "timing", .prompt = prompt, .max_new = 1 });
        defer s.deinit(b.gpa);
        const timer: WindowTimer = .{ .b = b, .s = &s, .prompt = prompt };
        for (0..5) |_| _ = try timer.time(max_rows - 1); // the widest window first, while the GPU's clocks ramp up
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

    /// Window entry `i` is `i + 1` rows, timed once; each keeps a row, so a full slot is refilled first.
    const WindowTimer = struct {
        b: *Metal,
        s: *lanes.Stream,
        prompt: []const u32,
        pub fn time(t: WindowTimer, i: usize) !f64 {
            if (t.b.runner.offsets[0] + i + 1 > t.b.runner.capacity) try t.b.refill(t.prompt);
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

    pub fn self(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }

    pub fn slotFor(b: *const Metal, s: *const lanes.Stream) !u32 {
        return b.slotOf(s) orelse error.UnknownStream;
    }

    fn push(b: *Metal, token: u32) u64 {
        b.drawn[b.next % ring] = token;
        b.next += 1;
        return b.next - 1;
    }

    pub fn value(b: *const Metal, feed: be.Feed) !u32 {
        return switch (feed) {
            .value => |v| v,
            .handle => |h| if (h < b.next and b.next - h <= ring) b.drawn[h % ring] else error.NoSuchToken,
        };
    }

    pub fn dropRound(b: *Metal) void {
        if (b.round) |*r| r.deinit();
        b.round = null;
        b.count = 0;
        b.early = @splat(0);
        b.advanced = false;
    }

    pub fn dropTree(b: *Metal) void {
        for (0..max_streams) |slot| b.dropTreeOf(@intCast(slot));
    }

    pub fn dropTreeOf(b: *Metal, slot: u32) void {
        if (b.trees[slot]) |*t| t.deinit();
        b.trees[slot] = null;
    }

    /// Slot 0's stream runs its prompt and single steps through the drafter's own generation.
    pub fn drafter(b: *const Metal, slot: u32) ?*Draft {
        return if (slot == 0) b.draft else null;
    }

    /// Logits rows [first, first + rows) drawn at their positions: argmax on the GPU, or the stream's keyed sampler.
    fn draw(b: *Metal, s: *const lanes.Stream, first: usize, rows: usize, positions: []const u64, out: []u32, masks: []const u32) !void {
        const logits = b.runner.model.frame.get(.logits);
        const vocab: u32 = @intCast(b.runner.model.config.vocab);
        const width = (@as(usize, vocab) + 31) / 32;
        if (masks.len != 0 and masks.len != rows * width) return error.InvalidQwenMasks;
        if (s.sampling != null or masks.len > 0) {
            const words: [*]const u16 = @ptrCast(@alignCast(logits.buffer.contents() + logits.offset));
            // the 2B's draw: top-k candidates, then the shared keyed sampler (the whole row's token)
            for (out[0..rows], positions[0..rows], 0..) |*t, p, row| {
                const mask = if (masks.len == 0) &.{} else masks[row * width ..][0..width];
                t.* = try @import("qwen27_masks.zig").draw(tf.qwen35.backend.draw, b.gpa, words[(first + row) * vocab ..][0..vocab], s.sampling, p, mask);
            }
            return;
        }
        try b.argmaxRows(first, rows);
        try b.picked(first, rows, out);
    }

    /// Rows [first, first + rows)'s greedy picks into `picks` at the same rows: one command buffer and one wait.
    fn argmaxRows(b: *Metal, first: usize, rows: usize) !void {
        const logits = b.runner.model.frame.get(.logits);
        const vocab: u32 = @intCast(b.runner.model.config.vocab);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = b.runner.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        var at = first;
        while (at < first + rows) : (at += contract.max_rows) {
            const n = @min(first + rows - at, contract.max_rows);
            try b.ops.argmax(e, .{ .buf = logits.buffer, .off = logits.offset + at * vocab * 2 }, .{ .buf = b.picks, .off = at * @sizeOf(contract.Pick) }, .{ .rows = @intCast(n), .vocab = vocab, .stride = vocab });
        }
        e.end();
        cb.commit();
        cb.wait();
        if (q.decode_round.gpuFailed(cb, "a draw")) return error.DrawGpuFailure;
    }

    /// The tokens argmaxRows picked for rows [first, first + rows).
    fn picked(b: *const Metal, first: usize, rows: usize, out: []u32) !void {
        const picks: [*]const contract.Pick = @ptrCast(@alignCast(b.picks.contents()));
        for (out[0..rows], picks[first..][0..rows]) |*t, p| {
            if (p.token == contract.invalid_token or p.nonfinite != 0) return error.NonfiniteLogits;
            t.* = p.token;
        }
    }

    /// Keep the last verify's rows on `paths`: slot 0 alone keeps on the GPU with its taps, else the runner and rings.
    fn keepPaths(b: *Metal, paths: []const []const u32) !void {
        defer b.dropRound();
        if (paths.len != b.count) return error.NothingToKeep;
        if (b.advanced) { // the verify kept each window's one row at its position; its logits already sit in row 0
            for (paths) |path| if (path.len != 1) return error.NothingToKeep;
            return;
        }
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
            if (b.early[slot] > 0) continue;
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
        if (q.decode_round.gpuFailed(cb, "a taps gather")) return error.DraftTapGpuFailure;
        var at: usize = 0;
        for (paths, b.windows[0..b.count]) |path, slot| {
            const g = b.ringOf(slot).?;
            if (b.early[slot] == 0) {
                try g.absorb(out, at * g.stride, @intCast(path.len));
                at += path.len;
            }
            if (g.end.* != r.offsets[slot]) return error.BadDraftPosition;
        }
    }

    /// One command buffer keeps slot 0's path rows, gathers their taps and moves its last logits to row 0.
    fn commitDevice(b: *Metal, d: *Draft, path: []const u32) !void {
        const g = &d.generation;
        // the result below is host-written, and a pending keep reads it
        std.debug.assert(g.keep == null);
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
        if (q.decode_round.gpuFailed(cb, "slot 0's keep")) {
            r.failed = true;
            return error.KeepGpuFailure;
        }
        try r.completeDeviceKeep(&b.round.?, path);
        try g.draft.absorb(.{ .buf = g.taps }, @intCast(path.len));
        if (g.draft.committed_end != r.offsets[0]) return error.BadDraftPosition;
    }

    /// A verify whose rows all stayed keeps them before anything else (keep is called only to drop rows).
    pub fn settle(b: *Metal) !void {
        if (b.round == null) return;
        var rows: [batch_rows]u32 = undefined;
        for (&rows, 0..) |*p, i| p.* = @intCast(i);
        var paths: [max_streams][]const u32 = undefined;
        for (paths[0..b.count], b.widths[0..b.count]) |*p, w| p.* = rows[0..w];
        try b.keepPaths(paths[0..b.count]);
    }

    // -- the vtable ---------------------------------------------------------------------------------------------

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const b = self(ptr);
        b.settle() catch {};
        b.dropRound();
        const slot = try slots.bind(b, s);
        errdefer b.owners[slot] = null;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try b.prefillSlot(s, slot);
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const b = self(ptr);
        _ = try b.slotFor(s);
        var token: [1]u32 = undefined;
        try b.draw(s, 0, 1, &.{position}, &token, &.{});
        return b.push(token[0]);
    }

    fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
        const b = self(ptr);
        _ = try b.slotFor(s);
        var token: [1]u32 = undefined;
        try b.draw(s, 0, 1, &.{position}, &token, mask);
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
        try b.draw(s, 0, 1, &.{position}, &next, &.{});
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
        var inputs: [max_streams]q.round_plan.Input = undefined;
        var total: usize = 0;
        b.from_tree = 0;
        var plain = b.draft == null; // every window one undrafted row: the verify keeps it, as a prompt's rows are
        for (windows, 0..) |w, i| {
            if (w.rows() != 1 or w.held != 0) plain = false;
            const slot = try b.slotFor(w.stream);
            if (w.held > 0 and w.tokens.len > 0) return error.HeldAndHostDrafts;
            if (w.held > 0 and !b.draftsOwn(slot)) return error.HeldWithoutDrafter;
            const rows = w.rows();
            const words = (r.model.config.vocab + 31) / 32;
            if (w.masks.len != 0 and w.masks.len != rows * words) return error.InvalidQwenMasks;
            if (rows > max_rows or total + rows > batch_rows) return error.WindowTooWide;
            const ids = b.ids[total..][0..rows];
            ids[0] = w.pending;
            if (w.held > 0) b.heldChain(slot, ids[1..], w.pending) else @memcpy(ids[1..], w.tokens);
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
        // every tree held is spent by this round, alone or shared
        if (windows.len > 1) b.dropTree();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        b.round = try Round.init(b.gpa, inputs[0..windows.len], @intCast(r.model.config.conv_kernel), r.slots, @intCast(r.model.config.vocab));
        b.count = windows.len;
        errdefer b.dropRound();
        if (plain) try r.advance(&b.round.?, .all) else try r.verifyHead(&b.round.?, .all);
        b.advanced = plain;
        b.rows = total;
        // every greedy window's rows from one argmax over the round; a sampled window draws on the host
        var greedy = false;
        for (windows) |w| greedy = greedy or (w.stream.sampling == null and w.masks.len == 0);
        if (greedy) try b.argmaxRows(0, total);
        for (windows, out, b.round.?.firsts) |w, o, first| {
            const rows = w.rows();
            if (w.stream.sampling != null or w.masks.len > 0) try b.draw(w.stream, first, rows, w.positions, o.sampled, w.masks) else try b.picked(first, rows, o.sampled);
            @memcpy(o.drafts, b.ids[first + 1 ..][0 .. rows - 1]);
        }
        if (windows.len == 1) @memcpy(b.sampled[0..b.rows], out[0].sampled);
    }

    /// How many drafts are the held tree's first nodes with their own parents (0: copies, fills or a chain).
    fn treeRows(b: *const Metal, w: be.Window) usize {
        const t = b.trees[0] orelse return 0;
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
    pub fn local(t: Tree, i: usize) f64 {
        const parent = t.parents[i];
        return t.scores[i] - (if (parent >= 0) t.scores[@intCast(parent)] else 0);
    }

    /// Count every proposed node whose parent row was kept, and those the target drew, by own log-probability.
    fn observe(b: *Metal, path: []const u32) void {
        const t = b.trees[0] orelse return;
        if (b.from_tree == 0) return;
        const keyed = if (b.owners[0]) |s| s.sampling != null else true;
        var kept: [max_rows]bool = @splat(false);
        for (path) |r| kept[r] = true;
        for (t.tokens, t.parents, 0..) |token, parent, i| {
            const parent_row: usize = @intCast(parent + 1);
            if (parent_row > b.from_tree or !kept[parent_row]) continue;
            b.landing.observe(keyed, local(t, i), token == b.sampled[parent_row]);
        }
    }

    /// Held drafts the core reads as a chain (a first round): the tree's first-child path, last token repeated.
    fn heldChain(b: *const Metal, slot: u32, out: []u32, pending: u32) void {
        var n: usize = 0;
        if (b.trees[slot]) |t| {
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

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        slots.endFill(self(ptr), s);
        self(ptr).unbind(s);
    }
};

/// Streams that draft their own trees in shared rounds, while no more are live (TF_Q27_DRAFT_SLOTS; 1: slot 0 alone).
fn draftSlots() u32 {
    const v = std.c.getenv("TF_Q27_DRAFT_SLOTS") orelse return 8;
    return std.math.clamp(std.fmt.parseInt(u32, std.mem.span(v), 10) catch 8, 1, max_streams);
}
