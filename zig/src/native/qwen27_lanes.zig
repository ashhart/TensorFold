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
pub const max_rows = @import("qwen27_window.zig").capacity;
pub fn windowRows(units: bool) !u32 {
    return @import("qwen27_window.zig").requested(units, if (platform.c.getenv("TF_QWEN27_WINDOW")) |v| std.mem.span(v) else null);
}
/// A shared round's rows at most: the frame's.
pub const batch_rows = 128;
/// Streams a runner holds at most (the attention plan's).
pub const max_streams = q.attention_plan.max_streams;
const ring = 64;
pub const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

comptime {
    std.debug.assert(@import("qwen27_window.zig").nodes(max_rows, false) <= q.dflash.runtime_model.max_nodes);
    std.debug.assert(max_rows <= contract.max_rows and max_rows <= ring);
    std.debug.assert(max_rows <= q.attention_plan.max_rows and max_rows <= q.attention_plan.path_width);
    std.debug.assert(max_rows <= q.gpu_frame.logit_rows and batch_rows <= q.gpu_frame.max_rows);
}

pub const Release = @import("qwen27_hooks.zig").Release;
pub const Grow = @import("qwen27_hooks.zig").Grow;

pub const Metal = struct {
    residency: ?mtl.ResidencySet.Follow = null,
    gpa: std.mem.Allocator,
    pass_profile: bool = false,
    runner: *q.decode_round.Runner,
    draft: ?*Draft,
    chunk: usize = 128, // prompt rows a pass
    window: u32 = 16, // the device per-stream verification width
    copies: bool = false, // lone context copies leave room past a 15-node tree
    nodes: u32 = 15, // a tree's drafts at most
    owners: [max_streams]?*lanes.Stream = @splat(null), // the stream each runner slot holds
    contexts: [max_streams]?*Context = @splat(null), // slots past 0: their taps (a kept state carries them) and drafter state
    draft_slots: u32 = 1, // slots below this draft their own trees while no more streams are live (TF_Q27_DRAFT_SLOTS)
    release_hook: ?Release = null,
    checkpoint_hook: ?lanes.stream.MarkHook = null, // unplanned prompt states only when the cache has room
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
    prefetched: [max_rows]u32 = undefined, // a greedy pick submission already gathered this complete path
    prefetched_count: usize = 0,
    prefetch_encoded: bool = false,
    prepared_keep: ?mtl.CommandBuffer = null, // encoded during verify, committed only after the core chooses its path
    sampled: [max_rows]u32 = undefined, // the last lone window's draws
    landing: Landing = .{}, // drafted nodes offered with their parent kept, and landed, by own log-probability
    vision: ?*@import("qwen27_images.zig").Vision = null, // --vision: the image tower and a prompt pass's image rows
    fills: std.AutoHashMapUnmanaged(*lanes.Stream, slots.Fill) = .empty, // prompt passes the lane host runs by chunks
    image_fill: ?*const lanes.Stream = null, // the pass whose images the vision rows hold, until it closes
    promote_due: bool = false, // slot 0 freed: a stream moves in at the next backend call (slots.promoteDue)
    standing: @import("qwen27_resume.zig").Standing = .{}, // what slot 0 stands at once its stream left (#1197)
    rewind: ?mtl.Buffer = null, // slot 0's DeltaNet states and taps at its last pass's history end
    slot_resume: bool = true, // TF_QWEN27_SLOT_RESUME: a later prompt resumes where slot 0 stands (0: off)

    pub fn init(gpa: std.mem.Allocator, runner: *q.decode_round.Runner, draft: ?*Draft) !*Metal {
        if (runner.slots == 0 or runner.slots > max_streams or runner.model.frame.capacity < batch_rows) return error.TargetBinding;
        const device = runner.model.device;
        const window = try windowRows(device.tensorUnits());
        const copies = try @import("qwen27_window.zig").copyPolicy(device.tensorUnits(), window, if (platform.c.getenv("TF_QWEN27_COPIES")) |v| std.mem.span(v) else null);
        const ops = try tf.tree_round_gpu.Ops.init(device);
        errdefer ops.deinit();
        const picks = try device.buffer(batch_rows * @sizeOf(contract.Pick), opts);
        errdefer picks.deinit();
        const kept = try device.buffer(batch_rows * 4, opts);
        errdefer kept.deinit();
        const b = try gpa.create(Metal);
        errdefer gpa.destroy(b);
        b.* = .{ .gpa = gpa, .runner = runner, .draft = draft, .window = window, .copies = copies, .nodes = @import("qwen27_window.zig").nodes(window, copies), .ops = ops, .picks = picks, .kept = kept, .draft_slots = draftSlots(), .slot_resume = slots.slotResumeEnv() };
        b.pass_profile = platform.c.getenv("TF_Q27_PASS_PROFILE") != null;
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
        b.settle() catch {};
        q.residency.setBack(b, null);
        b.standing.deinit(b.gpa);
        if (b.rewind) |x| x.deinit();
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
    pub const refill = slots.refill;

    pub fn backend(b: *Metal) be.Backend {
        return .{ .ptr = b, .defer_lone_draft = b.runner.slots == 1, .vtable = &.{ .prefill = prefillFn, .prefill_step = slots.prefillStep, .cancel_prefill = slots.cancelPrefill, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draft_fns.draftFn, .draft_capacity = draft_fns.capacityFn, .probabilities = draft_fns.probabilitiesFn, .tree = draft_fns.treeFn, .release = releaseFn, .reports_fill = true } };
    }

    /// What the round loop reads at setup: slot 0's stream drafts trees; every stream verifies copies.
    pub fn facts(b: *const Metal) lanes.Model {
        const drafting = b.draft != null;
        const shared = b.runner.slots > 1;
        // Context copies remain opt-in until their served cost is qualified.
        const lone_copies = b.copies;
        return .{ .exact_width = if (drafting or shared) b.window else 1, .mtp = drafting, .speculate = drafting, .speculate_early = false, .drafts = b.nodes, .draft_probabilities = true, .window_costs = b.costs[0..b.timed], .hidden_rows = shared, .streams_exact = true, .batch_rows = if (shared) batch_rows else b.window, .max_streams = b.runner.slots, .lone_copies = lone_copies, .context_copies = lone_copies, .allocate_shared_drafts = drafting, .draft_streams = drafting and b.draft_slots > 1 };
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

    /// Price only this device's active window widths; cache records carry that width.
    pub fn measure(b: *Metal, io: std.Io) !void {
        return @import("qwen27_costs.zig").measure(b, io);
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
        if (b.prepared_keep) |cb| cb.release();
        b.prepared_keep = null;
        if (b.round) |*r| r.deinit();
        b.round = null;
        b.count = 0;
        b.early = @splat(0);
        b.advanced = false;
        b.prefetched_count = 0;
        b.prefetch_encoded = false;
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
            const metric = mtl.pass_metrics.phase(.sampling);
            defer metric.end();
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
        const metric = mtl.pass_metrics.phase(.sampling);
        defer metric.end();
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
        defer mtl.pass_metrics.end();
        const metric = mtl.pass_metrics.phase(.keep);
        defer metric.end();
        defer b.dropRound();
        if (paths.len != b.count) return error.NothingToKeep;
        if (b.pass_profile and b.count == 1) std.log.info("q27_keep width={d} rows={d} kept={d}", .{ b.window, b.rows, paths[0].len });
        if (b.advanced) { // the verify kept each window's one row at its position; its logits already sit in row 0
            for (paths) |path| if (path.len != 1) return error.NothingToKeep;
            return;
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        if (b.count == 1 and b.windows[0] == 0) {
            b.observe(paths[0]);
            if (b.draft) |d| return @import("qwen27_keep.zig").commit(b, d, paths[0]);
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

    /// Completed tap bytes enter each slot's ring without a second GPU gather submission.
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
        // Runner.keep waited on this queue and checked its completion before clearing active, covering the tap writes.
        std.debug.assert(r.active == null);
        if (r.active != null) return error.RoundNotReady;
        try r.taps.complete();
        try q.tap_gather.gather(r.taps.buffer.contents()[0..r.taps.buffer.length()], r.taps.capacity, r.taps.rows, r.taps.width, rows[0..n], out.contents()[0..out.length()]);
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

    /// A verify whose rows all stayed keeps them before anything else (keep is called only to drop rows).
    pub fn settle(b: *Metal) !void {
        try b.settleRound();
        if (b.draft) |d| try d.generation.settle();
    }

    /// The next drafter may overlap a submitted keep; every target-state user calls settle instead.
    pub fn settleRound(b: *Metal) !void {
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
        slots.promoteDue(b);
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
        try b.settle();
        slots.promoteDue(b);
        const slot = try b.slotFor(s);
        const token = try b.value(feed);
        const before = b.runner.gpu_seconds;
        if (b.drafter(slot)) |d| try d.advance(token) else {
            try (q.session.Session{ .runner = b.runner, .slot = slot }).step(token);
            if (b.ringOf(slot)) |g| try b.ringChunk(g, 1);
        }
        slots.charge(b, s, before);
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
        slots.promoteDue(b);
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
            if (rows > b.window or total + rows > batch_rows) return error.WindowTooWide;
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
        b.rows = total;
        if (b.pass_profile and windows.len == 1) std.log.info("q27_offer width={d} rows={d} tree_nodes={d} copy_rows={d}", .{ b.window, total, b.from_tree, total - 1 - b.from_tree });
        const fused = windows.len == 1 and b.windows[0] == 0 and b.draft != null and total > 1 and windows[0].stream.sampling == null and windows[0].masks.len == 0;
        std.debug.assert(r.tail == null);
        if (fused) r.tail = .{ .ptr = b, .encode = @import("qwen27_keep.zig").encodeTail };
        defer r.tail = null;
        const before = r.gpu_seconds;
        if (plain) try r.advance(&b.round.?, .all) else try r.verifyHead(&b.round.?, .all);
        for (windows) |w| slots.charge(b, w.stream, before); // a shared round counts whole in each reply
        b.advanced = plain;
        b.rows = total;
        // every greedy window's rows from one argmax over the round; a sampled window draws on the host
        var greedy = false;
        for (windows) |w| greedy = greedy or (w.stream.sampling == null and w.masks.len == 0);
        if (fused) try @import("qwen27_keep.zig").finishPrefetch(b) else if (greedy) try b.argmaxRows(0, total);
        for (windows, out, b.round.?.firsts) |w, o, first| {
            const rows = w.rows();
            if (w.stream.sampling != null or w.masks.len > 0) try b.draw(w.stream, first, rows, w.positions, o.sampled, w.masks) else try b.picked(first, rows, o.sampled);
            @memcpy(o.drafts, b.ids[first + 1 ..][0 .. rows - 1]);
        }
        if (windows.len == 1) @memcpy(b.sampled[0..b.rows], out[0].sampled);
    }

    /// How many drafts are the held tree's first nodes with their own parents (0: copies, fills or a chain).
    pub fn treeRows(b: *const Metal, w: be.Window) usize {
        const t = b.trees[0] orelse return 0;
        if (w.held > 0) return 0;
        const n = @min(w.tokens.len, t.tokens.len);
        var prefix: usize = 0;
        for (w.tokens[0..n], t.tokens[0..n], 0..) |token, draft, i| {
            const parent = if (w.parents) |ps| ps[i + 1] - 1 else @as(i32, @intCast(i)) - 1;
            if (token != draft or parent != t.parents[i]) break;
            prefix += 1;
        }
        return prefix;
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
