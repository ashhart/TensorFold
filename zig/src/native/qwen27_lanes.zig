//! The lane core's backend for Qwen27: one stream's DFlash2 trees verified, kept and drafted on the shared round loop.
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
const contract = tf.tree_round;

/// A window's rows at most: the pending row and a DFlash2 tree.
pub const max_rows = 16;
const ring = 64;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

pub const Metal = struct {
    gpa: std.mem.Allocator,
    runner: *q.decode_round.Runner,
    draft: ?*Draft,
    chunk: usize = 128, // prompt rows a pass
    nodes: u32 = max_rows - 1, // a tree's drafts at most
    stream: ?*lanes.Stream = null, // the one stream the runner's slot holds
    drawn: [ring]u32 = undefined, // drawn tokens by handle
    next: u64 = 0,
    round: ?Round = null, // the last verify, until its rows are kept
    ids: [max_rows]u32 = undefined,
    parents: [max_rows]i32 = undefined,
    rows: usize = 0,
    tree: ?Tree = null, // the drafts held for the stream's next round
    ops: tf.tree_round_gpu.Ops,
    picks: mtl.Buffer,
    costs: [max_rows]lanes.config.Cost = undefined, // a window's ms by rows, timed by measure()
    timed: usize = 0,
    from_tree: usize = 0, // the last window's drafts that were the held tree's first nodes
    sampled: [max_rows]u32 = undefined, // the last window's draws
    gathered: usize = 0, // kept rows whose taps the last keep gathered for the drafter
    landing: [16][2]f64 = @splat(.{ 0, 0 }), // by a node's own log-probability bucket: offered with its parent kept, landed

    pub fn init(gpa: std.mem.Allocator, runner: *q.decode_round.Runner, draft: ?*Draft) !*Metal {
        if (runner.slots != 1) return error.TargetBinding;
        const ops = try tf.tree_round_gpu.Ops.init(runner.model.device);
        errdefer ops.deinit();
        const picks = try runner.model.device.buffer(max_rows * @sizeOf(contract.Pick), opts);
        errdefer picks.deinit();
        const b = try gpa.create(Metal);
        b.* = .{ .gpa = gpa, .runner = runner, .draft = draft, .ops = ops, .picks = picks };
        return b;
    }

    pub fn deinit(b: *Metal) void {
        b.dropRound();
        b.dropTree();
        b.picks.deinit();
        b.ops.deinit();
        b.gpa.destroy(b);
    }

    pub fn backend(b: *Metal) be.Backend {
        return .{ .ptr = b, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .probabilities = probabilitiesFn, .tree = treeFn, .release = releaseFn } };
    }

    /// What the round loop reads at setup: every round offers the drafter's whole tree (no timed costs yet).
    pub fn facts(b: *const Metal) lanes.Model {
        const drafting = b.draft != null;
        return .{ .exact_width = if (drafting) max_rows else 1, .mtp = drafting, .speculate = drafting, .speculate_early = false, .drafts = b.nodes, .draft_probabilities = true, .window_costs = b.costs[0..b.timed], .batch_rows = max_rows, .max_streams = 1 };
    }

    /// Take the stream whose prompt another driver prefilled into the runner and drafter (the serial host).
    pub fn attach(b: *Metal, s: *lanes.Stream) !void {
        if (b.stream != null) return error.StreamBusy;
        if (b.runner.offsets[0] != s.prompt_len) return error.CachePositionDiffers;
        b.dropRound();
        b.dropTree();
        b.stream = s;
    }

    /// Each window width's ms (verify, draws, keep) on a scratch context; resets the slot. Kept per build.
    pub fn measure(b: *Metal, io: std.Io) !void {
        const r = b.runner;
        if (b.stream != null or r.active != null) return error.StreamBusy;
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
        try (q.session.Session{ .runner = r }).prefill(&ids, 128);
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
        b.stream = s;
        defer b.stream = null;
        const w = be.Window{ .stream = s, .pending = 0, .held = 0, .tokens = drafts[0 .. rows - 1], .parents = null, .positions = positions[0..rows] };
        var out = [_]be.Verified{.{ .sampled = sampled[0..rows], .drafts = echoed[0 .. rows - 1] }};
        try verifyFn(b, &.{w}, &out);
        try b.commitPath(&.{0});
        return (mtl.clock.seconds() - t0) * 1e3;
    }

    fn self(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
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
    }

    fn dropTree(b: *Metal) void {
        if (b.tree) |*t| t.deinit();
        b.tree = null;
    }

    /// Each of the first `rows` logits rows drawn at its position: argmax on the GPU, or the stream's keyed sampler.
    fn draw(b: *Metal, s: *const lanes.Stream, rows: usize, positions: []const u64, out: []u32) !void {
        const logits = b.runner.model.frame.get(.logits);
        const vocab: u32 = @intCast(b.runner.model.config.vocab);
        if (s.sampling) |sampling| {
            const words: [*]const u16 = @ptrCast(@alignCast(logits.buffer.contents() + logits.offset));
            for (out[0..rows], positions[0..rows], 0..) |*t, p, r| t.* = try api.serial_host.choose(b.gpa, .{ .bf16 = words[r * vocab ..][0..vocab] }, p, sampling);
            return;
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = b.runner.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        try b.ops.argmax(e, .{ .buf = logits.buffer, .off = logits.offset }, .{ .buf = b.picks }, .{ .rows = @intCast(rows), .vocab = vocab, .stride = vocab });
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

    /// Keep the verified rows on `path`; the last kept row's logits become row 0, where a one-token step leaves them.
    fn commitPath(b: *Metal, path: []const u32) !void {
        defer b.dropRound();
        b.observe(path);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        if (b.draft) |d| return b.commitDevice(d, path);
        try b.runner.keep(&b.round.?, &.{path});
        const last = path[path.len - 1];
        if (last == 0) return;
        const logits = b.runner.model.frame.get(.logits);
        const bytes = b.runner.model.config.vocab * 2;
        const base = logits.buffer.contents() + logits.offset;
        @memcpy(base[0..bytes], base[@as(usize, last) * bytes ..][0..bytes]);
    }

    /// One command buffer keeps the path's cache rows, gathers its taps and moves its last logits to row 0.
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
        b.gathered = path.len;
    }

    /// A verify whose rows all stayed keeps them before anything else (keep is called only to drop rows).
    fn settle(b: *Metal) !void {
        if (b.round == null) return;
        var path: [max_rows]u32 = undefined;
        for (path[0..b.rows], 0..) |*p, i| p.* = @intCast(i);
        try b.commitPath(path[0..b.rows]);
    }

    /// The drafter reads the target's taps of the kept rows, which the keep gathered in path order.
    fn absorb(b: *Metal, d: *Draft, path: []const u32) !void {
        if (b.gathered != path.len) return error.TapsNotGathered;
        b.gathered = 0;
        try d.generation.draft.absorb(.{ .buf = d.generation.taps }, @intCast(path.len));
    }

    // -- the vtable ---------------------------------------------------------------------------------------------

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const b = self(ptr);
        if (b.stream) |o| if (o != s) return error.StreamBusy;
        const ids = s.prompt();
        if (ids.len == 0 or ids.len + s.max_new + max_rows > b.runner.capacity) return error.PromptTooLong;
        b.stream = s;
        b.settle() catch {};
        b.dropRound();
        b.dropTree();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        if (b.draft) |d| try d.reset() else try b.runner.reset(0);
        var at: usize = 0;
        while (at < ids.len) {
            if (s.isCancelled()) return error.Cancelled;
            const n = @min(b.chunk, ids.len - at);
            const part = ids[at..][0..n];
            if (b.draft) |d| try d.promptChunk(part, at + n == ids.len) else try (q.session.Session{ .runner = b.runner }).promptChunk(part, at + n == ids.len);
            at += n;
        }
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const b = self(ptr);
        if (b.stream != s) return error.UnknownStream;
        var token: [1]u32 = undefined;
        try b.draw(s, 1, &.{position}, &token);
        return b.push(token[0]);
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const b = self(ptr);
        if (b.stream != s) return error.UnknownStream;
        try b.settle();
        const token = try b.value(feed);
        if (b.draft) |d| try d.advance(token) else try (q.session.Session{ .runner = b.runner }).step(token);
        var next: [1]u32 = undefined;
        try b.draw(s, 1, &.{position}, &next);
        return b.push(next[0]);
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return self(ptr).value(.{ .handle = handle });
    }

    /// The stream's window (pending token, then its drafts as a chain or tree) in one forward, every row drawn.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = self(ptr);
        if (windows.len != 1) return error.SharedRoundsNotBuilt;
        const w = windows[0];
        if (b.stream != w.stream) return error.UnknownStream;
        if (w.held > 0 and w.tokens.len > 0) return error.HeldAndHostDrafts;
        try b.settle();
        const rows = w.rows();
        if (rows > max_rows) return error.WindowTooWide;
        b.ids[0] = w.pending;
        if (w.held > 0) b.heldChain(b.ids[1..rows]) else @memcpy(b.ids[1..rows], w.tokens);
        b.from_tree = b.treeRows(w);
        if (w.parents) |p| @memcpy(b.parents[0..rows], p) else for (b.parents[0..rows], 0..) |*p, r| {
            p.* = @as(i32, @intCast(r)) - 1;
        }
        const r = b.runner;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        b.round = try Round.init(b.gpa, &.{.{ .slot = 0, .start = r.offsets[0], .capacity = r.capacity, .ids = b.ids[0..rows], .parents = b.parents[0..rows] }}, @intCast(r.model.config.conv_kernel), r.slots, @intCast(r.model.config.vocab));
        errdefer b.dropRound();
        try r.verifyHead(&b.round.?, .all);
        b.rows = rows;
        try b.draw(w.stream, rows, w.positions, out[0].sampled);
        @memcpy(b.sampled[0..rows], out[0].sampled);
        @memcpy(out[0].drafts, b.ids[1..rows]);
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
    fn heldChain(b: *const Metal, out: []u32) void {
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
        const fill = if (n > 0) out[n - 1] else b.ids[0];
        for (out[n..]) |*x| x.* = fill;
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = self(ptr);
        if (windows.len != 1 or b.round == null or paths[0].len == 0 or paths[0][0] != 0) return error.NothingToKeep;
        try b.commitPath(paths[0]);
    }

    /// The drafter absorbs the kept rows, then proposes the next round's tree from the stream's pending token.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const b = self(ptr);
        if (requests.len != 1) return error.SharedRoundsNotBuilt;
        const r = requests[0];
        if (b.stream != r.stream) return error.UnknownStream;
        const d = b.draft orelse return error.NoDraftHead;
        try b.settle();
        b.dropTree();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        if (r.rows) |rows| try b.absorb(d, rows);
        const g = &d.generation;
        if (g.draft.committed_end != b.runner.offsets[0]) return error.BadDraftPosition;
        const room = b.runner.capacity - b.runner.offsets[0];
        if (r.depth == 0 or room < 2) return;
        const pending = if (r.rows != null) r.follow[r.follow.len - 1] else try b.value(r.first orelse return error.NoFirstToken);
        const nodes: u32 = @intCast(@min(@min(r.depth, b.nodes), room - 1));
        b.tree = if (g.proposer) |p| try p.call(p.ptr, pending, nodes) else try g.draft.propose(pending, nodes);
    }

    /// Each held node's chance of landing: its own given its parent (e^own as the prior) times its parent's.
    fn probabilitiesFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
        const b = self(ptr);
        if (b.stream != s) return false;
        const t = b.tree orelse return false;
        if (out.len > t.scores.len) return false;
        for (out, t.parents[0..out.len], 0..) |*p, parent, i| {
            const own = local(t, i);
            const x = b.landing[bucket(own)];
            p.* = (x[1] + 2 * @exp(own)) / (x[0] + 2) * (if (parent >= 0) out[@intCast(parent)] else 1);
        }
        return true;
    }

    fn treeFn(ptr: *anyopaque, s: *lanes.Stream, gpa: std.mem.Allocator) anyerror!?lanes.stream.Held {
        const b = self(ptr);
        if (b.stream != s) return null;
        const t = b.tree orelse return null;
        const tokens = try gpa.dupe(u32, t.tokens);
        errdefer gpa.free(tokens);
        return .{ .count = @intCast(t.tokens.len), .tokens = tokens, .parents = try gpa.dupe(i32, t.parents) };
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const b = self(ptr);
        if (b.stream != s) return;
        b.settle() catch {};
        b.dropRound();
        b.dropTree();
        b.stream = null;
    }
};
