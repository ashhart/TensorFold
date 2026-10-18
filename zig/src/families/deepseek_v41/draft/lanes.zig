//! DeepSeek-V4.1's lane backend over a `Target` and a DSpark `Pass`, the DSpark depth as the round's row policy.
const std = @import("std");
const Allocator = std.mem.Allocator;
const lanes = @import("lanes");
const be = lanes.backend;
const Stream = lanes.Stream;
const iface = @import("iface.zig");
const dspark = @import("dspark.zig");
const tree = @import("tree.zig");
const depth = @import("depth.zig");
const calib = @import("calib.zig");
const costs_mod = @import("costs.zig");
const Policy = @import("policy.zig").Policy;
const branches = @import("branches.zig");
const spec_mod = @import("spec.zig");

pub const max_block = 16;

pub const Options = struct {
    shape: dspark.Shape,
    depth: depth.Settings = .{},
    siblings: u32 = 0, // first-position siblings (TF_DSV41_TREE - 1; 0: chains only)
    sib_rows: u32 = 4, // a sibling branch's rows at most (TF_DSV41_TREE_ROWS)
    dup: tree.Dup = .{}, // sibling pricing: TF_DSV41_TREE_DUP_MS a sibling, its pending row counted
    slots: u32 = 4, // streams at once
    spec: spec_mod.Settings = .{}, // the speculative pass (TF_DSV41_SPEC_DRAFT / _SPEC_NUCLEUS)
};

/// A stream's slot and what the backend holds for it between rounds.
pub const St = struct {
    slot: u32,
    len: u64 = 0, // rows the target holds (provisional after a window until keep)
    base: u64 = 0, // the last window's start
    first: u32 = 0, // the prompt's choice (for `first`)
    held: [max_block]u32 = undefined, // the last pass's drafts (the main chain)
    conf: [max_block]f64 = undefined,
    n: u32 = 0, // drafts held (0: none this round)
    start: u64 = 0, // the pending token's position when they were drafted
    sib: [tree.max_siblings]tree.Sibling = undefined,
    sib_tokens: [tree.max_siblings][max_block]u32 = undefined, // each sibling's root then its continuation
    sib_len: [tree.max_siblings]u32 = @splat(0),
    n_sib: u32 = 0,
    shown: std.ArrayList(u32) = .empty, // the drafts this round's window verifies (node indices, window order)
    shown_set: bool = false,

    /// Node i of the held tree: the main chain's drafts, then each sibling's rows.
    pub fn nodes(st: *const St) u32 {
        var n = st.n;
        for (st.sib_len[0..st.n_sib]) |l| n += l;
        return n;
    }
};

pub const Lanes = struct {
    gpa: Allocator,
    target: iface.Target,
    pass: iface.Pass,
    opts: Options,
    streams: std.AutoHashMapUnmanaged(*Stream, *St) = .empty,
    free: std.ArrayList(u32) = .empty,
    drawn: std.ArrayList(u32) = .empty, // handle -> token
    policy: Policy,
    scratch: std.heap.ArenaAllocator,
    // a pass's buffers, sized for opts.slots
    drafts: []u32,
    conf: []f32,
    cand: []i32,
    base: []f32,
    /// the speculative pass (null: off)
    spec: ?spec_mod.Spec = null,
    /// each slot's stream state (null: free)
    by_slot: []?*St,
    retired: [16]bool = @splat(false),
    release_requested: [16]bool = @splat(false),

    pub const init = @import("lane_owner.zig").init;
    pub const deinit = @import("lane_owner.zig").deinit;

    pub fn drop(x: *Lanes, st: *St) void {
        st.shown.deinit(x.gpa);
        x.gpa.destroy(st);
    }

    pub fn backend(x: *Lanes) be.Backend {
        const pieces = x.target.pieces();
        return .{ .ptr = x, .vtable = if (pieces) &vtable_pieces else &vtable };
    }

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .probabilities = probabilities, .tree = heldTree, .release = release };
    /// a target that runs prompts in pieces (iface.Target begin / piece / finish): the round planner may use them
    const vtable_pieces: be.Backend.VTable = blk: {
        var v = vtable;
        v.begin = begin;
        v.piece = piece;
        v.finish = finish;
        v.join = join;
        v.pieces = roundPieces;
        v.finals = roundFinals;
        break :blk v;
    };

    /// The speculative pass's counts on stderr when it is on (the gates' and the server's line).
    pub fn reportSpec(x: *const Lanes) void {
        const sp = x.spec orelse return;
        var buf: [256]u8 = undefined;
        std.debug.print("dsv41 spec: {s}\n", .{sp.describe(&buf)});
    }

    /// Put the DSpark depth in charge of the engine's rows.
    pub fn attach(x: *Lanes, e: *lanes.Engine) void {
        e.trim = x.policy.rows();
    }

    /// What the engine's Config reads of this model: 16 rows a slot, `batch_rows` a shared forward, the table's costs.
    pub fn model(x: *Lanes, arena: Allocator, batch_rows: u32) !lanes.Model {
        x.policy.depth.forward_rows = batch_rows;
        const v = x.policy.depth.costs.verify;
        const window = try arena.alloc(lanes.config.Cost, @min(v.len, costs_mod.slot_rows));
        for (window, 0..) |*w, i| w.* = .{ .width = @intCast(i + 1), .ms = v[i] };
        const shared = try arena.alloc(lanes.config.Cost, @min(v.len, batch_rows));
        for (shared, 0..) |*w, i| w.* = .{ .width = @intCast(i + 1), .ms = v[i] };
        return .{ .exact_width = costs_mod.slot_rows, .mtp = true, .speculate = true, .speculate_early = false, .drafts = x.opts.shape.block, .window_costs = window, .mtp_step_ms = 0.0, .hidden_rows = true, .batch_rows = batch_rows, .max_streams = x.opts.slots, .shared_costs = shared, .draft_probabilities = true, .draft_streams = true, .join_tail = x.target.vtable.tail != null and joinTail(), .piece_runs = envOn("TF_DSV41_PIECE_RUNS"), .join_drafts = envOn("TF_DSV41_JOIN_DRAFTS"), .replay_runs = envOn("TF_DSV41_REPLAY_RUNS") };
    }

    /// TF_DSV41_TAIL_JOIN=1: a prompt in pieces leaves its last token to its first round's shared window (default off).
    fn joinTail() bool {
        return envOn("TF_DSV41_TAIL_JOIN");
    }

    /// TF_DSV41_PIECE_RUNS=1: a round's prompt pieces of several slots in one forward (default off).
    fn envOn(name: [*:0]const u8) bool {
        const v = std.c.getenv(name) orelse return false;
        return std.mem.eql(u8, std.mem.span(v), "1");
    }

    pub fn state(x: *Lanes, s: *Stream) ?*St {
        return x.streams.get(s);
    }

    fn self(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn get(x: *Lanes, s: *Stream) !*St {
        return x.streams.get(s) orelse error.UnknownStream;
    }

    fn value(x: *Lanes, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| x.drawn.items[h],
            .value => |v| v,
        };
    }

    fn ingest(x: *Lanes, st: *St, start: u64, rows: []const u32) !void {
        const span = iface.ingestSpan(rows.len, x.opts.shape.window);
        if (span.count == 0) return;
        const taps = x.target.taps(st.slot);
        const m = taps.map orelse return x.pass.ingest(st.slot, start + span.skip, taps, rows[span.skip..]);
        // a tree resolved as chains: the path's rows in the taps of the chain that holds it
        var mapped: [branches.max_rows]u32 = undefined;
        if (span.count > mapped.len) return error.TreeRows;
        for (rows[span.skip..], mapped[0..span.count]) |r, *o| {
            if (r >= m.len or m[r] == branches.unrun) return error.TreeTaps;
            o.* = m[r];
        }
        try x.pass.ingest(st.slot, start + span.skip, taps, mapped[0..span.count]);
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        if (x.streams.get(s) != null) return error.StreamLive;
        @import("lane_retire.zig").recover(x);
        const slot = x.free.pop() orelse return error.NoSlot;
        const st = x.gpa.create(St) catch |e| {
            x.free.appendAssumeCapacity(slot);
            return e;
        };
        st.* = .{ .slot = slot };
        x.by_slot[slot] = st;
        errdefer {
            x.retired[slot] = true;
            _ = @import("lane_retire.zig").retire(x, st);
        }
        try x.settle();
        if (x.spec) |*sp| sp.forget(slot);
        x.pass.reset(slot);
        x.policy.depth.reset(slot);
        const ids = s.prompt();
        x.target.admit(slot, s.max_new);
        st.first = try x.target.prefill(slot, ids, s.sampling, ids.len);
        st.len = ids.len;
        _ = x.scratch.reset(.retain_capacity);
        // the taps hold the prompt's last rows (after a verify-tail prompt only that row), at their positions
        const tr: usize = @min(x.target.taps(slot).rows, ids.len);
        const rows = try x.scratch.allocator().alloc(u32, tr);
        for (rows, 0..) |*r, i| r.* = @intCast(i);
        try x.ingest(st, ids.len - tr, rows);
        try x.streams.put(x.gpa, s, st);
        x.by_slot[slot] = st;
    }

    /// A prompt in pieces (lane_host.zig `Rounds`): new slot, drafter context and depth reset, target admits/restores.
    fn begin(ptr: *anyopaque, s: *Stream) anyerror!be.Backend.Begun {
        const x = self(ptr);
        if (x.streams.get(s) != null) return error.StreamLive;
        @import("lane_retire.zig").recover(x);
        const slot = x.free.pop() orelse return error.NoSlot;
        const st = x.gpa.create(St) catch |e| {
            x.free.appendAssumeCapacity(slot);
            return e;
        };
        st.* = .{ .slot = slot };
        x.by_slot[slot] = st;
        errdefer {
            x.retired[slot] = true;
            _ = @import("lane_retire.zig").retire(x, st);
        }
        try x.settle();
        if (x.spec) |*sp| sp.forget(slot);
        x.pass.reset(slot);
        x.policy.depth.reset(slot);
        x.target.admit(slot, s.max_new);
        const b = try x.target.begin(slot, s.prompt());
        try x.streams.put(x.gpa, s, st);
        x.by_slot[slot] = st;
        return .{ .cached = b.at, .damaged = b.damaged };
    }

    fn piece(ptr: *anyopaque, s: *Stream, start: u64, end: u64, save: bool) anyerror!void {
        const x = self(ptr);
        const st = try x.get(s);
        try x.settle();
        return x.target.piece(st.slot, s.prompt(), start, end, save);
    }

    /// A round's pieces of several streams: the target's (one forward over their slots' rows where it can).
    fn roundPieces(ptr: *anyopaque, list: []const be.Backend.Piece) anyerror!void {
        const x = self(ptr);
        try x.settle();
        var ps: [16]iface.Target.Piece = undefined;
        if (list.len > ps.len) return error.TooManySlots;
        for (list, ps[0..list.len]) |p, *q| {
            const st = try x.get(p.stream);
            q.* = .{ .slot = st.slot, .ids = p.stream.prompt(), .start = p.start, .end = p.end, .save = p.save };
        }
        return x.target.runPieces(ps[0..list.len]);
    }

    /// A round's finished prompts before their finishes: the target's shared work (their decoder replays in one run).
    fn roundFinals(ptr: *anyopaque, streams: []const *Stream) anyerror!void {
        const x = self(ptr);
        try x.settle();
        var fs: [16]iface.Target.Final = undefined;
        if (streams.len > fs.len) return error.TooManySlots;
        for (streams, fs[0..streams.len]) |s, *q| q.* = .{ .slot = (try x.get(s)).slot, .ids = s.prompt() };
        return x.target.runReplays(fs[0..streams.len]);
    }

    /// The prompt's rows are in: its last token's window (the first choice), its taps to the drafter as `prefill`'s.
    fn finish(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        const st = try x.get(s);
        try x.settle();
        const ids = s.prompt();
        return x.finished(st, ids, try x.target.finish(st.slot, ids, s.sampling, ids.len));
    }

    /// `finish` without the last token's window (TF_DSV41_TAIL_JOIN); true when the target ran the row anyway.
    fn join(ptr: *anyopaque, s: *Stream) anyerror!bool {
        const x = self(ptr);
        const st = try x.get(s);
        try x.settle();
        const ids = s.prompt();
        if (try x.target.tail(st.slot, ids, s.sampling, ids.len)) |first_| {
            try x.finished(st, ids, first_);
            return false;
        }
        st.len = ids.len - 1;
        st.base = st.len;
        return true;
    }

    fn finished(x: *Lanes, st: *St, ids: []const u32, first_: u32) !void {
        st.first = first_;
        st.len = ids.len;
        _ = x.scratch.reset(.retain_capacity);
        const tr: usize = @min(x.target.taps(st.slot).rows, ids.len);
        const rows = try x.scratch.allocator().alloc(u32, tr);
        for (rows, 0..) |*r, i| r.* = @intCast(i);
        try x.ingest(st, ids.len - tr, rows);
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const x = self(ptr);
        const st = try x.get(s);
        if (position != st.len) return error.PositionMismatch;
        try x.drawn.append(x.gpa, st.first);
        return x.drawn.items.len - 1;
    }

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const x = self(ptr);
        const st = try x.get(s);
        if (position != st.len + 1) return error.PositionMismatch;
        const token = [_]u32{x.value(feed)};
        try x.settle();
        if (x.spec) |*sp| sp.clear(); // an undrafted window: no speculation
        var choice: [1]u32 = undefined;
        var out = [_][]u32{&choice};
        try x.target.window(&.{.{ .slot = st.slot, .start = st.len, .tokens = &token, .parents = null, .draws = &.{position}, .sampling = s.sampling }}, &out);
        st.base = st.len;
        st.len += 1;
        try x.ingest(st, st.base, &.{0});
        try x.drawn.append(x.gpa, choice[0]);
        return x.drawn.items.len - 1;
    }

    fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return self(ptr).drawn.items[handle];
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        try x.settle();
        _ = x.scratch.reset(.retain_capacity);
        const a = x.scratch.allocator();
        const segs = try a.alloc(iface.Segment, windows.len);
        const choices = try a.alloc([]u32, windows.len);
        for (windows, segs, choices, out) |w, *g, *c, o| {
            const st = try x.get(w.stream);
            if (w.positions[0] != st.len + 1) return error.PositionMismatch;
            const tokens = try a.alloc(u32, w.rows());
            tokens[0] = w.pending;
            if (w.held > st.n) return error.HeldMismatch;
            @memcpy(tokens[1 .. 1 + w.held], st.held[0..w.held]);
            @memcpy(tokens[1 + w.held ..], w.tokens);
            const parents = w.parents; // each row's parent row (row 0: -1)
            g.* = .{ .slot = st.slot, .start = st.len, .tokens = tokens, .parents = parents, .draws = w.positions, .sampling = w.stream.sampling };
            c.* = o.sampled;
            @memcpy(o.drafts, tokens[1..]);
        }
        if (x.spec != null) x.armSpec(windows, segs);
        try x.target.window(segs, choices);
        for (windows) |w| {
            const st = try x.get(w.stream);
            st.base = st.len;
            st.len += w.rows();
        }
        if (x.spec != null) try x.speculate(a, windows, segs, choices);
    }

    /// Before the window: arm the slots `speculate` would launch, so passes start from the device pick (Pass.arm).
    fn armSpec(x: *Lanes, windows: []const be.Window, segs: []const iface.Segment) void {
        const arm = x.pass.vtable.arm orelse return;
        const sp = &x.spec.?;
        var tree_round = false;
        var nucleus = false;
        for (windows) |w| {
            tree_round = tree_round or w.parents != null;
            nucleus = nucleus or spec_mod.isNucleus(w.stream.sampling);
        }
        if (!sp.roundAllowed(tree_round, nucleus)) return;
        for (windows, segs) |w, g| {
            const st = x.get(w.stream) catch continue;
            if (!dsparkRound(st, w) or !sp.wouldWant(st.slot)) continue;
            // keyed rows: the sampler's device choice (vsample.choose); a nucleus row's statistics stay on the host
            const sampled = if (w.stream.sampling) |smp| smp.temperature > 0 else false;
            if (sampled and spec_mod.isNucleus(w.stream.sampling)) continue;
            arm(x.pass.ptr, st.slot, g.start, @intCast(g.tokens.len), sampled);
        }
    }

    /// A DSpark round: the window verifies the pass's drafts only (a prefix of the held chain counts too).
    fn dsparkRound(st: *const St, w: be.Window) bool {
        if (st.n == 0) return false;
        if (w.tokens.len == 0) return true;
        return w.held == 0 and w.parents == null and w.tokens.len <= st.n and std.mem.eql(u32, w.tokens, st.held[0..w.tokens.len]);
    }

    /// Collects a speculative pass still running (before any other pass, target or ingest work).
    pub fn settle(x: *Lanes) !void {
        if (x.spec) |*sp| try sp.settle(x.pass);
    }

    /// After a window: the next round's pass for its DSpark windows (spec.zig), their kept rows ingested first.
    fn speculate(x: *Lanes, a: Allocator, windows: []const be.Window, segs: []const iface.Segment, choices: []const []u32) !void {
        const sp = &x.spec.?;
        var tree_round = false;
        var nucleus = false;
        for (windows) |w| {
            tree_round = tree_round or w.parents != null;
            nucleus = nucleus or spec_mod.isNucleus(w.stream.sampling);
        }
        if (!sp.roundAllowed(tree_round, nucleus)) return sp.clear();
        const cands = try a.alloc(spec_mod.Candidate, windows.len);
        var n: usize = 0;
        for (windows, segs, choices) |w, g, c| {
            const st = try x.get(w.stream);
            // a DSpark round only (dsparkRound counts chain rounds read back as host tokens)
            if (!dsparkRound(st, w)) continue;
            var acc: u32 = 0;
            while (acc + 1 < g.tokens.len and g.tokens[acc + 1] == c[acc]) acc += 1;
            cands[n] = .{ .slot = st.slot, .start = g.start, .accepted = acc, .bonus = c[acc], .sampling = g.sampling };
            n += 1;
        }
        const asks = sp.launch(cands[0..n]);
        if (asks.len == 0) return;
        const rows = try a.alloc(u32, lanes_max_rows);
        for (rows, 0..) |*r, i| r.* = @intCast(i);
        for (asks) |ask| {
            const st = x.by_slot[ask.slot] orelse return error.UnknownStream;
            try x.ingest(st, st.base, rows[0 .. ask.start - st.base]);
        }
        try sp.run(x.pass, x.opts.siblings > 0);
    }

    const lanes_max_rows = 64;

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(ptr);
        for (windows, paths) |w, path| {
            const st = try x.get(w.stream);
            try x.target.keep(st.slot, path);
            st.len = st.base + path.len;
        }
    }

    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const x = self(ptr);
        try x.settle();
        _ = x.scratch.reset(.retain_capacity);
        const a = x.scratch.allocator();
        const asks = try a.alloc(iface.Ask, requests.len);
        const whose = try a.alloc(*St, requests.len);
        const depths = try a.alloc(u32, requests.len);
        const row0 = try a.alloc(iface.Warm, requests.len);
        const row0_ids = try a.alloc(u32, requests.len);
        var n_row0: usize = 0;
        var n: usize = 0;
        for (requests) |r| {
            const st = try x.get(r.stream);
            var at = st.len;
            if (r.rows) |path| {
                if (r.start != st.base) return error.PositionMismatch;
                const done = if (x.spec) |*sp| sp.ingested(st.slot, r.start, path) else false;
                if (!done) try x.ingest(st, r.start, path); // a shared round drafts before its keep: the path, not st.len
                at = r.start + path.len;
            }
            if (r.position != at + 1) return error.PositionMismatch;
            const anchor = if (r.first) |f| x.value(f) else r.follow[r.follow.len - 1];
            // commit warm: the next window's pending row, before the pass
            if (x.target.vtable.warm != null and n_row0 < row0.len) {
                row0_ids[n_row0] = anchor;
                row0[n_row0] = .{ .slot = st.slot, .start = at, .ids = row0_ids[n_row0..][0..1] };
                n_row0 += 1;
            }
            st.n = 0;
            st.n_sib = 0;
            st.start = at;
            if (r.depth == 0) continue;
            if (n == x.opts.slots) return error.TooManySlots;
            asks[n] = .{ .slot = st.slot, .anchor = anchor, .start = at, .params = dspark.Params.of(r.stream.sampling, at) };
            whose[n] = st;
            depths[n] = r.depth;
            n += 1;
        }
        x.target.warm(row0[0..n_row0]);
        if (n == 0) return;
        const b = x.opts.shape.block;
        const c = x.opts.shape.candidates;
        const markov = if (x.opts.siblings > 0) x.pass.markov() else null;
        const props = try a.alloc(iface.Proposal, n);
        for (props, 0..) |*p, i| p.* = .{
            .drafts = x.drafts[i * b ..][0..b],
            .conf = x.conf[i * b ..][0..b],
            .cand = if (markov != null) x.cand[i * b * c ..][0 .. b * c] else null,
            .base = if (markov != null) x.base[i * b * c ..][0 .. b * c] else null,
        };
        const took = if (x.spec) |*sp| sp.take(asks[0..n], props) else false;
        if (!took) try x.pass.propose(asks[0..n], props);
        // draft-end warm: the pending row and every drafted one, before the depth cuts them
        if (x.target.vtable.warm != null) {
            const ws = try a.alloc(iface.Warm, n);
            for (ws, props, asks[0..n]) |*w, p, ask| {
                const ids = try a.alloc(u32, 1 + b);
                ids[0] = ask.anchor;
                @memcpy(ids[1..], p.drafts[0..b]);
                w.* = .{ .slot = ask.slot, .start = ask.start, .ids = ids };
            }
            x.target.warm(ws);
        }
        for (props, whose[0..n], depths[0..n], asks[0..n]) |p, st, d, ask| {
            st.n = @min(d, b);
            @memcpy(st.held[0..st.n], p.drafts[0..st.n]);
            for (st.conf[0..st.n], p.conf[0..st.n]) |*q, f| q.* = f;
            if (markov) |m| try x.siblings(a, st, m, p, ask);
        }
    }

    /// The siblings after the pending token: the draft distribution's next first tokens and their Markov chains.
    fn siblings(x: *Lanes, a: Allocator, st: *St, m: dspark.Markov, p: iface.Proposal, ask: iface.Ask) !void {
        const c = x.opts.shape.candidates;
        const z0 = try a.alloc(f32, c);
        dspark.markovZ(m, p.cand.?[0..c], p.base.?[0..c], ask.anchor, z0);
        const smp = ask.params.sampling();
        const got = try tree.ranked(a, p.cand.?[0..c], z0, ask.start + 1, smp, st.held[0..1], st.sib[0..x.opts.siblings]);
        const rows = @min(x.opts.sib_rows, st.n);
        for (st.sib[0..got], 0..) |sib, j| {
            st.sib_tokens[j][0] = sib.token;
            const more = try tree.continuation(a, m, p.cand.?, p.base.?, c, sib.token, rows, ask.params, st.sib_tokens[j][1..]);
            st.sib_len[j] = @intCast(1 + more);
        }
        st.n_sib = @intCast(got);
    }

    /// The held drafts as host tokens: the main chain, then each sibling's root and its chain (gpa-owned).
    fn heldTree(ptr: *anyopaque, s: *Stream, gpa: Allocator) anyerror!?lanes.stream.Held {
        const x = self(ptr);
        const st = try x.get(s);
        if (st.n == 0) return null;
        const total = st.nodes();
        const tokens = try gpa.alloc(u32, total);
        errdefer gpa.free(tokens);
        @memcpy(tokens[0..st.n], st.held[0..st.n]);
        if (st.n_sib == 0) return .{ .count = total, .tokens = tokens };
        const parents = try gpa.alloc(i32, total);
        for (parents[0..st.n], 0..) |*q, i| q.* = @as(i32, @intCast(i)) - 1;
        var at: usize = st.n;
        for (st.sib_tokens[0..st.n_sib], st.sib_len[0..st.n_sib]) |toks, len| {
            for (0..len) |i| {
                tokens[at + i] = toks[i];
                parents[at + i] = if (i == 0) -1 else @intCast(at + i - 1);
            }
            at += len;
        }
        return .{ .count = total, .tokens = tokens, .parents = parents };
    }

    /// Each held draft's chance of landing from the raw confidences (the engine's own allocation, without the policy).
    fn probabilities(ptr: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
        const x = self(ptr);
        const st = try x.get(s);
        var run: f64 = 1.0;
        for (out, 0..) |*o, i| {
            run *= if (i < st.n) st.conf[i] else 0.0;
            o.* = run;
        }
        return st.n > 0;
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        const x = self(ptr);
        const kv = x.streams.fetchRemove(s) orelse return;
        const st = kv.value;
        x.retired[st.slot] = true;
        _ = @import("lane_retire.zig").retire(x, st);
    }
};
