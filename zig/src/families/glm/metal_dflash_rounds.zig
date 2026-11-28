//! GLM's DFlash2 draft rounds on Metal: per-stream drafter contexts, chain drafts, and their chances.
const std = @import("std");
const lanes = @import("lanes");
const be = lanes.backend;
const st = @import("state.zig");
const slots_mod = @import("slots.zig");
const engine_mod = @import("engine.zig");
const taps_mod = @import("metal_dflash_taps.zig");
const model_mod = @import("metal_dflash_model.zig");
const selector = @import("../qwen27/dflash/selector.zig");
const dflash_cfg = @import("../qwen27/dflash/config.zig");
const host = @import("backend.zig");
const source_mod = @import("metal_dflash_source.zig");
const head_mod = @import("metal_dflash_head_depth.zig");
const passes_mod = @import("metal_dflash_passes.zig");
pub const Job = passes_mod.Job;
pub const sharePass = passes_mod.sharePass;
pub const Plan = passes_mod.Plan;
pub const plan = passes_mod.plan;

/// The MTP head's chain when a stream drafts with it here and main's rule is not loaded: main's facts' drafts.
pub const mtp_drafts = 4;

/// A chain holds at most this many drafts: the window's last row is the verify's own pick.
pub const max_held = 7;
/// The drafter's pass, priced per draft by the depth rule.
const draft_step_ms = 1.6;

/// Where a chain's chance product stops it when TF_GLM_DRAFT_CONFIDENCE is unset on Metal.
pub const default_confidence = 0.5;

/// TF_GLM_DRAFT_CONFIDENCE, read once at load: where a DFlash2 chain's chance product stops it.
pub fn confidence() !f64 {
    return @import("sys").environment.read(f64, "TF_GLM_DRAFT_CONFIDENCE", default_confidence);
}

/// Device bytes one stream's drafter context holds: its taps ring plus every layer's rolling K/V.
pub fn contextBytes(c: dflash_cfg.Config) usize {
    const ring = @as(usize, c.window) * c.tapWidth() * 2;
    const caches = @as(usize, c.layers) * 2 * (@as(usize, c.window) + 16) * c.kvWidth() * 2;
    return ring + caches;
}

/// The best chain of `tree`: best root, then each step's best child, its own chance from the cumulative scores.
pub fn flatten(tree: *const selector.Tree, tokens: []u32, chances: []f64) u32 {
    if (tree.tokens.len == 0 or tokens.len == 0 or chances.len == 0) return 0;
    const depth_max = @min(@min(tokens.len, chances.len), max_held);
    var parent: i32 = -1;
    var above: f64 = 0; // the parent's cumulative score (the root's own chance stands alone)
    var depth: u32 = 0;
    while (depth < depth_max) : (depth += 1) {
        var best: ?usize = null;
        for (tree.parents, 0..) |p, i| {
            if (p != parent) continue;
            if (best == null or tree.scores[i] > tree.scores[best.?]) best = i;
        }
        const at = best orelse break;
        tokens[depth] = tree.tokens[at];
        chances[depth] = @exp(tree.scores[at] - above);
        parent = @intCast(at);
        above = tree.scores[at];
    }
    return depth;
}

/// The chain the lane core sizes windows from: the first draft stands, later ones stop under the cut.
pub fn cutChances(chances: []f64, cut: f64) void {
    var chain: f64 = 1;
    for (chances, 0..) |*c, j| {
        chain *= c.*;
        if (j > 0 and chain < cut) c.* = 0;
        if (c.* == 0) chain = 0;
    }
}

/// The held chain stops at the first zeroed chance: drafts past it cost verify rows with no expected gain.
pub fn pricedPrefix(chances: []const f64) u32 {
    var n: usize = 0;
    while (n < chances.len and chances[n] > 0) n += 1;
    return @intCast(n);
}

/// A stream's parked drafter state: its context, and the chain the last round held with its chances.
pub const StreamRounds = struct {
    ctx: *model_mod.Context,
    tokens: [max_held]u32 = @splat(0),
    chances: [max_held]f64 = @splat(0),
    held_n: u32 = 0,
    dflash: bool = true, // the policy at the prompt's start: false drafts this stream with the MTP head
    sampled: bool = false,
    used: ?source_mod.Source = null, // the source its current round drafts with
    last: u64 = 0, // host ticks at its last round's draft hook
    chooser: source_mod.Chooser = .{}, // its lone rounds' source chooser
    alone: bool = false, // its last round was its own, not a shared window's
    plain_next: bool = false, // lone stream's next round was asked zero drafts (skip inline pass)
    taps_paused: bool = false,
    head: head_mod.Stream = .{}, // its MTP head rounds, priced by main's rule
};

const dflash_prior = [_]f64{ 0.85, 0.80, 0.80, 0.80, 0.80, 0.75, 0.70 };
const sampled_rule: source_mod.Rule = .{ .first_dflash = 4 };

/// Prompts past this length draft with the MTP head (the head won at 64K).
pub const default_max_ctx = 32768;

/// Which streams DFlash2 drafts for: prompts within `max_ctx`; longer prompts use the MTP head.
pub const Policy = struct {
    max_ctx: u64 = default_max_ctx,

    pub fn ok(p: Policy, len: u64) bool {
        return p.max_ctx == 0 or len <= p.max_ctx;
    }
};

/// The rounds over the slots: a stream's StreamRounds by slot, its context made at its first token.
pub const Rounds = struct {
    gpa: std.mem.Allocator,
    sl: *slots_mod.Slots,
    model: *model_mod.Model,
    taps: *taps_mod.Taps,
    streams: []?*StreamRounds,
    cut: f64, // TF_GLM_DRAFT_CONFIDENCE, read once at load
    policy: Policy = .{},
    rule: source_mod.Rule = .{},
    head: ?*head_mod.Head = null, // main's depth rule for the MTP head's rounds (null: the head drafts mtp_drafts)
    follow: ?@import("metal").ResidencySet.Follow = null, // contexts join the target residency set
    inline_pass: ?*@import("metal_dflash_inline.zig").Inline = null, // same-buffer pass hook
    shared: source_mod.Shared = .{},
    shared_plain: bool = false, // multi-stream round asked zero drafts across all streams (skip inline pass)

    pub fn init(gpa: std.mem.Allocator, sl: *slots_mod.Slots, model: *model_mod.Model, taps: *taps_mod.Taps) !*Rounds {
        const cut = try confidence();
        const r = try gpa.create(Rounds);
        errdefer gpa.destroy(r);
        const streams = try gpa.alloc(?*StreamRounds, sl.slots.len);
        @memset(streams, null);
        r.* = .{ .gpa = gpa, .sl = sl, .model = model, .taps = taps, .streams = streams, .cut = cut };
        return r;
    }

    pub fn deinit(r: *Rounds) void {
        if (r.inline_pass) |x| x.deinit();
        for (r.streams) |s| if (s) |x| r.drop(x);
        if (r.head) |h| h.deinit();
        r.gpa.free(r.streams);
        r.gpa.destroy(r);
    }

    /// The MTP head's rounds take main's depth rule from main's facts (with no drafter loaded).
    pub fn priceHead(r: *Rounds, m: lanes.Model) !void {
        if (r.head) |h| h.deinit();
        r.head = null;
        r.head = try head_mod.Head.init(r.gpa, m);
    }

    fn drop(r: *Rounds, x: *StreamRounds) void {
        if (r.follow) |f| for (contextBuffers(x.ctx)) |b| f.leave(f.ctx, b);
        x.head.deinit(r.gpa);
        x.ctx.deinit(r.gpa);
        r.gpa.destroy(x);
    }

    /// The slot's rounds, or null when it has none (its stream drafts with the MTP head instead).
    pub fn get(r: *const Rounds, slot: u32) ?*StreamRounds {
        if (slot >= r.streams.len) return null;
        return r.streams[slot];
    }

    /// The slot's rounds, making them now; on failure null (the stream keeps the MTP head).
    pub fn make(r: *Rounds, slot: u32) ?*StreamRounds {
        if (slot >= r.streams.len) return null;
        if (r.streams[slot]) |x| return x;
        const x = r.create(slot) orelse return null;
        r.taps.setRoute(slot, &r.sl.slots[slot].s, x.ctx.pending);
        return x;
    }

    /// Every slot's rounds made at load, so their contexts are in a residency set before the first request.
    pub fn prepare(r: *Rounds) void {
        for (0..r.streams.len) |slot| if (r.streams[slot] == null) {
            _ = r.create(@intCast(slot));
        };
    }

    /// The slot's rounds and context, made now with no tap route.
    fn create(r: *Rounds, slot: u32) ?*StreamRounds {
        const x = r.gpa.create(StreamRounds) catch return null;
        const ctx = model_mod.Context.init(r.gpa, r.model) catch |err| {
            r.gpa.destroy(x);
            std.log.err("glm drafter: stream {d}'s context ({s}); it drafts with the MTP head", .{ slot, @errorName(err) });
            return null;
        };
        x.* = .{ .ctx = ctx };
        r.streams[slot] = x;
        if (r.follow) |f| for (contextBuffers(ctx)) |b| f.join(f.ctx, b);
        return x;
    }

    /// A new prompt in `slot`, `at` rows restored (a cache hit), `active` streams counted: the policy decides.
    pub fn begin(r: *Rounds, slot: u32, at: u64, active: u64, sampled: bool) ?*StreamRounds {
        if (active <= 1) {
            r.shared.reset();
            r.shared_plain = false;
        }
        const x = r.make(slot) orelse return null;
        x.ctx.reset(r.model.graph.config, at) catch |err| {
            std.log.err("glm drafter: stream {d}'s context reset ({s}); it drafts with the MTP head", .{ slot, @errorName(err) });
            r.discard(slot);
            return null;
        };
        x.held_n = 0;
        x.taps_paused = false;
        x.dflash = r.policy.ok(r.sl.slots[slot].prompt_len);
        x.sampled = sampled;
        x.used = null;
        x.last = 0;
        x.plain_next = false;
        x.chooser = .{};
        x.head.deinit(r.gpa);
        // a capped stream captures no taps: its windows launch nothing for a drafter it never uses
        if (x.dflash) r.taps.setRoute(slot, &r.sl.slots[slot].s, x.ctx.pending) else r.taps.clearRoute(&r.sl.slots[slot].s);
        if (!x.dflash) std.log.info("glm drafter: stream {d} drafts with the MTP head (prompt {d}, max_ctx {d})", .{ slot, r.sl.slots[slot].prompt_len, r.policy.max_ctx });
        return x;
    }

    /// The slot's window verified `drafts` drafts: what the head's last round, if it drafted, is priced by.
    pub fn sawWindow(r: *Rounds, slot: u32, drafts: u32) void {
        const x = r.get(slot) orelse return;
        if (x.head.asked != null) x.head.held = drafts;
    }

    /// The MTP head's drafts for `sr`'s round: main's priced depth within the core's ask.
    pub fn headDepth(r: *Rounds, sr: ?*StreamRounds, req: be.DraftRequest) !u32 {
        const h = r.head orelse return @min(req.depth, mtp_drafts);
        const x = sr orelse return @min(req.depth, mtp_drafts);
        return h.depth(&x.head, req.depth, req.rows == null);
    }

    /// The rows the next round verifies: the slot's held chain, as its priced prefix left it.
    pub fn heldRows(r: *Rounds, slot: u32) ?lanes.stream.Held {
        if (slot >= r.streams.len) return null;
        return .{ .count = r.sl.slots[slot].held_n };
    }

    /// The slot left the rounds: its tap route goes now.
    pub fn release(r: *Rounds, slot: u32) void {
        if (slot >= r.streams.len) return;
        if (r.streams[slot] != null) r.taps.clearRoute(&r.sl.slots[slot].s); // keep context for slot's next stream
    }

    /// The slot's rounds and context freed (a context that failed to reset).
    fn discard(r: *Rounds, slot: u32) void {
        r.release(slot);
        if (r.streams[slot]) |x| r.drop(x);
        r.streams[slot] = null;
    }

    fn ruleFor(r: *const Rounds, sr: *const StreamRounds) source_mod.Rule {
        return if (sr.sampled) sampled_rule else r.rule;
    }

    /// Whether the next round for `wins` is predicted to use DFlash2, so inline encoding only runs when needed.
    pub fn expectInlineDflash(r: *const Rounds, wins: []const slots_mod.Win) bool {
        if (wins.len == 0) return false;
        if (wins.len == 1) {
            const sr = r.get(wins[0].slot) orelse return false;
            return !sr.plain_next and sr.dflash and !sr.taps_paused and sr.chooser.peek(r.ruleFor(sr), true) == .dflash;
        }
        if (r.shared_plain) return false;
        var sampled = false;
        for (wins) |w| {
            const sr = r.get(w.slot) orelse return false;
            if (!sr.dflash or sr.taps_paused) return false;
            if (sr.sampled) sampled = true;
        }
        return r.shared.expect(source_mod.shared_rule, wins.len, sampled) == .dflash;
    }

    /// The stream's round: its committed taps end at `position - 1`, then a fixed block sliced to `depth` drafts.
    pub fn round(r: *Rounds, sr: *StreamRounds, position: u64, anchor: u32, depth: u32, out: *Round) !void {
        sr.ctx.committed_end = position - 1;
        sr.held_n = 0;
        const nodes = @min(depth, max_held);
        if (nodes == 0) return;
        const m = r.model;
        try m.swap(sr.ctx);
        var tree = m.proposeBlock(anchor, m.blockFor(max_held), .{ .nodes = nodes, .children = 1 }) catch |err| {
            m.swap(sr.ctx) catch {
                m.backend.failed = true; // the model holds this stream's context: recovery resets both
            };
            return err;
        };
        m.swap(sr.ctx) catch {
            tree.deinit(); // the tree was made: it goes before the context it was made in is lost
            m.backend.failed = true;
            return error.DraftContextSwap;
        };
        defer tree.deinit();
        r.holdChain(sr, &tree, nodes, out);
    }

    /// `sr`'s held chain from `tree`: its best chain, cut by the confidence, at most `depth` drafts.
    pub fn holdChain(r: *Rounds, sr: *StreamRounds, tree: *const selector.Tree, depth: u32, out: *Round) void {
        out.held_n = flatten(tree, &sr.tokens, &sr.chances);
        cutChances(sr.chances[0..out.held_n], r.cut);
        out.held_n = @min(pricedPrefix(sr.chances[0..out.held_n]), depth); // a shared pass's block may be longer
        sr.held_n = out.held_n;
        @memcpy(out.tokens[0..out.held_n], sr.tokens[0..out.held_n]);
    }

    /// `sr`'s source for `req`: a shared round's `window` source, else DFlash2 or its chooser (caps first).
    pub fn choose(r: *Rounds, sr: *StreamRounds, req: be.DraftRequest, t: u64, window: ?source_mod.Source) !source_mod.Source {
        const ms: f64 = if (sr.last > 0 and t > sr.last) @as(f64, @floatFromInt(t - sr.last)) / 24e3 else 0;
        if (r.head) |h| if (req.rows != null and req.follow.len > 0) {
            const kept_n: u32 = @intCast(req.follow.len - 1);
            if (window == null and sr.alone) try h.observe(&sr.head, kept_n, ms) else try h.observeShared(&sr.head, kept_n);
        };
        if (sr.used) |u| if (window == null and sr.alone and req.rows != null and ms > 0) sr.chooser.observe(u, @floatFromInt(req.follow.len), ms);
        sr.last = t;
        sr.alone = window == null;
        if (req.rows == null or !sr.dflash) {
            sr.used = .mtp;
            return .mtp;
        }
        if (sr.taps_paused) {
            const want = if (window) |w| w else sr.chooser.peek(r.ruleFor(sr), true);
            if (want == .dflash) {
                sr.used = .mtp;
                return .mtp;
            }
        }
        const s: source_mod.Source = if (window) |w| w else sr.chooser.next(r.ruleFor(sr), true);
        sr.used = s;
        return s;
    }

    /// The rounds of `jobs[members]` in one fixed-block batched pass: each chain slices its requested depth.
    pub fn roundBatch(r: *Rounds, jobs: []const Job, members: []const usize, out: []Round) !void {
        const block = r.model.blockFor(max_held);
        var entries: [8]model_mod.Model.Proposal = undefined;
        for (members, entries[0..members.len]) |i, *en| {
            const j = jobs[i];
            j.sr.ctx.committed_end = j.position - 1;
            j.sr.held_n = 0;
            en.* = .{ .context = j.sr.ctx, .pending = j.anchor, .options = .{ .nodes = @max(1, @min(j.depth, max_held)), .children = 1 }, .block = block };
        }
        var trees: [8]selector.Tree = undefined;
        try r.model.proposeBatch(entries[0..members.len], trees[0..members.len]);
        defer for (trees[0..members.len]) |*t| t.deinit();
        for (members, trees[0..members.len]) |i, *tree| r.holdChain(jobs[i].sr, tree, jobs[i].depth, &out[i]);
    }
};

/// A stream's context buffers a residency set holds: its taps ring and each layer's K/V (at most 8 layers).
fn contextBuffers(ctx: *const model_mod.Context) []const @import("metal").Buffer {
    const S = struct {
        threadlocal var list: [17]@import("metal").Buffer = undefined;
    };
    S.list[0] = ctx.pending;
    const n = @min(ctx.caches.len, 8);
    for (ctx.caches[0..n], 0..) |c, i| {
        S.list[1 + 2 * i] = c.keys;
        S.list[2 + 2 * i] = c.values;
    }
    return S.list[0 .. 1 + 2 * n];
}

/// A round's result: the held chain and its chances, for the slot's held rows and the core's sizing.
pub const Round = struct {
    tokens: [max_held]u32 = @splat(0),
    chances: [max_held]f64 = @splat(0),
    held_n: u32 = 0,
};

fn followAnchor(sl: *slots_mod.Slots, req: be.DraftRequest, one: *u32) ![]const u32 {
    if (req.rows != null) return req.follow;
    const f = req.first orelse return error.NoFirstToken;
    one.* = switch (f) {
        .handle => |h| try sl.ringGet(@intCast(h)),
        .value => |v| v,
    };
    return one[0..1];
}

fn syncTapPause(r_: *Rounds, sr: *StreamRounds, slot: u32, req: be.DraftRequest, window: ?source_mod.Source, sampled: bool, active: usize) !void {
    const paused = if (window == .mtp) r_.shared.settledMtp(source_mod.shared_rule, active, sampled) else if (window == null) sr.chooser.settledMtp(r_.ruleFor(sr)) else false;
    if (paused == sr.taps_paused) return;
    sr.taps_paused = paused;
    r_.taps.setActive(slot, !paused);
    if (!paused) try sr.ctx.reset(r_.model.graph.config, req.position - 1);
}

/// The backend's draft hook with the drafter loaded: each request's kept rows committed, a chain round, MTP fallback.
pub fn draft(b: *host.Backend, r_: *Rounds, requests: []const be.DraftRequest) anyerror!void {
    const sl = b.sl;
    var ones: [st.max_rows]u32 = undefined; // a prompt's first token, the one its head absorbs with
    var mtp: [st.max_rows]slots_mod.Draft = undefined;
    var mtp_n: usize = 0;
    if (requests.len > st.max_rows) return error.DraftOutOfStep;
    const t_enter = now();
    var t_ahead: u64 = t_enter;
    var t_passes: u64 = t_enter;
    defer if (sl.log) r_.model.backend.stamps.logPhases(sl.last_ret, sl.last_end, t_enter, t_ahead, t_passes, now());
    var slots: [st.max_rows]u32 = undefined;
    var jobs: [st.max_rows]Job = undefined;
    var job_of: [st.max_rows]?usize = @splat(null);
    var job_n: usize = 0;
    const window = sharedSource(b, r_, requests, t_enter);
    var all_plain = window != null;
    var any_sampled = false;
    for (requests) |req| if (req.stream.sampling != null) {
        any_sampled = true;
    };
    for (requests, 0..) |req, k| {
        if (req.early or req.lanes != null or req.ranks) return error.TreeDraftsUnsupported;
        if (req.rows) |path| {
            if (path.len != req.follow.len) return error.DraftOutOfStep;
            for (path, 0..) |x, j| if (x != j) return error.TreeDraftsUnsupported;
        }
        slots[k] = try b.slotOf(req.stream);
        const sr = r_.make(slots[k]);
        if (sr) |x| if (window == null) {
            x.plain_next = req.depth == 0;
        };
        if (req.depth > 0) all_plain = false;
        const was_paused = if (sr) |x| x.taps_paused else false;
        const want_dflash = if (sr) |x| req.rows != null and x.dflash and (if (window) |w| w == .dflash else x.chooser.peek(r_.ruleFor(x), true) == .dflash) else false;
        const drafted = if (sr) |x| try r_.choose(x, req, t_enter, window) == .dflash else false;
        if (drafted and req.depth > 0) {
            const follow = try followAnchor(sl, req, &ones[k]);
            jobs[job_n] = .{ .sr = sr.?, .position = req.position, .anchor = follow[follow.len - 1], .depth = req.depth };
            job_of[k] = job_n;
            job_n += 1;
        } else if (sr) |x| {
            x.ctx.committed_end = req.position - 1;
            if (req.rows != null and x.dflash) {
                if (was_paused and want_dflash) {
                    x.taps_paused = false;
                    r_.taps.setActive(slots[k], true);
                    try x.ctx.reset(r_.model.graph.config, req.position - 1);
                } else try syncTapPause(r_, x, slots[k], req, window, any_sampled, requests.len);
            }
        }
    }
    if (window != null) r_.shared_plain = all_plain;
    // the drafting streams' rounds: those of one block size in one pass; a stream drafting nothing costs nothing
    var rounds: [st.max_rows]Round = @splat(.{});
    sharePass(jobs[0..job_n]); // streams of different depths in one pass, not one pass a block size
    // a pass in the window's buffer serves the round when it followed exactly these streams; else today's passes
    const served = if (r_.inline_pass) |x| try x.serve(sl, jobs[0..job_n], rounds[0..job_n]) else false;
    if (job_n > 0 and !served) try sl.commitAhead(); // the keeps go ahead of the drafter's frames, on the same queue
    t_ahead = now();
    var blocks: [st.max_rows]u32 = undefined;
    const fixed_block = r_.model.blockFor(max_held);
    for (blocks[0..job_n]) |*blk| blk.* = fixed_block;
    const p = if (served) Plan{} else plan(blocks[0..job_n]);
    for (0..p.count) |g| {
        var members: [8]usize = undefined;
        var n: usize = 0;
        for (p.group[0..job_n], 0..) |x, i| if (x == g) {
            members[n] = i;
            n += 1;
        };
        if (n == 1) {
            const j = jobs[members[0]];
            try r_.round(j.sr, j.position, j.anchor, j.depth, &rounds[members[0]]);
        } else try r_.roundBatch(jobs[0..job_n], members[0..n], rounds[0..job_n]);
    }
    if (p.count > 0) t_passes = now();
    var mtp_of: [st.max_rows]?usize = @splat(null);
    var head_streams: [st.max_rows]*head_mod.Stream = undefined;
    var head_asks: [st.max_rows]u32 = undefined;
    var head_firsts: [st.max_rows]bool = undefined;
    var head_of: [st.max_rows]usize = undefined;
    var head_n: usize = 0;
    for (requests, 0..) |req, k| {
        const round = if (job_of[k]) |i| rounds[i] else Round{};
        const drafted = if (r_.get(slots[k])) |x| x.used == .dflash else false;
        const slot_ref = &sl.slots[slots[k]];
        slot_ref.held_n = round.held_n;
        if (round.held_n > 0) @memcpy(engine_mod.Engine.u32s(slot_ref.held, round.held_n), round.tokens[0..round.held_n]);
        if (!drafted or round.held_n == 0) {
            // clear stale DFlash2 chances on MTP or plain rounds so one low pick cannot freeze later rounds
            req.stream.draft_chances.clearRetainingCapacity();
            req.stream.draft_chances_cut = null;
        }
        if (!drafted) { // no DFlash2 for this stream: its MTP head takes this round's shape
            const follow = try followAnchor(sl, req, &ones[k]);
            const d = if (window == null) try r_.headDepth(r_.get(slots[k]), req) else @min(req.depth, mtp_drafts);
            if (window != null and r_.head != null) if (r_.get(slots[k])) |x| {
                head_streams[head_n] = &x.head;
                head_asks[head_n] = req.depth;
                head_firsts[head_n] = req.rows == null;
                head_of[head_n] = mtp_n;
                head_n += 1;
            };
            mtp[mtp_n] = .{ .slot = slots[k], .prompt = req.rows == null, .follow = follow, .depth = d, .sampling = req.stream.sampling };
            mtp_of[k] = mtp_n;
            mtp_n += 1;
        }
    }
    if (window != null and head_n > 0) if (r_.head) |h| {
        var depths: [st.max_rows]u32 = undefined;
        try h.sharedDepths(head_streams[0..head_n], head_asks[0..head_n], head_firsts[0..head_n], any_sampled, requests.len, depths[0..head_n]);
        for (head_of[0..head_n], depths[0..head_n]) |m_i, d| mtp[m_i].depth = d;
    };
    // GLM_WINDOWS: the round's source, the tokens its stream's last round gained, the drafts it now holds or asks
    if (sl.log) for (requests, 0..) |req, k| {
        const round = if (job_of[k]) |i| rounds[i] else Round{};
        if (mtp_of[k]) |m_i| {
            std.log.info("glm round: slot {d} source mtp gained {d} drafts {d}", .{ slots[k], req.follow.len, mtp[m_i].depth });
        } else {
            std.log.info("glm round: slot {d} source dflash gained {d} drafts {d}", .{ slots[k], req.follow.len, round.held_n });
        }
    };
    if (mtp_n == 0) return;
    try sl.checkDrafts(mtp[0..mtp_n]);
    try sl.draftAll(mtp[0..mtp_n]);
}

fn now() u64 {
    return @import("sys").c.mach_absolute_time();
}

/// Observe shared-window yield per stream and pick `.dflash` or `.mtp` for multi-stream rounds.
fn sharedSource(b: *host.Backend, r_: *Rounds, requests: []const be.DraftRequest, t: u64) ?source_mod.Source {
    if (requests.len <= 1) return null;
    var eligible: usize = 0;
    var decode_n: usize = 0;
    var sum_tokens: usize = 0;
    var sampled = false;
    for (requests) |req| {
        if (req.stream.sampling != null) sampled = true;
        const sr = r_.get(b.slotOf(req.stream) catch continue) orelse continue;
        if (!sr.dflash) continue;
        eligible += 1;
        if (req.rows != null) {
            decode_n += 1;
            sum_tokens += req.follow.len;
        }
    }
    return if (eligible == 0) .mtp else r_.shared.step(source_mod.shared_rule, requests.len, eligible, decode_n, sum_tokens, sampled, t);
}

/// The lane core's facts with the drafter loaded: chains of up to `max_held` drafts priced by their chances.
pub fn facts(b: *const host.Backend) lanes.Model {
    const single = b.sl.e.ep == null;
    return .{
        // shared_price fits per-row slope across DFlash2 and MTP without charging a fixed block penalty to MTP
        .fixed_draft_block = false,
        .allocate_shared_drafts = true,
        .limit_shared_drafts = false,
        .exact_width = st.max_rows,
        .first_copy_rows = 4,
        .mtp = true, // the drafter is the draft head: the lane core asks it for drafts
        .speculate = true,
        .speculate_early = false,
        .plain_guard = true,
        .drafts = max_held,
        .draft_prior = &dflash_prior,
        // the verify is the MTP head's forward: its one-Mac windows and shared forwards, learned where the head's are
        .learn_row_costs = single and !b.sl.e.device.tensorUnits(),
        .window_costs = &host.one_mac_window_costs,
        .shared_costs = if (single) &host.one_mac_shared_costs else &.{},
        .mtp_step_ms = draft_step_ms / @as(f64, max_held),
        .shared_step_ms = 0,
        .depth_rate = host.depth_rate,
        .streams_exact = true,
        .hidden_rows = true,
        .batch_rows = st.max_rows,
        .max_streams = @intCast(b.sl.slots.len),
        .draft_streams = true,
        .draft_probabilities = true,
        .gpu_tokens = single, // main's: a stream drafting nothing runs one-token rounds queued ahead
    };
}

/// The core's chances hook: each held draft's chance of landing, its ancestors' chained.
pub fn chancesFor(r_: *Rounds, slot: u32, out: []f64) bool {
    const sr = r_.get(slot) orelse return false;
    if (sr.used != .dflash or sr.held_n == 0) return false;
    const n = @min(out.len, sr.held_n);
    var chain: f64 = 1; // a draft lands only if every one before it does: the selector's chances chained
    for (out[0..n], sr.chances[0..n]) |*c, p| {
        chain *= p;
        c.* = chain;
    }
    for (out[n..]) |*c| c.* = 0;
    return true;
}

comptime {
    _ = st;
    _ = engine_mod;
    _ = dflash_cfg;
}
