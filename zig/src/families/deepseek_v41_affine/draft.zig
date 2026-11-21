//! DeepSeek-V4.1-Flash's DSpark draft head: three window-only MoE stages and a Markov-biased greedy chain.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const wts = @import("weights.zig");
const c = @import("config.zig");
const fwd = @import("forward.zig");
const Enc = fwd.Enc;
const Buffer = mtl.Buffer;
const moe = @import("moe.zig");
const confidence = @import("confidence.zig");

const MarkovArgs = extern struct { vocab: i32, rank: i32, step: i32, exclude: u32 };
/// The draft block's filler after its first token (dspark_noise_token_id).
const noise_token: u32 = 128799;

/// The draft head's view of the rows: main_norm(main_proj(target means)), each stage's window keys to its ring.
pub fn condition(x: Enc) void {
    const s = x.s;
    const stages = &x.m.weights.stages.?;
    x.linear(stages[0].main_proj.?, s.mh, 0, s.mainx, 0, x.rows);
    x.rms(s.mainx, 0, stages[0].main_norm.?, s.mainx, 0, x.rows, c.hidden);
    for (stages, 0..) |*stage, k| {
        const a = &stage.block.attn;
        x.linear(a.wkv, s.mainx, 0, s.kv, 0, x.rows);
        for (x.segs) |g| x.of(g).kvFinish(s.kv, a.kv_norm, x.m.freq(c.layers + k), g.cache.stage_rings.?[k]);
    }
}

/// One DSpark round from `at`: `token` and four noise rows through three stages, five Markov-biased picks to `out`.
pub fn draft(m: *Model, ds: *st.Scratch, cache: *st.Cache, token: u32, at: usize, out: *[c.draft_block]u32) !void {
    try draftPrefix(m, ds, cache, token, at, out);
}

/// Keep the block's original feature and projection paths, but score only the requested proposals.
pub fn draftPrefix(m: *Model, ds: *st.Scratch, cache: *st.Cache, token: u32, at: usize, out: []u32) !void {
    _ = try draftRun(m, ds, cache, token, at, out, null, 0);
}

pub fn draftScheduled(m: *Model, ds: *st.Scratch, cache: *st.Cache, token: u32, at: usize, out: []u32, policy: confidence.Config, row_base: usize) !confidence.State {
    return (try draftRun(m, ds, cache, token, at, out, policy, row_base)).?;
}

fn draftRun(m: *Model, ds: *st.Scratch, cache: *st.Cache, token: u32, at: usize, out: []u32, policy: ?confidence.Config, row_base: usize) !?confidence.State {
    const n = c.draft_block;
    const steps = out.len;
    std.debug.assert(row_base + steps <= n);
    const stages = &(m.weights.stages orelse return error.NoDraftHead);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    if (row_base == 0) {
        const ids = ds.ids.slice(u32, n);
        ids[0] = token;
        for (ids[1..]) |*v| v.* = noise_token;
        for (ds.pos.slice(i32, n), 0..) |*p, i| p.* = @intCast(at + 1 + i);
        for (ds.pre[0].slice(f32, n * c.hc), 0..) |*v, i| v.* = if (i % c.hc == 0) 1 else 0;
    }
    const draft_ids = ds.draft_ids.slice(u32, n + 1);
    if (policy != null) @memset(draft_ids, 0);
    draft_ids[row_base] = token;
    @memset(ds.img.slice(i32, n), 0); // drafts are text
    if (m.markov_audit or m.markov_top_two_audit) ds.markov_mismatch.slice(u32, 1)[0] = 0;

    if (ds.transition) |cache_state| {
        if (!m.markov_top_two) return error.MarkovTransitionCacheNeedsTopTwo;
        if (cache_state.audit) @memset(cache_state.counts.slice(u32, 3), 0);
    }
    var cb = m.queue.commandBufferUnretained(); // binds only model and scratch buffers; cb.wait() below
    var e = cb.compute(.concurrent);
    fwd.pending_expand = false;
    fwd.pending_topk = 0;
    const x: Enc = .{ .ep = &e, .cbp = &cb, .m = m, .s = ds, .rows = n, .base = at + 1 };
    if (row_base == 0) {
        x.pipe("ds_embed");
        x.buf(ds.ids, 0, 0);
        x.tensor(m.weights.embed, 1);
        x.buf(ds.x, 0, 2);
        x.threads(.{ c.hidden, n, 1 }, .{ 256, 1, 1 });
        for (stages, cache.stage_rings.?, 0..) |*stage, ring, k| {
            const l = &stage.block;
            x.boundary(l.hc_attn, ds.pre[0], ds.pre[1], l.attn_norm);
            stageAttention(x, l, ring, c.layers + k, at);
            x.expand();
            x.boundary(l.hc_ffn, ds.pre[1], ds.pre[0], l.ffn_norm);
            try moe.moe(x, &l.moe, c.draft_topk, c.draft_experts, null);
            x.expand();
        }
        x.boundary(null, ds.pre[0], ds.pre[1], stages[c.stages - 1].norm.?);
        x.linear(m.weights.head, ds.normed, 0, ds.logits, 0, n);
    }
    const last = &stages[c.stages - 1];
    if (policy) |choice| {
        const p = m.kernels.confidence.?;
        if (row_base == 0) {
            x.with();
            p.hidden(x, m.confidence.?, steps);
        }
        for (0..steps) |k| {
            const args: confidence.Args = .{
                .step = @intCast(k),
                .cap = @intCast(steps),
                .prune = @intFromBool(choice.prune),
                .markov_groups = (c.vocab + 255) / 256,
                .row_base = @intCast(row_base),
                .cost_ms = choice.costs,
                .margin = if (choice.margin and m.markov_group_policy) 2 else @intFromBool(choice.margin),
            };
            p.step(x, m.confidence.?, last.markov_embed.?, args);
            if (m.markov_top_two) {
                rankTwo(x, last, ds.draft_ids, row_base + k, args);
            } else {
                scoreMarkov(x, last, ds.draft_ids, row_base + k, std.math.maxInt(u32), ds.markov_part, null, true, .ordinary);
                p.pick(x, args);
            }
        }
    } else {
        // Each row's logits plus the Markov bias of the previous token.
        for (0..steps) |k| {
            if (m.markov_top_two) rankTwo(x, last, ds.draft_ids, k, null) else markovStep(x, last, ds.draft_ids, k, std.math.maxInt(u32));
        }
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.status() != .completed) return error.DeepseekGpuFailed;
    if (m.markov_audit and ds.markov_mismatch.slice(u32, 1)[0] != 0) return error.MarkovLayoutMismatch;
    if (m.markov_top_two_audit and ds.markov_mismatch.slice(u32, 1)[0] != 0) return error.MarkovTopTwoDiffers;
    if (ds.transition) |cache_state| {
        if (cache_state.audit) {
            const counts = cache_state.counts.slice(u32, 3);
            if (counts[2] != 0) return error.MarkovTransitionCacheMismatch;
            std.debug.print("markov-cache hits={d} misses={d}\n", .{ counts[0] * 256, counts[1] * 256 });
        }
    }
    const proposals = draft_ids[row_base + 1 ..];
    if (policy != null) {
        const state = ds.conf_state.slice(confidence.State, 1)[0];
        if (state.chosen > out.len or state.examined > out.len or state.full_passes > out.len or state.chosen > state.full_passes)
            return error.InvalidDeepseekConfidenceDecision;
        const count: usize = @intCast(state.chosen);
        for (proposals[0..count]) |id| if (id >= c.vocab) return error.InvalidDeepseekConfidenceToken;
        @memcpy(out[0..count], proposals[0..count]);
        return state;
    }
    @memcpy(out, proposals[0..out.len]);
    if (m.markov_top_two_audit) {
        for (ds.markov_ranks.slice(MarkovRank, steps), 0..) |rank, step| {
            std.debug.print("markov-top-two pos={d} step={d} first={d} second={d} score={d} runner_score={d}\n", .{
                at + 2, step, rank.first, rank.second, rank.first_score, rank.second_score,
            });
        }
    }
    if (m.confidence_probe) |projection|
        @import("confidence_probe.zig").emit(ds, projection, stages[c.stages - 1].markov_embed.?, at + 2);
    return null;
}

fn markovStep(x: Enc, stage: *const wts.Stage, ids: Buffer, k: usize, exclude: u32) void {
    const groups = (c.vocab + 255) / 256;
    if (x.m.markov_audit)
        scoreMarkov(x, stage, ids, k, exclude, x.s.markov_reference, null, false, .ordinary);
    scoreMarkov(x, stage, ids, k, exclude, x.s.markov_part, x.m.markov_packed, false, .ordinary);
    if (x.m.markov_audit) {
        x.pipe("ds_markov_audit");
        x.buf(x.s.markov_reference, 0, 0);
        x.buf(x.s.markov_part, 0, 1);
        x.buf(x.s.markov_mismatch, 0, 2);
        x.args(@as(u32, groups), 3);
        x.threads(.{ 1, 1, 1 }, .{ 1, 1, 1 });
    }
    x.pipe("ds_markov_pick");
    x.buf(x.s.markov_part, 0, 0);
    x.buf(ids, 0, 1);
    x.args([2]i32{ @intCast(groups), @intCast(k) }, 2);
    x.threads(.{ 1, 1, 1 }, .{ 1, 1, 1 });
}

fn scoreMarkov(x: Enc, stage: *const wts.Stage, ids: Buffer, k: usize, exclude: u32, parts: Buffer, packed_weights: ?Buffer, indirect: bool, comptime variant: enum { ordinary, two, exclusion }) void {
    switch (variant) {
        .ordinary => if (packed_weights != null) x.pipe("ds_markov_packed") else x.pipe("ds_markov"),
        .two => if (x.s.transition != null) x.pipe("tf_markov_cache_score") else x.pipe("ds_markov_top_two"),
        .exclusion => x.pipe("ds_markov_without_winner"),
    }
    if (variant == .two) {
        x.buf(x.s.markov_runners, 0, 6);
        if (x.s.transition) |cache_state| {
            std.debug.assert(exclude == std.math.maxInt(u32));
            x.buf(cache_state.tag, 0, 7);
            x.buf(cache_state.bias, 0, 8);
            x.buf(cache_state.counts, 0, 9);
            x.args([2]u32{ cache_state.log_slots, @intFromBool(cache_state.audit) }, 10);
        }
    }
    x.buf(x.s.logits, 0, 0);
    x.tensor(stage.markov_embed.?, 1);
    if (packed_weights) |weights| x.buf(weights, 0, 2) else x.tensor(stage.markov_head.?, 2);
    x.buf(ids, 0, 3);
    x.buf(parts, 0, 4);
    x.args(MarkovArgs{ .vocab = c.vocab, .rank = c.markov_rank, .step = @intCast(k), .exclude = exclude }, 5);
    if (indirect) {
        x.fence();
        x.ep.*.dispatchIndirect(x.s.conf_dispatch, 0, mtl.Size.of(256, 1, 1));
    } else {
        x.groups(.{ (c.vocab + 255) / 256, 1, 1 }, .{ 256, 1, 1 });
    }
}

fn inspectBranches(m: *Model, ds: *st.Scratch, pos: usize, anchor: u32, primary: *const [c.draft_block]u32, stage: *const wts.Stage) !void {
    for (0..2) |pivot| {
        const ids = ds.branch_ids.slice(u32, 5);
        ids[0] = anchor;
        @memcpy(ids[1 .. 1 + pivot], primary[0..pivot]);
        var cb = m.queue.commandBufferUnretained();
        var e = cb.compute(.concurrent);
        const x: Enc = .{ .ep = &e, .cbp = &cb, .m = m, .s = ds, .rows = c.draft_block, .base = pos - 1 };
        for (pivot..pivot + 3) |k| markovStep(x, stage, ds.branch_ids, k, if (k == pivot) primary[pivot] else std.math.maxInt(u32));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.status() != .completed) return error.DeepseekProbeGpuFailed;
        std.debug.print("branch-probe pos={d} anchor={d} pivot={d} primary={any} alternate={any}\n", .{
            pos, anchor, pivot, primary, ids[1 .. pivot + 4],
        });
    }
}

/// A draft stage's attention: its five rows over the stage's target-key window ending at `at` and their own keys.
pub fn stageAttention(x: Enc, l: *const wts.Layer, ring: Buffer, layer: usize, at: usize) void {
    const s = x.s;
    const a = &l.attn;
    const n = x.rows;
    const freq = x.m.freq(layer);
    x.linear(a.wq_a, s.normed, 0, s.qr, 0, n);
    x.with();
    x.linear(a.wkv, s.normed, 0, s.kv, 0, n);
    x.rms(s.qr, 0, a.q_norm, s.qr, 0, n, c.q_rank);
    x.with();
    x.kvFinish(s.kv, a.kv_norm, freq, null);
    x.linear(a.wq_b, s.qr, 0, s.q, 0, n);
    x.attend(s.q, ring, s.kv, s.draft_picks, c.draft_block, a.sink, s.o, @intCast(at), freq, 0);
    x.woA(a.wo_a, n);
    x.linear(a.wo_b, s.oa, 0, s.branch, 0, n);
}

const MarkovRank = extern struct {
    first: u32,
    second: u32,
    first_score: f32,
    second_score: f32,
};

const RankArgs = extern struct {
    groups: u32,
    row: u32,
    guard: u32,
    margin: u32,
    step: u32,
    cap: u32,
    costs: [c.draft_block + 1]f32,
};

fn rankTwo(x: Enc, stage: *const wts.Stage, ids: Buffer, step: usize, policy: ?confidence.Args) void {
    const indirect = policy != null;
    comptime std.debug.assert(@offsetOf(confidence.State, "score_step") == 5 * @sizeOf(u32));
    const s = x.s;
    const groups = (c.vocab + 255) / 256;
    const exclude = std.math.maxInt(u32);
    scoreMarkov(x, stage, ids, step, exclude, s.markov_part, null, indirect, .two);
    const parallel = x.m.markov_parallel;
    if (policy != null and policy.?.margin == 2) {
        x.pipe("tf_markov_pick_leader_policy");
    } else if (parallel) {
        x.pipe("tf_markov_pick_parallel");
    } else {
        x.pipe("tf_markov_pick_two");
    }
    x.buf(s.markov_part, 0, 0);
    x.buf(s.markov_runners, 0, 1);
    x.buf(ids, 0, 2);
    x.buf(s.markov_ranks, 0, 3);
    x.buf(s.conf_state, 0, 4);
    const args = [4]u32{ groups, @intCast(step), @intFromBool(indirect), 0 };
    const pick = RankArgs{ .groups = groups, .row = @intCast(step), .guard = @intFromBool(indirect), .margin = if (policy) |p| p.margin else 0, .step = if (policy) |p| p.step else 0, .cap = if (policy) |p| p.cap else 0, .costs = if (policy) |p| p.cost_ms else @splat(0) };
    x.args(pick, 5);
    x.groups(.{ 1, 1, 1 }, .{ if (parallel) 256 else 1, 1, 1 });
    if (x.m.markov_top_two_audit) {
        scoreMarkov(x, stage, ids, step, exclude, s.markov_reference, null, false, .ordinary);
        auditRankTwo(x, args);
        scoreMarkov(x, stage, ids, step, exclude, s.markov_reference, null, false, .exclusion);
        var second = args;
        second[2] = 1;
        auditRankTwo(x, second);
    }
}

fn auditRankTwo(x: Enc, args: [4]u32) void {
    x.pipe("tf_markov_audit_two");
    x.buf(x.s.markov_reference, 0, 0);
    x.buf(x.s.markov_part, 0, 1);
    x.buf(x.s.markov_ranks, 0, 2);
    x.buf(x.s.markov_mismatch, 0, 3);
    x.args(args, 4);
    x.groups(.{ 1, 1, 1 }, .{ 1, 1, 1 });
}
