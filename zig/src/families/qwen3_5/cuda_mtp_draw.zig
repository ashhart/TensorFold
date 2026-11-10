//! Coupled keyed-Gumbel and greedy draft draws for Qwen CUDA MTP across solo, tree, and batched passes.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const mtp = @import("cuda_mtp.zig");

const Mtp = mtp.Mtp;
const Rule = cuda.sample.Rule;
const ranks = lanes.shape.ranks;
const lp_top = lanes.draft_tree.lp_top;
const lp_words = mtp.lp_words;

/// One draft row's sampling rule and target verify position (1-based slot + 1, matching cuda_native.zig's verifyFn).
pub const Spec = struct {
    sampling: ?lanes.Sampling = null,
    position: u64 = 0,
};

/// sample.cu's Rule when `s` samples (temperature > 0), or null when `s` decodes greedily.
pub fn ruleOf(s: Spec) ?Rule {
    return cuda.sample.ruleFor(s.sampling);
}

/// sample.cu keys row r at static_cast<unsigned>(meta[r] + 1); wrapping subtraction keeps every u64 position exact.
pub fn metaOf(position: u64) c_int {
    return @bitCast(@as(u32, @truncate(position -% 1)));
}

/// Projects `specs.len` hidden rows at `input` to `d.logits` and launches greedy or target-coupled draws into `d.tok`.
pub fn launch(d: *Mtp, input: u64, specs: []const Spec, cands: bool) !void {
    const rows = specs.len;
    if (rows == 0) return;
    const e = d.e;
    try d.proj(e.head, input, d.logits, rows);
    var any_sampled = false;
    for (specs) |s| any_sampled = any_sampled or ruleOf(s) != null;
    if (!any_sampled) {
        const argmax_rows: usize = if (cands and d.trees) 1 else rows;
        try e.run(e.fns.argmax, .{ argmax_rows, 1, 1 }, .{ 1024, 1, 1 }, .{ d.logits, @as(c_int, @intCast(e.g.vocab)), d.tok });
        if (cands and d.trees) try e.draws.logprobRows(e.stream, d.logits, e.g.vocab, d.picks, ranks, false, d.words, rows);
        return;
    }
    const st = d.staged.bytes;
    const rules = std.mem.bytesAsSlice(Rule, st[Mtp.rules_at..][0 .. rows * @sizeOf(Rule)]);
    const meta = std.mem.bytesAsSlice(c_int, st[Mtp.meta_at..][0 .. rows * 4]);
    for (specs, 0..) |s, i| {
        rules[i] = ruleOf(s) orelse std.mem.zeroes(Rule);
        meta[i] = metaOf(s.position);
    }
    try cuda.copy_source.enqueue(e.d, d.rules, st[Mtp.rules_at..][0 .. rows * @sizeOf(Rule)], e.stream);
    try cuda.copy_source.enqueue(e.d, d.meta, st[Mtp.meta_at..][0 .. rows * 4], e.stream);
    const picks: ?u64 = if (e.picks != 0) e.picks else null;
    var i: usize = 0; // contiguous runs of sampled or greedy rows share one kernel launch, as cuda_engine.zig does
    while (i < rows) {
        const sampled = ruleOf(specs[i]) != null;
        var n: usize = 1;
        while (i + n < rows and (ruleOf(specs[i + n]) != null) == sampled) n += 1;
        if (sampled) {
            try e.draws.drawRows(e.stream, d.logits + i * e.g.vocab * 2, e.g.vocab, d.rules + i * @sizeOf(Rule), d.meta + i * 4, d.tok + i * 4, n, picks);
        } else {
            try e.run(e.fns.argmax, .{ n, 1, 1 }, .{ 1024, 1, 1 }, .{ d.logits + i * e.g.vocab * 2, @as(c_int, @intCast(e.g.vocab)), d.tok + i * 4 });
        }
        i += n;
    }
    if (cands and d.trees) try e.draws.logprobRows(e.stream, d.logits, e.g.vocab, d.picks, lp_top, false, d.words, rows);
}

/// Reads back `d.tok` (and `d.words` for trees) with one sync, filling `out_toks` and tree `cands`.
pub fn fetch(d: *Mtp, specs: []const Spec, out_toks: []u32, cands: [][ranks]u32) !void {
    const rows = specs.len;
    if (rows == 0) return;
    const e = d.e;
    const st = d.staged.bytes;
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(st.ptr + Mtp.tok_at, d.tok, rows * 4, e.stream.handle), "cuMemcpyDtoHAsync");
    if (d.trees and cands.len > 0) try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(st.ptr + Mtp.words_at, d.words, cands.len * lp_words * 4, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.stream.synchronize();
    const toks = std.mem.bytesAsSlice(u32, st[Mtp.tok_at..][0 .. rows * 4]);
    for (out_toks, 0..) |*dst, i| dst.* = toks[i];
    for (cands, 0..) |*c, r| {
        if (!d.trees) {
            c.* = @splat(toks[r]);
            continue;
        }
        const w = std.mem.bytesAsSlice(u32, st[Mtp.words_at + r * lp_words * 4 ..][0 .. lp_words * 4]);
        if (ruleOf(specs[r])) |rl| {
            c.* = rankTreeRow(rl, @truncate(specs[r].position), toks[r], w);
        } else {
            for (c, 0..) |*t, j| t.* = w[2 + j];
        }
    }
}

/// Anchors rank 0 to `picked` (sample.cu's full-vocab draw) and orders runner-up branches by the same Gumbel race.
pub fn rankTreeRow(rl: Rule, position: u32, picked: u32, w: []align(1) const u32) [ranks]u32 {
    var out: [ranks]u32 = @splat(picked);
    var v: [lp_top]f32 = undefined;
    var n_valid: usize = 0;
    while (n_valid < lp_top) : (n_valid += 1) {
        if (w[2 + n_valid] == 0xFFFF_FFFF) break;
        const logit: f32 = @bitCast(w[2 + lp_top + n_valid]);
        if (!std.math.isFinite(logit)) break;
        v[n_valid] = logit * rl.inv_t;
    }
    if (n_valid == 0) return out;
    const m = v[0];
    var keep: usize = if (rl.top_k == 0) n_valid else @min(n_valid, @as(usize, rl.top_k));
    if (rl.top_p > 0.0 and rl.top_p < 1.0 and keep > 0) {
        const denom_n: usize = if (rl.top_k != 0) keep else n_valid;
        var norm: f32 = 0.0;
        for (v[0..denom_n]) |x| norm += @exp(x - m);
        var cum: f32 = 0.0;
        for (v[0..keep], 0..) |x, j| {
            cum += @exp(x - m) / norm;
            if (cum >= rl.top_p) {
                keep = j + 1;
                break;
            }
        }
    }
    if (rl.min_log > -std.math.inf(f32)) {
        const floor = m + rl.min_log;
        var j: usize = 0;
        while (j < keep and v[j] >= floor) j += 1;
        keep = j;
    }
    const full = rl.top_k == 0 or rl.top_k > lanes.gpu_rule.candidates;
    var scores: [lp_top]f32 = undefined;
    var keys: [lp_top]u32 = undefined;
    var taken: [lp_top]bool = @splat(false);
    for (0..n_valid) |j| {
        const id = w[2 + j];
        taken[j] = id == picked;
        const u = if (full) lanes.gpu_full.raceUniform(rl.seed, position, id) else lanes.gpu_rule.uniform24(rl.seed, position, id);
        scores[j] = v[j] - @log(-@log(u));
        keys[j] = lanes.gpu_rule.key(v[j]);
    }
    for (out[1..]) |*slot| {
        var best: ?usize = null;
        for (0..n_valid) |j| {
            if (taken[j]) continue;
            if (best) |b| {
                const j_in = j < keep;
                const b_in = b < keep;
                if (j_in != b_in) {
                    if (j_in) best = j;
                    continue;
                }
                const id_j = w[2 + j];
                const id_b = w[2 + b];
                if (scores[j] > scores[b] or (scores[j] == scores[b] and (keys[j] > keys[b] or (keys[j] == keys[b] and id_j < id_b)))) best = j;
            } else best = j;
        }
        if (best) |b| {
            slot.* = w[2 + b];
            taken[b] = true;
        }
    }
    return out;
}
