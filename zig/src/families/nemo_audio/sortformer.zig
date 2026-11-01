//! The Sortformer's streaming speaker cache and FIFO on the host, in bf16 as SortformerModules keeps them in eval.
const std = @import("std");
const Config = @import("config.zig");
const mel = @import("mel.zig");
const Allocator = std.mem.Allocator;
const rbf = mel.rbf;
const S = Config.speakers;

pub const State = struct {
    gpa: Allocator,
    width: usize, // the Sortformer's width (512)
    spkcache: std.ArrayList(u16) = .empty, // [rows, width] bf16
    fifo: std.ArrayList(u16) = .empty,
    spkcache_preds: std.ArrayList(f32) = .empty, // [rows, 8], bf16 values
    fifo_preds: std.ArrayList(f32) = .empty,
    compressed: bool = false,

    pub fn deinit(st: *State) void {
        st.spkcache.deinit(st.gpa);
        st.fifo.deinit(st.gpa);
        st.spkcache_preds.deinit(st.gpa);
        st.fifo_preds.deinit(st.gpa);
    }

    pub fn spkRows(st: *const State) usize {
        return st.spkcache.items.len / st.width;
    }

    pub fn fifoRows(st: *const State) usize {
        return st.fifo.items.len / st.width;
    }

    /// streaming_update with a window's rows `chunk`, its predictions `preds` and its context rows `lc` and `rc`.
    pub fn update(st: *State, chunk: []const u16, preds: []const f32, lc: usize, rc: usize, sil: []const u16) !void {
        const w = st.width;
        const spk_n = st.spkRows();
        const fifo_n = st.fifoRows();
        const n = chunk.len / w;
        const chunk_len = n - lc - rc;
        try st.fifo_preds.resize(st.gpa, 0);
        try st.fifo_preds.appendSlice(st.gpa, preds[spk_n * S ..][0 .. fifo_n * S]);
        try st.fifo.appendSlice(st.gpa, chunk[lc * w ..][0 .. chunk_len * w]);
        try st.fifo_preds.appendSlice(st.gpa, preds[(spk_n + fifo_n + lc) * S ..][0 .. chunk_len * S]);
        if (fifo_n + chunk_len <= Config.fifo_len) return;
        var pop: usize = Config.spkcache_update_period;
        pop = @max(pop, (chunk_len + fifo_n) -| Config.fifo_len);
        pop = @min(pop, fifo_n + chunk_len);
        try st.spkcache.appendSlice(st.gpa, st.fifo.items[0 .. pop * w]);
        if (st.compressed) {
            try st.spkcache_preds.appendSlice(st.gpa, st.fifo_preds.items[0 .. pop * S]);
        } else {
            try st.spkcache_preds.resize(st.gpa, 0);
            try st.spkcache_preds.appendSlice(st.gpa, preds[0 .. spk_n * S]);
            try st.spkcache_preds.appendSlice(st.gpa, st.fifo_preds.items[0 .. pop * S]);
        }
        try dropFront(u16, st.gpa, &st.fifo, pop * w);
        try dropFront(f32, st.gpa, &st.fifo_preds, pop * S);
        if (st.spkRows() > Config.spkcache_len) {
            try st.compress(sil);
            st.compressed = true;
        }
    }

    /// _compress_spkcache: the cache cut to spkcache_len rows, the best-scored frames of each speaker first.
    fn compress(st: *State, sil: []const u16) !void {
        const gpa = st.gpa;
        const w = st.width;
        const n = st.spkRows();
        const preds = st.spkcache_preds.items;
        const per_spk = Config.spkcache_len / S - Config.sil_frames_per_spk;
        const strong: usize = @intFromFloat(@floor(@as(f64, @floatFromInt(per_spk)) * Config.strong_boost_rate));
        const weak: usize = @intFromFloat(@floor(@as(f64, @floatFromInt(per_spk)) * Config.weak_boost_rate));
        const min_pos: usize = @intFromFloat(@floor(@as(f64, @floatFromInt(per_spk)) * Config.min_pos_scores_rate));
        const nf = n + Config.sil_frames_per_spk;
        // scores [nf, 8]: log p - log(1 - p) + sum over speakers of log(1 - p) - log 0.5, each op rounded to bf16
        const scores = try gpa.alloc(f32, nf * S);
        defer gpa.free(scores);
        const log_half: f32 = @floatCast(@log(0.5));
        const inf = std.math.inf(f32);
        for (0..n) |f| {
            var l1: [S]f32 = undefined;
            var sum: f32 = 0;
            for (0..S) |s| {
                l1[s] = rbf(@log(@max(rbf(1.0 - preds[f * S + s]), Config.pred_score_threshold)));
                sum += l1[s];
            }
            sum = rbf(sum);
            for (0..S) |s| {
                const p = preds[f * S + s];
                const lp = rbf(@log(@max(p, Config.pred_score_threshold)));
                const v = rbf(rbf(rbf(lp - l1[s]) + sum) - log_half);
                scores[f * S + s] = if (p > 0.5) v else -inf;
            }
        }
        // _disable_low_scores: a speaker with min_pos positive frames loses its speech frames that are not positive
        for (0..S) |s| {
            var pos: usize = 0;
            for (0..n) |f| pos += @intFromBool(scores[f * S + s] > 0);
            if (pos < min_pos) continue;
            for (0..n) |f| if (!(scores[f * S + s] > 0) and preds[f * S + s] > 0.5) {
                scores[f * S + s] = -inf;
            };
        }
        if (Config.scores_boost_latest > 0) for (Config.spkcache_len..n) |f| for (0..S) |s| {
            scores[f * S + s] = rbf(scores[f * S + s] + Config.scores_boost_latest);
        };
        try boost(gpa, scores, n, strong, -2.0 * log_half);
        try boost(gpa, scores, n, weak, -1.0 * log_half);
        for (0..S) |s| scores[n * S + s] = inf; // the silence frame a speaker
        // _get_topk_indices: the spkcache_len best of every (speaker, frame), sorted speaker-major then by frame
        const flat = try gpa.alloc(usize, S * nf);
        defer gpa.free(flat);
        for (flat, 0..) |*o, i| o.* = i;
        const Key = struct {
            v: []const f32,
            nf: usize,
            fn at(k: @This(), i: usize) f32 {
                return k.v[(i % k.nf) * S + i / k.nf];
            }
            fn more(k: @This(), a: usize, b: usize) bool {
                const x = k.at(a);
                const y = k.at(b);
                return x > y or (x == y and a < b);
            }
        };
        const key: Key = .{ .v = scores, .nf = nf };
        std.mem.sort(usize, flat, key, Key.more);
        const take = flat[0..Config.spkcache_len];
        for (take) |*i| if (key.at(i.*) == -inf) {
            i.* = std.math.maxInt(usize);
        };
        std.mem.sort(usize, take, {}, std.sort.asc(usize));
        var emb = try gpa.alloc(u16, Config.spkcache_len * w);
        defer gpa.free(emb);
        var gp = try gpa.alloc(f32, Config.spkcache_len * S);
        defer gpa.free(gp);
        for (take, 0..) |i, r| {
            const frame = if (i == std.math.maxInt(usize)) nf else i % nf;
            if (frame >= n) {
                @memcpy(emb[r * w ..][0..w], sil);
                @memset(gp[r * S ..][0..S], 0);
            } else {
                @memcpy(emb[r * w ..][0..w], st.spkcache.items[frame * w ..][0..w]);
                @memcpy(gp[r * S ..][0..S], preds[frame * S ..][0..S]);
            }
        }
        try st.spkcache.resize(gpa, 0);
        try st.spkcache.appendSlice(gpa, emb);
        try st.spkcache_preds.resize(gpa, 0);
        try st.spkcache_preds.appendSlice(gpa, gp);
    }
};

/// _boost_topk_scores: each speaker's `k` best frames of [0, n) raised by `by`.
fn boost(gpa: Allocator, scores: []f32, n: usize, k: usize, by: f32) !void {
    const idx = try gpa.alloc(usize, n);
    defer gpa.free(idx);
    for (0..S) |s| {
        for (idx, 0..) |*o, i| o.* = i;
        const Key = struct {
            v: []const f32,
            s: usize,
            fn more(c: @This(), a: usize, b: usize) bool {
                const x = c.v[a * S + c.s];
                const y = c.v[b * S + c.s];
                return x > y or (x == y and a < b);
            }
        };
        std.mem.sort(usize, idx, Key{ .v = scores, .s = s }, Key.more);
        for (idx[0..@min(k, n)]) |f| scores[f * S + s] = rbf(scores[f * S + s] + by);
    }
}

fn dropFront(comptime T: type, gpa: Allocator, list: *std.ArrayList(T), n: usize) !void {
    const items = list.items;
    std.mem.copyForwards(T, items[0 .. items.len - n], items[n..]);
    try list.resize(gpa, items.len - n);
}

/// avg_pool1d over frames (kernel = stride = 8, ceil mode, padding not counted) of bf16 predictions [frames, 8].
pub fn downsample(gpa: Allocator, hr: []const f32) ![]f32 {
    const frames = hr.len / S;
    const out_n = (frames + Config.sub - 1) / Config.sub;
    const out = try gpa.alloc(f32, out_n * S);
    for (0..out_n) |o| {
        const lo = o * Config.sub;
        const hi = @min(lo + Config.sub, frames);
        for (0..S) |s| {
            var acc: f32 = 0;
            for (lo..hi) |f| acc += hr[f * S + s];
            out[o * S + s] = rbf(acc / @as(f32, @floatFromInt(hi - lo)));
        }
    }
    return out;
}

test "a FIFO past its length pops into the cache, and the cache compresses to its length" {
    const gpa = std.testing.allocator;
    var st: State = .{ .gpa = gpa, .width = 2 };
    defer st.deinit();
    const n = 500;
    const chunk = try gpa.alloc(u16, n * 2);
    defer gpa.free(chunk);
    for (chunk, 0..) |*c, i| c.* = @intCast(i);
    const preds = try gpa.alloc(f32, n * S);
    defer gpa.free(preds);
    for (preds, 0..) |*p, i| p.* = rbf(@as(f32, @floatFromInt(i % 97)) / 97.0);
    try st.update(chunk, preds, 0, 0, &.{ 7, 7 });
    try std.testing.expectEqual(@as(usize, Config.fifo_len), st.fifoRows());
    try std.testing.expectEqual(@as(usize, Config.spkcache_len), st.spkRows());
    try std.testing.expect(st.compressed);
}

test "pooling averages eight frames and the short last one by its own count" {
    const gpa = std.testing.allocator;
    var hr: [10 * S]f32 = @splat(0);
    for (0..10) |f| hr[f * S] = @floatFromInt(f);
    const lr = try downsample(gpa, &hr);
    defer gpa.free(lr);
    try std.testing.expectEqual(@as(f32, 3.5), lr[0]);
    try std.testing.expectEqual(@as(f32, 8.5), lr[S]);
}
