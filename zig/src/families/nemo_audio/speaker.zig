//! The speaker branch's streaming state between windows (speaker cache and queue, arXiv 2507.18446), on the host.
const std = @import("std");
const Config = @import("config.zig").Config;
const Allocator = std.mem.Allocator;

/// A cache row: a row of the call's sequence, or the learnable silence row.
pub const Pick = union(enum) { row: u32, sil };

/// The cache and queue a call starts with: rows [0, cache) of its sequence, then [cache, cache + fifo).
pub const State = struct {
    cache: usize = 0,
    fifo: usize = 0,
    sil: [Config.cache_rows]bool = @splat(false), // cache rows that are the silence row (their predictions read as 0)
};

/// The next call's cache and queue, as rows of this call's sequence.
pub const Next = struct {
    cache: [Config.cache_rows]Pick = undefined,
    cache_len: usize = 0,
    fifo: [Config.fifo_rows]u32 = undefined,
    fifo_len: usize = 0,
};

/// The state after a call ([cache; fifo; window], preds `p`, window rows [lo, hi)): queue, overflow, compressed cache.
pub fn update(a: Allocator, c: *const Config, st: *State, p: []const f32, lo: usize, hi: usize) !Next {
    const k = c.speakers;
    const base = st.cache + st.fifo;
    var queue: std.ArrayList(u32) = .empty;
    defer queue.deinit(a);
    for (st.cache..base) |r| try queue.append(a, @intCast(r));
    for (base + lo..base + hi) |r| try queue.append(a, @intCast(r));
    const popped = queue.items.len -| Config.fifo_rows;
    var next: Next = .{};
    for (queue.items[popped..], 0..) |r, i| next.fifo[i] = r;
    next.fifo_len = queue.items.len - popped;
    // the candidates: the old cache (silence rows predict 0), then what the queue let go
    const n = st.cache + popped;
    const rows = try a.alloc(Pick, n);
    defer a.free(rows);
    const preds = try a.alloc(f64, n * k);
    defer a.free(preds);
    for (0..n) |i| {
        const r: u32 = if (i < st.cache) @intCast(i) else queue.items[i - st.cache];
        const sil = i < st.cache and st.sil[i];
        rows[i] = if (sil) .sil else .{ .row = r };
        for (0..k) |s| preds[i * k + s] = if (sil) 0 else p[r * k + s];
    }
    if (n <= Config.cache_rows) {
        @memcpy(next.cache[0..n], rows);
        next.cache_len = n;
    } else {
        const chosen = try compress(a, c, preds, n, @max(st.cache, Config.cache_rows));
        defer a.free(chosen);
        for (chosen, 0..) |x, i| next.cache[i] = if (x) |t| rows[t] else .sil;
        next.cache_len = chosen.len;
    }
    st.cache = next.cache_len;
    st.fifo = next.fifo_len;
    for (next.cache[0..next.cache_len], 0..) |x, i| st.sil[i] = x == .sil;
    return next;
}

/// The cache's rows from `n` candidates, speaker by speaker in time order (null: silence); `new_from` on are new.
pub fn compress(a: Allocator, c: *const Config, preds: []const f64, n: usize, new_from: usize) ![]?usize {
    const k = c.speakers;
    const sil = c.sil_per_speaker;
    const col = n + sil; // a speaker's scores, its silence slots last
    const score = try a.alloc(f64, k * col);
    defer a.free(score);
    // a frame's score for speaker s: log p_s + sum over the others of log(1 - p_j); none under the floor
    for (0..n) |t| {
        var rest: f64 = 0;
        for (0..k) |s| rest += @log(clip(1 - preds[t * k + s]));
        for (0..k) |s| {
            const ps = preds[t * k + s];
            var v = @log(clip(ps)) + (rest - @log(clip(1 - ps)));
            if (ps < c.score_floor) v = -std.math.inf(f64) else if (t >= new_from) v += c.boost_latest;
            score[s * col + t] = v;
        }
    }
    // each speaker's best frames boosted twice, then a wider set once, so every speaker keeps some
    const per: f64 = @as(f64, @floatFromInt(Config.cache_rows - k * sil)) / @as(f64, @floatFromInt(k));
    const order = try a.alloc(usize, n);
    defer a.free(order);
    const half = -@log(0.5);
    for ([2]struct { f64, f64 }{ .{ c.strong_rate, 2 * half }, .{ c.weak_rate, half } }) |b| {
        const top: usize = @intFromFloat(@floor(per * b[0]));
        for (0..k) |s| {
            const v = score[s * col ..][0..n];
            rank(order, v);
            for (order[0..@min(top, n)]) |t| {
                if (std.math.isFinite(v[t])) v[t] += b[1];
            }
        }
    }
    for (0..k) |s| for (0..sil) |j| {
        score[s * col + n + j] = std.math.inf(f64);
    };
    const flat = try a.alloc(usize, k * col);
    defer a.free(flat);
    rank(flat, score);
    const keep = flat[0..Config.cache_rows];
    std.mem.sortUnstable(usize, keep, {}, std.sort.asc(usize));
    const out = try a.alloc(?usize, keep.len);
    for (keep, out) |i, *o| {
        const s = i / col;
        const t = i % col;
        o.* = if (t < n and std.math.isFinite(score[s * col + t])) t else null;
    }
    return out;
}

fn clip(x: f64) f64 {
    return std.math.clamp(x, 1e-30, 1.0);
}

/// order = the indices of v from highest to lowest, ties in index order.
fn rank(order: []usize, v: []const f64) void {
    for (order, 0..) |*o, i| o.* = i;
    const Ctx = struct {
        v: []const f64,
        fn above(ctx: @This(), x: usize, y: usize) bool {
            return ctx.v[x] > ctx.v[y];
        }
    };
    std.sort.block(usize, order, Ctx{ .v = v }, Ctx.above);
}

fn testConfig() Config {
    const stack: @import("config.zig").Stack = .{ .width = 512, .heads = 8, .layers = 1, .ffn = 1, .rot = 64, .qk_norm = false, .base = 10000 };
    return .{ .asr = stack, .spk = stack, .speakers = 8, .head_width = 192, .output = 1, .threshold = 0.5, .kernel_scale = 1, .boost_latest = 0.05, .score_floor = 0.25, .strong_rate = 0.75, .weak_rate = 1.5, .sil_per_speaker = 1 };
}

test "nemo audio speaker state: the first window fills the queue, its overflow joins the cache" {
    const a = std.testing.allocator;
    const c = testConfig();
    var st: State = .{};
    const p = try a.alloc(f32, 60 * 8);
    defer a.free(p);
    @memset(p, 0);
    const next = try update(a, &c, &st, p, 0, 60);
    try std.testing.expectEqual(@as(usize, 20), next.cache_len);
    try std.testing.expectEqual(@as(usize, 40), next.fifo_len);
    try std.testing.expectEqual(Pick{ .row = 19 }, next.cache[19]);
    try std.testing.expectEqual(@as(u32, 20), next.fifo[0]);
    try std.testing.expectEqual(@as(u32, 59), next.fifo[39]);
}

test "nemo audio speaker state: a short window pops nothing from a queue with room" {
    const a = std.testing.allocator;
    const c = testConfig();
    var st: State = .{};
    var p: [30 * 8]f32 = @splat(0);
    const next = try update(a, &c, &st, &p, 0, 30);
    try std.testing.expectEqual(@as(usize, 0), next.cache_len);
    try std.testing.expectEqual(@as(usize, 30), next.fifo_len);
}

test "nemo audio speaker cache: one active speaker keeps its best frames, silence rows fill the other speakers" {
    const a = std.testing.allocator;
    const c = testConfig();
    const n = 300;
    const preds = try a.alloc(f64, n * 8);
    defer a.free(preds);
    @memset(preds, 1e-6);
    // speaker 0 active on frames 100..299, most sure on the later ones
    for (100..n) |t| preds[t * 8] = 0.5 + 0.4 * @as(f64, @floatFromInt(t - 100)) / 200.0;
    const out = try compress(a, &c, preds, n, 188);
    defer a.free(out);
    try std.testing.expectEqual(@as(usize, 188), out.len);
    var frames: usize = 0;
    var last: usize = 0;
    for (out) |x| if (x) |t| {
        try std.testing.expect(t >= 100 and (frames == 0 or t > last));
        last = t;
        frames += 1;
    };
    // 180 of speaker 0's frames: 188 less every speaker's silence row
    try std.testing.expectEqual(@as(usize, 180), frames);
    try std.testing.expect(out[180] == null);
}
