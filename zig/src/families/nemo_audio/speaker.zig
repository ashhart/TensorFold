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
    // speaker s's frame score: log max(p_s, score_clamp) + the others' log(1 - p_j); none while p_s < 0.5
    for (0..n) |t| {
        var rest: f64 = 0;
        for (0..k) |s| rest += @log(clip(1 - preds[t * k + s]));
        for (0..k) |s| {
            const ps = preds[t * k + s];
            score[s * col + t] = if (ps < Config.speech) -std.math.inf(f64) else @log(clip(@max(ps, c.score_clamp))) + (rest - @log(clip(1 - ps)));
        }
    }
    // per: a speaker's whole share after the silence rows; floor(per x min_pos_rate) scores over log 0.5 drop the rest
    const per: f64 = @floatFromInt((Config.cache_rows - k * sil) / k);
    const need = @floor(per * @as(f64, c.min_pos_rate));
    const pos = @log(0.5);
    for (0..k) |s| {
        const v = score[s * col ..][0..n];
        var count: f64 = 0;
        for (v) |x| count += @floatFromInt(@intFromBool(x > pos));
        if (count >= need) for (v) |*x| {
            if (!(x.* > pos)) x.* = -std.math.inf(f64);
        };
    }
    // frames new to the cache: a small boost
    for (0..k) |s| for (@min(new_from, n)..n) |t| {
        score[s * col + t] += c.boost_latest;
    };
    // each speaker's best frames boosted twice, then a wider set once, so every speaker keeps some
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
    // in index order, then the picks that could not score (fewer than the cache's rows did): silence rows at the end
    const out = try a.alloc(?usize, keep.len);
    @memset(out, null);
    var o: usize = 0;
    for (keep) |i| {
        const t = i % col;
        if (score[i] == -std.math.inf(f64)) continue;
        out[o] = if (t < n and std.math.isFinite(score[i])) t else null;
        o += 1;
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
    return .{ .asr = stack, .spk = stack, .speakers = 8, .head_width = 192, .output = 1, .threshold = 0.5, .kernel_scale = 1, .boost_latest = 0.05, .score_clamp = 0.25, .min_pos_rate = 0.5, .strong_rate = 0.75, .weak_rate = 1.5, .sil_per_speaker = 1 };
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

/// brief_spk_1's first window (fp32), 460 rows: f32 [460][3], p of A, B, and 1 - prod(1 - p) of speakers 2-7 (< 0.02)
const brief = @embedFile("testdata/brief_spk_1_w0.f32");

fn briefCache(rate: f32, clamp: f32) !Next {
    return briefCacheSil(rate, clamp, 1);
}

fn briefCacheSil(rate: f32, clamp: f32, sil: usize) !Next {
    const a = std.testing.allocator;
    var c = testConfig();
    c.sil_per_speaker = sil;
    c.min_pos_rate = rate;
    c.score_clamp = clamp;
    var p: [500 * 8]f32 = @splat(0);
    for (0..460) |t| for (0..3) |s| {
        p[t * 8 + s] = @bitCast(std.mem.readInt(u32, brief[(t * 3 + s) * 4 ..][0..4], .little));
    };
    var st: State = .{};
    return update(a, &c, &st, &p, 0, 500);
}

/// The cache: speaker A's rows, its silence row, speaker B's rows, then the other six speakers' silence rows.
fn expectCache(next: Next, a_rows: []const [2]u32, b_rows: []const [2]u32) !void {
    return expectCacheSil(next, a_rows, b_rows, 1);
}

/// The same with `sil` silence rows a speaker.
fn expectCacheSil(next: Next, a_rows: []const [2]u32, b_rows: []const [2]u32, sil: usize) !void {
    var want: [Config.cache_rows]Pick = @splat(.sil);
    var i: usize = 0;
    for (a_rows) |r| for (r[0]..r[1]) |t| {
        want[i] = .{ .row = @intCast(t) };
        i += 1;
    };
    i += sil;
    for (b_rows) |r| for (r[0]..r[1]) |t| {
        want[i] = .{ .row = @intCast(t) };
        i += 1;
    };
    try std.testing.expectEqual(@as(usize, Config.cache_rows), next.cache_len);
    for (want, next.cache[0..next.cache_len]) |w, x| try std.testing.expectEqual(w, x);
}

const brief_a = [_][2]u32{ .{ 188, 211 }, .{ 213, 224 }, .{ 225, 228 }, .{ 241, 243 }, .{ 245, 247 }, .{ 290, 292 }, .{ 296, 298 }, .{ 299, 309 }, .{ 317, 327 }, .{ 333, 355 }, .{ 366, 375 }, .{ 385, 417 }, .{ 424, 460 } };

test "nemo audio speaker cache: a brief speaker with few positive scores keeps its 16 boosted rows (rate 0.5 and 1.0)" {
    const b = [_][2]u32{ .{ 260, 270 }, .{ 271, 277 } };
    try expectCache(try briefCache(0.5, 0.25), &brief_a, &b);
    try expectCache(try briefCache(1.0, 0.25), &brief_a, &b);
}

test "nemo audio speaker cache: rate 0 keeps only a brief speaker's positive scores, A's rows take the rest" {
    const a = [_][2]u32{ .{ 188, 230 }, .{ 233, 235 }, .{ 240, 247 }, .{ 290, 292 }, .{ 296, 309 }, .{ 317, 327 }, .{ 333, 355 }, .{ 366, 375 }, .{ 384, 417 }, .{ 424, 460 } };
    const b = [_][2]u32{ .{ 260, 261 }, .{ 267, 268 }, .{ 275, 277 } };
    try expectCache(try briefCache(0.0, 0.25), &a, &b);
}

test "nemo audio speaker cache: frames under p = 0.5 never score" {
    const a = std.testing.allocator;
    const c = testConfig();
    const n = 300;
    var preds: [n * 8]f64 = @splat(1e-6);
    for (0..n) |t| preds[t * 8] = 0.99;
    // speaker 1 at 0.3-0.49 on 40 frames where speaker 0 drops to 0.2: boosted, they would beat speaker 0's frames
    for (100..140) |t| {
        preds[t * 8] = 0.2;
        preds[t * 8 + 1] = 0.3 + 0.19 * @as(f64, @floatFromInt(t - 100)) / 40.0;
    }
    const out = try compress(a, &c, &preds, n, 188);
    defer a.free(out);
    // speaker 0's 180 frames, then eight silence rows
    for (out[0..180]) |x| try std.testing.expect(x != null and (x.? < 100 or x.? >= 140));
    for (out[180..]) |x| try std.testing.expect(x == null);
}

test "nemo audio speaker cache: a clamp of 0.6 ranks a brief speaker's frames under 0.6 by the other speakers alone" {
    // NeMo's cache for this window with pred_score_threshold 0.6: B's row 273 (p 0.598, A 0.176) gives way to 270
    const b = [_][2]u32{ .{ 260, 273 }, .{ 274, 277 } };
    try expectCache(try briefCache(0.5, 0.6), &brief_a, &b);
}

test "nemo audio speaker cache: the rule needs floor(22 x min_pos_rate) positive scores" {
    const a = std.testing.allocator;
    var c = testConfig();
    const n = 400;
    var preds: [n * 8]f64 = @splat(1e-6);
    for (0..n) |t| preds[t * 8] = 0.99;
    // speaker 1: 11 positive frames (0.62 against 0.1) and 5 more just over 0.5
    for (100..116) |t| {
        preds[t * 8] = 0.1;
        preds[t * 8 + 1] = if (t < 111) 0.62 else 0.52;
    }
    for ([_]f32{ 0.54, 0.55 }, [_]usize{ 11, 16 }) |rate, want| {
        c.min_pos_rate = rate;
        const out = try compress(a, &c, &preds, n, 188);
        defer a.free(out);
        var kept: usize = 0;
        for (out) |x| kept += @intFromBool(x != null and x.? >= 100 and x.? < 116);
        try std.testing.expectEqual(want, kept);
    }
}

/// gap37 (brief_spk_1's first 8 s, 37 s of zeros, the rest), window 0 (fp32): f32 [460], A's p (others < 0.001)
const gap = @embedFile("testdata/gap37_w0.f32");

test "nemo audio speaker cache: with only 92 frames over p = 0.5, the rest of the cache is silence rows at its end" {
    const a = std.testing.allocator;
    var c = testConfig();
    var p: [500 * 8]f32 = @splat(0);
    for (0..460) |t| p[t * 8] = @bitCast(std.mem.readInt(u32, gap[t * 4 ..][0..4], .little));
    // NeMo's cache after this window, with pred_score_threshold 0.25 and 0.1 alike
    for ([_]f32{ 0.25, 0.1 }) |clamp| {
        c.score_clamp = clamp;
        var st: State = .{};
        const next = try update(a, &c, &st, &p, 0, 500);
        try std.testing.expectEqual(@as(usize, Config.cache_rows), next.cache_len);
        var i: usize = 0;
        for ([_][2]u32{ .{ 6, 30 }, .{ 32, 100 } }) |r| for (r[0]..r[1]) |t| {
            try std.testing.expectEqual(Pick{ .row = @intCast(t) }, next.cache[i]);
            i += 1;
        };
        for (next.cache[i..next.cache_len]) |x| try std.testing.expectEqual(Pick.sil, x);
    }
}

test "nemo audio speaker cache: two silence rows a speaker share floor((188 - 16) / 8) = 21 rows each" {
    const a = std.testing.allocator;
    var c = testConfig();
    c.sil_per_speaker = 2;
    const n = 400;
    var preds: [n * 8]f64 = @splat(1e-6);
    for (0..n) |t| preds[t * 8] = 0.99;
    // speaker 1: 11 positive frames and 5 more just over 0.5; floor(21 x 0.565) = 11 fires, 21.5 or 22 would need 12
    for (100..116) |t| {
        preds[t * 8] = 0.1;
        preds[t * 8 + 1] = if (t < 111) 0.62 else 0.52;
    }
    c.min_pos_rate = 0.565;
    const out = try compress(a, &c, &preds, n, 188);
    defer a.free(out);
    // A's frames, its two silence rows (161-162), B's 11, then 14 silence rows (174-187), as in NeMo's sil-2 cache
    const a_rows = Config.cache_rows - 2 - 11 - 14;
    for (out[0..a_rows]) |x| try std.testing.expect(x != null and (x.? < 100 or x.? >= 116));
    for (out[a_rows..][0..2]) |x| try std.testing.expect(x == null);
    for (out[a_rows + 2 ..][0..11], 100..111) |x, t| try std.testing.expectEqual(@as(?usize, t), x);
    for (out[a_rows + 13 ..]) |x| try std.testing.expect(x == null);
}

test "nemo audio speaker cache: with two silence rows, a brief speaker's boosts count from the share 21 (15 rows, not 16)" {
    // NeMo's fp32 sil-2 cache, window 0: B's 4 positives don't fire the rule, so it keeps floor(21 x 0.75) = 15 rows
    const a = [_][2]u32{ .{ 188, 211 }, .{ 214, 218 }, .{ 220, 224 }, .{ 226, 228 }, .{ 242, 243 }, .{ 245, 247 }, .{ 290, 291 }, .{ 296, 298 }, .{ 299, 308 }, .{ 317, 327 }, .{ 333, 355 }, .{ 366, 375 }, .{ 385, 417 }, .{ 424, 460 } };
    const b = [_][2]u32{ .{ 260, 270 }, .{ 272, 277 } };
    try expectCacheSil(try briefCacheSil(0.5, 0.25, 2), &a, &b, 2);
}
