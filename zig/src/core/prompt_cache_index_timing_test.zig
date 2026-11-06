//! The snapshot index's find timed at 16 to 1,024 kept states of 2K to 64K tokens: a hit, an edited-prefix hit, a miss.
const std = @import("std");
const builtin = @import("builtin");
const c = @import("sys").c;
const pc = @import("prompt_cache.zig");
const oracle = @import("prompt_cache_index_oracle.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const gpa = std.testing.allocator;

const vocab: u32 = 151_000;
const im_start = vocab + 1;
const im_end = vocab + 2;
const system = vocab + 3;
const edit = vocab + 6;
const nl: u32 = 2; // the second most frequent word ends every turn, so some unrelated rows equal a state's last token
const shortest = 2048;
const longest = 65_536;
const sizes = [_]usize{ 16, 64, 256, 1024 };
const samples = 101;
const flush_bytes = 1 << 29;

/// A word of a Zipf-like vocabulary: rank k with probability near 1/(k ln vocab).
fn word(r: std.Random) u32 {
    return @intFromFloat(@exp(r.float(f64) * @log(@as(f64, vocab))));
}

fn logUniform(r: std.Random, lo: usize, hi: usize) usize {
    const l: f64 = @log(@as(f64, @floatFromInt(lo)));
    return @intFromFloat(@exp(l + r.float(f64) * (@log(@as(f64, @floatFromInt(hi))) - l)));
}

/// One chat-template turn of `words` words; a kept state ends after its closing newline.
fn turn(list: *std.ArrayList(u32), r: std.Random, words: usize) !void {
    try list.appendSlice(gpa, &.{ im_start, vocab + 4 + r.uintLessThan(u32, 2), nl });
    for (0..words) |_| try list.append(gpa, word(r));
    try list.appendSlice(gpa, &.{ im_end, nl });
}

const Talk = struct { tokens: std.ArrayList(u32) = .empty, goal: usize };
/// A kept state: conversation `talk`'s first `at` tokens, at a turn's end.
const State = struct { talk: usize, at: u32 };

/// Conversations interleaved in creation order: eight shared system prompts or their own, forks, then turns.
const World = struct {
    talks: std.ArrayList(Talk) = .empty,
    states: std.ArrayList(State) = .empty,
    systems: [8][]u32 = @splat(&.{}),

    fn deinit(w: *World) void {
        for (w.talks.items) |*t| t.tokens.deinit(gpa);
        w.talks.deinit(gpa);
        w.states.deinit(gpa);
        for (w.systems) |s| gpa.free(s);
    }

    fn prompt(w: *const World, st: State) []const u32 {
        return w.talks.items[st.talk].tokens.items[0..st.at];
    }

    fn build(w: *World, r: std.Random) !void {
        for (&w.systems) |*s| {
            s.* = try gpa.alloc(u32, 1024 + r.uintLessThan(usize, 5120));
            for (s.*) |*t| t.* = word(r);
        }
        var active: std.ArrayList(usize) = .empty;
        defer active.deinit(gpa);
        while (w.states.items.len < sizes[sizes.len - 1]) {
            const roll = r.uintLessThan(u8, 10);
            if (active.items.len == 0 or roll < 3) { // a new conversation: one in four with a system prompt of its own
                var t: Talk = .{ .goal = logUniform(r, shortest, longest) };
                try t.tokens.appendSlice(gpa, &.{ im_start, system, nl });
                if (r.uintLessThan(u8, 4) == 0) {
                    for (0..1024 + r.uintLessThan(usize, 5120)) |_| try t.tokens.append(gpa, word(r));
                } else try t.tokens.appendSlice(gpa, w.systems[r.uintLessThan(usize, w.systems.len)]);
                try t.tokens.appendSlice(gpa, &.{ im_end, nl });
                if (r.uintLessThan(u8, 4) == 0 and t.goal > t.tokens.items.len + shortest) { // a pasted document first
                    try turn(&t.tokens, r, t.goal - t.tokens.items.len - 5);
                    try w.states.append(gpa, .{ .talk = w.talks.items.len, .at = @intCast(t.tokens.items.len) });
                }
                try w.talks.append(gpa, t);
                try active.append(gpa, w.talks.items.len - 1);
                try w.advance(&active, active.items.len - 1, r);
            } else if (roll == 3 and w.states.items.len > 0) { // a fork: a regenerated or edited turn after a kept state
                const from = w.states.items[r.uintLessThan(usize, w.states.items.len)];
                var t: Talk = .{ .goal = @max(logUniform(r, shortest, longest), from.at + 4096) };
                try t.tokens.appendSlice(gpa, w.prompt(from));
                try w.talks.append(gpa, t);
                try active.append(gpa, w.talks.items.len - 1);
                try w.advance(&active, active.items.len - 1, r);
            } else try w.advance(&active, r.uintLessThan(usize, active.items.len), r);
        }
    }

    /// The next turn of an active conversation, kept from 2K tokens; a conversation past its goal stops instead.
    fn advance(w: *World, active: *std.ArrayList(usize), slot: usize, r: std.Random) !void {
        const i = active.items[slot];
        const t = &w.talks.items[i];
        const words = 256 + r.uintLessThan(usize, 3840);
        if (t.tokens.items.len + words + 5 > @min(t.goal, longest)) {
            _ = active.swapRemove(slot);
            return;
        }
        try turn(&t.tokens, r, words);
        if (t.tokens.items.len >= shortest) try w.states.append(gpa, .{ .talk = i, .at = @intCast(t.tokens.items.len) });
    }
};

/// How many entries find examines for a prompt of `len` tokens: from the first shorter state through the winner.
fn probes(s: *const pc.Store, len: usize, winner: ?*const pc.Entry) usize {
    var n: usize = 0;
    for (s.by_position.items) |e| if (e.at < len) {
        n += 1;
        if (e == winner) break;
    };
    return n;
}

const Work = struct { probes: usize = 0, compares: usize = 0, tokens: usize = 0 };

/// What find does for `q`: entries examined, whole-prefix compares the last-token check lets through, tokens compared.
fn work(s: *const pc.Store, q: []const u32, winner: ?*const pc.Entry) Work {
    var w: Work = .{};
    for (s.by_position.items) |e| if (e.at < q.len) {
        w.probes += 1;
        const n = e.tokens.len;
        if (q[n - 1] == e.tokens[n - 1]) {
            w.compares += 1;
            w.tokens += if (std.mem.indexOfDiff(u32, q[0..n], e.tokens)) |d| d + 1 else n;
        }
        if (e == winner) break;
    };
    return w;
}

const Case = struct { name: []const u8, query: []u32, history: u32, want: ?*pc.Entry };

fn joined(head: []const u32, tail: []const u32) ![]u32 {
    const q = try gpa.alloc(u32, head.len + tail.len);
    @memcpy(q[0..head.len], head);
    @memcpy(q[head.len..], tail);
    return q;
}

/// The next turn after the kept state with the most shorter-than-query states ahead of it: the costliest such hit.
fn hit(s: *const pc.Store, w: *const World, n: usize, next: []const u32) !Case {
    var best: usize = 0;
    var most: usize = 0;
    for (s.entries.items[0..n], 0..) |e, i| {
        const p = probes(s, e.at + next.len, e);
        if (p >= most) best, most = .{ i, p };
    }
    const st = w.states.items[best];
    return .{ .name = "hit", .query = try joined(w.prompt(st), next), .history = st.at, .want = s.entries.items[best] };
}

/// The next turn after the longest kept state: the first entry examined wins, after its whole-prefix compare.
fn hitLongest(s: *const pc.Store, w: *const World, n: usize, next: []const u32) !Case {
    var best: usize = 0;
    for (w.states.items[0..n], 0..) |st, i| if (st.at > w.states.items[best].at) {
        best = i;
    };
    const st = w.states.items[best];
    return .{ .name = "hit_longest", .query = try joined(w.prompt(st), next), .history = st.at, .want = s.entries.items[best] };
}

/// A turn after a kept state, edited just past its earliest kept prefix (which wins): the costliest such pair.
fn edited(s: *const pc.Store, w: *const World, n: usize, next: []const u32) !Case {
    var tip: ?usize = null;
    var first: usize = 0;
    var most: usize = 0;
    for (w.states.items[0..n], 0..) |long, i| {
        var low: ?usize = null;
        for (w.states.items[0..n], 0..) |short, j| if (short.at < long.at and (low == null or short.at < w.states.items[low.?].at) and
            std.mem.eql(u32, w.prompt(short), w.prompt(long)[0..short.at]))
        {
            low = j;
        };
        const j = low orelse continue;
        const p = probes(s, long.at + next.len, s.entries.items[j]);
        if (p >= most) tip, first, most = .{ i, j, p };
    }
    const st = w.states.items[tip orelse return error.NoChain];
    const q = try joined(w.prompt(st), next);
    q[w.states.items[first].at + 3] = edit; // the first word of the turn after the earliest state
    return .{ .name = "edited", .query = q, .history = st.at, .want = s.entries.items[first] };
}

/// A new conversation on the first shared system prompt, longer than every kept state: no state is its prefix.
fn miss(w: *const World, r: std.Random, next: []const u32) !Case {
    var q: std.ArrayList(u32) = .empty;
    errdefer q.deinit(gpa);
    try q.appendSlice(gpa, &.{ im_start, system, nl });
    try q.appendSlice(gpa, w.systems[0]);
    try q.appendSlice(gpa, &.{ im_end, nl });
    while (q.items.len <= longest) try turn(&q, r, 256 + r.uintLessThan(usize, 3840));
    const history: u32 = @intCast(q.items.len);
    try q.appendSlice(gpa, next);
    return .{ .name = "miss", .query = try q.toOwnedSlice(gpa), .history = history, .want = null };
}

/// Earlier loads complete and the clock is read in order, so a single timed call cannot end before its misses do.
fn fence() void {
    switch (builtin.cpu.arch) {
        .aarch64 => asm volatile ("dsb sy\nisb" ::: .{ .memory = true }),
        .x86_64 => asm volatile ("mfence\nlfence" ::: .{ .memory = true }),
        else => {},
    }
}

fn now() i128 {
    fence();
    var t: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC_RAW, &t);
    return @as(i128, t.sec) * std.time.ns_per_s + t.nsec;
}

const Stats = struct { median: f64, low: f64, high: f64 };

fn stats(xs: []f64) Stats {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return .{ .median = xs[xs.len / 2], .low = xs[0], .high = xs[xs.len - 1] };
}

fn batch(s: *pc.Store, q: []const u32, reps: usize) f64 {
    const t0 = now();
    for (0..reps) |_| std.mem.doNotOptimizeAway(s.find(q, &.{}, &.{}));
    return @floatFromInt(now() - t0);
}

/// Warm caches: each sample a batch of back-to-back finds lasting at least 20 us, after 20 warm-up batches.
fn warm(s: *pc.Store, q: []const u32) Stats {
    var reps: usize = 1;
    while (batch(s, q, reps) < 20_000 and reps < 1 << 24) reps *= 2;
    for (0..20) |_| _ = batch(s, q, reps);
    var xs: [samples]f64 = undefined;
    for (&xs) |*x| x.* = batch(s, q, reps) / @as(f64, @floatFromInt(reps));
    return stats(&xs);
}

fn sweep(flush: []const u64) void {
    var sum: u64 = 0;
    var i: usize = 0;
    while (i < flush.len) : (i += 8) sum +%= flush[i];
    std.mem.doNotOptimizeAway(sum);
}

/// Cold caches: each sample one find after reading a buffer larger than the chip's caches, after 5 warm-ups.
fn cold(s: *pc.Store, q: []const u32, flush: []const u64) Stats {
    var xs: [samples]f64 = undefined;
    for (0..5 + samples) |k| {
        sweep(flush);
        const t0 = now();
        std.mem.doNotOptimizeAway(s.find(q, &.{}, &.{}));
        if (k >= 5) xs[k - 5] = @floatFromInt(now() - t0);
    }
    return stats(&xs);
}

/// The whole request lookup around find (marks, reservation, interior cut), then its release, with warm caches.
fn lookup(s: *pc.Store, q: []const u32, history: u32, buf: []u8) !Stats {
    var xs: [samples]f64 = undefined;
    for (0..20 + samples) |k| {
        var fixed = std.heap.FixedBufferAllocator.init(buf);
        const t0 = now();
        const l = try s.lookup(fixed.allocator(), q, history, &.{}, &.{}, &.{});
        if (l.entry) |e| s.release(e);
        if (k >= 20) xs[k - 20] = @floatFromInt(now() - t0);
    }
    return stats(&xs);
}

/// Kept states sharing at least 1K tokens with another, a prefix of another, or sharing only the template header.
fn mix(w: *const World, n: usize) struct { long: usize, prefix: usize, header: usize, median_at: u32 } {
    var long: usize = 0;
    var prefix: usize = 0;
    var header: usize = 0;
    var ats: [sizes[sizes.len - 1]]u32 = undefined;
    for (w.states.items[0..n], 0..) |a, i| {
        ats[i] = a.at;
        var most: usize = 0;
        var inside = false;
        for (w.states.items[0..n], 0..) |b, j| if (i != j) {
            const pa, const pb = .{ w.prompt(a), w.prompt(b) };
            const same = std.mem.indexOfDiff(u32, pa, pb) orelse @min(pa.len, pb.len);
            most = @max(most, same);
            inside = inside or (same == pa.len and pa.len < pb.len);
        };
        long += @intFromBool(most >= 1024);
        prefix += @intFromBool(inside);
        header += @intFromBool(most <= 3);
    }
    std.mem.sort(u32, ats[0..n], {}, std.sort.asc(u32));
    return .{ .long = long, .prefix = prefix, .header = header, .median_at = ats[n / 2] };
}

test "snapshot index timing: find at 16 64 256 1024 kept states" {
    if (builtin.mode != .fast) return error.SkipZigTest; // timings are for ReleaseFast builds only
    var prng = std.Random.DefaultPrng.init(0x7f4a_2295_c0de);
    const r = prng.random();
    var w: World = .{};
    defer w.deinit();
    try w.build(r);
    var next: std.ArrayList(u32) = .empty;
    defer next.deinit(gpa);
    try turn(&next, r, 1022);
    const flush = try std.heap.page_allocator.alloc(u64, flush_bytes / 8);
    defer std.heap.page_allocator.free(flush);
    for (flush, 0..) |*x, i| x.* = i;
    const buf = try gpa.alloc(u8, 1 << 20);
    defer gpa.free(buf);
    const missed = try miss(&w, r, next.items);
    defer gpa.free(missed.query);
    for (sizes) |n| {
        var f: Fake = .{ .gpa = gpa };
        var s = pc.Store.init(gpa, f.snapshots(), .{}, 1 << 40);
        defer s.deinit();
        for (w.states.items[0..n]) |st| {
            f.at = st.at;
            try std.testing.expect(s.keep(w.prompt(st), st.at, null, &.{}, &.{}));
        }
        try std.testing.expectEqual(n, s.entries.items.len);
        const m = mix(&w, n);
        std.debug.print("CACHE_INDEX_MIX states={d} share_1k={d} prefix_of_another={d} header_only={d} median_at={d} min_at={d} max_at={d}\n", .{ n, m.long, m.prefix, m.header, m.median_at, s.by_position.items[n - 1].at, s.by_position.items[0].at });
        const cases = [_]Case{ try hit(&s, &w, n, next.items), try edited(&s, &w, n, next.items), missed, try hitLongest(&s, &w, n, next.items) };
        defer for (cases) |k| if (k.query.ptr != missed.query.ptr) gpa.free(k.query);
        for (cases) |k| {
            try std.testing.expect(oracle.find(&s, k.query, &.{}, &.{}) == k.want);
            try std.testing.expect(s.find(k.query, &.{}, &.{}) == k.want);
            const did = work(&s, k.query, k.want);
            const hot, const fresh, const whole = .{ warm(&s, k.query), cold(&s, k.query, flush), try lookup(&s, k.query, k.history, buf) };
            std.debug.print("CACHE_INDEX states={d} case={s} query={d} resumes_at={d} probes={d} compares={d} tokens_compared={d} warm_ns={d:.1} warm_min={d:.1} warm_max={d:.1} cold_ns={d:.0} cold_min={d:.0} cold_max={d:.0} lookup_ns={d:.0} lookup_min={d:.0} lookup_max={d:.0}\n", .{ n, k.name, k.query.len, if (k.want) |e| e.at else 0, did.probes, did.compares, did.tokens, hot.median, hot.low, hot.high, fresh.median, fresh.low, fresh.high, whole.median, whole.low, whole.high });
        }
    }
}
