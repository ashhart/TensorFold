//! Which draft source a GLM stream's next round uses: DFlash2 or the MTP head, by each one's measured tokens a ms.
const std = @import("std");

pub const Source = enum { dflash, mtp };

/// The rule's constants: DFlash2 rounds before the first choice, the tokens a round that keep it, probes, averaging.
pub const Rule = struct {
    first_dflash: u32 = 8, // DFlash2 rounds a stream runs before its first choice
    keep_tokens: f64 = 2.3, // DFlash2 stays, unprobed, at this many tokens a round times its rounds' ms over the head's
    probe_mtp: u32 = 5, // MTP head rounds that settle a choice the drafter's tokens did not
    probe_every: u32 = 128, // rounds between probes of the head while on DFlash2
    back_every: u32 = 64, // rounds between probes of DFlash2 while on the head
    probe_len: u32 = 2,
    watch: u32 = 8, // the latest rounds each source's yield averages (at most `Yield.most`)
    reprobe_margin: f64 = 1.0, // periodic probe back from MTP to DFlash2 must beat MTP by this ratio
};

/// Multi-stream shared rounds settle faster on workload changes and avoid noisy re-probes back into DFlash2.
pub const shared_rule: Rule = .{
    .first_dflash = 4,
    .probe_mtp = 4,
    .back_every = 256,
    .watch = 6,
    .reprobe_margin = 1.05,
};

/// A source's latest rounds: tokens (kept drafts and the verify's own) and ms, the newest `most` of them.
pub const Yield = struct {
    pub const most = 16;
    tokens: [most]f64 = @splat(0),
    ms: [most]f64 = @splat(0),
    n: u32 = 0,

    pub fn add(y: *Yield, tokens: f64, ms: f64) void {
        y.tokens[y.n % most] = tokens;
        y.ms[y.n % most] = ms;
        y.n += 1;
    }

    fn sum(xs: *const [most]f64, n: u32, k: u32) f64 {
        var total: f64 = 0;
        var i: u32 = 0;
        while (i < @min(k, @min(n, most))) : (i += 1) total += xs[(n - 1 - i) % most];
        return total;
    }

    /// Tokens a round over the latest `k` rounds (0 before any).
    pub fn perRound(y: *const Yield, k: u32) f64 {
        const c = @min(k, @min(y.n, most));
        return if (c == 0) 0 else sum(&y.tokens, y.n, k) / @as(f64, @floatFromInt(c));
    }

    /// Ms a round over the latest `k` rounds (0 before any).
    pub fn msRound(y: *const Yield, k: u32) f64 {
        const c = @min(k, @min(y.n, most));
        return if (c == 0) 0 else sum(&y.ms, y.n, k) / @as(f64, @floatFromInt(c));
    }

    /// Tokens a ms over the latest `k` rounds (0 before any).
    pub fn perMs(y: *const Yield, k: u32) f64 {
        const ms = sum(&y.ms, y.n, k);
        return if (ms <= 0) 0 else sum(&y.tokens, y.n, k) / ms;
    }
};

fn other(s: Source) Source {
    return if (s == .dflash) .mtp else .dflash;
}

/// One stream's choice: DFlash2 first, kept on its own tokens a round, else settled by a short MTP probe.
pub const Chooser = struct {
    current: Source = .dflash,
    yields: [2]Yield = .{ .{}, .{} },
    chosen: bool = false,
    probed: bool = false, // the first choice's MTP probe ran
    since: u32 = 0, // rounds on `current` since the last probe ended
    probing: u32 = 0, // probe rounds still to run on `probe`
    probe: Source = .mtp,
    settle: bool = false, // a probe just ended: its rounds decide before the next round

    /// The round that ran on `used` gained `tokens` (its kept drafts and the verify's own) in `ms` of wall time.
    pub fn observe(c: *Chooser, used: Source, tokens: f64, ms: f64) void {
        if (ms <= 0) return;
        c.yields[@backingInt(used)].add(tokens, ms);
    }

    fn y(c: *const Chooser, s: Source) *const Yield {
        return &c.yields[@backingInt(s)];
    }

    /// DFlash2's tokens a round over `k` rounds clear the bar its rounds' extra time sets (none without head rounds).
    fn drafterKeeps(c: *const Chooser, rule: Rule, k: u32) bool {
        const d = c.y(.dflash);
        const m = c.y(.mtp);
        if (m.n == 0 or m.msRound(rule.watch) <= 0) return false;
        return d.perRound(k) >= rule.keep_tokens * d.msRound(rule.watch) / m.msRound(rule.watch);
    }

    fn probeOf(c: *Chooser, s: Source, rounds: u32) Source {
        c.probe = s;
        c.probing = rounds -| 1;
        c.settle = true;
        return s;
    }

    /// What `next` will return when a round finishes without mutating the chooser state.
    pub fn peek(c: *const Chooser, rule: Rule, dflash_ok: bool) Source {
        if (!dflash_ok) return .mtp;
        if (c.probing > 0) return c.probe;
        if (!c.chosen) {
            if (c.y(.dflash).n < rule.first_dflash) return .dflash;
            if (!c.probed and !c.drafterKeeps(rule, rule.first_dflash)) return .mtp;
            const d = c.y(.dflash);
            const m = c.y(.mtp);
            const cur: Source = if (!c.probed or d.perMs(rule.watch) >= m.perMs(rule.watch)) .dflash else .mtp;
            return if (rule.probe_len > 0 and c.since + 1 >= (if (cur == .mtp) rule.back_every else rule.probe_every)) other(cur) else cur;
        }
        var cur = c.current;
        if (c.settle) {
            if (c.probe == .dflash and c.y(.dflash).perMs(rule.probe_len) > c.y(.mtp).perMs(rule.watch) * rule.reprobe_margin) cur = .dflash;
            if (c.probe == .mtp and c.y(.mtp).perMs(rule.probe_len) > c.y(.dflash).perMs(rule.watch)) cur = .mtp;
        }
        if (cur == .dflash and c.since >= rule.watch and c.y(.mtp).n >= rule.probe_len and c.y(.dflash).n >= rule.watch and c.y(.mtp).perMs(rule.watch) > c.y(.dflash).perMs(rule.watch)) cur = .mtp;
        if (c.since + 1 >= (if (cur == .mtp) rule.back_every else rule.probe_every) and rule.probe_len > 0) return other(cur);
        return cur;
    }

    /// The source the stream's next round uses. `dflash_ok`: the caps let it draft with DFlash2 at all.
    pub fn next(c: *Chooser, rule: Rule, dflash_ok: bool) Source {
        if (!dflash_ok) return .mtp;
        if (c.probing > 0) {
            c.probing -= 1;
            return c.probe;
        }
        const d = c.y(.dflash);
        const m = c.y(.mtp);
        if (!c.chosen) {
            if (d.n < rule.first_dflash) return .dflash;
            if (!c.probed and !c.drafterKeeps(rule, rule.first_dflash)) {
                c.probed = true;
                return c.probeOf(.mtp, rule.probe_mtp);
            }
            c.chosen = true;
            c.settle = false;
            c.current = if (!c.probed or d.perMs(rule.watch) >= m.perMs(rule.watch)) .dflash else .mtp;
        }
        if (c.settle) { // a periodic probe ended: the source that yielded more tokens a ms takes the rounds
            c.settle = false;
            if (c.probe == .dflash and d.perMs(rule.probe_len) > m.perMs(rule.watch) * rule.reprobe_margin) c.current = .dflash;
            if (c.probe == .mtp and m.perMs(rule.probe_len) > d.perMs(rule.watch)) c.current = .mtp;
        }
        if (c.current == .dflash and c.since >= rule.watch and m.n >= rule.probe_len and d.n >= rule.watch and m.perMs(rule.watch) > d.perMs(rule.watch)) c.current = .mtp;
        c.since += 1;
        if (c.since >= (if (c.current == .mtp) rule.back_every else rule.probe_every) and rule.probe_len > 0) {
            c.since = 0;
            return c.probeOf(other(c.current), rule.probe_len);
        }
        return c.current;
    }
};

/// Multi-stream shared window source tracker, keyed by stream count (sampled shared rounds use MTP).
pub const Shared = struct {
    choosers: [16]Chooser = @splat(.{}),
    last: u64 = 0,
    used: ?Source = null,
    bucket: usize = 0,
    n: usize = 0,

    pub fn reset(s: *Shared) void {
        s.* = .{};
    }

    pub fn expect(s: *const Shared, rule: Rule, streams: usize, sampled: bool) Source {
        if (sampled) return .mtp;
        const b = @min(streams - 1, s.choosers.len - 1);
        return s.choosers[b].peek(rule, true);
    }

    pub fn step(s: *Shared, rule: Rule, streams: usize, eligible: usize, decode_n: usize, tokens: usize, sampled: bool, t: u64) Source {
        if (sampled) {
            s.used = .mtp;
            s.last = 0;
            return .mtp;
        }
        const b = @min(streams - 1, s.choosers.len - 1);
        if (decode_n == eligible and s.used != null and s.last > 0 and t > s.last and
            s.bucket == b and s.n == eligible)
        {
            const ms = @as(f64, @floatFromInt(t - s.last)) / 24e3;
            const nf: f64 = @floatFromInt(eligible);
            s.choosers[b].observe(s.used.?, @as(f64, @floatFromInt(tokens)) / nf, ms / nf);
        }
        const chosen: Source = if (decode_n == 0) .mtp else if (decode_n == eligible)
            s.choosers[b].next(rule, true)
        else
            s.choosers[b].peek(rule, true);
        s.used = chosen;
        s.last = if (decode_n == eligible or decode_n == 0) t else 0;
        s.bucket = b;
        s.n = eligible;
        return chosen;
    }
};

/// A stream's sources yield `tokens` a round in `ms` (dflash, mtp), its first round on the head: the sources used.
fn run(rule: Rule, ok: bool, tokens: [2]u32, ms: [2]f64, rounds: usize, out: []Source) void {
    var c: Chooser = .{};
    for (out[0..rounds], 0..) |*s, i| {
        s.* = if (i == 0) .mtp else c.next(rule, ok);
        c.observe(s.*, @floatFromInt(tokens[@backingInt(s.*)]), ms[@backingInt(s.*)]);
    }
}

fn count(xs: []const Source, s: Source) usize {
    var n: usize = 0;
    for (xs) |x| n += @intFromBool(x == s);
    return n;
}

test "a chat-like stream (the drafter's rounds cost more than they gain) moves to the MTP head within its first probe" {
    var used: [130]Source = undefined;
    run(.{}, true, .{ 2, 2 }, .{ 32.3, 24.7 }, 130, &used); // tokens a round 2.4 and 2.0 in the cells, rounded
    try std.testing.expect(count(used[0..20], .dflash) <= 9);
    try std.testing.expect(count(used[20..], .dflash) <= 2 * (110 / 64 + 1)); // only the periodic probes return
}

test "a code-like stream keeps DFlash2 without an MTP probe, the head only in its rare probes" {
    var used: [100]Source = undefined;
    run(.{}, true, .{ 5, 2 }, .{ 43.7, 26.9 }, 100, &used);
    try std.testing.expectEqual(@as(usize, 1), count(&used, .mtp)); // the prompt's first round: no probe within 128
    try std.testing.expectEqual(Source.dflash, used[99]);
}

test "a sampled code stream that leaves DFlash2 on a poor start returns at its next probe, not never" {
    var c: Chooser = .{};
    const rule = Rule{};
    var on_dflash: usize = 0;
    var dflash_rounds: usize = 0;
    for (0..200) |i| {
        const s = if (i == 0) Source.mtp else c.next(rule, true);
        dflash_rounds += @intFromBool(s == .dflash);
        const tokens: u32 = if (s == .mtp) 2 else if (dflash_rounds <= 8) 1 else 4; // eight poor drafter rounds
        c.observe(s, @floatFromInt(tokens), if (s == .mtp) 27.1 else 40.5);
        if (i >= 100 and s == .dflash) on_dflash += 1;
    }
    try std.testing.expect(on_dflash >= 90);
}

test "a stream on the MTP head returns to DFlash2 when a probe finds the drafter's tokens again" {
    var c: Chooser = .{};
    const rule = Rule{};
    var on_dflash: usize = 0;
    for (0..400) |i| {
        const s = if (i == 0) Source.mtp else c.next(rule, true);
        const code = i >= 150;
        const tokens: u32 = if (s == .mtp) 2 else if (code) 5 else 2;
        c.observe(s, @floatFromInt(tokens), if (s == .mtp) 24.7 else 32.3);
        if (i >= 300 and s == .dflash) on_dflash += 1;
    }
    try std.testing.expect(on_dflash >= 90);
}

test "a stream on DFlash2 leaves it when the head out-yields it over the watched rounds" {
    var c: Chooser = .{};
    const rule = Rule{};
    var on_mtp: usize = 0;
    for (0..400) |i| {
        const s = if (i == 0) Source.mtp else c.next(rule, true);
        const tokens: u32 = if (s == .mtp) 2 else if (i < 150) 5 else 2;
        c.observe(s, @floatFromInt(tokens), if (s == .mtp) 24.7 else 32.3);
        if (i >= 300 and s == .mtp) on_mtp += 1;
    }
    try std.testing.expect(on_mtp >= 90);
}

test "dflash_ok false keeps a stream on the MTP head regardless of yield" {
    var used: [64]Source = undefined;
    run(.{}, false, .{ 9, 1 }, .{ 30, 30 }, 64, &used);
    try std.testing.expectEqual(@as(usize, 64), count(&used, .mtp));
}

test "shared window chooser adapts per stream count and resets on new burst" {
    var s: Shared = .{};
    const rule = Rule{};
    try std.testing.expectEqual(Source.mtp, s.expect(rule, 3, true));
    try std.testing.expectEqual(Source.mtp, s.step(rule, 3, 3, 0, 0, true, 24000));
    var mtp_rounds: usize = 0;
    var t: u64 = 48000;
    for (0..30) |i| {
        const dt: u64 = if (s.used == .mtp) 24000 * 40 else 24000 * 60;
        t += dt;
        const src = if (i == 0) s.step(rule, 3, 3, 0, 0, false, t) else s.step(rule, 3, 3, 3, 6, false, t);
        if (i > 0 and src == .mtp) mtp_rounds += 1;
        if (i > rule.first_dflash + rule.probe_mtp + 2) try std.testing.expectEqual(src, s.expect(rule, 3, false));
    }
    try std.testing.expect(mtp_rounds > 0);
    try std.testing.expectEqual(Source.mtp, s.expect(rule, 3, true));
    try std.testing.expectEqual(Source.mtp, s.step(rule, 3, 3, 3, 6, true, t + 24000));
    s.reset();
    try std.testing.expectEqual(Source.dflash, s.expect(rule, 3, false));
}

test "a source's yield averages its latest rounds only" {
    var y: Yield = .{};
    for (0..20) |_| y.add(1, 10);
    for (0..4) |_| y.add(5, 10);
    try std.testing.expectApproxEqAbs(@as(f64, 5), y.perRound(4), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 2), y.perRound(16), 1e-9); // 4 x 5 and 12 x 1 over 16
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), y.perMs(16), 1e-9);
}

test "shared_rule switches to MTP after 4 losing DFlash2 rounds and reprobe_margin suppresses noisy switches" {
    var used: [32]Source = undefined;
    run(shared_rule, true, .{ 1, 1 }, .{ 35.0, 19.2 }, 32, &used);
    try std.testing.expectEqual(@as(usize, 4), count(used[1..], .dflash));
    try std.testing.expectEqual(Source.mtp, used[31]);
    var c: Chooser = .{};
    for (0..10) |_| {
        _ = c.next(shared_rule, true);
        c.observe(.dflash, 1.0, 35.0);
        c.observe(.mtp, 1.0, 19.2);
    }
    var back: usize = 0;
    for (0..shared_rule.back_every + 4) |_| {
        const s = c.next(shared_rule, true);
        if (s == .dflash) {
            back += 1;
            // Marginally above 1.0x (19.0 vs 19.2 -> 1.01x) does not clear reprobe_margin (1.05x).
            c.observe(.dflash, 1.0, 19.0);
        } else c.observe(.mtp, 1.0, 19.2);
    }
    try std.testing.expectEqual(@as(usize, shared_rule.probe_len), back);
    try std.testing.expectEqual(Source.mtp, c.current);
}
