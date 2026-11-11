//! Remote prefill's gate and decider: one pull at a time reaches the pair, and a prompt goes there only when it pays.
const std = @import("std");

/// TF_PREFILL_ROUTE: `auto` weighs each prompt (see `Route.weigh`); `always` sends every long prompt, behind any pull.
pub const Mode = enum { auto, always };

/// A queued prompt as the source sees it: its length, the tokens the engine's cache has, the streams decoding.
pub const Ask = struct { len: u64, cached: u64, decoding: u32 };

/// A source's answer for a queued prompt: the pair takes it, it never qualifies (`small`, `off`), or not now (`later`).
pub const Verdict = enum {
    remote,
    small,
    off,
    busy,
    slower,
    down,

    /// A reason that holds for now only: the prompt is weighed again the next time the queue is looked at.
    pub fn later(v: Verdict) bool {
        return switch (v) {
            .busy, .slower, .down => true,
            .remote, .small, .off => false,
        };
    }

    pub fn why(v: Verdict) []const u8 {
        return switch (v) {
            .remote => "the pair takes it",
            .small => "it is too short to pull",
            .off => "remote prefill is off",
            .busy => "the pair is busy with another prompt",
            .slower => "this machine alone is predicted faster, and nothing is decoding",
            .down => "the pair is not answering",
        };
    }
};

/// A pass the lane host or a pull timed: this machine's own prompt pass, a pull end to end, or the last-token pass.
pub const Pass = enum { local, remote, tail };
pub const Measured = struct { pass: Pass, rows: u64, ns: u64 };

/// What a prompt costs on each side, in nanoseconds: the pair's pull end to end, and this machine's own prefill.
pub const Estimate = struct { remote_ns: u64, local_ns: u64 };

/// What `enter` says to a prompt: the pair is its own now, wait a little for it, or prefill here.
pub const Gate = enum { entered, wait, local };

/// How long a pull waiting for the pair sleeps between looks (ms).
const poll_ms = 5;
/// A pass of fewer rows says little about a rate: a fixed cost of 0.5 s or more a forward dominates it.
const min_rows = 1024;

/// A rate or time, seeded and then blended with what is measured.
const Rate = struct {
    v: f64,
    seen: bool = false,

    /// A sample within a third to three times the standing value: the first replaces the seed, then 70% / 30%.
    fn blend(r: *Rate, sample: f64) void {
        if (!std.math.isFinite(sample) or sample <= 0) return;
        const s = std.math.clamp(sample, r.v / 3, r.v * 3);
        r.v = if (r.seen) 0.7 * r.v + 0.3 * s else s;
        r.seen = true;
    }
};

/// One pair, one pull at a time. Several threads may ask; the lock only guards the bookkeeping, never a pull.
pub const Route = struct {
    mode: Mode = .auto,
    lock: std.atomic.Mutex = .unlocked,
    held: bool = false,
    /// when the pull under way is expected to end, on the clock of `enter`'s callers
    ends: u64 = 0,
    /// this machine's prompt tokens a second (seed: 950, the notes' Mac)
    local_tps: Rate = .{ .v = 950 },
    /// a pull's prefix tokens a second, end to end (seed: 1000, the pair's export profile)
    remote_tps: Rate = .{ .v = 1000 },
    /// the Mac's pass after the pull, in seconds: restore, the last token, its first reply (seed: 1)
    tail_s: Rate = .{ .v = 1 },

    /// The monotonic clock every `enter` and `weigh` caller must use.
    pub fn nowNs(io: std.Io) u64 {
        return @intCast(std.Io.Clock.awake.now(io).toNanoseconds());
    }

    /// TF_PREFILL_ROUTE, TF_PREFILL_LOCAL_TPS and TF_PREFILL_REMOTE_TPS, each checked by name.
    pub fn fromEnv() !Route {
        const env = @import("sys").environment;
        const seeds: Route = .{};
        return build(try env.read(u64, "TF_PREFILL_ROUTE", 0), try env.read(f64, "TF_PREFILL_LOCAL_TPS", seeds.local_tps.v), try env.read(f64, "TF_PREFILL_REMOTE_TPS", seeds.remote_tps.v));
    }

    /// A route from the knobs' values (`mode`: the word's index in TF_PREFILL_ROUTE's list).
    pub fn build(mode: u64, local_tps: f64, remote_tps: f64) Route {
        return .{ .mode = if (mode == 0) .auto else .always, .local_tps = .{ .v = local_tps }, .remote_tps = .{ .v = remote_tps } };
    }

    fn acquire(r: *Route) void {
        while (!r.lock.tryLock()) std.atomic.spinLoopHint();
    }

    fn cost(r: *const Route, prefix: u64, rest: u64) Estimate {
        const s: f64 = std.time.ns_per_s;
        const pulled = @as(f64, @floatFromInt(prefix)) / r.remote_tps.v + r.tail_s.v;
        return .{ .remote_ns = @intFromFloat(pulled * s), .local_ns = @intFromFloat(@as(f64, @floatFromInt(rest)) / r.local_tps.v * s) };
    }

    /// A prefix of `prefix` tokens pulled, against `rest` tokens the engine's cache lacks prefilled here.
    pub fn estimate(r: *Route, prefix: u64, rest: u64) Estimate {
        r.acquire();
        defer r.lock.unlock();
        return r.cost(prefix, rest);
    }

    /// A pass this machine or a pull timed: measured rates replace the seeds, a third to three times at a time.
    pub fn observe(r: *Route, m: Measured) void {
        if (m.ns == 0) return;
        const s = @as(f64, @floatFromInt(m.ns)) / 1e9;
        const per_s = @as(f64, @floatFromInt(m.rows)) / s;
        r.acquire();
        defer r.lock.unlock();
        switch (m.pass) {
            .local => if (m.rows >= min_rows) r.local_tps.blend(per_s),
            .remote => if (m.rows >= min_rows) r.remote_tps.blend(per_s),
            .tail => r.tail_s.blend(s),
        }
    }

    /// Whether a prompt that came at `arrived` and looks again at `now` is better off here than behind the pull.
    fn goesLocal(r: *const Route, now: u64, arrived: u64, e: Estimate) bool {
        if (r.mode == .always) return false;
        return (now -| arrived) +| (r.ends -| now) +| e.remote_ns > e.local_ns;
    }

    /// Where a queued prompt goes: not behind a costly pull; else to the pair if a stream decodes or it is no slower.
    pub fn weigh(r: *Route, now: u64, ask: Ask, prefix: u64) Verdict {
        r.acquire();
        defer r.lock.unlock();
        if (r.mode == .always) return .remote;
        const e = r.cost(prefix, ask.len - ask.cached);
        if (r.held and r.goesLocal(now, now, e)) return .busy;
        return if (ask.decoding > 0 or e.remote_ns <= e.local_ns) .remote else .slower;
    }

    /// The pair for this prompt (`entered`, until `leave`), a short wait for the pull under way, or the Mac's prefill.
    pub fn enter(r: *Route, now: u64, arrived: u64, e: Estimate) Gate {
        r.acquire();
        defer r.lock.unlock();
        if (r.held) return if (r.goesLocal(now, arrived, e)) .local else .wait;
        r.held = true;
        r.ends = now +| e.remote_ns;
        return .entered;
    }

    /// The pull that `enter` let in is over.
    pub fn leave(r: *Route) void {
        r.acquire();
        defer r.lock.unlock();
        r.held = false;
    }

    /// `enter`, waiting for the pull under way only while that beats prefilling here; null: the Mac prefills it.
    pub fn admit(r: *Route, io: std.Io, e: Estimate, closing: *const std.atomic.Value(bool)) ?u64 {
        const arrived = Route.nowNs(io);
        while (!closing.load(.acquire)) {
            const now = Route.nowNs(io);
            switch (r.enter(now, arrived, e)) {
                .entered => return now - arrived,
                .local => return null,
                .wait => std.Io.sleep(io, .fromMilliseconds(poll_ms), .awake) catch return null,
            }
        }
        return null;
    }
};
