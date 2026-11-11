//! Remote prefill's gate: the pair takes one pull at a time, and a prompt that finds it busy is prefilled here.
const std = @import("std");

/// TF_PREFILL_ROUTE: `auto` prefills here a prompt that finds the pair busy unless waiting is faster, `always` waits.
pub const Mode = enum { auto, always };

/// What a prompt costs on each side, in nanoseconds: the pair's pull end to end, and the Mac's own prefill.
pub const Estimate = struct { remote_ns: u64, local_ns: u64 };

/// What `enter` says to a prompt: the pair is its own now, wait a little for it, or prefill here.
pub const Gate = enum { entered, wait, local };

/// How long a pull waiting for the pair sleeps between looks (ms).
const poll_ms = 5;

const local_tps = 950.0; // the Mac's own prefill, prompt tokens a second
const pair_tps = 1000.0; // the pair's export profile, prompt tokens a second
const overhead_s = 1.0; // the Mac's pull, landing and last token once the pair is done

/// One pair, one pull at a time. Several threads may ask; the lock only guards the bookkeeping, never a pull.
pub const Route = struct {
    mode: Mode = .auto,
    lock: std.atomic.Mutex = .unlocked,
    held: bool = false,
    /// when the pull under way is expected to end, on the clock of `enter`'s callers
    ends: u64 = 0,

    /// The monotonic clock every `enter` and `busy` caller must use.
    pub fn nowNs(io: std.Io) u64 {
        return @intCast(std.Io.Clock.awake.now(io).toNanoseconds());
    }

    pub fn fromEnv() !Route {
        const word = try @import("sys").environment.read(u64, "TF_PREFILL_ROUTE", 0);
        return .{ .mode = if (word == 0) .auto else .always };
    }

    /// A prefix of `prefix` tokens pulled, against `rest` tokens the Mac's cache lacks prefilled here.
    pub fn estimate(prefix: u64, rest: u64) Estimate {
        const s: f64 = std.time.ns_per_s;
        return .{
            .remote_ns = @intFromFloat((@as(f64, @floatFromInt(prefix)) / pair_tps + overhead_s) * s),
            .local_ns = @intFromFloat(@as(f64, @floatFromInt(rest)) / local_tps * s),
        };
    }

    fn acquire(r: *Route) void {
        while (!r.lock.tryLock()) std.atomic.spinLoopHint();
    }

    /// Whether a prompt that came at `arrived` and looks again at `now` is better off here than behind the pull.
    fn goesLocal(r: *const Route, now: u64, arrived: u64, e: Estimate) bool {
        if (r.mode == .always) return false;
        const waited = now -| arrived;
        return waited +| (r.ends -| now) +| e.remote_ns > e.local_ns;
    }

    /// Whether a prompt arriving at `now` would be prefilled here at once: the pair is busy and waiting costs more.
    pub fn busy(r: *Route, now: u64, e: Estimate) bool {
        r.acquire();
        defer r.lock.unlock();
        return r.held and r.goesLocal(now, now, e);
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
