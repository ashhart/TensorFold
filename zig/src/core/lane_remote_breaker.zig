//! The remote pair's circuit breaker: closed, open for a cool-down after failures, half-open for one probing pull.
const std = @import("std");

/// How a pull that went through ended: the pair worked, was unreachable, or answered with something else.
pub const Outcome = enum { ok, down, other };

/// What `door` tells a pull that holds the gate: go, go after checking the pair is alive, or prefill here.
pub const Door = enum { through, probe, local };

/// What `finish` changed, for the log.
pub const Change = enum { none, opened, closed };

/// Errors that say the pair or its link is not there; a refusal or a mismatch means it answered.
pub fn pairDown(err: anyerror) bool {
    return switch (err) {
        error.Refused, error.Mismatch, error.Protocol, error.Corrupt, error.BadManifest, error.NotHeld, error.Unauthorized => false,
        error.PrefixTooLong, error.PartialPrefix, error.PartialState, error.OutOfMemory, error.NoEntropy, error.WriteFailed => false,
        error.TriggerClosed, error.Aborted, error.TriggerUnauthorized, error.TriggerRefused => false,
        else => true,
    };
}

/// Opens after `trip` failures in a row (0: never), waits `cooldown_ns`, then lets one pull through to try the pair.
pub const Breaker = struct {
    trip: u64 = 3,
    cooldown_ns: u64 = 30 * std.time.ns_per_s,
    fails: u64 = 0,
    /// when the open breaker lets a trial through, on the clock of the callers; 0: closed
    until: u64 = 0,
    trial: bool = false,

    /// Whether a prompt is kept off the pair now: the breaker is open, or its one trial is under way.
    pub fn shut(b: *const Breaker, now: u64) bool {
        return b.trip != 0 and (b.trial or (b.until != 0 and now < b.until));
    }

    /// A pull at the gate: through while closed; after the cool-down one is the trial (`probe`), the rest stay local.
    pub fn door(b: *Breaker, now: u64) Door {
        if (b.trip == 0 or b.until == 0) return .through;
        if (b.trial or now < b.until) return .local;
        b.trial = true;
        return .probe;
    }

    /// The pull that `door` let through is over.
    pub fn finish(b: *Breaker, now: u64, outcome: Outcome) Change {
        if (b.trip == 0) return .none;
        const was_trial = b.trial;
        b.trial = false;
        switch (outcome) {
            .ok => {
                const was_open = b.until != 0;
                b.fails = 0;
                b.until = 0;
                return if (was_open) .closed else .none;
            },
            .down => {
                b.fails +|= 1;
                if (!was_trial and b.fails < b.trip) return .none;
                b.until = now +| @max(b.cooldown_ns, 1);
                return .opened;
            },
            .other => return .none,
        }
    }
};
