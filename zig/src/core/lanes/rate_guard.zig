//! Per-stream plain and drafted throughput measurements and their probe decisions.
const depth = @import("depth.zig");
const Rule = depth.Rule;
const Who = depth.Who;

pub const min_samples = 4;

pub const Constants = struct {
    leave_plain_margin: f64 = 1.2,
    wait_most: u64 = 64,
    weight: f64 = 0.25,
};

pub const State = struct {
    grant_plain: ?u64 = null,
    grant_wait: ?u64 = null,
    plain_tok: f64 = 0,
    plain_ms: f64 = 0,
    plain_n: u32 = 0,
    draft_tok: f64 = 0,
    draft_ms: f64 = 0,
    draft_n: u32 = 0,
    chain: bool = true,
};

/// A finished round's tokens and wall ms.
pub fn observeRate(r: *Rule, who: Who, tokens: usize, ms: f64) !void {
    _ = try r.rates(who);
    const st = &who.state.*.?.rate;
    const drafted = st.chain;
    const tok = if (drafted) &st.draft_tok else &st.plain_tok;
    const span = if (drafted) &st.draft_ms else &st.plain_ms;
    const n = if (drafted) &st.draft_n else &st.plain_n;
    n.* +|= 1;
    // a true mean over the first rounds (a round's wall time is noisy), then a moving average
    const w = @max(r.k.rate.weight, 1.0 / @as(f64, @floatFromInt(n.*)));
    const t: f64 = @floatFromInt(tokens);
    tok.* += w * (t - tok.*);
    span.* += w * (ms - span.*);
}

/// Whether the stream's next round runs plain, based on its measured token rates.
pub fn preferPlain(r: *Rule, who: Who) !bool {
    _ = try r.rates(who);
    const st = &who.state.*.?;
    const plain_round = blk: {
        if (st.rate.draft_n < min_samples) break :blk false;
        if (st.rate.plain_n < min_samples) break :blk true;
        const plain = st.rate.plain_tok / st.rate.plain_ms;
        const drafted = st.rate.draft_tok / st.rate.draft_ms;
        const margin = if (st.rate.chain) r.k.plain_margin else r.k.rate.leave_plain_margin;
        if (drafted >= plain * margin) {
            st.plain = 0;
            st.wait = r.k.depth_probe_every;
            break :blk false;
        }
        break :blk !r.probePlain(st);
    };
    st.rate.chain = !plain_round;
    return plain_round;
}

/// Whether the coming round must tap hidden rows for the head: not while it runs plain with no probe due.
pub fn rowsWanted(r: *const Rule, who: Who) bool {
    const st = if (who.state.*) |*x| x else return true;
    if (!r.rate_guard or st.rate.chain) return true;
    return (st.plain orelse 0) + 1 >= (st.wait orelse r.k.depth_probe_every);
}

/// Whether every stream of a shared round prefers plain (each is asked, so each keeps its probes); else all draft.
pub fn allPlain(r: *Rule, whos: []Who) !bool {
    var all = true;
    for (whos) |w| {
        if (!try preferPlain(r, w)) all = false;
    }
    if (!all) for (whos) |w| {
        w.state.*.?.rate.chain = true;
    };
    return all;
}

pub fn probeCount(r: *Rule, plain: *?u64, delay: *?u64) bool {
    plain.* = (plain.* orelse 0) + 1;
    const wait = delay.* orelse r.k.depth_probe_every;
    if (plain.*.? < wait) return false;
    plain.* = 0;
    delay.* = @min(2 * wait, if (r.rate_guard) r.k.rate.wait_most else r.k.plain_wait_most);
    return true;
}

/// Keep a separate zero-grant backoff while the rate guard also probes.
pub fn grantAllows(r: *Rule, who: Who, granted: u32) bool {
    const st = &who.state.*.?;
    const plain = if (r.rate_guard) &st.rate.grant_plain else &st.plain;
    const wait = if (r.rate_guard) &st.rate.grant_wait else &st.wait;
    if (granted == 0) return !r.plain_guard or probeCount(r, plain, wait);
    plain.* = 0;
    wait.* = r.k.depth_probe_every;
    return true;
}
