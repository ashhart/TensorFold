//! The drafter's host policy: depth, the lookup hand-off, and the drafting governor, all behind env switches.
const std = @import("std");
const Backend = @import("backend.zig").Backend;

var fast_argmax_cache: ?bool = null;
pub fn fastArgmax() bool {
    if (fast_argmax_cache) |v| return v;
    const s = std.c.getenv("TF_DSV41_FAST_ARGMAX");
    fast_argmax_cache = if (s) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
    return fast_argmax_cache.?;
}

/// The tokens that followed the last earlier occurrence of `s`'s own tail, at most `out.len` of them.
pub fn tokAt(p: []const u32, h: []const u32, k: usize) u32 {
    return if (k < p.len) p[k] else h[k - p.len];
}

/// The tokens that followed the last earlier occurrence of the stream's own tail, at most `out.len` of them.
pub fn lookupTokens(p: []const u32, h: []const u32, out: []u32) usize {
    const total = p.len + h.len;
    if (total < 4 or out.len == 0) return 0;
    var n: usize = @min(total - 1, 12);
    while (n >= 2) : (n -= 1) {
        const tail = total - n;
        var i: usize = 0;
        while (i + n < total) : (i += 1) {
            var j: usize = 0;
            while (j < n and tokAt(p, h, i + j) == tokAt(p, h, tail + j)) j += 1;
            if (j == n) {
                const start = i + n;
                const avail = @min(out.len, total - start);
                for (out[0..avail], 0..) |*t, k| t.* = tokAt(p, h, start + k);
                return avail;
            }
        }
    }
    return 0;
}

/// The hand-off's cap (the drafts a round); the default is the shipped three.
pub fn mtpDepthMax() u32 {
    const v = std.c.getenv("TF_DSV41_MTP_DEPTH_MAX") orelse return 3;
    const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 3;
    return @min(@max(n, 1), 5);
}

var lookup_depth: ?u32 = null;
/// TF_DSV41_LOOKUP_DRAFTS=<depth>: draft that many tokens a round from the stream's own suffix matches.
pub fn lookupDepth() u32 {
    if (lookup_depth) |d| return d;
    var out: u32 = 0;
    if (std.c.getenv("TF_DSV41_LOOKUP_DRAFTS")) |v| {
        std.log.info("deepseek-v41: lookupDepth reads TF_DSV41_LOOKUP_DRAFTS={s}", .{std.mem.span(v)});
        out = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
        if (out > 7) out = 7;
    }
    lookup_depth = out;
    return out;
}

var no_refeed: ?bool = null;
/// TF_DSV41_NO_REFEED=0 restores the re-feed; the no-re-feed port is the default.
pub fn noRefeedOn() bool {
    if (no_refeed) |v| return v;
    const raw = std.c.getenv("TF_DSV41_NO_REFEED") orelse {
        no_refeed = true;
        return true;
    };
    no_refeed = !std.mem.eql(u8, std.mem.span(raw), "0");
    return no_refeed.?;
}

var draft_rows_follow: ?bool = null;
/// TF_DSV41_MTP_DRAFT_ROWS=0 restores the old form (the drafter always proposes three, four rows).
pub fn draftRowsFollowWindow() bool {
    if (draft_rows_follow) |v| return v;
    const raw = std.c.getenv("TF_DSV41_MTP_DRAFT_ROWS") orelse {
        draft_rows_follow = true;
        return true;
    };
    draft_rows_follow = !std.mem.eql(u8, std.mem.span(raw), "0");
    return draft_rows_follow.?;
}

var round_signal: ?u8 = null; // 0 = first-draft (shipped), 1 = round, 2 = hybrid
/// TF_DSV41_MTP_ADAPT_SIGNAL=round feeds the policy the round's own per-draft acceptance.
pub fn roundSignalMode() u8 {
    if (round_signal) |v| return v;
    const raw = std.c.getenv("TF_DSV41_MTP_ADAPT_SIGNAL") orelse {
        round_signal = 0;
        return 0;
    };
    round_signal = if (std.mem.eql(u8, std.mem.span(raw), "round"))
        1
    else if (std.mem.eql(u8, std.mem.span(raw), "hybrid"))
        2
    else
        0;
    return round_signal.?;
}

pub fn roundSignalOn() bool {
    return roundSignalMode() != 0;
}

var mtp_confirm: ?u32 = null;
/// TF_DSV41_MTP_ADAPT_CONFIRM=k requires the up-condition for k consecutive rounds before the deep window is taken.
pub fn mtpConfirmRounds() u32 {
    if (mtp_confirm) |v| return v;
    const raw = std.c.getenv("TF_DSV41_MTP_ADAPT_CONFIRM") orelse {
        mtp_confirm = 1;
        return 1;
    };
    const n = std.fmt.parseInt(u32, std.mem.span(raw), 10) catch 1;
    mtp_confirm = @min(@max(n, 1), 8);
    return mtp_confirm.?;
}

/// The shipped band's signal is depth-blind: at depth 3 the first draft usually lands while drafts 2-3 do not.
var ratchet_cache: ?f64 = null;
pub fn mtpRatchetDown() f64 {
    if (ratchet_cache) |v| return v;
    const raw = std.c.getenv("TF_DSV41_MTP_ADAPT_RATCHET") orelse {
        ratchet_cache = 0.90;
        return 0.90;
    };
    const t = std.fmt.parseFloat(f64, std.mem.span(raw)) catch 0;
    ratchet_cache = if (t > 0) t else 0;
    return ratchet_cache.?;
}

var draft_off_on: ?bool = null;
/// THE DRAFTING GOVERNOR: it measures the request's own yield and drops to the plain path below break-even.
pub fn draftOffArmed() bool {
    if (draft_off_on) |v| return v;
    const raw = std.c.getenv("TF_DSV41_DRAFT_OFF") orelse {
        draft_off_on = true;
        return true;
    };
    draft_off_on = !std.mem.eql(u8, std.mem.span(raw), "0");
    return draft_off_on.?;
}

var draft_off_below: ?f64 = null;
pub fn draftOffBelow() f64 {
    if (draft_off_below) |v| return v;
    const raw = std.c.getenv("TF_DSV41_DRAFT_OFF_BELOW") orelse {
        draft_off_below = 1.75;
        return 1.75;
    };
    const t = std.fmt.parseFloat(f64, std.mem.span(raw)) catch 0;
    draft_off_below = @max(t, 0);
    return draft_off_below.?;
}

var draft_off_after: ?u32 = null;
pub fn draftOffAfter() u32 {
    if (draft_off_after) |v| return v;
    const raw = std.c.getenv("TF_DSV41_DRAFT_OFF_AFTER") orelse {
        draft_off_after = 24;
        return 24;
    };
    const n = std.fmt.parseInt(u32, std.mem.span(raw), 10) catch 24;
    draft_off_after = @max(n, 2);
    return draft_off_after.?;
}

var draft_off_gate_ctx: ?f64 = null;
/// THE CONTEXT-GATED GOVERNOR SAMPLE: short below a context threshold, flat 24 at or above it.
pub fn draftOffGateCtx() f64 {
    if (draft_off_gate_ctx) |v| return v;
    const raw = std.c.getenv("TF_DSV41_DRAFT_OFF_GATE_CTX") orelse {
        draft_off_gate_ctx = 32768;
        return 32768;
    };
    const t = std.fmt.parseFloat(f64, std.mem.span(raw)) catch 0;
    draft_off_gate_ctx = @max(t, 0);
    return draft_off_gate_ctx.?;
}

var draft_off_low: ?u32 = null;
/// The short-context sample the gate arms below `draftOffGateCtx`.
pub fn draftOffLow() u32 {
    if (draft_off_low) |v| return v;
    const raw = std.c.getenv("TF_DSV41_DRAFT_OFF_SAMPLE_LOW") orelse {
        draft_off_low = 8;
        return 8;
    };
    const n = std.fmt.parseInt(u32, std.mem.span(raw), 10) catch 8;
    draft_off_low = @max(n, 2);
    return draft_off_low.?;
}

/// The sample this request's governor needs.
pub fn draftOffAfterFor(b: *Backend, i: usize) u32 {
    if (std.c.getenv("TF_DSV41_DRAFT_OFF_AFTER") != null) return draftOffAfter();
    const gate = draftOffGateCtx();
    if (gate <= 0 or i >= b.model.caches.len) return 24;
    return if (@as(f64, @floatFromInt(b.model.caches[i].position)) < gate) draftOffLow() else 24;
}

const MtpPolicy = struct { on: bool, up: f64, down: f64 };
var mtp_policy: ?MtpPolicy = null;
/// The adaptive depth policy, shared with the early `draft` hook so the drafter sizes itself to the policy.
pub fn mtpDepthFor(b: *Backend, slot: u32) u32 {
    const p = mtp_policy orelse blk: {
        const raw = std.c.getenv("TF_DSV41_MTP_ADAPT");
        const single: ?f64 = if (raw) |v| (std.fmt.parseFloat(f64, std.mem.span(v)) catch null) else null;
        const up: f64 = if (single) |t| t else (std.fmt.parseFloat(f64, std.mem.span(std.c.getenv("TF_DSV41_MTP_ADAPT_UP") orelse "0.7")) catch 0.7);
        const down: f64 = if (single) |t| t else (std.fmt.parseFloat(f64, std.mem.span(std.c.getenv("TF_DSV41_MTP_ADAPT_DOWN") orelse "0.7")) catch 0.7);
        const off = raw != null and std.mem.eql(u8, std.mem.span(raw.?), "0");
        const v = MtpPolicy{ .on = !off, .up = up, .down = down };
        mtp_policy = v;
        break :blk v;
    };
    if (!p.on) return lookupDepth();
    const mode = roundSignalMode();
    const up_rate = if (mode == 1) b.model.mtpRoundSlot(slot) else b.model.mtpAcceptSlot(slot);
    const stay_rate = if (mode >= 1) b.model.mtpRoundSlot(slot) else up_rate;
    const prev: u32 = if (slot < b.last_depth.len) b.last_depth[slot] else 0;
    if (mtpRatchetDown() > 0 and slot < b.shallow.len and b.shallow[slot]) return 1;
    return if (up_rate > p.up)
        lookupDepth()
    else if (prev >= 2 and stay_rate >= p.down)
        lookupDepth()
    else
        1;
}

pub fn draftGovernorOff(b: *Backend, i: usize) bool {
    if (!draftOffArmed()) return false;
    // A shared round decodes plain by contract, so the governor runs only while it is the only admitted stream.
    if (b.by.count() > 1) return false;
    if (i >= b.gov_off.len) return false;
    if (b.gov_off[i]) return true;
    const rounds = b.gov_rounds[i];
    const sample = draftOffAfterFor(b, i);
    if (rounds < sample) return false;
    const yield_per_round = @as(f64, @floatFromInt(b.gov_committed[i])) / @as(f64, @floatFromInt(rounds));
    const first = b.model.mtpAcceptSlot(@intCast(i));
    if (yield_per_round < draftOffBelow() and first < 0.9) {
        b.gov_off[i] = true;
        std.log.info("dsv41: drafting off at round {d}: {d:.3} tokens a round, first-draft {d:.3}, break-even {d:.2}, context {d} sample {d}", .{ rounds, yield_per_round, first, draftOffBelow(), b.model.caches[i].position, sample });
        return true;
    }
    return false;
}

/// The proposer's hand-off is a per-slot fact (the pending token and the lane's own conditioning).
pub const MtpCache = struct { pending: u32, drafts: [4]u32, count: u32 = 3 };
