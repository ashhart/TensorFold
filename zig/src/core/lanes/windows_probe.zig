//! A stream's copied rows' trust and landing histories: the prices a copy window's chances come from.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const lone_copy = @import("lone_copy.zig");

/// The weight a stream's copy history keeps a round, so its copies' landing rate follows its content.
const copy_decay = 0.9;

/// A context match this long earns cfg.copy_rate's prior (Qwen3.8-27B, 134 prompts: 85% of such copies' rows landed).
pub const long_match = 20;
/// A copy out of the prompt this early in the reply earns it too (89% on code: restating the asked-for function).
pub const early_reply = 64;
/// Any other copy's prior: about the share of their rows that landed (0 to 13%).
pub const short_rate = 0.08;

/// A copy after a long match, or out of the prompt within the reply's first early_reply tokens.
pub fn trustedCopy(s: *const Stream) bool {
    const pr = s.proposer orelse return true;
    if (pr.lastMatch() >= long_match) return true;
    const at = pr.lastEnd() orelse return false;
    return at <= s.prompt_len and s.context.items.len -| s.prompt_len < early_reply;
}

/// A stream's copy history from its prior: sixteen rows' worth, cfg.copy_rate's share (trusted) or short_rate's landed.
fn copyPrime(e: *const Engine, s: *Stream, trusted: bool) void {
    const i = @intFromBool(trusted);
    if (s.copy_landed[i] >= 0) return;
    const rate = if (trusted) e.cfg.copy_rate else short_rate;
    s.copy_landed[i] = 16 * rate;
    s.copy_broke[i] = 16 * (1 - rate);
}

/// A copied row's landing rate for this stream's trusted copies or the others': landed rows over those plus breaks.
pub fn copyRate(e: *const Engine, s: *Stream, trusted: bool) f64 {
    copyPrime(e, s, trusted);
    const i = @intFromBool(trusted);
    return s.copy_landed[i] / (s.copy_landed[i] + s.copy_broke[i]);
}

/// A verified copy of `offered` rows (anchor landed), `kept` landing; a trusted one teaches both histories.
pub fn observeCopy(e: *const Engine, s: *Stream, trusted: bool, kept: u64, offered: u64) void {
    for ([_]bool{ false, true }) |which| {
        if (which and !trusted) continue;
        copyPrime(e, s, which);
        const i = @intFromBool(which);
        s.copy_landed[i] = copy_decay * s.copy_landed[i] + @as(f64, @floatFromInt(kept));
        s.copy_broke[i] = copy_decay * s.copy_broke[i] + @as(f64, @floatFromInt(@intFromBool(kept < offered)));
    }
}

/// A copy window's history weighs this many rows of cfg.copy_rate's prior before the stream's own copies teach it.
const copied_prior = 4;

/// A copy window's row landing rate for this stream and source: landed rows over those plus breaks, decayed a copy.
pub fn copiedRate(e: *const Engine, s: *Stream, recalled: bool) f64 {
    const i = @intFromBool(recalled);
    if (s.copied_landed[i] < 0) {
        s.copied_landed[i] = copied_prior * e.cfg.copy_rate;
        s.copied_broke[i] = copied_prior * (1 - e.cfg.copy_rate);
    }
    return s.copied_landed[i] / (s.copied_landed[i] + s.copied_broke[i]);
}

/// Strong context matches learn separately from weaker copies; Recall keeps its own source history.
pub fn pricedCopiedRate(e: *const Engine, s: *Stream, recalled: bool, trusted: bool) f64 {
    if (!e.cfg.own_copy_rates) return e.cfg.copy_rate;
    if (e.cfg.matched_copy_rates and !recalled and trusted) return copyRate(e, s, true);
    return copiedRate(e, s, recalled);
}

/// Update both the existing source history and, when enabled, the strong-match history.
pub fn observePricedCopied(e: *const Engine, s: *Stream, recalled: bool, trusted: bool, kept: usize, offered: usize) void {
    observeCopied(e, s, recalled, kept, offered);
    if (e.cfg.matched_copy_rates and !recalled and trusted) observeCopy(e, s, true, kept, offered);
}

/// A verified copy window of `offered` rows, `kept` landing: its source's counters, then its own rate's history.
pub fn observeCopied(e: *const Engine, s: *Stream, recalled: bool, kept: usize, offered: usize) void {
    if (recalled) {
        if (s.recall) |r| r.observe(@intCast(offered), @intCast(kept), s.rounds);
    } else if (s.proposer) |pr| pr.observe(@intCast(offered), @intCast(kept));
    // a lone priced stream keeps the history its copies' chances come from (lanes/lone_copy.zig)
    if (!e.cfg.own_copy_rates and !lone_copy.enabled(e, s)) return;
    _ = copiedRate(e, s, recalled);
    const i = @intFromBool(recalled);
    s.copied_landed[i] = copy_decay * s.copied_landed[i] + @as(f64, @floatFromInt(kept));
    s.copied_broke[i] = copy_decay * s.copied_broke[i] + @as(f64, @floatFromInt(@intFromBool(kept < offered)));
}
