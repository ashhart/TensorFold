//! The GLM backend's settings from the environment (TF_GLM_*, TF_ROCE*) and its timing helper.
const std = @import("std");
const copy = @import("copy.zig");
const native = @import("native.zig");
const max_rows = native.max_rows;
const draft_confidence = native.draft_confidence;

/// TF_GLM_DRAFT_MARGINAL=p (DFlash2): stop at the first draft whose own chance is below p; 0 keeps the product rule.
pub fn marginalFromEnv() f64 {
    const v = std.c.getenv("TF_GLM_DRAFT_MARGINAL") orelse return 0;
    return std.fmt.parseFloat(f64, std.mem.span(v)) catch 0;
}

pub fn confidenceFromEnv() f64 {
    const v = std.c.getenv("TF_GLM_DRAFT_CONFIDENCE") orelse return draft_confidence;
    return std.fmt.parseFloat(f64, std.mem.span(v)) catch draft_confidence;
}

/// TF_GLM_WINDOW: a stream's verify window at most, 2 to 16 rows (16: long copy drafts verified in one window).
pub fn windowFromEnv() u32 {
    const v = std.c.getenv("TF_GLM_WINDOW") orelse return max_rows;
    const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch return max_rows;
    return @max(2, @min(n, max_rows));
}

pub fn envUsize(name: [:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

pub fn drafterWanted() bool {
    const v = std.c.getenv("TF_GLM_DRAFTER") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "dflash2");
}

/// TF_GLM_DFLASH2_STEP_MS: a DFlash2 round's cost spread over its 7 drafts (1.6 ms: about 11 ms a round).
pub fn dflashStepMs() f64 {
    const v = std.c.getenv("TF_GLM_DFLASH2_STEP_MS") orelse return 1.6;
    return std.fmt.parseFloat(f64, std.mem.span(v)) catch 1.6;
}

pub fn roceWanted() bool {
    const v = std.c.getenv("TF_ROCE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

pub fn roceMaxBytes() usize {
    const v = std.c.getenv("TF_ROCE_MAX_KB") orelse return 512 << 10;
    return (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 512) << 10;
}

pub fn msSince(io: std.Io, t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e6;
}
