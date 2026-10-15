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

/// The settings that shape what both ranks run together: each rank reads its own environment, so they are compared
/// over the link before the first collective and a pair that differs does not start.
pub const Agreed = extern struct {
    prefill_rows: i64, // TF_GLM_PREFILL_ROWS: a prompt chunk's rows
    index_ring: i64, // TF_GLM_INDEX_RING: the index rings' rows (before the capacity caps them)
    mix: u8, // TF_GLM_MIX
    split: u8, // TF_GLM_DFLASH2_SPLIT
    drafter: u8, // TF_GLM_DRAFTER=dflash2
    full_head: u8, // TF_GLM_DFLASH2_HEAD=full
    _pad: [4]u8 = @splat(0),

    comptime { // the swap sends every byte: no padding left unset
        std.debug.assert(@sizeOf(Agreed) == 24);
    }

    pub fn fromEnv() Agreed {
        return .{
            .prefill_rows = @import("forward.zig").prefillRows(),
            .index_ring = @import("seq.zig").ringRows(std.math.maxInt(i64)),
            .mix = @intFromBool(mixWanted()),
            .split = @intFromBool(@import("dflash_load.zig").splitWanted()),
            .drafter = @intFromBool(drafterWanted()),
            .full_head = @intFromBool(@import("dflash_load.zig").fullHeadWanted()),
        };
    }
};

/// Both ranks' Agreed over ``link``; error.SettingsDiffer on both when they differ (each logs both sides).
pub fn agree(link: anytype, mine: Agreed) !void {
    var theirs: Agreed = undefined;
    try link.swap(std.mem.asBytes(&mine), std.mem.asBytes(&theirs));
    if (std.meta.eql(mine, theirs)) return;
    std.log.err("glm: the ranks' settings differ (this rank {any}, the other {any}); set TF_GLM_PREFILL_ROWS, TF_GLM_INDEX_RING, TF_GLM_MIX, TF_GLM_DFLASH2_SPLIT, TF_GLM_DRAFTER and TF_GLM_DFLASH2_HEAD alike on both", .{ mine, theirs });
    return error.SettingsDiffer;
}

/// TF_GLM_MIX=0: with DFlash2 loaded, it drafts every round (else each stream picks a source a round).
pub fn mixWanted() bool {
    const v = std.c.getenv("TF_GLM_MIX") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}
