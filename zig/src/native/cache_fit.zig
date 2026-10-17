//! The prompt cache's budget from memory: up to 16 GiB of what 70% of RAM leaves once the server loads, less a margin.
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");
const mtl = @import("metal");

pub const GiB: u64 = 1 << 30;
/// The share of RAM a served model may hold: the rest stays with macOS and other processes.
pub const SHARE_PERCENT = mtl.ResidencySet.share_percent;
/// Kept back from the cache for buffers that grow with long prompts.
pub const MARGIN: u64 = mtl.ResidencySet.margin;
/// The default budget's most, so a server leaves the rest of the RAM to other apps; --prompt-cache-gib sets more.
pub const DEFAULT_CEILING: u64 = 16 * GiB;

/// The cache's budget, what fits under the cap, and the cap itself (bytes).
pub const Fit = struct { budget: u64, room: u64, cap: u64 };

/// For `total` bytes of RAM and `ready` once loaded: `gib` if it fits (or `over`), else what fits up to the ceiling.
pub fn fit(total: u64, ready: u64, gib: ?f64, over: bool) error{CacheOverCap}!Fit {
    const cap = total / 100 * SHARE_PERCENT;
    const room = cap -| ready -| MARGIN;
    const want = if (gib) |g| (if (g > 0) std.math.lossyCast(u64, g * GiB) else 0) else @min(room, DEFAULT_CEILING);
    if (want > room and !over) return error.CacheOverCap;
    return .{ .budget = want, .room = room, .cap = cap };
}

/// Whether `bytes` held at once and the margin stay inside the 70% share of `total` bytes of RAM (0: unknown, never).
pub fn fitsLoadLimit(bytes: u64, total: u64) bool {
    return residentFits(0, bytes, total);
}

/// Whether a residency set holding `held` bytes may take `added` more inside the 70% limit (RAM null: never).
pub const residentFits = mtl.ResidencySet.fits;

/// This machine's RAM (null where the OS does not say).
pub fn ram() ?u64 {
    return std.process.totalSystemMemory() catch null;
}

extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *anyopaque) c_int;
extern "c" fn getpid() c_int;

/// This process's physical footprint as macOS counts it, Metal buffers included (rusage_info_v4; null elsewhere).
pub fn footprint() ?u64 {
    if (builtin.os.tag != .macos) return null;
    var info: [43]u64 = undefined;
    if (proc_pid_rusage(getpid(), 4, &info) != 0) return null;
    return info[9];
}

/// The Mac hosts' session peak: the footprint's high-water mark since the last `reset_peak`.
var host_peak: std.atomic.Value(u64) = .init(0);

/// The mark after a reading: a reset restarts it at the footprint now, else it climbs to the reading's most.
fn peakAfter(mark: *std.atomic.Value(u64), reading: u64, reset_peak: bool) u64 {
    if (reset_peak) mark.store(reading, .monotonic) else _ = mark.fetchMax(reading, .monotonic);
    return mark.load(.monotonic);
}

/// A Mac host's memory counts from the process footprint; a zero limit means the family has none configured.
pub fn hostMemory(limit_bytes: u64, reset_peak: bool) ?api.Memory {
    const active = footprint() orelse return null;
    const peak = peakAfter(&host_peak, active, reset_peak);
    return .{
        .active = active,
        .cache = 0,
        .peak = peak,
        .pinned = 0,
        .limit = if (limit_bytes > 0) limit_bytes else null,
    };
}

/// The cache plan: `gib` if it fits (or `over`), else what 70% of RAM leaves up to 16 GiB (no reading: `spare` for it).
pub fn budget(gib: ?f64, over: bool, spare: u64, a: std.mem.Allocator, why: *[]const u8, rank: []const u8) !api.PromptCachePlan {
    return budgetBeside(gib, over, spare, 0, a, why, rank);
}

/// budget, with `later` bytes the server takes after load (streams' caches on first use) counted as in use.
pub fn budgetBeside(gib: ?f64, over: bool, spare: u64, later: u64, a: std.mem.Allocator, why: *[]const u8, rank: []const u8) !api.PromptCachePlan {
    const total = ram() orelse 0;
    const loaded = footprint() orelse 0;
    if (total == 0 or loaded == 0) {
        const b = if (gib) |g| (if (g > 0) std.math.lossyCast(u64, g * GiB) else 0) else @min(spare, DEFAULT_CEILING);
        std.log.info("prompt cache: {d:.1} GiB for kept prompt states{s} (no memory reading)", .{ gibs(b), rank });
        return .{ .source = if (gib != null) .explicit else .metal_working_set, .budget_bytes = b, .explicit_budget = gib != null };
    }
    const ready = loaded + later;
    const streams = if (later > 0) try std.fmt.allocPrint(a, " and {d:.1} GiB its streams take on first use", .{gibs(later)}) else "";
    const f = fit(total, ready, gib, over) catch |e| {
        const left = (fit(total, ready, null, false) catch unreachable).room; // without a given budget it never refuses
        why.* = try std.fmt.allocPrint(a, "--prompt-cache-gib {d} would take this server past 70% of this Mac's memory: it holds {d:.1} GiB once loaded{s}, 70% of {d:.0} GiB is {d:.1} GiB, and {d} GiB stays free for prompt buffers, so {d:.1} GiB is left for kept prompt states. Pass a smaller one{s}, leave --prompt-cache-gib out for the default {d:.1} GiB, or add --prompt-cache-over-cap to keep {d} GiB anyway.", .{ gib.?, gibs(loaded), streams, gibs(total), gibs(total / 100 * SHARE_PERCENT), MARGIN >> 30, gibs(left), if (later > 0) ", serve a smaller --parallel" else "", gibs(@min(left, DEFAULT_CEILING)), gib.? });
        return e;
    };
    const ceiling = if (gib == null and f.room > f.budget) std.fmt.comptimePrint("; {d} GiB by default: --prompt-cache-gib takes more, up to what is free", .{DEFAULT_CEILING >> 30}) else "";
    std.log.info("prompt cache: {d:.1} GiB from {d:.1} GiB free under the 70% cap ({d:.1} GiB in use once loaded{s}, {d:.0} GiB of RAM, {d} GiB kept for prompt buffers){s}{s}{s}", .{ gibs(f.budget), gibs(f.room), gibs(loaded), streams, gibs(total), MARGIN >> 30, if (f.budget > f.room) ", past the cap by --prompt-cache-over-cap" else "", rank, ceiling });
    return .{
        .source = .physical_footprint,
        .budget_bytes = f.budget,
        .explicit_budget = gib != null,
        .over_cap = f.budget > f.room,
        .ram_bytes = total,
        .ready_footprint_bytes = loaded,
        .cap_bytes = f.cap,
        .room_bytes = f.room,
        .margin_bytes = MARGIN,
    };
}

/// GiB with one decimal, for log lines and messages.
pub fn gibs(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / GiB;
}

test "the default budget is what 70% of RAM leaves past the footprint at ready and the margin" {
    const ram256: u64 = 256 * GiB;
    const f = try fit(ram256, 172 * GiB, null, false); // this tip on a 256 GB M5 Ultra
    try std.testing.expectEqual(ram256 / 100 * 70, f.cap);
    try std.testing.expectEqual(f.cap - 174 * GiB, f.budget);
    try std.testing.expect(gibs(f.budget) > 5.0 and gibs(f.budget) < 5.3);
    try std.testing.expectEqual(@as(u64, 0), (try fit(ram256, 179 * GiB, null, false)).budget); // nothing left: no cache
    try std.testing.expectEqual(@as(u64, 0), (try fit(192 * GiB, 172 * GiB, null, false)).budget); // a 192 GB Mac: past the cap already
}

test "the default stops at 16 GiB, a given budget takes up to the room, and 0 turns the cache off" {
    const ram128: u64 = 128 * GiB;
    const ready: u64 = 18 * GiB + 3 * GiB / 4; // Nemotron's server on a 128 GB M5 Max
    const f = try fit(ram128, ready, null, false);
    try std.testing.expectEqual(DEFAULT_CEILING, f.budget);
    try std.testing.expect(gibs(f.room) > 68.8 and gibs(f.room) < 68.9);
    try std.testing.expectEqual(@as(u64, 40 * GiB), (try fit(ram128, ready, 40, false)).budget);
    try std.testing.expectEqual(@as(u64, 68 * GiB), (try fit(ram128, ready, 68, false)).budget); // the old default's room
    try std.testing.expectError(error.CacheOverCap, fit(ram128, ready, 69, false));
    try std.testing.expectEqual(@as(u64, 0), (try fit(ram128, ready, 0, false)).budget);
}

test "a given budget that fits is kept, a larger one is refused unless the override is set" {
    const ram256: u64 = 256 * GiB;
    try std.testing.expectEqual(@as(u64, 4 * GiB), (try fit(ram256, 172 * GiB, 4, false)).budget);
    try std.testing.expectEqual(@as(u64, 0), (try fit(ram256, 172 * GiB, 0, false)).budget);
    try std.testing.expectError(error.CacheOverCap, fit(ram256, 172 * GiB, 8, false));
    try std.testing.expectError(error.CacheOverCap, fit(ram256, 172 * GiB, 16, false));
    try std.testing.expectEqual(@as(u64, 16 * GiB), (try fit(ram256, 172 * GiB, 16, true)).budget);
    try std.testing.expectEqual(@as(u64, 64 * GiB), (try fit(512 * GiB, 172 * GiB, 64, false)).budget); // a 512 GB Mac has room
}

test "a load fits only while it and the margin stay inside 70% of RAM" {
    const ram256: u64 = 256 * GiB;
    try std.testing.expect(fitsLoadLimit(172 * GiB, ram256)); // Flash Next's 6-bit checkpoint on a 256 GB M5 Ultra
    try std.testing.expect(!fitsLoadLimit(181 * GiB, ram256)); // the 6 Oct panic's load
    try std.testing.expect(fitsLoadLimit(ram256 / 100 * 70 - MARGIN, ram256));
    try std.testing.expect(!fitsLoadLimit(ram256 / 100 * 70 - MARGIN + 1, ram256));
    try std.testing.expect(!fitsLoadLimit(1, 0)); // no RAM reading: nothing fits
    try std.testing.expect(!fitsLoadLimit(16 * GiB + (5760 << 20), 32 * GiB)); // the 27B and its Neural Engine share on a 32 GB Mac
    try std.testing.expect(fitsLoadLimit(16 * GiB + (5760 << 20), 48 * GiB));
}

test "a residency set takes more only while what it holds, what joins and the margin stay inside 70% of RAM" {
    const ram256: u64 = 256 * GiB;
    const edge = ram256 / 100 * 70 - MARGIN;
    try std.testing.expect(residentFits(0, 172 * GiB, ram256)); // Flash Next 6-bit's set on a 256 GB Mac
    try std.testing.expect(residentFits(172 * GiB, edge - 172 * GiB, ram256)); // kept states up to the edge
    try std.testing.expect(!residentFits(172 * GiB, edge - 172 * GiB + 1, ram256)); // one byte past it
    try std.testing.expect(!residentFits(172 * GiB, 3 << 30, 240 * GiB)); // 3 GiB more on a 240 GiB Mac
    try std.testing.expect(!residentFits(0, 181 * GiB, ram256)); // GLM-5.3-Flash's 4-bit checkpoint, GLM_RESIDENCY
    try std.testing.expect(!residentFits(0, 1, null)); // RAM that could not be read: nothing joins
    try std.testing.expect(!residentFits(0, 1, 0));
    try std.testing.expect(!residentFits(std.math.maxInt(u64), 1, ram256)); // sums saturate, never wrap
    try std.testing.expect(residentFits(edge, 0, ram256));
}

test "the OS readings: RAM and this process's footprint" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const r = ram() orelse return error.SkipZigTest;
    const p = footprint() orelse return error.SkipZigTest;
    try std.testing.expect(r >= 8 * GiB and p > 0 and p < r);
}

test "a host's memory counts keep the footprint, its measured peak and the limit apart" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const limit = 235 * GiB;
    const m = hostMemory(limit, false) orelse return error.SkipZigTest;
    try std.testing.expect(m.active > 0); // this test process's own footprint
    try std.testing.expect(m.peak >= m.active); // the high-water mark covers now
    try std.testing.expectEqual(limit, m.limit.?);
    try std.testing.expectEqual(@as(u64, 0), m.cache);
    try std.testing.expectEqual(@as(u64, 0), m.pinned);
    const after_reset = hostMemory(limit, true).?; // a reset starts a new peak at the footprint now
    try std.testing.expectEqual(after_reset.active, after_reset.peak); // the reset's own reading, not an older mark
    const again = hostMemory(limit, false).?;
    try std.testing.expect(again.peak >= again.active); // the mark covers the footprint now
}

test "the high-water mark rises, holds, resets and the next read climbs from the reset" {
    var mark: std.atomic.Value(u64) = .init(0);
    try std.testing.expectEqual(@as(u64, 100), peakAfter(&mark, 100, false)); // the first read is the mark
    try std.testing.expectEqual(@as(u64, 100), peakAfter(&mark, 60, false)); // a lower reading leaves the mark
    try std.testing.expectEqual(@as(u64, 120), peakAfter(&mark, 120, false)); // a higher one raises it
    try std.testing.expectEqual(@as(u64, 40), peakAfter(&mark, 40, true)); // a reset starts the mark at the reading
    try std.testing.expectEqual(@as(u64, 50), peakAfter(&mark, 50, false)); // the next read reports 50, not 120
    try std.testing.expectEqual(@as(u64, 50), peakAfter(&mark, 45, false)); // and it holds against lower readings
    try std.testing.expectEqual(@as(u64, 70), peakAfter(&mark, 70, false)); // then climbs again
}

test "a host without a footprint reading reports no memory rather than zeros" {
    if (builtin.os.tag == .macos) return error.SkipZigTest; // on macOS the reading always works
    try std.testing.expect(hostMemory(1 << 30, false) == null);
}
