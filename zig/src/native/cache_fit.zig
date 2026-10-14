//! The prompt cache's budget from memory: up to 16 GiB of what 70% of RAM leaves once the server loads, less a margin.
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");

pub const GiB: u64 = 1 << 30;
/// The share of RAM a served model may hold: the rest stays with macOS and other processes.
pub const SHARE_PERCENT = 70;
/// Kept back from the cache for buffers that grow with long prompts.
pub const MARGIN: u64 = 2 * GiB;
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

/// This process's peak physical footprint over its lifetime (rusage_info_v4 slot 30; null elsewhere).
pub fn peakFootprint() ?u64 {
    if (builtin.os.tag != .macos) return null;
    var info: [43]u64 = undefined;
    if (proc_pid_rusage(getpid(), 4, &info) != 0) return null;
    return info[30];
}

/// A Mac family host's memory counts for `/stats`, `/memory` and `/metrics`, from the process footprint.
pub fn hostMemory(limit_bytes: u64, reset_peak: bool) ?api.Memory {
    _ = reset_peak;
    const active = footprint() orelse return null;
    const peak = peakFootprint() orelse active;
    return .{ .active = active, .cache = 0, .peak = peak, .pinned = 0, .limit = limit_bytes };
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
    try std.testing.expect(m.peak >= m.active); // the kernel's lifetime peak covers now
    try std.testing.expectEqual(limit, m.limit.?);
    try std.testing.expectEqual(@as(u64, 0), m.cache);
    try std.testing.expectEqual(@as(u64, 0), m.pinned);
    try std.testing.expectEqual(m.active, hostMemory(limit, true).?.active); // reset_peak changes nothing here
    try std.testing.expectEqual(m.peak, hostMemory(limit, true).?.peak);
}

test "a host without a footprint reading reports no memory rather than zeros" {
    if (builtin.os.tag == .macos) return error.SkipZigTest; // on macOS the reading always works
    try std.testing.expect(hostMemory(1 << 30, false) == null);
}
