//! GLM_POOL_TOKENS: the shared latent pool sized apart from the per-stream window (unset: the pool is the window).
const std = @import("std");
const platform = @import("sys");
const st = @import("state.zig");
const pool = @import("pool.zig");
const cfg = @import("config.zig");
const load_plan = @import("load_plan.zig");

/// Tokens of cache an engine is loaded with beyond the pool's last whole block's reach.
pub const slack: u32 = 64;

pub const name = "GLM_POOL_TOKENS";

/// The setting, whole and checked by platform/environment.zig; null when unset or blank.
pub fn fromEnv() !?u32 {
    return platform.environment.read(?u32, name, null);
}

/// The caches an engine is loaded with: the window's (unset) or the pool's tokens, and the slack.
pub fn cacheTokens(window: u32, tokens: ?u32) u32 {
    return (tokens orelse window) + slack;
}

/// The setting needs the shared pool: one Mac and GLM_POOL not 0; without it a stream would own the whole pool.
pub fn needsPool(tokens: ?u32, pair: bool) error{PoolTokensNeedPool}!void {
    if (tokens == null) return;
    if (!pair and pool.wanted()) return;
    std.log.err("glm: {s} sizes the shared pool of one Mac: it cannot be used with the two-Mac pair or GLM_POOL=0", .{name});
    return error.PoolTokensNeedPool;
}

/// The tokens the pool holds for an engine loaded with `cap` tokens: whole blocks, less the pooled entry.
pub fn poolTokens(cap: u32, kpool: u32) u32 {
    const p: pool.Pool = .{ .base = undefined, .blocks = cap / pool.Pool.block, .kpool = kpool };
    return p.tokens(p.blocks);
}

/// The most tokens one stream serves: its window, within the pool less the rows a pass holds beside its last window.
pub fn streamWindow(window: u32, capacity: u32) u32 {
    return @min(window, capacity -| (st.max_rows + 1));
}

/// The refusal text when the caches counted for the pool pass the load limit; null when they fit.
pub fn problem(buf: []u8, t: u32, need: usize, used: usize, limit: usize, source: []const u8) ?[]const u8 {
    if (used + need <= limit) return null;
    const gb = struct {
        fn of(b: usize) f64 {
            return @as(f64, @floatFromInt(b)) / 1e9;
        }
    }.of;
    return std.fmt.bufPrint(buf, "{s}={d} needs {d:.1} GB of caches beside {d:.1} GB of weights and images, past this Mac's {d:.1} GB load limit ({s}); " ++
        "lower {s} or raise the limit", .{ name, t, gb(need), gb(used), gb(limit), source, name }) catch name;
}

/// A load's early refusal text, before any cache is allocated: the pool's least caches, counted, against `limit`.
pub fn check(buf: []u8, gpa: std.mem.Allocator, c: *const cfg.Config, t: u32, cap: u32, chunked: bool, used: usize, limit: usize, source: []const u8) !?[]const u8 {
    return problem(buf, t, try load_plan.leastArena(gpa, c, cap, chunked), used, limit, source);
}

/// The startup line's context words: the window alone, or a stream's window and the pool it shares.
pub fn contextWords(buf: []u8, window: u32, tokens: ?u32, capacity: u32) []const u8 {
    const t = tokens orelse return std.fmt.bufPrint(buf, "context {d} tokens", .{window}) catch "context";
    return std.fmt.bufPrint(buf, "context {d} tokens a stream in a shared pool of {d} tokens ({s}={d})", .{ window, capacity, name, t }) catch "context";
}
