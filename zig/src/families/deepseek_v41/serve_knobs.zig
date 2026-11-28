//! The served engine knobs the Zig engine reads under their shared names; every rank reads the same environment.
const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel

const Config = @import("config.zig").Config;
const knob_types = @import("knob_types.zig");
const PfDense = knob_types.PfDense;

pub const Error = error{BadTable};

/// Reads one setting; null when unset or empty (the environment in `env`, a fixture in tests).
pub const Get = *const fn ([]const u8) ?[]const u8;

/// TF_DSV41_PF_DENSE (off | fused; 0 / no / false = off): the fused path's settings, null when off.
pub fn pfDense(a: std.mem.Allocator, io: std.Io, get: Get) !?*const PfDense {
    if (!try @import("sys").environment.value(bool, "TF_DSV41_PF_DENSE", get("TF_DSV41_PF_DENSE"), false)) return null;
    const p = try a.create(PfDense);
    errdefer a.destroy(p);
    p.* = .{ .override = if (get("TF_DSV41_PF_DENSE_CFG")) |raw| try PfDense.parseOverride(raw) else null };
    if (try @import("sys").environment.path("TF_DSV41_PF_DENSE_TABLE", get("TF_DSV41_PF_DENSE_TABLE"))) |path| {
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24)) catch |e| {
            if (!quiet) std.log.warn("TF_DSV41_PF_DENSE_TABLE={s}: {t}", .{ path, e });
            return e;
        };
        p.entries = PfDense.parseTable(a, bytes) catch |e| {
            if (!quiet) std.log.warn("TF_DSV41_PF_DENSE_TABLE={s}: not a pfdense tuning table ({t})", .{ path, e });
            return e;
        };
    }
    return p;
}

/// pf4k.TARGET: the only prefill segment above 2,048 rows
pub const pf4k_rows: u32 = 4096;

/// TF_DSV41_PF_4K ("0" or "1"): 4,096-row prefill segments, refused without TF_DSV41_PF_4K_MEMORY_OK=1.
pub fn pf4k(get: Get) !bool {
    if (!try @import("sys").environment.value(bool, "TF_DSV41_PF_4K", get("TF_DSV41_PF_4K"), false)) return false;
    if (!try @import("sys").environment.value(bool, "TF_DSV41_PF_4K_MEMORY_OK", get("TF_DSV41_PF_4K_MEMORY_OK"), false)) {
        if (!quiet) std.log.warn("TF_DSV41_PF_4K=1 requires TF_DSV41_PF_4K_MEMORY_OK=1", .{});
        return error.InvalidEnvironment;
    }
    return true;
}

/// A prompt segment's rows: min(TF_DSV41_PREFILL_CHUNK, _ROWS) on the 16-row grid, 32-2,048; 4,096 under PF_4K.
pub fn prefillRows(get: Get) !?u32 {
    const big = try pf4k(get);
    var rows: ?u32 = null;
    for ([_][:0]const u8{ "TF_DSV41_PREFILL_CHUNK", "TF_DSV41_PREFILL_ROWS" }) |name| {
        const v = (try @import("sys").environment.value(?u64, name, get(name), null)) orelse continue;
        if ((big and v != pf4k_rows) or v < 32 or (v > 2048 and !big) or v % 16 != 0) {
            if (!quiet) std.log.warn("{s}={d}: expected multiples of 16 in 32..2048, or exactly 4096 with TF_DSV41_PF_4K=1", .{ name, v });
            return error.InvalidEnvironment;
        }
        rows = if (rows) |r| @min(r, @as(u32, @intCast(v))) else @intCast(v);
    }
    return if (big) pf4k_rows else rows;
}

/// TF_DSV41_PF_OVERLAP (unset / 0 / 1: off): a prefill segment's two exchanges a layer in this many row pieces (2-8).
pub fn pfOverlap(get: Get) !i64 {
    const v = try @import("sys").environment.value(i64, "TF_DSV41_PF_OVERLAP", get("TF_DSV41_PF_OVERLAP"), 0);
    return if (v < 2) 0 else v;
}

/// TF_DSV41_MHC_SITE_ROWS (unset / 0 off, else 256-4,096, a multiple of 16): a prefill boundary site's rows a piece.
pub fn mhcSiteRows(get: Get) !i64 {
    return @import("sys").environment.value(i64, "TF_DSV41_MHC_SITE_ROWS", get("TF_DSV41_MHC_SITE_ROWS"), 0);
}

/// TF_DSV41_GM_GU2 (0 / 1, with TF_DSV41_GM_V2=1): a 4,096-row segment's routed gate / up at x3gm's 128-member tile.
pub fn gmGu2(get: Get) !bool {
    return @import("sys").environment.value(bool, "TF_DSV41_GM_GU2", get("TF_DSV41_GM_GU2"), false);
}

/// TF_DSV41_PF_OVERLAP_SITE (0 / 1, with PF_OVERLAP >= 2): every exchange piece forked, the next site in pieces.
pub fn pfOverlapSite(get: Get) !bool {
    return @import("sys").environment.value(bool, "TF_DSV41_PF_OVERLAP_SITE", get("TF_DSV41_PF_OVERLAP_SITE"), false);
}

/// TF_DSV41_PF_TBO (ours; 0 / 1, default 0): CED encoder segments two at a time on two streams (block_prefill.emitTbo).
pub fn pfTbo(get: Get) !bool {
    return @import("sys").environment.value(bool, "TF_DSV41_PF_TBO", get("TF_DSV41_PF_TBO"), false);
}

/// TF_DSV41_PF_TBO_GM_SMS (under TF_DSV41_PF_TBO): the SMs x3gm's persistent grid covers at most, 8-48; 0: every SM.
pub const tbo_gm_sms: u32 = 0;
pub fn pfTboGmSms(get: Get) !u32 {
    return @import("sys").environment.value(u32, "TF_DSV41_PF_TBO_GM_SMS", get("TF_DSV41_PF_TBO_GM_SMS"), tbo_gm_sms);
}

/// TF_DSV41_STREAM_RB (unset / 0 off, 1, 2 or 4): the stream top-k's twin: 1 `_stream_pf`, 2 / 4 `_stream_rb<N>`.
pub fn streamRb(get: Get) !i64 {
    return @import("sys").environment.value(i64, "TF_DSV41_STREAM_RB", get("TF_DSV41_STREAM_RB"), 0);
}

/// TF_DSV41_INDEX_BOUND (0 by default): decode windows' dense index scores as `_scores_b` (block.Options.index_bound).
pub fn indexBound(get: Get) !bool {
    return @import("sys").environment.value(bool, "TF_DSV41_INDEX_BOUND", get("TF_DSV41_INDEX_BOUND"), false);
}

/// TF_DSV41_PREFETCH_AHEAD (0 by default): the next prompt segment's Engram reads before this one runs.
pub fn prefetchAhead(get: Get) !bool {
    return @import("sys").environment.value(bool, "TF_DSV41_PREFETCH_AHEAD", get("TF_DSV41_PREFETCH_AHEAD"), false);
}

/// TF_DSV41_KV_NORM_STORE=1: KV-only RMS + SWA RoPE/FP8 store, default off.
pub fn kvNormStore(get: Get) !bool {
    return @import("sys").environment.value(bool, "TF_DSV41_KV_NORM_STORE", get("TF_DSV41_KV_NORM_STORE"), false);
}

/// TF_DSV41_FLOAT_LINEAR=1: a projection a pack keeps as a float matrix alone is planned and run as one; default off.
pub fn floatLinear(get: Get) !bool {
    return @import("sys").environment.value(bool, "TF_DSV41_FLOAT_LINEAR", get("TF_DSV41_FLOAT_LINEAR"), false);
}

/// TF_DSV41_ROUTER_GROUP_ROT=1: histogram grouping + input rotation, default off.
pub fn routerGroupRot(get: Get) !bool {
    return @import("sys").environment.value(bool, "TF_DSV41_ROUTER_GROUP_ROT", get("TF_DSV41_ROUTER_GROUP_ROT"), false);
}

/// TF_DSV41_X3LD_EPI (unset / 0 off, 1 on, else refused): the decode MoE's expert epilogues in x3ld's tail.
pub fn x3ldEpi(get: Get) !bool {
    return @import("sys").environment.value(bool, "TF_DSV41_X3LD_EPI", get("TF_DSV41_X3LD_EPI"), false);
}

/// The decode R1 set (ATTN_SPLIT=4 and seven switches at 1), or TF_DSV41_R1 (wins): R1 runs whole or not at all.
pub fn r1(get: Get) !bool {
    if (get("TF_DSV41_R1")) |v| return @import("sys").environment.value(bool, "TF_DSV41_R1", v, false);
    const names = [_][]const u8{ "TF_DSV41_ATTN_SPLIT", "TF_DSV41_ATTN_ROPE", "TF_DSV41_MHC_TAIL", "TF_DSV41_MHC_DEFER", "TF_DSV41_MOE_BF16", "TF_DSV41_PRUNE_KIT", "TF_DSV41_RG_PRUNE", "TF_DSV41_RMS_FOLD" };
    var set: usize = 0;
    var served: usize = 0;
    for (names, 0..) |n, i| if (get(n)) |raw| {
        set += 1;
        if (std.mem.eql(u8, std.mem.trim(u8, raw, " "), if (i == 0) "4" else "1")) served += 1;
    };
    if (served == names.len) return true;
    if (set > 0) if (!quiet) std.log.warn("decode R1: {d} of its {d} Python knobs at the reference's values; Zig runs R1 whole (TF_DSV41_R1=1, or all of ATTN_SPLIT=4 ATTN_ROPE MHC_TAIL MHC_DEFER MOE_BF16 PRUNE_KIT RG_PRUNE RMS_FOLD = 1) or not at all: off", .{ served, names.len });
    return false;
}

/// TF_DSV41_MHC_PFDEC: unset / 0 off, else the fewest rows (1-64) a window's mixing mHC sites take mhc_pf at.
pub fn mhcPfRows(get: Get) !i64 {
    return @import("sys").environment.value(i64, "TF_DSV41_MHC_PFDEC", get("TF_DSV41_MHC_PFDEC"), 0);
}

/// TF_DSV41_INDEX_BUDGET_MIB (1-4,096): the indexer's scores and sort keys a launch; prefill row blocks follow it.
pub fn indexBudget(get: Get) !?i64 {
    const v = (try @import("sys").environment.value(?i64, "TF_DSV41_INDEX_BUDGET_MIB", get("TF_DSV41_INDEX_BUDGET_MIB"), null)) orelse return null;
    return v << 20;
}

/// block_prefill's stream threshold (TF_DSV41_INDEX_STREAM_MIN, 4,096 visible keys): the emitter's constant.
pub const stream_min: i64 = 4096;

/// The long-prompt stream top-k's scratch the buffer plan holds for `limit` positions and `rows`-row segments.
pub fn streamScratch(cfg: *const Config, limit: i64, rows: i64) u64 {
    const longpf = @import("longpf.zig");
    const k: i64 = cfg.index_topk;
    if (rows <= 16 or @popCount(k) != 1) return 0;
    var most: i64 = 0;
    for (0..cfg.layers) |i| {
        const L: u32 = @intCast(i);
        if (cfg.mode(L) != .full or !cfg.isIndexSource(L) or cfg.isCandidateSource(L)) continue;
        const keys = @divFloor(limit, cfg.compressRatio(L));
        if (keys < stream_min) continue;
        most = @max(most, longpf.plan(rows, keys, k) * longpf.splits(keys) * 2 * k * 8);
    }
    return @intCast(most);
}

pub const PrefillMode = enum { full, replay };

/// TF_DSV41_PREFILL: the prompt's prefill mode (full | replay); unset is `full`.
pub fn prefillMode(get: Get) !PrefillMode {
    return @fromBackingInt(@intCast(try @import("sys").environment.value(u8, "TF_DSV41_PREFILL", get("TF_DSV41_PREFILL"), 0)));
}

/// Where a served prompt's last token runs (TF_DSV41_PROMPT_TAIL): `verify` as a decode row (default), or `prefill`.
pub const PromptTail = enum { verify, prefill };

pub fn promptTail(get: Get) !PromptTail {
    return @fromBackingInt(@intCast(try @import("sys").environment.value(u8, "TF_DSV41_PROMPT_TAIL", get("TF_DSV41_PROMPT_TAIL"), 0)));
}

/// A prompt of `n` rows in `tail`'s shape: the rows through the prefill, then the rows of one decode window.
pub fn promptSplit(n: usize, tail: PromptTail) struct { prefill: usize, window: usize } {
    return switch (tail) {
        .prefill => .{ .prefill = n, .window = 0 },
        .verify => .{ .prefill = n -| 1, .window = @min(n, 1) },
    };
}

pub fn env(name: []const u8) ?[]const u8 {
    var buf: [64]u8 = undefined;
    if (name.len >= buf.len) return null;
    @memcpy(buf[0..name.len], name);
    buf[name.len] = 0;
    const v = @import("sys").environment.Process.get(.{}, buf[0..name.len :0]) orelse return null;
    return if (std.mem.trim(u8, v, " \t\r\n").len == 0) null else v;
}
