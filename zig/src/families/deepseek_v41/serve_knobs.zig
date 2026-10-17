//! The served engine knobs the Zig engine reads under their shared names; every rank reads the same environment.
const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const dk = @import("dsv41_kernels");
const Config = @import("config.zig").Config;
const knob_types = @import("knob_types.zig");
const PfDense = knob_types.PfDense;

const pfd = dk.ops.pfd;

pub const Error = error{ BadKnob, BadTable };

/// Reads one setting; null when unset or empty (the environment in `env`, a fixture in tests).
pub const Get = *const fn ([]const u8) ?[]const u8;

/// TF_DSV41_PF_DENSE (off | fused; 0 / no / false = off): the fused path's settings, null when off.
pub fn pfDense(a: std.mem.Allocator, io: std.Io, get: Get) !?*const PfDense {
    const raw = get("TF_DSV41_PF_DENSE") orelse return null;
    const v = std.mem.trim(u8, raw, " ");
    const off = [_][]const u8{ "off", "0", "no", "false" };
    for (off) |o| if (std.ascii.eqlIgnoreCase(v, o)) return null;
    if (!std.ascii.eqlIgnoreCase(v, "fused")) {
        if (!quiet) std.log.warn("TF_DSV41_PF_DENSE={s}: expected off or fused", .{raw});
        return error.BadKnob;
    }
    const p = try a.create(PfDense);
    p.* = .{};
    if (get("TF_DSV41_PF_DENSE_CFG")) |c| p.override = PfDense.parseOverride(c) catch {
        if (!quiet) std.log.warn("TF_DSV41_PF_DENSE_CFG={s}: expected id[,group], id 0-{d}, group 1-1024", .{ c, pfd.cfgs.len - 1 });
        return error.BadKnob;
    };
    if (get("TF_DSV41_PF_DENSE_TABLE")) |path| {
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
    const raw = std.mem.trim(u8, get("TF_DSV41_PF_4K") orelse "0", " ");
    if (std.mem.eql(u8, raw, "0")) return false;
    if (!std.mem.eql(u8, raw, "1")) {
        if (!quiet) std.log.warn("TF_DSV41_PF_4K={s}: 0 or 1", .{raw});
        return error.BadKnob;
    }
    const ok = std.mem.trim(u8, get("TF_DSV41_PF_4K_MEMORY_OK") orelse "", " ");
    if (!std.mem.eql(u8, ok, "1")) {
        if (!quiet) std.log.warn("TF_DSV41_PF_4K=1: 4,096-row prefill segments hold about 0.66 GiB more worker memory than 2,048-row ones; set TF_DSV41_PF_4K_MEMORY_OK=1 to acknowledge", .{});
        return error.BadKnob;
    }
    return true;
}

/// A prompt segment's rows: min(TF_DSV41_PREFILL_CHUNK, _ROWS) on the 16-row grid, 32-2,048; 4,096 under PF_4K.
pub fn prefillRows(get: Get) !?u32 {
    const big = try pf4k(get);
    var rows: ?u32 = null;
    for ([_][]const u8{ "TF_DSV41_PREFILL_CHUNK", "TF_DSV41_PREFILL_ROWS" }) |name| if (get(name)) |raw| {
        const v = std.fmt.parseInt(u32, std.mem.trim(u8, raw, " "), 10) catch return badRows(name, raw, big);
        if (big and v != pf4k_rows) return badRows(name, raw, big);
        if (v < 32 or (v > 2048 and !big) or v % 16 != 0) return badRows(name, raw, big);
        rows = if (rows) |r| @min(r, v) else v;
    };
    return if (big) pf4k_rows else rows;
}

fn badRows(name: []const u8, raw: []const u8, big: bool) error{BadKnob} {
    if (!quiet) if (big) std.log.warn("{s}={s}: TF_DSV41_PF_4K=1 means 4,096-row segments (unset it or set 4096)", .{ name, raw }) else std.log.warn("{s}={s}: expected a multiple of 16 in 32..2048", .{ name, raw });
    return error.BadKnob;
}

/// TF_DSV41_PF_OVERLAP (unset / 0 / 1: off): a prefill segment's two exchanges a layer in this many row pieces (2-8).
pub fn pfOverlap(get: Get) !i64 {
    const raw = std.mem.trim(u8, get("TF_DSV41_PF_OVERLAP") orelse "0", " ");
    const v = std.fmt.parseInt(i64, raw, 10) catch -1;
    if (v < 0 or v > 8) {
        if (!quiet) std.log.warn("TF_DSV41_PF_OVERLAP={s}: 0 (off) or row pieces 2-8", .{raw});
        return error.BadKnob;
    }
    return if (v < 2) 0 else v;
}

/// TF_DSV41_MHC_SITE_ROWS (unset / 0 off, else 256-4,096, a multiple of 16): a prefill boundary site's rows a piece.
pub fn mhcSiteRows(get: Get) !i64 {
    const raw = std.mem.trim(u8, get("TF_DSV41_MHC_SITE_ROWS") orelse "0", " ");
    const v = std.fmt.parseInt(i64, raw, 10) catch -1;
    if (v == 0) return 0;
    if (v < 256 or v > 4096 or @rem(v, 16) != 0) {
        if (!quiet) std.log.warn("TF_DSV41_MHC_SITE_ROWS={s}: 0 (off) or 256-4096, a multiple of 16", .{raw});
        return error.BadKnob;
    }
    return v;
}

/// TF_DSV41_GM_GU2 (0 / 1, with TF_DSV41_GM_V2=1): a 4,096-row segment's routed gate / up at x3gm's 128-member tile.
pub fn gmGu2(get: Get) !bool {
    const raw = std.mem.trim(u8, get("TF_DSV41_GM_GU2") orelse "0", " ");
    if (std.mem.eql(u8, raw, "0")) return false;
    if (std.mem.eql(u8, raw, "1")) return true;
    if (!quiet) std.log.warn("TF_DSV41_GM_GU2={s}: 0 or 1", .{raw});
    return error.BadKnob;
}

/// TF_DSV41_PF_OVERLAP_SITE (0 / 1, with PF_OVERLAP >= 2): every exchange piece forked, the next site in pieces.
pub fn pfOverlapSite(get: Get) !bool {
    const raw = std.mem.trim(u8, get("TF_DSV41_PF_OVERLAP_SITE") orelse "0", " ");
    if (std.mem.eql(u8, raw, "0")) return false;
    if (std.mem.eql(u8, raw, "1")) return true;
    if (!quiet) std.log.warn("TF_DSV41_PF_OVERLAP_SITE={s}: 0 or 1", .{raw});
    return error.BadKnob;
}

/// TF_DSV41_PF_TBO (ours; 0 / 1, default 0): CED encoder segments two at a time on two streams (block_prefill.emitTbo).
pub fn pfTbo(get: Get) !bool {
    const raw = std.mem.trim(u8, get("TF_DSV41_PF_TBO") orelse "0", " ");
    if (std.mem.eql(u8, raw, "0")) return false;
    if (std.mem.eql(u8, raw, "1")) return true;
    if (!quiet) std.log.warn("TF_DSV41_PF_TBO={s}: 0 or 1", .{raw});
    return error.BadKnob;
}

/// TF_DSV41_PF_TBO_GM_SMS (under TF_DSV41_PF_TBO): the SMs x3gm's persistent grid covers at most, 8-48; 0: every SM.
pub const tbo_gm_sms: u32 = 0;
pub fn pfTboGmSms(get: Get) !u32 {
    const raw = std.mem.trim(u8, get("TF_DSV41_PF_TBO_GM_SMS") orelse return tbo_gm_sms, " ");
    const v = std.fmt.parseInt(u32, raw, 10) catch return badSms(raw);
    if (v != 0 and (v < 8 or v > 48)) return badSms(raw);
    return v;
}

fn badSms(raw: []const u8) error{BadKnob} {
    if (!quiet) std.log.warn("TF_DSV41_PF_TBO_GM_SMS={s}: 0 (every SM) or 8-48", .{raw});
    return error.BadKnob;
}

/// TF_DSV41_STREAM_RB (unset / 0 off, 1, 2 or 4): the stream top-k's twin: 1 `_stream_pf`, 2 / 4 `_stream_rb<N>`.
pub fn streamRb(get: Get) !i64 {
    const raw = std.mem.trim(u8, get("TF_DSV41_STREAM_RB") orelse "0", " ");
    if (std.mem.eql(u8, raw, "0")) return 0;
    if (std.mem.eql(u8, raw, "1")) return 1;
    if (std.mem.eql(u8, raw, "2")) return 2;
    if (std.mem.eql(u8, raw, "4")) return 4;
    if (!quiet) std.log.warn("TF_DSV41_STREAM_RB={s}: 0, 1, 2 or 4", .{raw});
    return error.BadKnob;
}

/// TF_DSV41_INDEX_BOUND (0 by default): decode windows' dense index scores as `_scores_b` (block.Options.index_bound).
pub fn indexBound(get: Get) !bool {
    const raw = std.mem.trim(u8, get("TF_DSV41_INDEX_BOUND") orelse "0", " ");
    if (std.mem.eql(u8, raw, "0")) return false;
    if (std.mem.eql(u8, raw, "1")) return true;
    if (!quiet) std.log.warn("TF_DSV41_INDEX_BOUND={s}: 0 or 1", .{raw});
    return error.BadKnob;
}

/// TF_DSV41_PREFETCH_AHEAD (0 by default): the next prompt segment's Engram reads before this one runs.
pub fn prefetchAhead(get: Get) bool {
    const v = get("TF_DSV41_PREFETCH_AHEAD") orelse return false;
    return !std.mem.eql(u8, std.mem.trim(u8, v, " "), "0");
}

/// TF_DSV41_KV_NORM_STORE=1: KV-only RMS + SWA RoPE/FP8 store, default off.
pub fn kvNormStore(get: Get) bool {
    const v = get("TF_DSV41_KV_NORM_STORE") orelse return false;
    return std.mem.eql(u8, std.mem.trim(u8, v, " "), "1");
}

/// TF_DSV41_ROUTER_GROUP_ROT=1: histogram grouping + input rotation, default off.
pub fn routerGroupRot(get: Get) bool {
    const v = get("TF_DSV41_ROUTER_GROUP_ROT") orelse return false;
    return std.mem.eql(u8, std.mem.trim(u8, v, " "), "1");
}

/// TF_DSV41_X3LD_EPI (unset / 0 off, 1 on, else refused): the decode MoE's expert epilogues in x3ld's tail.
pub fn x3ldEpi(get: Get) !bool {
    const raw = std.mem.trim(u8, get("TF_DSV41_X3LD_EPI") orelse "0", " ");
    if (std.mem.eql(u8, raw, "0")) return false;
    if (std.mem.eql(u8, raw, "1")) return true;
    if (!quiet) std.log.warn("TF_DSV41_X3LD_EPI={s}: 0 or 1", .{raw});
    return error.BadKnob;
}

/// The decode R1 set (ATTN_SPLIT=4 and seven switches at 1), or TF_DSV41_R1 (wins): R1 runs whole or not at all.
pub fn r1(get: Get) bool {
    if (get("TF_DSV41_R1")) |v| return std.mem.eql(u8, std.mem.trim(u8, v, " "), "1");
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
    const raw = std.mem.trim(u8, get("TF_DSV41_MHC_PFDEC") orelse return 0, " \t");
    const v = std.fmt.parseInt(i64, raw, 10) catch -1;
    if (v < 0 or v > 64) {
        if (!quiet) std.log.warn("TF_DSV41_MHC_PFDEC={s}: expected 0 (off) or the fewest rows, 1..64", .{raw});
        return error.BadKnob;
    }
    return v;
}

/// TF_DSV41_INDEX_BUDGET_MIB (1-4,096): the indexer's scores and sort keys a launch; prefill row blocks follow it.
pub fn indexBudget(get: Get) !?i64 {
    const raw = get("TF_DSV41_INDEX_BUDGET_MIB") orelse return null;
    const v = std.fmt.parseInt(i64, std.mem.trim(u8, raw, " "), 10) catch return badBudget(raw);
    if (v < 1 or v > 4096) return badBudget(raw);
    return v << 20;
}

fn badBudget(raw: []const u8) error{BadKnob} {
    if (!quiet) std.log.warn("TF_DSV41_INDEX_BUDGET_MIB={s}: expected 1..4096", .{raw});
    return error.BadKnob;
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
    const raw = get("TF_DSV41_PREFILL") orelse return .full;
    return std.meta.stringToEnum(PrefillMode, std.mem.trim(u8, raw, " ")) orelse {
        if (!quiet) std.log.warn("TF_DSV41_PREFILL={s}: expected full or replay", .{raw});
        return error.BadKnob;
    };
}

/// Where a served prompt's last token runs (TF_DSV41_PROMPT_TAIL): `verify` as a decode row (default), or `prefill`.
pub const PromptTail = enum { verify, prefill };

pub fn promptTail(get: Get) !PromptTail {
    const raw = get("TF_DSV41_PROMPT_TAIL") orelse return .verify;
    return std.meta.stringToEnum(PromptTail, std.mem.trim(u8, raw, " ")) orelse {
        if (!quiet) std.log.warn("TF_DSV41_PROMPT_TAIL={s}: expected verify or prefill", .{raw});
        return error.BadKnob;
    };
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
    const v = std.mem.span(std.c.getenv(buf[0..name.len :0]) orelse return null);
    return if (v.len == 0) null else v;
}
