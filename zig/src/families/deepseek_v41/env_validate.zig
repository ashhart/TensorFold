//! Validate optional DeepSeek settings before opening resources, including settings read only by later paths.
const std = @import("std");
const strict = @import("sys").strict_env;
pub const Get = *const fn ([:0]const u8) ?[]const u8;
const Flag = struct { name: [:0]const u8, mode: strict.Flag = .{}, sample: []const u8 = "1" };
pub const flags = [_]Flag{
    .{ .name = "TF_DSV41_MHC_DEFER", .mode = .{ .trim = " \t", .words = true, .empty_off = true } },
    .{ .name = "TF_DSV41_COEF_LATE", .mode = .{ .trim = " \t", .words = true, .empty_off = true } },
    .{ .name = "TF_DSV41_LOOKUP", .mode = .{ .trim = " \t", .words = true, .no = true, .empty_off = true } },
    .{ .name = "TF_DSV41_LOOKUP_PLAN", .mode = .{ .trim = " \t", .words = true, .no = true, .empty_off = true } },
    .{ .name = "TF_DSV41_GM_V3" },
    .{ .name = "TF_DSV41_OWN_PREFILL" },
    .{ .name = "TF_DSV41_DRAFTS" },
    .{ .name = "TF_DSV41_GRAPHS" },
    .{ .name = "TF_DSV41_AOT_GUARD" },
    .{ .name = "TF_DSV41_ENGRAM_PREFETCH" },
    .{ .name = "TF_DSV41_BIAS_VL", .mode = .{ .trim = " ", .empty_off = true } },
    .{ .name = "TF_DSV41_FLOOR_PRICE", .mode = .{ .trim = " " } },
    .{ .name = "TF_DSV41_SESSIONS", .mode = .{ .trim = " " } },
    .{ .name = "TF_DSV41_SAMPLING" },
    .{ .name = "TF_DSV41_SLOT_DRAFTS" },
    .{ .name = "TF_DSV41_FORMS_INPLACE", .mode = .{ .trim = " " } },
    .{ .name = "TF_DSV41_EX_STACKS", .mode = .{ .trim = " " } },
    .{ .name = "TF_DSV41_STREAM_LAZY", .mode = .{ .trim = " " } },
    .{ .name = "TF_DSV41_GRAPH_LOG" },
    .{ .name = "TF_DSV41_WIN_PROF" },
    .{ .name = "TF_DSV41_BOOT_TRACE", .mode = .{ .trim = " " } },
    .{ .name = "TF_DSV41_PREFILL_PIECES", .mode = .{ .trim = " " } },
    .{ .name = "TF_DSV41_GREEDY_GPU" },
    .{ .name = "TF_DSV41_DRAFT_TIMED" },
    .{ .name = "TF_DSV41_SYNC_EACH" },
    .{ .name = "TF_DSV41_ENGRAM_POLL" },
    .{ .name = "TF_DSV41_ENGRAM_AIO" },
    .{ .name = "TF_DSV41_ENGRAM_URING" },
    .{ .name = "TF_DSV41_KV_SPLIT" },
    .{ .name = "TF_DSV41_NEEDS_POOL" },
    .{ .name = "TF_DSV41_KV_SPLIT_DEDUP", .mode = .{ .trim = " \t" } },
    .{ .name = "TF_DSV41_KV_SPLIT_EARLY", .mode = .{ .trim = " \t" } },
    .{ .name = "TF_DSV41_KV_SPLIT_HISTORY", .mode = .{ .trim = " \t" } },
    .{ .name = "TF_DSV41_PREFILL_ADAPT", .mode = .{ .trim = " " } },
    .{ .name = "TF_DSV41_R1", .mode = .{ .trim = " " } },
    .{ .name = "TF_DSV41_PREFETCH_AHEAD", .mode = .{ .trim = " " } },
    .{ .name = "TF_DSV41_KV_NORM_STORE", .mode = .{ .trim = " " } },
    .{ .name = "TF_DSV41_ROUTER_GROUP_ROT", .mode = .{ .trim = " " } },
    .{ .name = "TF_DSV41_SCHEMA_PROMPT", .mode = .{ .trim = " \t\r\n" } },
    .{ .name = "TF_DSV41_TOOL_GRAMMAR", .mode = .{ .trim = " " } },
    .{ .name = "TF_DSV41_PIECE_RUNS" },
    .{ .name = "TF_DSV41_REPLAY_RUNS" },
    .{ .name = "TF_DSV41_JOIN_DRAFTS" },
    .{ .name = "TF_DSV41_TAIL_JOIN" },
    .{ .name = "TF_DSV41_PF_WS_SHARE" },
    .{ .name = "TF_DSV41_SESSION_SKIP_COVERED", .mode = .{ .trim = " " } },
    .{ .name = "TF_DSV41_SESSION_DELTA", .mode = .{ .trim = " " } },
    .{ .name = "TF_DSV41_SESSION_SLOTS", .mode = .{ .trim = " " } },
};
const Int = struct { name: [:0]const u8, min: u64 = 0, max: u64 = std.math.maxInt(u64), sample: []const u8 = "1", trim: bool = false };
pub const integers = [_]Int{
    .{ .name = "TF_DSV41_ENGRAM_QD", .max = std.math.maxInt(u32) },
    .{ .name = "TF_DSV41_ENGRAM_READERS", .max = 32 },
    .{ .name = "TF_DSV41_GEN_STREAMS", .min = 1 },
    .{ .name = "TF_DSV41_CONTEXT", .min = 1, .max = std.math.maxInt(i64) },
    .{ .name = "TF_DSV41_SLOTS", .min = 1, .max = 16 },
    .{ .name = "TF_DSV41_NUCLEUS", .max = std.math.maxInt(u32) },
    .{ .name = "TF_DSV41_POOL_TOKENS" },
    .{ .name = "TF_DSV41_KV_SPLIT_UNION_MIB", .max = std.math.maxInt(u64) >> 20 },
    .{ .name = "TF_DSV41_PREFILL_ROWS", .trim = true },
    .{ .name = "TF_DSV41_INDEX_BUDGET_MIB", .min = 1, .max = 4096, .trim = true },
    .{ .name = "TF_DSV41_BATCH_SHORT", .trim = true },
    .{ .name = "TF_DSV41_LONG_PROMPT", .min = 1, .trim = true },
    .{ .name = "TF_DSV41_PREFILL_CONCURRENT", .trim = true },
    .{ .name = "TF_DSV41_PREFILL_ADAPT_MIN", .min = 16, .sample = "16", .trim = true },
};
const Real = struct { name: [:0]const u8, min: f64 = 0, max: f64 = std.math.floatMax(f64), sample: []const u8 = "1", trim: bool = false };
pub const reals = [_]Real{
    .{ .name = "TF_DSV41_FLOOR_HARD_GIB", .min = 0x1p-1074 },
    .{ .name = "TF_DSV41_PF_WS_FLOOR_GIB" },
    .{ .name = "TF_DSV41_GRAPH_FLOOR_GIB" },
    .{ .name = "TF_DSV41_GRAPH_COST_MIB" },
    .{ .name = "TF_DSV41_EXPERT_TOPP", .min = -std.math.floatMax(f64) },
    .{ .name = "TF_DSV41_PREFILL_SHARE", .min = 0x1p-1074, .max = 1, .trim = true },
    .{ .name = "TF_DSV41_FLOOR_GIB", .min = 0x1p-1074, .trim = true },
    .{ .name = "TF_DSV41_FLOOR_PF_CHUNK_GIB", .trim = true },
    .{ .name = "TF_DSV41_FLOOR_PF_INDEX_GIB", .trim = true },
    .{ .name = "TF_DSV41_FLOOR_PF_HOST_GIB", .trim = true },
    .{ .name = "TF_DSV41_FLOOR_PF_OTHER_GIB", .trim = true },
    .{ .name = "TF_DSV41_ADMIT_FREE_FLOOR_GIB", .trim = true },
    .{ .name = "TF_DSV41_FLOOR_REFUSE_S", .trim = true },
    .{ .name = "TF_DSV41_ADMIT_CACHE_KEEP_GB", .trim = true },
    .{ .name = "TF_DSV41_PREFILL_ADAPT_GIB", .trim = true },
    .{ .name = "TF_DSV41_PREFILL_ADAPT_UP_GIB", .trim = true },
};

/// The same parser produces startup refusals and each individual reader's diagnostic.
pub fn validate(p: *strict.Parser, get: Get) !void {
    for (flags) |f| {
        if (std.mem.eql(u8, f.name, "TF_DSV41_TOOL_GRAMMAR") and get("TENSORFOLD_TOOL_GRAMMAR") != null) continue;
        _ = try p.flag(f.name, get(f.name), false, f.mode);
    }
    inline for (integers) |n| {
        if (get(n.name)) |raw| {
            _ = p.integer(u64, n.name, if (n.trim) std.mem.trim(u8, raw, " ") else raw, 0, n.min, n.max, comptime std.fmt.comptimePrint("whole number {d}..{d}", .{ n.min, n.max })) catch |e| {
                p.problem.?.value = raw;
                return e;
            };
        }
    }
    inline for (reals) |n| {
        if (get(n.name)) |raw| {
            _ = p.real(n.name, if (n.trim) std.mem.trim(u8, raw, " ") else raw, 0, n.min, n.max, if (n.min < 0) "finite number" else if (n.max == 1) "finite number greater than 0 and at most 1" else if (n.min > 0) "finite number greater than 0" else "finite number at least 0") catch |e| {
                p.problem.?.value = raw;
                return e;
            };
        }
    }
    if (get("TF_DSV41_IMAGES")) |raw| {
        const v = std.mem.trim(u8, raw, " ");
        if (!std.mem.eql(u8, v, "native") and !std.mem.eql(u8, v, "0")) return p.reject("TF_DSV41_IMAGES", raw, "native or 0");
    }
    if (get("TF_DSV41_LAYERS")) |raw| _ = try layerRange(p, raw);
    if (get("TF_DSV41_ROWS_CAP")) |raw| {
        const v = try p.integer(u32, "TF_DSV41_ROWS_CAP", raw, 16, 1, 64, "1, 2, 4, 8, 16, 32 or 64");
        if (v & (v - 1) != 0) return p.reject("TF_DSV41_ROWS_CAP", raw, "1, 2, 4, 8, 16, 32 or 64");
    }
}

pub fn layerRange(p: *strict.Parser, raw: []const u8) ![2]u32 {
    const accepted = "lo-hi, unsigned integers with lo <= hi";
    var it = std.mem.splitScalar(u8, raw, '-');
    const lo = std.fmt.parseInt(u32, it.next() orelse "", 10) catch return p.reject("TF_DSV41_LAYERS", raw, accepted);
    const hi = std.fmt.parseInt(u32, it.next() orelse "", 10) catch return p.reject("TF_DSV41_LAYERS", raw, accepted);
    if (it.next() != null or hi < lo) return p.reject("TF_DSV41_LAYERS", raw, accepted);
    return .{ lo, hi };
}

pub fn startup() !void {
    var p: strict.Parser = .{};
    validate(&p, strict.get) catch |e| {
        p.log();
        return e;
    };
}

test {
    _ = @import("env_validate_test.zig");
}

/// BIAS_VL has always selected the bias when nonempty, including 0; validation does not reinterpret that value.
pub fn imageBiasEnv() !bool {
    const raw = strict.get("TF_DSV41_BIAS_VL") orelse return false;
    _ = try strict.flag("TF_DSV41_BIAS_VL", raw, false, .{ .trim = " ", .empty_off = true });
    return std.mem.trim(u8, raw, " ").len > 0;
}
