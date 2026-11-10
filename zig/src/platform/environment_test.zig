//! Startup refusals carry names, values and accepted forms; valid settings preserve defaults and clamps.
const std = @import("std");
const e = @import("environment.zig");
const t = std.testing;
const a = t.allocator;
const all: e.Flags = .{ .cuda = true, .metal = true, .xpu = true, .slide = true, .hooks = true, .dsv41 = true, .gemma4 = true, .drama3 = true, .glm = true, .qwen = true };
const Env = struct {
    name: []const u8,
    raw: ?[]const u8,
    pub fn get(v: Env, name: []const u8) ?[]const u8 {
        if (std.mem.eql(u8, name, v.name)) return v.raw;
        if (std.mem.eql(u8, name, "TF_CUDA_CARVEOUT")) return "1";
        return null;
    }
};

test "environment: every strict reader refuses bad text before opening a model, naming value and accepted forms" {
    for (e.rules) |r| {
        try t.expectEqual(@as(?u64, null), try e.parse(r.name, null));
        const bad: []const []const u8 = if (r.kind == .path) &.{ "key\nfile", "key\x1b[2Kfile", "\x7f" } else if (e.isReal(r.kind)) &.{ "typo", "nan", "-inf", "1\nwrong" } else &.{ "typo", "-1", "18446744073709551616", "1\nwrong" };
        for (bad) |raw| {
            try t.expectError(error.InvalidEnvironment, e.parse(r.name, raw));
            const why = (try e.refusal(a, Env{ .name = r.name, .raw = raw }, all)).?;
            defer a.free(why);
            try t.expect(std.mem.indexOf(u8, why, r.name) != null);
            try t.expect(std.mem.indexOf(u8, why, r.forms) != null);
            if (for (raw) |ch| {
                if (ch < 0x20 or ch == 0x7f) break false;
            } else true) try t.expect(std.mem.indexOf(u8, why, raw) != null);
            for (why) |byte| try t.expect(byte >= 0x20 and byte <= 0x7e);
        }
        const untouched = try e.refusal(a, Env{ .name = r.name, .raw = null }, all);
        try t.expect(untouched == null);
    }
}

const Case = struct { name: []const u8, raw: []const u8, n: u64 };
const valid = [_]Case{
    .{ .name = "TENSORFOLD_VIDEO_TOKENS", .raw = "0", .n = 0 },
    .{ .name = "TENSORFOLD_VIDEO_TOKENS", .raw = "18446744073709551615", .n = std.math.maxInt(u64) },
    .{ .name = "TENSORFOLD_MAX_PDF_PAGES", .raw = "4294967295", .n = std.math.maxInt(u32) },
    .{ .name = "TENSORFOLD_PRESSURE_CACHE_PERCENT", .raw = "\t100 ", .n = 100 },
    .{ .name = "TENSORFOLD_PRESSURE_CACHE_PERCENT", .raw = "0", .n = 0 },
    .{ .name = "TENSORFOLD_PREFIX_WAIT_MS", .raw = "\t4294967295 ", .n = std.math.maxInt(u32) },
    .{ .name = "TENSORFOLD_RESTORE_PREFETCH", .raw = " 0\t", .n = 0 },
    .{ .name = "TENSORFOLD_RESTORE_PREFETCH", .raw = "65537", .n = 65537 },
    .{ .name = "TENSORFOLD_RESTORE_PREFETCH_GIB", .raw = "16777217", .n = 16777217 },
    .{ .name = "TENSORFOLD_RESTORE_PREFETCH_MS", .raw = "1099511627777", .n = 1099511627777 },
    .{ .name = "TENSORFOLD_TOOL_ID_KEY_FILE", .raw = "/etc/tensorfold/tool id.key", .n = 0 },
    .{ .name = "TF_STARTUP_RECEIPT", .raw = "0", .n = 0 },
    .{ .name = "TF_STARTUP_RECEIPT", .raw = "1", .n = 1 },
    .{ .name = "TF_CUDA_DEVICE", .raw = " 3 ", .n = 3 },
    .{ .name = "TF_CUDA_DEVICE", .raw = "4294967295", .n = std.math.maxInt(u32) },
    .{ .name = "TF_CUDA_SEGMENTS", .raw = " 1 ", .n = 1 },
    .{ .name = "TF_CUDA_SEGMENTS", .raw = "4", .n = 4 },
    .{ .name = "TF_CUDA_CARVEOUT", .raw = "0", .n = 0 },
    .{ .name = "TF_CUDA_CARVEOUT", .raw = "1", .n = 1 },
    .{ .name = "TF_CUDA_CARVEOUT_MIB", .raw = "1792", .n = 1792 },
    .{ .name = "TF_MTP_DRAFTS", .raw = "0", .n = 0 },
    .{ .name = "TF_MTP_DRAFTS", .raw = "4294967295", .n = std.math.maxInt(u32) },
    .{ .name = "TF_DRAFT_NODES", .raw = "0", .n = 0 },
    .{ .name = "TF_DRAFT_NODES", .raw = "64", .n = 64 },
    .{ .name = "TF_QMMF_HALF_BELOW", .raw = "0", .n = 0 },
    .{ .name = "TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS", .raw = "0", .n = 0 },
    .{ .name = "TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS", .raw = "1", .n = 1 },
    .{ .name = "TF_ROCE_GID_INDEX", .raw = "2147483647", .n = 2147483647 },
    .{ .name = "TF_DRAMA_PREFILL_ROWS", .raw = " 128 ", .n = 128 },
    .{ .name = "TF_DRAMA_REFERENCE_CACHE", .raw = "0", .n = 0 },
    .{ .name = "TF_GEMMA4_CHUNK_SOLO", .raw = "2048", .n = 2048 },
    .{ .name = "TF_GEMMA4_CHUNK_SOLO", .raw = "0", .n = 0 },
    .{ .name = "TF_GEMMA4_CHUNK_SOLO", .raw = "1000", .n = 1000 },
    .{ .name = "TF_GEMMA4_CHUNK_SOLO", .raw = "8192", .n = 8192 },
    .{ .name = "TF_DRAMA_PREFILL_ROWS", .raw = "1025", .n = 1025 },
    .{ .name = "TF_DRAMA_REFERENCE_CACHE", .raw = "17", .n = 17 },
    .{ .name = "TF_Q27_COPY_ROWS", .raw = "0", .n = 0 },
    .{ .name = "TF_Q27_COPY_ROWS", .raw = "48", .n = 48 },
    .{ .name = "TF_Q27_DRAFT_SLOTS", .raw = "0", .n = 0 },
    .{ .name = "TF_Q27_DRAFT_SLOTS", .raw = "4294967295", .n = std.math.maxInt(u32) },
    .{ .name = "TF_HEAD_GEOMETRY", .raw = "0", .n = 0 },
    .{ .name = "TF_PROMPT_ROWS", .raw = "128", .n = 128 },
    .{ .name = "TF_PROMPT_ROWS", .raw = "2048", .n = 2048 },
    .{ .name = "TF_ANE_COLUMNS", .raw = "0", .n = 0 },
    .{ .name = "TF_ANE_COLUMNS", .raw = "3072", .n = 3072 },
    .{ .name = "TF_FLASHNEXT_SCALE_INDEX", .raw = "0", .n = 0 },
    .{ .name = "TF_FLASHNEXT_SCALE_INDEX", .raw = "1", .n = 1 },
    .{ .name = "GLM_EHPROJ_NAX", .raw = "0", .n = 0 },
    .{ .name = "GLM_EHPROJ_NAX", .raw = "1", .n = 1 },
    .{ .name = "XPU_A8_BITS", .raw = "4", .n = 4 },
    .{ .name = "XPU_A8_BITS", .raw = "8", .n = 8 },
    .{ .name = "LFM3_FUSE", .raw = "7", .n = 7 },
    .{ .name = "TF_SLIDE_POCKET_KEY", .raw = "auto", .n = 0 },
    .{ .name = "TF_SLIDE_POCKET_KEY", .raw = "reply", .n = 1 },
    .{ .name = "TF_SLIDE_POCKET_KEY", .raw = "subject", .n = 2 },
    .{ .name = "TF_SLIDE_POCKET_KEY", .raw = "both", .n = 3 },
    .{ .name = "TF_SLIDE_FIT_GIB", .raw = "4", .n = 4 },
    .{ .name = "TF_SLIDE_POCKET_DUMP", .raw = "/tmp/pocket rows", .n = 0 },
    .{ .name = "TF_SLIDE_FORGET_FAIL", .raw = " 0 ", .n = 0 },
    .{ .name = "TF_CUBLAS_BRIDGE", .raw = "runtime/tf_cublaslt_bridge.dll", .n = 0 },
    .{ .name = "TF_CUBLAS_PREFILL", .raw = "1", .n = 1 },
    .{ .name = "TF_CUDA_WARPS8", .raw = "7", .n = 7 },
    .{ .name = "TF_DRAFT_BF16", .raw = "31", .n = 31 },
    .{ .name = "TF_DRAFT_BF16_EMBED_DIR", .raw = "models/draft-bf16", .n = 0 },
    .{ .name = "TF_DRAFT_BLOCK", .raw = "8", .n = 8 },
    .{ .name = "TF_DRAFT_BLOCK", .raw = "16", .n = 16 },
    .{ .name = "TF_DRAFT_COPY_MATCH", .raw = "2", .n = 2 },
    .{ .name = "TF_DRAFT_COPY_MATCH", .raw = "20", .n = 20 },
    .{ .name = "TF_DRAFT_DATASET_DIR", .raw = "output/dataset", .n = 0 },
    .{ .name = "TF_DRAFT_FEEDBACK_TRACE", .raw = "0", .n = 0 },
    .{ .name = "TF_DRAFT_FEEDBACK_WEIGHT", .raw = "0.2", .n = @bitCast(@as(f64, 0.2)) },
    .{ .name = "TF_DRAFT_GRAPH", .raw = "1", .n = 1 },
    .{ .name = "TF_DRAFT_LANES", .raw = "1", .n = 1 },
    .{ .name = "TF_DRAFT_LANES", .raw = "2", .n = 2 },
    .{ .name = "TF_DRAFT_LANES", .raw = "4", .n = 4 },
    .{ .name = "TF_DRAFT_LANE_BASE_WEIGHT", .raw = "0.5", .n = @bitCast(@as(f64, 0.5)) },
    .{ .name = "TF_DRAFT_LATTICE_TRACE", .raw = "0", .n = 0 },
    .{ .name = "TF_DRAFT_MTP_DIR", .raw = "models/mtp", .n = 0 },
    .{ .name = "TF_DRAFT_MTP_GRAFT", .raw = "0", .n = 0 },
    .{ .name = "TF_DRAFT_MTP_PREFIX", .raw = "0", .n = 0 },
    .{ .name = "TF_DRAFT_MTP_QUERIES", .raw = "1", .n = 1 },
    .{ .name = "TF_DRAFT_MTP_QUERIES", .raw = "63", .n = 63 },
    .{ .name = "TF_DRAFT_MTP_WEIGHT", .raw = "0.3", .n = @bitCast(@as(f64, 0.3)) },
    .{ .name = "TF_DRAFT_QWEN3_NORM", .raw = "1", .n = 1 },
    .{ .name = "TF_DRAFT_REFERENCE_DIR", .raw = "output/reference", .n = 0 },
    .{ .name = "TF_DRAFT_REFINE", .raw = "0", .n = 0 },
    .{ .name = "TF_DRAFT_WIDE_TRACE", .raw = "0", .n = 0 },
    .{ .name = "TF_QWEN_AFFINE_PREPARED", .raw = "0", .n = 0 },
    .{ .name = "TF_QWEN_ATTN_EXACT", .raw = "7", .n = 7 },
    .{ .name = "TF_QWEN_EXPAND_KV", .raw = "1", .n = 1 },
    .{ .name = "TF_QWEN_FAST_PREFILL", .raw = "0", .n = 0 },
    .{ .name = "TF_QWEN_FP8_ATTENTION", .raw = "2", .n = 2 },
    .{ .name = "TF_QWEN_FP8_TILE", .raw = "3", .n = 3 },
    .{ .name = "TF_QWEN_GDN_CHUNKED", .raw = "0", .n = 0 },
    .{ .name = "TF_QWEN_NVFP4_MLP_DIR", .raw = "models/alternate", .n = 0 },
    .{ .name = "TF_QWEN_NVFP4_PREFILL", .raw = "1", .n = 1 },
    .{ .name = "TF_QWEN_NVFP4_SCOPE", .raw = "mlp", .n = 0 },
    .{ .name = "TF_QWEN_NVFP4_SCOPE", .raw = "layers", .n = 1 },
    .{ .name = "TF_QWEN_NVFP4_TMA", .raw = "1", .n = 1 },
    .{ .name = "TF_QWEN_PREFILL_GRAPH", .raw = "0", .n = 0 },
    .{ .name = "TF_QWEN_PREFILL_PROFILE", .raw = "0", .n = 0 },
    .{ .name = "TF_QWEN_PREFILL_ROWS", .raw = "64", .n = 64 },
    .{ .name = "TF_QWEN_PREFILL_ROWS", .raw = "7680", .n = 7680 },
};

test "environment: valid forms remain raw numbers before existing consumer clamps and defaults" {
    for (valid) |c| {
        try t.expectEqual(@as(?u64, c.n), try e.parse(c.name, c.raw));
        const why = try e.refusal(a, Env{ .name = c.name, .raw = c.raw }, all);
        defer if (why) |v| a.free(v);
        try t.expect(why == null);
    }
    for (e.rules) |r| {
        for ([_][]const u8{ "", " ", "\t \n" }) |raw| try t.expectEqual(@as(?u64, null), try e.parse(r.name, raw));
        // the DeepSeek and GLM rows are each covered by deepseek_v41/env_validate_test.zig and glm/cuda_env_test.zig
        if (r.scope == .dsv41 or r.scope == .glm) continue;
        var covered = false;
        for (valid) |c| if (std.mem.eql(u8, r.name, c.name)) {
            covered = true;
        };
        try t.expect(covered);
    }
}

test "environment: range, enum, whitespace and native prompt geometry refusals are explicit" {
    for ([_][2][]const u8{
        .{ "TENSORFOLD_MAX_PDF_PAGES", "0" },   .{ "TENSORFOLD_PRESSURE_CACHE_PERCENT", "101" },
        .{ "TENSORFOLD_PREFIX_WAIT_MS", "+1" }, .{ "TENSORFOLD_RESTORE_PREFETCH", "+1" },
        .{ "TF_CUDA_SEGMENTS", "0" },           .{ "TF_CUDA_SEGMENTS", "5" },
        .{ "TF_CUDA_CARVEOUT", "true" },        .{ "TF_CUDA_CARVEOUT", "01" },
        .{ "TF_CUDA_CARVEOUT_MIB", "0" },       .{ "TF_CUDA_CARVEOUT_MIB", "18446744073709551615" },
        .{ "TF_ROCE_GID_INDEX", "2147483648" }, .{ "TF_PROMPT_ROWS", "16" },
        .{ "TF_PROMPT_ROWS", "129" },           .{ "TF_PROMPT_ROWS", "2064" },
        .{ "TF_PROMPT_ROWS", " 512" },          .{ "XPU_A8_BITS", "04" },
        .{ "XPU_A8_BITS", "16" },               .{ "TF_SLIDE_POCKET_KEY", "AUTO" },
        .{ "TF_DRAMA_PREFILL_ROWS", "4096x" },  .{ "TF_DRAMA_REFERENCE_CACHE", "4294967296" },
    }) |c| try t.expectError(error.InvalidEnvironment, e.parse(c[0], c[1]));
    try t.expectEqual(@as(?u64, null), try e.parse("TF_CUDA_DEVICE", "   "));
    try t.expectEqual(@as(?u64, 0), try e.parse("TF_Q27_COPY_ROWS", "+0"));
}

test "environment: a path setting keeps its text exactly, reads blank as unset and refuses control characters" {
    const name = "TENSORFOLD_TOOL_ID_KEY_FILE";
    try t.expectEqualStrings("relative/k ey", (try e.path(name, "relative/k ey")).?);
    try t.expectEqualStrings(" /lead space", (try e.path(name, " /lead space")).?);
    for ([_]?[]const u8{ null, "", " \t" }) |raw| try t.expectEqual(@as(?[]const u8, null), try e.path(name, raw));
    for ([_][]const u8{ "/k\n", "/k\x00", "\t/k" }) |raw| try t.expectError(error.InvalidEnvironment, e.path(name, raw));
    // a key pasted into the setting by mistake: the refusal names the setting and never shows its text
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const pasted = "0123456789abcdefPASTEDKEY0123456789abcdef\x1b";
    const said = try e.describe(arena.allocator(), name, pasted);
    try t.expect(std.mem.indexOf(u8, said, "PASTEDKEY") == null and std.mem.startsWith(u8, said, name ++ ": expected "));
    const refused = (try e.refusal(arena.allocator(), Env{ .name = name, .raw = pasted }, all)).?;
    try t.expect(std.mem.indexOf(u8, refused, "PASTEDKEY") == null);
}

test "environment: explicit flags and disabled features retain their old precedence" {
    try t.expectEqual(@as(u32, 7), try e.device(7, "bad"));
    try t.expectEqual(@as(u32, 3), try e.device(null, " 3 "));
    try t.expectEqual(@as(u32, 0), try e.device(null, null));
    try t.expectEqual(@as(u32, 0), try e.device(null, ""));
    var f = all;
    f.device = 3;
    try t.expect((try e.refusal(a, Env{ .name = "TF_CUDA_DEVICE", .raw = "bad" }, f)) == null);
    f.segments = 2;
    try t.expect((try e.refusal(a, Env{ .name = "TF_CUDA_SEGMENTS", .raw = "bad" }, f)) == null);
    try t.expect((try e.refusal(a, Env{ .name = "TF_CUDA_DEVICE", .raw = "bad" }, .{})) == null);
    try t.expect((try e.refusal(a, Env{ .name = "TF_SLIDE_FORGET_FAIL", .raw = "bad" }, .{})) == null);
}

const Inactive = struct {
    pub fn get(_: Inactive, name: []const u8) ?[]const u8 {
        if (std.mem.eql(u8, name, "TF_CUDA_CARVEOUT")) return "0";
        if (std.mem.eql(u8, name, "TF_CUDA_CARVEOUT_MIB")) return "bad-size";
        if (std.mem.eql(u8, name, "TF_NO_TREES")) return "";
        if (std.mem.eql(u8, name, "TF_DRAFT_NODES")) return "bad-nodes";
        return null;
    }
};

test "environment: inactive carveout size and disabled trees are still ignored" {
    try t.expect((try e.refusal(a, Inactive{}, all)) == null);
    var enabled = all;
    enabled.carveout = true;
    const why = (try e.refusal(a, Inactive{}, enabled)).?;
    defer a.free(why);
    try t.expect(std.mem.indexOf(u8, why, "TF_CUDA_CARVEOUT_MIB=\"bad-size\"") != null);
}

test "environment: video refusal describes the token budget without advertising a storage maximum" {
    const why = try e.describe(a, "TENSORFOLD_VIDEO_TOKENS", "bad-value");
    defer a.free(why);
    try t.expectEqualStrings(
        "TENSORFOLD_VIDEO_TOKENS=\"bad-value\": expected a nonnegative whole video-token budget (default 16384; unsigned 64-bit count)",
        why,
    );
    try t.expect(std.mem.indexOf(u8, why, "18446744073709551615") == null);
}

test "environment: word switches take 0 and 1 and the word forms in any case; exact switches refuse words" {
    for ([_][]const u8{ "0", "false", "OFF", "No" }) |raw| try t.expect(!try e.value(bool, "TF_DSV41_LOOKUP_PLAN", raw, true));
    // MHC_DEFER and COEF_LATE read "no" as on before, so it is refused there rather than read as off
    for ([_][]const u8{ "no", "NO", "No" }) |raw| {
        try t.expectError(error.InvalidEnvironment, e.parse("TF_DSV41_MHC_DEFER", raw));
        try t.expectError(error.InvalidEnvironment, e.parse("TF_DSV41_COEF_LATE", raw));
    }
    for ([_][]const u8{ "1", "TRUE", "on", "Yes" }) |raw| try t.expect(try e.value(bool, "TF_DSV41_COEF_LATE", raw, false));
    try t.expect(try e.value(bool, "TF_DSV41_GRAPHS", null, true));
    try t.expect(try e.value(bool, "TF_DSV41_GRAPHS", " ", true));
    for ([_][]const u8{ "2", "00", "truee", "0 trailing" }) |raw| try t.expectError(error.InvalidEnvironment, e.parse("TF_DSV41_COEF_LATE", raw));
    try t.expect(try e.value(bool, "TF_DSV41_MHC_DEFER", "\ton ", false));
    // main read these as on for any text but "0", so a word form must be refused, never read the other way
    for ([_][]const u8{ "true", "on", "yes", "false", "off", "no", " 1", "2" }) |raw| {
        try t.expectError(error.InvalidEnvironment, e.parse("TF_DSV41_GM_V3", raw));
        try t.expectError(error.InvalidEnvironment, e.parse("TF_DSV41_DRAFTS", raw));
        try t.expectError(error.InvalidEnvironment, e.parse("TF_DSV41_SAMPLING", raw));
    }
    try t.expectError(error.InvalidEnvironment, e.parse("TF_DSV41_KV_SPLIT", "true"));
}

test "environment: the keep-others switch is exact 0 or 1, checked when the CUDA backend starts" {
    const name = "TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS";
    for ([_][]const u8{ "true", "on", "yes", "01", " 1", "1 ", "2" }) |raw| {
        try t.expectError(error.InvalidEnvironment, e.parse(name, raw));
        const why = (try e.refusal(a, Env{ .name = name, .raw = raw }, .{ .cuda = true })).?;
        a.free(why);
        try t.expect((try e.refusal(a, Env{ .name = name, .raw = raw }, .{ .metal = true })) == null);
    }
    try t.expect(!try e.value(bool, name, null, false));
    try t.expect(try e.value(bool, name, "1", false));
}

test "environment: real settings are finite, keep their lower bound and optional cap, and read as floats" {
    try t.expectEqual(@as(f64, 3.25), try e.value(f64, "TF_DSV41_GRAPH_FLOOR_GIB", "3.25", 0));
    try t.expectEqual(@as(f64, 0), try e.value(f64, "TF_DSV41_GRAPH_FLOOR_GIB", "0", 1));
    try t.expectEqual(@as(f64, 4), try e.value(f64, "TF_DSV41_FLOOR_HARD_GIB", null, 4));
    try t.expectEqual(@as(f64, 4), try e.value(f64, "TF_DSV41_FLOOR_HARD_GIB", " 4", 1));
    try t.expectEqual(@as(f64, 6.5), try e.value(f64, "TF_DSV41_FLOOR_GIB", " 6.5 ", 5));
    try t.expectEqual(@as(f64, 1), try e.value(f64, "TF_DSV41_PREFILL_SHARE", "1", 0.5));
    try t.expectEqual(@as(f64, -2.5), try e.value(f64, "TF_DSV41_EXPERT_TOPP", "-2.5", 0));
    for ([_][2][]const u8{
        .{ "TF_DSV41_GRAPH_FLOOR_GIB", "-1" }, .{ "TF_DSV41_GRAPH_FLOOR_GIB", "1e999" },
        .{ "TF_DSV41_FLOOR_HARD_GIB", "0" },   .{ "TF_DSV41_FLOOR_HARD_GIB", "-1" },
        .{ "TF_DSV41_PREFILL_SHARE", "1.1" },  .{ "TF_DSV41_EXPERT_TOPP", "nan" },
        .{ "TF_DSV41_EXPERT_TOPP", "-inf" },   .{ "TF_DSV41_FLOOR_GIB", "4 GiB" },
    }) |c| try t.expectError(error.InvalidEnvironment, e.parse(c[0], c[1]));
}

test "environment: ranges, word lists and whole-number bounds refuse ignored tails and values outside them" {
    try t.expectEqual(@as(?[2]u32, .{ 2, 4 }), try e.range("TF_DSV41_LAYERS", "2-4"));
    try t.expectEqual(@as(?[2]u32, .{ 0, 0 }), try e.range("TF_DSV41_LAYERS", "0-0"));
    try t.expectEqual(@as(?[2]u32, null), try e.range("TF_DSV41_LAYERS", ""));
    for ([_][]const u8{ "2", "4-2", "1-2-3", "a-b", "-1", "0--1", "0-4294967296" }) |raw| try t.expectError(error.InvalidEnvironment, e.parse("TF_DSV41_LAYERS", raw));
    try t.expectEqual(@as(?u64, 1), try e.parse("TF_DSV41_IMAGES", " native "));
    try t.expectEqual(@as(?u64, 0), try e.parse("TF_DSV41_IMAGES", "0"));
    for ([_][]const u8{ "Native", "1", "nativee" }) |raw| try t.expectError(error.InvalidEnvironment, e.parse("TF_DSV41_IMAGES", raw));
    for ([_][2][]const u8{ .{ "TF_DSV41_ENGRAM_READERS", "32" }, .{ "TF_DSV41_SLOTS", "16" }, .{ "TF_DSV41_ROWS_CAP", "12" }, .{ "TF_DSV41_CONTEXT", "9223372036854775807" } }) |c|
        try t.expect(try e.parse(c[0], c[1]) != null);
    for ([_][2][]const u8{ .{ "TF_DSV41_ENGRAM_READERS", "33" }, .{ "TF_DSV41_SLOTS", "0" }, .{ "TF_DSV41_SLOTS", "17" }, .{ "TF_DSV41_ROWS_CAP", "9" }, .{ "TF_DSV41_CONTEXT", "9223372036854775808" }, .{ "TF_DSV41_PREFILL_ADAPT_MIN", "15" } }) |c|
        try t.expectError(error.InvalidEnvironment, e.parse(c[0], c[1]));
}

const Alias = struct {
    canonical: ?[]const u8,
    video: ?[]const u8 = null,
    pub fn get(v: Alias, name: []const u8) ?[]const u8 {
        if (std.mem.eql(u8, name, "TENSORFOLD_TOOL_GRAMMAR")) return v.canonical;
        if (std.mem.eql(u8, name, "TF_DSV41_TOOL_GRAMMAR")) return "bad";
        return if (std.mem.eql(u8, name, "TENSORFOLD_VIDEO_TOKENS")) v.video else null;
    }
};

test "environment: a legacy alias is checked only while its canonical name is unset, and a scope check stays in its scope" {
    try t.expectError(error.InvalidEnvironment, e.checkScope(.dsv41, Alias{ .canonical = null }));
    try e.checkScope(.dsv41, Alias{ .canonical = "0", .video = "bad" });
    const why = (try e.refusal(a, Alias{ .canonical = null }, .{ .dsv41 = true })).?;
    defer a.free(why);
    try t.expectEqualStrings("TF_DSV41_TOOL_GRAMMAR=\"bad\": expected 0 or 1 (ignored while TENSORFOLD_TOOL_GRAMMAR is set)", why);
    for ([_][]const u8{ "", "0", "1" }) |canonical| try t.expect((try e.refusal(a, Alias{ .canonical = canonical }, .{ .dsv41 = true })) == null);
    try t.expect((try e.refusal(a, Alias{ .canonical = null }, .{})) == null);
}

test "environment: Qwen selectors keep their exact domains and family checks retain geometry ownership" {
    for ([_][2][]const u8{
        .{ "TF_CUDA_WARPS8", "8" },            .{ "TF_DRAFT_BF16", "32" },
        .{ "TF_DRAFT_BLOCK", "4" },            .{ "TF_DRAFT_BLOCK", "32" },
        .{ "TF_DRAFT_COPY_MATCH", "1" },       .{ "TF_DRAFT_COPY_MATCH", "21" },
        .{ "TF_DRAFT_LANES", "0" },            .{ "TF_DRAFT_LANES", "3" },
        .{ "TF_DRAFT_MTP_QUERIES", "0" },      .{ "TF_DRAFT_MTP_QUERIES", "64" },
        .{ "TF_QWEN_ATTN_EXACT", "8" },        .{ "TF_QWEN_ATTN_EXACT", "01" },
        .{ "TF_QWEN_FP8_ATTENTION", "3" },     .{ "TF_QWEN_FP8_ATTENTION", "single" },
        .{ "TF_QWEN_FP8_TILE", "4" },          .{ "TF_QWEN_FP8_TILE", "01" },
        .{ "TF_QWEN_NVFP4_SCOPE", "all" },     .{ "TF_QWEN_NVFP4_SCOPE", "MLP" },
        .{ "TF_QWEN_PREFILL_ROWS", "0" },      .{ "TF_QWEN_PREFILL_ROWS", "63" },
        .{ "TF_QWEN_PREFILL_ROWS", "65" },     .{ "TF_DRAFT_FEEDBACK_TRACE", "yes" },
        .{ "TF_DRAFT_LATTICE_TRACE", "true" }, .{ "TF_QWEN_PREFILL_PROFILE", "01" },
    }) |c| try t.expectError(error.InvalidEnvironment, e.parse(c[0], c[1]));
    for ([_][:0]const u8{ "TF_DRAFT_FEEDBACK_WEIGHT", "TF_DRAFT_LANE_BASE_WEIGHT", "TF_DRAFT_MTP_WEIGHT" }) |name| {
        for ([_][]const u8{ "0", "1" }) |raw| try t.expect(try e.parse(name, raw) != null);
        for ([_][]const u8{ "-0.1", "1.01", "nan", "inf", "-inf" }) |raw|
            try t.expectError(error.InvalidEnvironment, e.parse(name, raw));
    }
}

test "environment: Qwen scope validation is explicit and disabled for unrelated startup flags" {
    const env: Env = .{ .name = "TF_CUBLAS_PREFILL", .raw = "invalid" };
    try t.expect((try e.refusal(a, env, .{})) == null);
    try t.expect((try e.refusal(a, env, .{ .cuda = true, .metal = true })) == null);
    try e.checkScope(.cuda, env);
    try t.expectError(error.InvalidEnvironment, e.checkScope(.qwen, env));
    const why = (try e.refusal(a, env, .{ .qwen = true })).?;
    defer a.free(why);
    try t.expectEqualStrings("TF_CUBLAS_PREFILL=\"invalid\": expected 0 or 1", why);
}
