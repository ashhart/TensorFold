//! Startup refusals carry names, values and accepted forms; valid settings preserve defaults and clamps.
const std = @import("std");
const e = @import("environment.zig");
const t = std.testing;
const a = t.allocator;
const all: e.Flags = .{ .cuda = true, .metal = true, .xpu = true, .slide = true, .hooks = true, .dsv41 = true, .gemma4 = true, .drama3 = true, .glm = true, .flashnext = true, .nemotron = true, .minimax_m3 = true, .chat = true, .hub = true };
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
        const bad: []const []const u8 = if (r.kind == .path) &.{ "key\nfile", "key\x1b[2Kfile", "\x7f" } else if (r.kind == .names) &.{ "a,b,c", ",", "1\nwrong" } else if (e.isReal(r.kind)) &.{ "typo", "nan", "-inf", "1\nwrong" } else &.{ "typo", "-1", "18446744073709551616", "1\nwrong" };
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

const valid = @import("environment_valid_cases.zig").valid;

test "environment: valid forms remain raw numbers before existing consumer clamps and defaults" {
    for (valid) |c| {
        try t.expectEqual(@as(?u64, c.n), try e.parse(c.name, c.raw));
        const why = try e.refusal(a, Env{ .name = c.name, .raw = c.raw }, all);
        defer if (why) |v| a.free(v);
        try t.expect(why == null);
    }
    for (e.rules) |r| {
        for ([_][]const u8{ "", " ", "\t \n" }) |raw| {
            if (r.blank_default) try t.expectEqual(@as(?u64, null), try e.parse(r.name, raw)) else try t.expectError(error.InvalidEnvironment, e.parse(r.name, raw));
        }
        // own scope tests: deepseek_v41/env_validate_test.zig, glm/cuda_env_test.zig, environment_flashnext_test.zig
        if (r.scope == .dsv41 or r.scope == .glm or r.scope == .flashnext) continue;
        var covered = false;
        for (valid) |c| if (std.mem.eql(u8, r.name, c.name)) {
            covered = true;
        };
        try t.expect(covered);
    }
}

test "environment: range, enum, whitespace and native prompt geometry refusals are explicit" {
    for ([_][2][]const u8{
        .{ "TENSORFOLD_MAX_PDF_PAGES", "0" },          .{ "TENSORFOLD_PRESSURE_CACHE_PERCENT", "101" },
        .{ "TENSORFOLD_PREFIX_WAIT_MS", "+1" },        .{ "TENSORFOLD_RESTORE_PREFETCH", "+1" },
        .{ "TF_CUDA_SEGMENTS", "0" },                  .{ "TF_CUDA_SEGMENTS", "5" },
        .{ "TF_CUDA_CARVEOUT", "true" },               .{ "TF_CUDA_CARVEOUT", "01" },
        .{ "TF_CUDA_CARVEOUT_MIB", "0" },              .{ "TF_CUDA_CARVEOUT_MIB", "18446744073709551615" },
        .{ "TF_ROCE_GID_INDEX", "2147483648" },        .{ "TF_PROMPT_ROWS", "16" },
        .{ "TF_PROMPT_ROWS", "129" },                  .{ "TF_PROMPT_ROWS", "2064" },
        .{ "TF_PROMPT_ROWS", " 512" },                 .{ "XPU_A8_BITS", "04" },
        .{ "XPU_A8_BITS", "16" },                      .{ "XPU_TP", "0" },
        .{ "XPU_TP", "3" },                            .{ "XPU_TP", " 4" },
        .{ "XPU_CTX", "0" },                           .{ "XPU_CTX", "262145" },
        .{ "XPU_CHUNK", "0" },                         .{ "XPU_CHUNK", "2049" },
        .{ "XPU_CHUNK_SOLO", "0" },                    .{ "XPU_CHUNK_SOLO", "2049" },
        .{ "XPU_MTP_DEPTH", "0" },                     .{ "XPU_MTP_DEPTH", "16" },
        .{ "XPU_DEVICE_ID", "0" },                     .{ "XPU_DEVICE_ID", "0xzz" },
        .{ "TF_SLIDE_POCKET_KEY", "AUTO" },            .{ "TF_DRAMA_PREFILL_ROWS", "4096x" },
        .{ "TF_DRAMA_REFERENCE_CACHE", "4294967296" }, .{ "TF_DSPINE_WINDOW", "0" },
        .{ "TF_DSPINE_WINDOW", "15" },                 .{ "TF_DSPINE_WINDOW", "8193" },
        .{ "TF_QWEN_KV_PAGE_TEST", "2147483648" },     .{ "TF_QWEN_KV_PAGE_TEST", "0" },
        .{ "TF_Q27_GEMM", "wide" },                    .{ "TF_Q27_GEMM", "NARROW" },
        .{ "TF_Q27_TAIL", "on" },                      .{ "TF_Q27_TAIL", "OFF" },
        .{ "TF_Q27_POCKET_OFF", "no\ncontrol" },       .{ "TF_QWEN_KEEP_LATE", "2" },
        .{ "TF_TIMING", "2" },                         .{ "TF_NO_TREES", "true" },
        .{ "TF_GEMMA4_ARGMAX_GRAPH", "2" },            .{ "TF_GEMMA4_ARGMAX_GRAPH", "on" },
        .{ "TF_Q35_GROUP_PROJ", "2" },                 .{ "TF_Q35_GROUP_PROJ", "on" },
        .{ "TF_QMMF_HALF_BN", "2" },                   .{ "TF_QMMF_HALF_BN", "on" },
        .{ "TF_MINIMAX_GRAPHS", "2" },                 .{ "TF_MINIMAX_GRAPHS", "on" },
        .{ "TF_MTP_TREES", "yes" },                    .{ "TF_MTP_BATCH", "off" },
        .{ "TF_QWEN_POOL_LOG", "true" },               .{ "TF_COST_KEY_TRACE", "on" },
        .{ "TF_COST_KEY_TRACE", "on" },                .{ "TF_ROCE_HCA", "rocep1s0f0,roceP2p1s0f0,rocep1s0f1" },
        .{ "GLM_POOL_TOKENS", "0" },                   .{ "GLM_POOL_TOKENS", "1023" },
        .{ "GLM_POOL_TOKENS", "4294967232" },          .{ "GLM_POOL_TOKENS", "2k" },
        .{ "GLM_POOL_TOKENS", "+2000000" },            .{ "GLM_POOL_TOKENS", "-1" },
        .{ "GLM_POOL_TOKENS", "1e6" },                 .{ "GLM_POOL_TOKENS", "lots" },
        .{ "TF_MTP_BITS", "8" },
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
        try t.expectError(error.InvalidEnvironment, e.parse("TF_DSV41_COPY_DRAFTER", raw));
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

test "environment: LFM2's and LFM3's page switches are exact 0 or 1, refused by name, and unset keeps pages on" {
    inline for (.{ "TENSORFOLD_LFM2_KV_PAGES", "TENSORFOLD_LFM3_KV_PAGES" }) |name| {
        for ([_][]const u8{ "off", "false", "no", "on", "2", " 0" }) |raw| {
            try t.expectError(error.InvalidEnvironment, e.value(bool, name, raw, true));
            const why = (try e.refusal(a, Env{ .name = name, .raw = raw }, .{ .cuda = true })).?;
            defer a.free(why);
            try t.expect(std.mem.indexOf(u8, why, name) != null);
        }
        try t.expect(try e.value(bool, name, null, true));
        try t.expect(try e.value(bool, name, "1", true));
        try t.expect(!try e.value(bool, name, "0", true));
    }
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

test "environment: the Metal Qwen and slide-learning switches take their forms, read blank as unset and name a refusal" {
    const flags = [_][:0]const u8{ "GLM_ROUND_PROF", "TF_Q27_MTP_GPU_PICK", "TF_Q27_METAL_PROMPT_NO_RESCALE", "TF_QWEN27_MESSAGE_STARTS", "TF_QWEN27_SLOT_RESUME", "TF_QWEN27_SHARED_PREFIX", "TF_QWEN27_METAL_KV_PAGES", "TF_QWEN27_JOIN", "TF_QWEN27_TIED_HEAD", "TF_QWEN35_METAL_KV_PAGES", "TF_INKLING2_DRAFT_HEAD_Q4", "GLM_DRAFT_HEAD_Q4", "TF_SLIDE_TAKE_OUT_CHECK", "TF_SLIDE_TAP_CHECK" };
    for (flags) |name| {
        try t.expectEqual(@as(?u64, 0), try e.parse(name, "0"));
        try t.expectEqual(@as(?u64, 1), try e.parse(name, "1"));
        for ([_][]const u8{ "", " ", "\t" }) |raw| try t.expectEqual(@as(?u64, null), try e.parse(name, raw));
        for ([_][]const u8{ "00", "false", "2", "yes" }) |raw| {
            try t.expectError(error.InvalidEnvironment, e.parse(name, raw));
            const why = (try e.refusal(a, Env{ .name = name, .raw = raw }, all)).?;
            defer a.free(why);
            try t.expect(std.mem.indexOf(u8, why, name) != null);
        }
    }
    // unset and every valid value keep today's meaning: resume on unless 0, pages off unless 1
    try t.expect(try e.value(bool, "TF_QWEN27_SLOT_RESUME", null, true));
    try t.expect(try e.value(bool, "TF_QWEN27_SLOT_RESUME", "", true));
    try t.expect(!(try e.value(bool, "TF_QWEN27_SLOT_RESUME", "0", true)));
    try t.expect(!(try e.value(bool, "TF_QWEN27_METAL_KV_PAGES", null, false)));
    try t.expect(try e.value(bool, "TF_QWEN27_JOIN", null, true) and !(try e.value(bool, "TF_QWEN27_JOIN", "0", true)));
    try t.expect(!(try e.value(bool, "TF_QWEN27_TIED_HEAD", null, false)) and try e.value(bool, "TF_QWEN27_TIED_HEAD", "1", false));
    try t.expect(try e.value(bool, "TF_QWEN35_METAL_KV_PAGES", "1", false));
    try t.expect(!(try e.value(bool, "TF_Q27_MTP_GPU_PICK", null, false)) and try e.value(bool, "TF_Q27_MTP_GPU_PICK", "1", false));
    try t.expect(!(try e.value(bool, "TF_Q27_METAL_PROMPT_NO_RESCALE", null, false)) and !(try e.value(bool, "TF_Q27_METAL_PROMPT_NO_RESCALE", " ", false)) and !(try e.value(bool, "TF_Q27_METAL_PROMPT_NO_RESCALE", "0", false)) and try e.value(bool, "TF_Q27_METAL_PROMPT_NO_RESCALE", "1", false));
    for ([_][]const u8{ "true", "01", "2", " 1" }) |raw| try t.expectError(error.InvalidEnvironment, e.parse("TF_Q27_METAL_PROMPT_NO_RESCALE", raw));
    try t.expect(!(try e.value(bool, "TF_INKLING2_DRAFT_HEAD_Q4", null, false)) and !(try e.value(bool, "TF_INKLING2_DRAFT_HEAD_Q4", " ", false)) and !(try e.value(bool, "TF_INKLING2_DRAFT_HEAD_Q4", "0", false)) and try e.value(bool, "TF_INKLING2_DRAFT_HEAD_Q4", "1", false));
}

test "rank aliases name their flag and refuse from their retirement release" {
    const Row = struct { name: [:0]const u8, scope: e.Scope, flag: []const u8 };
    const rows = [_]Row{
        .{ .name = "TF_GLM_RANK", .scope = .glm, .flag = "--rank" },            .{ .name = "TF_GLM_WORLD", .scope = .glm, .flag = "--tp" },
        .{ .name = "TF_MINIMAX_RANK", .scope = .minimax_m3, .flag = "--rank" }, .{ .name = "TF_MINIMAX_WORLD", .scope = .minimax_m3, .flag = "--tp" },
        .{ .name = "TF_NEMOTRON_RANK", .scope = .nemotron, .flag = "--rank" },  .{ .name = "TF_NEMOTRON_WORLD", .scope = .nemotron, .flag = "--tp" },
    };
    for (rows) |row| {
        const name = row.name;
        try t.expectEqual(row.scope, e.rule(name).scope);
        try t.expectEqualStrings(row.flag, e.rule(name).flag.?);
        try t.expectEqualStrings(e.alias_retire, e.rule(name).retire.?);
        var buf: [64]u8 = undefined;
        try t.expect(e.retired(name, "0.0.1") == null);
        try t.expect(e.retired(name, try std.fmt.bufPrint(&buf, "{s}-rc.1", .{e.alias_retire})) == null); // a pre-release is before it
        try t.expect(e.retired(name, e.alias_retire) != null);
        try t.expect(e.retired(name, "999.0.0") != null);
        try t.expectEqual(@as(?u64, 2), try e.parse(name, "+2"));
        try t.expectError(error.InvalidEnvironment, e.parse(name, " 2"));
        try t.expectError(error.InvalidEnvironment, e.parse(name, "4294967296"));
    }
    try t.expect(e.retired("TF_GLM_PROF", "9.9.9") == null);
    try t.expect(e.retired("TF_NOT_A_ROW", "9.9.9") == null);
}

test "environment: TENSORFOLD_PARALLEL takes the flag over the setting, blank keeps the default, garbage is refused" {
    try t.expectEqualStrings("4", (try e.parallel("4", "8")).?);
    try t.expectEqualStrings("auto", (try e.parallel("auto", "8")).?);
    try t.expectEqualStrings("8", (try e.parallel(null, " 8 ")).?);
    try t.expectEqualStrings("Auto", (try e.parallel(null, "Auto")).?); // the text passes through; cli.parallel matches it without case
    try t.expect((try e.parallel(null, null)) == null);
    try t.expect((try e.parallel(null, " \t")) == null);
    for ([_][]const u8{ "0", "4097", "typo", "-1", "8\nwrong" }) |raw| {
        try t.expectError(error.InvalidEnvironment, e.parallel(null, raw));
        const why = (try e.refusal(a, Env{ .name = "TENSORFOLD_PARALLEL", .raw = raw }, all)).?;
        defer a.free(why);
        try t.expect(std.mem.indexOf(u8, why, "TENSORFOLD_PARALLEL") != null);
        try t.expect(std.mem.indexOf(u8, why, "or auto") != null);
    }
}

test "environment: a malformed calibration setting stops a Metal start naming the setting, its text and the accepted forms" {
    const metal: e.Flags = .{ .metal = true };
    const bad = [_]struct { name: []const u8, raw: []const u8 }{
        .{ .name = "TF_DSV41_CALIB_BUDGET_MS", .raw = "abc" },
        .{ .name = "TF_DSV41_CALIB_BUDGET_MS", .raw = "600001" },
        .{ .name = "TF_DSV41_CALIB_REPS", .raw = "0" },
        .{ .name = "TF_DSV41_CALIB_REPS", .raw = "51" },
        .{ .name = "TF_DSV41_CALIB", .raw = "stored" },
        .{ .name = "TF_DSV41_CALIB_STAT", .raw = "mean" },
    };
    for (bad) |c| {
        const why = (try e.refusal(a, Env{ .name = c.name, .raw = c.raw }, metal)).?;
        defer a.free(why);
        try t.expect(std.mem.indexOf(u8, why, c.name) != null);
        try t.expect(std.mem.indexOf(u8, why, c.raw) != null);
        try t.expect(std.mem.indexOf(u8, why, e.rule(c.name).forms) != null);
        // a CUDA start does not check these rows: the engine's own reader refuses them when it measures
        try t.expect((try e.refusal(a, Env{ .name = c.name, .raw = c.raw }, .{ .cuda = true })) == null);
    }
}
