//! Runtime settings keep source forms, dynamic-reader limits and family-only startup validation.
const std = @import("std");
const e = @import("environment.zig");
const t = std.testing;

const Fixture = struct {
    name: []const u8,
    raw: []const u8,
    pub fn get(f: Fixture, name: []const u8) ?[]const u8 {
        return if (std.mem.eql(u8, name, f.name)) f.raw else null;
    }
};

test "DeepSeek runtime: a malformed graph count is refused before the loader needs resources" {
    const Bad = struct {
        pub fn get(_: @This(), name: []const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_DSV41_GRAPHS_MAX")) "mistyped" else null;
        }
    };
    const why = try e.refusal(std.testing.allocator, Bad{}, .{ .dsv41 = true });
    defer if (why) |v| std.testing.allocator.free(v);
    try std.testing.expect(why != null);
    try std.testing.expect(std.mem.startsWith(u8, why.?, "TF_DSV41_GRAPHS_MAX="));
    try std.testing.expectError(error.InvalidEnvironment, e.checkScope(.dsv41, Bad{}));
}

test "DeepSeek runtime: every new declaration has a valid value and a named startup refusal" {
    const cases = [_][2][]const u8{
        .{ "TF_DSV41_AOT_TRACE", "relative/path with space" },
        .{ "TF_DSV41_GRAMMAR_TOKENIZER", "relative/path with space" },
        .{ "TF_DSV41_ACCEPT_LOG", "relative/path with space" },
        .{ "TF_DSV41_PHASES_OUT", "relative/path with space" },
        .{ "TF_DSV41_PREPARED", "relative/path with space" },
        .{ "TF_DSV41_PROFILE", "relative/path with space" },
        .{ "TF_DSV41_TRACE_TOKENS", "relative/path with space" },
        .{ "TF_DSV41_CALIB_DIR", "relative/path with space" },
        .{ "TF_DSV41_CALIB_FILE", "relative/path with space" },
        .{ "TF_DSV41_SESSION_DISK", "relative/path with space" },
        .{ "TF_DSV41_PF_DENSE_TABLE", "relative/path with space" },
        .{ "TF_TP_KERNEL_IMAGE", "relative/path with space" },
        .{ "TF_DSV41_BRANCHES", "0" },
        .{ "TF_DSV41_BRANCHES_ROWS", "1" },
        .{ "TF_DSV41_BRANCHES_PRIO", "main" },
        .{ "TF_DSV41_DRAFT_GRAPHS", "0" },
        .{ "TF_DSV41_DRAFT_GRAPHS_MAX", "1" },
        .{ "TF_DSV41_GRAPH_FLOOR", "shed" },
        .{ "TF_DSV41_DRAFT_CHAIN", "1" },
        .{ "TF_DSV41_DRAFT_OVERLAP", "1" },
        .{ "TF_DSV41_DRAFT_CAPTURE", "1" },
        .{ "TF_DSV41_SAMP_BATCH", "1" },
        .{ "TF_DSV41_DRAFT_BATCH_PROJ", "0" },
        .{ "TF_DSV41_SLOT_DRAFT_GROUP", "1" },
        .{ "TF_DSV41_ENGRAM_GATE", "1" },
        .{ "TF_DSV41_ENGRAM_GATE_ROWS", "1" },
        .{ "TF_DSV41_ENGRAM_GATE_TIMEOUT_S", "1" },
        .{ "TF_DSV41_ENGRAM_WAIT_S", "1" },
        .{ "TF_DSV41_GRAPHS_MAX", "1" },
        .{ "TF_DSV41_GRAPH_BUCKET", "1" },
        .{ "TF_DSV41_GRAPH_BUCKET_GROW", "1" },
        .{ "TF_DSV41_GRAPH_ROWS_MAX", "1" },
        .{ "TF_DSV41_LIN2", "0" },
        .{ "TF_DSV41_LIN2_PF", "1" },
        .{ "TF_DSV41_LIN2_CLUSTER", "1" },
        .{ "TF_DSV41_LIN2_WK", "4" },
        .{ "TF_DSV41_L2PF", "0" },
        .{ "TF_DSV41_L2PF_ROWS", "1" },
        .{ "TF_DSV41_L2PF_MB", "1" },
        .{ "TF_DSV41_L2PF_CHUNK_KB", "1" },
        .{ "TF_DSV41_L2PF_PACE_GBPS", "1" },
        .{ "TF_DSV41_L2PF_PACE_CTAS", "1" },
        .{ "TF_DSV41_L2PF_DELAY_US", "1" },
        .{ "TF_DSV41_L2PF_SITES", "x,o,f" },
        .{ "TF_DSV41_L2PF_PACE_SITES", "x,f" },
        .{ "TF_DSV41_L2PF_AT", "before" },
        .{ "TF_DSV41_PROMPT_PROF", "1" },
        .{ "TF_DSV41_PHASES", "0" },
        .{ "TF_DSV41_PREPARED_WRITE", "1" },
        .{ "TF_DSV41_PREPARED_VERIFY", "1" },
        .{ "TF_DSV41_ROUND_GRAPH", "split" },
        .{ "TF_DSV41_ROUND_GRAPH_HEAD", "1" },
        .{ "TF_DSV41_SAMP_STATS", "split" },
        .{ "TF_DSV41_PIECES_FLOOR", "1" },
        .{ "TF_DSV41_SESSION_DISK_MIN", "1" },
        .{ "TF_DSV41_SESSION_RESIDENT", "1" },
        .{ "TF_DSV41_SESSION_RAM_MIB", "1" },
        .{ "TF_DSV41_SESSION_STAGING_MIB", "1" },
        .{ "TF_DSV41_SESSION_DISK_GIB", "1" },
        .{ "TF_DSV41_SESSION_RESIDENT_GIB", "1" },
        .{ "TF_DSV41_ENGRAM_WARM", "1" },
        .{ "TF_DSV41_STALL_EXIT_S", "1" },
        .{ "TF_DSV41_GM_GU2", "1" },
        .{ "TF_DSV41_PF_4K", "1" },
        .{ "TF_DSV41_PF_4K_MEMORY_OK", "1" },
        .{ "TF_DSV41_PF_OVERLAP_SITE", "1" },
        .{ "TF_DSV41_PF_TBO", "1" },
        .{ "TF_DSV41_INDEX_BOUND", "1" },
        .{ "TF_DSV41_X3LD_EPI", "1" },
        .{ "TF_DSV41_PF_DENSE", "off" },
        .{ "TF_DSV41_GM_V2", "0" },
        .{ "TF_DSV41_PF_DENSE_CFG", "7,1024" },
        .{ "TF_DSV41_PF_OVERLAP", "1" },
        .{ "TF_DSV41_MHC_SITE_ROWS", "256" },
        .{ "TF_DSV41_PF_TBO_GM_SMS", "8" },
        .{ "TF_DSV41_STREAM_RB", "0" },
        .{ "TF_DSV41_MHC_PFDEC", "1" },
        .{ "TF_DSV41_PREFILL", "full" },
        .{ "TF_DSV41_PROMPT_TAIL", "verify" },
        .{ "TF_DSV41_MHC_DEFER_AT", "0" },
        .{ "TF_DSV41_MHC_DEFER_AT_ROWS", "1" },
        .{ "TF_DSV41_ATTN_SPLIT", "1" },
        .{ "TF_DSV41_ATTN_ROPE", "1" },
        .{ "TF_DSV41_MHC_TAIL", "1" },
        .{ "TF_DSV41_MOE_BF16", "1" },
        .{ "TF_DSV41_PRUNE_KIT", "1" },
        .{ "TF_DSV41_RG_PRUNE", "1" },
        .{ "TF_DSV41_RMS_FOLD", "1" },
        .{ "TF_DSV41_CALIB", "cached" },
        .{ "TF_DSV41_CALIB_SHAPE", "raw" },
        .{ "TF_DSV41_CALIB_STAT", "median" },
        .{ "TF_DSV41_CALIB_REPS", "1" },
        .{ "TF_DSV41_CALIB_WARM", "1" },
        .{ "TF_DSV41_CALIB_RECHECK", "1" },
        .{ "TF_DSV41_CALIB_CYCLE", "1" },
        .{ "TF_DSV41_CALIB_DEEP_REPS", "1" },
        .{ "TF_DSV41_CODE_ACCEPT", "1" },
        .{ "TF_DSV41_DRAFT_SKIP", "1" },
        .{ "TF_DSV41_SPEC_DRAFT", "1" },
        .{ "TF_DSV41_SPEC_NUCLEUS", "1" },
        .{ "TF_DSV41_DEPTH", "cost" },
        .{ "TF_DSV41_DRAFT_DEPTH", "1" },
        .{ "TF_DSV41_DEPTH_JOINT", "0" },
        .{ "TF_DSV41_DEPTH_FAIR", "1" },
        .{ "TF_DSV41_DEPTH_HOST_MS", "1" },
        .{ "TF_DSV41_DRAFT_PLAIN_STREAMS", "1" },
        .{ "TF_DSV41_LOOKUP_MIN", "1" },
        .{ "TF_DSV41_MAX_ROWS", "2" },
        .{ "TF_DSV41_TREE", "0" },
        .{ "TF_DSV41_TREE_DUP_MS", "1" },
        .{ "TF_DSV41_TREE_ROWS", "1" },
        .{ "TF_DSV41_TREE_PC", "0" },
        .{ "TF_DSV41_POST_BOOT_GIB", "1" },
        .{ "TF_DSV41_BOOT_GROWTH_GIB", "1" },
        .{ "TF_DSV41_SERVE_DIP_GIB", "1" },
        .{ "TF_DSV41_BOOT_MEASURED", "auto" },
        .{ "TF_DSV41_KV_SPLIT_COMPACT", "0" },
        .{ "TF_DSV41_PREFILL_CHUNK", "32" },
    };
    for (cases) |c| {
        try t.expectEqual(e.Scope.dsv41, e.rule(c[0]).scope);
        try t.expect(try e.parse(c[0], c[1]) != null);
        const fx: Fixture = .{ .name = c[0], .raw = if (e.rule(c[0]).kind == .path) "bad\npath" else "invalid" };
        try t.expect(try e.refusal(t.allocator, fx, .{ .cuda = true, .metal = true, .xpu = true, .slide = true, .hooks = true, .glm = true, .flashnext = true }) == null);
        if (e.rule(c[0]).word_fallback != null) {
            try t.expectEqual(@as(?u64, 0), try e.parse(c[0], fx.raw));
            continue;
        }
        try t.expectError(error.InvalidEnvironment, e.checkScope(.dsv41, fx));
        const why = (try e.refusal(t.allocator, fx, .{ .dsv41 = true })).?;
        defer t.allocator.free(why);
        try t.expect(std.mem.startsWith(u8, why, c[0]));
    }
}

test "DeepSeek runtime: legacy aliases, sets, case and pair grammar retain accepted forms" {
    for ([_][3][]const u8{
        .{ "BRANCHES", "StReAmS", "1" },        .{ "BRANCHES", "e4", "0" },
        .{ "LIN2", "LaNeS", "2" },              .{ "LIN2", "LINEAR", "3" },
        .{ "DRAFT_GRAPHS", "device", "1" },     .{ "DRAFT_BATCH_PROJ", "check", "2" },
        .{ "ROUND_GRAPH", "split, gate", "3" }, .{ "ROUND_GRAPH", "gate,gate", "2" },
        .{ "L2PF_SITES", "X, o, f", "7" },      .{ "L2PF_SITES", "x,x,x", "1" },
        .{ "L2PF_AT", ",BEFORE,", "1" },        .{ "KV_SPLIT_COMPACT", "YES", "1" },
        .{ "KV_SPLIT_COMPACT", "FORCE", "2" },
    }) |c| {
        var buf: [64]u8 = undefined;
        const name = try std.fmt.bufPrintSentinel(&buf, "TF_DSV41_{s}", .{c[0]}, 0);
        try t.expectEqual(try std.fmt.parseInt(u64, c[2], 10), try e.value(u64, name, c[1], 99));
    }
    try t.expectEqual(@as(?[2]u32, .{ 0, 8 }), try e.pair("TF_DSV41_PF_DENSE_CFG", ",0,"));
    try t.expectEqual(@as(?[2]u32, .{ 7, 1024 }), try e.pair("TF_DSV41_PF_DENSE_CFG", " 7, 1024 "));
    for ([_][]const u8{ "8", "0,0", "0,1025", "0,1,2", "bad" }) |raw|
        try t.expectError(error.InvalidEnvironment, e.pair("TF_DSV41_PF_DENSE_CFG", raw));
    for ([_][]const u8{ "CODE_ACCEPT", "DRAFT_SKIP", "SPEC_DRAFT", "SPEC_NUCLEUS" }) |suffix| {
        var buf: [64]u8 = undefined;
        const name = try std.fmt.bufPrintSentinel(&buf, "TF_DSV41_{s}", .{suffix}, 0);
        try t.expect(try e.value(bool, name, "TRUE", false));
        try t.expect(!try e.value(bool, name, "Off", true));
        for ([_][]const u8{ "yes", "no", "bad" }) |v| try t.expectError(error.InvalidEnvironment, e.parse(name, v));
    }
}

test "DeepSeek runtime: bounds and tool controls never leak into common startup" {
    for ([_][2][]const u8{
        .{ "TF_DSV41_MHC_SITE_ROWS", "257" },     .{ "TF_DSV41_PF_TBO_GM_SMS", "7" },
        .{ "TF_DSV41_GRAPH_BUCKET_GROW", "0.5" }, .{ "TF_DSV41_GRAPH_BUCKET_GROW", "17" },
        .{ "TF_DSV41_MAX_ROWS", "17" },           .{ "TF_DSV41_TREE", "1" },
        .{ "TF_DSV41_DRAFT_DEPTH", "16" },        .{ "TF_DSV41_STALL_EXIT_S", "18446744074" },
    }) |c| try t.expectError(error.InvalidEnvironment, e.parse(c[0], c[1]));
    for ([_][]const u8{ "COUNT", "FN_LAYERS", "TF_TP_WORLD", "TF_TP_RANK", "HOME", "TF_EXPERIMENTAL", "TF_DSV41_DECODE_TP_LANES", "TF_DSV41_KV_CHECK_TOPP", "TF_DSV41_SESSION_SLOTS_TURNS" }) |name|
        for (e.rules) |r| try t.expect(!std.mem.eql(u8, name, r.name));
    try t.expectEqual(@as(?u64, 0), try e.parse("TF_DSV41_MHC_SITE_ROWS", "0"));
    try t.expectEqual(@as(?u64, 0), try e.parse("TF_DSV41_PF_TBO_GM_SMS", "0"));
    try t.expectEqual(@as(?u64, 0), try e.parse("TF_DSV41_L2PF_ROWS", "0"));
    try t.expectEqual(@as(?u64, 18446744073709551615), try e.parse("TF_DSV41_GRAPH_ROWS_MAX", "18446744073709551615"));
}
