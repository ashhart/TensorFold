//! Startup refusals carry names, values and accepted forms; valid settings preserve defaults and clamps.
const std = @import("std");
const e = @import("environment.zig");
const extensions = @import("environment_extensions.zig");
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

const Case = struct { name: []const u8, raw: []const u8, n: u64 };
const valid = [_]Case{
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

test "environment extensions: every production rule is present exactly once" {
    try t.expectEqual(e.rules.len + extensions.rules.len, e.registry.len);
    for (e.registry, 0..) |r, i| for (e.registry[0..i]) |prior|
        try t.expect(!std.mem.eql(u8, r.name, prior.name));
    for (extensions.rules) |r| try t.expectEqualStrings(r.name, e.rule(r.name).name);
}

test "environment extensions: every strict reader refuses bad text before opening a model, naming value and accepted forms" {
    for (extensions.rules) |r| {
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

test "environment extensions: valid forms remain raw numbers before existing consumer clamps and defaults" {
    for (valid) |c| {
        try t.expectEqual(@as(?u64, c.n), try e.parse(c.name, c.raw));
        const why = try e.refusal(a, Env{ .name = c.name, .raw = c.raw }, all);
        defer if (why) |v| a.free(v);
        try t.expect(why == null);
    }
    for (extensions.rules) |r| {
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
