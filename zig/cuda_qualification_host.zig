//! Model-free serving matrix, through the actual pre-driver native admission rather than a policy mock.
const std = @import("std");
const gate = @import("qualified_refusal");
const t = std.testing;
const Pair = struct { name: [:0]const u8, value: ?[]const u8 };
const Env = struct {
    entries: []const Pair = &.{},
    changed: ?Pair = null,
    pub fn get(e: Env, name: [:0]const u8) ?[]const u8 {
        if (e.changed) |v| if (std.mem.eql(u8, v.name, name)) return v.value;
        for (e.entries) |v| if (std.mem.eql(u8, v.name, name)) return v.value;
        return null;
    }
};
const active = [_]Pair{
    .{ .name = "TF_CUBLAS_PREFILL", .value = "1" },
    .{ .name = "TF_CUBLAS_BRIDGE", .value = "./runtime/bridge.dll" },
    .{ .name = "TF_QWEN_NVFP4_PREFILL", .value = "1" },
    .{ .name = "TF_QWEN_NVFP4_TMA", .value = "1" },
    .{ .name = "TF_QWEN_NVFP4_MLP_DIR", .value = "./alternate-checkpoint" },
    .{ .name = "TF_QWEN_NVFP4_SCOPE", .value = "layers" },
    .{ .name = "TF_CUDA_WARPS8", .value = "2" },
    .{ .name = "TF_QWEN_EXPAND_KV", .value = "1" },
    .{ .name = "TF_QWEN_PREFILL_ROWS", .value = "7680" },
    .{ .name = "TF_DRAFT_BLOCK", .value = "8" },
    .{ .name = "TF_DRAFT_GRAPH", .value = "1" },
    .{ .name = "TF_DRAFT_QWEN3_NORM", .value = "1" },
    .{ .name = "TF_DRAFT_FEEDBACK_WEIGHT", .value = "0.2" },
    .{ .name = "TF_QWEN_KV_PAGES", .value = "512" },
    .{ .name = "TF_QWEN_KV_PAGES_ALL", .value = "1" },
    .{ .name = "TF_QWEN_KV_SHARE", .value = "64" },
};
fn check(a: std.mem.Allocator, model: []const u8, e: Env, needle: ?[]const u8) !void {
    var why: []const u8 = "";
    const refused = try gate.startupWith(a, .{ .dir = "not-opened", .model_type = model }, &why, e);
    if (needle) |word| {
        try t.expect(refused);
        try t.expect(std.mem.indexOf(u8, why, word) != null);
    } else try t.expect(!refused);
}

test "CUDA qualification: main defaults and upstream settings retain admission" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const entries = [_]Pair{
        .{ .name = "TF_QWEN_PREFILL_ROWS", .value = "4096" },
        .{ .name = "TF_CUDA_TEST_MLP128", .value = "1" },
        .{ .name = "TF_DRAFT_NODES", .value = "32" },
        .{ .name = "TF_QWEN_KV_PAGES", .value = "512" },
        .{ .name = "TF_QWEN_KV_PAGES_ALL", .value = "1" },
    };
    try check(arena.allocator(), "qwen3_5", .{ .entries = &entries }, null);
}

test "CUDA qualification: recorded fast tuple is admitted only by a wide enough build" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    try check(arena.allocator(), "qwen3_5", .{ .entries = &active }, if (@import("qwen_prefill_config").max_rows >= 7680) null else "7680");
    try check(arena.allocator(), "qwen3_5_moe", .{ .entries = &.{.{ .name = "TF_QWEN_PREFILL_ROWS", .value = "4096" }} }, null);
}

test "CUDA qualification: new experimental choices and incomplete target tuples refuse by name" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const rejected = [_]Pair{
        .{ .name = "TF_QWEN_FAST_PREFILL", .value = "1" },
        .{ .name = "TF_QWEN_FP8_TILE", .value = "1" },
        .{ .name = "TF_QWEN_FP8_TILE", .value = "3" },
        .{ .name = "TF_QWEN_AFFINE_PREPARED", .value = "1" },
        .{ .name = "TF_QWEN_FP8_ATTENTION", .value = "1" },
        .{ .name = "TF_QWEN_FP8_ATTENTION", .value = "2" },
        .{ .name = "TF_QWEN_GDN_CHUNKED", .value = "1" },
        .{ .name = "TF_QWEN_ATTN_EXACT", .value = "7" },
        .{ .name = "TF_QWEN_PREFILL_GRAPH", .value = "1" },
        .{ .name = "TF_DRAFT_BF16", .value = "1" },
        .{ .name = "TF_DRAFT_BF16", .value = "31" },
        .{ .name = "TF_DRAFT_BF16_EMBED_DIR", .value = "./proposal-checkpoint" },
        .{ .name = "TF_DRAFT_BLOCK", .value = "16" },
        .{ .name = "TF_DRAFT_COPY_MATCH", .value = "2" },
        .{ .name = "TF_DRAFT_LANES", .value = "2" },
        .{ .name = "TF_DRAFT_LANES", .value = "4" },
        .{ .name = "TF_DRAFT_REFINE", .value = "1" },
        .{ .name = "TF_DRAFT_LANE_BASE_WEIGHT", .value = "0.6" },
        .{ .name = "TF_DRAFT_MTP_DIR", .value = "./proposal-checkpoint" },
        .{ .name = "TF_DRAFT_MTP_GRAFT", .value = "1" },
        .{ .name = "TF_DRAFT_MTP_PREFIX", .value = "1" },
        .{ .name = "TF_DRAFT_MTP_QUERIES", .value = "8" },
        .{ .name = "TF_DRAFT_MTP_WEIGHT", .value = "0.4" },
        .{ .name = "TF_DRAFT_FEEDBACK_WEIGHT", .value = "0.1" },
        .{ .name = "TF_CUDA_WARPS8", .value = "1" },
        .{ .name = "TF_CUDA_WARPS8", .value = "7" },
        .{ .name = "TF_CUBLAS_PREFILL", .value = "0" },
        .{ .name = "TF_QWEN_NVFP4_PREFILL", .value = "0" },
        .{ .name = "TF_QWEN_NVFP4_TMA", .value = "0" },
        .{ .name = "TF_QWEN_NVFP4_MLP_DIR", .value = null },
        .{ .name = "TF_QWEN_NVFP4_SCOPE", .value = "mlp" },
        .{ .name = "TF_QWEN_EXPAND_KV", .value = "0" },
        .{ .name = "TF_QWEN_PREFILL_ROWS", .value = "4096" },
        .{ .name = "TF_QWEN_PREFILL_ROWS", .value = "8192" },
        .{ .name = "TF_CUBLAS_BRIDGE", .value = null },
        .{ .name = "TF_DRAFT_BLOCK", .value = null },
        .{ .name = "TF_DRAFT_GRAPH", .value = "0" },
        .{ .name = "TF_DRAFT_QWEN3_NORM", .value = "0" },
        .{ .name = "TF_DRAFT_FEEDBACK_WEIGHT", .value = "0" },
    };
    for (rejected) |v| try check(arena.allocator(), "qwen3_5", .{ .entries = &active, .changed = v }, v.name);
}

test "CUDA qualification: unrelated families keep upstream modes and ignore unused Qwen knobs" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const entries = [_]Pair{
        .{ .name = "TF_QWEN_FAST_PREFILL", .value = "1" },
        .{ .name = "TF_DRAFT_BF16", .value = "31" },
        .{ .name = "TF_CUDA_TEST_MLP128", .value = "1" },
        .{ .name = "TF_DRAFT_NODES", .value = "32" },
    };
    try check(arena.allocator(), "nemotron_h", .{ .entries = &entries }, null);
    try check(arena.allocator(), "nemotron_h", .{ .entries = &entries, .changed = .{ .name = "TF_CUDA_WARPS8", .value = "2" } }, "TF_CUDA_WARPS8");
}

test "CUDA qualification: syntax remains strict before profile admission" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    try check(arena.allocator(), "qwen3_5", .{ .entries = &active, .changed = .{ .name = "TF_QWEN_FAST_PREFILL", .value = "typo" } }, "0 or 1");
}

test "CUDA qualification: disabled prefill profiling refuses by name before checkpoint or device work" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const upstream = [_]Pair{.{ .name = "TF_QWEN_PREFILL_ROWS", .value = "4096" }};
    for ([_][]const u8{ "qwen3_5", "qwen3_5_moe" }) |model| {
        try check(arena.allocator(), model, .{ .entries = &upstream, .changed = .{ .name = "TF_QWEN_PREFILL_PROFILE", .value = "1" } }, "TF_QWEN_PREFILL_PROFILE");
        for ([_]?[]const u8{ null, "", " \t", "0" }) |value|
            try check(arena.allocator(), model, .{ .entries = &upstream, .changed = .{ .name = "TF_QWEN_PREFILL_PROFILE", .value = value } }, null);
        try check(arena.allocator(), model, .{ .entries = &upstream, .changed = .{ .name = "TF_QWEN_PREFILL_PROFILE", .value = "typo" } }, "0 or 1");
    }
    try check(arena.allocator(), "nemotron_h", .{ .changed = .{ .name = "TF_QWEN_PREFILL_PROFILE", .value = "1" } }, null);
}

test "CUDA qualification: the recorded dense profile is not an unqualified MoE opt-in" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    try check(arena.allocator(), "qwen3_5_moe", .{ .entries = &active }, "dense");
}

test "CUDA qualification: numeric spellings and blank optional values keep their parsed meaning" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    if (@import("qwen_prefill_config").max_rows >= 7680) {
        for ([_]Pair{
            .{ .name = "TF_CUDA_WARPS8", .value = "02" },
            .{ .name = "TF_DRAFT_FEEDBACK_WEIGHT", .value = "2e-1" },
            .{ .name = "TF_DRAFT_BLOCK", .value = "08" },
            .{ .name = "TF_DRAFT_BF16_EMBED_DIR", .value = " \t" },
        }) |v| try check(arena.allocator(), "qwen3_5", .{ .entries = &active, .changed = v }, null);
    }
    try check(arena.allocator(), "qwen3_5", .{ .entries = &.{.{ .name = "TF_QWEN_PREFILL_ROWS", .value = "4096" }}, .changed = .{ .name = "TF_DRAFT_COPY_MATCH", .value = " \t" } }, null);
}

test "CUDA qualification: a wide build cannot silently select its unrecorded serving width" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    try check(arena.allocator(), "qwen3_5", .{}, if (@import("qwen_prefill_config").max_rows == 4096) null else "TF_QWEN_PREFILL_ROWS");
}
