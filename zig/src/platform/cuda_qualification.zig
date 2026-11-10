//! Serving admits upstream arithmetic or the recorded dense NVFP4 profile; operator contracts stay independent.
const std = @import("std");
const environment = @import("environment.zig");
pub const Failure = struct { err: anyerror, message: []const u8 };
const Limit = union(enum) {
    whole: struct { values: []const u64, default: u64 },
    real: struct { values: []const f64, default: f64 },
    present: bool,
};
const Rule = struct { name: [:0]const u8, limit: Limit, failure: Failure };

fn rule(comptime name: [:0]const u8, comptime limit: Limit, comptime err: anyerror, comptime forms: []const u8) Rule {
    return .{ .name = name, .limit = limit, .failure = .{ .err = err, .message = name ++ ": " ++ forms } };
}
fn number(comptime name: [:0]const u8, comptime values: []const u64, comptime default: u64, comptime err: anyerror, comptime forms: []const u8) Rule {
    return rule(name, .{ .whole = .{ .values = values, .default = default } }, err, forms);
}
fn real(comptime name: [:0]const u8, comptime values: []const f64, comptime default: f64, comptime err: anyerror, comptime forms: []const u8) Rule {
    return rule(name, .{ .real = .{ .values = values, .default = default } }, err, forms);
}
fn whole(env: anytype, name: [:0]const u8, default: u64) environment.Error!u64 {
    return environment.value(u64, name, env.get(name), default);
}
fn floating(env: anytype, name: [:0]const u8, default: f64) environment.Error!f64 {
    return environment.value(f64, name, env.get(name), default);
}
fn present(env: anytype, name: [:0]const u8) environment.Error!bool {
    return (try environment.parse(name, env.get(name))) != null;
}
fn accepts(r: Rule, env: anytype) environment.Error!bool {
    return switch (r.limit) {
        .whole => |n| std.mem.indexOfScalar(u64, n.values, try whole(env, r.name, n.default)) != null,
        .real => |n| std.mem.indexOfScalar(f64, n.values, try floating(env, r.name, n.default)) != null,
        .present => |yes| (try present(env, r.name)) == yes,
    };
}

const limited = [_]Rule{
    number("TF_QWEN_FAST_PREFILL", &.{0}, 0, error.UnqualifiedQwenFastPrefill, "FP8 activation prefill is not admitted for serving; use 0 or unset"),
    number("TF_QWEN_FP8_TILE", &.{0}, 0, error.UnqualifiedQwenFp8Tile, "experimental FP8 tiles are not admitted for serving; use 0 or unset"),
    number("TF_QWEN_AFFINE_PREPARED", &.{0}, 0, error.UnqualifiedQwenAffinePrepared, "prepared FP8 affine activations are not admitted for serving; use 0 or unset"),
    number("TF_QWEN_FP8_ATTENTION", &.{0}, 0, error.UnqualifiedQwenFp8Attention, "experimental FP8 attention is not admitted for serving; use 0 or unset"),
    number("TF_QWEN_GDN_CHUNKED", &.{0}, 0, error.UnqualifiedQwenGdnChunked, "the alternate GDN reduction is not admitted for serving; use 0 or unset"),
    number("TF_QWEN_ATTN_EXACT", &.{0}, 0, error.UnqualifiedQwenAttentionMode, "alternate attention algorithms are not admitted for serving; use 0 or unset"),
    number("TF_QWEN_PREFILL_GRAPH", &.{0}, 0, error.UnqualifiedQwenPrefillGraph, "target prefill graphs are not admitted for serving; use 0 or unset"),
    number("TF_DRAFT_BF16", &.{0}, 0, error.UnqualifiedQwenDraftBf16, "proposal precision overrides are not admitted for serving; use 0 or unset"),
    rule("TF_DRAFT_BF16_EMBED_DIR", .{ .present = false }, error.UnqualifiedQwenDraftEmbedding, "alternate proposal embeddings are not admitted for serving; leave unset"),
    number("TF_DRAFT_BLOCK", &.{8}, 8, error.UnqualifiedQwenDraftBlock, "leave unset for upstream behavior or use 8 in the recorded NVFP4 profile"),
    rule("TF_DRAFT_COPY_MATCH", .{ .present = false }, error.UnqualifiedQwenCopyMatch, "copy-match overrides are not admitted for serving; leave unset"),
    number("TF_DRAFT_LANES", &.{1}, 1, error.UnqualifiedQwenDraftLanes, "cooperative proposal lanes are not admitted for serving; use 1 or unset"),
    number("TF_DRAFT_REFINE", &.{0}, 0, error.UnqualifiedQwenDraftRefine, "proposal refinement is not admitted for serving; use 0 or unset"),
    real("TF_DRAFT_LANE_BASE_WEIGHT", &.{0.5}, 0.5, error.UnqualifiedQwenDraftLaneWeight, "alternate proposal lane weights are not admitted for serving; use 0.5 or unset"),
    rule("TF_DRAFT_MTP_DIR", .{ .present = false }, error.UnqualifiedQwenMtpEnsemble, "the additional MTP ensemble is not admitted for serving; leave unset"),
    number("TF_DRAFT_MTP_GRAFT", &.{0}, 0, error.UnqualifiedQwenMtpGraft, "MTP grafting is not admitted for serving; use 0 or unset"),
    number("TF_DRAFT_MTP_PREFIX", &.{0}, 0, error.UnqualifiedQwenMtpPrefix, "MTP prefix proposals are not admitted for serving; use 0 or unset"),
    rule("TF_DRAFT_MTP_QUERIES", .{ .present = false }, error.UnqualifiedQwenMtpQueries, "MTP query overrides are not admitted for serving; leave unset"),
    real("TF_DRAFT_MTP_WEIGHT", &.{0.3}, 0.3, error.UnqualifiedQwenMtpWeight, "alternate MTP proposal weights are not admitted for serving; use 0.3 or unset"),
    real("TF_DRAFT_FEEDBACK_WEIGHT", &.{ 0, 0.2 }, 0, error.UnqualifiedQwenFeedbackWeight, "use 0 or unset for upstream behavior, or 0.2 in the recorded NVFP4 profile"),
    number("TF_CUDA_WARPS8", &.{ 0, 2 }, 0, error.UnqualifiedCudaWarpMode, "use 0 or unset for upstream behavior, or 2 in the recorded dense NVFP4 profile"),
};
const required = [_]Rule{
    rule("TF_QWEN_NVFP4_MLP_DIR", .{ .present = true }, error.IncompleteQwenNvfp4Source, "the recorded NVFP4 profile requires an alternate checkpoint with NVFP4/BF16 layers"),
    rule("TF_CUBLAS_BRIDGE", .{ .present = true }, error.IncompleteQwenCublasBridge, "the recorded NVFP4 profile requires the cuBLAS bridge library"),
    number("TF_CUBLAS_PREFILL", &.{1}, 0, error.IncompleteQwenCublasPrefill, "the recorded NVFP4 profile requires 1; partial precision profiles are not admitted"),
    number("TF_QWEN_NVFP4_PREFILL", &.{1}, 0, error.IncompleteQwenNativePrefill, "the recorded NVFP4 profile requires 1; partial precision profiles are not admitted"),
    number("TF_QWEN_NVFP4_TMA", &.{1}, 0, error.IncompleteQwenTma, "the recorded NVFP4 profile requires 1; other native NVFP4 paths are not admitted"),
    number("TF_QWEN_NVFP4_SCOPE", &.{1}, 0, error.IncompleteQwenNvfp4Scope, "the recorded NVFP4 profile requires layers; the MLP-only alternate source is not admitted"),
    number("TF_CUDA_WARPS8", &.{2}, 0, error.IncompleteQwenWarps, "the recorded NVFP4 profile requires 2"),
    number("TF_QWEN_EXPAND_KV", &.{1}, 0, error.IncompleteQwenExpandedKv, "the recorded NVFP4 profile requires 1"),
    number("TF_QWEN_PREFILL_ROWS", &.{7680}, 0, error.UnqualifiedQwenPrefillWidth, "the recorded NVFP4 profile requires 7680 and a build supporting that width"),
    number("TF_DRAFT_BLOCK", &.{8}, 0, error.IncompleteQwenDraftBlock, "the recorded NVFP4 profile requires an explicit 8"),
    number("TF_DRAFT_GRAPH", &.{1}, 0, error.IncompleteQwenDraftGraph, "the recorded NVFP4 profile requires 1"),
    number("TF_DRAFT_QWEN3_NORM", &.{1}, 0, error.IncompleteQwenDraftNorm, "the recorded NVFP4 profile requires 1"),
    real("TF_DRAFT_FEEDBACK_WEIGHT", &.{0.2}, 0, error.IncompleteQwenFeedback, "the recorded NVFP4 profile requires 0.2"),
};
const dense_only: Failure = .{ .err = error.UnqualifiedQwenFastTarget, .message = "The recorded NVFP4 profile requires the dense affine4/group64 Qwen target; MoE keeps its upstream modes" };
const upstream_warps: Failure = .{ .err = error.UnqualifiedCudaWarpMode, .message = "TF_CUDA_WARPS8: other CUDA families retain 0 or unset; 2 belongs to the recorded dense NVFP4 profile" };
const upstream_width: Failure = .{ .err = error.UnqualifiedQwenPrefillWidth, .message = "TF_QWEN_PREFILL_ROWS: upstream serving uses 4096; the recorded NVFP4 profile uses 7680 with a wide enough build" };

pub fn isQwen(model_type: []const u8) bool {
    return std.mem.eql(u8, model_type, "qwen3_5") or std.mem.eql(u8, model_type, "qwen3_5_moe");
}

/// No device or file access occurs here; the existing source loader validates the actual checkpoint formats.
pub fn refusal(model_type: []const u8, compiled_rows: usize, env: anytype) environment.Error!?Failure {
    const warp = try whole(env, "TF_CUDA_WARPS8", 0);
    if (!isQwen(model_type)) return if (warp == 0) null else upstream_warps;
    inline for (limited) |r| if (!try accepts(r, env)) return r.failure;
    const rows = try whole(env, "TF_QWEN_PREFILL_ROWS", 0);
    const selected = try present(env, "TF_QWEN_NVFP4_MLP_DIR") or
        try whole(env, "TF_CUBLAS_PREFILL", 0) != 0 or try whole(env, "TF_QWEN_NVFP4_PREFILL", 0) != 0 or
        try whole(env, "TF_QWEN_NVFP4_TMA", 0) != 0 or try whole(env, "TF_QWEN_NVFP4_SCOPE", 0) != 0 or
        warp != 0 or try whole(env, "TF_QWEN_EXPAND_KV", 0) != 0 or
        try present(env, "TF_DRAFT_BLOCK") or try whole(env, "TF_DRAFT_GRAPH", 0) != 0 or
        try whole(env, "TF_DRAFT_QWEN3_NORM", 0) != 0 or try floating(env, "TF_DRAFT_FEEDBACK_WEIGHT", 0) != 0 or
        (rows != 0 and rows != 4096);
    if (!selected) return if ((if (rows == 0) compiled_rows else rows) == 4096) null else upstream_width;
    if (!std.mem.eql(u8, model_type, "qwen3_5")) return dense_only;
    inline for (required) |r| if (!try accepts(r, env)) return r.failure;
    if (compiled_rows < 7680) return upstream_width;
    return null;
}

pub fn check(model_type: []const u8, compiled_rows: usize, env: anytype) anyerror!void {
    if (try refusal(model_type, compiled_rows, env)) |why| return why.err;
}

pub fn explain(err: anyerror) ?[]const u8 {
    inline for (limited ++ required) |r| if (r.failure.err == err) return r.failure.message;
    for ([_]Failure{ dense_only, upstream_warps, upstream_width }) |why| if (why.err == err) return why.message;
    return null;
}
