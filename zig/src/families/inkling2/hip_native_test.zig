//! Inkling 2 Mini's HIP refusals through the hook the HIP host calls before any load (native/hip_families.zig).
const std = @import("std");
const api = @import("engine_api");
const fam = @import("hip_families");
const F = @import("hip_native.zig");
const t = std.testing;

/// What `o` gets from the HIP host's refusal hook for this family, or "" when it passes.
fn why(o: api.Open) []const u8 {
    return fam.refused(F, o) orelse "";
}

const mini: api.Open = .{ .dir = "d", .model_type = F.model_type };

fn with(comptime field: []const u8, value: anytype) api.Open {
    var o = mini;
    @field(o, field) = value;
    return o;
}

test "inkling2 hip refusals: the family reads Inkling 2 Mini's model_type" {
    try t.expectEqualStrings("inkling2_loop", F.model_type);
    try t.expect(fam.reads(F, "inkling2_loop") and !fam.reads(F, "inkling2"));
}

test "inkling2 hip refusals: --tp above one rank" {
    try t.expectEqualStrings("inkling2_loop: this HIP engine serves one rank; serve with --tp 1", why(with("tp", 2)));
}

test "inkling2 hip refusals: classifier-free guidance" {
    try t.expectEqualStrings("inkling2_loop: classifier-free guidance is unsupported; serve without --cfg-gamma, --cfg-allow-request-gamma and --cfg-batched", why(with("cfg_enabled", true)));
}

test "inkling2 hip refusals: batched classifier-free guidance" {
    try t.expectEqualStrings("inkling2_loop: classifier-free guidance is unsupported; serve without --cfg-gamma, --cfg-allow-request-gamma and --cfg-batched", why(with("cfg_batched", true)));
}

test "inkling2 hip refusals: --loops" {
    try t.expectEqualStrings("inkling2_loop: the model has exactly three core passes; --loops cannot override them", why(with("loops", @as(u32, 3))));
}

test "inkling2 hip refusals: --mtp-drafts, zero included" {
    const msg = "inkling2_loop: external draft controls are unsupported; serve without --mtp-drafts and --mtp-confidence";
    try t.expectEqualStrings(msg, why(with("mtp_drafts", @as(u32, 2))));
    try t.expectEqualStrings(msg, why(with("mtp_drafts", @as(u32, 0))));
}

test "inkling2 hip refusals: --mtp-confidence" {
    try t.expectEqualStrings("inkling2_loop: external draft controls are unsupported; serve without --mtp-drafts and --mtp-confidence", why(with("mtp_confidence", @as(f64, 0.5))));
}

test "inkling2 hip refusals: --precision, full and checkpoint" {
    const msg = "inkling2_loop: the checkpoint's NVFP4, MXFP8 and BF16 arithmetic is fixed; serve without --precision";
    try t.expectEqualStrings(msg, why(with("precision", api.Precision.full)));
    try t.expectEqualStrings(msg, why(with("precision", api.Precision.checkpoint)));
}

test "inkling2 hip refusals: --prompt-precision int8" {
    try t.expectEqualStrings("inkling2_loop: the prompt matmuls keep the checkpoint's arithmetic; serve without --prompt-precision int8", why(with("prompt_int8", true)));
}

test "inkling2 hip refusals: --learn" {
    try t.expectEqualStrings("inkling2_loop: learned disk states are not implemented; serve without --learn and --learn-conversations", why(with("learn", @as([]const u8, "/learned"))));
}

test "inkling2 hip refusals: --learn-conversations" {
    try t.expectEqualStrings("inkling2_loop: learned disk states are not implemented; serve without --learn and --learn-conversations", why(with("learn_conversations", true)));
}

test "inkling2 hip refusals: a non-finite or huge --prompt-cache-gib; a negative one is the host's (checkGroup)" {
    const msg = "inkling2_loop: --prompt-cache-gib must be a finite budget of at most 1048576 GiB; serve with a smaller one";
    for ([_]f64{ std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64), 1024 * 1024 + 1 }) |g| try t.expectEqualStrings(msg, why(with("prompt_cache_gib", g)));
    for ([_]f64{ 0, 1024 * 1024, -1 }) |g| try t.expectEqualStrings("", why(with("prompt_cache_gib", g)));
}

test "inkling2 hip refusals: one rank's text settings pass" {
    var o = mini;
    o.context = 65536;
    o.lanes = 4;
    o.lanes_fixed = true;
    o.drafts = false;
    o.prompt_cache_gib = 8;
    o.prompt_cache_over_cap = true;
    o.keep = 16;
    o.decode_share = 0.5;
    o.device = 0;
    o.recall = true;
    o.execution_witness = true;
    o.vision = .{}; // the host serves text only and logs it (native/hip.zig openWith)
    try t.expectEqualStrings("", why(o));
    try t.expectEqualStrings("", why(mini));
}

test "inkling2 hip refusals: the shared HIP refusals still follow the family's" {
    try t.expectEqualStrings("--rope-yarn: the native HIP engine has no YaRN rope; serve without --rope-yarn, or on CUDA", why(with("rope_yarn", @as(f64, 4.0))));
}
