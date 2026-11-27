//! GLM cold-prefill ledger setting and stage vocabulary, without a device.
const std = @import("std");
const env = @import("sys").environment;
pub const name = "TF_GLM_PREFILL_TRACE";
pub const Stage = enum {
    control_reduce,
    token_gather,
    cache_write,
    prompt_score,
    drafter,
    replicated_qakva,
    projections,
    index_update,
    index_score,
    index_select,
    mla,
    kda,
    ffn,
    router,
    experts,
    shared_expert,
    output_projection,
    glue,
    partials,
    glued,
    all_gather,
    wait_filled,
    wait_gathered,
    wait_glues,
    wait_shared,
    embed,
    head,
    gate,
    draw,
    mtp,
    commit,
    norms,
    dense_gemm,
    request,
    chunk,
};
pub const Site = enum { chunk, mixer, ffn };
pub const Meta = struct { rank: usize, slot: u32, start: usize, rows: usize };

pub fn path(raw: ?[]const u8) !?[]const u8 {
    return env.path(name, raw);
}

test "GLM prefill ledger: strict optional path is registered" {
    const found = for (env.rules) |rule| {
        if (std.mem.eql(u8, rule.name, name)) break rule;
    } else null;
    try std.testing.expect(found != null);
    if (found) |rule| {
        try std.testing.expectEqual(.path, rule.kind);
        try std.testing.expectEqual(.glm, rule.scope);
    }
}

test "GLM prefill ledger: stage vocabulary includes standalone norms and dense GEMMs" {
    try std.testing.expect(std.meta.stringToEnum(Stage, "norms") != null);
    try std.testing.expect(std.meta.stringToEnum(Stage, "dense_gemm") != null);
}

pub const Kind = enum { dsa, kda, mla, outside_layer };
pub const Ffn = enum { dense, moe, outside_layer };
pub const Layer = struct { kind: Kind, ffn: Ffn };
pub const Class = enum { indexer, attention, projections, kda, hc_and_fused_norms, norms, routing, routed_experts, shared_expert, dense_gemm, collective, exchange_wait, final_logits, aggregate, other };
pub fn class(stage: Stage) Class {
    return switch (stage) {
        .index_update, .index_score, .index_select => .indexer,
        .mla => .attention,
        .replicated_qakva, .projections, .output_projection => .projections,
        .kda => .kda,
        .glue => .hc_and_fused_norms,
        .norms => .norms,
        .router => .routing,
        .experts => .routed_experts,
        .shared_expert => .shared_expert,
        .dense_gemm => .dense_gemm,
        .control_reduce, .token_gather, .partials, .glued, .all_gather => .collective,
        .wait_filled, .wait_gathered, .wait_glues, .wait_shared => .exchange_wait,
        .head => .final_logits,
        .request, .chunk, .ffn => .aggregate,
        .cache_write, .prompt_score, .drafter, .embed, .gate, .draw, .mtp, .commit => .other,
    };
}

test "GLM prefill ledger: blank disables, paths preserve bytes and controls refuse" {
    try std.testing.expectEqual(@as(?[]const u8, null), try path(null));
    try std.testing.expectEqual(@as(?[]const u8, null), try path(" \t\n"));
    try std.testing.expectEqualStrings("results/cold run", (try path("results/cold run")).?);
    inline for (.{ "a\nb", "a\tb", "a\x00b", "a\x7fb" }) |raw| try std.testing.expectError(error.InvalidEnvironment, path(raw));
    inline for (std.meta.tags(Stage)) |stage| _ = class(stage);
    try std.testing.expectEqual(Class.collective, class(.partials));
    try std.testing.expectEqual(Class.hc_and_fused_norms, class(.glue));
}
