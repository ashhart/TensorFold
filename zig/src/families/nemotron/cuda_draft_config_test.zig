//! The drafter config's topk geometry (PR 5102 owner A3): the lilicorr correlator's fixed
//! band is exactly 1..8 — nine or more candidates misread the fixed head, and a zero keeps
//! nothing — enforced at config load, before any allocation, under the named error whose
//! explain words live in cuda_native.zig's explain() (the DrafterBusy table). The DFlash2
//! selector's own band is untouched: its recipe checkpoint declares selector_top_k 16 and
//! cuda_draft_config.zig's bolt test pins it. The A5 cells pin the frame's row-0 identity
//! rule: a config that DECLARES parallel_drafting_token_id wins, a SILENT one keeps the
//! pending token's identity (dcfg.row0Token), and declared-0 is a declaration, not silence.
const std = @import("std");
const dcfg = @import("cuda_draft_config.zig");

// The bolt recipe's declaration shape with the selector rank/top-k swapped for the correlator's:
// widths, taps, window and the sliding declarations stay exactly the recipe's, so the only
// geometry variable across these cells is lilicorr_candidate_topk.
const lilicorr_band_config =
    \\{
    \\ "architectures": ["DFlash2DraftModel"],
    \\ "attention_bias": false,
    \\ "block_size": 8,
    \\ "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
    \\  "mask_token_id": 990, "lilicorr_enabled": true, "lilicorr_candidate_topk": 4,
    \\  "target_layer_ids": [1, 18, 35, 51, 68, 85]},
    \\ "head_dim": 128,
    \\ "hidden_act": "silu",
    \\ "hidden_size": 4096,
    \\ "intermediate_size": 12288,
    \\ "model_type": "qwen3",
    \\ "num_attention_heads": 32,
    \\ "num_hidden_layers": 5,
    \\ "num_key_value_heads": 2,
    \\ "rms_norm_eps": 1e-06,
    \\ "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    \\ "sliding_window": 1024,
    \\ "use_sliding_window": true,
    \\ "vocab_size": 131072
    \\}
;

const band_target = dcfg.Target{ .hidden = 4096, .vocab = 131072, .layers = 86 };

fn withTopk(a: std.mem.Allocator, topk: []const u8) ![]u8 {
    const repl = try std.fmt.allocPrint(a, "\"lilicorr_candidate_topk\": {s}", .{topk});
    defer a.free(repl);
    return std.mem.replaceOwned(u8, a, lilicorr_band_config, "\"lilicorr_candidate_topk\": 4", repl);
}

test "a candidate_topk inside the correlator's 1..8 band loads at config (owner A3)" {
    const a = std.testing.allocator;
    const c = try dcfg.Config.parse(a, lilicorr_band_config, band_target);
    try std.testing.expectEqual(dcfg.Kind.lilicorr, c.kind);
    try std.testing.expectEqual(@as(usize, 4), c.topk);
    try std.testing.expectEqual(@as(usize, 0), c.rank); // the correlator drops the selector's rank
}

test "a candidate_topk above eight refuses at load with the named geometry error (owner A3)" {
    const a = std.testing.allocator;
    for ([_][]const u8{ "9", "17", "256" }) |over| {
        const cfg = try withTopk(a, over);
        defer a.free(cfg);
        try std.testing.expectError(error.DraftTopkGeometry, dcfg.Config.parse(a, cfg, band_target));
    }
}

test "a candidate_topk of zero keeps nothing and refuses under the same named rule (owner A3)" {
    const a = std.testing.allocator;
    const zero = try withTopk(a, "0");
    defer a.free(zero);
    try std.testing.expectError(error.DraftTopkGeometry, dcfg.Config.parse(a, zero, band_target));
    const neg = try withTopk(a, "-2");
    defer a.free(neg);
    try std.testing.expectError(error.DraftTopkGeometry, dcfg.Config.parse(a, neg, band_target));
}

// -- owner A5, part 2: the row-0 identity is config-backed (declared wins, silent = pending) --

fn withParallelDecl(a: std.mem.Allocator, value: []const u8) ![]u8 {
    // Raw multiline literals (escape-free needling of the recipe shape): the declared cell
    // splices "parallel_drafting_token_id": <value> INTO the dflash_config object (the parse
    // reads the key there, beside mask_token_id — the checkpoint family's own shape).
    const needle =
        \\ "dflash_config": {"block_size": 8,
    ;
    const full = try std.fmt.allocPrint(a, "{s} \"parallel_drafting_token_id\": {s},", .{ needle, value });
    defer a.free(full);
    return std.mem.replaceOwned(u8, a, lilicorr_band_config, needle, full);
}

test "a config declaring parallel_drafting_token_id marks it declared and carries its value (owner A5)" {
    const a = std.testing.allocator;
    const with = try withParallelDecl(a, "990");
    defer a.free(with);
    const c = try dcfg.Config.parse(a, with, band_target);
    try std.testing.expect(c.parallel_declared);
    try std.testing.expectEqual(@as(u32, 990), c.parallel);
    try std.testing.expectEqual(@as(u32, 990), dcfg.row0Token(&c, 7)); // declared: config wins
}

test "a silent config keeps the pending identity (owner A5, the served LiLiCorr shape)" {
    const a = std.testing.allocator;
    const c = try dcfg.Config.parse(a, lilicorr_band_config, band_target);
    try std.testing.expect(!c.parallel_declared); // no parallel_drafting_token_id key anywhere
    try std.testing.expectEqual(@as(u32, 7), dcfg.row0Token(&c, 7)); // silent: pending rides row 0
    const declared_zero = try withParallelDecl(a, "0");
    defer a.free(declared_zero);
    const z = try dcfg.Config.parse(a, declared_zero, band_target);
    try std.testing.expect(z.parallel_declared); // declared 0 is a declaration, not silence
    try std.testing.expectEqual(@as(u32, 0), dcfg.row0Token(&z, 7));
}
