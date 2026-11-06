//! Native learner launches declare their geometry; CUDA compiler receipts stay absent until measured per image and SM.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");
const names = .{ [_][]const u8{ "tf_lw_linear_back", "tf_lw_rms_back", "tf_lw_hc_act_back", "tf_lw_hc_mix_back", "tf_lw_hc_branch_back", "tf_lw_swiglu_back", "tf_lw_loss", "tf_lw_nudge", "tf_lw_gate_back", "tf_lw_rope_rms_back", "tf_lw_pocket_act", "tf_lw_pocket_add", "tf_lw_pocket_back", "tf_lw_pocket_coeff" }, [_][]const u8{ "tf_lw_gdn_front", "tf_lw_gdn_norm", "tf_lw_gdn_readout_back", "tf_lw_gdn_scan_back", "tf_lw_gdn_qk_back", "tf_lw_gdn_gates_back", "tf_lw_gdn_conv_back" }, [_][]const u8{ "tf_lw_qsa_scores", "tf_lw_qsa_softmax_back", "tf_lw_qsa_query_back", "tf_lw_qsa_kv_back" }, [_][]const u8{ "tf_lw_g4_copy", "tf_lw_g4_zero", "tf_lw_g4_move", "tf_lw_g4_rms_back", "tf_lw_g4_geglu_back", "tf_lw_g4_softcap_back", "tf_lw_g4_scale", "tf_lw_g4_rope_back" } };
const suffixes = [_][]const u8{ "", "_bad_nan", "_bad_repeat", "_bad_tail", "_bad_sums" };
pub const count = (names[0].len + names[1].len + names[2].len + names[3].len) * suffixes.len;
pub const cuda = blk: {
    @setEvalBranchQuota(50000);
    var out: [count]registry.CudaRecord = undefined;
    var at = 0;
    for (names, .{ "living_rows", "living_gdn", "living_qsa", "living_gemma" }) |symbols, image| for (symbols) |base| for (suffixes) |suffix| {
        const symbol = base ++ suffix;
        out[at] = .{ .launch = .{ .name = image ++ " " ++ symbol, .symbol = symbol, .image = image, .threads = 128 }, .usage = registry.none, .later_pending = true };
        at += 1;
    };
    break :blk out;
};
pub const metal = blk: {
    @setEvalBranchQuota(50000);
    var out: [count]limits.MetalLaunch = undefined;
    var at = 0;
    for (names, .{ "rows", "gdn", "qsa", "gemma" }) |symbols, label| for (symbols) |base| for (suffixes) |suffix| {
        const symbol = base ++ suffix;
        out[at] = .{ .name = "living-native " ++ label ++ " " ++ symbol, .threads = 128, .tg_memory = 0, .reserved = 128 };
        at += 1;
    };
    break :blk out;
};
pub fn source(comptime Family: type, a: std.mem.Allocator, name: []const u8) ![]u8 {
    return Family.train_source.named(a, name);
}

test "living native resources: absent compiler records never qualify a CUDA image" {
    for (cuda) |record| {
        try std.testing.expectEqual(limits.sms.len, try registry.checkCuda(std.testing.allocator, &.{record}, true));
        try std.testing.expectEqual(registry.none, registry.usageFor(record, .{ .major = 13, .minor = 2 }));
    }
    try std.testing.expectEqual(@as(usize, 0), try registry.checkMetal(std.testing.allocator, &metal, false));
}
