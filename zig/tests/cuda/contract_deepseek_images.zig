//! Every registered DeepSeek image is explicit; unknown names continue to the refusing common resolver.
const std = @import("std");
pub const names = [_][]const u8{ "dsv41_exl3_experts", "dsv41_x3ld", "dsv41_x3pf", "dsv41_x3gm", "dsv41_linear", "dsv41_x3seg", "dsv41_dense3", "dsv41_attn", "dsv41_topk", "dsv41_mhc", "dsv41_mhc_pf", "dsv41_router_gemv", "dsv41_pfdense_k8", "dsv41_pfdense_k10", "dsv41_pfdense_k12", "dsv41_pfdense_k16", "dsv41_engram_gate", "dsv41_l2pace", "dsv41_l2pf", "dsv41_x3gm_plan", "dsv41_topk_keys", "dsv41_pointwise", "dsv41_kvsplit", "dsv41_glue", "dsv41_x3gm3", "dsv41_pfglue", "dsv41_router_glue", "dsv41_kv_glue", "dsv41_x3ld_epi", "dsv41_topk_b", "dsv41_sampling", "dsv41_lin2" };
fn Blob(comptime name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile("contract_fatbin_" ++ name).*;
    };
}
pub fn image(name: []const u8) ?[]const u8 {
    inline for (names) |n| if (std.mem.eql(u8, name, n)) return &Blob(n).bytes;
    return null;
}
