//! Whole-head chain and its unchanged baseline launches; compile usage must come from the device builder.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "gdn_prefill tf_gdn_chain_norm", .symbol = "tf_gdn_chain_norm", .image = "gdn_prefill", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "gdn_prefill chain_kernel", .symbol = "_ZN14tf_gdn_prefill12chain_kernelI13__nv_bfloat16S1_Li64ELi32EEEvPKT_S4_PKS1_PKfPKT0_S8_PfPS1_iii", .image = "gdn_prefill", .threads = 128 }, .usage = registry.none },
    .{ .launch = .{ .name = "qwen35 tf_q35_gdn_post", .symbol = "tf_q35_gdn_post", .image = "qwen35", .threads = 32 }, .usage = registry.none },
};
