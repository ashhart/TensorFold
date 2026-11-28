//! Mini DFlash 2 Metal launches (zig/kernels/metal/inkling2/dflash2.metal).
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "metal inkling2 dflash2 embed tf_ink2_df2_embed", .threads = 256, .tg_memory = 32, .reserved = 256 },
    .{ .name = "metal inkling2 dflash2 norm tf_ink2_df2_norm", .threads = 256, .tg_memory = 32, .reserved = 256 },
    .{ .name = "metal inkling2 dflash2 norm16 tf_ink2_df2_norm16", .threads = 256, .tg_memory = 32, .reserved = 256 },
    .{ .name = "metal inkling2 dflash2 conv tf_ink2_df2_conv", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 dflash2 ctx_kv tf_ink2_df2_ctx_kv", .threads = 32, .tg_memory = 512, .reserved = 32 },
    .{ .name = "metal inkling2 dflash2 block_qkv tf_ink2_df2_block_qkv", .threads = 32, .tg_memory = 512, .reserved = 32 },
    .{ .name = "metal inkling2 dflash2 attend tf_ink2_df2_attend", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 dflash2 attend_merge tf_ink2_df2_attend_merge", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 dflash2 silu_mul tf_ink2_df2_silu_mul", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 dflash2 topk16 tf_ink2_df2_topk16", .threads = 512, .tg_memory = 10080, .reserved = 512 },
    .{ .name = "metal inkling2 dflash2 walk tf_ink2_df2_walk", .threads = 256, .tg_memory = 528, .reserved = 256 },
};

pub const fast_records = [_]M{
    .{ .name = "metal inkling2 draft_head_q4 batch tf_ink2_draft_head_q4_logits", .threads = 128, .tg_memory = 11008, .reserved = 128 },
    .{ .name = "metal inkling2 dflash2_fast bf16_nax_16 tf_ink2_df2_bf16_nax_16", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 dflash2_fast bf16_nax_32 tf_ink2_df2_bf16_nax_32", .threads = 128, .tg_memory = 0, .reserved = 128 },
};
