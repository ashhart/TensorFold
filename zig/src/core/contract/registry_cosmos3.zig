//! Cosmos3-Edge's Metal launches (cosmos3.metal) for gate 5: threads a group, threadgroup bytes, M1/M2 compiled limits.
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "cosmos3 8-bit embedding rows cos3_embed_b8", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "cosmos3 image rows cos3_place", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "cosmos3 RMS norm cos3_norm", .threads = 512, .tg_memory = 144, .m1 = 1024, .m2 = 1024 },
    .{ .name = "cosmos3 M-RoPE and cache write cos3_qk", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "cosmos3 relu^2 cos3_relu2", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "cosmos3 decode attention cos3_attn_128", .threads = 32, .tg_memory = 512, .m1 = 1024, .m2 = 1024 },
    .{ .name = "cosmos3 text flash attention cos3_flash_128", .threads = 128, .tg_memory = 22656, .m1 = 512, .m2 = 512 },
    .{ .name = "cosmos3 tower flash attention cos3_flash_72", .threads = 128, .tg_memory = 15488, .m1 = 768, .m2 = 768 },
    .{ .name = "cosmos3 layer norm cos3_layer_norm", .threads = 256, .tg_memory = 144, .m1 = 1024, .m2 = 1024 },
    .{ .name = "cosmos3 bias and GELU cos3_bias", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "cosmos3 add cos3_add", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
};
