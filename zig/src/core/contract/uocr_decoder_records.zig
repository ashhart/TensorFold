//! Unlimited-OCR-2's decoder launches: each reserves its group's threads; M1/M2 from tf-kernel-contract --offline.
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "uocr decoder rows ud_gather", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr decoder step products ud_matvec", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr decoder norm ud_norm", .threads = 256, .tg_memory = 144, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr decoder rotary ud_qk", .threads = 128, .tg_memory = 1168, .reserved = 128, .m1 = 128, .m2 = 128 },
    .{ .name = "uocr decoder block keys ud_block_keys", .threads = 32, .tg_memory = 0, .reserved = 32, .m1 = 32, .m2 = 32 },
    .{ .name = "uocr decoder router ud_route", .threads = 256, .tg_memory = 17232, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr decoder attention ud_attend", .threads = 256, .tg_memory = 10432, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr decoder expert gate ud_moe_gate", .threads = 1024, .tg_memory = 256, .reserved = 1024, .m1 = 1024, .m2 = 1024 },
    .{ .name = "uocr decoder swiglu ud_silu_mul", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr decoder step experts ud_expert_gu", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr decoder step experts ud_expert_down", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr decoder combine ud_combine", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
};
