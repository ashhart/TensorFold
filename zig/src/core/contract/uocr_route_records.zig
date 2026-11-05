//! Unlimited-OCR-2 route launches (tests/contract/metal_uocr_route.zig); m1/m2 from the M1 and M2 compiles.
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    // one 32-thread group a block: wts[32]
    .{ .name = "uocr route block keys ud_block_keys", .threads = 32, .tg_memory = 128, .m1 = 32, .m2 = 32 },
    // one simdgroup a query: x[4096] fp32 and cand[4096] bytes, 20,480 of 32,768 (4,096 blocks: 131,072 keys)
    .{ .name = "uocr route router ud_route", .threads = 32, .tg_memory = 20480, .m1 = 32, .m2 = 32 },
    // one 64-thread group a row: part[33] and score[64], 388 bytes padded to 400
    .{ .name = "uocr route moe gate ud_moe_gate", .threads = 64, .tg_memory = 400, .m1 = 64, .m2 = 64 },
    .{ .name = "uocr route moe combine ud_moe_combine", .threads = 256, .tg_memory = 0, .m1 = 256, .m2 = 256 },
};
