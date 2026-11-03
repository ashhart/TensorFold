//! Unlimited-OCR-2's matvec launches (zig/tests/contract/metal_uocr_matvec.zig): each kernel's source reserves its 256
//! threads ([[max_total_threads_per_threadgroup(256)]]), so M1 and M2 build them at 256 or not at all.
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "uocr matvec decode projection ud_matvec", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr matvec routed gate up ud_expert_gu", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr matvec routed down ud_expert_down", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
};
