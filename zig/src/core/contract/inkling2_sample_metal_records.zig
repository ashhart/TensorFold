//! Inkling 2 Mini f32 GPU sampler launches (zig/tests/contract/metal_inkling2_sample.zig).
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "metal inkling2 sample tf_sample_full_f32", .threads = 1024, .tg_memory = 9760, .reserved = 1024, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal inkling2 sample tf_sample_full_ids_f32", .threads = 1024, .tg_memory = 9760, .reserved = 1024, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal inkling2 sample tf_topk_slices_f32", .threads = 512, .tg_memory = 12304, .reserved = 512, .m1 = 512, .m2 = 512 },
    .{ .name = "metal inkling2 sample tf_ink2_topk_rule_f32", .threads = 64, .tg_memory = 17168, .reserved = 64, .m1 = 64, .m2 = 64 },
    .{ .name = "metal inkling2 sample tf_ink2_topk_rule_ids_f32", .threads = 64, .tg_memory = 17168, .reserved = 64, .m1 = 64, .m2 = 64 },
    .{ .name = "metal inkling2 sample tf_ink2_rule_1024_f32", .threads = 1024, .tg_memory = 13472, .reserved = 1024, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal inkling2 sample tf_ink2_rule_1024_ids_f32", .threads = 1024, .tg_memory = 13472, .reserved = 1024, .m1 = 1024, .m2 = 1024 },
};
