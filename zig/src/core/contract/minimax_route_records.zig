//! MiniMax-M3.1-Flash's route launches (metal_minimax_route*.zig); mm_router reserves the 256 threads it launches.
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "minimax_m3 route router mm_router", .threads = 256, .tg_memory = 2048, .reserved = 256 },
    .{ .name = "minimax_m3 route pick mm_route_pick", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 route groups mm_route_groups", .threads = 1024, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 route sort mm_sort", .threads = 1024, .tg_memory = 18944, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 route rows mm_rows_by_order", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 route prep mm_prep", .threads = 32, .tg_memory = 512, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 route index scores mm_index_scores", .threads = 128, .tg_memory = 17408, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 route index select mm_index_select", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 route decode partials mm_decode_vec", .threads = 512, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 route decode merge mm_merge_vec", .threads = 32, .tg_memory = 0, .m1 = 928, .m2 = 928 },
};
