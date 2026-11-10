//! Flash Next's grouped expert launches (gx.zig): the thread counts fn_gx.metal reserves, and its threadgroup arrays.
const M = @import("limits.zig").MetalLaunch;
pub const metal = [_]M{
    .{ .name = "flashnext grouped gx_pick", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "flashnext grouped gx_list", .threads = 512, .tg_memory = 7488, .reserved = 512 },
    .{ .name = "flashnext grouped gx_gateup", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "flashnext grouped gx_down_one", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "flashnext grouped gx_down_tile", .threads = 64, .tg_memory = 1280, .reserved = 64 },
};
