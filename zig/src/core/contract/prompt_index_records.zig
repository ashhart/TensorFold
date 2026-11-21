//! The indexed prompt gathers: the 128-thread groups the patched text reserves, and the gathers' dequantized tiles.
const M = @import("limits.zig").MetalLaunch;
const tile = @import("fsr_records.zig").gather_memory;
pub const metal = [_]M{
    .{ .name = "flashnext prompt index tf_gather_qmmw_nax_32", .threads = 128, .tg_memory = tile, .reserved = 128 },
    .{ .name = "flashnext prompt index tf_gather_qmmw_nax_64", .threads = 128, .tg_memory = tile, .reserved = 128 },
    .{ .name = "flashnext prompt index tf_gather_qmmw_nax_128", .threads = 128, .tg_memory = tile, .reserved = 128 },
    .{ .name = "flashnext prompt index tf_gather_guw_nax_32", .threads = 128, .tg_memory = 2 * tile, .reserved = 128 },
    .{ .name = "flashnext prompt index tf_gather_guw_nax_64", .threads = 128, .tg_memory = 2 * tile, .reserved = 128 },
    .{ .name = "flashnext grouped index gx_pick", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "flashnext grouped index gx_list", .threads = 512, .tg_memory = 7488, .reserved = 512 },
    .{ .name = "flashnext grouped index gx_gateup", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "flashnext grouped index gx_down_one", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "flashnext grouped index gx_down_tile", .threads = 64, .tg_memory = 1280, .reserved = 64 },
};
