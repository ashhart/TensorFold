//! Nemotron 3.5 Super VL (nemotron_h_omni) on CUDA: the Nemotron engine and lane backend under the omni model type.
const std = @import("std");
const native = @import("cuda_native.zig");
const Config = @import("config.zig").Config;
const latent = @import("cuda_latent.zig");

pub const model_type = "nemotron_h_omni";
pub const formats = native.formats;
/// --vision-image-tokens unless given: four images at the most tokens NVIDIA's processor gives one (3,328 each).
pub const default_image_tokens: u32 = 13312;
pub const default_context = native.default_context;
pub const max_segments = native.max_segments;
pub const prompt_rows = native.prompt_rows;
pub const Options = native.Options;
pub const explains_open = true;
pub const Loaded = native.Loaded;
pub const LoneRun = native.LoneRun;
pub const open = native.open;
pub const explain = native.explain;
pub const warm_quiet = native.warm_quiet;
pub const lone_min_budget = native.lone_min_budget;
pub const snap_bytes = native.snap_bytes;
pub const snap_save = native.snap_save;
pub const snap_restore = native.snap_restore;
pub const snap_drop = native.snap_drop;
pub const snap_slots = native.snap_slots;
pub const snap_place = native.snap_place;
pub const window_rows = native.window_rows;
pub const attach = native.attach;
pub const fits = native.fits;

/// NVFP4 bytes a parameter takes: a 4-bit code and an e4m3 scale a 16.
const nvfp4_bytes: f64 = 0.5625;

/// Bytes this rank's weights take: rank 0 all but the other ranks' expert shares, rank r > 0 its expert shares alone.
pub fn weightShare(a: std.mem.Allocator, io: std.Io, dir: []const u8, total: u64) u64 {
    var c = Config.read(a, io, dir) catch return total;
    latent.place(&c) catch return total;
    if (c.world < 2 or c.latent == 0) return total;
    const moe: f64 = @floatFromInt(c.count(.moe) + 1); // the MTP head's MoE too
    const per: f64 = @floatFromInt(c.experts * 2 * c.expert_width * c.latent + 2 * c.shared_width * c.hidden);
    const experts: u64 = @intFromFloat(moe * per * nvfp4_bytes);
    const share = experts / c.world;
    if (c.rank > 0) return share;
    return total -| (experts - share);
}

pub const tensorWeightBytes = native.tensorWeightBytes;
