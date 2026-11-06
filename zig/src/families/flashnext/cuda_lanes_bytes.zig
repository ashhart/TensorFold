//! The Flash Next CUDA engine's device bytes: the checkpoint's weights, a placement's, the vision tower's.
const std = @import("std");
const weights = @import("cuda_prompt_wt.zig");
const mtp = @import("cuda_mtp.zig");
const config = @import("cuda_config.zig");

/// Device bytes of the checkpoint: the n-gram tables stay mapped, the MTP head counts; 0 if unread.
pub fn deviceWeightBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) u64 {
    const main = weights.deviceBytes(gpa, io, dir) catch return 0;
    return main + mtp.headBytes(headFp8(gpa, io, dir) catch return 0);
}

pub fn placementWeightBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, world: usize, rank: usize, drafts: bool) !u64 {
    if ((world != 1 and world != 2) or rank >= world) return error.BadRank;
    const mode = try config.mode(gpa, io, dir);
    if (mode == .modelopt) {
        const main = try @import("cuda_rank_memory.zig").estimate(gpa, io, dir, world, rank);
        // Dense draft projections and vocabulary rows exceed the affine head's estimate.
        return main + if (drafts and rank == 0) mtp.headBytes(try headFp8(gpa, io, dir)) + (1 << 30) else @as(u64, 0);
    }
    if (world != 1) return error.UnsupportedQuantization;
    const main = try weights.deviceBytes(gpa, io, dir);
    return main + if (drafts) mtp.headBytes(try headFp8(gpa, io, dir)) else @as(u64, 0);
}

/// Whether the MTP head's routed experts are block FP8: its first expert's gate weight is F8_E4M3 (absent: false).
pub fn headFp8(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !bool {
    var store = try weights.Store.open(gpa, io, dir);
    defer store.deinit();
    // the store's aliases take the checkpoint's mtp.layers.0... as well as language_model.mtp.layers.0...
    const name = "language_model.mtp.layers.0.mlp.experts.0.gate_proj.weight";
    if (!store.has(name)) return false;
    return (try store.tensor(name)).dtype == .f8_e4m3;
}

/// The tower's weights and one call's scratch, which rank zero holds beside the text weights with --vision.
pub fn visionBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
    const v = @import("qwen_vision");
    const c = ((try v.Settings.read(a, io, dir)) orelse return error.NoVisionTower).tower;
    return v.Tower.weightBytes(c) + v.Tower.scratchBytes(c, v.max_patches);
}
