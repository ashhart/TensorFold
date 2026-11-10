//! The Flash Next CUDA engine's device bytes: the checkpoint's weights, a placement's, the vision tower's.
const std = @import("std");
const core = @import("core");
const weights = @import("cuda_prompt_wt.zig");
const mtp = @import("cuda_mtp.zig");
const config = @import("cuda_config.zig");
const exl3_memory = @import("exl3/pack/cuda_exl3_memory.zig");

/// Device bytes of the checkpoint: the n-gram tables stay mapped, the MTP head counts; 0 if unread.
pub fn deviceWeightBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) u64 {
    if (@import("cuda_tfq3.zig").isTfq3(gpa, io, dir)) return @import("cuda_tfq3.zig").deviceWeightBytes(gpa, io, dir, true) catch 0;
    if (exl3_memory.isExl3(gpa, io, dir)) return exl3_memory.weightBytes(gpa, io, dir);
    const main = weights.deviceBytes(gpa, io, dir) catch return 0;
    return main + mtp.headBytes(headFp8(gpa, io, dir) catch return 0, headShared(gpa, io, dir) catch mtp.Shared{ .width = weights.moe_w });
}

pub fn placementWeightBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, world: usize, rank: usize, drafts: bool) !u64 {
    if ((world != 1 and world != 2) or rank >= world) return error.BadRank;
    const mode = try config.mode(gpa, io, dir);
    if (mode == .tfq3) {
        if (world != 1) return error.UnsupportedQuantization;
        return @import("cuda_tfq3.zig").deviceWeightBytes(gpa, io, dir, drafts);
    }
    if (mode == .exl3) return exl3_memory.placement(gpa, io, dir, world, rank);
    if (mode == .modelopt) {
        const main = try @import("cuda_rank_memory.zig").estimate(gpa, io, dir, world, rank);
        // Dense draft projections and vocabulary rows exceed the affine head's estimate.
        return main + if (drafts and rank == 0) mtp.headBytes(try headFp8(gpa, io, dir), try headShared(gpa, io, dir)) + (1 << 30) else @as(u64, 0);
    }
    if (world != 1) return error.UnsupportedQuantization;
    const main = try weights.deviceBytes(gpa, io, dir);
    return main + if (drafts) mtp.headBytes(try headFp8(gpa, io, dir), try headShared(gpa, io, dir)) else @as(u64, 0);
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

/// The head's shared expert on the device: its gate_proj's rows and the form each of its projections is stored in.
pub fn headShared(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !mtp.Shared {
    var store = try weights.Store.open(gpa, io, dir);
    defer store.deinit();
    const root = "language_model.mtp.layers.0.mlp.shared_expert";
    if (!store.has(root ++ ".gate_proj.weight")) return .{ .width = weights.moe_w };
    // whole routed experts beside it are the modelopt load, which holds every row; the legacy path keeps moe_w
    if (!store.has("language_model.mtp.layers.0.mlp.experts.0.gate_proj.weight")) return .{ .width = weights.moe_w };
    const gate = try store.tensor(root ++ ".gate_proj.weight");
    const up = try store.tensor(root ++ ".up_proj.weight");
    const down = try store.tensor(root ++ ".down_proj.weight");
    return .{ .width = gate.dim(0), .gate = form(gate), .up = form(up), .down = form(down) };
}

/// One shared projection's stored form, from the tensor the checkpoint holds: E4M3 codes MXFP8, BF16 kept, else affine.
fn form(t: core.safetensors.Tensor) mtp.Format {
    return switch (t.dtype) {
        .bf16 => .bf16,
        .f8_e4m3 => .mxfp8,
        else => .affine,
    };
}

/// The worlds Flash Next serves on CUDA; `native/cuda_refusal.place` refuses others by name before admission.
pub const worlds = struct {
    pub const ids = [_]u32{ 1, 2 };
    pub const refusal = "Flash Next on CUDA runs on one GPU or at --tp 2 (two ranks): serve with --tp 1 or --tp 2";
};

/// The tower's weights and one call's scratch, which rank zero holds beside the text weights with --vision.
pub fn visionBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, rank: u32, _: u32) !u64 {
    if (rank != 0) return 0;
    const v = @import("qwen_vision");
    const c = ((try v.Settings.read(a, io, dir)) orelse return 0).tower; // no tower: open refuses --vision by name
    return v.Tower.weightBytes(c) + v.Tower.scratchBytes(c, v.max_patches);
}
