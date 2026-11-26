//! Per-stream slot state and verify/draft request shapes for GLM-5.3 decode rounds.
const std = @import("std");
const gpu = @import("gpu.zig");
const tr = @import("tree.zig");

const R = gpu.max_rows;

/// Drafts a round at most (r=1..8 full block; GPU early-stop gate skips low-confidence tail steps).
pub const max_depth = 7;

/// A verify window: its slot and input tokens (the pending token, then the drafts).
pub const Win = struct { slot: u32, tokens: []const u32 };

/// A draft request: the kept rows (`prompt`: the prompt's last row) and the tokens after them, then a chain.
pub const Draft = struct { slot: u32, prompt: bool, follow: []const u32, depth: u32, path: []const u32 = &.{} };

pub const Slot = struct {
    used: bool = false,
    /// Cache rows written (the stream's tokens so far).
    len: u32 = 0,
    prompt: std.ArrayList(u32) = .empty,
    /// The last window's first row here, its rows not yet settled (kept unless a keep says), the last settle's rows.
    win_row: u32 = 0,
    win_tok0: u32 = 0,
    open: u32 = 0,
    kept: u32 = 0,
    held: [max_depth]u32 = @splat(0),
    held_probs: [max_depth]f32 = @splat(0),
    held_top4_ids: [4]u32 = @splat(0),
    held_top4_p: [4]f32 = @splat(0),
    held_probs_valid: bool = false,
    held_n: u32 = 0,
    fused_start: u32 = 0,
    fused_depth: u32 = 0,
    fused_valid: bool = false,
    fused_b1: bool = false,
    cal0_scale: f64 = 1.0,
    cal_scale: f64 = 1.0,
    fused_picks: [R]u32 = @splat(0),
    fused_path: [8]u32 = .{ 0, 1, 2, 3, 4, 5, 6, 7 },
    tree: tr.Plan = .{},
};
