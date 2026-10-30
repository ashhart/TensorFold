//! The lanes gate on the GPU: the lanes engine over the GPU target on rank 0; tokens must equal serial greedy.

const std = @import("std");
const iface = @import("draft/iface.zig");
const dspark = @import("draft/dspark.zig");
const drive = @import("draft/drive.zig");
const fwd = @import("forward.zig");
const target_mod = @import("target.zig");
const tree = @import("draft/tree.zig");
const branches = @import("draft/branches.zig");
const branches_oracle = @import("draft/branches_oracle.zig");

pub const Result = drive.Result;

/// DSpark's shape at this world (candidates a position: every rank's best min(64, 128 / world), gathered).
pub fn shapeOf(f: *const fwd.Forward) dspark.Shape {
    return shapeOfCfg(f.cfg, f.comm.world());
}

/// DSpark's shape from the checkpoint config and the world, for a backend without a CUDA Forward.
pub fn shapeOfCfg(cfg: *const @import("config.zig").Config, world: u32) dspark.Shape {
    const w = world;
    return .{ .block = cfg.dspark_block, .noise = cfg.dspark_noise_token, .window = cfg.window, .rank = cfg.dspark_markov_rank, .hidden = cfg.hidden, .candidates = w * @min(64, 128 / w) };
}

/// Rank 0: `max_new` tokens after `prompt` (already in the slot) through lanes with `pass`; `max_rows` the plan's.
pub fn generate(gpa: std.mem.Allocator, f: *fwd.Forward, pass: iface.Pass, drafts: bool, prompt: []const u32, first: u32, max_new: u32, max_rows: u32) !Result {
    var gt: target_mod.GpuTarget = .{ .f = f, .resume_first = first, .max_rows = max_rows };
    return drive.generate(gpa, gt.target(), pass, shapeOf(f), prompt, max_new, drafts);
}

/// Rank 0: as `generate`, with draft trees (`set`: TF_DSV41_TREE's knobs); tree counts and chains come back.
pub fn generateTrees(gpa: std.mem.Allocator, f: *fwd.Forward, pass: iface.Pass, prompt: []const u32, first: u32, max_new: u32, max_rows: u32, set: tree.Settings) !TreeResult {
    var gt: target_mod.GpuTarget = .{ .f = f, .resume_first = first, .max_rows = max_rows };
    const r = try branches_oracle.generate(gpa, gt.target(), pass, shapeOf(f), prompt, max_new, set);
    return .{ .r = r, .chains = gt.tree.stats };
}

pub const TreeResult = struct { r: branches_oracle.Result, chains: branches.Resolver.Stats };
