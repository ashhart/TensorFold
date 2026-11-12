//! The candidate-block selection of one launch: block maxima of the index scores, the exact top blocks as a byte mask.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const glue = @import("glue.zig");

pub const Ref = glue.Ref;
/// Threads of the one group a row.
pub const threads = 1024;
/// The most blocks a row keeps.
pub const max_topk = 2048;

pub fn load(device: mtl.Device) !mtl.Pipeline {
    const lib = try mtl.Library.fromSource(device, sources.dsv41.cand, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, "tf_cand", false);
}

/// Candidate blocks of scores [rows][keys] (vis[row] keys count): mask bytes, 1 = candidate; keyed u32 scratch.
pub fn encode(pipe: mtl.Pipeline, e: mtl.ComputeEncoder, scores: Ref, vis: Ref, keyed: Ref, mask: Ref, rows: u32, keys: u32, blocks: u32, topk: u32) void {
    e.setPipeline(pipe);
    e.setValue([3]u32{ keys, blocks, @min(topk, max_topk) }, 0);
    e.setBuffer(scores.buf, scores.off, 1);
    e.setBuffer(vis.buf, vis.off, 2);
    e.setBuffer(keyed.buf, keyed.off, 3);
    e.setBuffer(mask.buf, mask.off, 4);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(threads, 1, 1));
}
