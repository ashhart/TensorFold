const std = @import("std");
const gpu = @import("metal.zig");
const Tensor = gpu.Tensor;
const mtl = gpu.mtl;
const sources = @import("kernel_sources");

fn reserve(g: *gpu.Gpu, slot: *?Tensor, count: usize) !Tensor {
    if (slot.*) |current| if (current.count >= count) return current;
    if (g.batch == null) {
        if (slot.*) |current| current.deinit();
        slot.* = null;
    }
    const replacement = try g.allocDedicated(count, .f32);
    if (slot.*) |current| current.deinit();
    slot.* = replacement;
    return replacement;
}

const Workspace = struct { probabilities: Tensor, stats: Tensor };

fn workspace(g: *gpu.Gpu, rows: usize, blocks: usize, heads: usize, batch: usize) !Workspace {
    if (rows == 0 or blocks == 0 or heads == 0 or batch == 0) return error.AttentionShape;
    if (blocks > std.math.maxInt(i32) / 32) return error.AttentionTooLarge;
    const planes = try std.math.mul(usize, try std.math.mul(usize, batch, heads), rows);
    const probabilities = try reserve(g, &g.attention_probabilities, try std.math.mul(usize, planes, blocks * 32));
    const stats = try reserve(g, &g.attention_stats, try std.math.mul(usize, planes, blocks + 1));
    return .{ .probabilities = probabilities, .stats = stats };
}

pub fn prepare(g: *gpu.Gpu, rows: usize, keys: usize, heads: usize, batch: usize) !void {
    if (g.batch != null) return error.AttentionWorkspaceInBatch;
    _ = try workspace(g, rows, try std.math.divCeil(usize, keys, 32), heads, batch);
}

pub fn run(g: *gpu.Gpu, q: Tensor, k: Tensor, v: Tensor, out: Tensor, p: [24]i32, scale: f32) !void {
    const work = try workspace(g, @intCast(p[2]), @intCast(p[6]), @intCast(p[1]), @intCast(p[0]));
    const scores = try g.pipeline("im_attention_scores256");
    const values = try g.pipeline("im_attention_values256");
    const pass = g.command();
    sources.tensor_attention_passes.encode(mtl, pass.enc, scores, values, [_]Tensor{ q, k, v, work.probabilities, work.stats, out }, p, scale);
    try g.finish(pass.cb, pass.enc);
}
