//! GLM-5.3-Flash's latent attention for rows whose keys all fit index_topk: every key, the 64 heads as one matrix.
const std = @import("std");
const mtl = @import("metal");
const fwd = @import("forward.zig");
const Ref = @import("weights.zig").Ref;
const Ctx = fwd.Ctx;
const bind = fwd.bind;
const shape = fwd.shape;
const size = fwd.size;

/// One row's 64 heads over keys [0, n) as one 64-row matrix against the latent keys: scores, a precise softmax, values.
pub fn attendDense(x: *const Ctx, e: mtl.ComputeEncoder, keys: Ref, q: Ref, scores: Ref, probs: Ref, out: Ref, n: u32) void {
    const k = x.k;
    const H = x.c.mla_heads;
    e.setPipeline(k.latent_scores);
    bind(e, 0, .{ q, keys, scores });
    e.setValue([2]i32{ @intCast(n), @intFromBool(n < 256) }, 3); // fewer than 256 keys: the 512 dims in two halves
    e.dispatchGroups(size((n + 31) / 32, 1, 1), size(2 * H, 1, 1)); // a simdgroup each 16 heads
    e.setPipeline(k.softmax);
    bind(e, 0, .{scores});
    shape(e, 1, .{n});
    bind(e, 2, .{probs});
    e.dispatchGroups(size(H, 1, 1), size(((n + 3) / 4 + 31) / 32 * 32, 1, 1));
    e.setPipeline(k.latent_values);
    bind(e, 0, .{ probs, keys, out });
    e.setValue([2]i32{ @intCast(n), if (n > 1024) 1024 else 0 }, 3); // past 1,024 keys: the first 1,024 apart
    e.dispatchGroups(size(x.c.kv_lora / 32, 1, 1), size(2 * H, 1, 1));
}

/// `rows` rows (keys n0, n0 + 1, ..) each as attendDense leaves it, in three launches for all of them.
pub fn attendRows(x: *const Ctx, e: mtl.ComputeEncoder, keys: Ref, q: Ref, scores: Ref, probs: Ref, out: Ref, n0: u32, rows: u32) void {
    const k = x.k;
    const H = x.c.mla_heads;
    const lat: i32 = @intCast(H * x.c.kv_lora); // a row's latent queries and outputs
    const plane: i32 = @intCast(H * x.c.i_topk); // and its scores
    const n_max = n0 + rows - 1;
    const args = [4]i32{ @intCast(n0), lat, plane, lat };
    e.setPipeline(k.latent_scores_rows);
    bind(e, 0, .{ q, keys, scores });
    e.setBytes(std.mem.asBytes(&args), 3);
    e.dispatchGroups(size((n_max + 31) / 32, H / 16, rows), size(32, 1, 1)); // a simdgroup each 16 heads
    e.setPipeline(k.softmax_rows);
    bind(e, 0, .{scores});
    e.setValue([2]i32{ @intCast(n0), plane }, 1);
    bind(e, 2, .{probs});
    e.dispatchGroups(size(H, rows, 1), size(((n_max + 3) / 4 + 31) / 32 * 32, 1, 1));
    e.setPipeline(k.latent_values_rows);
    bind(e, 0, .{ probs, keys, out });
    e.setBytes(std.mem.asBytes(&args), 3);
    e.dispatchGroups(size(x.c.kv_lora / 32, H / 16, rows), size(32, 1, 1));
}
