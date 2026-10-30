//! Shared host metadata describes causal rows and completed index blocks; it never reads inference tensors.
const std = @import("std");
const qsa = @import("cuda_attention.zig");
pub const Plan = struct {
    rows: usize,
    start: usize,
    first_pool: usize,
    pool_count: usize,
    positions: [128]i32 = undefined,
    complete: [128]i32 = undefined,
    ends: [128]i32 = undefined,
    keys: [128]i32 = undefined,
    sparse: [128]i32 = undefined,
    pub fn init(g: qsa.Geometry, p: qsa.Call) !Plan {
        _ = try qsa.sizes(g, p);
        var out: Plan = .{ .rows = p.rows, .start = p.start, .first_pool = p.start / g.ratio, .pool_count = (p.start + p.rows) / g.ratio - p.start / g.ratio };
        for (0..p.rows) |r| {
            const end = p.start + r + 1;
            const blocks = end / g.ratio;
            out.positions[r] = @intCast(end - 1);
            out.complete[r] = @intCast(blocks);
            out.ends[r] = @intCast(end);
            out.sparse[r] = @intFromBool(blocks > g.top);
            out.keys[r] = @intCast(if (blocks > g.top) g.top * g.ratio + end % g.ratio else end);
        }
        return out;
    }
};

test "QSA plan: every prompt cut retains only complete pooled blocks and its own causal tail" {
    const t = std.testing;
    const g: qsa.Geometry = .{ .heads = 24, .kv = 4, .index_heads = 4, .index_dim = 128, .rotary = 64, .ratio = 4, .top = 2 };
    for (0..20) |start| for (1..17) |rows| {
        const plan = try Plan.init(g, .{ .rows = rows, .start = start, .capacity = 64, .eps = 1e-6 });
        try t.expectEqual((start + rows) / 4 - start / 4, plan.pool_count);
        for (0..rows) |r| {
            const one = try Plan.init(g, .{ .rows = 1, .start = start + r, .capacity = 64, .eps = 1e-6 });
            try t.expectEqual(one.positions[0], plan.positions[r]);
            try t.expectEqual(one.complete[0], plan.complete[r]);
            try t.expectEqual(one.keys[0], plan.keys[r]);
            try t.expectEqual(one.sparse[0], plan.sparse[r]);
        }
    };
}
