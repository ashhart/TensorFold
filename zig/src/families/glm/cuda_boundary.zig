//! GLM's one reduction boundary: each rank's fp32 partial gathered in rank order, which hcPost then adds rank 0 first.
const std = @import("std");
const tp = @import("tp");
const contract = @import("mesh").contract;
const prof_mod = @import("cuda_prof.zig");
const Forward = @import("cuda_forward.zig").Forward;

/// What every GLM layer sum, and the MTP block's two, still owe when their partials leave the kernels.
pub const owes: contract.Owes = .{ .axis = .tensor, .contract = .rank_fold };

comptime {
    if (contract.need(owes) != .gather_partials) @compileError("GLM's epilogue folds gathered partials (hcPost, residualAdd)");
}

/// Every rank's partial in rank order; an eligible TP2 RoCE gather may consume `post` for its caller.
pub fn resolve(f: *const Forward, rows: usize, post: ?@import("cuda_forward.zig").ResolvePost) !bool {
    const stamp = prof_mod.span(f.prof, f.stream, .all_gather);
    defer stamp.end();
    const c = &f.ranks[0].w.config;
    const bytes = rows * c.hidden * 4;
    switch (f.exchange) {
        .local => for (f.ranks) |to| for (f.ranks) |from| {
            if (to.w.share.rank == from.w.share.rank) continue;
            const off = from.w.share.rank * bytes;
            try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(to.b.gathered + off, from.b.gathered + off, bytes, f.stream.handle), "cuMemcpyDtoDAsync");
        },
        .nccl => |n| {
            const r = f.ranks[0];
            const mine = r.b.gathered + r.w.share.rank * bytes;
            if (post) |p| if (n.fuse) if (n.group) |g| {
                if (try g.gatherPost(f.stream, mine, r.b.gathered, bytes, c.hidden, p)) return true;
            };
            if (n.collective) |coll| {
                coll.allGather(mine, r.b.gathered, rows * c.hidden, .f32, f.stream.handle) catch |e| return tp.group.underlying(coll, e);
            } else {
                try n.lib.check(n.lib.api.ncclAllGather(mine, r.b.gathered, rows * c.hidden, .f32, n.comm, f.stream.handle), "ncclAllGather");
            }
        },
    }
    return false;
}

test "GLM owes a rank_fold over its tensor group" {
    try std.testing.expectEqual(contract.Contract.rank_fold, owes.contract);
}
