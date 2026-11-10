//! Draft proposal extensions keep their established arithmetic around the normal facts constructor.
const std = @import("std");
const dr = @import("lanes").drafter;
const feedback = @import("cuda_draft_feedback.zig");

pub fn Facts(comptime D: type, comptime make: anytype) type {
    return struct {
        pub fn facts(p: *anyopaque) dr.Facts {
            const d: *D = @ptrCast(@alignCast(p));
            var result = make(p);
            result.history_rows = @intCast(d.c.window);
            if (d.c.copy_match > 0) result.copy_match = d.c.copy_match;
            if (d.c.kind == .dflash2) result.tree_nodes = @intCast(d.tree_nodes);
            return result;
        }
    };
}

pub fn clearGraphs(d: anytype) void {
    if (d.graphs) |*cache| cache.dropAll();
}

pub fn nodeBudget(d: anytype) usize {
    return if (d.c.kind == .dflash2) @max(d.tree_nodes, d.auxiliary_nodes) else if (d.c.block == 16) d.tree_nodes else 0;
}

pub fn selectHold(d: anytype, ctx: anytype, it: dr.Hold) !void {
    const rows = d.c.block - 1;
    ctx.grown = 0;
    if (d.c.kind == .dflash2) try @import("cuda_draft_trace.zig").emit(d, it);
    // a sampled hold's candidates ranked as its target draw ranks them (its seed, each draft's position)
    d.drawn = .{ .sampling = it.sampling, .position = it.position };
    defer d.drawn = .{};
    if (it.lanes) |shape| return d.selectLanes(ctx, it.pending, rows, shape);
    const limit = nodeBudget(d);
    if (ctx.seed_n > 0) return @import("cuda_draft_refine.zig").select(d.candidates(rows), ctx, if (it.depth == 0) 0 else limit);
    if (limit > 0) {
        // the whole budget whenever drafts are asked: the block's forward costs the same at any depth
        const n = if (it.depth == 0) 0 else limit;
        ctx.grown = feedback.grow(d.candidates(rows), &ctx.feedback, it.position, d.feedback_weight, it.pending, rows, n, &ctx.held, &ctx.parents, &ctx.own);
        ctx.held_n = ctx.grown;
    } else {
        d.select(ctx, it.pending, rows);
        ctx.held_n = @min(it.depth, rows);
    }
}
