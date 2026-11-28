//! Opt-in short-context proposal passes; the target and N15 tree budget keep their existing rules.
const std = @import("std");
const draft = @import("cuda_draft.zig");
const dr = @import("lanes").drafter;
pub const environment = @import("sys").environment;
pub const cached_threshold: u32 = 16384;
pub const setting = "TF_Q35_DFLASH_CONTEXT_HOLD";

/// Read only when the adapter is constructed; an unset switch retains every original callback.
pub fn enabled() bool {
    return environment.read(bool, setting, false) catch false;
}

/// The complete shared frame stays full if any member is long, deep or has an explicit lane shape.
pub fn rows(items: []const dr.Hold, full: usize) usize {
    if (full != 16 or items.len == 0) return full;
    for (items) |it| {
        if (it.position == 0 or it.position - 1 >= cached_threshold or it.depth > 7 or it.lanes != null) return full;
    }
    return 8;
}

pub fn selectHold(d: *draft.Draft, ctx: *draft.Ctx, it: dr.Hold) !void {
    return @call(.always_inline, selectHoldBlock, .{ d, ctx, it, d.c.block });
}

pub fn selectHoldBlock(d: *draft.Draft, ctx: *draft.Ctx, it: dr.Hold, logical_block: usize) !void {
    const row_count = d.c.block - 1;
    ctx.grown = 0;
    // a sampled hold's candidates ranked as its target draw ranks them (its seed, each draft's position)
    d.drawn = .{ .sampling = it.sampling, .position = it.position };
    defer d.drawn = .{};
    if (it.lanes) |shape| return d.selectLanes(ctx, it.pending, row_count, shape);
    const limit = if (logical_block == 16) d.tree_nodes else 0;
    if (limit > 0) {
        // the whole budget whenever drafts are asked: the block's forward costs the same at any depth
        const n = if (it.depth == 0) 0 else limit;
        const capture: ?*draft.tree.GrowTrace = if (d.trace == null or d.trace.?.captures.len == 0) null else &d.trace.?.captures[ctx.trace_base / (d.c.block - 1)];
        ctx.grown = draft.tree.growTraced(d.candidates(row_count), it.pending, row_count, n, &ctx.held, &ctx.parents, &ctx.own, capture);
        ctx.held_n = ctx.grown;
    } else {
        d.select(ctx, it.pending, row_count);
        ctx.held_n = @min(it.depth, row_count);
    }
}

pub fn adapt(d: *draft.Draft, original: dr.Drafter, on: bool) dr.Drafter {
    if (!on) return original;
    if (d.c.kind != .dflash2 or d.c.block != 16 or d.tree_nodes != 15) return original;
    var changed = original;
    changed.vtable = &vtable;
    return changed;
}

const vtable: dr.Drafter.VTable = blk: {
    var v = draft.Draft.drafter_vtable;
    v.hold = hold;
    break :blk v;
};

fn hold(p: *anyopaque, items: []const dr.Hold) anyerror!void {
    const d: *draft.Draft = @ptrCast(@alignCast(p));
    const original = draft.Draft.drafter_vtable.hold;
    if (items.len == 0 or items.len > d.frame_streams) return original(p, items);
    const block = rows(items, d.c.block);
    if (block == d.c.block) return original(p, items);
    const t0 = std.Io.Clock.awake.now(d.io);
    defer draft.timing("context_hold_rows8", d.io, t0);
    var contexts: [draft.frame.max_streams]*draft.Ctx = undefined;
    for (items, contexts[0..items.len]) |it, *ctx| {
        ctx.* = d.streams.get(it.stream) orelse return error.UnknownStream;
        draft.frame.scan(ctx.*, it.tokens);
    }
    return draft.frame.forwardBlock(d, items, contexts[0..items.len], block);
}
