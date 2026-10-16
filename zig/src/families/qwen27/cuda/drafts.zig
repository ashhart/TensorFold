//! DFlash2 for a shared round's streams (multi.py's drafter halves): copies first, else every block in one batch.

const std = @import("std");
const lanes = @import("lanes");
const st = @import("state.zig");
const kern = @import("kernels.zig");
const dr = @import("draft.zig");
const dl = @import("draft_load.zig");
const lone = @import("lone.zig");

const bf = 2;
pub const taps_width = dl.taps.len * dl.hidden;
/// A shared window's drafts at most (MultiDecoder's max_rows - 1).
pub const max_nodes = st.max_rows - 1;

pub const Kind = enum { none, copy, tree };

/// One stream's drafting: its drafter context, its copy index, and the drafts it holds for its next window.
pub const Stream = struct {
    gpa: std.mem.Allocator,
    ctx: dr.Context,
    copies: lone.CopyIndex,
    kind: Kind = .none,
    tokens: std.ArrayList(u32) = .empty,
    parents: std.ArrayList(i32) = .empty, // each draft's parent draft (-1: under the pending row)
    scores: std.ArrayList(f64) = .empty, // each draft's path score (best_first's; 0 for a copy)

    pub fn init(gpa: std.mem.Allocator, ctx: dr.Context) Stream {
        return .{ .gpa = gpa, .ctx = ctx, .copies = .{ .gpa = gpa } };
    }

    pub fn deinit(x: *Stream) void {
        x.ctx.deinit();
        x.copies.deinit();
        x.tokens.deinit(x.gpa);
        x.parents.deinit(x.gpa);
        x.scores.deinit(x.gpa);
    }

    fn clear(x: *Stream) void {
        x.kind = .none;
        x.tokens.clearRetainingCapacity();
        x.parents.clearRetainingCapacity();
        x.scores.clearRetainingCapacity();
    }
};

/// A stream asking for its next drafts: its context through the pending token, and its sampling.
pub const Ask = struct { x: *Stream, context: []const u32, sampling: ?lanes.Sampling };

/// _mode, launch_blocks and finish_tree: a copied continuation where the context repeats, else a DFlash2 tree.
pub fn propose(d: *dr.DFlash2, asks: []const Ask) !void {
    var blocks: [st.max_streams]dr.Block = undefined;
    var which: [st.max_streams]usize = undefined;
    var n: usize = 0;
    for (asks, 0..) |a, k| {
        const x = a.x;
        x.clear();
        const copied = try x.copies.propose(a.context, max_nodes);
        if (copied.len > 0) {
            x.kind = .copy;
            try x.tokens.appendSlice(x.gpa, copied);
            for (0..copied.len) |r| {
                try x.parents.append(x.gpa, @as(i32, @intCast(r)) - 1);
                try x.scores.append(x.gpa, 0);
            }
        } else if (x.ctx.len > 0) {
            blocks[n] = .{ .ctx = &x.ctx, .pending = a.context[a.context.len - 1] };
            which[n] = k;
            n += 1;
        }
    }
    if (n == 0) return;
    try d.launchMany(blocks[0..n], @min(dr.block, max_nodes + 1));
    for (blocks[0..n], which[0..n], 0..) |b, k, j| {
        const a = asks[k];
        try d.finish(j, b.pending, a.context.len, max_nodes, a.sampling, &d.tree);
        a.x.kind = .tree;
        try a.x.tokens.appendSlice(a.x.gpa, d.tree.tokens.items);
        try a.x.parents.appendSlice(a.x.gpa, d.tree.parents.items);
        try a.x.scores.appendSlice(a.x.gpa, d.tree.scores.items);
    }
}

/// A stream's kept rows of the last round (rows of the round's taps, in path order).
pub const Kept = struct { x: *Stream, rows: []const u32 };

/// add_taps_streams: every stream's kept rows' taps gathered after the round's rows, then absorbed in one batch.
pub fn absorb(d: *dr.DFlash2, ops: kern.Ops, taps: u64, copy_items: u64, kept: []const Kept) !void {
    var items: std.ArrayList(kern.Copy) = .empty;
    defer items.deinit(d.gpa);
    var absorbs: [st.max_streams]dr.Absorb = undefined;
    const row = taps_width * bf;
    const gathered = taps + st.round_rows * row;
    for (kept, 0..) |k, j| {
        for (k.rows) |r| try items.append(d.gpa, .{ .dst = gathered + items.items.len * row, .src = taps + r * row, .bytes = row });
        absorbs[j] = .{ .ctx = &k.x.ctx, .rows = k.rows.len };
    }
    if (items.items.len == 0) return;
    try ops.upload(copy_items, std.mem.sliceAsBytes(items.items));
    try ops.copies(copy_items, items.items.len);
    try d.absorbMany(absorbs[0..kept.len], gathered);
}
