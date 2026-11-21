//! A profile's tape: a chunk's routes and key lists played back, so knock-outs leave later layers' loads as served.
const std = @import("std");
const mtl = @import("metal");
const wts = @import("weights.zig");
const fwd = @import("forward.zig");
const Ref = wts.Ref;

pub const Mode = enum { off, record, play };

pub const Tape = struct {
    route: Ref, // a MoE layer's routes: weights and order (pairs), then the experts' first pairs
    keys: Ref, // an MLA layer's key lists: [rows, width] i32
    route_mode: Mode = .off,
    keys_mode: Mode = .off,
    moe_at: usize = 0, // the pass's next MoE layer
    mla_at: usize = 0, // and MLA layer
    rows: u32,

    /// Every MoE layer's routes and every MLA layer's key lists of `rows`-row chunks.
    pub fn init(arena: anytype, moe_layers: u32, mla_layers: u32, rows: u32, topk: u32, experts: u32, width: u32) !Tape {
        return .{
            .route = try arena.buffer(@as(usize, moe_layers) * routeWords(rows, topk, experts) * 4),
            .keys = try arena.buffer(@as(usize, mla_layers) * rows * width * 4),
            .rows = rows,
        };
    }

    /// A pass starts at the first layers.
    pub fn rewind(t: *Tape) void {
        t.moe_at = 0;
        t.mla_at = 0;
    }
};

fn routeWords(rows: u32, topk: u32, experts: u32) usize {
    return 2 * @as(usize, rows) * topk + experts;
}

fn copy(x: *const fwd.Ctx, e: mtl.ComputeEncoder, src: Ref, dst: Ref, words: usize) void {
    e.setPipeline(x.k.copy_u32);
    fwd.bind(e, 0, .{ src, dst });
    e.setValue(@as(u32, @intCast(words)), 2);
    e.dispatchThreads(fwd.size(words, 1, 1), fwd.size(256, 1, 1));
}

/// A MoE layer's routes: false to compute them (then `keepRoute`), true when the tape played them into place.
pub fn playRoute(tape: ?*Tape, x: *const fwd.Ctx, e: mtl.ComputeEncoder, w: Ref, order: Ref, offsets: Ref, M: u32) bool {
    const t = tape orelse return false;
    if (t.route_mode != .play) return false;
    routeCopies(t, x, e, w, order, offsets, M, true);
    return true;
}

pub fn keepRoute(tape: ?*Tape, x: *const fwd.Ctx, e: mtl.ComputeEncoder, w: Ref, order: Ref, offsets: Ref, M: u32) void {
    const t = tape orelse return;
    if (t.route_mode == .record) routeCopies(t, x, e, w, order, offsets, M, false) else t.moe_at += 1;
}

fn routeCopies(t: *Tape, x: *const fwd.Ctx, e: mtl.ComputeEncoder, w: Ref, order: Ref, offsets: Ref, M: u32, play: bool) void {
    std.debug.assert(M <= t.rows);
    const n = @as(usize, M) * x.c.topk;
    const at = t.route.at(t.moe_at * routeWords(t.rows, x.c.topk, x.c.experts) * 4);
    const parts = [3]Ref{ w, order, offsets };
    const offs = [3]usize{ 0, n, 2 * n };
    const lens = [3]usize{ n, n, x.c.experts };
    for (parts, offs, lens) |live, o, len| {
        if (play) copy(x, e, at.at(o * 4), live, len) else copy(x, e, live, at.at(o * 4), len);
    }
    t.moe_at += 1;
}

/// An MLA layer's key lists: false to select them (then `keepKeys`), true when the tape played them into place.
pub fn playKeys(tape: ?*Tape, x: *const fwd.Ctx, e: mtl.ComputeEncoder, indices: Ref, M: u32) bool {
    const t = tape orelse return false;
    if (t.keys_mode != .play) return false;
    keyCopies(t, x, e, indices, M, true);
    return true;
}

pub fn keepKeys(tape: ?*Tape, x: *const fwd.Ctx, e: mtl.ComputeEncoder, indices: Ref, M: u32) void {
    const t = tape orelse return;
    if (t.keys_mode == .record) keyCopies(t, x, e, indices, M, false) else t.mla_at += 1;
}

fn keyCopies(t: *Tape, x: *const fwd.Ctx, e: mtl.ComputeEncoder, indices: Ref, M: u32, play: bool) void {
    std.debug.assert(M <= t.rows);
    const width = x.c.keyWidth();
    const at = t.keys.at(t.mla_at * @as(usize, t.rows) * width * 4);
    const words = @as(usize, M) * width;
    if (play) copy(x, e, at, indices, words) else copy(x, e, indices, at, words);
    t.mla_at += 1;
}
