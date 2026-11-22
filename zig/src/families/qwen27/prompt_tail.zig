//! A prompt pass's last layer past its keys and values: none of its rows (keysOnly), or the kept row's block (first).
const std = @import("std");
const mtl = @import("metal");
const forward = @import("forward.zig");
const wts = @import("gpu_weights.zig");
const attn = @import("attention.zig");
const Frame = @import("gpu_frame.zig").Frame;
const Runner = @import("decode_round.zig").Runner;
const tail = @import("core").prompt_tail;
const lane = @import("core").lane_projection;

/// The tail starts on a lane block, so its q, o and MLP dispatches are the ones a whole pass makes there.
pub const block: usize = lane.reg_rows;
comptime {
    std.debug.assert(block == tail.rows);
}

/// What reads the last layer's rows besides the head, and whether the pass keeps a row at all.
pub const Facts = struct {
    /// The pass keeps no row's logits (a prompt chunk before the last).
    keeps_none: bool,
    /// Decoded rows (no prompt kernels), a verify that is not an accepted chain, or a head other than the last row.
    plain: bool = false,
    /// A drafter's or MTP head's tap on the last layer's rows.
    last_tapped: bool = false,
    /// A tensor-parallel pair: each layer's seams are counted on both ranks.
    pair: bool = false,
    /// The Neural Engine's MLP share steps through every layer.
    ane: bool = false,
    images: bool = false,
    moe: bool = false,
    /// A last-layer projection whose dispatches depend on the row count past a 128-row block (raw or long rows).
    unaligned: bool = false,
    last_attention: bool = true,
};

pub fn keysOnly(f: Facts) bool {
    return f.keeps_none and !f.last_tapped and !f.pair and !f.ane and f.last_attention;
}

/// The tail's first row: the last lane block of the chunk's rows, or null when every row runs the last layer.
pub fn first(f: Facts, total: usize, nseg: usize) ?u32 {
    const readers = f.keeps_none or f.plain or f.pair or f.ane or f.images or f.last_tapped;
    _ = tail.start(.{ .total = total, .nseg = nseg, .readers = readers, .fits = !f.moe and !f.unaligned }) orelse return null;
    return @intCast(block * ((total - 1) / block));
}

/// What reads runner `r`'s last layer rows in a pass at `head`, as decode_round's verify reads it.
pub fn facts(r: anytype, head: anytype, accepted: bool, images: bool) Facts {
    const m = r.model;
    const last = m.config.layers - 1;
    const layer = m.weights.layers[last];
    const out = switch (layer.mixer) {
        .linear => |g| g.out,
        .attention => |a| a.out,
    };
    var aligned = m.kernels.blockRows(out) and m.kernels.blockRows(layer.gu) and m.kernels.blockRows(layer.down);
    if (layer.mixer == .attention) aligned = aligned and m.kernels.blockRows(layer.mixer.attention.q);
    return .{
        .keeps_none = head == .none,
        .plain = !m.kernels.prompt or !accepted or head != .last or r.tail != null,
        .last_tapped = r.capture_taps and std.mem.indexOfScalar(usize, r.taps.ids[0..r.taps.count], last) != null,
        .pair = r.reduce != null,
        .ane = m.ane != null,
        .images = images,
        .moe = layer.moe != null,
        .unaligned = !aligned,
        .last_attention = layer.mixer == .attention,
    };
}

/// Whether `r`'s pass at `head` runs its last layer's keys and values alone (these facts on the runner).
pub fn on(r: *const Runner, head: forward.Head) bool {
    const last = r.model.config.layers - 1;
    return keysOnly(.{
        .keeps_none = head == .none,
        .last_tapped = r.capture_taps and std.mem.indexOfScalar(usize, r.taps.ids[0..r.taps.count], last) != null,
        .pair = r.reduce != null,
        .ane = r.model.ane != null,
        .last_attention = r.model.weights.layers[last].mixer == .attention,
    });
}

/// The last layer of a pass that keeps no row: k and v into the cache, as the runner's attentionForward writes them.
pub fn keysForward(ptr: *anyopaque, e: mtl.ComputeEncoder, attn_block: usize, weights: wts.Attention, f: *Frame, rows: u32) !void {
    const r: *Runner = @ptrCast(@alignCast(ptr));
    const plan = (r.current_attention orelse return error.RoundNotReady).*;
    const layer = r.attention_index[attn_block];
    var views: [attn.planning.max_streams]attn.Cache = undefined;
    const caches = r.layerCaches(layer, r.current_round orelse return error.RoundNotReady, &views);
    try r.bindings[layer].upload(plan, caches);
    try r.model.kernels.quant(e, weights.kv, f.get(.input), forward.sums(f, .sums, rows), f.get(.dims), f.get(.mixed), rows);
    e.barrier();
    try r.attention.keysOnly(e, &r.scratch, &r.bindings[layer], plan, .{ .qg = f.get(.q_gate), .mixed = f.get(.mixed), .q_gain = wts.ref(weights.q_norm), .k_gain = wts.ref(weights.k_norm), .norm_q = f.get(.norm_query), .norm_k = f.get(.norm_key), .query = f.get(.query), .rotated_key = f.get(.rotated_key), .positions = f.get(.positions), .rows = rows, .eps = r.model.config.eps, .theta = r.model.config.rope_theta, .h_end = r.current_placed.h_end, .w_end = r.current_placed.w_end, .plane = f.capacity, .geometry = r.geometry, .shift = r.model.config.shift_norms, .rope = r.model.config.rope }, caches);
}

test "qwen27 prompt tail: each planted reader of the last layer's rows refuses it" {
    const open: Facts = .{ .keeps_none = true };
    try std.testing.expect(keysOnly(open));
    var refused: usize = 0;
    inline for (.{ "last_tapped", "pair", "ane" }) |name| {
        var f = open;
        @field(f, name) = true;
        if (!keysOnly(f)) refused += 1 else std.debug.print("tail kept with {s}\n", .{name});
    }
    for ([_]Facts{ .{ .keeps_none = false }, .{ .keeps_none = true, .last_attention = false } }) |f| {
        if (!keysOnly(f)) refused += 1;
    }
    try std.testing.expectEqual(@as(usize, 5), refused);
}

test "qwen27 prompt tail: a plain chunk past one block starts on its last lane block" {
    for ([_]usize{ 129, 256, 257, 300, 2048 }) |total| {
        const s = first(.{ .keeps_none = false }, total, 1).?;
        try std.testing.expect(s % block == 0 and s < total and total - s <= block);
    }
    try std.testing.expectEqual(@as(?u32, null), first(.{ .keeps_none = false }, block, 1));
}

test "qwen27 prompt tail: each planted reader or misfit refuses the block" {
    const open: Facts = .{ .keeps_none = false };
    try std.testing.expect(first(open, 4096, 1) != null);
    var refused: usize = 0;
    const names = .{ "keeps_none", "plain", "pair", "ane", "images", "last_tapped", "moe", "unaligned" };
    inline for (names) |name| {
        var f = open;
        @field(f, name) = true;
        if (first(f, 4096, 1) == null) refused += 1 else std.debug.print("tail kept with {s}\n", .{name});
    }
    try std.testing.expectEqual(@as(usize, names.len), refused);
    try std.testing.expectEqual(@as(?u32, null), first(open, 4096, 2));
}
