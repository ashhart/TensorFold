//! A prompt pass's last layer past its keys and values: none of its rows (keysOnly), or the kept row's block (first).
const std = @import("std");
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
