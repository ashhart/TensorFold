//! Qwen 3.5's facts for core's prompt tail: what reads the last layer's other rows, and when that layer fits.
const std = @import("std");
const core = @import("core");
const tail = core.prompt_tail;

pub const rows = tail.rows;

/// A partial final-layer tap need: compute from its first needed row, unless another reader blocks the tail.
pub fn tapTailStart(candidate: ?usize, tapped: bool, taps_active: bool, tap_from: ?usize) ?usize {
    const tail_first = candidate orelse return null;
    if (!tapped or !taps_active) return tail_first;
    const from = tap_from orelse return null;
    if (from == 0) return null;
    return @min(tail_first, from);
}

/// Every reader of the last layer's other rows on this family, and what its last layer must be.
pub const Facts = struct {
    hidden: bool = false,
    hooks: bool = false,
    int8: bool = false,
    images: bool = false,
    odd_group: bool = false,
    last_attention: bool = true,
    last_tapped: bool = false,

    pub fn of(f: Facts, total: usize, nseg: usize) tail.Facts {
        return .{
            .total = total,
            .nseg = nseg,
            .readers = f.hidden or f.hooks or f.int8 or f.images or f.last_tapped,
            .fits = f.last_attention and !f.odd_group,
        };
    }
};

/// Whether something reads the last layer's rows in a prompt chunk besides the head.
fn lastTapped(e: anytype) bool {
    const last = e.layers.len - 1;
    for (e.taps[0..e.n_taps]) |t| if (t == last) return true;
    return false;
}

/// Whether this prompt chunk writes its taps: not when it ends before the MTP tap boundary (cuda_engine's taps_active).
fn tapsWritten(e: anytype) bool {
    const T = @TypeOf(e);
    const E = if (@typeInfo(T) == .pointer) @typeInfo(T).pointer.child else T;
    return if (@hasField(E, "mtp_prompt")) e.mtp_prompt.mode != .none else true;
}

/// True when nothing reads this prompt chunk's rows: its last layer stops at the state writes.
pub fn unread(e: anytype, images: bool) bool {
    return !(tapsWritten(e) and lastTapped(e)) and e.learning == null and e.mlp == null and !e.pockets.any() and !e.int8.on and !images;
}

/// The tail's first row for engine `e`'s prompt chunk of `total` rows in `nseg` parts, or null.
pub fn first(e: anytype, total: usize, nseg: usize, hidden: bool, taps_active: bool, tap_from: ?usize) ?usize {
    const last = e.layers.len - 1;
    const tapped = lastTapped(e);
    const f: Facts = .{
        .hidden = hidden,
        .hooks = e.learning != null or e.mlp != null or e.pockets.any(),
        .int8 = e.int8.on,
        .images = e.features != 0,
        .odd_group = (e.g.q_heads / e.g.kv_heads) % 2 != 0,
        .last_attention = e.layers[last].mixer == .attention,
        .last_tapped = taps_active and tapped and (tap_from == null or tap_from.? == 0),
    };
    return tapTailStart(tail.start(f.of(total, nseg)), tapped, taps_active, tap_from);
}

test "partial tap capture computes the prefix tail while preserving every needed row" {
    try std.testing.expectEqual(@as(?usize, 2047), tapTailStart(3968, true, true, 2047));
    try std.testing.expectEqual(@as(?usize, null), tapTailStart(3968, true, true, 0));
    try std.testing.expectEqual(@as(?usize, 3968), tapTailStart(3968, true, false, null));
    try std.testing.expectEqual(@as(?usize, 3968), tapTailStart(3968, false, true, null));
    try std.testing.expectEqual(@as(?usize, null), tapTailStart(null, false, false, null));
}

test "qwen3_5 prompt tail: a chunk is unread only when no reader is armed" {
    const Fake = struct {
        layers: [2]void = .{ {}, {} },
        taps: [4]u32 = undefined,
        n_taps: usize = 0,
        learning: ?*const anyopaque = null,
        mlp: ?*const anyopaque = null,
        pockets: struct {
            held: bool = false,
            pub fn any(p: @This()) bool {
                return p.held;
            }
        } = .{},
        int8: struct { on: bool = false } = .{},
    };
    var e: Fake = .{};
    try std.testing.expect(unread(&e, false));
    try std.testing.expect(!unread(&e, true));
    e.taps[0] = 0; // an earlier layer's tap still reads its own rows
    e.n_taps = 1;
    try std.testing.expect(unread(&e, false));
    e.taps[0] = 1; // the last layer's tap reads every row
    try std.testing.expect(!unread(&e, false));
    e.n_taps = 0;
    e.learning = @ptrCast(&e);
    try std.testing.expect(!unread(&e, false));
    e.learning = null;
    e.mlp = @ptrCast(&e);
    try std.testing.expect(!unread(&e, false));
    e.mlp = null;
    e.pockets.held = true;
    try std.testing.expect(!unread(&e, false));
    e.pockets.held = false;
    e.int8.on = true;
    try std.testing.expect(!unread(&e, false));
}

test "qwen3_5 prompt tail: each planted reader of the other rows refuses it" {
    const open: Facts = .{};
    try std.testing.expect(tail.start(open.of(4096, 1)) != null);
    var refused: usize = 0;
    inline for (.{ "hidden", "hooks", "int8", "images", "odd_group", "last_tapped" }) |name| {
        var f = open;
        @field(f, name) = true;
        if (tail.start(f.of(4096, 1)) == null) refused += 1 else std.debug.print("tail kept with {s}\n", .{name});
    }
    var f = open;
    f.last_attention = false;
    if (tail.start(f.of(4096, 1)) == null) refused += 1;
    try std.testing.expectEqual(@as(usize, 7), refused);
}

test "qwen3_5 prompt tail: a chunk before the MTP tap boundary writes no taps, so a tapped last layer is unread" {
    const Fake = struct {
        layers: [2]void = .{ {}, {} },
        taps: [4]u32 = .{ 1, 0, 0, 0 },
        n_taps: usize = 1,
        mtp_prompt: @import("cuda_mtp_prompt.zig").State = .{},
        learning: ?*const anyopaque = null,
        mlp: ?*const anyopaque = null,
        pockets: struct {
            pub fn any(_: @This()) bool {
                return false;
            }
        } = .{},
        int8: struct { on: bool = false } = .{},
    };
    var e: Fake = .{};
    for ([_]@import("cuda_mtp_prompt.zig").Mode{ .normal, .mtp, .mtp_direct }) |mode| {
        e.mtp_prompt.mode = mode;
        try std.testing.expect(!unread(&e, false));
    }
    const previous = e.mtp_prompt.setForPrompt(8192, 4096, false, true); // the chunk ends before the boundary
    try std.testing.expectEqual(@import("cuda_mtp_prompt.zig").Mode.none, e.mtp_prompt.mode);
    try std.testing.expect(unread(&e, false));
    try std.testing.expect(!unread(&e, true)); // images still read every row
    e.learning = @ptrCast(&e);
    try std.testing.expect(!unread(&e, false));
    e.learning = null;
    e.mtp_prompt.restoreTapConfig(previous);
    try std.testing.expect(!unread(&e, false));
    _ = e.mtp_prompt.setForPrompt(4096, 8192, false, true); // a chunk past the boundary writes its taps
    try std.testing.expect(!unread(&e, false));
}
