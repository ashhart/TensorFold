//! Qwen 3.5's facts for core's prompt tail: what reads the last layer's other rows, and when that layer fits.
const std = @import("std");
const core = @import("core");
const tail = core.prompt_tail;

pub const rows = tail.rows;

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

/// The tail's first row for engine `e`'s prompt chunk of `total` rows in `nseg` parts, or null.
pub fn first(e: anytype, total: usize, nseg: usize, hidden: bool) ?usize {
    const last = e.layers.len - 1;
    var tapped = false;
    for (e.taps[0..e.n_taps]) |t| tapped = tapped or t == last;
    const f: Facts = .{
        .hidden = hidden,
        .hooks = e.learning != null or e.mlp != null or e.pockets.any(),
        .int8 = e.int8.on,
        .images = e.features != 0,
        .odd_group = (e.g.q_heads / e.g.kv_heads) % 2 != 0,
        .last_attention = e.layers[last].mixer == .attention,
        .last_tapped = tapped,
    };
    return tail.start(f.of(total, nseg));
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
