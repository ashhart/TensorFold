//! A prompt pass that keeps no row's logits: its last layer writes keys and values only, when nothing else reads it.
const std = @import("std");

/// What reads the last layer's rows besides the head, and whether the pass keeps a row at all.
pub const Facts = struct {
    /// The pass keeps no row's logits (a prompt chunk before the last).
    keeps_none: bool,
    /// A drafter's or MTP head's tap on the last layer's rows.
    last_tapped: bool = false,
    /// A tensor-parallel pair: each layer's seams are counted on both ranks.
    pair: bool = false,
    /// The Neural Engine's MLP share steps through every layer.
    ane: bool = false,
    last_attention: bool = true,
};

pub fn keysOnly(f: Facts) bool {
    return f.keeps_none and !f.last_tapped and !f.pair and !f.ane and f.last_attention;
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
