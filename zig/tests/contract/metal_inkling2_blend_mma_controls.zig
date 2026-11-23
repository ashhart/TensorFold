//! Count-checked mutation of the simdgroup-matrix loop blend: the ranks added in another order.
const std = @import("std");
const src = @import("inkling2_metal");
const g = @import("contract").gates;

pub const Mutation = enum {
    none,
    ranks,

    pub fn name(m: Mutation) []const u8 {
        return if (m == .none) "" else "control: metal inkling2 mma tf_ink2_loop_blend_mma ranks added in reverse";
    }

    pub fn gate(m: Mutation) ?g.Gate {
        return if (m == .none) null else .shapes;
    }

    const from = "for (uint o = 0u; o < INKLING_LOOP_RANK / 8u; ++o)";
    const to = "for (uint o = INKLING_LOOP_RANK / 8u; o-- != 0u;)";
};

pub fn apply(a: std.mem.Allocator, m: Mutation) ![]u8 {
    if (m == .none) return a.dupe(u8, src.loop_mma);
    if (std.mem.count(u8, src.loop_mma, Mutation.from) != 1) return error.ControlEditCount;
    return std.mem.replaceOwned(u8, a, src.loop_mma, Mutation.from, Mutation.to);
}

test "the blend control edits exactly its rank loop" {
    const a = std.testing.allocator;
    const edited = try apply(a, .ranks);
    defer a.free(edited);
    try std.testing.expect(std.mem.indexOf(u8, edited, Mutation.to) != null);
    try std.testing.expect(std.mem.indexOf(u8, edited, Mutation.from) == null);
    try std.testing.expectEqual(g.Gate.shapes, Mutation.ranks.gate().?);
    const plain = try apply(a, .none);
    defer a.free(plain);
    try std.testing.expectEqualStrings(src.loop_mma, plain);
    try std.testing.expect(std.mem.indexOf(u8, src.loop_mma, "tf_ink2_loop_blend_mma") != null);
}
