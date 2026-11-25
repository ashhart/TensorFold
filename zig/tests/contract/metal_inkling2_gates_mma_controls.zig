//! Count-checked mutations of the simdgroup-matrix loop gates: each adds one chain in another order.
const std = @import("std");
const src = @import("inkling2_metal");
const g = @import("contract").gates;

pub const Mutation = enum {
    none,
    chunks,
    ranks,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .chunks => "control: metal inkling2 mma tf_inkling2_loop_gates_mma zeta columns in reverse",
            .ranks => "control: metal inkling2 mma tf_inkling2_loop_gates_mma ranks added in reverse",
        };
    }

    pub fn gate(m: Mutation) ?g.Gate {
        return if (m == .none) null else .shapes;
    }

    const Edit = struct { from: []const u8, to: []const u8 };
    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .chunks => .{
                .from = "for (uint chunk = 0; chunk < 2 * INKLING_LOOP_H / INKLING_LOOP_MMA_STAGE; ++chunk) {",
                .to = "for (uint chunk = 2 * INKLING_LOOP_H / INKLING_LOOP_MMA_STAGE; chunk-- != 0;) {",
            },
            .ranks => .{
                .from = "for (uint o = 0; o < INKLING_LOOP_RANK / 8u; ++o) {",
                .to = "for (uint o = INKLING_LOOP_RANK / 8u; o-- != 0u;) {",
            },
        };
    }
};

pub fn apply(a: std.mem.Allocator, m: Mutation) ![]u8 {
    if (m == .none) return a.dupe(u8, src.loop_mma);
    const change = m.edit();
    if (std.mem.count(u8, src.loop_mma, change.from) != 1) return error.ControlEditCount;
    return std.mem.replaceOwned(u8, a, src.loop_mma, change.from, change.to);
}

test "every simdgroup-matrix gates control edits exactly its planted chain" {
    const a = std.testing.allocator;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const edited = try apply(a, m);
        defer a.free(edited);
        try std.testing.expect(std.mem.indexOf(u8, edited, m.edit().to) != null);
        try std.testing.expect(std.mem.indexOf(u8, edited, m.edit().from) == null);
        try std.testing.expectEqual(g.Gate.shapes, m.gate().?);
        try std.testing.expect(std.mem.startsWith(u8, m.name(), "control: metal inkling2 mma tf_inkling2_loop_gates_mma"));
    }
    const plain = try apply(a, .none);
    defer a.free(plain);
    try std.testing.expectEqualStrings(src.loop_mma, plain);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src.loop_mma, "void tf_inkling2_loop_gates_mma("));
}
