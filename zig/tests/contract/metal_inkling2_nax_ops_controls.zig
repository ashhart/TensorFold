//! Count-checked mutations for Inkling 2 Mini's M5 NAX route_scores, loop_blend, and rel_table kernels.
const std = @import("std");
const src = @import("inkling2_metal");
const g = @import("contract").gates;

pub const Mutation = enum {
    none,
    route_16,
    route_64,
    loop_blend,
    rel_table,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .route_16 => "control: metal inkling2 dense_units tf_ink2_route_scores_units_16 k-step",
            .route_64 => "control: metal inkling2 dense_units tf_ink2_route_scores_units_64 k-step",
            .loop_blend => "control: metal inkling2 dense_units tf_ink2_loop_blend_units drop_right",
            .rel_table => "control: metal inkling2 dense_units tf_ink2_rel_table_units bad_col",
        };
    }

    pub fn gate(m: Mutation) ?g.Gate {
        return if (m == .none) null else .shapes;
    }

    const Edit = struct { from: []const u8, to: []const u8 };
    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .route_16 => .{ .from = "ink2_route_core<1u, false>(", .to = "ink2_route_core<1u, true>(" },
            .route_64 => .{ .from = "ink2_route_core<4u, false>(", .to = "ink2_route_core<4u, true>(" },
            .loop_blend => .{ .from = "const float right = g * mv[j];", .to = "const float right = 0.0f;" },
            .rel_table => .{ .from = "const uint col0 = col_base + 32u * t;", .to = "const uint col0 = 32u * t;" },
        };
    }
};

pub fn apply(a: std.mem.Allocator, m: Mutation) ![]u8 {
    if (m == .none) return a.dupe(u8, src.dense_units);
    const change = m.edit();
    if (std.mem.count(u8, src.dense_units, change.from) != 1) return error.ControlEditCount;
    return std.mem.replaceOwned(u8, a, src.dense_units, change.from, change.to);
}

test "every nax_ops control edits exactly its planted arithmetic" {
    const a = std.testing.allocator;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const edited = try apply(a, m);
        defer a.free(edited);
        try std.testing.expect(std.mem.indexOf(u8, edited, m.edit().to) != null);
        try std.testing.expectEqual(g.Gate.shapes, m.gate().?);
        try std.testing.expect(std.mem.startsWith(u8, m.name(), "control: metal inkling2 dense_units "));
    }
    const plain = try apply(a, .none);
    defer a.free(plain);
    try std.testing.expectEqualStrings(src.dense_units, plain);
}
