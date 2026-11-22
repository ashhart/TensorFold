//! Count-checked mutations of Mini DSpark's tensor-unit attention: each of its two kernels has a genuine control.
const std = @import("std");
const src = @import("inkling2_metal");
const g = @import("contract").gates;
const Kind = @import("metal_inkling2_dspark_units.zig").Kind;

pub const Mutation = enum {
    none,
    attend_log2,
    merge_keep,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .attend_log2 => "control: metal inkling2 dspark_units tf_ink2_dspark_attn_units context scores without the log2 factor",
            .merge_keep => "control: metal inkling2 dspark_units tf_ink2_dspark_attn_merge earlier mass kept unscaled",
        };
    }

    pub fn gate(m: Mutation) ?g.Gate {
        return if (m == .none) null else .shapes;
    }

    const Edit = struct { from: []const u8, to: []const u8 };
    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .attend_log2 => .{ .from = "score[f][e] = live ? score[f][e] * DSU_SCALE * DSU_LOG2E : -INFINITY;", .to = "score[f][e] = live ? score[f][e] * DSU_SCALE : -INFINITY;" },
            .merge_keep => .{ .from = "ll = ll * x + ml.y * y;", .to = "ll = ll + ml.y * y;" },
        };
    }
};

pub fn of(kind: Kind) Mutation {
    return switch (kind) {
        .attend => .attend_log2,
        .merge => .merge_keep,
    };
}

pub fn apply(a: std.mem.Allocator, m: Mutation) ![]u8 {
    if (m == .none) return a.dupe(u8, src.inkling2_dspark_units);
    const change = m.edit();
    if (std.mem.count(u8, src.inkling2_dspark_units, change.from) != 1) return error.ControlEditCount;
    return std.mem.replaceOwned(u8, a, src.inkling2_dspark_units, change.from, change.to);
}

test "each tensor-unit attention control edits exactly its planted line" {
    const a = std.testing.allocator;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const edited = try apply(a, m);
        defer a.free(edited);
        try std.testing.expect(std.mem.indexOf(u8, edited, m.edit().to) != null);
        try std.testing.expectEqual(g.Gate.shapes, m.gate().?);
        try std.testing.expect(std.mem.startsWith(u8, m.name(), "control: metal inkling2 dspark_units "));
    }
    inline for ([_][]const u8{ "tf_ink2_dspark_attn_units", "tf_ink2_dspark_attn_merge" }) |name|
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src.inkling2_dspark_units, name ++ "("));
}
