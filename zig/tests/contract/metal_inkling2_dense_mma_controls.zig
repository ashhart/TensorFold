//! Count-checked mutations of the simdgroup-matrix projection: each moves one rounding in the sums.
const std = @import("std");
const src = @import("inkling2_metal");
const g = @import("contract").gates;

pub const Mutation = enum {
    none,
    halves,
    groups,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .halves => "control: metal inkling2 mma tf_ink2_dense_mma halves added one by one",
            .groups => "control: metal inkling2 mma tf_ink2_dense_mma groups added in a line",
        };
    }

    pub fn gate(m: Mutation) ?g.Gate {
        return if (m == .none) null else .shapes;
    }

    const Edit = struct { from: []const u8, to: []const u8 };
    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .halves => .{
                .from = "acc[i][j][c].thread_elements()[0] += d0.thread_elements()[0] + d1.thread_elements()[0];",
                .to = "acc[i][j][c].thread_elements()[0] = (acc[i][j][c].thread_elements()[0] + d0.thread_elements()[0]) + d1.thread_elements()[0];",
            },
            .groups => .{
                .from = "(part[0][i][j][m][n] + part[1][i][j][m][n]) + (part[2][i][j][m][n] + part[3][i][j][m][n])",
                .to = "((part[0][i][j][m][n] + part[1][i][j][m][n]) + part[2][i][j][m][n]) + part[3][i][j][m][n]",
            },
        };
    }
};

pub fn apply(a: std.mem.Allocator, m: Mutation) ![]u8 {
    if (m == .none) return a.dupe(u8, src.dense_mma);
    const change = m.edit();
    if (std.mem.count(u8, src.dense_mma, change.from) != 1) return error.ControlEditCount;
    return std.mem.replaceOwned(u8, a, src.dense_mma, change.from, change.to);
}

test "every simdgroup-matrix projection control edits exactly its planted arithmetic" {
    const a = std.testing.allocator;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const edited = try apply(a, m);
        defer a.free(edited);
        try std.testing.expect(std.mem.indexOf(u8, edited, m.edit().to) != null);
        try std.testing.expectEqual(g.Gate.shapes, m.gate().?);
        try std.testing.expect(std.mem.startsWith(u8, m.name(), "control: metal inkling2 mma tf_ink2_dense_mma"));
    }
    const plain = try apply(a, .none);
    defer a.free(plain);
    try std.testing.expectEqualStrings(src.dense_mma, plain);
}
