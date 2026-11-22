//! Count-checked mutations of Mini DSpark's two chain kernels: each has a genuine control.
const std = @import("std");
const src = @import("inkling2_metal");
const g = @import("contract").gates;

pub const Mutation = enum {
    none,
    step_token,
    pick_tie,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .step_token => "control: metal inkling2 dspark chain_step tf_ink2_dspark_chain_step next token",
            .pick_tie => "control: metal inkling2 dspark chain_pick tf_ink2_dspark_chain_pick later tie",
        };
    }

    pub fn gate(m: Mutation) ?g.Gate {
        return if (m == .none) null else .shapes;
    }

    const Edit = struct { from: []const u8, to: []const u8 };
    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .step_token => .{ .from = "lambda, held[a.token], i);", .to = "lambda, held[a.token + 1u], i);" },
            .pick_tie => .{ .from = "if (ov > best || (ov == best && oi < at)) { best = ov; at = oi; }", .to = "if (ov > best || (ov == best && oi > at)) { best = ov; at = oi; }" },
        };
    }
};

pub fn apply(a: std.mem.Allocator, m: Mutation) ![]u8 {
    if (m == .none) return a.dupe(u8, src.inkling2_dspark);
    const change = m.edit();
    if (std.mem.count(u8, src.inkling2_dspark, change.from) != 1) return error.ControlEditCount;
    return std.mem.replaceOwned(u8, a, src.inkling2_dspark, change.from, change.to);
}

test "each chain control edits exactly its planted line" {
    const a = std.testing.allocator;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const edited = try apply(a, m);
        defer a.free(edited);
        try std.testing.expect(std.mem.indexOf(u8, edited, m.edit().to) != null);
        try std.testing.expectEqual(g.Gate.shapes, m.gate().?);
        try std.testing.expect(std.mem.startsWith(u8, m.name(), "control: metal inkling2 dspark chain_"));
    }
    inline for ([_][]const u8{ "tf_ink2_dspark_chain_step", "tf_ink2_dspark_chain_pick" }) |name|
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src.inkling2_dspark, name ++ "("));
}
