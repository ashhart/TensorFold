//! Count-checked mutations of the register-tiled bf16 projection: each shifts one rounding or narrows accumulation.
const std = @import("std");
const src = @import("inkling2_metal");
const g = @import("contract").gates;

pub const Mutation = enum {
    none,
    halves,
    second_col,
    bf16_acc,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .halves => "control: metal inkling2 dense tf_ink2_dense_tiled halves added one by one",
            .second_col => "control: metal inkling2 dense tf_ink2_dense_tiled second column halves added one by one",
            .bf16_acc => "control: metal inkling2 dense tf_ink2_dense_tiled bf16 running sum",
        };
    }

    pub fn gate(m: Mutation) ?g.Gate {
        return switch (m) {
            .none => null,
            .halves, .second_col => .shapes,
            .bf16_acc => .sums,
        };
    }

    const Edit = struct { from: []const u8, to: []const u8 };
    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .halves => .{
                .from = "acc0[r] += dot(w0_lo, x0) + dot(w0_hi, x1);",
                .to = "acc0[r] = (acc0[r] + dot(w0_lo, x0)) + dot(w0_hi, x1);",
            },
            .second_col => .{
                .from = "acc1[r] += dot(w1_lo, x0) + dot(w1_hi, x1);",
                .to = "acc1[r] = (acc1[r] + dot(w1_lo, x0)) + dot(w1_hi, x1);",
            },
            .bf16_acc => .{
                .from = "acc0[r] += dot(w0_lo, x0) + dot(w0_hi, x1);",
                .to = "acc0[r] = float(bfloat(acc0[r] + (dot(w0_lo, x0) + dot(w0_hi, x1))));",
            },
        };
    }
};

pub fn apply(a: std.mem.Allocator, m: Mutation) ![]u8 {
    if (m == .none) return a.dupe(u8, src.dense_tiled);
    const change = m.edit();
    if (std.mem.count(u8, src.dense_tiled, change.from) != 1) return error.ControlEditCount;
    return std.mem.replaceOwned(u8, a, src.dense_tiled, change.from, change.to);
}

test "every register-tiled projection control edits exactly its planted arithmetic" {
    const a = std.testing.allocator;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const edited = try apply(a, m);
        defer a.free(edited);
        try std.testing.expect(std.mem.indexOf(u8, edited, m.edit().to) != null);
        try std.testing.expect(m.gate() != null);
        try std.testing.expect(std.mem.startsWith(u8, m.name(), "control: metal inkling2 dense tf_ink2_dense_tiled"));
    }
    const plain = try apply(a, .none);
    defer a.free(plain);
    try std.testing.expectEqualStrings(src.dense_tiled, plain);
}
