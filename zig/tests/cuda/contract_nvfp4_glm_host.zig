//! The actual factored gate-up contract includes GLM's TP shards and wide verify rows.
const std = @import("std");
const spec = @import("contract_nvfp4_fact.zig").spec;

test "GLM factored contract: K4096 gate-up at TP1, TP2 and TP4 includes 32 rows" {
    const s = spec(.up);
    try std.testing.expectEqualStrings("nvfp4_expert_kernel<2, 2, 4>", s.replaces.?);
    try std.testing.expectEqual(.f32, s.accum);
    for ([_]usize{ 2048, 1024, 512 }) |n| for ([_]usize{ 1, 16, 32 }) |rows| {
        var found = false;
        for (s.served) |shape| if (shape.k == 4096 and shape.n == n and shape.rows == rows) {
            found = true;
        };
        try std.testing.expect(found);
    };
}
