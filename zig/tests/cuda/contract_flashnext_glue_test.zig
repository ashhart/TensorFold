//! The flashnext prompt contract's host-visible declarations: the served row range and the state alias.
const std = @import("std");
const core = @import("core");
const flash = @import("flashnext");
const glue = @import("contract_flashnext_glue.zig");

const contract = core.contract;
const c = contract.spec;
const parts = flash.batch_state;

test "the flashnext prompt specs serve the admitted row range with the dispatch and tile seams" {
    for ([_]glue.Kind{ .chain, .fused }) |kind| {
        const s = glue.spec(kind);
        var served: [3]bool = @splat(false);
        for (s.served) |x| for ([_]usize{ 64, parts.default_chunk, parts.max_chunk }, 0..) |want, i| {
            if (x.rows == want) served[i] = true;
        };
        for (served) |yes| try std.testing.expect(yes);
        const list = try c.shapes(std.testing.allocator, s);
        defer std.testing.allocator.free(list);
        var seams: [4]bool = @splat(false);
        for (list) |x| for ([_]usize{ 1, 15, 16, 17 }, 0..) |want, i| {
            if (x.rows == want) seams[i] = true;
        };
        for (seams) |yes| try std.testing.expect(yes);
    }
}

test "the fused block declares the state alias and binds one buffer for it" {
    try std.testing.expect(glue.spec(.fused).in_place);
    try std.testing.expect(!glue.spec(.chain).in_place); // the chain's kernel marks its state input __restrict__
    if (@hasDecl(glue, "fusedArgs")) {
        var ins: [7]u64 = .{ 1, 2, 3, 4, 5, 6, 7 };
        var o: [4]u64 = .{ 8, 9, 10, 11 };
        var a = glue.fusedArgs(&ins, &o, 64, 12);
        try std.testing.expectEqual(a.integer(5), a.integer(6)); // the state in and the state out are one buffer
        try std.testing.expectEqual(@as(?u64, 10), a.integer(5));
        try std.testing.expectEqual(@as(?u64, 64), a.integer(7));
    } else return error.MissingFusedArgs;
}
