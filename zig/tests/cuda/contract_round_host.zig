//! The actual round adapter uses the CUDA ABI and refuses unmeasured launch qualification.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const fixture = @import("contract_tree_round.zig");
export fn round_contract_hooks(r: *fixture.Round) *const core.contract.gates.VTable {
    return r.kernel().vt;
}
test "round contract uses CUDA result stride after the common oracle widened" {
    const Stored = @typeInfo(@FieldType(fixture.Round, "results")).pointer.child;
    try std.testing.expectEqual(@sizeOf(cuda.tree_round.Result), @sizeOf(Stored));
    try std.testing.expect(@sizeOf(core.tree_round.Result) > @sizeOf(Stored));
    var r: fixture.Round = .{ .rig = undefined, .k = undefined, .which = .match, .shape = .{ .rows = 3, .n = 1, .k = 16 } };
    try std.testing.expectEqual(3 * @sizeOf(cuda.tree_round.Result), r.count());
}
test "five common round launches and Gemma picks have measured 13.0 resources" {
    const pending = [_][]const u8{ "tf_round_bf16_argmax", "tf_tree_round_match", "tf_round_keep_rows", "tf_round_gather_rows", "tf_round_keep_head" };
    var seen: [pending.len]bool = @splat(false);
    var measured: usize = 0;
    for (core.contract.registry.all_cuda) |r| if (std.mem.eql(u8, r.launch.image, "tree_round")) {
        if (std.mem.eql(u8, r.launch.symbol, "tf_round_pick_ids")) {
            measured += 1;
            const regs = [_]u32{ 10, 10, 10, 12, 12 };
            for (r.usage, regs) |u, n| {
                try std.testing.expectEqual(n, u.regs);
                try std.testing.expectEqual(@as(u32, 0), u.static);
                try std.testing.expectEqual(@as(u32, 256), u.bound);
            }
            try std.testing.expectEqual(@as(usize, 0), try core.contract.registry.checkCuda(std.testing.allocator, &.{r}, true));
            continue;
        }
        const i = for (pending, 0..) |symbol, i| {
            if (std.mem.eql(u8, r.launch.symbol, symbol)) break i;
        } else return error.UnexpectedRoundLaunch;
        try std.testing.expect(!seen[i]);
        seen[i] = true;
        for (r.usage) |u| try std.testing.expect(u.regs > 0);
        try std.testing.expectEqual(@as(usize, 0), try core.contract.registry.checkCuda(std.testing.allocator, &.{r}, true));
        try std.testing.expectEqualDeep(core.contract.registry.none, core.contract.registry.usageFor(r, .{ .major = 13, .minor = 2 }));
    };
    for (seen) |present| try std.testing.expect(present);
    try std.testing.expectEqual(@as(usize, 1), measured);
}
