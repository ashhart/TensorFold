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
test "all five round launches are present and explicitly unmeasured" {
    var seen: usize = 0;
    for (core.contract.registry.all_cuda) |r| if (std.mem.eql(u8, r.launch.image, "tree_round")) {
        seen += 1;
        for (r.usage) |u| try std.testing.expectEqual(@as(u32, 0), u.regs);
        try std.testing.expectEqual(core.contract.limits.sms.len, try core.contract.registry.checkCuda(std.testing.allocator, &.{r}, true));
    };
    try std.testing.expectEqual(@as(usize, 5), seen);
}
