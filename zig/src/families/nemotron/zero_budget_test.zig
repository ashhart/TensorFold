//! The real Metal adapter policy with a counted host target and no device allocations.
const lanes = @import("lanes");
const Metal = @import("backend.zig").Metal;
const mtp = @import("mtp.zig");

test "zero grant: Nemotron Metal preserves later drafts through a long shared skip" {
    var b: Metal = undefined;
    b.head = @as(mtp.Head, undefined);
    b.o = .{ .capacity = 512, .streams = 16 };
    b.costs = .{};
    try lanes.shared_draft_check.check(b.facts(), false);
}
