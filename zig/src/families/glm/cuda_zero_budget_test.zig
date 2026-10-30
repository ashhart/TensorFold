//! Price the actual family facts and exercise the DFlash2 zero-block boundary without device execution.
const std = @import("std");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const dflash = @import("dflash.zig");
const proposals = @import("dflash_proposals.zig");

test "zero grant: GLM CUDA MTP and DFlash2 preserve tokens through a long skip" {
    var e: engine.Engine = undefined;
    e.mtp = true;
    for ([_]bool{ false, true }) |block| {
        e.drafter = if (block) @ptrFromInt(16) else null;
        var b: @import("cuda_lanes.zig").Lanes = undefined;
        b.x.e = &e;
        b.alone = false;
        b.mtp_confidence = null;
        try lanes.shared_draft_check.check(b.facts(), false);
    }
}

test "zero grant: GLM DFlash2 skips the actual block pass and transmits a stable proposal mask" {
    var e: engine.Engine = undefined;
    var dr: dflash.Drafter = undefined;
    dr.gpa = std.testing.allocator;
    dr.prof = null;
    dr.verifies = 0;
    dr.calls = 0;
    dr.slots = .empty;
    defer dr.slots.deinit(std.testing.allocator);
    var ring: @import("cuda").DeviceBuffer = undefined;
    ring.ptr = 16;
    try dr.slots.append(std.testing.allocator, .{ .ring = ring });
    const zero: dflash.Req = .{ .slot = 0, .pos = 512, .n = 0, .anchor = 11, .propose = false };
    for (0..180) |_| try dr.draft(&e, &.{zero});
    try std.testing.expectEqual(@as(u64, 0), dr.calls);
    var useful = zero;
    useful.propose = true;
    useful.slot = 7;
    const reqs = [_]dflash.Req{ zero, useful, zero, useful };
    const bits = proposals.mask(&reqs);
    try std.testing.expectEqual(@as(u32, 10), bits);
    var decoded = reqs;
    for (&decoded, 0..) |*q, i| q.propose = proposals.enabled(bits, i);
    var out: [4]dflash.Req = undefined;
    const selected = proposals.active(&decoded, &out);
    try std.testing.expectEqual(@as(usize, 2), selected.len);
    for (selected) |q| {
        try std.testing.expectEqual(@as(u32, 7), q.slot);
        try std.testing.expectEqual(useful.anchor, q.anchor);
        try std.testing.expectEqual(useful.pos, q.pos);
    }
}
