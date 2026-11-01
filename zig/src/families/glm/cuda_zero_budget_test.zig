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

test "shared head costs: GLM CUDA MTP is unpriced and DFlash2 already uses node budgets" {
    var e: engine.Engine = undefined;
    e.mtp = true;
    var b: @import("cuda_lanes.zig").Lanes = undefined;
    b.x.e = &e;
    b.alone = false;
    b.mtp_confidence = null;
    for ([_]bool{ false, true }) |block| {
        e.drafter = if (block) @ptrFromInt(16) else null;
        const m = b.facts();
        try std.testing.expect(!m.limit_shared_drafts);
        if (!block) {
            try std.testing.expectEqual(@as(f64, 0), m.mtp_step_ms);
            try std.testing.expect(!m.draft_probabilities);
            continue;
        }
        try std.testing.expect(m.mtp_step_ms > 0 and m.draft_probabilities);
        var cfg = try lanes.Config.init(std.testing.allocator, m, m.exact_width, m.drafts);
        defer cfg.deinit(std.testing.allocator);
        var fake: lanes.fake.Fake = .{ .gpa = std.testing.allocator };
        defer fake.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(std.testing.allocator, &cfg, fake.backend(), clock.clock());
        defer core.deinit();
        var stream = try lanes.Stream.init(std.testing.allocator, .{ .id = "node", .prompt = &.{1}, .max_new = 64 });
        defer stream.deinit(std.testing.allocator);
        stream.granted = 1;
        var who = [_]lanes.depth.Who{lanes.windows.who(&stream)};
        var old: [1]i64 = undefined;
        var expensive: [1]i64 = undefined;
        try core.rule.budgets(&who, lanes.windows.probe(&core), false, &old);
        core.rule.mtp_step_ms = 1000;
        try core.rule.budgets(&who, lanes.windows.probe(&core), false, &expensive);
        try std.testing.expectEqual(@as(i64, 3), old[0]);
        try std.testing.expectEqualSlices(i64, &old, &expensive);
    }
}
