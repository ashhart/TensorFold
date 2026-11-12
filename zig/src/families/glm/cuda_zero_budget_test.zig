//! Price the actual family facts and exercise the DFlash2 zero-block boundary without device execution.
const std = @import("std");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const dflash = @import("dflash.zig");
const proposals = @import("dflash_proposals.zig");

test "zero grant: GLM CUDA MTP preserves tokens through a long skip" {
    var e: engine.Engine = undefined;
    e.mtp = true;
    e.drafter = null;
    var b: @import("cuda_lanes.zig").Lanes = undefined;
    b.x.e = &e;
    b.alone = false;
    b.mtp_confidence = null;
    b.costs = null;
    try lanes.shared_draft_check.check(b.facts(), false);
}

/// The fake target with DFlash2's chain cut: one stream's first block held a weak draft and zeros, its later ones sure.
const Cut = struct {
    target: lanes.fake.Fake = .{ .gpa = std.testing.allocator },
    stream: ?*lanes.Stream = null,
    cuts: u32 = 1, // the stream's blocks still to cut
    cut: bool = false, // its held block was cut
    depths: std.ArrayList(u32) = .empty, // its depth in each shared round from its first grant of none

    fn of(ptr: *anyopaque) *Cut {
        return @fieldParentPtr("target", @as(*lanes.fake.Fake, @ptrCast(@alignCast(ptr))));
    }

    fn draft(ptr: *anyopaque, reqs: []const lanes.backend.DraftRequest) !void {
        const c = of(ptr);
        for (reqs) |r| if (r.stream == c.stream) {
            if (reqs.len > 1 and (c.depths.items.len > 0 or r.stream.granted == 0)) try c.depths.append(std.testing.allocator, r.depth);
            c.cut = r.depth > 0 and c.cuts > 0;
            if (c.cut) c.cuts -= 1;
        };
        try c.target.backend().vtable.draft(ptr, reqs);
    }

    fn chances(ptr: *anyopaque, s: *lanes.Stream, out: []f64) !bool {
        const c = of(ptr);
        var chain: f64 = 1;
        for (out, 0..) |*o, j| {
            chain *= if (s != c.stream or !c.cut) 0.9 else if (j == 0) 0.05 else 0;
            o.* = chain;
        }
        return true;
    }
};

/// Three streams decoded together, the first cut once: its depths from its first grant of none.
fn cutDepths(m: lanes.Model, shared: bool) !std.ArrayList(u32) {
    const a = std.testing.allocator;
    var c: Cut = .{};
    defer c.target.deinit();
    errdefer c.depths.deinit(a);
    var table = c.target.backend().vtable.*;
    table.draft = Cut.draft;
    table.probabilities = Cut.chances;
    var model = m;
    model.allocate_shared_drafts = shared;
    var cfg = try lanes.Config.init(a, model, model.exact_width, model.drafts);
    defer cfg.deinit(a);
    cfg.timed = false;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, .{ .ptr = &c.target, .vtable = &table }, clock.clock());
    defer e.deinit();
    var streams: [3]lanes.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    for (&streams, 0..) |*s, i| {
        s.* = try lanes.Stream.init(a, .{ .id = "cut", .prompt = &.{ 7, 13, @intCast(i) }, .max_new = 96 });
        made += 1;
    }
    c.stream = &streams[0];
    for (&streams) |*s| try e.addStream(s);
    for (0..1000) |_| if (e.activeCount() > 0) try e.step();
    try std.testing.expectEqual(@as(usize, 0), e.activeCount());
    return c.depths;
}

test "zero grant: a GLM DFlash2 stream cut to none drafts again within a few rounds while its peers decode" {
    var e: engine.Engine = undefined;
    e.mtp = true;
    e.drafter = @ptrFromInt(16);
    var b: @import("cuda_lanes.zig").Lanes = undefined;
    b.x.e = &e;
    b.alone = false;
    b.mtp_confidence = null;
    b.costs = null;
    const m = b.facts();
    var fixed = try cutDepths(m, m.allocate_shared_drafts);
    defer fixed.deinit(std.testing.allocator);
    try std.testing.expect(fixed.items.len >= 3 and std.mem.max(u32, fixed.items[0..3]) > 0);
    // priced from its cut chances, which only a drafted block refreshes, it stays at none while its peers decode
    var priced = try cutDepths(m, true);
    defer priced.deinit(std.testing.allocator);
    try std.testing.expect(priced.items.len >= 32 and std.mem.allEqual(u32, priced.items[0..32], 0));
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
    b.costs = null;
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
