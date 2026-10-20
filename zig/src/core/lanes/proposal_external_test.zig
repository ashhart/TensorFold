const std = @import("std");
const fake = @import("fake.zig");
const sm = @import("stream.zig");
const dr = @import("drafter.zig");
const Drafted = @import("drafted.zig").Drafted;
const FakeDrafter = @import("drafted_test.zig").FakeDrafter;
const gpa = std.testing.allocator;

const ProposalFixture = struct {
    pub fn isLegacy(_: *const ProposalFixture) bool {
        return false;
    }
    fingerprint: [32]u8 = @splat(1),
    pub fn ids(_: *const ProposalFixture) []const u32 {
        return &.{ 0, 3, 7 };
    }
    pub fn global(x: *const ProposalFixture, at: usize) ?u32 {
        return if (at < x.ids().len) x.ids()[at] else null;
    }
    pub fn column(x: *const ProposalFixture, id: u32) ?usize {
        return std.mem.indexOfScalar(u32, x.ids(), id);
    }
    pub fn matches(_: *const ProposalFixture, actual: anytype) bool {
        return actual.vocab == fake.vocab;
    }
};
fn plannedProposalHold(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
    for (items) |it| {
        try std.testing.expect(it.plan != null);
        try std.testing.expectEqual(@import("proposal_head.zig").Kind.subset, it.plan.?.kind);
    }
    const x: *FakeDrafter = @ptrCast(@alignCast(ptr));
    return x.drafter().hold(items);
}
fn clearProposalHold(ptr: *anyopaque, s: *sm.Stream) void {
    const x: *FakeDrafter = @ptrCast(@alignCast(ptr));
    if (x.lanes.getPtr(s)) |l| l.held.clearRetainingCapacity();
}
test "proposal router: external zero-depth fallback absorbs omitted target rows and resumes after sixteen covered commits" {
    const a: ProposalFixture = .{};
    const service = try @import("proposal_head.zig").Service.calibrated(gpa, &a, .{ .vocab = fake.vocab }, .{ .subset = true, .full = false });
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var inner: FakeDrafter = .{};
    defer inner.deinit();
    var d = inner.drafter();
    var vt = d.vtable.*;
    vt.hold_planned = plannedProposalHold;
    vt.clear_proposals = clearProposalHold;
    d.vtable = &vt;
    d.proposal_head = &service;
    var wrapper = try Drafted.init(gpa, target.backend(), d);
    defer wrapper.deinit();
    const b = wrapper.backend();
    var s = try sm.Stream.init(gpa, .{ .id = "external-proposal", .prompt = &(@as([16]u32, @splat(3))), .max_new = 1000, .eos = &.{} });
    defer s.deinit(gpa);
    try b.prefill(&s);
    defer b.release(&s);
    try b.draftIn(gpa, &.{.{ .stream = &s, .follow = &.{}, .first = .{ .value = 3 }, .rows = null, .start = 16, .position = 17, .depth = 2 }});
    _ = try s.commit(gpa, &.{3}, &.{});
    try std.testing.expectEqual(@as(u32, 1), inner.holds);
    try target.lanes.getPtr(&s).?.history.append(gpa, 3);
    _ = try s.commit(gpa, &.{9}, &.{});
    try b.draftIn(gpa, &.{.{ .stream = &s, .follow = &.{9}, .rows = &.{0}, .start = 16, .position = 18, .depth = 2 }});
    const paused_absorbs = inner.absorbs;
    try std.testing.expectEqual(@as(u32, 1), inner.holds);
    try std.testing.expectEqual(@as(usize, 0), inner.lanes.getPtr(&s).?.held.items.len);
    for (0..16) |j| {
        try target.lanes.getPtr(&s).?.history.append(gpa, if (j == 0) 9 else 3);
        _ = try s.commit(gpa, &.{3}, &.{});
        try b.draftIn(gpa, &.{.{ .stream = &s, .follow = &.{3}, .rows = &.{0}, .start = 17 + j, .position = 19 + j, .depth = 2 }});
        try std.testing.expectEqual(@as(u32, if (j == 15) 2 else 1), inner.holds);
    }
    try std.testing.expectEqual(paused_absorbs + 16, inner.absorbs);
    try std.testing.expect(s.drafts);
    try std.testing.expectEqualSlices(u32, target.lanes.getPtr(&s).?.history.items, inner.lanes.getPtr(&s).?.history.items);
    inner.fail = .absorb;
    try std.testing.expectError(error.Injected, b.draftIn(gpa, &.{.{ .stream = &s, .follow = &.{3}, .rows = &.{0}, .start = 32, .position = 34, .depth = 2 }}));
    try std.testing.expectEqual(@as(usize, 0), inner.lanes.getPtr(&s).?.held.items.len);
}

fn decodedProposal(drafts: bool, sampling: ?@import("sampling.zig").Sampling) ![]u32 {
    const a: ProposalFixture = .{};
    const service = try @import("proposal_head.zig").Service.calibrated(gpa, &a, .{ .vocab = fake.vocab }, .{ .subset = true, .full = false });
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var inner: FakeDrafter = .{};
    defer inner.deinit();
    var d = inner.drafter();
    var vt = d.vtable.*;
    vt.hold_planned = plannedProposalHold;
    vt.clear_proposals = clearProposalHold;
    d.vtable = &vt;
    d.proposal_head = &service;
    var wrapper = try Drafted.init(gpa, target.backend(), d);
    defer wrapper.deinit();
    var cfg = try @import("config.zig").Config.init(gpa, wrapper.facts(.{ .exact_width = 16, .hidden_rows = true }), 16, 15);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var e = @import("engine.zig").Engine.init(gpa, &cfg, wrapper.backend(), clock.clock());
    defer e.deinit();
    var s = try sm.Stream.init(gpa, .{ .id = "equality", .prompt = &(@as([16]u32, @splat(3))), .max_new = 40, .eos = &.{}, .drafts = drafts, .sampling = sampling });
    defer s.deinit(gpa);
    try e.addStream(&s);
    while (!s.finished) try e.step();
    if (drafts) try std.testing.expect(inner.live_rows > 0);
    return gpa.dupe(u32, s.emitted());
}
test "proposal router: absorb-only omitted target tokens equal plain greedy and keyed sampled replies" {
    for ([_]?@import("sampling.zig").Sampling{ null, .{ .seed = 1234 } }) |sampling| {
        const plain = try decodedProposal(false, sampling);
        defer gpa.free(plain);
        const routed = try decodedProposal(true, sampling);
        defer gpa.free(routed);
        try std.testing.expectEqualSlices(u32, plain, routed);
        try std.testing.expectEqual(@as(usize, 40), routed.len);
        try std.testing.expect(std.mem.indexOfScalar(u32, routed, 9) != null or std.mem.indexOfScalar(u32, routed, 4) != null or routed[0] != 3);
    }
}
