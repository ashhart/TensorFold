//! Shared proposal routing against real artifact admission, lane requests and external absorption.
const std = @import("std");
const lanes = @import("lanes");
const vocab = @import("core").proposal_vocab;
const h = lanes.proposal_head;
const be = lanes.backend;
const dr = lanes.drafter;
const Stream = lanes.Stream;
const t = std.testing;
const gpa = t.allocator;
const fixture = @embedFile("src/core/testdata/proposal-vocab.json");
fn artifact() !vocab.Artifact {
    const d = try std.json.parseFromSlice(vocab.Document, gpa, fixture, .{});
    defer d.deinit();
    return vocab.load(gpa, fixture, d.value.identity, &.{0});
}
fn stream() !Stream {
    return Stream.init(gpa, .{ .id = "proposal", .prompt = &(@as([16]u32, @splat(3))), .max_new = 1000, .eos = &.{} });
}
fn request(s: *Stream) be.DraftRequest {
    return .{ .stream = s, .follow = &.{3}, .rows = &.{0}, .start = 15, .position = 17, .depth = 2 };
}
const Spy = struct {
    legacy_calls: u32 = 0,
    routed_calls: u32 = 0,
    clears: u32 = 0,
    held: bool = false,
    fail: bool = false,
    cancel_after: ?*bool = null,
    plans: [8]h.Kind = @splat(.paused),
    depths: [8]u32 = @splat(0),
    ids: [4]u32 = @splat(0),
    fn self(p: *anyopaque) *Spy {
        return @ptrCast(@alignCast(p));
    }
    fn backend(x: *Spy, service: ?*const h.Service) be.Backend {
        return .{ .ptr = x, .proposal_head = service, .vtable = &.{ .prefill = undefined, .first = undefined, .queue = undefined, .read = undefined, .verify = undefined, .keep = undefined, .release = undefined, .draft = legacy, .proposal_draft = planned, .clear_proposals = clear } };
    }
    fn legacy(p: *anyopaque, rs: []const be.DraftRequest) anyerror!void {
        const x = self(p);
        x.legacy_calls += 1;
        for (rs, 0..) |r, i| {
            try t.expect(r.plan == null);
            x.depths[i] = r.depth;
        }
        x.ids = .{ 9, 1, 7, 3 };
        if (x.fail) return error.LegacyFailure;
    }
    fn planned(p: *anyopaque, rs: []const be.DraftRequest) anyerror!void {
        const x = self(p);
        x.routed_calls += 1;
        for (rs, 0..) |r, i| {
            const plan = r.plan orelse return error.MissingPlan;
            x.plans[i] = plan.kind;
            x.depths[i] = r.depth;
            if (plan.kind == .subset) {
                try t.expectEqual(@as(?u32, 7), plan.map.?.global(2));
                try t.expectEqual(@as(?u32, null), plan.map.?.global(3));
            }
            if (plan.kind == .paused) try t.expect(r.depth == 0 and r.lanes == null and !r.ranks);
        }
        x.held = true;
        if (x.cancel_after) |flag| flag.* = true;
        if (x.fail) return error.Injected;
    }
    fn clear(p: *anyopaque, _: *Stream) void {
        const x = self(p);
        x.clears += 1;
        x.held = false;
    }
};

test "proposal router: no service and explicit legacy preserve calls depths IDs and failures" {
    var s = try stream();
    defer s.deinit(gpa);
    var old: Spy = .{};
    var legacy: Spy = .{};
    const service: h.Service = .{ .gpa = gpa };
    try old.backend(null).draft(&.{request(&s)});
    try legacy.backend(&service).draft(&.{request(&s)});
    try t.expectEqualDeep(old, legacy);
    try t.expectEqual(@as(u32, 1), legacy.legacy_calls);
    try t.expectEqual(@as(u32, 0), legacy.routed_calls);
    try t.expectEqual(@as(u32, 0), legacy.clears);
    try t.expectEqual(@as(?u32, null), s.proposal_depth);
    old.fail = true;
    legacy.fail = true;
    try t.expectError(error.LegacyFailure, old.backend(null).draft(&.{request(&s)}));
    try t.expectError(error.LegacyFailure, legacy.backend(&service).draft(&.{request(&s)}));
    try t.expectEqualDeep(old, legacy);
}

test "proposal router: artifact identity unsupported formats full diagnostics and placement refuse" {
    var a = try artifact();
    defer a.deinit();
    var wrong = a.identity();
    wrong.hidden += 1;
    try t.expectError(error.ProposalIdentityMismatch, h.Service.calibrated(gpa, &a, wrong, .{ .subset = true, .full = true }));
    try t.expectError(error.ProposalSubsetUnsupported, h.Service.calibrated(gpa, &a, a.identity(), .{ .subset = false, .full = true }));
    try t.expectError(error.ProposalPlacementUnsupported, h.Service.calibrated(gpa, &a, a.identity(), .{ .subset = true, .full = true, .ranks = 2 }));
    try t.expectError(error.ProposalFullUnsupported, h.Service.full(gpa, 11, false));
    var s = try stream();
    defer s.deinit(gpa);
    const service = try h.Service.full(gpa, 11, true);
    var x: Spy = .{};
    var b = x.backend(&service);
    var vt = b.vtable.*;
    vt.proposal_draft = null;
    b.vtable = &vt;
    try t.expectError(error.ProposalServiceUnsupported, b.draft(&.{request(&s)}));
    try t.expectEqual(@as(u32, 0), x.clears + x.legacy_calls + x.routed_calls);
}

test "proposal router: mixed spans forced OOV grammar EOS unknown history and resume are per stream" {
    var a = try artifact();
    defer a.deinit();
    const service = try h.Service.calibrated(gpa, &a, a.identity(), .{ .subset = true, .full = true });
    var one = try stream();
    defer one.deinit(gpa);
    var two = try stream();
    defer two.deinit(gpa);
    var x: Spy = .{};
    const b = x.backend(&service);
    try two.context.append(gpa, 9);
    try b.draft(&.{ request(&one), request(&two) });
    try t.expectEqualSlices(h.Kind, &.{ .subset, .full }, x.plans[0..2]);
    for (0..15) |_| try two.context.append(gpa, 3);
    try b.draft(&.{request(&two)});
    try t.expectEqual(h.Kind.full, x.plans[0]);
    try two.context.append(gpa, 7);
    try b.draft(&.{request(&two)});
    try t.expectEqual(h.Kind.subset, x.plans[0]);
    try one.force.append(gpa, 9);
    try b.draft(&.{request(&one)});
    try t.expectEqual(h.Kind.full, x.plans[0]);
    one.force.clearRetainingCapacity();
    one.eos = &.{9};
    try b.draft(&.{request(&one)});
    try t.expectEqual(h.Kind.full, x.plans[0]);
    one.eos = &.{};
    var constraint: lanes.grammar.Constraint = undefined;
    one.grammar = &constraint;
    try b.draft(&.{request(&one)});
    try t.expectEqual(h.Kind.full, x.plans[0]);
    one.grammar = null;
    var r = request(&one);
    r.first = .{ .value = 9 };
    try b.draft(&.{r});
    try t.expectEqual(h.Kind.full, x.plans[0]);
    r.first = .{ .handle = 123 };
    try b.draft(&.{r});
    try t.expectEqual(h.Kind.full, x.plans[0]);
    one.context.shrinkRetainingCapacity(8);
    try b.draft(&.{request(&one)});
    try t.expectEqual(h.Kind.full, x.plans[0]);
    const diagnostic = try h.Service.full(gpa, 11, true);
    try x.backend(&diagnostic).draft(&.{request(&one)});
    try t.expectEqual(h.Kind.full, x.plans[0]);
}

fn cancelled(p: *anyopaque) bool {
    const flag: *bool = @ptrCast(@alignCast(p));
    return flag.*;
}
test "proposal router: askDrafts pauses without stale holds and clears failures cancellation and allocation refusal" {
    var a = try artifact();
    defer a.deinit();
    var service = try h.Service.calibrated(gpa, &a, a.identity(), .{ .subset = true, .full = false });
    var s = try stream();
    defer s.deinit(gpa);
    var x: Spy = .{};
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .mtp = true, .speculate = true }, 8, 7);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpa, &cfg, x.backend(&service), clock.clock());
    defer e.deinit();
    try e.askDrafts(&.{request(&s)});
    try t.expectEqual(@as(u32, 2), s.next.?.count);
    try s.context.append(gpa, 9);
    try e.askDrafts(&.{request(&s)});
    try t.expectEqual(@as(u32, 0), s.next.?.count);
    try t.expect(s.drafts);
    try t.expectEqual(@as(u32, 0), x.depths[0]);
    s.context.items[s.context.items.len - 1] = 3;
    try e.askDrafts(&.{request(&s)});
    try t.expectEqual(@as(u32, 2), s.next.?.count);
    x.fail = true;
    try t.expectError(error.Injected, e.askDrafts(&.{request(&s)}));
    const p = try lanes.windows.plan(&e, &s, null);
    try t.expectEqual(@as(usize, 0), p.count());
    try t.expect(!x.held);
    x.fail = false;
    var flag = true;
    s.cancel_check = .{ .ptr = &flag, .check = cancelled };
    const before = x.routed_calls;
    try t.expectError(error.Cancelled, e.askDrafts(&.{request(&s)}));
    try t.expectEqual(before, x.routed_calls);
    flag = false;
    x.cancel_after = &flag;
    try t.expectError(error.Cancelled, e.askDrafts(&.{request(&s)}));
    try t.expect(!x.held);
    x.cancel_after = null;
    flag = false;
    var failing = std.testing.FailingAllocator.init(gpa, .{ .fail_index = 0 });
    service.gpa = failing.allocator();
    try t.expectError(error.OutOfMemory, e.askDrafts(&.{request(&s)}));
    try t.expect(!x.held);
}

const FakeDrafter = @import("lanes").drafted;
const External = @import("lanes").fake;

test "proposal router: an omitted target EOS is committed by the existing verifier correction" {
    var a = try artifact();
    defer a.deinit();
    try t.expect(a.column(9) == null);
    const path = try lanes.accept.acceptPath(gpa, &.{ 3, 7 }, &.{ -1, 0 }, &.{ 9, 0 });
    defer gpa.free(path);
    try t.expectEqualSlices(u32, &.{0}, path);
    var s = try stream();
    defer s.deinit(gpa);
    s.eos = &.{9};
    _ = try s.commit(gpa, &.{9}, &.{});
    try t.expect(s.finished and s.reason == .stop);
    try t.expectEqualSlices(u32, &.{9}, s.emitted());
}

test "proposal router: legacy ignores unrouted proposal state" {
    var s = try stream();
    defer s.deinit(gpa);
    s.proposal_depth = 0;
    var x: Spy = .{};
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .mtp = true, .speculate = true }, 8, 7);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    const legacy: h.Service = .{ .gpa = gpa };
    for ([_]?*const h.Service{ null, &legacy }) |service| {
        var e = lanes.Engine.init(gpa, &cfg, x.backend(service), clock.clock());
        defer e.deinit();
        try e.askDrafts(&.{request(&s)});
        try t.expectEqual(@as(u32, 2), s.next.?.count);
        const plan = try lanes.windows.plan(&e, &s, null);
        try t.expectEqual(@as(usize, 2), plan.count());
    }
}

test "proposal router: recorder forwards the optional service without routing a batch twice" {
    var a = try artifact();
    defer a.deinit();
    const service = try h.Service.calibrated(gpa, &a, a.identity(), .{ .subset = true, .full = true });
    var s = try stream();
    defer s.deinit(gpa);
    var x: Spy = .{};
    var recorded: lanes.recorder.Recorded = .{ .inner = x.backend(&service), .recorder = undefined };
    const b = recorded.backend();
    try t.expectEqual(@as(?*const h.Service, &service), b.proposal_head);
    try b.draft(&.{request(&s)});
    try t.expectEqual(@as(u32, 1), x.routed_calls);
    try t.expectEqual(@as(u32, 1), x.clears);
    try t.expectEqual(@as(u32, 0), x.legacy_calls);
}
