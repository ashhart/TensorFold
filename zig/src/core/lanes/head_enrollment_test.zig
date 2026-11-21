//! Request-stable head enrollment through real wrapper/Engine rounds, with history-dependent target tokens.
const std = @import("std");
const be = @import("backend.zig");
const dr = @import("drafter.zig");
const dt = @import("drafted_test.zig");
const fake = @import("fake.zig");
const Stream = @import("stream.zig").Stream;
const Engine = @import("engine.zig").Engine;
const Config = @import("config.zig").Config;
const Drafted = @import("drafted.zig").Drafted;
const Sampling = @import("sampling.zig").Sampling;
const Proposer = @import("proposer.zig").Proposer;
const win = @import("windows.zig");
const hp = @import("hold_price.zig");
const gpa = std.testing.allocator;

/// The existing history-rebuilding head, with a deterministic admission policy instead of device pricing.
const Provider = struct {
    head: dt.FakeDrafter = .{},
    table: dr.Drafter.VTable = undefined,
    track: bool = true,
    only: ?*Stream = null,
    max_heads: usize = 32,
    admissions: [32]dr.Enrollment = undefined,
    admitted: usize = 0,
    prior: [4]f64 = @splat(0.95),
    proposed: u64 = 0,
    accepted: u64 = 0,
    invalid_observation: bool = false,

    fn of(ptr: *anyopaque) *Provider {
        return @fieldParentPtr("head", @as(*dt.FakeDrafter, @ptrCast(@alignCast(ptr))));
    }
    fn enroll(ptr: *anyopaque, s: *Stream, context: dr.Enrollment) bool {
        const x = of(ptr);
        x.admissions[x.admitted] = context;
        x.admitted += 1;
        s.head_prior = &x.prior;
        return x.track and (x.only == null or x.only == s) and context.heads <= x.max_heads;
    }
    fn observe(ptr: *anyopaque, s: *Stream, proposed: u32, accepted: u32) void {
        const x = of(ptr);
        if (!s.headDrafts() or !x.head.lanes.contains(s) or proposed == 0 or accepted > proposed) x.invalid_observation = true;
        x.proposed += proposed;
        x.accepted += accepted;
    }
    fn chances(ptr: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
        const lane = of(ptr).head.lanes.getPtr(s) orelse return false;
        if (out.len > lane.held.items.len) return error.TooFewDrafts;
        var joint: f64 = 1;
        for (out) |*p| {
            joint *= 0.95;
            p.* = joint;
        }
        return true;
    }
    fn drafter(x: *Provider) dr.Drafter {
        const inner = x.head.drafter();
        x.table = inner.vtable.*;
        x.table.enroll = enroll;
        x.table.observe_head = observe;
        x.table.chances = chances;
        return .{ .ptr = &x.head, .vtable = &x.table };
    }
};

/// A genuine priced copy consumer: proposes a target continuation from the complete committed history.
const Copier = struct {
    sampling: ?Sampling = null,
    enabled: bool = true,
    tokens: [3]u32 = undefined,
    rows: u64 = 0,
    kept: u64 = 0,

    fn propose(ptr: *anyopaque, context: []const u32, most: i64) anyerror![]const u32 {
        const x: *Copier = @ptrCast(@alignCast(ptr));
        if (!x.enabled or most < 3) return &.{};
        var history: [512]u32 = undefined;
        @memcpy(history[0..context.len], context);
        for (&x.tokens, 0..) |*t, i| {
            t.* = fake.next(history[0 .. context.len + i], x.sampling, context.len + i);
            history[context.len + i] = t.*;
        }
        return &x.tokens;
    }
    fn observe(ptr: *anyopaque, rows: i64, kept: i64) void {
        const x: *Copier = @ptrCast(@alignCast(ptr));
        x.rows += @intCast(rows);
        x.kept += @intCast(kept);
    }
    fn match(_: *anyopaque) i64 {
        return 0;
    }
    fn proposer(x: *Copier) Proposer {
        return .{ .ptr = x, .vtable = &.{ .propose = propose, .last_match = match, .observe = observe, .priced = true } };
    }
};

fn plain(prompt: []const u32, max_new: u32, sampling: ?Sampling) ![]u32 {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var cfg = try Config.init(gpa, dt.base(), 16, 15);
    defer cfg.deinit(gpa);
    cfg.timed = false;
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var s = try Stream.init(gpa, .{ .id = "plain", .prompt = prompt, .max_new = max_new, .sampling = sampling, .drafts = false });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    while (engine.activeCount() > 0) try engine.step();
    return gpa.dupe(u32, s.emitted());
}

const cheap_holds = [_]hp.HoldCost{ .{ .streams = 1, .depth = 4, .ms = 0.01 }, .{ .streams = 2, .depth = 4, .ms = 0.02 } };
const contexts = [_]hp.ContextCost{ .{ .streams = 1, .rows = 512, .ms = 0.01 }, .{ .streams = 2, .rows = 512, .ms = 0.02 } };
fn pricedFacts(holds: []const hp.HoldCost) dr.Facts {
    return .{ .depth = 4, .step_ms = 0.1, .prior = &.{ 0.95, 0.95, 0.95, 0.95 }, .batched = true, .chances = true, .plain_guard = true, .hold_costs = holds, .hold_context_costs = &contexts, .hold_rows = 32, .hold_block_rows = 9 };
}

fn routes(count: usize, mixed: bool, sampling: ?Sampling) !void {
    const max_new = 60;
    const want = try plain(&dt.p1, max_new, sampling);
    defer gpa.free(want);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var strict: dt.Strict = .{ .inner = target.backend() };
    defer strict.deinit();
    var provider: Provider = .{ .track = mixed, .head = .{ .facts_ = pricedFacts(&cheap_holds) } };
    defer provider.head.deinit();
    var wrapped = try Drafted.init(gpa, strict.backend(), provider.drafter());
    defer wrapped.deinit();
    var cfg = try Config.init(gpa, wrapped.facts(dt.base()), 16, 15);
    defer cfg.deinit(gpa);
    cfg.timed = false;
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
    defer engine.deinit();
    var streams: [2]Stream = undefined;
    var copiers = [_]Copier{ .{ .sampling = sampling }, .{ .sampling = sampling, .enabled = !mixed } };
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(gpa);
    defer for (streams[0..made]) |*s| if (!s.finished) engine.discard(s);
    if (mixed) provider.only = &streams[1];
    for (streams[0..count], copiers[0..count]) |*s, *copy| {
        s.* = try Stream.init(gpa, .{ .id = "route", .prompt = &dt.p1, .max_new = max_new, .sampling = sampling, .proposer = copy.proposer() });
        made += 1;
        try engine.addStream(s);
    }
    try std.testing.expect(streams[0].drafts and !streams[0].headDrafts());
    try std.testing.expectEqual(@as(usize, 0), streams[0].head_prior.len);
    try std.testing.expectEqual(@as(usize, @intFromBool(mixed)), wrapped.heads);
    try std.testing.expectEqual(@as(usize, @intFromBool(mixed)), strict.snaps.count());
    try std.testing.expectEqual(count, wrapped.lanes.count());
    var copied_rounds: usize = 0;
    while (engine.activeCount() > 0) {
        if (mixed) {
            copiers[1].enabled = streams[1].accepted > 0 and copied_rounds < 3;
            if (copiers[1].enabled) copied_rounds += 1;
        }
        try engine.step();
        if (!streams[0].finished) {
            try std.testing.expect(streams[0].depth == null and streams[0].next == null);
            try std.testing.expect(!provider.head.lanes.contains(&streams[0]));
            try std.testing.expect(!strict.snaps.contains(&streams[0]));
        }
    }
    for (streams[0..count]) |*s| try std.testing.expectEqualSlices(u32, want, s.emitted());
    try std.testing.expect(copiers[0].rows > 0 and copiers[0].kept == copiers[0].rows);
    try std.testing.expect(streams[0].accepted > 0);
    if (mixed) {
        try std.testing.expect(streams[1].accepted > 0 and provider.head.holds > 0 and provider.proposed > 0);
        try std.testing.expect(copiers[1].rows > 0);
        try std.testing.expectEqual(streams[1].drafted - copiers[1].rows, provider.proposed);
        try std.testing.expectEqual(streams[1].accepted - copiers[1].kept, provider.accepted);
    } else {
        try std.testing.expectEqual(@as(u32, 0), provider.head.opens + provider.head.absorbs + provider.head.holds);
        try std.testing.expectEqual(@as(u64, 0), provider.proposed);
        try std.testing.expectEqual(@as(usize, 0), wrapped.windows.capacity);
    }
    try std.testing.expect(!provider.invalid_observation);
    try std.testing.expectEqual(provider.proposed, strict.head_proposed);
    try std.testing.expectEqual(provider.accepted, strict.head_accepted);
    try std.testing.expectEqual(@as(usize, 0), wrapped.lanes.count());
    try std.testing.expectEqual(@as(usize, 0), wrapped.heads);
    for (streams[0..count]) |s| try std.testing.expect(s.head_context == null and s.head_prior.len == 0);
}

test "head enrollment: target-only lone and shared copies retain exact target tokens without head state" {
    for ([_]?Sampling{ null, dt.sampled }) |sampling| for ([_]usize{ 1, 2 }) |count| try routes(count, false, sampling);
}

test "head enrollment: mixed requests keep tracked features and observe only the real head prefix" {
    for ([_]?Sampling{ null, dt.sampled }) |sampling| try routes(2, true, sampling);
}

test "head enrollment: tracked zero-funded rounds retain complete context and recover real accepted drafts" {
    const max_new = 100;
    for ([_]?Sampling{ null, dt.sampled }) |sampling| {
        const want = try plain(&dt.p1, max_new, sampling);
        defer gpa.free(want);
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var strict: dt.Strict = .{ .inner = target.backend() };
        defer strict.deinit();
        var holds = [_]hp.HoldCost{.{ .streams = 1, .depth = 4, .ms = 1000 }};
        var provider: Provider = .{ .head = .{ .facts_ = pricedFacts(&holds) } };
        defer provider.head.deinit();
        var wrapped = try Drafted.init(gpa, strict.backend(), provider.drafter());
        defer wrapped.deinit();
        var cfg = try Config.init(gpa, wrapped.facts(dt.base()), 16, 15);
        defer cfg.deinit(gpa);
        cfg.timed = false;
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
        defer engine.deinit();
        var s = try Stream.init(gpa, .{ .id = "recover", .prompt = &dt.p1, .max_new = max_new, .sampling = sampling });
        defer s.deinit(gpa);
        defer if (!s.finished) engine.discard(&s);
        try engine.addStream(&s);
        for (0..4) |_| {
            try engine.step();
            try std.testing.expect(s.headDrafts() and s.next.?.count == 0);
            try std.testing.expectEqualSlices(u32, s.context.items[0..@intCast(s.cache_len)], provider.head.lanes.getPtr(&s).?.history.items);
        }
        try std.testing.expectEqual(@as(u32, 0), provider.head.holds);
        try std.testing.expectEqual(@as(u64, 0), provider.proposed);
        // The same enrolled request receives a paying quote later; no reopened lane or fabricated failure.
        holds[0].ms = 0.01;
        for (0..64) |_| {
            try engine.step();
            if (s.accepted > 0) break;
        }
        try std.testing.expect(s.accepted > 0 and provider.head.holds > 0);
        try std.testing.expectEqual(@as(usize, 1), provider.admitted);
        try std.testing.expectEqual(@as(u32, 1), provider.head.opens);
        while (engine.activeCount() > 0) try engine.step();
        try std.testing.expectEqualSlices(u32, want, s.emitted());
        try std.testing.expectEqual(s.drafted, provider.proposed);
        try std.testing.expectEqual(s.accepted, provider.accepted);
    }
}

const Stop = struct {
    stopped: bool = false,
    fn check(ptr: *anyopaque) bool {
        return @as(*Stop, @ptrCast(@alignCast(ptr))).stopped;
    }
};

test "head enrollment: prompt chunks never rechoose and cancellation releases both routes before readmission" {
    for ([_]bool{ false, true }) |track| {
        var target: fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
        defer target.deinit();
        var provider: Provider = .{ .track = track };
        defer provider.head.deinit();
        var wrapped = try Drafted.init(gpa, target.stepped(), provider.drafter());
        defer wrapped.deinit();
        var cfg = try Config.init(gpa, wrapped.facts(dt.base()), 16, 15);
        defer cfg.deinit(gpa);
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
        defer engine.deinit();
        var stop: Stop = .{};
        var s = try Stream.init(gpa, .{ .id = "chunks", .prompt = &dt.long, .max_new = 20, .cancel_check = .{ .ptr = &stop, .check = Stop.check } });
        defer s.deinit(gpa);
        try std.testing.expect(!try engine.fillStream(&s, true));
        provider.track = !track;
        try std.testing.expect(!try engine.fillStream(&s, false));
        try std.testing.expectEqual(@as(?bool, track), s.head_context);
        try std.testing.expectEqual(@as(usize, 1), provider.admitted);
        try std.testing.expect(s.fill_at.load(.monotonic) > 0 and s.emitted().len == 0);
        stop.stopped = true;
        try std.testing.expectError(error.Cancelled, engine.fillStream(&s, false));
        engine.backend.release(&s); // the host's redundant error cleanup must not release another admission
        try std.testing.expectEqual(@as(usize, 0), wrapped.lanes.count());
        try std.testing.expectEqual(@as(usize, 0), wrapped.heads);
        try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
        try std.testing.expectEqual(@as(u32, @intFromBool(track)), provider.head.releases);
        try std.testing.expect(s.head_context == null and s.head_prior.len == 0);
        stop.stopped = false;
        var first = true;
        while (!try engine.fillStream(&s, first)) first = false;
        try std.testing.expectEqual(@as(?bool, !track), s.head_context);
        try std.testing.expectEqual(@as(usize, 1), provider.admissions[1].streams);
        try std.testing.expectEqual(@as(usize, 1), provider.admissions[1].heads);
        while (engine.activeCount() > 0) try engine.step();
        const want = try plain(&dt.long, 20, null);
        defer gpa.free(want);
        try std.testing.expectEqualSlices(u32, want, s.emitted());
    }
}

test "head enrollment: failed target admission and head open do not consume the next request's cohort" {
    const Streamed = struct {
        fn prefill(ptr: *anyopaque, s: *Stream, taps: []const u32, sink: be.FeatureSink) anyerror!void {
            const target: *fake.Fake = @ptrCast(@alignCast(ptr));
            const b = target.backend();
            try b.prefill(s);
            try sink.take(sink.ptr, s, try b.features(s, taps, 0, @intCast(s.prompt_len)), 0);
        }
    };
    for ([_]bool{ false, true }) |streamed| for ([_]bool{ false, true }) |track| for ([_]bool{ false, true }) |fail_head| {
        if (!track and fail_head) continue;
        var target: fake.Fake = .{ .gpa = gpa, .memory_lanes = if (fail_head) null else 0 };
        defer target.deinit();
        var provider: Provider = .{ .track = track, .max_heads = 1 };
        defer provider.head.deinit();
        if (fail_head) provider.head.fail = .open;
        var target_table = target.backend().vtable.*;
        target_table.prefill_features = if (streamed) Streamed.prefill else null;
        var wrapped = try Drafted.init(gpa, .{ .ptr = &target, .vtable = &target_table }, provider.drafter());
        defer wrapped.deinit();
        var cfg = try Config.init(gpa, wrapped.facts(dt.base()), 16, 15);
        defer cfg.deinit(gpa);
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
        defer engine.deinit();
        var s = try Stream.init(gpa, .{ .id = "failure", .prompt = &dt.p1, .max_new = 20 });
        defer s.deinit(gpa);
        if (fail_head) try std.testing.expectError(error.Injected, engine.addStream(&s)) else try std.testing.expectError(error.LaneMemory, engine.addStream(&s));
        try std.testing.expect(s.head_context == null and s.head_prior.len == 0);
        try std.testing.expectEqual(@as(usize, 0), wrapped.lanes.count());
        try std.testing.expectEqual(@as(usize, 0), wrapped.heads);
        try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
        target.memory_lanes = null;
        provider.head.fail = .none;
        provider.track = true;
        try engine.addStream(&s);
        try std.testing.expect(s.headDrafts());
        try std.testing.expectEqual(@as(usize, 1), provider.admissions[1].streams);
        try std.testing.expectEqual(@as(usize, 1), provider.admissions[1].heads);
        while (engine.activeCount() > 0) try engine.step();
        const want = try plain(&dt.p1, 20, null);
        defer gpa.free(want);
        try std.testing.expectEqualSlices(u32, want, s.emitted());
    };
}

test "head enrollment: detach adopt preserves route and context but invalidates the reserved copy" {
    for ([_]bool{ false, true }) |track| {
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var strict: dt.Strict = .{ .inner = target.backend() };
        defer strict.deinit();
        var provider: Provider = .{ .track = track, .head = .{ .facts_ = pricedFacts(&cheap_holds) } };
        defer provider.head.deinit();
        var wrapped = try Drafted.init(gpa, strict.backend(), provider.drafter());
        defer wrapped.deinit();
        var cfg = try Config.init(gpa, wrapped.facts(dt.base()), 16, 15);
        defer cfg.deinit(gpa);
        cfg.pipelined = false; // handover's explicit contract: no queued draw
        cfg.timed = false;
        var clock: fake.FixedClock = .{};
        var before = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
        defer before.deinit();
        var after = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
        defer after.deinit();
        var copier: Copier = .{};
        var s = try Stream.init(gpa, .{ .id = "handover", .prompt = &dt.p1, .max_new = 60, .proposer = copier.proposer() });
        defer s.deinit(gpa);
        try before.addStream(&s);
        try before.step();
        _ = try win.reserveCopy(&before, &s);
        try std.testing.expect(s.reserved_copy != null);
        const context_rows = if (track) provider.head.lanes.getPtr(&s).?.history.items.len else 0;
        try std.testing.expect(before.detach(&s));
        provider.track = !track;
        try std.testing.expectEqual(@as(?bool, track), s.head_context);
        try std.testing.expect(s.reserved_copy == null and s.copy_tokens.items.len == 0);
        try std.testing.expectEqual(@as(usize, 1), wrapped.lanes.count());
        if (track) try std.testing.expectEqual(context_rows, provider.head.lanes.getPtr(&s).?.history.items.len);
        try after.adopt(&s);
        try after.step();
        try std.testing.expectEqual(@as(usize, 1), provider.admitted);
        try std.testing.expect(copier.kept > 0);
        while (after.activeCount() > 0) try after.step();
        const want = try plain(&dt.p1, 60, null);
        defer gpa.free(want);
        try std.testing.expectEqualSlices(u32, want, s.emitted());
        try std.testing.expectEqual(@as(usize, 0), wrapped.lanes.count());
        try std.testing.expectEqual(@as(u32, @intFromBool(track)), provider.head.releases);
    }
}

test "head enrollment: admission counts target-only peers and discard releases their cohort slot" {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var provider: Provider = .{ .track = false, .max_heads = 1 };
    defer provider.head.deinit();
    var wrapped = try Drafted.init(gpa, target.backend(), provider.drafter());
    defer wrapped.deinit();
    var cfg = try Config.init(gpa, wrapped.facts(dt.base()), 16, 15);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
    defer engine.deinit();
    var ss: [3]Stream = undefined;
    var made: usize = 0;
    defer for (ss[0..made]) |*s| s.deinit(gpa);
    defer for (ss[0..made]) |*s| if (!s.finished) engine.discard(s);
    for (&ss, 0..) |*s, i| {
        s.* = try Stream.init(gpa, .{ .id = "cohort", .prompt = &dt.p1, .max_new = 20 });
        made += 1;
        provider.track = i > 0;
        try engine.addStream(s);
    }
    try std.testing.expect(ss[0].head_context == false and ss[1].head_context == true and ss[2].head_context == false);
    try std.testing.expectEqual(@as(usize, 2), provider.admissions[1].streams);
    try std.testing.expectEqual(@as(usize, 1), provider.admissions[1].heads);
    try std.testing.expectEqual(@as(usize, 3), provider.admissions[2].streams);
    try std.testing.expectEqual(@as(usize, 2), provider.admissions[2].heads);
    try engine.step();
    const want = try plain(&dt.p1, 20, null);
    defer gpa.free(want);
    for (&ss) |*s| try std.testing.expectEqualSlices(u32, want[0..s.emitted().len], s.emitted());
    engine.discard(&ss[0]);
    engine.discard(&ss[1]);
    var next = try Stream.init(gpa, .{ .id = "next", .prompt = &dt.p1, .max_new = 20 });
    defer next.deinit(gpa);
    try engine.addStream(&next);
    defer if (!next.finished) engine.discard(&next);
    try std.testing.expect(next.headDrafts());
    try std.testing.expectEqual(@as(usize, 2), provider.admissions[3].streams);
    try std.testing.expectEqual(@as(usize, 1), provider.admissions[3].heads);
    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expectEqualSlices(u32, want, next.emitted());
    try std.testing.expectEqualSlices(u32, want, ss[2].emitted());
}

test "head enrollment: provider prior changes initial priced depth without sharing mutable rule state" {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var provider: Provider = .{};
    defer provider.head.deinit();
    var wrapped = try Drafted.init(gpa, target.backend(), provider.drafter());
    defer wrapped.deinit();
    var cfg = try Config.init(gpa, wrapped.facts(dt.base()), 16, 15);
    defer cfg.deinit(gpa);
    cfg.timed = false;
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
    defer engine.deinit();
    var good = try Stream.init(gpa, .{ .id = "good", .prompt = &dt.p1, .max_new = 40 });
    defer good.deinit(gpa);
    var poor = try Stream.init(gpa, .{ .id = "poor", .prompt = &dt.p1, .max_new = 40 });
    defer poor.deinit(gpa);
    try engine.addStream(&good);
    defer if (!good.finished) engine.discard(&good);
    const seeded = good.depth.?.p[0];
    provider.prior = @splat(0.01);
    try engine.addStream(&poor);
    defer if (!poor.finished) engine.discard(&poor);
    try std.testing.expect(good.next.?.count > poor.next.?.count);
    try std.testing.expectEqual(seeded, good.depth.?.p[0]);
    try std.testing.expect(good.depth.?.p.ptr != poor.depth.?.p.ptr);
    while (engine.activeCount() > 0) try engine.step();
    const want = try plain(&dt.p1, 40, null);
    defer gpa.free(want);
    try std.testing.expectEqualSlices(u32, want, good.emitted());
    try std.testing.expectEqualSlices(u32, want, poor.emitted());
}

test "head enrollment: restored-prompt fillers do not train provider acceptance before a real hold" {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var early = try Stream.init(gpa, .{ .id = "saved", .prompt = &dt.p1, .max_new = 1 });
    defer early.deinit(gpa);
    try target.backend().prefill(&early);
    const saved = try target.save(&early, dt.p1.len);
    defer target.drop(saved);
    target.backend().release(&early);
    var strict: dt.Strict = .{ .inner = target.backend() };
    defer strict.deinit();
    var provider: Provider = .{};
    defer provider.head.deinit();
    var wrapped = try Drafted.init(gpa, strict.backend(), provider.drafter());
    defer wrapped.deinit();
    var cfg = try Config.init(gpa, wrapped.facts(dt.base()), 16, 15);
    defer cfg.deinit(gpa);
    cfg.timed = false;
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
    defer engine.deinit();
    var s = try Stream.init(gpa, .{ .id = "restored", .prompt = &dt.p1, .max_new = 40, .reuse = .{ .saved = saved, .at = dt.p1.len } });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    defer if (!s.finished) engine.discard(&s);
    try std.testing.expect(s.next.?.count > 0 and provider.head.holds == 0);
    try engine.step();
    try std.testing.expectEqual(@as(u64, 0), provider.proposed);
    try std.testing.expect(provider.head.holds > 0);
    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expect(provider.proposed > 0 and !provider.invalid_observation);
    const want = try plain(&dt.p1, 40, null);
    defer gpa.free(want);
    try std.testing.expectEqualSlices(u32, want, s.emitted());
}

test "head enrollment: interrupted-prefill callback failures and successful cancellation release the same admission" {
    const Cancel = struct {
        fn call(ptr: *anyopaque, s: *Stream) anyerror!void {
            const target: *fake.Fake = @ptrCast(@alignCast(ptr));
            const fail: *bool = @ptrCast(@alignCast(target.prefill_hook_ctx.?));
            try std.testing.expect(s.fill_at.load(.monotonic) > 0 and s.emitted().len == 0);
            if (fail.*) return error.SaveFailed;
        }
    };
    for ([_]bool{ false, true }) |track| for ([_]bool{ false, true }) |failed| {
        var target: fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
        defer target.deinit();
        var fail = failed;
        target.prefill_hook_ctx = &fail;
        var target_table = target.stepped().vtable.*;
        target_table.cancel_prefill = Cancel.call;
        var provider: Provider = .{ .track = track };
        defer provider.head.deinit();
        var wrapped = try Drafted.init(gpa, .{ .ptr = &target, .vtable = &target_table }, provider.drafter());
        defer wrapped.deinit();
        var cfg = try Config.init(gpa, wrapped.facts(dt.base()), 16, 15);
        defer cfg.deinit(gpa);
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
        defer engine.deinit();
        var s = try Stream.init(gpa, .{ .id = "interrupted", .prompt = &dt.long, .max_new = 20 });
        defer s.deinit(gpa);
        try std.testing.expect(!try engine.fillStream(&s, true));
        const backend = engine.backend;
        if (failed) try std.testing.expectError(error.SaveFailed, backend.vtable.cancel_prefill.?(backend.ptr, &s)) else try backend.vtable.cancel_prefill.?(backend.ptr, &s);
        engine.discard(&s);
        try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
        try std.testing.expectEqual(@as(usize, 0), wrapped.lanes.count());
        try std.testing.expectEqual(@as(usize, 0), wrapped.heads);
        try std.testing.expectEqual(@as(u32, @intFromBool(track)), provider.head.releases);
        try std.testing.expect(s.head_context == null and s.head_prior.len == 0);
        var next = try Stream.init(gpa, .{ .id = "after-interrupt", .prompt = &dt.p1, .max_new = 20 });
        defer next.deinit(gpa);
        try engine.addStream(&next);
        defer if (!next.finished) engine.discard(&next);
        try std.testing.expectEqual(@as(usize, 1), provider.admissions[1].streams);
        try std.testing.expectEqual(@as(usize, 1), provider.admissions[1].heads);
        while (engine.activeCount() > 0) try engine.step();
        const want = try plain(&dt.p1, 20, null);
        defer gpa.free(want);
        try std.testing.expectEqualSlices(u32, want, next.emitted());
    };
}
