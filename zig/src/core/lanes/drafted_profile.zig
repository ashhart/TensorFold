//! The drafted wrapper's counters and round-profile state: each gate read once, the phases fused per round.
const std = @import("std");

pub const DraftStats = struct {
    rounds: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    draft_rounds: std.atomic.Value(u64) = std.atomic.Value(u64).init(0), // windows with held > 0
    offered: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    kept: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    held_max: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    kept_max: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
};
pub var draft_stats: DraftStats = .{};
pub var draft_stats_env: ?bool = null; // the TF_DRAFT_STATS gate, read once (the hot path must not getenv per round)
pub var draft_debug_env: ?bool = null; // the TF_DRAFT_DEBUG gate, read once (the hot path must not getenv per round)

/// TF_ROUND_PROFILE: one line a drafted round, its phases in host-wall ms - held (the drafts'
/// DtoH waiting on the last hold), fwd (the verify window), keep, absorb, hold (the drafter's
/// launch). A phase a round skips rides into the next line; the wall's remainder = the lane
/// host's own time. The depth buckets accumulate across the run (kept/offers by held depth).
pub var profile_env: ?bool = null;
pub var profile_round: u64 = 0;
pub var profile: struct {
    held: f64 = 0,
    fwd: f64 = 0,
    keep: f64 = 0,
    absorb: f64 = 0,
    hold: f64 = 0,
    rows: usize = 0,
    offers: usize = 0,
    kept: usize = 0,
} = .{};
pub var profile_depths: [9]u64 = @splat(0);
pub var profile_offers_at: [9]u64 = @splat(0);

/// Clock reads the round profile has taken (profileNow invocations). A disabled TF_ROUND_PROFILE
/// performs zero of them: every timed site opens through profStart, which reads the clock only
/// under the gate, and closes inside an `if (profileOn())` - so a pass of the round path with the
/// gate latched off must leave this at 0 (the zero-clock-read contract's own counter).
pub var clock_reads: u64 = 0;

pub fn profileOn() bool {
    if (profile_env == null) profile_env = std.c.getenv("TF_ROUND_PROFILE") != null;
    return profile_env.?;
}

/// A timed section's start: the clock is read ONLY when TF_ROUND_PROFILE is set, so a disabled
/// profile does zero clock reads - the site's own cost is then an unclocked flow straight through.
/// The end site (an `if (profileOn()) profile.X += profileNow() - start`) re-checks the gate, and
/// the gate is latched after its first read, so `0` can never mix with a real start.
pub inline fn profStart() f64 {
    return if (profileOn()) profileNow() else 0;
}

pub fn profileNow() f64 {
    clock_reads += 1;
    const c = @import("sys").c;
    var t: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &t);
    return @as(f64, @floatFromInt(t.sec)) * 1e3 + @as(f64, @floatFromInt(t.nsec)) / 1e6;
}

const be = @import("backend.zig");
const dr = @import("drafter.zig");
const Stream = @import("stream.zig").Stream;
const Drafted = @import("drafted.zig").Drafted;

// The disabled profile's zero-clock-read contract, proven on the full round path (the drafted
// wrapper's own profiled sections — the undrafted verify route included — over a mock target and
// drafter): with the gate latched off the path does its real work while clock_reads and every
// accumulator stay at zero, and the same path under the gate reads clocks on every section.
test "round profile off reads zero clocks: disabled gate, all profiled sites" {
    const gpa = std.testing.allocator;
    const Mock = struct {
        absorbs: u32 = 0,
        holds: u32 = 0,
        verifies: u32 = 0,
        keeps: u32 = 0,
        absorb_reads: u64 = 0, // clock_reads at the mock's entry: >0 means the calling section opened with a real clock read
        hold_reads: u64 = 0,
        keep_reads: u64 = 0,
        verify_reads: u64 = 0,
        held_reads: u64 = 0,
        work: u32 = 0, // burn ever-advances it: every profiled section's real work is measurable
        rows: [4]u32 = .{ 1, 2, 3, 4 },
        fn cast(ptr: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ptr));
        }
        // Real section work: consecutive clock reads on an idle mock can land in the same f64
        // ms bucket at large MONOTONIC magnitudes (a zero measured difference would make the
        // enabled arm's > 0 assertions flake instead of measuring). The burn guarantees every
        // profiled section spans a strictly positive, representable elapsed.
        fn burn(s: *u32) void {
            var acc: u64 = 0;
            var i: u32 = 0;
            while (i < 300_000) : (i += 1) acc +%= @as(u64, i) *% 0x9e3779b97f4a7c15;
            s.* +%= @intCast(acc & 1); // the shift-out keeps the loop from folding away
        }
        fn taps(_: *anyopaque) []const u32 {
            return &.{0};
        }
        fn facts(_: *anyopaque) dr.Facts {
            return .{ .depth = 2, .step_ms = 0.5 };
        }
        fn open(_: *anyopaque, _: *Stream) anyerror!void {}
        fn absorb(ptr: *anyopaque, _: []const dr.Absorb) anyerror!void {
            const m = cast(ptr);
            m.absorbs += 1;
            m.absorb_reads = clock_reads;
            burn(&m.work);
        }
        fn hold(ptr: *anyopaque, _: []const dr.Hold) anyerror!void {
            const m = cast(ptr);
            m.holds += 1;
            m.hold_reads = clock_reads;
            burn(&m.work);
        }
        fn held(ptr: *anyopaque, _: []const *Stream, _: []const []u32) anyerror!void {
            const m = cast(ptr);
            m.held_reads = clock_reads;
            burn(&m.work);
        }
        fn release(_: *anyopaque, _: *Stream) void {}
        fn prefill(_: *anyopaque, _: *Stream) anyerror!void {}
        fn prefillFeatures(ptr: *anyopaque, s: *Stream, _: []const u32, sink: be.FeatureSink) anyerror!void {
            const m = cast(ptr);
            try sink.take(sink.ptr, s, .{ .buffer = @intFromPtr(&m.rows), .rows = 4, .row_bytes = 4, .space = .host, .dtype = .u32 }, 0);
        }
        fn features(ptr: *anyopaque, _: *Stream, _: []const u32, _: u64, n: u32) anyerror!be.Features {
            return .{ .buffer = @intFromPtr(&cast(ptr).rows), .rows = n, .row_bytes = 4, .space = .host, .dtype = .u32 };
        }
        fn first(_: *anyopaque, _: *Stream, _: u64) anyerror!u64 {
            return 0;
        }
        fn queue(_: *anyopaque, _: *Stream, _: be.Feed, _: u64) anyerror!u64 {
            return 0;
        }
        fn read(_: *anyopaque, _: u64) anyerror!u32 {
            return 7;
        }
        fn verify(ptr: *anyopaque, _: []const be.Window, _: []be.Verified) anyerror!void {
            const m = cast(ptr);
            m.verifies += 1;
            m.verify_reads = clock_reads;
            burn(&m.work);
        }
        fn keep(ptr: *anyopaque, _: []const be.Window, _: []const []const u32) anyerror!void {
            const m = cast(ptr);
            m.keeps += 1;
            m.keep_reads = clock_reads;
            burn(&m.work);
        }
        fn draftFn(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {}
    };
    var mock: Mock = .{};
    const target: be.Backend = .{ .ptr = &mock, .vtable = &.{ .prefill = Mock.prefill, .prefill_features = Mock.prefillFeatures, .features = Mock.features, .first = Mock.first, .queue = Mock.queue, .read = Mock.read, .verify = Mock.verify, .keep = Mock.keep, .draft = Mock.draftFn, .release = Mock.release } };
    // draft_round stays unset: the round runs its absorb and its hold (the wrapper's separate profiled sections).
    const drafter: dr.Drafter = .{ .ptr = &mock, .vtable = &.{ .taps = Mock.taps, .facts = Mock.facts, .open = Mock.open, .absorb = Mock.absorb, .hold = Mock.hold, .held = Mock.held, .release = Mock.release } };
    const saved_profile_env = profile_env;
    profile_env = false; // the disabled profile, latched (the gate would otherwise read the ambient env)
    clock_reads = 0; // the disabled arm's zero is exact regardless of any earlier test's reads in this binary
    defer {
        profile_env = saved_profile_env;
        clock_reads = 0;
        profile = .{};
        profile_round = 0;
        @memset(&profile_depths, 0);
        @memset(&profile_offers_at, 0);
    }
    var w0 = try Drafted.init(gpa, target, drafter);
    defer w0.deinit();
    var s1 = try Stream.init(gpa, .{ .id = "pr1", .prompt = &.{ 10, 20, 30, 40 }, .max_new = 4 });
    defer s1.deinit(gpa);
    var s2 = try Stream.init(gpa, .{ .id = "pr2", .prompt = &.{ 11, 21, 31, 41 }, .max_new = 4 });
    defer s2.deinit(gpa);
    const pos = [_]u64{ 4, 5, 6 };
    try w0.backend().prefill(&s1);
    // The disabled arm: the full round path — undrafted verify, keep, hold-draft, drafted verify — with the gate off.
    try w0.backend().verify(&.{.{ .stream = &s1, .pending = 40, .held = 0, .tokens = &.{}, .parents = null, .positions = pos[0..1] }}, &.{});
    try w0.backend().keep(&.{.{ .stream = &s1, .pending = 40, .held = 0, .tokens = &.{}, .parents = null, .positions = pos[0..1] }}, &.{&.{}});
    try w0.backend().draft(&.{.{ .stream = &s1, .follow = &.{}, .first = .{ .value = 50 }, .rows = null, .start = 4, .position = 5, .depth = 2 }});
    try w0.backend().verify(&.{.{ .stream = &s1, .pending = 50, .held = 2, .tokens = &.{}, .parents = null, .positions = pos[0..2] }}, &.{});
    try std.testing.expectEqual(@as(u32, 2), mock.absorbs);
    try std.testing.expectEqual(@as(u32, 1), mock.holds);
    try std.testing.expectEqual(@as(u32, 2), mock.verifies);
    try std.testing.expectEqual(@as(u32, 1), mock.keeps);
    // Every profiled section's work actually ran (its mocks counted), yet no clock was read anywhere around it.
    try std.testing.expectEqual(@as(u64, 0), clock_reads);
    try std.testing.expectEqual(@as(u64, 0), mock.absorb_reads);
    try std.testing.expectEqual(@as(u64, 0), mock.hold_reads);
    try std.testing.expectEqual(@as(u64, 0), mock.keep_reads);
    try std.testing.expectEqual(@as(u64, 0), mock.verify_reads);
    try std.testing.expectEqual(@as(u64, 0), mock.held_reads);
    try std.testing.expectEqual(@as(f64, 0), profile.fwd);
    try std.testing.expectEqual(@as(f64, 0), profile.held);
    try std.testing.expectEqual(@as(f64, 0), profile.keep);
    try std.testing.expectEqual(@as(f64, 0), profile.absorb);
    try std.testing.expectEqual(@as(f64, 0), profile.hold);
    for (profile_depths) |d| try std.testing.expectEqual(@as(u64, 0), d);
    for (profile_offers_at) |o| try std.testing.expectEqual(@as(u64, 0), o);
    // The enabled arm: the same round path under the gate, so the disabled zero is its gate, not a dead counter.
    profile_env = true;
    clock_reads = 0;
    profile = .{};
    profile_round = 0;
    try w0.backend().prefill(&s2);
    try w0.backend().verify(&.{.{ .stream = &s2, .pending = 41, .held = 0, .tokens = &.{}, .parents = null, .positions = pos[0..1] }}, &.{});
    try w0.backend().keep(&.{.{ .stream = &s2, .pending = 41, .held = 0, .tokens = &.{}, .parents = null, .positions = pos[0..1] }}, &.{&.{}});
    try w0.backend().draft(&.{.{ .stream = &s2, .follow = &.{}, .first = .{ .value = 51 }, .rows = null, .start = 4, .position = 5, .depth = 2 }});
    try w0.backend().verify(&.{.{ .stream = &s2, .pending = 51, .held = 2, .tokens = &.{}, .parents = null, .positions = pos[0..2] }}, &.{});
    // Exact per-section pin: the six profiled sections (undrafted verify, keep, absorb, hold,
    // held readback, drafted verify) each open through profStart and close under the gate -
    // exactly two real reads per section, so one number pins open AND close on every section.
    try std.testing.expectEqual(@as(u64, 12), clock_reads);
    // The mocks capture the cumulative counter at their section's entry: one real read per
    // completed earlier section plus this section's own open — 1, 3, 5, 7, 9, 11 in wrapper
    // order (undrafted verify, keep, absorb, hold, held, drafted-verify fwd; the fwd section's
    // own open is its 11th). A missing open shifts every later capture down; a missing
    // close shifts the total below 12.
    try std.testing.expectEqual(@as(u64, 3), mock.keep_reads);
    try std.testing.expectEqual(@as(u64, 5), mock.absorb_reads);
    try std.testing.expectEqual(@as(u64, 7), mock.hold_reads);
    try std.testing.expectEqual(@as(u64, 9), mock.held_reads);
    try std.testing.expectEqual(@as(u64, 11), mock.verify_reads);
    try std.testing.expect(profile.fwd > 0 and profile.held > 0); // the drafted verify's open and close measured real elapsed ms
    try std.testing.expectEqual(@as(u64, 1), profile_round); // the round line printed and the round state reset under the gate
    try w0.backend().draft(&.{.{ .stream = &s2, .follow = &.{}, .first = .{ .value = 52 }, .rows = null, .start = 6, .position = 7, .depth = 0 }});
    try std.testing.expectEqual(@as(u64, 2), profile_round); // mid-round state rides into the next round line alive
}
