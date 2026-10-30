//! The next draft starts after a host step returns, with owned rows and unchanged target tokens.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const Config = @import("config.zig").Config;
const Stream = @import("stream.zig").Stream;
const fake = @import("fake.zig");
const a = std.testing.allocator;
fn config() !Config {
    return Config.init(a, .{ .exact_width = 16, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 15, .hidden_rows = true, .batch_rows = 16 }, 16, 15);
}

test "late draft: a greedy step returns before drafting again, then owns its rows across arena reset" {
    var cfg = try config();
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var backend = target.backend();
    backend.defer_lone_draft = true;
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, backend, clock.clock());
    defer e.deinit();
    var s = try Stream.init(a, .{ .id = "deferred", .prompt = &.{ 3, 1, 4 }, .max_new = 48 });
    defer s.deinit(a);
    try e.addStream(&s);
    const initial = target.drafts_calls;
    try e.step();
    try std.testing.expectEqual(initial, target.drafts_calls);
    try std.testing.expect(e.late.request != null);
    const r = e.late.request.?;
    try std.testing.expectEqual(s.pending.?, e.late.follow[r.follow.len - 1]);
    try e.step();
    try std.testing.expectEqual(initial + 1, target.drafts_calls);
    while (e.activeCount() > 0) try e.step();
    try std.testing.expect(e.late.request == null);
    for (s.emitted(), 0..) |t, i| try std.testing.expectEqual(fake.next(s.context.items[0 .. s.prompt_len + i], null, s.prompt_len + i), t);
}

test "late draft: detach and cancellation drop the retained request before a stream can be freed" {
    var cfg = try config();
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var backend = target.backend();
    backend.defer_lone_draft = true;
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, backend, clock.clock());
    defer e.deinit();
    var s = try Stream.init(a, .{ .id = "released", .prompt = &.{ 3, 1, 4 }, .max_new = 48 });
    defer s.deinit(a);
    try e.addStream(&s);
    try e.step();
    try std.testing.expect(e.late.request != null);
    try std.testing.expect(e.detach(&s));
    try std.testing.expect(e.late.request == null);
    try e.adopt(&s);
    try e.step();
    try std.testing.expect(e.late.request != null);
    e.discard(&s);
    try std.testing.expect(e.late.request == null);
}

test "late draft: default and sampled paths keep their immediate scheduling" {
    for ([_]bool{ false, true }) |enabled| {
        var cfg = try config();
        defer cfg.deinit(a);
        var target: fake.Fake = .{ .gpa = a };
        defer target.deinit();
        var backend = target.backend();
        backend.defer_lone_draft = enabled;
        var clock: fake.FixedClock = .{};
        var e = Engine.init(a, &cfg, backend, clock.clock());
        defer e.deinit();
        var s = try Stream.init(a, .{ .id = "unchanged", .prompt = &.{ 3, 1, 4 }, .max_new = 48, .sampling = if (enabled) .{ .seed = 9, .temperature = 1, .top_k = 0, .top_p = 1 } else null });
        defer s.deinit(a);
        try e.addStream(&s);
        const initial = target.drafts_calls;
        try e.step();
        try std.testing.expectEqual(initial + 1, target.drafts_calls);
        try std.testing.expect(e.late.request == null);
    }
}

test "late draft: retained rows own their bytes and an ask failure clears the request" {
    const State = @import("late_draft.zig").State;
    const be = @import("backend.zig");
    const Recording = struct {
        live: struct { items: []const *Stream },
        calls: usize = 0,
        fail: bool = false,
        fn ask(r: *@This(), requests: []const be.DraftRequest) !void {
            r.calls += 1;
            try std.testing.expectEqualSlices(u32, &.{ 7, 8 }, requests[0].follow);
            try std.testing.expectEqualSlices(u32, &.{ 0, 3 }, requests[0].rows.?);
            if (r.fail) return error.InjectedDraftFailure;
        }
    };
    var s = try Stream.init(a, .{ .id = "owned", .prompt = &.{1}, .max_new = 48 });
    defer s.deinit(a);
    var live = [_]*Stream{&s};
    var recording = Recording{ .live = .{ .items = &live } };
    var state: State = .{};
    var follow = [_]u32{ 7, 8 };
    var rows = [_]u32{ 0, 3 };
    const request: be.DraftRequest = .{ .stream = &s, .follow = &follow, .rows = &rows, .start = 1, .position = 3, .depth = 15 };
    try std.testing.expect(state.put(true, request));
    @memset(&follow, 99);
    @memset(&rows, 99);
    try state.flush(&recording, Recording.ask);
    try std.testing.expect(state.request == null);
    follow = .{ 7, 8 };
    rows = .{ 0, 3 };
    try std.testing.expect(state.put(true, request));
    recording.fail = true;
    try std.testing.expectError(error.InjectedDraftFailure, state.flush(&recording, Recording.ask));
    try std.testing.expect(state.request == null);
    try std.testing.expectEqual(@as(usize, 2), recording.calls);
    try std.testing.expect(!state.put(false, request));
}
