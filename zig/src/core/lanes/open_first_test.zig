//! Single-stream opt-in admission preserves committed tokens and default ordering across terminal and failure paths.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const Config = @import("config.zig").Config;
const be = @import("backend.zig");
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const events = @import("events.zig");
const a = std.testing.allocator;
const ids = [_]u32{ 3, 1, 4 };

const Probe = struct {
    target: fake.Fake = .{ .gpa = a },
    table: be.Backend.VTable = undefined,
    hooks: usize = 0,
    drafts: usize = 0,
    releases: usize = 0,
    committed_in_draft: bool = false,
    hook_in_draft: bool = false,
    fail_draft: bool = false,
    cancel: bool = false,
    selected: u32 = 0,

    fn backend(p: *Probe, early: bool, stepped: bool) be.Backend {
        p.table = (if (stepped) p.target.stepped() else p.target.backend()).vtable.*;
        p.table.draft = draft;
        p.table.release = release;
        return .{ .ptr = &p.target, .vtable = &p.table, .first_before_draft = early };
    }
    fn self(ptr: *anyopaque) *Probe {
        const t: *fake.Fake = @ptrCast(@alignCast(ptr));
        return @fieldParentPtr("target", t);
    }
    fn draft(ptr: *anyopaque, req: []const be.DraftRequest) !void {
        const p = self(ptr);
        p.drafts += 1;
        p.committed_in_draft = req[0].stream.emitted().len == 1;
        p.hook_in_draft = p.hooks == 1;
        if (req[0].first) |f| p.selected = switch (f) {
            .value => |v| v,
            .handle => |h| try p.target.backend().read(h),
        };
        if (p.fail_draft) return error.InitialDraftFailed;
        try p.target.backend().draft(req);
    }
    fn release(ptr: *anyopaque, s: *sm.Stream) void {
        const p = self(ptr);
        p.releases += 1;
        p.target.backend().release(s);
    }
    fn hook(ptr: *anyopaque) void {
        const p: *Probe = @ptrCast(@alignCast(ptr));
        p.hooks += 1;
    }
    fn cancelled(ptr: *anyopaque) bool {
        const p: *Probe = @ptrCast(@alignCast(ptr));
        return p.cancel and p.hooks > 0;
    }
};

fn config() !Config {
    return Config.init(a, .{ .exact_width = 8, .hidden_rows = true, .mtp = true, .speculate = true, .drafts = 7 }, 8, 7);
}

fn runOrder(early: bool) !void {
    var p: Probe = .{};
    defer p.target.deinit();
    var cfg = try config();
    defer cfg.deinit(a);
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, p.backend(early, false), clock.clock());
    defer e.deinit();
    var s = try sm.Stream.init(a, .{ .id = "order", .prompt = &ids, .max_new = 8 });
    defer s.deinit(a);
    try e.addStreamCommitted(&s, .{ .ptr = &p, .call = Probe.hook });
    try std.testing.expectEqual(early, p.committed_in_draft);
    try std.testing.expectEqual(early, p.hook_in_draft);
    try std.testing.expectEqual(@as(usize, @intFromBool(early)), p.hooks);
    try std.testing.expectEqual(@as(usize, 1), s.emitted().len);
    e.discard(&s);
    try std.testing.expectEqual(@as(usize, 1), p.releases);
}

test "opt-in commits and hands off before draft; default retains old ordering" {
    try runOrder(true);
    try runOrder(false);
}

test "max-one and EOS terminate successfully without preparing any drafts" {
    for ([_]bool{ false, true }) |eos| {
        var p: Probe = .{ .fail_draft = true };
        defer p.target.deinit();
        var cfg = try config();
        defer cfg.deinit(a);
        var clock: fake.FixedClock = .{};
        var e = Engine.init(a, &cfg, p.backend(true, false), clock.clock());
        defer e.deinit();
        const token = fake.next(&ids, null, ids.len);
        var s = try sm.Stream.init(a, .{ .id = "terminal", .prompt = &ids, .max_new = if (eos) 8 else 1, .eos = if (eos) &.{token} else &.{} });
        defer s.deinit(a);
        try e.addStreamCommitted(&s, .{ .ptr = &p, .call = Probe.hook });
        try std.testing.expect(s.finished);
        try std.testing.expectEqual(@as(usize, 0), p.drafts);
        try std.testing.expectEqual(@as(usize, 1), p.hooks);
        try std.testing.expectEqual(@as(usize, 1), p.releases);
        try std.testing.expectEqual(@as(usize, 0), e.activeCount());
    }
}

test "forced first and its logprob are committed exactly once before a later draft failure" {
    var p: Probe = .{ .fail_draft = true };
    defer p.target.deinit();
    var cfg = try config();
    defer cfg.deinit(a);
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, p.backend(true, false), clock.clock());
    defer e.deinit();
    var s = try sm.Stream.init(a, .{ .id = "forced", .prompt = &ids, .max_new = 8, .logprobs = 2 });
    defer s.deinit(a);
    try s.force.append(a, 42);
    try std.testing.expectError(error.InitialDraftFailed, e.addStreamCommitted(&s, .{ .ptr = &p, .call = Probe.hook }));
    try std.testing.expectEqualSlices(u32, &.{42}, s.emitted());
    try std.testing.expectEqual(@as(u32, 42), p.selected);
    try std.testing.expectEqual(@as(usize, 1), s.rows.items.len);
    try std.testing.expectEqual(@as(u32, 42), s.rows.items[0].token);
    try std.testing.expectEqual(@as(usize, 1), p.hooks);
    e.discard(&s);
    try std.testing.expectEqual(@as(usize, 1), p.releases);
}

test "cancellation at handoff releases once and skips initial drafts" {
    var p: Probe = .{ .cancel = true };
    defer p.target.deinit();
    var cfg = try config();
    defer cfg.deinit(a);
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, p.backend(true, false), clock.clock());
    defer e.deinit();
    var s = try sm.Stream.init(a, .{ .id = "cancel", .prompt = &ids, .max_new = 8, .cancel_check = .{ .ptr = &p, .check = Probe.cancelled } });
    defer s.deinit(a);
    try std.testing.expectError(error.Cancelled, e.addStreamCommitted(&s, .{ .ptr = &p, .call = Probe.hook }));
    try std.testing.expectEqual(@as(usize, 1), p.hooks);
    try std.testing.expectEqual(@as(usize, 1), p.releases);
    try std.testing.expectEqual(@as(usize, 0), p.drafts);
}

test "stepped prefill uses the same committed handoff only after its final chunk" {
    var p: Probe = .{};
    p.target.prefill_chunks = 2;
    defer p.target.deinit();
    var cfg = try config();
    defer cfg.deinit(a);
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, p.backend(true, true), clock.clock());
    defer e.deinit();
    var s = try sm.Stream.init(a, .{ .id = "step", .prompt = &ids, .max_new = 8 });
    defer s.deinit(a);
    try std.testing.expect(!try e.fillStreamCommitted(&s, true, .{ .ptr = &p, .call = Probe.hook }));
    try std.testing.expectEqual(@as(usize, 0), p.hooks);
    while (!try e.fillStreamCommitted(&s, false, .{ .ptr = &p, .call = Probe.hook })) {}
    try std.testing.expectEqual(@as(usize, 1), p.hooks);
    try std.testing.expect(p.committed_in_draft and p.hook_in_draft);
    e.discard(&s);
}

test "capability alone does not reorder admissions without active request drafting" {
    for ([_]bool{ false, true }) |family_mtp| {
        var p: Probe = .{};
        defer p.target.deinit();
        var cfg = try Config.init(a, .{ .exact_width = 8, .hidden_rows = true, .mtp = family_mtp, .speculate = family_mtp, .drafts = 7 }, 8, 7);
        defer cfg.deinit(a);
        var clock: fake.FixedClock = .{};
        var e = Engine.init(a, &cfg, p.backend(true, false), clock.clock());
        defer e.deinit();
        var s = try sm.Stream.init(a, .{ .id = "no-draft", .prompt = &ids, .max_new = 8, .drafts = !family_mtp });
        defer s.deinit(a);
        try e.addStreamCommitted(&s, .{ .ptr = &p, .call = Probe.hook });
        try std.testing.expectEqual(@as(usize, 0), p.hooks);
        try std.testing.expectEqual(@as(usize, 0), p.drafts);
        try std.testing.expectEqual(@as(usize, 1), s.emitted().len);
        e.discard(&s);
        try std.testing.expectEqual(@as(usize, 1), p.releases);
    }
}

test "terminal admission releases its lane if a queued trace cannot resolve" {
    var p: Probe = .{};
    defer p.target.deinit();
    var cfg = try config();
    defer cfg.deinit(a);
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, p.backend(true, false), clock.clock());
    defer e.deinit();
    var log = events.Log.init(a);
    defer log.deinit();
    e.log = &log;
    const event = try log.add(&.{events.f("ev", .{ .str = "missing-token" })});
    try e.unread.append(a, .{ .event = event, .handle = 0 });
    var s = try sm.Stream.init(a, .{ .id = "trace", .prompt = &ids, .max_new = 1 });
    defer s.deinit(a);
    try std.testing.expectError(error.NoSuchField, e.addStreamCommitted(&s, .{ .ptr = &p, .call = Probe.hook }));
    try std.testing.expect(s.finished);
    try std.testing.expectEqual(@as(usize, 1), p.releases);
    try std.testing.expectEqual(@as(usize, 0), p.hooks);
    try std.testing.expectEqual(@as(usize, 0), p.drafts);
}

test "terminal trace allocation failures release once before any handoff" {
    var terminal_failures: usize = 0;
    var completed = false;
    for (0..8) |index| {
        var p: Probe = .{};
        defer p.target.deinit();
        var cfg = try config();
        defer cfg.deinit(a);
        var clock: fake.FixedClock = .{};
        var e = Engine.init(a, &cfg, p.backend(true, false), clock.clock());
        defer e.deinit();
        var fail = std.testing.FailingAllocator.init(a, .{ .fail_index = index, .resize_fail_index = 0 });
        var log = events.Log.init(fail.allocator());
        defer log.deinit();
        e.log = &log;
        var s = try sm.Stream.init(a, .{ .id = "trace", .prompt = &ids, .max_new = 1 });
        defer s.deinit(a);
        e.addStreamCommitted(&s, .{ .ptr = &p, .call = Probe.hook }) catch |err| {
            try std.testing.expectEqual(error.OutOfMemory, err);
            if (s.finished) terminal_failures += 1 else e.discard(&s);
            try std.testing.expectEqual(@as(usize, 1), p.releases);
            try std.testing.expectEqual(@as(usize, 0), p.hooks);
            continue;
        };
        try std.testing.expect(s.finished);
        try std.testing.expectEqual(@as(usize, 1), p.releases);
        try std.testing.expectEqual(@as(usize, 1), p.hooks);
        completed = true;
        break;
    }
    try std.testing.expect(terminal_failures > 0);
    try std.testing.expect(completed);
}
