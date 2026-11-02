//! A stream that finishes on its first token releases its lane even when its trace fails (the default open path).
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
    releases: usize = 0,

    fn backend(p: *Probe) be.Backend {
        p.table = p.target.backend().vtable.*;
        p.table.release = release;
        return .{ .ptr = &p.target, .vtable = &p.table };
    }
    fn release(ptr: *anyopaque, s: *sm.Stream) void {
        const t: *fake.Fake = @ptrCast(@alignCast(ptr));
        const p: *Probe = @fieldParentPtr("target", t);
        p.releases += 1;
        p.target.backend().release(s);
    }
};

fn config() !Config {
    return Config.init(a, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
}

test "a first-token finish releases its lane if a queued trace cannot resolve" {
    var p: Probe = .{};
    defer p.target.deinit();
    var cfg = try config();
    defer cfg.deinit(a);
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, p.backend(), clock.clock());
    defer e.deinit();
    var log = events.Log.init(a);
    defer log.deinit();
    e.log = &log;
    const event = try log.add(&.{events.f("ev", .{ .str = "missing-token" })});
    try e.unread.append(a, .{ .event = event, .handle = 0 });
    var s = try sm.Stream.init(a, .{ .id = "trace", .prompt = &ids, .max_new = 1 });
    defer s.deinit(a);
    try std.testing.expectError(error.NoSuchField, e.addStream(&s));
    try std.testing.expect(s.finished);
    try std.testing.expectEqual(@as(usize, 1), p.releases);
    try std.testing.expectEqual(@as(usize, 0), e.activeCount());
}

test "a first-token finish releases its lane once whichever trace allocation fails" {
    var finished_failures: usize = 0;
    var completed = false;
    for (0..16) |index| {
        var p: Probe = .{};
        defer p.target.deinit();
        var cfg = try config();
        defer cfg.deinit(a);
        var clock: fake.FixedClock = .{};
        var e = Engine.init(a, &cfg, p.backend(), clock.clock());
        defer e.deinit();
        var fail = std.testing.FailingAllocator.init(a, .{ .fail_index = index, .resize_fail_index = 0 });
        var log = events.Log.init(fail.allocator());
        defer log.deinit();
        e.log = &log;
        var s = try sm.Stream.init(a, .{ .id = "trace", .prompt = &ids, .max_new = 1 });
        defer s.deinit(a);
        e.addStream(&s) catch |err| {
            try std.testing.expectEqual(error.OutOfMemory, err);
            // the host discards only an unfinished stream, so a finished one must already be released
            if (s.finished) finished_failures += 1 else e.discard(&s);
            try std.testing.expectEqual(@as(usize, 1), p.releases);
            continue;
        };
        try std.testing.expect(s.finished);
        try std.testing.expectEqual(@as(usize, 1), p.releases);
        completed = true;
        break;
    }
    try std.testing.expect(completed);
    try std.testing.expect(finished_failures > 0);
}
