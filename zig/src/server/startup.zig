//! The model's text loads on another thread while the engine opens on this one: both come up, or neither is left open.
const std = @import("std");

/// What `both` hands back: the loaded text and the opened engine.
pub fn Both(comptime Load: type, comptime Open: type) type {
    const Text = @typeInfo(@typeInfo(Load).@"fn".return_type.?).error_union.payload;
    const Engine = @typeInfo(@typeInfo(@typeInfo(Open).@"fn".return_type.?).error_union.payload).optional.child;
    return struct { text: Text, engine: Engine };
}

/// Load text on another thread; its error wins, then open's error or refusal, cleaning whichever succeeded.
pub fn both(io: std.Io, load: anytype, load_args: std.meta.ArgsTuple(@TypeOf(load)), open: anytype, open_args: std.meta.ArgsTuple(@TypeOf(open))) !?Both(@TypeOf(load), @TypeOf(open)) {
    var loading = io.async(load, load_args);
    const engine = @call(.auto, open, open_args) catch |e| {
        const text = try loading.await(io);
        text.deinit();
        return e;
    } orelse {
        const text = try loading.await(io);
        text.deinit();
        return null;
    };
    const text = loading.await(io) catch |e| {
        engine.close(engine.ctx);
        return e;
    };
    return .{ .text = text, .engine = engine };
}

const Fake = struct {
    var closed: u32 = 0;
    var freed: u32 = 0;
    var text: Text = .{};
    const Text = struct {
        fn deinit(_: *Text) void {
            freed += 1;
        }
    };
    const Engine = struct { ctx: *anyopaque, close: *const fn (*anyopaque) void };
    const Outcome = enum { ok, fails, refuses };

    fn reset() void {
        closed = 0;
        freed = 0;
    }

    fn close(_: *anyopaque) void {
        closed += 1;
    }

    fn load(io: std.Io, ms: i64, fails: bool) error{TextFailed}!*Text {
        std.Io.sleep(io, .fromMilliseconds(ms), .awake) catch {};
        return if (fails) error.TextFailed else &text;
    }

    fn open(io: std.Io, ms: i64, outcome: Outcome) error{EngineFailed}!?Engine {
        std.Io.sleep(io, .fromMilliseconds(ms), .awake) catch {};
        return switch (outcome) {
            .ok => .{ .ctx = &text, .close = close },
            .fails => error.EngineFailed,
            .refuses => null,
        };
    }
};

test "a text that fails after the engine opened closes the engine" {
    const io = std.testing.io;
    Fake.reset();
    try std.testing.expectError(error.TextFailed, both(io, Fake.load, .{ io, 30, true }, Fake.open, .{ io, 0, .ok }));
    try std.testing.expectEqual(@as(u32, 1), Fake.closed);
    try std.testing.expectEqual(@as(u32, 0), Fake.freed);
}

test "an engine that fails or refuses after the text loaded frees the text" {
    const io = std.testing.io;
    Fake.reset();
    try std.testing.expectError(error.EngineFailed, both(io, Fake.load, .{ io, 0, false }, Fake.open, .{ io, 30, .fails }));
    try std.testing.expectEqual(@as(u32, 1), Fake.freed);
    try std.testing.expect((try both(io, Fake.load, .{ io, 0, false }, Fake.open, .{ io, 30, .refuses })) == null);
    try std.testing.expectEqual(@as(u32, 2), Fake.freed);
    try std.testing.expectEqual(@as(u32, 0), Fake.closed);
}

test "both failing reports the text's error, as when it loaded first; both up hands both back" {
    const io = std.testing.io;
    Fake.reset();
    try std.testing.expectError(error.TextFailed, both(io, Fake.load, .{ io, 0, true }, Fake.open, .{ io, 0, .fails }));
    try std.testing.expectError(error.TextFailed, both(io, Fake.load, .{ io, 0, true }, Fake.open, .{ io, 0, .refuses }));
    const up = (try both(io, Fake.load, .{ io, 10, false }, Fake.open, .{ io, 0, .ok })).?;
    try std.testing.expectEqual(&Fake.text, up.text);
    try std.testing.expectEqual(@as(u32, 0), Fake.closed + Fake.freed);
}
