//! The speech host's encode hand-off over a fake host: idle lanes answer, busy ones wait, a bad file keeps it open.
const std = @import("std");
const Queue = @import("drama_host_voice.zig").Queue;
const t = std.testing;

const Tts = struct {
    c: struct { gpa: std.mem.Allocator } = .{ .gpa = t.allocator },
    fail: ?anyerror = null,
    calls: usize = 0,

    pub fn encodeWav(tts: *Tts, wav: []const u8) ![][10]i32 {
        tts.calls += 1;
        if (tts.fail) |e| return switch (e) {
            error.NotWav => error.NotWav,
            error.UnsupportedWav => error.UnsupportedWav,
            else => error.LaunchFailed,
        };
        const codes = try tts.c.gpa.alloc([10]i32, wav.len);
        for (codes, wav) |*row, byte| row.* = @splat(byte);
        return codes;
    }
};

const Host = struct {
    io: std.Io = t.io,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    closing: bool = false,
    failed: struct { closed: bool = false } = .{},
    jobs: [2]?u8 = @splat(null),
    queue: struct { lanes: usize = 2 } = .{},
    tts: [2]Tts = @splat(.{}),
    sessions: [2]struct { t: *Tts } = undefined,
    voices: Queue = .{},

    fn lock(h: *Host) void {
        h.mutex.lockUncancelable(h.io);
    }
    fn unlock(h: *Host) void {
        h.mutex.unlock(h.io);
    }
};
const Voice = @import("drama_host_voice.zig").For(Host);

const Asked = struct {
    h: *Host,
    out: @import("engine_api").EncodeVoiceError![][10]i32 = &.{},
    fn run(a: *Asked) void {
        a.out = Voice.encode(a.h, t.allocator, "ab");
    }
};

/// One encode from another thread, answered by this one as the host loop would.
fn ask(h: *Host) ![][10]i32 {
    var asked: Asked = .{ .h = h };
    const thread = try std.Thread.spawn(.{}, Asked.run, .{&asked});
    while (true) {
        h.lock();
        if (h.voices.ask != null and Voice.step(h)) break;
        h.unlock();
        std.Thread.yield() catch {};
    }
    thread.join();
    return asked.out;
}

test "drama voice encode: an idle lane answers; a bad file keeps the engine open, a device error closes it" {
    var h: Host = .{};
    for (&h.sessions, &h.tts) |*s, *tts| s.* = .{ .t = tts };
    h.jobs[0] = 1; // lane 0 busy: lane 1 encodes
    const codes = try ask(&h);
    defer t.allocator.free(codes);
    try t.expectEqualSlices([10]i32, &.{ @splat('a'), @splat('b') }, codes);
    try t.expectEqual(@as(usize, 1), h.tts[1].calls);
    h.tts[1].fail = error.NotWav;
    try t.expectError(error.BadWav, ask(&h));
    h.tts[1].fail = error.UnsupportedWav;
    try t.expectError(error.BadWav, ask(&h));
    try t.expect(!h.failed.closed);
    h.tts[1].fail = error.LaunchFailed;
    const level = t.log_level;
    t.log_level = .err; // the closing warning is the expected outcome here
    defer t.log_level = level;
    try t.expectError(error.Failed, ask(&h));
    try t.expect(h.failed.closed);
    try t.expectError(error.Closed, Voice.encode(&h, t.allocator, "ab"));
    try t.expectEqual(@as(usize, 0), h.tts[0].calls);
}

test "drama voice encode: every lane busy keeps the ask waiting; closing answers it" {
    var h: Host = .{};
    for (&h.sessions, &h.tts) |*s, *tts| s.* = .{ .t = tts };
    h.jobs = .{ 1, 2 };
    var asked: Asked = .{ .h = &h };
    const thread = try std.Thread.spawn(.{}, Asked.run, .{&asked});
    while (true) {
        h.lock();
        defer h.unlock();
        if (h.voices.ask != null) break;
    }
    h.lock();
    try t.expect(!Voice.step(&h)); // nothing ran: the mutex is still held
    h.closing = true;
    Voice.refuse(&h);
    h.unlock();
    thread.join();
    try t.expectError(error.Closed, asked.out);
    try t.expectEqual(@as(usize, 0), h.tts[0].calls + h.tts[1].calls);
}
