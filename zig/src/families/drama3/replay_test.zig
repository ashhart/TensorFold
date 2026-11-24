//! Production replay adapter exercised with deterministic frame and codec stand-ins.
const std = @import("std");
const Fake = struct {
    pub const gpu = struct {};
    pub const model = struct {
        pub const Frame = [11]i32;
    };
    pub const Sampling = struct {};
    pub const Engine = struct {
        a: std.mem.Allocator,
        forced: usize = 17,
        frame: i32 = 1,
        fork_frame: i32 = 1,
        frames_out: u64 = 0,
        c: struct { ids: struct { eos: i32 = -1 } = .{} } = .{},
        pub fn generate(e: *@This(), _: std.Io, _: []const model.Frame, _: Sampling, limit: usize, _: bool, _: usize) !struct { frames: []model.Frame } {
            const frames = try e.a.alloc(model.Frame, @min(limit, e.forced));
            for (frames) |*f| {
                f.* = @splat(1);
            }
            return .{ .frames = frames };
        }
        pub fn poison(_: *@This()) !void {}
        pub fn fork(e: *@This()) !*@This() {
            const f = try e.a.create(@This());
            f.* = e.*;
            f.frame = e.fork_frame;
            return f;
        }
        pub fn destroy(e: *@This()) void {
            e.a.destroy(e);
        }
        pub fn prefill(_: *@This(), _: []const model.Frame, _: ?*anyopaque) !void {}
        pub fn frameStep(_: *@This(), _: Sampling, _: bool, _: ?*anyopaque) !void {}
        pub fn download(e: *@This(), _: u64, out: []u8) !void {
            const f: model.Frame = @splat(e.frame);
            @memcpy(out, std.mem.asBytes(&f));
        }
    };
    const Codec = struct {
        pub fn reset(_: *@This()) !void {}
        pub fn poison(_: *@This()) !void {}
    };
    pub const tts = struct {
        pub const Tts = struct {
            a: std.mem.Allocator,
            c: Codec = .{},
            max_chunk: usize = 16,
            saw16: bool = false,
            saw_tail: bool = false,
            /// A four- or sixteen-frame decode's sample; chunk 1 decodes write 0.
            pcm4: f32 = 0,
            pcm16: f32 = 0,
            pub fn flush(t: *@This(), from: usize, to: usize, sink: anytype) !void {
                var at = from;
                while (at < to) {
                    const n = @min(t.max_chunk, to - at);
                    t.saw16 = t.saw16 or n == 16;
                    t.saw_tail = t.saw_tail or (from == 16 and n == 1);
                    const pcm: [16 * 2048]f32 = @splat(if (n == 16) t.pcm16 else if (n == 4) t.pcm4 else 0);
                    try sink.audio(sink.ctx, pcm[0 .. n * 2048]);
                    at += n;
                }
            }
            pub fn fork(t: *@This(), _: *Engine) !*@This() {
                const f = try t.a.create(@This());
                f.* = t.*;
                return f;
            }
            pub fn destroy(t: *@This()) void {
                t.a.destroy(t);
            }
        };
    };
};
const Replay = @import("replay.zig").Replay(Fake);
test "drama replay: refuses short fixtures and insufficient codec capacity" {
    var e: Fake.Engine = .{ .a = std.testing.allocator, .forced = 4 };
    var t: Fake.tts.Tts = .{ .a = std.testing.allocator };
    const io = std.testing.io;
    try std.testing.expectError(error.ShortReplayFixture, Replay.verify(e.a, io, &e, &t, &.{}, .{}, false));
    e.forced = 17;
    t.max_chunk = 1;
    try std.testing.expectError(error.ReplayCodecCapacity, Replay.verify(e.a, io, &e, &t, &.{}, .{}, false));
}
test "drama replay: actually decodes a sixteen-frame chunk and its one-frame tail" {
    var e: Fake.Engine = .{ .a = std.testing.allocator };
    var t: Fake.tts.Tts = .{ .a = std.testing.allocator };
    try Replay.verify(e.a, std.testing.io, &e, &t, &.{}, .{}, false);
    try std.testing.expect(t.saw16 and t.saw_tail);
}
test "drama replay: a codec chunk 16 PCM difference is PCM-only once every exact check passed" {
    var e: Fake.Engine = .{ .a = std.testing.allocator };
    var t: Fake.tts.Tts = .{ .a = std.testing.allocator, .pcm16 = 1.0 / 32768.0 };
    try std.testing.expectError(error.ReplayPcmOnly, Replay.verify(e.a, std.testing.io, &e, &t, &.{}, .{}, false));
}
test "drama replay: an over-limit chunk 4 PCM difference fails despite in-bound chunk 16" {
    var e: Fake.Engine = .{ .a = std.testing.allocator };
    var t: Fake.tts.Tts = .{ .a = std.testing.allocator, .pcm4 = 0.25, .pcm16 = 1.0 / 32768.0 };
    try std.testing.expectError(error.ReplayPcmBound, Replay.verify(e.a, std.testing.io, &e, &t, &.{}, .{}, false));
}

test "drama replay: an over-limit chunk 16 PCM difference fails" {
    var e: Fake.Engine = .{ .a = std.testing.allocator };
    var t: Fake.tts.Tts = .{ .a = std.testing.allocator, .pcm16 = 0.25 };
    try std.testing.expectError(error.ReplayPcmBound, Replay.verify(e.a, std.testing.io, &e, &t, &.{}, .{}, false));
}
test "drama replay: a forked frame mismatch fails behind a codec chunk 16 PCM difference" {
    var e: Fake.Engine = .{ .a = std.testing.allocator, .fork_frame = 2 };
    var t: Fake.tts.Tts = .{ .a = std.testing.allocator, .pcm16 = 0.25 };
    try std.testing.expectError(error.ReplayMismatch, Replay.verify(e.a, std.testing.io, &e, &t, &.{}, .{}, false));
}
