//! Silence redraws on the wire, against the scripted engine of speech_stream_test.zig.
const std = @import("std");
const api = @import("engine_api");
const harness = @import("speech_stream_test.zig");
const drama_allow = @import("drama_allow.zig");
const drama_prompt = @import("drama_prompt.zig");
const quantizeInPlace = @import("speech.zig").quantizeInPlace;
const Policy = @import("speech_redraw.zig").Policy;
const Script = harness.Script;
const Served = harness.Served;
const frame = harness.frame;
const expectHeader = harness.expectHeader;

const salt = 5;

fn start(s: *Served, script: *Script, redraw: Policy, headers: []const u8, body: []const u8) !void {
    try s.open(script, .{ .served_name = "m", .model_ids = &.{"m"}, .seed_salt = salt, .redraw = redraw }, headers, "HTTP/1.1", body, 2000);
}

fn whole(comptime input: []const u8) []const u8 {
    return "{\"model\":\"m\",\"input\":\"" ++ input ++ "\",\"response_format\":\"pcm\"}";
}

/// The seed stock speech gives `input` under salt + `k`.
fn seed(input: []const u8, k: i64) !u64 {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var bytes: harness.Bytes = .{};
    return api.seedFor(try drama_prompt.encode(arena.allocator(), bytes.text(), input), salt + k);
}

/// The reply is one frame of take `k`'s fill.
fn expectTake(body: []const u8, k: usize) !void {
    var sample = [_]f32{0.5 / @as(f32, @floatFromInt(k + 1))};
    try std.testing.expectEqual(@as(usize, 2 * frame), body.len);
    try std.testing.expectEqualSlices(u8, try quantizeInPlace(&sample), body[0..2]);
}

const Reply = struct { head: []const u8, body: []const u8 };

/// Reads the whole reply and waits for the route and the engine, so a failed check never outlives them.
fn finish(s: *Served, script: *Script) !Reply {
    defer s.join(script);
    _ = try s.readHead();
    const body = try s.readRest();
    return .{ .head = s.got.items[0..s.read_at], .body = body };
}

/// Serves one whole reply and checks its status line, finish reason, take and engine submissions.
fn expectReply(script: *Script, redraw: Policy, headers: []const u8, comptime input: []const u8, reason: []const u8, take: usize, takes: usize) !void {
    var s: Served = undefined;
    try start(&s, script, redraw, headers, whole(input));
    defer s.deinit();
    const reply = try finish(&s, script);
    try expectHeader(reply.head, "HTTP/1.1 200");
    try expectHeader(reply.head, reason);
    if (redraw.count() > 0) {
        var header: [40]u8 = undefined;
        try expectHeader(reply.head, try std.fmt.bufPrint(&header, "X-Silence-Redraws: {d}\r\n", .{takes - 1}));
    } else try std.testing.expect(std.mem.indexOf(u8, reply.head, "X-Silence-Redraws") == null);
    try expectTake(reply.body, take);
    try std.testing.expectEqual(takes, script.takes);
    try std.testing.expectEqual(try seed(input, 0), script.seeds[0]);
}

test "redraws off, omitted or an explicit 0, send a silenced reply once and never read X-Deadline-S" {
    for ([_]Policy{ .{}, .{ .max = 0 } }) |off| {
        var script: Script = .{ .steps = &.{frame}, .reasons = &.{ .silence, .stop }, .inline_ = true };
        try expectReply(&script, off, "X-Deadline-S: soon\r\n", "hi", "X-Finish-Reason: silence\r\n", 0, 1);
    }
}

test "a silenced reply is redrawn with seed salt + k until a take finishes" {
    var script: Script = .{ .steps = &.{frame}, .reasons = &.{ .silence, .silence, .stop }, .inline_ = true };
    try expectReply(&script, .{ .max = 3 }, "", "hi", "X-Finish-Reason: stop\r\n", 2, 3);
    for (script.seeds[0..3], 0..) |got, k| try std.testing.expectEqual(try seed("hi", @intCast(k)), got);
}

test "the last silenced take is sent once redraws run out" {
    var script: Script = .{ .steps = &.{frame}, .reason = .silence, .inline_ = true };
    try expectReply(&script, .{ .max = 1 }, "", "hi", "X-Finish-Reason: silence\r\n", 1, 2);
}

test "a refused or failed redraw leaves the last take with audio as the reply" {
    var busy: Script = .{ .steps = &.{frame}, .reason = .silence, .busy_from = 1, .inline_ = true };
    try expectReply(&busy, .{ .max = 2 }, "", "hi", "X-Finish-Reason: silence\r\n", 0, 1);
    var failed: Script = .{ .steps = &.{frame}, .reasons = &.{ .silence, .failed }, .message = "codec failed", .inline_ = true };
    try expectReply(&failed, .{ .max = 2 }, "", "hi", "X-Finish-Reason: silence\r\n", 0, 2);
}

test "no redraw starts past the server budget or X-Deadline-S" {
    const cases = [_]struct { Policy, []const u8, usize }{
        .{ .{ .max = 1, .budget_s = 1e-9 }, "", 1 },
        .{ .{ .max = 1 }, "X-Deadline-S: 1e-9\r\n", 1 },
        .{ .{ .max = 1, .budget_s = 60 }, "X-Deadline-S: 1e-9\r\n", 1 },
        .{ .{ .max = 1, .budget_s = 1e-9 }, "X-Deadline-S: 60\r\n", 1 },
        .{ .{ .max = 1, .budget_s = 60 }, "X-Deadline-S: 60\r\n", 2 },
    };
    for (cases) |case| {
        var script: Script = .{ .steps = &.{frame}, .reason = .silence, .inline_ = true };
        try expectReply(&script, case[0], case[1], "hi", "X-Finish-Reason: silence\r\n", case[2] - 1, case[2]);
    }
}

test "a malformed X-Deadline-S is refused when redraws are on" {
    var script: Script = .{ .steps = &.{frame}, .reason = .silence, .inline_ = true };
    var s: Served = undefined;
    try start(&s, &script, .{ .max = 1 }, "X-Deadline-S: 0\r\n", whole("hi"));
    defer s.deinit();
    const reply = try finish(&s, &script);
    try expectHeader(reply.head, "HTTP/1.1 400");
    try expectHeader(reply.head, "X-Silence-Redraws: 0\r\n");
    try expectHeader(reply.body, "X-Deadline-S must be a positive number of seconds");
    try std.testing.expectEqual(@as(usize, 0), script.takes);
}

test "redraw controls filter the redrawn input; an input with nothing left is not redrawn" {
    var script: Script = .{ .steps = &.{frame}, .reasons = &.{ .silence, .stop }, .inline_ = true };
    try expectReply(&script, .{ .max = 1, .controls = try drama_allow.parse("sighs") }, "", "hi [sighs] [laughs] there", "X-Finish-Reason: stop\r\n", 1, 2);
    try std.testing.expectEqual(try seed("hi [sighs] there", 1), script.seeds[1]);
    var empty: Script = .{ .steps = &.{frame}, .reasons = &.{ .silence, .stop }, .inline_ = true };
    try expectReply(&empty, .{ .max = 1, .controls = try drama_allow.parse("") }, "", "[sighs]", "X-Finish-Reason: silence\r\n", 0, 1);
}

/// 200 characters, the fewest a stop is judged short on.
const long: []const u8 = &@as([200]u8, @splat('a'));

/// The rate at which one take, a frame at 44100 Hz, is exactly a quarter of `long`'s expected length.
const quarter: f64 = 0.25 * 200 * 44100 / @as(f64, frame);

fn short(chars_per_s: f64) Policy {
    return .{ .max = 2, .short_ratio = 0.25, .short_chars_per_s = chars_per_s };
}

test "a stop under the ratio of its expected length is redrawn with seed salt + k while redraws last" {
    var script: Script = .{ .steps = &.{frame}, .inline_ = true };
    try expectReply(&script, short(quarter / 2), "", long, "X-Finish-Reason: stop\r\n", 2, 3);
    for (script.seeds[0..3], 0..) |got, k| try std.testing.expectEqual(try seed(long, @intCast(k)), got);
}

test "a stop at or above the ratio of its expected length is sent" {
    for ([_]f64{ quarter, 2 * quarter }) |rate| {
        var script: Script = .{ .steps = &.{frame}, .inline_ = true };
        try expectReply(&script, short(rate), "", long, "X-Finish-Reason: stop\r\n", 0, 1);
    }
}

test "a length finish, or a stop of fewer than 200 characters, is never judged short" {
    var length: Script = .{ .steps = &.{frame}, .reason = .length, .inline_ = true };
    try expectReply(&length, short(quarter / 2), "", long, "X-Finish-Reason: length\r\n", 0, 1);
    var fewer: Script = .{ .steps = &.{frame}, .inline_ = true };
    try expectReply(&fewer, short(quarter / 2), "", long[1..], "X-Finish-Reason: stop\r\n", 0, 1);
}

test "a redraw is judged short by the input it spoke after its controls" {
    var script: Script = .{ .steps = &.{frame}, .inline_ = true };
    const p: Policy = .{ .max = 2, .controls = try drama_allow.parse(""), .short_ratio = 0.25, .short_chars_per_s = quarter / 2 };
    try expectReply(&script, p, "", @as([195]u8, @splat('a')) ++ " [sighs]", "X-Finish-Reason: stop\r\n", 1, 2);
}

test "without the short reply flags a short stop is sent once" {
    var script: Script = .{ .steps = &.{frame}, .inline_ = true };
    try expectReply(&script, .{ .max = 1 }, "", long, "X-Finish-Reason: stop\r\n", 0, 1);
}

test "an empty redraw list seeds the reference plain text" {
    var script: Script = .{ .steps = &.{frame}, .reasons = &.{ .silence, .stop }, .inline_ = true };
    try expectReply(&script, .{ .max = 1, .controls = try drama_allow.parse("") }, "", "[sighs] hello [unknown]", "X-Finish-Reason: stop\r\n", 1, 2);
    try std.testing.expectEqual(try seed("hello", 1), script.seeds[1]);
}

test "streams are never redrawn and report zero redraws" {
    var script: Script = .{ .steps = &.{frame}, .reasons = &.{ .silence, .stop }, .inline_ = true };
    var s: Served = undefined;
    try start(&s, &script, .{ .max = 2 }, "X-Deadline-S: 60\r\n", harness.stream_body);
    defer s.deinit();
    const reply = try finish(&s, &script);
    try expectHeader(reply.head, "HTTP/1.1 200");
    try expectHeader(reply.head, "X-Silence-Redraws: 0\r\n");
    try std.testing.expect(std.mem.endsWith(u8, reply.body, "\r\n0\r\nX-Finish-Reason: silence\r\n\r\n"));
    try std.testing.expectEqual(@as(usize, 1), script.takes);
}

test "stream deadlines are checked only with redraws enabled" {
    for ([_]Policy{ .{}, .{ .max = 0 }, .{ .max = 1 } }) |policy| {
        var script: Script = .{ .steps = &.{frame}, .inline_ = true };
        var s: Served = undefined;
        try start(&s, &script, policy, "X-Deadline-S: soon\r\n", harness.stream_body);
        defer s.deinit();
        const reply = try finish(&s, &script);
        try expectHeader(reply.head, if (policy.count() > 0) "HTTP/1.1 400" else "HTTP/1.1 200");
        if (policy.count() > 0) {
            try expectHeader(reply.head, "X-Silence-Redraws: 0\r\n");
            try std.testing.expectEqual(@as(usize, 0), script.takes);
        } else try std.testing.expect(std.mem.indexOf(u8, reply.head, "X-Silence-Redraws") == null);
    }
}

test "an enabled guard reports zero without a trip and on a request refusal" {
    var script: Script = .{ .steps = &.{frame}, .inline_ = true };
    try expectReply(&script, .{ .max = 1 }, "", "hi", "X-Finish-Reason: stop\r\n", 0, 1);
    var s: Served = undefined;
    try start(&s, &script, .{ .max = 1 }, "", "{\"model\":\"m\"}");
    defer s.deinit();
    const reply = try finish(&s, &script);
    try expectHeader(reply.head, "HTTP/1.1 400");
    try expectHeader(reply.head, "X-Silence-Redraws: 0\r\n");
}

test "a busy redraw is admitted after backoff within its deadline" {
    var script: Script = .{
        .steps = &.{frame},
        .reasons = &.{ .silence, .stop },
        .inline_ = true,
        .busy_redraws = 1,
    };
    try expectReply(&script, .{ .max = 1 }, "X-Deadline-S: 60\r\n", "hi", "X-Finish-Reason: stop\r\n", 1, 2);
    try std.testing.expectEqual(@as(usize, 1), script.busy_replies);
}

test "a busy redraw stops at its latest start without counting an unsubmitted take" {
    var script: Script = .{
        .steps = &.{frame},
        .reason = .silence,
        .inline_ = true,
        .busy_redraws = std.math.maxInt(usize),
    };
    try expectReply(&script, .{ .max = 1 }, "X-Deadline-S: 0.1\r\n", "hi", "X-Finish-Reason: silence\r\n", 0, 1);
}
