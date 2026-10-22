//! The register guard on synthetic tones: the measure alone, then whole replies against the scripted engine.
const std = @import("std");
const api = @import("engine_api");
const harness = @import("speech_stream_test.zig");
const drama_prompt = @import("drama_prompt.zig");
const quantizeInPlace = @import("speech.zig").quantizeInPlace;
const Policy = @import("speech_redraw.zig").Policy;
const Rule = @import("speech_register.zig").Rule;
const jump = @import("speech_register.zig").jump;
const Allocator = std.mem.Allocator;

/// Seconds, then the pitch a part glides from and to; 0 Hz is a rest.
const Part = struct { f64, f64, f64 };
const rest: Part = .{ 0.3, 0, 0 };
const rule: Rule = .{ .st = 5, .high = 30, .high_hz = 250 };

/// Four falling phrases near 125 Hz, the first opening at `first` Hz and the third at `third`.
fn reply(first: f64, third: f64) [7]Part {
    return .{ .{ 0.6, first, first * 0.88 }, rest, .{ 0.6, 125, 110 }, rest, .{ 0.6, third, third * 0.85 }, rest, .{ 0.6, 128, 112 } };
}

/// The parts at `rate`: three harmonics of each glide, its phase the integral of its frequency.
fn tones(a: Allocator, rate: f64, parts: []const Part) ![]f32 {
    var out: std.ArrayList(f32) = .empty;
    for (parts) |part| {
        const seconds, const from, const to = part;
        for (0..@intFromFloat(seconds * rate)) |i| {
            const t = @as(f64, @floatFromInt(i)) / rate;
            const phase = 2 * std.math.pi * (from * t + (to - from) * t * t / (2 * seconds));
            try out.append(a, @floatCast(0.3 * (@sin(phase) + @sin(2 * phase) / 2 + @sin(3 * phase) / 3)));
        }
    }
    return out.items;
}

fn measure(a: Allocator, parts: []const Part) !?f64 {
    return jump(a, try tones(a, 44100, parts), 44100, rule);
}

test "the register rule is off unless every threshold is set, each in range" {
    try std.testing.expect(!(Rule{}).on() and rule.on());
    try (Rule{}).validate();
    try rule.validate();
    for ([_]Rule{ .{ .st = 5 }, .{ .high = 30, .high_hz = 250 }, .{ .st = 5, .high = 30 } }) |r|
        try std.testing.expectError(error.MissingRegisterThreshold, r.validate());
    for ([_]f64{ 0, -1, std.math.inf(f64), std.math.nan(f64) }) |bad| {
        try std.testing.expectError(error.BadRegisterThreshold, (Rule{ .st = bad, .high = 30, .high_hz = 250 }).validate());
        try std.testing.expectError(error.BadRegisterThreshold, (Rule{ .st = 5, .high = bad, .high_hz = 250 }).validate());
    }
    for ([_]f64{ 74, 601, std.math.nan(f64) }) |bad| try std.testing.expectError(error.BadRegisterThreshold, (Rule{ .st = 5, .high = 30, .high_hz = bad }).validate());
    try std.testing.expectError(error.BadRegisterThreshold, (Rule{ .st = 5, .high = 101, .high_hz = 250 }).validate());
}

test "a phrase opening far above the others trips; a flat reply, a high first phrase or a jump under the line does not" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // the third phrase opens near 319 Hz, 0.125 s into its glide; the others' openings have a median near 124.5 Hz
    try std.testing.expectApproxEqAbs(12 * std.math.log2(319.0 / 124.5), (try measure(a, &reply(130, 330))).?, 0.3);
    const flat = try tones(a, 44100, &reply(130, 127));
    try std.testing.expectEqual(@as(?f64, 0), try jump(a, flat, 44100, rule));
    try std.testing.expectEqual(@as(?f64, 0), try jump(a, flat, 44100, .{ .st = 5, .high = 30, .high_hz = 75 }));
    try std.testing.expectEqual(@as(?f64, 0), try measure(a, &reply(330, 127)));
    try std.testing.expectEqual(@as(?f64, 0), try measure(a, &reply(130, 220)));
    // a high blip voiced for less than an opening is no phrase
    try std.testing.expectEqual(@as(?f64, 0), try measure(a, &(comptime reply(130, 127) ++ [_]Part{ rest, .{ 0.2, 400, 400 } })));
    // a hum far under the peak leaves the rests silent
    const hummed = try tones(a, 44100, &reply(130, 330));
    for (hummed, 0..) |*x, i| x.* += @floatCast(0.003 * @sin(2 * std.math.pi * 300 * @as(f64, @floatFromInt(i)) / 44100));
    try std.testing.expect((try jump(a, hummed, 44100, rule)).? > 0);
}

test "steady phrases 3 and 7 semitones up read the larger jump across the pitch range and sample rates" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]f64{ 16000, 24000, 44100 }) |rate| for ([_]f64{ 90, 200, 380 }) |hz| {
        const up3 = hz * std.math.pow(f64, 2, 3.0 / 12.0);
        const up7 = hz * std.math.pow(f64, 2, 7.0 / 12.0);
        const parts = [_]Part{ .{ 0.6, hz, hz }, rest, .{ 0.6, up3, up3 }, rest, .{ 0.6, hz, hz }, rest, .{ 0.6, up7, up7 } };
        const got = try jump(a, try tones(a, rate, &parts), @intFromFloat(rate), .{ .st = 2, .high = 30, .high_hz = 75 });
        try std.testing.expectApproxEqAbs(7, got.?, 0.05);
    };
}

test "non-finite audio is refused and audio under 8 kHz is not measured" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectError(error.NonFinite, jump(a, &.{ 0.1, std.math.nan(f32) }, 44100, rule));
    try std.testing.expectEqual(null, try jump(a, try tones(a, 4000, &reply(130, 330)), 4000, rule));
}

const salt = 5;

/// The seed stock speech gives "hi" under salt + `k`.
fn seed(a: Allocator, k: i64) !u64 {
    var bytes: harness.Bytes = .{};
    return api.seedFor(try drama_prompt.encode(a, bytes.text(), "hi"), salt + k);
}

/// Serves `audio` as successive takes of one whole reply; the reply is take `sent`, after `takes` seeded salt + k.
fn expectSent(a: Allocator, audio: []const []const f32, reasons: []const api.Reason, redraw: Policy, headers: []const u8, sent: usize, takes: usize) !void {
    return expectSentWith(a, audio, reasons, redraw, headers, sent, takes, false, null);
}

fn expectSentWith(a: Allocator, audio: []const []const f32, reasons: []const api.Reason, redraw: Policy, headers: []const u8, sent: usize, takes: usize, guard: bool, expected: ?[]const u8) !void {
    const n = [_]usize{audio[0].len};
    var script: harness.Script = .{ .steps = &n, .audio = audio, .reasons = reasons, .max_frames = @intCast(n[0] / harness.frame + 1), .inline_ = true };
    var s: harness.Served = undefined;
    try s.open(&script, .{ .served_name = "m", .model_ids = &.{"m"}, .seed_salt = salt, .redraw = redraw, .silence_guard = guard }, headers, "HTTP/1.1", "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"pcm\"}", 2000);
    defer s.deinit();
    const body = body: {
        defer s.join(&script);
        _ = try s.readHead();
        break :body try s.readRest();
    };
    try harness.expectHeader(s.got.items[0..s.read_at], "HTTP/1.1 200");
    if (expected) |value| try harness.expectHeader(s.got.items[0..s.read_at], value);
    if (!redraw.register.on()) try std.testing.expect(std.mem.indexOf(u8, s.got.items[0..s.read_at], "X-Register-") == null);
    try std.testing.expectEqualSlices(u8, try quantizeInPlace(try a.dupe(f32, audio[sent])), body);
    try std.testing.expectEqual(takes, script.takes);
    for (script.seeds[0..takes], 0..) |got, k| try std.testing.expectEqual(try seed(a, @intCast(k)), got);
}

test "a high-onset take is redrawn once and the take with the smaller jump is sent" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const high = try tones(a, 44100, &reply(130, 330));
    const lower = try tones(a, 44100, &reply(130, 290));
    const flat = try tones(a, 44100, &reply(130, 127));
    const on: Policy = .{ .register = rule };
    try expectSent(a, &.{ high, lower }, &.{}, on, "", 1, 2);
    try expectSent(a, &.{ lower, high }, &.{}, on, "", 0, 2);
    try expectSent(a, &.{ high, flat }, &.{ .stop, .silence }, on, "", 0, 2);
    try expectSent(a, &.{flat}, &.{}, on, "", 0, 1);
    try expectSent(a, &.{ high, flat }, &.{}, on, "X-Deadline-S: 1e-9\r\n", 0, 1);
}

test "an unmeasured redraw keeps the tripping first take; an unmeasured first take does not trip" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const high = try tones(a, 44100, &reply(130, 330));
    const zero = try a.alloc(f32, high.len);
    @memset(zero, 0);
    const single = try tones(a, 44100, &.{.{ 3.3, 130, 130 }});
    const flat = try tones(a, 44100, &reply(130, 127));
    const on: Policy = .{ .register = rule };
    for ([_][]const f32{ zero, single }) |unmeasured| {
        try std.testing.expectEqual(null, try jump(a, unmeasured, 44100, rule));
        try expectSent(a, &.{ high, unmeasured }, &.{ .stop, .stop }, on, "", 0, 2);
        try expectSent(a, &.{unmeasured}, &.{.stop}, on, "", 0, 1);
    }
    try expectSent(a, &.{ high, flat }, &.{ .stop, .stop }, on, "", 1, 2);
}

test "the register guard measures the take silence redraws leave, and off sends the first take unmeasured" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const high = try tones(a, 44100, &reply(130, 330));
    const flat = try tones(a, 44100, &reply(130, 127));
    try expectSent(a, &.{ flat, high, flat }, &.{ .silence, .stop, .stop }, .{ .max = 1, .register = rule }, "", 2, 3);
    try expectSent(a, &.{ high, flat }, &.{}, .{}, "X-Deadline-S: soon\r\n", 0, 1);
}

test "register metadata counts admission not selection and latches trips even without budget" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const high = try tones(a, 44100, &reply(130, 330));
    const flat = try tones(a, 44100, &reply(130, 127));
    const lower = try tones(a, 44100, &reply(130, 290));
    const on: Policy = .{ .register = rule };
    const used = "X-Register-Trip: 1\r\nX-Register-Redraws: 1\r\n";
    try expectSentWith(a, &.{ high, flat }, &.{}, on, "", 1, 2, false, used);
    try expectSentWith(a, &.{ lower, high }, &.{}, on, "", 0, 2, false, used);
    try expectSentWith(a, &.{high}, &.{}, on, "X-Deadline-S: 1e-9\r\n", 0, 1, false, "X-Register-Trip: 1\r\nX-Register-Redraws: 0\r\n");
    try expectSentWith(a, &.{flat}, &.{}, on, "", 0, 1, false, "X-Register-Trip: 0\r\nX-Register-Redraws: 0\r\n");
    try expectSentWith(a, &.{ flat, high, flat }, &.{ .silence, .stop, .stop }, .{ .max = 1, .register = rule }, "", 2, 3, true, "X-Silence-Redraws: 1\r\nX-Register-Trip: 1\r\nX-Register-Redraws: 1\r\n");
}

test "register selection keeps ties, accepts a lower length take, and rejects silence" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const high = try tones(a, 44100, &reply(130, 330));
    const flat = try tones(a, 44100, &reply(130, 127));
    const on: Policy = .{ .register = rule };
    try expectSentWith(a, &.{ high, high }, &.{ .stop, .length }, on, "", 0, 2, true, "X-Finish-Reason: stop\r\n");
    try expectSentWith(a, &.{ high, flat }, &.{ .stop, .length }, on, "", 1, 2, true, "X-Finish-Reason: length\r\n");
    try expectSentWith(a, &.{ high, flat }, &.{ .stop, .silence }, on, "", 0, 2, true, "X-Finish-Reason: stop\r\n");
}

test "an unmeasured register redraw can win only with the silence guard on and no silence finish" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const high = try tones(a, 44100, &reply(130, 330));
    const unmeasured = try a.alloc(f32, high.len);
    @memset(unmeasured, 0);
    const on: Policy = .{ .register = rule };
    const used = "X-Register-Trip: 1\r\nX-Register-Redraws: 1\r\n";
    try expectSentWith(a, &.{ high, unmeasured }, &.{ .stop, .stop }, on, "", 1, 2, true, used);
    try expectSentWith(a, &.{ high, unmeasured }, &.{ .stop, .silence }, on, "", 0, 2, true, used);
    try expectSentWith(a, &.{ high, unmeasured }, &.{ .stop, .stop }, on, "", 0, 2, false, used);
}

test "register headers are absent on streams and errors, and register-only streams validate deadlines" {
    const Case = struct { body: []const u8, headers: []const u8 = "", status: []const u8 };
    for ([_]Case{
        .{ .body = harness.stream_body, .status = "HTTP/1.1 200" },
        .{ .body = harness.stream_body, .headers = "X-Deadline-S: soon\r\n", .status = "HTTP/1.1 400" },
        .{ .body = "{\"model\":\"m\"}", .status = "HTTP/1.1 400" },
    }) |case| {
        var script: harness.Script = .{ .steps = &.{harness.frame}, .inline_ = true };
        var s: harness.Served = undefined;
        try s.open(&script, .{ .served_name = "m", .model_ids = &.{"m"}, .redraw = .{ .register = rule } }, case.headers, "HTTP/1.1", case.body, 2000);
        defer s.deinit();
        const head = head: {
            defer s.join(&script);
            _ = try s.readHead();
            _ = try s.readRest();
            break :head s.got.items[0..s.read_at];
        };
        try harness.expectHeader(head, case.status);
        try std.testing.expect(std.mem.indexOf(u8, head, "X-Register-") == null);
        try std.testing.expect(std.mem.indexOf(u8, head, "X-Silence-Redraws") == null);
    }
}

test "register retries reenter the silence pipeline before comparison without overwriting the held take" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const high = try tones(a, 44100, &reply(130, 330));
    const lower = try tones(a, 44100, &reply(130, 290));
    const higher = try tones(a, 44100, &reply(130, 390));
    const flat = try tones(a, 44100, &reply(130, 127));
    const on: Policy = .{ .max = 1, .register = rule };
    const used = "X-Silence-Redraws: 1\r\nX-Register-Trip: 1\r\nX-Register-Redraws: 1\r\n";
    try expectSentWith(a, &.{ high, flat, lower }, &.{ .stop, .silence, .stop }, on, "", 2, 3, true, used);
    try expectSentWith(a, &.{ high, flat, higher }, &.{ .stop, .silence, .stop }, on, "", 0, 3, true, used);
    try expectSentWith(a, &.{ flat, high, flat, lower }, &.{ .silence, .stop, .silence, .stop }, .{ .max = 2, .register = rule }, "", 3, 4, true, "X-Silence-Redraws: 2\r\nX-Register-Trip: 1\r\nX-Register-Redraws: 1\r\n");
    try expectSentWith(a, &.{ high, high }, &.{ .silence, .silence }, on, "", 1, 2, true, "X-Silence-Redraws: 1\r\nX-Register-Trip: 0\r\nX-Register-Redraws: 0\r\n");
}

test "short and register rules together: short takes go unmeasured, and a short register retry keeps the held take" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const guard: api.speech_guard.Config = .{ .floor_dbfs = -60, .seconds = 1 };
    const both = struct {
        fn policy(chars_per_s: f64) Policy {
            return .{ .max = 2, .short_ratio = 0.25, .short_chars_per_s = chars_per_s, .register = rule };
        }
    }.policy;
    try both(10).validate(guard);
    try std.testing.expectError(error.MissingRedrawCount, (Policy{ .short_ratio = 0.25, .short_chars_per_s = 10, .register = rule }).validate(.{}));
    try std.testing.expect(both(10).refusal("nemotron_h") != null and both(10).refusal("fish_omni") == null);
    const high = try tones(a, 44100, &reply(130, 330));
    const flat = try tones(a, 44100, &reply(130, 127));
    const input: [200]u8 = @splat('a');
    var bytes: harness.Bytes = .{};
    const prompt = try drama_prompt.encode(a, bytes.text(), &input);
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"input\":\"{s}\",\"response_format\":\"pcm\"}}", .{&input});
    // a 3.3 s stop of 200 characters is short at 10 a second (under 5 s), not at 20 (under 2.5 s); a length never is
    const Case = struct { []const []const f32, []const api.Reason, f64, usize, usize, []const u8 };
    for ([_]Case{
        .{ &.{ high, high, high }, &.{}, 10, 2, 3, "X-Silence-Redraws: 2\r\nX-Register-Trip: 0\r\nX-Register-Redraws: 0\r\n" },
        .{ &.{ high, flat }, &.{}, 20, 1, 2, "X-Silence-Redraws: 0\r\nX-Register-Trip: 1\r\nX-Register-Redraws: 1\r\n" },
        .{ &.{ high, high, flat, flat }, &.{ .stop, .length, .stop, .length }, 10, 3, 4, "X-Silence-Redraws: 2\r\nX-Register-Trip: 1\r\nX-Register-Redraws: 1\r\n" },
        .{ &.{ high, flat, flat, flat }, &.{ .length, .stop, .stop, .stop }, 10, 0, 4, "X-Silence-Redraws: 2\r\nX-Register-Trip: 1\r\nX-Register-Redraws: 1\r\n" },
    }) |case| {
        const audio, const reasons, const chars_per_s, const sent, const takes, const headers = case;
        const n = [_]usize{audio[0].len};
        var script: harness.Script = .{ .steps = &n, .audio = audio, .reasons = reasons, .max_frames = @intCast(n[0] / harness.frame + 1), .inline_ = true };
        var s: harness.Served = undefined;
        try s.open(&script, .{ .served_name = "m", .model_ids = &.{"m"}, .seed_salt = salt, .redraw = both(chars_per_s), .silence_guard = true }, "", "HTTP/1.1", body, 2000);
        defer s.deinit();
        const got = got: {
            defer s.join(&script);
            _ = try s.readHead();
            break :got try s.readRest();
        };
        try harness.expectHeader(s.got.items[0..s.read_at], "HTTP/1.1 200");
        try harness.expectHeader(s.got.items[0..s.read_at], headers);
        try std.testing.expectEqualSlices(u8, try quantizeInPlace(try a.dupe(f32, audio[sent])), got);
        try std.testing.expectEqual(takes, script.takes);
        for (script.seeds[0..takes], 0..) |seen, k| try std.testing.expectEqual(api.seedFor(prompt, salt + @as(i64, @intCast(k))), seen);
    }
}
