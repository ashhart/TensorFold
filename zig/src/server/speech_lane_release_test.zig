const std = @import("std");
const harness = @import("speech_stream_test.zig");
const Script = harness.Script;
const Served = harness.Served;
const LaneProbe = harness.LaneProbe;
const frame = harness.frame;
const expectHeader = harness.expectHeader;
const gpa = std.testing.allocator;
const whole_body = "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"wav\"}";

test "invalid advertised speech output releases the lane before the first write" {
    var script: Script = .{ .steps = &.{frame}, .max_frames = 0 };
    var probe: LaneProbe = .{};
    defer probe.first_bytes.deinit(gpa);
    var s: Served = undefined;
    try s.openProbed(&script, .{ .served_name = "m", .model_ids = &.{"m"} }, "", "HTTP/1.1", whole_body, 2000, &probe);
    defer s.deinit();
    s.join(&script);
    try std.testing.expect(probe.writes >= 1);
    try std.testing.expect(std.mem.startsWith(u8, probe.first_bytes.items, "HTTP/1.1 500"));
    try expectHeader(probe.first_bytes.items, "the engine advertises no usable speech output");
    try std.testing.expectEqual(@as(u32, 0), s.srv.speech_inflight.load(.acquire));
    std.debug.print("invalid output: first-write inflight={d}\n", .{probe.first_inflight});
    try std.testing.expectEqual(@as(u32, 0), probe.first_inflight);
}

test "speech submit errors release the lane before the first write" {
    const cases = .{
        .{ error.Busy, "HTTP/1.1 503", "the engine is busy; retry shortly" },
        .{ error.Closed, "HTTP/1.1 400", "the scheduler is closed" },
        .{ error.InvalidRequest, "HTTP/1.1 400", "the engine refused this speech request's prompt, frame limit or sampling" },
        .{ error.Unsupported, "HTTP/1.1 400", "this model does not support speech synthesis" },
    };
    var failed = false;
    inline for (cases) |case| {
        var script: Script = .{ .steps = &.{frame}, .submit_error = case[0] };
        var probe: LaneProbe = .{};
        defer probe.first_bytes.deinit(gpa);
        var s: Served = undefined;
        try s.openProbed(&script, .{ .served_name = "m", .model_ids = &.{"m"} }, "", "HTTP/1.1", whole_body, 2000, &probe);
        defer s.deinit();
        s.join(&script);
        try std.testing.expect(probe.writes >= 1);
        try std.testing.expect(std.mem.startsWith(u8, probe.first_bytes.items, case[1]));
        try expectHeader(probe.first_bytes.items, case[2]);
        try std.testing.expectEqual(@as(u32, 0), s.srv.speech_inflight.load(.acquire));
        std.debug.print("{s}: first-write inflight={d}\n", .{ @errorName(case[0]), probe.first_inflight });
        std.testing.expectEqual(@as(u32, 0), probe.first_inflight) catch {
            failed = true; // Exercise every submit error even when an earlier case holds the lane.
        };
    }
    try std.testing.expect(!failed);
}

test "the speech lane is free while the reply body is still being written" {
    var script: Script = .{ .steps = &.{3072 * frame}, .max_frames = 3072 };
    var s: Served = undefined;
    try s.start(&script, "HTTP/1.1", whole_body, 30000);
    defer s.deinit();
    var joined = false;
    defer if (!joined) {
        _ = std.c.shutdown(s.client, 2); // unblock the writer when an assertion fails before draining
        s.join(&script);
    };
    const head = try s.readHead();
    try expectHeader(head, "200 OK");
    try expectHeader(head, "Content-Length:");
    while (s.got.items.len - s.read_at < 1024 * 1024 + 44) {
        try std.testing.expect(try s.more(2000));
    }
    // At least 11 MB remains unwritten, so the route's deferred release cannot have run.
    try std.testing.expectEqual(@as(u32, 0), s.srv.speech_inflight.load(.acquire));
    const rest = try s.readRest();
    s.join(&script);
    joined = true;
    try std.testing.expectEqual(@as(usize, 44 + 2 * frame * 3072), rest.len);
    try std.testing.expectEqualStrings("RIFF", rest[0..4]);
    try std.testing.expectEqualSlices(u8, &.{ 0xff, 0x3f }, rest[44..46]);
    try std.testing.expectEqualSlices(u8, &.{ 0xff, 0x3f }, rest[rest.len - 2 ..]);
    try std.testing.expectEqual(@as(u32, 0), s.srv.speech_inflight.load(.acquire));
}

test "the speech lane is free while an error reply body is still being written" {
    const message = try gpa.alloc(u8, 12 * 1024 * 1024);
    defer gpa.free(message);
    @memset(message, 'x');
    var script: Script = .{ .steps = &.{}, .reason = .failed, .message = message };
    var s: Served = undefined;
    try s.start(&script, "HTTP/1.1", harness.stream_body, 30000);
    defer s.deinit();
    var joined = false;
    defer if (!joined) {
        _ = std.c.shutdown(s.client, 2); // unblock the writer when an assertion fails before draining
        s.join(&script);
    };
    const head = try s.readHead();
    try expectHeader(head, "HTTP/1.1 500");
    try expectHeader(head, "Content-Length:");
    while (s.got.items.len - s.read_at < 1024 * 1024) {
        try std.testing.expect(try s.more(2000));
    }
    // At least 11 MB remains unwritten, so the error reply's deferred release cannot have run.
    try std.testing.expectEqual(@as(u32, 0), s.srv.speech_inflight.load(.acquire));
    const rest = try s.readRest();
    s.join(&script);
    joined = true;
    try std.testing.expect(std.mem.indexOf(u8, rest, message[0..80]) != null);
    try std.testing.expectEqual(@as(u32, 0), s.srv.speech_inflight.load(.acquire));
}
