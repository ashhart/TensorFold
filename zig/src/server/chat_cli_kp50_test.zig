//! KP50 host regressions for truncation, streamed hold-back and encoded credentials.
const std = @import("std");
const fake = @import("benchmark_fake.zig");
const fixtures = @import("chat_cli_test.zig");
const Allocator = std.mem.Allocator;

const ascii_key = "fixture-ascii/token+part";
const ascii_encoded = "fixture-ascii%2Ftoken%2Bpart";
const latin_key = "fixture-caf\u{e9}/ol\u{f1}";
const latin_encoded = "fixture-caf%E9%2Fol%F1";

fn postBody(f: *fake.Fake, wanted: usize) ![]const u8 {
    var seen: usize = 0;
    for (0..f.count()) |i| if (std.mem.eql(u8, f.request(i).method, "POST")) {
        if (seen == wanted) return f.request(i).body;
        seen += 1;
    };
    return error.TestUnexpectedResult;
}

fn absent(text: []const u8, secrets: []const []const u8) !void {
    for (secrets) |secret| try std.testing.expect(std.mem.indexOf(u8, text, secret) == null);
}

fn stream(a: Allocator, parts: []const []const u8) ![]const fake.Write {
    var bytes: std.ArrayList(u8) = .empty;
    try bytes.appendSlice(a, try fake.headOf(a, "200 OK", "text/event-stream"));
    for (parts) |part| try bytes.appendSlice(a, part);
    return a.dupe(fake.Write, &.{.{ .bytes = bytes.items }});
}

fn delta(a: Allocator, content: []const u8) ![]const u8 {
    return fake.event(a, try std.fmt.allocPrint(a, "{{\"choices\": [{{\"index\": 0, \"delta\": {{\"content\": \"{s}\"}}, \"finish_reason\": null}}]}}", .{content}));
}

fn successful(a: Allocator, content: []const u8) ![]const fake.Write {
    return stream(a, try a.dupe([]const u8, &.{
        try delta(a, content),
        try fake.event(a, "{\"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": \"stop\"}]}"),
        try fake.event(a, "[DONE]"),
    }));
}

fn httpError(a: Allocator, message: []const u8) ![]const fake.Write {
    return fake.jsonReply(a, "500 Internal Server Error", try std.fmt.allocPrint(a, "{{\"error\": {{\"message\": \"{s}\"}}}}", .{message}));
}

fn sseError(a: Allocator, message: []const u8) ![]const fake.Write {
    return stream(a, try a.dupe([]const u8, &.{
        try fake.event(a, try std.fmt.allocPrint(a, "{{\"error\": {{\"message\": \"{s}\"}}}}", .{message})),
        try fake.event(a, "[DONE]"),
    }));
}

const Bodies = struct { plain: []const u8, json: []const u8 };
const message_head = "{\"error\":{\"message\":\"";

fn truncationBodies(a: Allocator, controls: []const u8, key: []const u8) !Bodies {
    return .{
        .plain = try std.mem.concat(a, u8, &.{ controls, key }),
        .json = try std.mem.concat(a, u8, &.{ message_head, controls, key, "\"}}" }),
    };
}

/// A declared body whose first write stops short of the 16 KiB capture until after its 100 ms deadline.
fn timedError(a: Allocator, body: []const u8, kept: usize) ![]const fake.Write {
    const head = try std.fmt.allocPrint(a, "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{body.len});
    const writes = try a.alloc(fake.Write, 2);
    writes[0] = .{ .bytes = try std.mem.concat(a, u8, &.{ head, body[0..kept] }), .pause_ms = 300 };
    writes[1] = .{ .bytes = body[kept..] };
    return writes;
}

test "chat CLI: KP50 fixed-length plain and JSON-message bodies expose no truncated key prefix" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const key = try a.alloc(u8, 256);
    @memset(key, 'K');
    const controls = try a.alloc(u8, 16_129);
    @memset(controls, 1);
    const bodies = try truncationBodies(a, controls, key);
    const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{
        try fake.jsonReply(a, "500 Internal Server Error", bodies.plain),
        try fake.jsonReply(a, "500 Internal Server Error", bodies.json),
    }));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", key);
    r.input = "hi";
    var shown: std.ArrayList(u8) = .empty;
    for (0..2) |_| {
        try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "--url", try f.url(a, "") }));
        try shown.appendSlice(a, r.out.written());
        try shown.appendSlice(a, r.err.written());
    }
    try absent(shown.items, &.{ key[0..255], key[0..128] });
}

test "chat CLI: KP50 genuinely timed plain and JSON-message bodies expose no retained key prefix" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const key = try a.alloc(u8, 256);
    @memset(key, 'K');
    const plain = key;
    const json = try std.mem.concat(a, u8, &.{ message_head, key, "\"}}" });
    const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{
        try timedError(a, plain, 128),
        try timedError(a, json, message_head.len + 128),
    }));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", key);
    r.input = "hi";
    var shown: std.ArrayList(u8) = .empty;
    for (0..2) |_| {
        try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "--url", try f.url(a, "") }));
        try shown.appendSlice(a, r.out.written());
        try shown.appendSlice(a, r.err.written());
    }
    try absent(shown.items, &.{key[0..128]});
}

test "chat CLI: KP50 a successful final ambiguous SSE prefix stays out of output and history" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const prefix = "fixture-secre";
    const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{
        try successful(a, prefix),
        try successful(a, "next"),
    }));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", "fixture-secret");
    r.interactive = true;
    r.input = "first\nsecond\n/bye\n";
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "--url", try f.url(a, "") }));
    try absent(r.out.written(), &.{prefix});
    try absent(r.err.written(), &.{prefix});
    try absent(try postBody(f, 1), &.{prefix});
}

test "chat CLI: KP50 error-event and truncated-EOF replies never flush an ambiguous prefix" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const prefix = "fixture-secre";
    const failing = try stream(a, try a.dupe([]const u8, &.{
        try delta(a, prefix),
        try fake.event(a, "{\"error\": {\"message\": \"generation failed\"}}"),
        try fake.event(a, "[DONE]"),
    }));
    const truncated = try stream(a, try a.dupe([]const u8, &.{try delta(a, prefix)}));
    const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{ failing, truncated }));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", "fixture-secret");
    r.input = "hi";
    var shown: std.ArrayList(u8) = .empty;
    for (0..2) |_| {
        try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "--url", try f.url(a, "") }));
        try shown.appendSlice(a, r.out.written());
        try shown.appendSlice(a, r.err.written());
    }
    try absent(shown.items, &.{prefix});
}

test "chat CLI: KP50 a shorter key never releases a longer key suffix across deltas or history" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const split = try stream(a, try a.dupe([]const u8, &.{
        try delta(a, "fixture"),
        try delta(a, "-secret"),
        try fake.event(a, "{\"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": \"stop\"}]}"),
        try fake.event(a, "[DONE]"),
    }));
    const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{ split, try successful(a, "next") }));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", "fixture,fixture-secret");
    r.interactive = true;
    r.input = "first\nsecond\n/bye\n";
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "--url", try f.url(a, "") }));
    try absent(r.out.written(), &.{ "-secret", "[redacted]-secret" });
    try absent(r.err.written(), &.{ "-secret", "[redacted]-secret" });
    try absent(try postBody(f, 1), &.{ "-secret", "[redacted]-secret" });
}

test "chat CLI: KP50 literal-percent keys consume their full encoded span in content and history" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{ try successful(a, "%2525"), try successful(a, "a%25%62"), try successful(a, "done") }));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", "%25,a%25b");
    r.interactive = true;
    r.input = "first\nsecond\nthird\n/bye\n";
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "--url", try f.url(a, "") }));
    try absent(r.out.written(), &.{ "%2525", "a%25%62", "%62", "[redacted]25" });
    try absent(r.err.written(), &.{ "%2525", "a%25%62", "%62", "[redacted]25" });
    try absent(try postBody(f, 1), &.{ "%2525", "[redacted]25" });
    try absent(try postBody(f, 2), &.{ "%2525", "a%25%62", "%62", "[redacted]25" });
}

test "chat CLI: KP50 percent-encoded credentials are absent from HTTP and SSE errors" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{
        try httpError(a, ascii_encoded),
        try httpError(a, latin_encoded),
        try httpError(a, "%2525"),
        try httpError(a, "a%25%62"),
        try sseError(a, ascii_encoded),
        try sseError(a, latin_encoded),
        try sseError(a, "%2525"),
        try sseError(a, "a%25%62"),
    }));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", try std.mem.concat(a, u8, &.{ ascii_key, ",", latin_key, ",%25,a%25b" }));
    r.input = "hi";
    var shown: std.ArrayList(u8) = .empty;
    for (0..8) |_| {
        try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "--url", try f.url(a, "") }));
        try shown.appendSlice(a, r.out.written());
        try shown.appendSlice(a, r.err.written());
    }
    try absent(shown.items, &.{ ascii_encoded, latin_encoded, "%2525", "a%25%62", "%62", "[redacted]25" });
}

test "chat CLI: KP50 discovered encoded IDs leave neither the banner nor the follow-up POST" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const model = try std.mem.concat(a, u8, &.{ ascii_encoded, "|", latin_encoded, "|%2525|a%25%62" });
    const models = try std.fmt.allocPrint(a, "{{\"data\": [{{\"id\": \"{s}\"}}]}}", .{model});
    const f = try fixtures.world(a, models, try a.dupe([]const fake.Write, &.{try successful(a, "ok")}));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", try std.mem.concat(a, u8, &.{ ascii_key, ",", latin_key, ",%25,a%25b" }));
    var shown: std.ArrayList(u8) = .empty;
    r.interactive = true;
    r.input = "/bye\n";
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "--url", try f.url(a, "") }));
    try shown.appendSlice(a, r.out.written());
    try shown.appendSlice(a, r.err.written());
    r.interactive = false;
    r.input = "hi";
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "--url", try f.url(a, "") }));
    try shown.appendSlice(a, r.out.written());
    try shown.appendSlice(a, r.err.written());
    try absent(shown.items, &.{ ascii_encoded, latin_encoded, "%2525", "a%25%62", "%62", "[redacted]25" });
    try absent(try postBody(f, 0), &.{ ascii_encoded, latin_encoded, "%2525", "a%25%62", "%62", "[redacted]25" });
}
