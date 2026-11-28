//! KR67 regressions for post-redaction cleaning and credential-valued replacement text.
const std = @import("std");
const chat = @import("chat_cli.zig");
const fake = @import("benchmark_fake.zig");
const fixtures = @import("chat_cli_test.zig");
const Allocator = std.mem.Allocator;

const marker = "[redacted]";
const marker_encoded = "%5Bredacted%5D";
const marker_nested = "%255Bredacted%255D";
const substring_encoded = "%72%65%64%61%63%74%65%64";
const substring_nested = "%2572%2565%2564%2561%2563%2574%2565%2564";
const marker_input = marker ++ " " ++ marker_encoded ++ " " ++ marker_nested ++ " redacted " ++ substring_encoded ++ " " ++ substring_nested;
const marker_secrets = [_][]const u8{ marker, "redacted", marker_encoded, marker_nested, substring_encoded, substring_nested };
const synthesized = "prefix-[redacted]-suffix";
const two_key_input = "prefix-token-suffix";

fn absent(text: []const u8, secrets: []const []const u8) !void {
    for (secrets) |secret| try std.testing.expect(std.mem.indexOf(u8, text, secret) == null);
}

fn terminalCommand(r: *fixtures.Run, argv: []const []const u8) !u8 {
    r.out.clearRetainingCapacity();
    r.err.clearRetainingCapacity();
    r.stdin = .fixed(r.input);
    const ctx: chat.Context = .{
        .a = r.a,
        .io = std.testing.io,
        .environ = &r.env,
        .out = &r.out.writer,
        .err = &r.err.writer,
        .input = r.reader orelse &r.stdin,
        .interactive = r.interactive,
        .terminal = true,
        .self_exe = r.self_exe,
    };
    return chat.command(&ctx, argv);
}

fn postBody(f: *fake.Fake, wanted: usize) ![]const u8 {
    var seen: usize = 0;
    for (0..f.count()) |i| if (std.mem.eql(u8, f.request(i).method, "POST")) {
        if (seen == wanted) return f.request(i).body;
        seen += 1;
    };
    return error.TestUnexpectedResult;
}

fn postCount(f: *fake.Fake) usize {
    var count: usize = 0;
    for (0..f.count()) |i| count += @intFromBool(std.mem.eql(u8, f.request(i).method, "POST"));
    return count;
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

fn httpError401(a: Allocator, message: []const u8) ![]const fake.Write {
    return fake.jsonReply(a, "401 Unauthorized", try std.fmt.allocPrint(a, "{{\"error\": {{\"message\": \"{s}\"}}}}", .{message}));
}
fn sseError(a: Allocator, message: []const u8) ![]const fake.Write {
    return stream(a, try a.dupe([]const u8, &.{
        try fake.event(a, try std.fmt.allocPrint(a, "{{\"error\": {{\"message\": \"{s}\"}}}}", .{message})),
        try fake.event(a, "[DONE]"),
    }));
}

fn controlMaterials(a: Allocator) !struct { key: []u8, split: []const u8, raw: []const u8 } {
    const key = try a.alloc(u8, 256);
    for (key, 0..) |*c, i| c.* = 'a' + @as(u8, @intCast(i % 26));
    const printable = "123456789012345678901234567890123456789012345";
    const split = try std.mem.concat(a, u8, &.{ printable, key[0..128], "\\u0001", key[128..] });
    const raw = try std.mem.concat(a, u8, &.{ printable, key[0..128], "\x01", key[128..] });
    return .{ .key = key, .split = split, .raw = raw };
}

fn expectClean(texts: []const []const u8, secrets: []const []const u8) !void {
    for (texts) |text| try absent(text, secrets);
}

test "chat CLI: KR67 ESC sanitization preserves adjacent Unicode in terminal and history" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{
        try successful(a, "whole\\u001bé bracket\\u001b[é"),
        try stream(a, try a.dupe([]const u8, &.{
            try delta(a, "split\\u001b"),
            try delta(a, "é csi\\u001b["),
            try delta(a, "é"),
            try fake.event(a, "{\"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": \"stop\"}]}"),
            try fake.event(a, "[DONE]"),
        })),
        try successful(a, "done"),
    }));
    defer f.stop();
    r.interactive = true;
    r.input = "whole\nsplit\nthird\n/bye\n";
    try std.testing.expectEqual(@as(u8, 0), try terminalCommand(&r, &.{ "--url", try f.url(a, "") }));
    try std.testing.expect(std.mem.indexOf(u8, r.out.written(), "wholeé bracketé") != null);
    try std.testing.expect(std.mem.indexOf(u8, r.out.written(), "splité csié") != null);
    try std.testing.expect(std.mem.indexOf(u8, try postBody(f, 1), "whole\\u00e9 bracket\\u00e9") != null);
    try std.testing.expect(std.mem.indexOf(u8, try postBody(f, 2), "split\\u00e9 csi\\u00e9") != null);
}

test "chat CLI: KR67 control-split keys leave neither terminal streams, history nor HTTP and SSE errors" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const control = try controlMaterials(a);
    const errors = try std.mem.concat(a, u8, &.{ "A\\u0001B ", control.split });
    const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{
        try successful(a, "A\\u0001B"),
        try successful(a, control.split),
        try successful(a, "done"),
        try httpError(a, errors),
        try sseError(a, errors),
    }));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", try std.mem.concat(a, u8, &.{ "AB,", control.key }));
    r.interactive = true;
    r.input = "first\nsecond\nthird\n/bye\n";
    try std.testing.expectEqual(@as(u8, 0), try terminalCommand(&r, &.{ "--url", try f.url(a, "") }));
    var shown: std.ArrayList(u8) = .empty;
    try shown.appendSlice(a, r.out.written());
    try shown.appendSlice(a, r.err.written());
    const secrets = [_][]const u8{ "AB", control.key, control.key[0..255] };
    try expectClean(&.{ shown.items, try postBody(f, 0), try postBody(f, 1), try postBody(f, 2) }, &secrets);
    r.interactive = false;
    r.input = "hi";
    for (0..2) |_| {
        try std.testing.expectEqual(@as(u8, 1), try terminalCommand(&r, &.{ "--url", try f.url(a, "") }));
        try shown.appendSlice(a, r.out.written());
        try shown.appendSlice(a, r.err.written());
    }
    try expectClean(&.{ shown.items, try postBody(f, 3), try postBody(f, 4) }, &secrets);
}

fn discoveryRun(a: Allocator, tmp: *std.testing.TmpDir, keys: []const u8, model: []const u8, secrets: []const []const u8) !void {
    var r = try fixtures.Run.init(a, tmp);
    const models = try std.fmt.allocPrint(a, "{{\"data\": [{{\"id\": \"{s}\"}}]}}", .{model});
    const f = try fixtures.world(a, models, try a.dupe([]const fake.Write, &.{try successful(a, "done")}));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", keys);
    r.interactive = true;
    r.input = "/bye\n";
    _ = try terminalCommand(&r, &.{ "--url", try f.url(a, "") });
    var shown: std.ArrayList(u8) = .empty;
    try shown.appendSlice(a, r.out.written());
    try shown.appendSlice(a, r.err.written());
    r.interactive = false;
    r.input = "hi";
    _ = try terminalCommand(&r, &.{ "--url", try f.url(a, "") });
    try shown.appendSlice(a, r.out.written());
    try shown.appendSlice(a, r.err.written());
    try expectClean(&.{ shown.items, try postBody(f, 0) }, secrets);
}

test "chat CLI: KR67 discovered control-split AB leaves neither its banner nor follow-up POST" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try discoveryRun(a, &tmp, "AB", "A\\u0001B", &.{"AB"});
}

test "chat CLI: KR67 discovery truncation exposes neither a split 256-byte key nor its 255-byte prefix" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const control = try controlMaterials(a);
    try discoveryRun(a, &tmp, control.key, control.split, &.{ control.key, control.key[0..255] });
}

test "chat CLI: KR67 exact marker and substring keys vanish from raw and encoded streams and history" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{
        try successful(a, marker_input),
        try successful(a, marker_input),
        try successful(a, "done"),
    }));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", "[redacted],redacted");
    r.interactive = true;
    r.input = "first\nsecond\nthird\n/bye\n";
    try std.testing.expectEqual(@as(u8, 0), try terminalCommand(&r, &.{ "--url", try f.url(a, "") }));
    try expectClean(&.{ r.out.written(), r.err.written(), try postBody(f, 0), try postBody(f, 1), try postBody(f, 2) }, &marker_secrets);
}

test "chat CLI: KR67 exact marker and substring discovery suppresses the unsafe model before any POST" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const models = try std.fmt.allocPrint(a, "{{\"data\": [{{\"id\": \"{s}\"}}]}}", .{marker_input});
    const f = try fixtures.world(a, models, try a.dupe([]const fake.Write, &.{try successful(a, "unused")}));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", "[redacted],redacted");
    r.interactive = true;
    r.input = "/bye\n";
    _ = try terminalCommand(&r, &.{ "--url", try f.url(a, "") });
    try expectClean(&.{ r.out.written(), r.err.written() }, &marker_secrets);
    try std.testing.expectEqual(@as(usize, 0), postCount(f));
}

test "chat CLI: KR67 exact marker and substring keys vanish from HTTP and SSE errors" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{ try httpError(a, marker_input), try sseError(a, marker_input) }));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", "[redacted],redacted");
    r.input = "hi";
    var shown: std.ArrayList(u8) = .empty;
    const url = try f.url(a, "");
    for (0..2) |_| {
        _ = try terminalCommand(&r, &.{ "--url", url, "safe-model" });
        try shown.appendSlice(a, r.out.written());
        try shown.appendSlice(a, r.err.written());
    }
    try expectClean(&.{ shown.items, try postBody(f, 0), try postBody(f, 1) }, &marker_secrets);
}

test "chat CLI: KR67 containment runs after composing the HTTP 401 credential hint" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{try httpError401(a, "prefix")}));
    defer f.stop();
    const synthesized_hint = "TENSORFOLD_API_KEY";
    try r.env.put("TENSORFOLD_API_KEY", synthesized_hint);
    r.input = "hi";
    _ = try terminalCommand(&r, &.{ "--url", try f.url(a, ""), "safe-model" });
    try expectClean(&.{ r.out.written(), r.err.written(), try postBody(f, 0) }, &.{synthesized_hint});
}

test "chat CLI: KR67 token replacement cannot synthesize another key in streams or history" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{
        try successful(a, two_key_input),
        try successful(a, two_key_input),
        try successful(a, "done"),
    }));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", "token," ++ synthesized);
    r.interactive = true;
    r.input = "first\nsecond\nthird\n/bye\n";
    try std.testing.expectEqual(@as(u8, 0), try terminalCommand(&r, &.{ "--url", try f.url(a, "") }));
    try expectClean(&.{ r.out.written(), r.err.written(), try postBody(f, 0), try postBody(f, 1), try postBody(f, 2) }, &.{ "token", synthesized });
}

test "chat CLI: KR67 synthesized-key discovery suppresses the unsafe model before any POST" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const models = try std.fmt.allocPrint(a, "{{\"data\": [{{\"id\": \"{s}\"}}]}}", .{two_key_input});
    const f = try fixtures.world(a, models, try a.dupe([]const fake.Write, &.{try successful(a, "unused")}));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", "token," ++ synthesized);
    r.interactive = true;
    r.input = "/bye\n";
    _ = try terminalCommand(&r, &.{ "--url", try f.url(a, "") });
    try expectClean(&.{ r.out.written(), r.err.written() }, &.{ "token", synthesized });
    try std.testing.expectEqual(@as(usize, 0), postCount(f));
}

test "chat CLI: KR67 token replacement cannot synthesize another key in HTTP or SSE errors" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const f = try fixtures.world(a, fixtures.models_doc, try a.dupe([]const fake.Write, &.{ try httpError(a, two_key_input), try sseError(a, two_key_input) }));
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", "token," ++ synthesized);
    r.input = "hi";
    var shown: std.ArrayList(u8) = .empty;
    const url = try f.url(a, "");
    for (0..2) |_| {
        _ = try terminalCommand(&r, &.{ "--url", url, "safe-model" });
        try shown.appendSlice(a, r.out.written());
        try shown.appendSlice(a, r.err.written());
    }
    try expectClean(&.{ shown.items, try postBody(f, 0), try postBody(f, 1) }, &.{ "token", synthesized });
}
