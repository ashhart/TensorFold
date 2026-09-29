const std = @import("std");
const long_request = "{\"prompt\":\"Count upwards, one number per line.\",\"max_tokens\":200000,\"ignore_eos\":true,\"temperature\":0,\"stream\":true}";

fn connect(io: std.Io, port: u16) !std.Io.net.Stream {
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", port);
    return address.connect(io, .{ .mode = .stream });
}

fn headers(io: std.Io, socket: std.Io.net.Stream, length: usize) !void {
    var buffer: [2048]u8 = undefined;
    var writer = socket.writer(io, &buffer);
    try writer.interface.print("POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{length});
    try writer.interface.flush();
}

fn write(io: std.Io, socket: std.Io.net.Stream, bytes: []const u8) !void {
    var buffer: [2048]u8 = undefined;
    var writer = socket.writer(io, &buffer);
    try writer.interface.writeAll(bytes);
    try writer.interface.flush();
}

fn post(io: std.Io, port: u16, body: []const u8) !std.Io.net.Stream {
    return postRoute(io, port, "/v1/completions", body);
}

fn postRoute(io: std.Io, port: u16, route: []const u8, body: []const u8) !std.Io.net.Stream {
    const socket = try connect(io, port);
    errdefer socket.close(io);
    var buffer: [2048]u8 = undefined;
    var writer = socket.writer(io, &buffer);
    try writer.interface.print("POST {s} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{ route, body.len });
    try writer.interface.flush();
    try write(io, socket, body);
    return socket;
}

fn readAll(a: std.mem.Allocator, io: std.Io, socket: std.Io.net.Stream) ![]u8 {
    var buffer: [8192]u8 = undefined;
    var reader = socket.reader(io, &buffer);
    return reader.interface.allocRemaining(a, .limited(4 * 1024 * 1024));
}

fn memoryWaiting(a: std.mem.Allocator, io: std.Io, port: u16) !i64 {
    return (try health(a, io, port)).object.get("memory").?.object.get("waiting_requests").?.integer;
}

fn health(a: std.mem.Allocator, io: std.Io, port: u16) !std.json.Value {
    const socket = try connect(io, port);
    defer socket.close(io);
    try write(io, socket, "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    const response = try readAll(a, io, socket);
    const start = (std.mem.indexOf(u8, response, "\r\n\r\n") orelse return error.MissingHttpBody) + 4;
    const body = try std.json.parseFromSlice(std.json.Value, a, response[start..], .{});
    return body.value;
}

const CacheCounts = struct {
    enabled: bool,
    bytes: u64,
    entries: usize,
    hits: u64,
    misses: u64,
    evictions: u64,

    fn read(a: std.mem.Allocator, io: std.Io, port: u16) !CacheCounts {
        const result = try std.json.parseFromValue(CacheCounts, a, (try health(a, io, port)).object.get("prompt_cache").?, .{});
        return result.value;
    }
};

fn waitForMemory(a: std.mem.Allocator, io: std.Io, port: u16, expected: i64) !void {
    for (0..1000) |_| {
        if (try memoryWaiting(a, io, port) == expected) return;
        try std.Io.sleep(io, .fromMilliseconds(25), .awake);
    }
    return error.MissingMemoryWaitState;
}

fn firstEvent(io: std.Io, socket: std.Io.net.Stream) !void {
    var buffer: [8192]u8 = undefined;
    var reader = socket.reader(io, &buffer);
    if (!std.mem.startsWith(u8, try reader.interface.takeSentinel('\n'), "HTTP/1.1 200")) return error.ExpectedStreamingResponse;
    while (true) {
        const line = try reader.interface.takeSentinel('\n');
        if (std.mem.startsWith(u8, line, "data: ")) {
            if (std.mem.indexOf(u8, line, "\"error\"") != null or std.mem.indexOf(u8, line, "[DONE]") != null) return error.ExpectedGeneratedToken;
            return;
        }
    }
}

fn assertCancelled(bytes: []const u8) !void {
    if (std.mem.indexOf(u8, bytes, "\"finish_reason\":\"length\"") != null or std.mem.indexOf(u8, bytes, "\"finish_reason\":\"stop\"") != null) return error.CancelledRequestCompleted;
    if (bytes.len != 0 and std.mem.indexOf(u8, bytes, "RequestTimedOut") == null and std.mem.indexOf(u8, bytes, "ServerStopping") == null and std.mem.indexOf(u8, bytes, "data: ") == null) {
        std.debug.print("Unexpected cancellation response: {s}\n", .{bytes});
        return error.MissingCancellation;
    }
}

const Output = struct {
    content: []const u8,
    reasoning: []const u8,
    finish: []const u8,
    usage: ?[]const u8 = null,

    fn parse(a: std.mem.Allocator, bytes: []const u8, streaming: bool) !Output {
        if (!std.mem.startsWith(u8, bytes, "HTTP/1.1 200")) return error.HttpRequestFailed;
        if (!streaming) {
            const start = (std.mem.indexOf(u8, bytes, "\r\n\r\n") orelse return error.MissingHttpBody) + 4;
            const body = try std.json.parseFromSlice(std.json.Value, a, bytes[start..], .{});
            const choice = body.value.object.get("choices").?.array.items[0];
            const message = choice.object.get("message");
            return .{ .content = if (message) |m| m.object.get("content").?.string else choice.object.get("text").?.string, .reasoning = if (message) |m| m.object.get("reasoning_content").?.string else "", .finish = choice.object.get("finish_reason").?.string, .usage = try std.json.Stringify.valueAlloc(a, body.value.object.get("usage").?, .{}) };
        }
        var content: std.ArrayList(u8) = .empty;
        var reasoning: std.ArrayList(u8) = .empty;
        var finish: ?[]const u8 = null;
        var done = false;
        var lines = std.mem.splitScalar(u8, bytes, '\n');
        while (lines.next()) |line| {
            if (!std.mem.startsWith(u8, line, "data: ")) continue;
            const data = std.mem.trimEnd(u8, line[6..], "\r");
            if (std.mem.eql(u8, data, "[DONE]")) {
                done = true;
                continue;
            }
            const body = try std.json.parseFromSlice(std.json.Value, a, data, .{});
            if (body.value.object.contains("error")) return error.StreamFailed;
            const choice = body.value.object.get("choices").?.array.items[0];
            if (choice.object.get("text")) |text| try content.appendSlice(a, text.string);
            if (choice.object.get("delta")) |delta| {
                if (delta.object.get("content")) |text| try content.appendSlice(a, text.string);
                if (delta.object.get("reasoning_content")) |text| try reasoning.appendSlice(a, text.string);
            }
            if (choice.object.get("finish_reason")) |reason| if (reason == .string) {
                finish = reason.string;
            };
        }
        if (!done or finish == null) return error.IncompleteStream;
        return .{ .content = content.items, .reasoning = reasoning.items, .finish = finish.? };
    }

    fn compare(expected: Output, actual: Output) !void {
        if (!std.mem.eql(u8, expected.content, actual.content) or !std.mem.eql(u8, expected.reasoning, actual.reasoning) or !std.mem.eql(u8, expected.finish, actual.finish)) return error.ConcurrentOutputMismatch;
        if (actual.usage) |usage| if (!std.mem.eql(u8, expected.usage.?, usage)) return error.ConcurrentUsageMismatch;
    }
};

const Scenario = struct {
    init: std.process.Init,
    child: std.process.Child,
    idle: bool,
    rounds: bool = false,
    memory: bool = false,
    prefixes: bool = false,
    cache_enabled: bool = true,
    cache_oversize: bool = false,
    image: []const u8 = "",
    http_checks: []const u8 = "",

    fn checkPrefixes(s: *Scenario, port: u16) !void {
        const a = s.init.arena.allocator();
        const io = s.init.io;
        var tokens: [2051]i32 = undefined;
        for (&tokens, 0..) |*id, i| id.* = @intCast(10 + i % 93);
        const body = try std.json.Stringify.valueAlloc(a, .{ .prompt = &tokens, .max_tokens = @as(usize, 16), .ignore_eos = true, .temperature = @as(f64, 0.7), .top_k = @as(usize, 12), .top_p = @as(f64, 0.8), .seed = @as(usize, 123) }, .{});
        const cold = try post(io, port, body);
        defer cold.close(io);
        const expected = try Output.parse(a, try readAll(a, io, cold), false);
        var counts = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(s.cache_enabled, counts.enabled);
        try std.testing.expectEqual(@as(u64, 0), counts.hits);
        try std.testing.expectEqual(@as(usize, @intFromBool(s.cache_enabled)), counts.entries);
        if (s.cache_enabled) try std.testing.expect(counts.bytes > 0);
        if (s.cache_oversize) try std.testing.expect(counts.bytes > 1074);
        for ([_]bool{ false, true }) |stream| {
            var request = try std.json.parseFromSlice(std.json.Value, a, body, .{});
            try request.value.object.put(a, "stream", .{ .bool = stream });
            const repeated = try post(io, port, try std.json.Stringify.valueAlloc(a, request.value, .{}));
            defer repeated.close(io);
            try expected.compare(try Output.parse(a, try readAll(a, io, repeated), stream));
        }
        counts = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(@as(u64, if (s.cache_enabled) 2 else 0), counts.hits);
        {
            var sockets: [3]std.Io.net.Stream = undefined;
            var opened: usize = 0;
            defer for (sockets[0..opened]) |socket| socket.close(io);
            for (&sockets) |*socket| {
                socket.* = try post(io, port, body);
                opened += 1;
            }
            for (sockets) |socket| try expected.compare(try Output.parse(a, try readAll(a, io, socket), false));
        }
        counts = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(@as(u64, if (s.cache_enabled) 5 else 0), counts.hits);
        tokens[0] = 101;
        const changed = try std.json.Stringify.valueAlloc(a, .{ .prompt = &tokens, .max_tokens = @as(usize, 1), .ignore_eos = true }, .{});
        const different = try post(io, port, changed);
        defer different.close(io);
        _ = try Output.parse(a, try readAll(a, io, different), false);
        counts = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(@as(u64, if (s.cache_enabled) 1 else 0), counts.evictions);
        const restored = try post(io, port, body);
        defer restored.close(io);
        try expected.compare(try Output.parse(a, try readAll(a, io, restored), false));
        counts = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(@as(u64, if (s.cache_enabled) 3 else 0), counts.misses);
        var ongoing = try std.json.parseFromSlice(std.json.Value, a, body, .{});
        try ongoing.value.object.put(a, "stream", .{ .bool = true });
        try ongoing.value.object.put(a, "max_tokens", .{ .integer = 10000 });
        const cancelled = try post(io, port, try std.json.Stringify.valueAlloc(a, ongoing.value, .{}));
        var open = true;
        defer if (open) cancelled.close(io);
        try firstEvent(io, cancelled);
        cancelled.close(io);
        open = false;
        const recovery = try post(io, port, body);
        defer recovery.close(io);
        try expected.compare(try Output.parse(a, try readAll(a, io, recovery), false));
        counts = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(@as(u64, if (s.cache_enabled) 7 else 0), counts.hits);
        const system = try a.alloc(u8, 5 * 320);
        for (0..320) |i| @memcpy(system[i * 5 ..][0..5], "word ");
        const conversation = try std.json.Stringify.valueAlloc(a, .{ .messages = &.{ .{ .role = "system", .content = system }, .{ .role = "user", .content = "Reply briefly." } }, .reasoning_effort = "none", .max_tokens = @as(usize, 16), .ignore_eos = true, .temperature = @as(f64, 0.7), .top_k = @as(usize, 12), .top_p = @as(f64, 0.8), .seed = @as(usize, 21) }, .{});
        const cold_chat = try postRoute(io, port, "/v1/chat/completions", conversation);
        defer cold_chat.close(io);
        const expected_chat = try Output.parse(a, try readAll(a, io, cold_chat), false);
        const usage = try std.json.parseFromSlice(std.json.Value, a, expected_chat.usage.?, .{});
        const prompt_tokens = usage.value.object.get("prompt_tokens").?.integer;
        try std.testing.expect(prompt_tokens > 256 and prompt_tokens < 2048);
        var chat = try std.json.parseFromSlice(std.json.Value, a, conversation, .{});
        try chat.value.object.put(a, "stream", .{ .bool = true });
        const cached_chat = try postRoute(io, port, "/v1/chat/completions", try std.json.Stringify.valueAlloc(a, chat.value, .{}));
        defer cached_chat.close(io);
        try expected_chat.compare(try Output.parse(a, try readAll(a, io, cached_chat), true));
        counts = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(@as(u64, if (s.cache_enabled) 8 else 0), counts.hits);
        std.debug.print("PASS: HTTP prefix cache enabled={any}: cold/reused/concurrent JSON/SSE agree; eviction, cache counters and cancellation match policy\n", .{s.cache_enabled});
        std.debug.print("PASS: {d}-token adaptive chat JSON/SSE agree, cache enabled={any}\n", .{ prompt_tokens, s.cache_enabled });
        try std.posix.kill(s.child.id.?, .TERM);
        if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
    }

    fn checkMemory(s: *Scenario, port: u16) !void {
        const a = s.init.arena.allocator();
        const io = s.init.io;
        const short = "{\"prompt\":\"Hello\",\"max_tokens\":12,\"temperature\":0}";
        const baseline = try post(io, port, short);
        defer baseline.close(io);
        const expected = try Output.parse(a, try readAll(a, io, baseline), false);
        const reserved = "{\"prompt\":\"Count upwards.\",\"max_tokens\":260000,\"ignore_eos\":true,\"temperature\":0,\"stream\":true}";
        var active = try post(io, port, reserved);
        var active_open = true;
        defer if (active_open) active.close(io);
        try firstEvent(io, active);
        const cancelled = try post(io, port, reserved);
        var cancelled_open = true;
        defer if (cancelled_open) cancelled.close(io);
        try waitForMemory(a, io, port, 1);
        cancelled.close(io);
        cancelled_open = false;
        try waitForMemory(a, io, port, 0);
        var next = try post(io, port, reserved);
        var next_open = true;
        defer if (next_open) next.close(io);
        try waitForMemory(a, io, port, 1);
        active.close(io);
        active_open = false;
        try firstEvent(io, next);
        try waitForMemory(a, io, port, 0);
        next.close(io);
        next_open = false;
        const recovery = try post(io, port, short);
        defer recovery.close(io);
        try expected.compare(try Output.parse(a, try readAll(a, io, recovery), false));
        std.debug.print("PASS: memory-limited request waits for release; queued cancellation clears the wait; next request matches its isolated output\n", .{});

        var prefix: [2051]i32 = undefined;
        for (&prefix, 0..) |*id, i| id.* = @intCast(10 + i % 93);
        const warm_body = try std.json.Stringify.valueAlloc(a, .{ .prompt = &prefix, .max_tokens = @as(usize, 1), .ignore_eos = true }, .{});
        const warm = try post(io, port, warm_body);
        defer warm.close(io);
        _ = try Output.parse(a, try readAll(a, io, warm), false);
        const retained = try CacheCounts.read(a, io, port);
        try std.testing.expect(retained.entries > 0 and retained.bytes > 0);
        const tokens = try a.alloc(i32, 262000);
        @memset(tokens, 1001);
        const oversized = try std.json.Stringify.valueAlloc(a, .{ .prompt = tokens, .max_tokens = @as(usize, 1) }, .{});
        const too_long = try post(io, port, oversized);
        defer too_long.close(io);
        const refusal = try readAll(a, io, too_long);
        if (std.mem.indexOf(u8, refusal, "RequestExceedsMemoryBudget") == null) return error.MissingPromptMemoryRefusal;
        const after_refusal = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(retained.evictions, after_refusal.evictions);
        try std.testing.expectEqual(retained.bytes, after_refusal.bytes);
        const image = try std.Io.Dir.cwd().readFileAlloc(io, s.image, a, .limited(10 * 1024 * 1024));
        const encoded = try a.alloc(u8, std.base64.standard.Encoder.calcSize(image.len));
        _ = std.base64.standard.Encoder.encode(encoded, image);
        const url = try std.mem.concat(a, u8, &.{ "data:image/jpeg;base64,", encoded });
        const image_body = try std.json.Stringify.valueAlloc(a, .{ .messages = &.{.{ .role = "user", .content = .{ .{ .type = "text", .text = "Describe this image." }, .{ .type = "image_url", .image_url = .{ .url = url, .detail = "high" } } } }}, .max_tokens = @as(usize, 1) }, .{});
        const too_large = try postRoute(io, port, "/v1/chat/completions", image_body);
        defer too_large.close(io);
        const image_refusal = try readAll(a, io, too_large);
        if (std.mem.indexOf(u8, image_refusal, "RequestExceedsMemoryBudget") == null) return error.MissingImageMemoryRefusal;
        const final = try post(io, port, short);
        defer final.close(io);
        try expected.compare(try Output.parse(a, try readAll(a, io, final), false));
        std.debug.print("PASS: oversized prompt and image workspace refused before inference; server recovers unchanged\n", .{});
        try std.posix.kill(s.child.id.?, .TERM);
        if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
    }

    fn checkRounds(s: *Scenario, port: u16) !void {
        const a = s.init.arena.allocator();
        const io = s.init.io;
        const image = try std.Io.Dir.cwd().readFileAlloc(io, s.image, a, .limited(10 * 1024 * 1024));
        const encoded = try a.alloc(u8, std.base64.standard.Encoder.calcSize(image.len));
        _ = std.base64.standard.Encoder.encode(encoded, image);
        const image_url = try std.mem.concat(a, u8, &.{ "data:image/png;base64,", encoded });
        const image_body = try std.json.Stringify.valueAlloc(a, .{ .messages = &.{.{ .role = "user", .content = .{ .{ .type = "text", .text = "Describe this image briefly." }, .{ .type = "image_url", .image_url = .{ .url = image_url, .detail = "low" } } } }}, .reasoning_effort = "none", .max_tokens = @as(usize, 12), .temperature = @as(f64, 0.8), .seed = @as(usize, 9182) }, .{});
        const cases = [_]struct { route: []const u8, body: []const u8 }{
            .{ .route = "/v1/completions", .body = "{\"prompt\":\"Name three colors:\",\"max_tokens\":24,\"temperature\":0,\"seed\":12}" },
            .{ .route = "/v1/completions", .body = "{\"prompt\":\"A short story about a fox:\",\"max_tokens\":31,\"temperature\":0.9,\"seed\":919}" },
            .{ .route = "/v1/chat/completions", .body = image_body },
        };
        var expected: [cases.len]Output = undefined;
        for (cases, &expected) |case, *value| {
            const before = try CacheCounts.read(a, io, port);
            const socket = try postRoute(io, port, case.route, case.body);
            defer socket.close(io);
            value.* = try Output.parse(a, try readAll(a, io, socket), false);
            if (std.mem.eql(u8, case.route, "/v1/chat/completions")) {
                const after = try CacheCounts.read(a, io, port);
                try std.testing.expectEqual(before.hits, after.hits);
                try std.testing.expectEqual(before.misses, after.misses);
                try std.testing.expectEqual(before.entries, after.entries);
            }
        }
        for ([_]bool{ false, true }) |stream| {
            const background = try post(io, port, long_request);
            var background_open = true;
            defer if (background_open) background.close(io);
            var buffer: [8192]u8 = undefined;
            var reader = background.reader(io, &buffer);
            while (true) {
                const line = try reader.interface.takeSentinel('\n');
                if (std.mem.startsWith(u8, line, "data: ")) break;
            }
            var sockets: [cases.len]std.Io.net.Stream = undefined;
            var opened: usize = 0;
            defer for (sockets[0..opened]) |socket| socket.close(io);
            for (cases, &sockets) |case, *socket| {
                var body = try std.json.parseFromSlice(std.json.Value, a, case.body, .{});
                try body.value.object.put(a, "stream", .{ .bool = stream });
                socket.* = try postRoute(io, port, case.route, try std.json.Stringify.valueAlloc(a, body.value, .{}));
                opened += 1;
            }
            if (stream) {
                background.close(io);
                background_open = false;
            }
            for (sockets, expected) |socket, prior| {
                const actual = try Output.parse(a, try readAll(a, io, socket), stream);
                try prior.compare(actual);
            }
            if (!stream) {
                // The short requests finished while this unbounded request was still active.
                const remainder = reader.interface.buffered();
                if (std.mem.indexOf(u8, remainder, "[DONE]") != null) return error.BackgroundFinishedBeforeShortRequests;
            }
        }
        std.debug.print("PASS: concurrent greedy/sampled/image requests match isolated JSON/SSE; short requests progress during long inference; disconnect preserves other requests\n", .{});
        const url = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}/v1/chat/completions", .{port});
        for ([_][]const u8{ "--controls-only", "--tool-stream-only", s.image }) |mode| {
            const result = try std.process.run(a, io, .{ .argv = &.{ s.http_checks, url, mode }, .stderr_limit = .limited(4 * 1024 * 1024) });
            std.debug.print("{s}", .{result.stderr});
            if (!result.term.success()) return error.HttpCompatibilityFailed;
        }
        try std.posix.kill(s.child.id.?, .TERM);
        if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
    }

    fn run(s: *Scenario) anyerror!void {
        const io = s.init.io;
        const a = s.init.arena.allocator();
        var stderr_buffer: [8192]u8 = undefined;
        var stderr = s.child.stderr.?.reader(io, &stderr_buffer);
        const prefix = "Native inference listening at http://127.0.0.1:";
        const port = while (true) {
            const line = try stderr.interface.takeSentinel('\n');
            if (std.mem.startsWith(u8, line, "Native memory admission:")) std.debug.print("{s}\n", .{line});
            if (std.mem.indexOf(u8, line, prefix)) |start| {
                const value = line[start + prefix.len ..];
                const end = std.mem.indexOfScalar(u8, value, ' ') orelse return error.InvalidListenAddress;
                break try std.fmt.parseInt(u16, value[0..end], 10);
            }
        };
        if (s.memory) return s.checkMemory(port);
        if (s.prefixes) return s.checkPrefixes(port);
        if (s.rounds) return s.checkRounds(port);
        if (s.idle) {
            try std.posix.kill(s.child.id.?, .INT);
            if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
            std.debug.print("PASS: SIGINT exits an idle server cleanly\n", .{});
            return;
        }

        const slow_head = try connect(io, port);
        defer slow_head.close(io);
        try write(io, slow_head, "POST /v1/completions HTTP/1.1\r\n");
        const slow_body = try connect(io, port);
        defer slow_body.close(io);
        try headers(io, slow_body, 9999);
        try write(io, slow_body, "{");
        const queued = try connect(io, port);
        defer queued.close(io);
        try headers(io, queued, long_request.len);
        try std.Io.sleep(io, .fromMilliseconds(250), .awake);
        const active = try post(io, port, long_request);
        defer active.close(io);
        try std.Io.sleep(io, .fromMilliseconds(100), .awake);
        try write(io, queued, long_request);
        try assertCancelled(try readAll(a, io, queued));
        try assertCancelled(try readAll(a, io, active));
        _ = try readAll(a, io, slow_head);
        _ = try readAll(a, io, slow_body);
        // A GPU operation already in flight may finish after the socket deadline.
        var recovered = false;
        for (0..16) |_| {
            const recovery = try post(io, port, "{\"prompt\":\"Hello\",\"max_tokens\":0}");
            defer recovery.close(io);
            const response = try readAll(a, io, recovery);
            if (std.mem.indexOf(u8, response, "200 OK") != null and std.mem.indexOf(u8, response, "\"completion_tokens\":0") != null) {
                recovered = true;
                break;
            }
            try assertCancelled(response);
        }
        if (!recovered) return error.ServerDidNotRecover;
        std.debug.print("PASS: deadlines stop active/queued inference and partial requests; next request succeeds\n", .{});

        const generating = try post(io, port, long_request);
        defer generating.close(io);
        var buffer: [8192]u8 = undefined;
        var reader = generating.reader(io, &buffer);
        while (true) {
            const line = try reader.interface.takeSentinel('\n');
            if (std.mem.startsWith(u8, line, "data: ")) break;
        }
        const waiting = try post(io, port, long_request);
        defer waiting.close(io);
        const stalled = try connect(io, port);
        defer stalled.close(io);
        try write(io, stalled, "POST /v1/completions HTTP/1.1\r\n");
        try std.Io.sleep(io, .fromMilliseconds(100), .awake);
        try std.posix.kill(s.child.id.?, .TERM);
        try assertCancelled(try reader.interface.allocRemaining(a, .limited(4 * 1024 * 1024)));
        try assertCancelled(try readAll(a, io, waiting));
        _ = try readAll(a, io, stalled);
        if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
        std.debug.print("PASS: SIGTERM cancels active/queued inference, releases stalled clients and exits cleanly\n", .{});
    }
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3 and args.len != 5) return error.ExpectedExecutableAndModel;
    if (args.len == 5) {
        const memory = std.mem.eql(u8, args[3], "--memory-only");
        const prefixes = std.mem.eql(u8, args[3], "--cache-only");
        var environment = try init.environ_map.clone(init.arena.allocator());
        defer environment.deinit();
        if (memory) try environment.put("TENSORFOLD_MEMORY_LIMIT_GB", "70");
        var scenario = Scenario{ .init = init, .idle = false, .rounds = !memory and !prefixes, .memory = memory, .prefixes = prefixes, .cache_enabled = !std.mem.eql(u8, args[4], "0"), .cache_oversize = std.mem.eql(u8, args[4], "0.000001"), .image = if (memory) args[4] else args[3], .http_checks = args[4], .child = try std.process.spawn(init.io, .{ .argv = &.{ args[1], "serve", args[2], "--port", "0", "--batch-streams", "4", "--shutdown-grace-seconds", "1", "--checkpoint-slots", if (prefixes) "1" else "12", "--prompt-cache-gib", if (prefixes) args[4] else "16" }, .environ_map = &environment, .stderr = .pipe }) };
        defer scenario.child.kill(init.io);
        const Event = union(enum) { done: anyerror!void, timeout: std.Io.Cancelable!void };
        var events: [2]Event = undefined;
        var select = std.Io.Select(Event).init(init.io, &events);
        defer select.cancelDiscard();
        try select.concurrent(.done, Scenario.run, .{&scenario});
        try select.concurrent(.timeout, std.Io.sleep, .{ init.io, std.Io.Duration.fromSeconds(300), .awake });
        switch (try select.await()) {
            .done => |result| try result,
            .timeout => return error.ServerRoundsCheckTimedOut,
        }
        return;
    }
    var environment = try init.environ_map.clone(init.arena.allocator());
    defer environment.deinit();
    for ([_][]const u8{ "nan", "0", "1" }) |budget| {
        try environment.put("TENSORFOLD_MEMORY_LIMIT_GB", budget);
        const result = try std.process.run(init.arena.allocator(), init.io, .{ .argv = &.{ args[1], "serve", args[2], "--port", "0" }, .environ_map = &environment, .stderr_limit = .limited(64 * 1024) });
        const expected = if (std.mem.eql(u8, budget, "1")) "WeightsExceedMemoryBudget" else "InvalidMemoryBudget";
        if (result.term.success() or std.mem.indexOf(u8, result.stderr, expected) == null or std.mem.indexOf(u8, result.stderr, "Loading ") != null or std.mem.indexOf(u8, result.stderr, "Native inference listening") != null) return error.InvalidMemoryBudgetLoadedModel;
    }
    std.debug.print("PASS: invalid/insufficient process budgets fail before model weights load\n", .{});
    for ([_]bool{ false, true }) |idle| {
        var scenario = Scenario{ .init = init, .idle = idle, .child = try std.process.spawn(init.io, .{ .argv = &.{ args[1], "serve", args[2], "--port", "0", "--request-timeout-seconds", if (idle) "0" else "2", "--shutdown-grace-seconds", "1", "--no-thinking" }, .stderr = .pipe }) };
        defer scenario.child.kill(init.io);
        const Event = union(enum) { done: anyerror!void, timeout: std.Io.Cancelable!void };
        var events: [2]Event = undefined;
        var select = std.Io.Select(Event).init(init.io, &events);
        defer select.cancelDiscard();
        try select.concurrent(.done, Scenario.run, .{&scenario});
        try select.concurrent(.timeout, std.Io.sleep, .{ init.io, std.Io.Duration.fromSeconds(90), .awake });
        switch (try select.await()) {
            .done => |result| try result,
            .timeout => return error.ServerLifecycleCheckTimedOut,
        }
    }
}
