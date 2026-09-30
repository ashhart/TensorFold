const std = @import("std");
const V = std.json.Value;

const Client = struct {
    init: std.process.Init,
    port: u16,

    fn value(c: Client, data: anytype) !V {
        const a = c.init.arena.allocator();
        return (try std.json.parseFromSlice(V, a, try std.json.Stringify.valueAlloc(a, data, .{}), .{})).value;
    }

    fn open(c: Client, method: []const u8, target: []const u8, body: V) !std.Io.net.Stream {
        const io = c.init.io;
        const socket = try (try std.Io.net.IpAddress.parse("127.0.0.1", c.port)).connect(io, .{ .mode = .stream });
        errdefer socket.close(io);
        var buffer: [8192]u8 = undefined;
        var writer = socket.writer(io, &buffer);
        const bytes = if (body == .null) "" else try std.json.Stringify.valueAlloc(c.init.arena.allocator(), body, .{});
        try writer.interface.print("{s} {s} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ method, target, bytes.len, bytes });
        try writer.interface.flush();
        return socket;
    }

    fn read(c: Client, socket: std.Io.net.Stream, status: u16) ![]const u8 {
        var buffer: [8192]u8 = undefined;
        var reader = socket.reader(c.init.io, &buffer);
        const bytes = try reader.interface.allocRemaining(c.init.arena.allocator(), .limited(4 * 1024 * 1024));
        const prefix = try std.fmt.allocPrint(c.init.arena.allocator(), "HTTP/1.1 {d}", .{status});
        if (!std.mem.startsWith(u8, bytes, prefix)) {
            std.debug.print("Responses HTTP failed: {s}\n", .{bytes[0..@min(bytes.len, 4096)]});
            return error.UnexpectedResponseStatus;
        }
        const start = (std.mem.indexOf(u8, bytes, "\r\n\r\n") orelse return error.MissingHttpBody) + 4;
        return bytes[start..];
    }

    fn call(c: Client, method: []const u8, target: []const u8, body: V, status: u16) !V {
        const socket = try c.open(method, target, body);
        defer socket.close(c.init.io);
        return (try std.json.parseFromSlice(V, c.init.arena.allocator(), try c.read(socket, status), .{})).value;
    }

    fn streamed(c: Client, socket: std.Io.net.Stream) !V {
        const a = c.init.arena.allocator();
        const bytes = try c.read(socket, 200);
        var sequence: i64 = 0;
        var final: ?V = null;
        var name: ?[]const u8 = null;
        var text: std.StringHashMapUnmanaged(std.ArrayList(u8)) = .empty;
        var lines = std.mem.splitScalar(u8, bytes, '\n');
        while (lines.next()) |line| {
            if (std.mem.startsWith(u8, line, "event: ")) name = line[7..];
            if (!std.mem.startsWith(u8, line, "data: ")) continue;
            if (final != null) return error.EventAfterResponseFinished;
            const event = (try std.json.parseFromSlice(V, a, line[6..], .{})).value;
            const kind = field(event, "type").string;
            try std.testing.expectEqualStrings(kind, name orelse return error.MissingResponseEventName);
            name = null;
            try std.testing.expectEqual(sequence, field(event, "sequence_number").integer);
            sequence += 1;
            if (std.mem.eql(u8, kind, "response.output_text.delta") or std.mem.eql(u8, kind, "response.reasoning_text.delta") or std.mem.eql(u8, kind, "response.function_call_arguments.delta")) {
                const entry = try text.getOrPut(a, field(event, "item_id").string);
                if (!entry.found_existing) entry.value_ptr.* = .empty;
                try entry.value_ptr.appendSlice(a, field(event, "delta").string);
            }
            if (std.mem.eql(u8, kind, "response.completed") or std.mem.eql(u8, kind, "response.incomplete")) final = field(event, "response");
            if (std.mem.eql(u8, kind, "response.failed")) return error.ResponseStreamFailed;
        }
        const result = final orelse return error.MissingResponseCompletion;
        for (field(result, "output").array.items) |item| {
            const expected = if (std.mem.eql(u8, field(item, "type").string, "function_call")) field(item, "arguments").string else field(field(item, "content").array.items[0], "text").string;
            const accumulated = text.get(field(item, "id").string);
            try std.testing.expectEqualStrings(expected, if (accumulated) |t| t.items else "");
        }
        return result;
    }

    fn path(c: Client, response: V) ![]const u8 {
        return std.fmt.allocPrint(c.init.arena.allocator(), "/v1/responses/{s}", .{field(response, "id").string});
    }
};

fn field(value: V, key: []const u8) V {
    return if (value == .object) value.object.get(key) orelse .null else .null;
}

fn equal(expected: V, actual: V) anyerror!void {
    if (expected == .object and actual == .object) {
        try std.testing.expectEqual(expected.object.count(), actual.object.count());
        var iter = expected.object.iterator();
        while (iter.next()) |entry| try equal(entry.value_ptr.*, field(actual, entry.key_ptr.*));
    } else if (expected == .array and actual == .array) {
        try std.testing.expectEqual(expected.array.items.len, actual.array.items.len);
        for (expected.array.items, actual.array.items) |lhs, rhs| try equal(lhs, rhs);
    } else if (expected == .string and actual == .string) try std.testing.expectEqualStrings(expected.string, actual.string) else try std.testing.expectEqualDeep(expected, actual);
}

fn sameResponse(c: Client, expected: V, actual: V) !void {
    try equal(field(expected, "status"), field(actual, "status"));
    const lhs = field(expected, "output").array.items;
    const rhs = field(actual, "output").array.items;
    try std.testing.expectEqual(lhs.len, rhs.len);
    for (lhs, rhs) |left, right| {
        var l = try c.value(left);
        var r = try c.value(right);
        _ = l.object.swapRemove("id");
        _ = r.object.swapRemove("id");
        _ = l.object.swapRemove("call_id");
        _ = r.object.swapRemove("call_id");
        try equal(l, r);
    }
    const first = field(expected, "usage");
    const second = field(actual, "usage");
    for ([_][]const u8{ "input_tokens", "output_tokens", "total_tokens", "output_tokens_details" }) |key| try equal(field(first, key), field(second, key));
    for ([_]V{ first, second }) |usage| {
        const cached = field(field(usage, "input_tokens_details"), "cached_tokens").integer;
        try std.testing.expect(cached >= 0 and cached <= field(usage, "input_tokens").integer);
    }
}

pub fn check(init: std.process.Init, port: u16, image: []const u8) !void {
    const c = Client{ .init = init, .port = port };
    const a = init.arena.allocator();
    const request = try c.value(.{ .input = "Say hello.", .instructions = "Be kind.", .reasoning = .{ .effort = "none" }, .max_output_tokens = @as(usize, 8), .ignore_eos = true, .temperature = @as(f64, 0.7), .seed = @as(usize, 21) });
    const response = try c.call("POST", "/v1/responses", request, 200);
    var serial = try c.value(request);
    try serial.object.put(a, "draft", .{ .bool = false });
    try sameResponse(c, response, try c.call("POST", "/v1/responses", serial, 200));
    try std.testing.expectEqualStrings("incomplete", field(response, "status").string);
    try std.testing.expectEqualStrings("max_output_tokens", field(field(response, "incomplete_details"), "reason").string);
    const chat = try c.call("POST", "/v1/chat/completions", try c.value(.{ .messages = &.{ .{ .role = "system", .content = "Be kind." }, .{ .role = "user", .content = "Say hello." } }, .reasoning_effort = "none", .max_tokens = @as(usize, 8), .ignore_eos = true, .temperature = @as(f64, 0.7), .seed = @as(usize, 21) }), 200);
    try std.testing.expectEqualStrings(field(field(field(response, "output").array.items[0], "content").array.items[0], "text").string, field(field(field(chat, "choices").array.items[0], "message"), "content").string);
    try equal(field(field(chat, "usage"), "prompt_tokens"), field(field(response, "usage"), "input_tokens"));
    try equal(response, try c.call("GET", try c.path(response), .null, 200));
    var stream_request = try c.value(request);
    try stream_request.object.put(a, "stream", .{ .bool = true });
    var sockets: [2]std.Io.net.Stream = undefined;
    var opened: usize = 0;
    defer for (sockets[0..opened]) |socket| socket.close(init.io);
    for (&sockets) |*socket| {
        socket.* = try c.open("POST", "/responses/", stream_request);
        opened += 1;
    }
    for (sockets) |socket| {
        const streamed = try c.streamed(socket);
        try sameResponse(c, response, streamed);
        try equal(streamed, try c.call("GET", try c.path(streamed), .null, 200));
    }
    var chained = try c.value(request);
    _ = chained.object.swapRemove("instructions");
    try chained.object.put(a, "input", .{ .string = "And now?" });
    try chained.object.put(a, "previous_response_id", field(response, "id"));
    const continued = try c.call("POST", "/v1/responses", chained, 200);
    var whole = try c.value(chained);
    _ = whole.object.swapRemove("previous_response_id");
    var input = std.json.Array.init(a);
    try input.append(try c.value(.{ .role = "user", .content = "Say hello." }));
    try input.appendSlice(field(response, "output").array.items);
    try input.append(try c.value(.{ .role = "user", .content = "And now?" }));
    try whole.object.put(a, "input", .{ .array = input });
    try whole.object.put(a, "store", .{ .bool = false });
    const manual = try c.call("POST", "/v1/responses", whole, 200);
    try sameResponse(c, continued, manual);
    _ = try c.call("GET", try c.path(manual), .null, 404);
    _ = try c.call("DELETE", try c.path(response), .null, 200);
    _ = try c.call("GET", try c.path(response), .null, 404);
    _ = try c.call("DELETE", try c.path(response), .null, 404);
    _ = try c.call("POST", "/v1/responses", chained, 400);
    for ([_][]const u8{ "{\"input\":\"x\",\"background\":true}", "{\"input\":\"x\",\"tools\":[{\"type\":\"web_search\"}]}", "{\"input\":\"x\",\"top_k\":\"many\"}", "{\"input\":\"x\",\"text\":{\"format\":{\"type\":\"json_object\"}}}" }) |bad| {
        _ = try c.call("POST", "/v1/responses", (try std.json.parseFromSlice(V, a, bad, .{})).value, 400);
    }
    const tool_request = (try std.json.parseFromSlice(V, a,
        \\{"input":"What is the weather in Oslo?","tools":[{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],"tool_choice":"required","reasoning":{"effort":"none"},"max_output_tokens":128,"temperature":0,"seed":42}
    , .{})).value;
    const tool_response = try c.call("POST", "/v1/responses", tool_request, 200);
    const call_item = field(tool_response, "output").array.items[0];
    try std.testing.expectEqualStrings("function_call", field(call_item, "type").string);
    var streamed_tool = try c.value(tool_request);
    try streamed_tool.object.put(a, "stream", .{ .bool = true });
    const tool_socket = try c.open("POST", "/v1/responses", streamed_tool);
    defer tool_socket.close(init.io);
    try sameResponse(c, tool_response, try c.streamed(tool_socket));
    const tool_result = try c.value(.{ .type = "function_call_output", .call_id = field(call_item, "call_id"), .output = "sunny" });
    const tool_chain = try c.value(.{ .previous_response_id = field(tool_response, "id"), .input = &.{tool_result}, .tools = field(tool_request, "tools"), .reasoning = .{ .effort = "none" }, .max_output_tokens = @as(usize, 8), .temperature = @as(f64, 0), .seed = @as(usize, 42) });
    const tool_continued = try c.call("POST", "/v1/responses", tool_chain, 200);
    var tool_whole = try c.value(tool_chain);
    _ = tool_whole.object.swapRemove("previous_response_id");
    var tool_input = std.json.Array.init(a);
    try tool_input.append(try c.value(.{ .role = "user", .content = "What is the weather in Oslo?" }));
    try tool_input.appendSlice(field(tool_response, "output").array.items);
    try tool_input.append(tool_result);
    try tool_whole.object.put(a, "input", .{ .array = tool_input });
    try sameResponse(c, tool_continued, try c.call("POST", "/v1/responses", tool_whole, 200));
    var thinking = try c.value(request);
    try thinking.object.put(a, "reasoning", try c.value(.{ .effort = "low" }));
    try thinking.object.put(a, "thinking_budget", .{ .integer = 1 });
    const thought = try c.call("POST", "/v1/responses", thinking, 200);
    try std.testing.expectEqual(@as(i64, 2), field(field(field(thought, "usage"), "output_tokens_details"), "reasoning_tokens").integer);
    try thinking.object.put(a, "stream", .{ .bool = true });
    const thinking_socket = try c.open("POST", "/v1/responses", thinking);
    defer thinking_socket.close(init.io);
    try sameResponse(c, thought, try c.streamed(thinking_socket));
    const image_bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, image, a, .limited(4 * 1024 * 1024));
    const encoded = try a.alloc(u8, std.base64.standard.Encoder.calcSize(image_bytes.len));
    const url = try std.fmt.allocPrint(a, "data:image/png;base64,{s}", .{std.base64.standard.Encoder.encode(encoded, image_bytes)});
    var image_request = try c.value(request);
    const parts = try c.value(.{ .input = &.{.{ .role = "user", .content = .{ .{ .type = "input_text", .text = "Describe this image." }, .{ .type = "input_image", .image_url = url } } }} });
    try image_request.object.put(a, "input", field(parts, "input"));
    const picture = try c.call("POST", "/v1/responses", image_request, 200);
    try image_request.object.put(a, "stream", .{ .bool = true });
    const image_socket = try c.open("POST", "/v1/responses", image_request);
    defer image_socket.close(init.io);
    try sameResponse(c, picture, try c.streamed(image_socket));
    var cancelled = try c.value(stream_request);
    try cancelled.object.put(a, "max_output_tokens", .{ .integer = 4096 });
    const cancelled_id = blk: {
        const socket = try c.open("POST", "/v1/responses", cancelled);
        defer socket.close(init.io);
        var buffer: [8192]u8 = undefined;
        var reader = socket.reader(init.io, &buffer);
        while (true) {
            const line = try reader.interface.takeSentinel('\n');
            if (!std.mem.startsWith(u8, line, "data: ")) continue;
            const event = (try std.json.parseFromSlice(V, a, line[6..], .{})).value;
            try std.testing.expectEqualStrings("response.created", field(event, "type").string);
            break :blk try a.dupe(u8, field(field(event, "response"), "id").string);
        }
    };
    try sameResponse(c, response, try c.call("POST", "/v1/responses", request, 200));
    _ = try c.call("GET", try std.fmt.allocPrint(a, "/v1/responses/{s}", .{cancelled_id}), .null, 404);
    std.debug.print("PASS: Responses HTTP chat/serial parity, JSON/SSE, concurrent requests, tool/history chains, images, reasoning usage, GET/DELETE, store:false and disconnect recovery\n", .{});
}
