const std = @import("std");
const V = std.json.Value;

fn post(init: std.process.Init, url: []const u8, body: V) ![]const u8 {
    const a = init.arena.allocator();
    const bytes = try std.json.Stringify.valueAlloc(a, body, .{});
    const path = "build/native-checks/http-request.json";
    const file = try std.Io.Dir.cwd().createFile(init.io, path, .{});
    defer file.close(init.io);
    try file.writeStreamingAll(init.io, bytes);
    const result = try std.process.run(a, init.io, .{ .argv = &.{ "curl", "--silent", "--show-error", "--max-time", "120", url, "-H", "Content-Type: application/json", "--data-binary", "@" ++ path }, .stdout_limit = .limited(4 * 1024 * 1024) });
    if (!result.term.success()) {
        std.debug.print("HTTP check failed: {s}\n", .{result.stderr});
        return error.HttpRequestFailed;
    }
    return result.stdout;
}

fn compare(init: std.process.Init, url: []const u8, body: V) !usize {
    const a = init.arena.allocator();
    var request = body;
    const plain = try std.json.parseFromSlice(V, a, try post(init, url, request), .{});
    if (plain.value.object.get("error")) |err| {
        std.debug.print("Server error: {s}\n", .{try std.json.Stringify.valueAlloc(a, err, .{})});
        return error.ServerError;
    }
    const choice = plain.value.object.get("choices").?.array.items[0];
    const message = choice.object.get("message").?;
    const expected_calls = message.object.get("tool_calls") orelse V{ .array = std.json.Array.init(a) };
    const arguments = try a.alloc(std.ArrayList(u8), expected_calls.array.items.len);
    for (arguments) |*value| value.* = .empty;
    const named = try a.alloc(bool, arguments.len);
    @memset(named, false);
    try request.object.put(a, "stream", .{ .bool = true });
    const stream = try post(init, url, request);
    var content: std.ArrayList(u8) = .empty;
    var reasoning: std.ArrayList(u8) = .empty;
    var finished = false;
    var done = false;
    var lines = std.mem.splitScalar(u8, stream, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "data: ")) continue;
        if (std.mem.eql(u8, line[6..], "[DONE]")) {
            done = true;
            continue;
        }
        const chunk = try std.json.parseFromSlice(V, a, line[6..], .{});
        const item = chunk.value.object.get("choices").?.array.items[0];
        const delta = item.object.get("delta").?;
        if (delta != .object) return error.InvalidChatDelta;
        if (delta.object.get("content")) |value| try content.appendSlice(a, value.string);
        if (delta.object.get("reasoning_content")) |value| try reasoning.appendSlice(a, value.string);
        if (delta.object.get("tool_calls")) |calls| for (calls.array.items) |call| {
            const index: usize = @intCast(call.object.get("index").?.integer);
            if (index >= arguments.len) return error.UnexpectedToolCall;
            const function = call.object.get("function").?;
            if (function.object.get("name")) |name| {
                if (!std.mem.eql(u8, name.string, expected_calls.array.items[index].object.get("function").?.object.get("name").?.string)) return error.ToolNameMismatch;
                if (call.object.get("id").?.string.len == 0) return error.MissingToolId;
                named[index] = true;
            }
            if (function.object.get("arguments")) |value| try arguments[index].appendSlice(a, value.string);
        };
        if (item.object.get("finish_reason")) |value| if (value == .string) {
            if (!std.mem.eql(u8, value.string, choice.object.get("finish_reason").?.string)) return error.FinishMismatch;
            finished = true;
        };
    }
    if (!finished or !done) return error.MissingStreamEnd;
    if (!std.mem.eql(u8, content.items, message.object.get("content").?.string) or !std.mem.eql(u8, reasoning.items, message.object.get("reasoning_content").?.string)) return error.StreamContentMismatch;
    for (expected_calls.array.items, arguments, named) |call, value, name_seen| if (!name_seen or !std.mem.eql(u8, call.object.get("function").?.object.get("arguments").?.string, value.items)) return error.ToolArgumentsMismatch;
    std.debug.print("PASS: JSON/SSE agree, {d} content bytes, {d} reasoning bytes, {d} tool calls\n", .{ content.items.len, reasoning.items.len, arguments.len });
    return arguments.len;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) return error.ExpectedChatUrlAndImagePath;
    if (std.mem.eql(u8, args[2], "--tools-only")) {
        const tool = try std.json.parseFromSlice(V, a,
            \\{"messages":[{"role":"user","content":"What is the weather in Copenhagen?"}],"tools":[{"type":"function","function":{"name":"weather","description":"Get the weather in a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":{"type":"function","function":{"name":"weather"}},"parallel_tool_calls":false,"max_tokens":128,"seed":1234}
        , .{});
        if (try compare(init, args[1], tool.value) != 1) return error.MissingRequiredToolCall;
        return;
    }
    const base = try std.json.parseFromSlice(V, a,
        \\{"messages":[{"role":"user","content":"Reply with one word: hello"}],"max_tokens":8,"seed":1234}
    , .{});
    _ = try compare(init, args[1], base.value);
    const thinking = try std.json.parseFromSlice(V, a,
        \\{"messages":[{"role":"user","content":"What is 2 + 3?"}],"reasoning_effort":"low","max_tokens":24,"seed":1234}
    , .{});
    _ = try compare(init, args[1], thinking.value);
    const image = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(10 * 1024 * 1024));
    const encoder = std.base64.standard.Encoder;
    const encoded = try a.alloc(u8, encoder.calcSize(image.len));
    _ = encoder.encode(encoded, image);
    const url = try std.mem.concat(a, u8, &.{ "data:image/png;base64,", encoded });
    const source = try std.json.Stringify.valueAlloc(a, .{ .messages = &.{.{ .role = "user", .content = .{ .{ .type = "text", .text = "Describe this image briefly." }, .{ .type = "image_url", .image_url = .{ .url = url, .detail = "low" } } } }}, .max_tokens = @as(usize, 8), .seed = @as(usize, 1234) }, .{});
    const request = try std.json.parseFromSlice(V, a, source, .{});
    _ = try compare(init, args[1], request.value);
}
