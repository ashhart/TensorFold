const std = @import("std");
const V = std.json.Value;

pub fn compareUsage(expected: V, actual: V) !void {
    for ([_]V{ expected, actual }) |usage| {
        const prompt = usage.object.get("prompt_tokens").?.integer;
        const completion = usage.object.get("completion_tokens").?.integer;
        const cached = usage.object.get("prompt_tokens_details").?.object.get("cached_tokens").?.integer;
        const reasoning = usage.object.get("completion_tokens_details").?.object.get("reasoning_tokens").?.integer;
        if (prompt < 0 or completion < 0 or cached < 0 or cached > prompt or reasoning < 0 or reasoning > completion or usage.object.get("total_tokens").?.integer != prompt + completion) return error.InvalidUsage;
    }
    for ([_][]const u8{ "prompt_tokens", "completion_tokens", "total_tokens" }) |name| {
        if (expected.object.get(name).?.integer != actual.object.get(name).?.integer) return error.UsageMismatch;
    }
    if (expected.object.get("completion_tokens_details").?.object.get("reasoning_tokens").?.integer != actual.object.get("completion_tokens_details").?.object.get("reasoning_tokens").?.integer) return error.ReasoningUsageMismatch;
}

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
    return compareStream(init, url, body, false);
}

fn compareStream(init: std.process.Init, url: []const u8, body: V, incremental: bool) !usize {
    const a = init.arena.allocator();
    var request = body;
    const plain = try std.json.parseFromSlice(V, a, try post(init, url, request), .{});
    if (plain.value.object.get("error")) |err| {
        std.debug.print("Server error: {s}\n", .{try std.json.Stringify.valueAlloc(a, err, .{})});
        return error.ServerError;
    }
    const choice = plain.value.object.get("choices").?.array.items[0];
    const message = choice.object.get("message").?;
    const reasoning_tokens = plain.value.object.get("usage").?.object.get("completion_tokens_details").?.object.get("reasoning_tokens").?.integer;
    if (body.object.get("reasoning_effort")) |effort| if (effort == .string and std.mem.eql(u8, effort.string, "none") and reasoning_tokens != 0) return error.UnexpectedReasoningUsage;
    if (body.object.get("thinking_budget")) |budget| if (budget == .integer and budget.integer == 1) {
        if (std.mem.trim(u8, message.object.get("reasoning_content").?.string, " \r\n\t").len != 0) return error.ThinkingBudgetExceeded;
        if (message.object.get("content").?.string.len == 0) return error.MissingAnswerAfterThinkingBudget;
        if (reasoning_tokens != 2) return error.ThinkingBudgetUsageMismatch;
    };
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
    var usage_seen = false;
    var argument_fragments: usize = 0;
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
                if (named[index]) return error.DuplicateToolName;
                if (!std.mem.eql(u8, name.string, expected_calls.array.items[index].object.get("function").?.object.get("name").?.string)) return error.ToolNameMismatch;
                if (call.object.get("id").?.string.len == 0) return error.MissingToolId;
                named[index] = true;
            }
            if (function.object.get("arguments")) |value| {
                try arguments[index].appendSlice(a, value.string);
                if (value.string.len > 0) argument_fragments += 1;
            }
        };
        if (item.object.get("finish_reason")) |value| if (value == .string) {
            if (!std.mem.eql(u8, value.string, choice.object.get("finish_reason").?.string)) return error.FinishMismatch;
            finished = true;
            const usage = chunk.value.object.get("usage") orelse return error.MissingStreamUsage;
            try compareUsage(plain.value.object.get("usage").?, usage);
            usage_seen = true;
        };
    }
    if (!finished or !done or !usage_seen) return error.MissingStreamEnd;
    if (!std.mem.eql(u8, content.items, message.object.get("content").?.string) or !std.mem.eql(u8, reasoning.items, message.object.get("reasoning_content").?.string)) return error.StreamContentMismatch;
    for (expected_calls.array.items, arguments, named) |call, value, name_seen| if (!name_seen or !std.mem.eql(u8, call.object.get("function").?.object.get("arguments").?.string, value.items)) return error.ToolArgumentsMismatch;
    if (incremental and argument_fragments <= 2 * arguments.len) return error.ToolArgumentsNotIncremental;
    std.debug.print("PASS: JSON/SSE agree, {d} content bytes, {d} reasoning bytes, {d} tool calls\n", .{ content.items.len, reasoning.items.len, arguments.len });
    return arguments.len;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3 and args.len != 4) return error.ExpectedChatUrlAndImagePath;
    if (args.len == 4) {
        if (!std.mem.startsWith(u8, args[2], "https://")) return error.ExpectedRemoteImageUrl;
        const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[3], a, .limited(10 * 1024 * 1024));
        const encoder = std.base64.standard.Encoder;
        const encoded = try a.alloc(u8, encoder.calcSize(bytes.len));
        _ = encoder.encode(encoded, bytes);
        const data_url = try std.mem.concat(a, u8, &.{ "data:image/png;base64,", encoded });
        var expected: ?V = null;
        for ([_][]const u8{ data_url, args[2] }) |image_url| {
            const source = try std.json.Stringify.valueAlloc(a, .{ .messages = &.{.{ .role = "user", .content = .{ .{ .type = "text", .text = "Describe this image briefly." }, .{ .type = "image_url", .image_url = .{ .url = image_url, .detail = "low" } } } }}, .reasoning_effort = "none", .max_tokens = @as(usize, 8), .temperature = @as(f64, 0), .seed = @as(usize, 1234) }, .{});
            const body = try std.json.parseFromSlice(V, a, source, .{});
            const reply = try std.json.parseFromSlice(V, a, try post(init, args[1], body.value), .{});
            if (reply.value.object.contains("error")) return error.RemoteImageRequestFailed;
            if (expected) |prior| {
                for ([_][]const u8{ "choices", "usage" }) |key| if (!std.mem.eql(u8, try std.json.Stringify.valueAlloc(a, prior.object.get(key).?, .{}), try std.json.Stringify.valueAlloc(a, reply.value.object.get(key).?, .{}))) return error.RemoteImageMismatch;
                _ = try compare(init, args[1], body.value);
            } else expected = reply.value;
        }
        for ([_][]const u8{ "https://localhost/image.png", "https://example.com:8443/image.png", "http://example.com/image.png" }) |url| {
            const source = try std.json.Stringify.valueAlloc(a, .{ .messages = &.{.{ .role = "user", .content = &.{.{ .type = "image_url", .image_url = .{ .url = url } }} }}, .max_tokens = @as(usize, 0) }, .{});
            const body = try std.json.parseFromSlice(V, a, source, .{});
            const reply = try std.json.parseFromSlice(V, a, try post(init, args[1], body.value), .{});
            if (!reply.value.object.contains("error")) return error.UnsafeImageUrlAccepted;
        }
        std.debug.print("PASS: HTTPS/data image outputs and usage match; JSON/SSE agree; invalid destinations rejected\n", .{});
        return;
    }
    if (std.mem.eql(u8, args[2], "--controls-only")) {
        var defaults = try std.json.parseFromSlice(V, a,
            \\{"messages":[{"role":"user","content":"Name three colors."}],"reasoning_effort":"none","max_tokens":24,"seed":1234,"temperature":null,"top_k":null,"top_p":null}
        , .{});
        const first = try std.json.parseFromSlice(V, a, try post(init, args[1], defaults.value), .{});
        try defaults.value.object.put(a, "temperature", .{ .float = 1.0 });
        try defaults.value.object.put(a, "top_k", .{ .integer = 20 });
        try defaults.value.object.put(a, "top_p", .{ .float = 0.95 });
        const second = try std.json.parseFromSlice(V, a, try post(init, args[1], defaults.value), .{});
        for ([_][]const u8{ "choices", "usage" }) |key| {
            if (std.mem.eql(u8, key, "usage")) {
                try compareUsage(first.value.object.get(key).?, second.value.object.get(key).?);
                continue;
            }
            const lhs = try std.json.Stringify.valueAlloc(a, first.value.object.get(key).?, .{});
            const rhs = try std.json.Stringify.valueAlloc(a, second.value.object.get(key).?, .{});
            if (!std.mem.eql(u8, lhs, rhs)) return error.ModelSamplingDefaultsMismatch;
        }
        std.debug.print("PASS: omitted/null sampling matches Qwen generation_config defaults\n", .{});
        for ([_][]const u8{
            \\{"messages":[{"role":"user","content":"What is 2 + 3?"}],"thinking_budget":1,"max_tokens":48,"seed":1234}
            ,
            \\{"messages":[{"role":"user","content":"What is 2 + 3?"}],"thinking_budget":4,"max_tokens":48,"seed":1234}
            ,
            \\{"messages":[{"role":"user","content":"Say hello."}],"reasoning_effort":"none","thinking_budget":4,"max_tokens":16,"seed":1234}
            ,
            \\{"messages":[{"role":"user","content":"What is the weather in Copenhagen?"}],"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":"required","parallel_tool_calls":false,"thinking_budget":4,"max_tokens":128,"seed":1234}
            ,
        }, 0..) |source, index| {
            const body = try std.json.parseFromSlice(V, a, source, .{});
            const calls = try compare(init, args[1], body.value);
            if (index == 3 and calls != 1) return error.MissingRequiredToolCall;
        }
        return;
    }
    if (std.mem.eql(u8, args[2], "--tools-only")) {
        const tool = try std.json.parseFromSlice(V, a,
            \\{"messages":[{"role":"user","content":"What is the weather in Copenhagen?"}],"tools":[{"type":"function","function":{"name":"weather","description":"Get the weather in a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":{"type":"function","function":{"name":"weather"}},"parallel_tool_calls":false,"reasoning_effort":"none","max_tokens":128,"seed":1234}
        , .{});
        if (try compare(init, args[1], tool.value) != 1) return error.MissingRequiredToolCall;
        return;
    }
    if (std.mem.eql(u8, args[2], "--tool-stream-only")) {
        const tool = try std.json.parseFromSlice(V, a,
            \\{"messages":[{"role":"user","content":"Call write with path notes.txt, days 2, and content containing a greeting in Danish, Chinese and English. Include two paragraphs with a newline between them. Use at least 50 words in the content."}],"tools":[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"},"days":{"type":"integer"},"content":{"type":"string"}},"required":["path","days","content"]}}}],"tool_choice":"required","reasoning_effort":"none","temperature":0,"max_tokens":384,"seed":1234}
        , .{});
        if (try compareStream(init, args[1], tool.value, true) != 1) return error.MissingRequiredToolCall;
        std.debug.print("PASS: tool arguments arrive in incremental SSE fragments without duplicate calls\n", .{});
        return;
    }
    const base = try std.json.parseFromSlice(V, a,
        \\{"messages":[{"role":"user","content":"Reply with one word: hello"}],"reasoning_effort":"none","max_tokens":8,"seed":1234}
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
