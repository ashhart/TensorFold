//! Ollama translation host tests: options, refusals, body shapes and NDJSON framing.
const std = @import("std");
const json = @import("json");
const ollama = @import("ollama.zig");
const errors = @import("errors.zig");
const t = std.testing;

fn body(a: std.mem.Allocator, text: []const u8) json.Value {
    return (json.parse(a, text) catch unreachable).ok;
}

test "ollama options map onto the chat request and think becomes a template kwarg" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const raw = body(a,
        \\{"model":"m","messages":[{"role":"user","content":"H"}],"think":true,"options":{"temperature":0.3,"top_k":20,"top_p":0.9,"min_p":0.1,"num_predict":7,"seed":4,"presence_penalty":0.2,"frequency_penalty":0.3}}
    );
    const chat = try ollama.chatBody(a, "local-model", 4096, raw, &cx);
    try t.expectEqualStrings("local-model", chat.get("model").?.string);
    try t.expect(chat.get("return_token_ids").?.bool);
    try t.expectApproxEqAbs(@as(f64, 0.3), chat.get("temperature").?.float, 1e-9);
    try t.expectEqualStrings("7", chat.get("max_tokens").?.int);
    try t.expect(chat.get("chat_template_kwargs").?.get("enable_thinking").?.bool);
}

test "ollama refuses unsupported options by name and a num_ctx past the window" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    for ([_][]const u8{ "mirostat", "repeat_penalty", "tfs_z", "num_ctx" }) |option| {
        cx.message = "";
        const text = std.fmt.allocPrint(a, "{{\"prompt\":\"H\",\"options\":{{\"{s}\":{s}}}}}", .{ option, if (std.mem.eql(u8, option, "num_ctx")) "8192" else "1" }) catch unreachable;
        try t.expectError(error.Refused, ollama.chatBody(a, "local-model", 4096, body(a, text), &cx));
        if (std.mem.eql(u8, option, "num_ctx"))
            try t.expect(std.mem.indexOf(u8, cx.message, "context window") != null)
        else
            try t.expect(std.mem.indexOf(u8, cx.message, option) != null);
    }
}

test "ollama generate prompt and system become chat messages" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const raw = body(a, "{\"prompt\":\"H\",\"system\":\"rules\"}");
    const chat = try ollama.chatBody(a, "local-model", 4096, raw, &cx);
    const messages = chat.get("messages").?.array;
    try t.expectEqual(@as(usize, 2), messages.len);
    try t.expectEqualStrings("system", messages[0].get("role").?.string);
    try t.expectEqualStrings("rules", messages[0].get("content").?.string);
    try t.expectEqualStrings("user", messages[1].get("role").?.string);
    try t.expectEqualStrings("H", messages[1].get("content").?.string);
}

test "ollama stops and seed pass through unchanged" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const raw = body(a, "{\"prompt\":\"H\",\"options\":{\"stop\":[\"END\"],\"seed\":9,\"top_k\":40}}");
    const chat = try ollama.chatBody(a, "local-model", 4096, raw, &cx);
    try t.expectEqual(@as(usize, 1), chat.get("stop").?.array.len);
    try t.expectEqualStrings("9", chat.get("seed").?.int);
    try t.expectEqualStrings("40", chat.get("top_k").?.int);
}

test "ollama maps json format and tools, and refuses unknown fields by name" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const mapped = body(a,
        \\{"prompt":"H","format":"json","tools":[{"type":"function","function":{"name":"lookup"}}],"tool_choice":"auto","stream":false}
    );
    const chat = try ollama.chatBody(a, "local-model", 4096, mapped, &cx);
    try t.expectEqualStrings("json_object", chat.get("response_format").?.get("type").?.string);
    try t.expectEqual(@as(usize, 1), chat.get("tools").?.array.len);
    try t.expect(chat.get("stream").?.bool == false);
    for ([_][]const u8{ "format", "tools", "images", "stream" }) |field| {
        cx.message = "";
        const raw = body(a, try std.fmt.allocPrint(a, "{{\"prompt\":\"H\",\"{s}\":{s}}}", .{ field, if (std.mem.eql(u8, field, "format")) "\"xml\"" else if (std.mem.eql(u8, field, "tools")) "{}" else if (std.mem.eql(u8, field, "images")) "[]" else "\"yes\"" }));
        try t.expectError(error.Refused, ollama.chatBody(a, "local-model", 4096, raw, &cx));
        try t.expect(std.mem.indexOf(u8, cx.message, field) != null);
    }
}
