//! LFM2's own chat template (the checkpoint's chat_template.jinja) over the history a tool-calling client sends.
const std = @import("std");
const json = @import("json");
const template = @import("template");
const errors = @import("errors.zig");
const messages_mod = @import("messages.zig");
const hf_text = @import("hf_text.zig");

const source = @embedFile("fixtures/lfm2_chat_template.jinja");

/// `body` as the server hands the template: normalized, call arguments as objects, then rendered.
fn render(a: std.mem.Allocator, body: []const u8, tools: ?[]const u8) ![]const u8 {
    var cx: errors.Cx = .{ .a = a };
    const messages = (try json.parse(a, body)).ok;
    const ready = try messages_mod.toolArguments(&cx, try messages_mod.normalize(&cx, messages, "system", false));
    var globals: std.json.ObjectMap = .empty;
    try globals.put(a, "bos_token", .{ .string = "<|startoftext|>" });
    const offered: std.json.Value = if (tools) |t| try hf_text.toStd(a, (try json.parse(a, t)).ok) else .null;
    const compiled = try template.compile(a, source, null);
    const out = try template.renderProvenanceGlobals(a, &compiled, try hf_text.toStd(a, ready), offered, .{ .object = .empty }, .{ .object = globals }, true, null);
    return out.s;
}

const history =
    \\[{"role":"user","content":"Compare two things."},
    \\ {"role":"assistant","content":"","tool_calls":[
    \\  {"id":"c1","type":"function","function":{"name":"lookup","arguments":@A}},
    \\  {"id":"c2","type":"function","function":{"name":"web-search","arguments":@B}}]},
    \\ {"role":"tool","tool_call_id":"c1","content":"42 rows"},
    \\ {"role":"tool","tool_call_id":"c2","content":"{\"hits\": 3}"}]
;

const want = "<|startoftext|><|im_start|>user\nCompare two things.<|im_end|>\n" ++
    "<|im_start|>assistant\n<|tool_call_start|>[lookup(query='it\\'s', limit=4, filters={\"a\": [1, true]}), " ++
    "web-search(q='zig')]<|tool_call_end|><|im_end|>\n" ++
    "<|im_start|>tool\n42 rows<|im_end|>\n<|im_start|>tool\n{\"hits\": 3}<|im_end|>\n<|im_start|>assistant\n<think>";

test "lfm2 template: one assistant message with two calls, then two tool results, renders in order" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const mapping = [_][]const u8{ "{\"query\":\"it's\",\"limit\":4,\"filters\":{\"a\":[1,true]}}", "{\"q\":\"zig\"}" };
    // arguments as a mapping, and as the JSON string OpenAI clients send, give the same prompt
    const strings = [_][]const u8{
        "\"{\\\"query\\\":\\\"it's\\\",\\\"limit\\\":4,\\\"filters\\\":{\\\"a\\\":[1,true]}}\"",
        "\"{\\\"q\\\":\\\"zig\\\"}\"",
    };
    for ([_]*const [2][]const u8{ &mapping, &strings }) |args| {
        const body = try std.mem.replaceOwned(u8, a, try std.mem.replaceOwned(u8, a, history, "@A", args[0]), "@B", args[1]);
        try std.testing.expectEqualStrings(want, try render(a, body, null));
    }
}

test "lfm2 template: a call with no arguments, null content, and the tool list with a system message" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools =
        \\[{"type":"function","function":{"name":"now","parameters":{"type":"object","properties":{}}}}]
    ;
    const body =
        \\[{"role":"system","content":"Be brief."},{"role":"user","content":"Time?"},
        \\ {"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"now","arguments":"{}"}},
        \\  {"id":"b","type":"function","function":{"name":"now","arguments":""}},
        \\  {"id":"c","type":"function","function":{"name":"now"}}]},
        \\ {"role":"tool","tool_call_id":"a","content":"noon"},{"role":"tool","tool_call_id":"b","content":"noon"},
        \\ {"role":"tool","tool_call_id":"c","content":"noon"}]
    ;
    const got = try render(a, body, tools);
    try std.testing.expectEqualStrings("<|startoftext|><|im_start|>system\nBe brief.\nList of tools: [" ++
        "{\"type\": \"function\", \"function\": {\"name\": \"now\", \"parameters\": {\"type\": \"object\", \"properties\": {}}}}]<|im_end|>\n" ++
        "<|im_start|>user\nTime?<|im_end|>\n<|im_start|>assistant\n<|tool_call_start|>[now(), now(), now()]<|tool_call_end|><|im_end|>\n" ++
        "<|im_start|>tool\nnoon<|im_end|>\n<|im_start|>tool\nnoon<|im_end|>\n<|im_start|>tool\nnoon<|im_end|>\n<|im_start|>assistant\n<think>", got);
}
