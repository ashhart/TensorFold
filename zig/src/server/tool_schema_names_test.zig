//! LFM2's schema check reads keywords at keyword positions: a parameter named `pattern` or `format` is a name (1.3.4).
const std = @import("std");
const json = @import("json");
const tool_schema = @import("tool_schema.zig");
const pre = @import("stream_preflight_test.zig");

fn parsed(a: std.mem.Allocator, text: []const u8) !json.Value {
    return (try json.parse(a, text)).ok;
}

test "LFM2 schema: names under properties, $defs and definitions and data under enum or const are not keywords" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{
        "{\"type\":\"object\",\"properties\":{\"pattern\":{\"type\":\"string\"},\"format\":{\"type\":\"string\"}}}",
        "{\"type\":\"object\",\"properties\":{\"not\":{\"type\":\"integer\"},\"if\":{\"type\":\"boolean\"},\"contains\":{\"type\":\"string\"}}}",
        "{\"type\":\"object\",\"properties\":{\"x\":{\"$ref\":\"#/$defs/format\"}},\"$defs\":{\"format\":{\"type\":\"string\"}}}",
        "{\"type\":\"object\",\"properties\":{\"x\":{\"$ref\":\"#/definitions/pattern\"}},\"definitions\":{\"pattern\":{\"type\":\"string\"}}}",
        "{\"type\":\"object\",\"properties\":{\"x\":{\"enum\":[{\"not\":1},\"if\"],\"default\":{\"pattern\":\"a\"}}}}",
        "{\"type\":\"object\",\"properties\":{\"x\":{\"const\":{\"format\":\"date\"},\"examples\":[{\"then\":2}]}}}",
    }) |text| try std.testing.expect(tool_schema.supported(try parsed(a, text)));
    for ([_][]const u8{
        "{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\",\"pattern\":\"^a\"}}}",
        "{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\",\"format\":\"date\"}}}",
        "{\"type\":\"object\",\"properties\":{\"pattern\":{\"type\":\"string\",\"pattern\":\"^a\"}}}",
        "{\"type\":\"object\",\"properties\":{\"q\":{\"$ref\":\"#/$defs/d\"}},\"$defs\":{\"d\":{\"not\":{\"type\":\"null\"}}}}",
        "{\"type\":\"object\",\"patternProperties\":{\"^x\":{\"type\":\"string\"}}}",
        "{\"type\":\"object\",\"properties\":{\"q\":{\"anyOf\":[{\"type\":\"string\",\"format\":\"email\"}]}}}",
    }) |text| try std.testing.expect(!tool_schema.supported(try parsed(a, text)));
    // a value valid for a schema with a property named `pattern` is valid
    const schema = try parsed(a, "{\"type\":\"object\",\"properties\":{\"pattern\":{\"type\":\"string\"}},\"required\":[\"pattern\"]}");
    try std.testing.expect(try tool_schema.valid(a, try parsed(a, "{\"pattern\":\"x\"}"), schema));
}

fn body(comptime param: []const u8) []const u8 {
    return "{\"model\":\"test-model\",\"max_tokens\":2000,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"grep\",\"parameters\":{\"type\":\"object\"," ++
        "\"properties\":{" ++ param ++ "}}}}]";
}

fn calls(a: std.mem.Allocator, request: []const u8, reply: []const u8, want: usize) !void {
    const tokens = try pre.TestText.encode(undefined, a, reply, false);
    var backend: pre.CallEngine = .{ .tokens = tokens };
    const whole = try pre.sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ request, ",\"stream\":false}" }), .{});
    const streamed = try pre.sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ request, ",\"stream\":true}" }), .{});
    try std.testing.expectEqual(@as(?u16, 200), whole.status);
    try std.testing.expectEqual(want, whole.calls);
    try std.testing.expectEqual(want, streamed.calls);
}

test "LFM2 schema: tool_choice auto keeps calls to a tool whose parameters are named pattern and format" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const named = body("\"pattern\":{\"type\":\"string\"},\"format\":{\"type\":\"string\"}");
    try calls(a, named, "<|tool_call_start|>[grep(pattern='^a', format='json')]<|tool_call_end|>", 1);
    try calls(a, named, "<|tool_call_start|>[{\"name\":\"grep\",\"arguments\":{\"pattern\":\"^a\",\"format\":\"json\"}}]<|tool_call_end|>", 1);
    // a real pattern keyword is still one the post-validator cannot enforce: the call stays text
    const keyword = body("\"q\":{\"type\":\"string\",\"pattern\":\"^a\"}");
    try calls(a, keyword, "<|tool_call_start|>[grep(q='abc')]<|tool_call_end|>", 0);
}
