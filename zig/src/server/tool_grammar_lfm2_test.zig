//! LFM2's required-call grammar: its text for named and required requests, and replies it admits through the parser.
const std = @import("std");
const json = @import("json");
const lfm2 = @import("tool_grammar_lfm2.zig");
const tool_lfm2 = @import("tool_lfm2.zig");
const tool_parse = @import("tool_parse.zig");
const Value = json.Value;

const tools_json =
    \\[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{
    \\"query":{"type":"string"},"limit":{"type":"integer"},"mode":{"enum":["fast","slow"]},"tags":{"type":"array"},
    \\"opt":{"type":["number","null"]},"on":{"type":"boolean"},"bad-key":{"type":"string"}},"required":["query","limit"]}}},
    \\ {"type":"function","name":"now"}]
;

fn offered(a: std.mem.Allocator) ![]const Value {
    return (try json.parse(a, tools_json)).ok.array;
}

fn has(text: []const u8, part: []const u8) !void {
    if (std.mem.indexOf(u8, text, part) == null) {
        std.debug.print("missing [{s}] in:\n{s}\n", .{ part, text });
        return error.TestExpectedEqual;
    }
}

test "lfm2 grammar: a named tool, or the only offered one, is one call in one list with no more" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try lfm2.ebnf(a, (try offered(a))[1..], false);
    try has(text, "root ::= blank \"<|tool_call_start|>[\" call \"]<|tool_call_end|>\" blank\n");
    try has(text, "call ::= call_0\n");
    try has(text, "call_0 ::= \"now(\" a_0_0_0 \")\"\na_0_0_0 ::= \"\"\n");
    try std.testing.expect(std.mem.indexOf(u8, text, "lookup") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "call)*") == null);
}

test "lfm2 grammar: required tools are alternatives, and without parallel_tool_calls=false calls repeat" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const many = try lfm2.ebnf(a, try offered(a), true);
    try has(many, "root ::= blank \"<|tool_call_start|>[\" call (\", \" call)* \"]<|tool_call_end|>\" blank\n");
    try has(many, "call ::= call_0 | call_1\n");
    try has(many, "call_0 ::= \"lookup(\" a_0_0_0 \")\"\n");
    try has(many, "call_1 ::= \"now(\" a_1_0_0 \")\"\n");
    const single = try lfm2.ebnf(a, try offered(a), false);
    try has(single, "call ::= call_0 | call_1\n");
    try std.testing.expect(std.mem.indexOf(u8, single, "call)*") == null);
}

test "lfm2 grammar: each argument is a keyword typed by its schema, required ones forced, a separator only after one" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try lfm2.ebnf(a, try offered(a), true);
    try has(text, "a_0_0_0 ::= \"query=\" (py_str) a_0_1_1\n");
    try has(text, "a_0_0_1 ::= \", \" \"query=\" (py_str) a_0_1_1\n");
    try has(text, "a_0_1_1 ::= \", \" \"limit=\" (py_int) a_0_2_1\n");
    try has(text, "a_0_2_0 ::= \"mode=\" (\"\\\"fast\\\"\" | \"'fast'\" | \"\\\"slow\\\"\" | \"'slow'\") a_0_3_1 | a_0_3_0\n");
    try has(text, "a_0_3_1 ::= \", \" \"tags=\" (py_list) a_0_4_1 | a_0_4_1\n");
    try has(text, "a_0_4_1 ::= \", \" \"opt=\" (py_num | py_none) a_0_5_1 | a_0_5_1\n");
    try has(text, "a_0_5_1 ::= \", \" \"on=\" (py_bool) a_0_6_1 | a_0_6_1\n");
    // a name Python cannot spell as a keyword is left out, and the chain ends after the last property it can
    try std.testing.expect(std.mem.indexOf(u8, text, "bad-key") == null);
    try has(text, "a_0_6_0 ::= \"\"\n");
}

test "lfm2 grammar: the structural tag holds the grammar as its one format" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try offered(a);
    const tag = (try json.parse(a, try lfm2.build(a, tools, true))).ok;
    try std.testing.expectEqualStrings("structural_tag", tag.strField("type").?);
    const format = tag.get("format").?;
    try std.testing.expectEqualStrings("grammar", format.strField("type").?);
    try std.testing.expectEqualStrings(try lfm2.ebnf(a, tools, true), format.strField("grammar").?);
}

test "lfm2 grammar: only a template that writes the probe call in LFM2's form picks it" {
    const probe = "tfprobe_fn";
    const key = "tfprobe_k";
    try std.testing.expect(lfm2.isStyle("<|tool_call_start|>[tfprobe_fn(tfprobe_k='v')]<|tool_call_end|>", probe, key));
    try std.testing.expect(!lfm2.isStyle("<tool_call>{\"name\": \"tfprobe_fn\", \"arguments\": {\"tfprobe_k\": \"v\"}}</tool_call>", probe, key));
    try std.testing.expect(!lfm2.isStyle("<|tool_call>call:tfprobe_fn{tfprobe_k:<|\"|>v<|\"|>}<tool_call|>", probe, key));
}

test "lfm2 grammar: replies the grammar admits read through the parser as calls with typed arguments" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try offered(a);
    const cases = [_]struct { []const u8, []const []const u8 }{
        .{ "[lookup(query='fast', limit=4)]", &.{"lookup {\"query\":\"fast\",\"limit\":4}"} },
        .{ "[lookup(query=\"say \\\"hi\\\"\", limit=-12, mode='slow', opt=1.5e3, on=True)]", &.{"lookup {\"query\":\"say \\\"hi\\\"\",\"limit\":-12,\"mode\":\"slow\",\"opt\":1500.0,\"on\":true}"} },
        .{ "[lookup(query='it\\'s', limit=0, tags=[1, 'a', None, {\"k\": [2.5]}]), now()]", &.{ "lookup {\"query\":\"it's\",\"limit\":0,\"tags\":[1,\"a\",null,{\"k\":[2.5]}]}", "now {}" } },
        .{ "[now()]", &.{"now {}"} },
    };
    for (cases) |c| {
        const reply = try std.mem.concat(a, u8, &.{ "\n", tool_lfm2.open, c[0], tool_lfm2.close, "\n" });
        const r = try tool_parse.parse(a, try tool_lfm2.dsml(a, reply, tools, true), tools, null);
        try std.testing.expectEqualStrings("", r.content);
        try std.testing.expectEqual(c[1].len, r.calls.?.len);
        for (c[1], r.calls.?) |want, call| {
            const f = call.get("function").?;
            const got = try std.fmt.allocPrint(a, "{s} {s}", .{ f.get("name").?.string, f.get("arguments").?.string });
            try std.testing.expectEqualStrings(want, got);
        }
    }
}
