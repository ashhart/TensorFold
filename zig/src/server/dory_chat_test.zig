//! Dory's real chat template through HfText: prompts, EOS ids, the reasoning split and a qwen3_coder call.
const std = @import("std");
const json = @import("json");
const HfText = @import("hf_text.zig").HfText;
const model_text = @import("model_text.zig");
const reply_text = @import("reply_text.zig");
const tool_parse = @import("tool_parse.zig");
const fields = @import("fields.zig");

const template = @embedFile("fixtures/dory/chat_template.jinja");
const cases_json = @embedFile("fixtures/dory/render_cases.json");
const tokenizer_json =
    \\{"model":{"type":"BPE","vocab":{"<unk>":0,"<s>":1,"</s>":2,"[INST]":3,"[/INST]":4,"[A]":5,"[B]":6,"[C]":7,"[D]":8,"[E]":9,"<|im_start|>":10,"<|im_end|>":11,"<think>":12,"</think>":13,"<tool_call>":14,"</tool_call>":15,"<tool_response>":16,"</tool_response>":17,"H":18},"merges":[]},"decoder":{"type":"Fuse"},"added_tokens":[
    \\{"id":0,"content":"<unk>","special":true},{"id":1,"content":"<s>","special":true},{"id":2,"content":"</s>","special":true},
    \\{"id":3,"content":"[INST]","special":true},{"id":4,"content":"[/INST]","special":true},{"id":5,"content":"[A]","special":true},
    \\{"id":6,"content":"[B]","special":true},{"id":7,"content":"[C]","special":true},{"id":8,"content":"[D]","special":true},{"id":9,"content":"[E]","special":true},
    \\{"id":10,"content":"<|im_start|>","special":true},{"id":11,"content":"<|im_end|>","special":true},
    \\{"id":12,"content":"<think>","special":false},{"id":13,"content":"</think>","special":false},
    \\{"id":14,"content":"<tool_call>","special":false},{"id":15,"content":"</tool_call>","special":false},
    \\{"id":16,"content":"<tool_response>","special":false},{"id":17,"content":"</tool_response>","special":false}]}
;

fn load(tmp: *std.testing.TmpDir, a: std.mem.Allocator) !*HfText {
    const io = std.testing.io;
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = tokenizer_json });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"dory\",\"vocab_size\":131072}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"bos_token\":\"<s>\",\"eos_token\":\"<|im_end|>\",\"pad_token\":\"<|im_end|>\"}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "generation_config.json", .data = "{\"eos_token_id\":[2,11],\"pad_token_id\":11}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "chat_template.jinja", .data = template });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    var problem: []const u8 = "";
    return HfText.load(std.testing.allocator, io, dir, null, a, &problem);
}

test "Dory chat: chat_template.jinja renders every case as the Jinja reference does and EOS is both ids" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const t = try load(&tmp, a);
    defer t.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 2, 11 }, t.text().eosIds());
    try std.testing.expectEqual(@as(?u32, 13), t.text().tokenId("</think>"));
    const cases = (try json.parse(a, cases_json)).ok.array;
    for (cases) |case| {
        const req = case.get("request").?;
        const thinking = if (req.get("enable_thinking")) |v| v.bool else true;
        var problem: []const u8 = "";
        const out = try t.text().render(a, req.get("messages").?, .{ .tools = req.get("tools"), .enable_thinking = thinking }, &problem);
        std.testing.expectEqualStrings(case.get("prompt").?.string, out.text) catch |e| {
            std.debug.print("case {s}\n", .{case.get("name").?.string});
            return e;
        };
    }
}

test "Dory chat: thinking is the template's own switch, and the reply splits at the close token" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const levels = try fields.effortLevels(a, template);
    try std.testing.expectEqual(@as(usize, 0), levels.len);
    const markers = reply_text.think_markers;
    // thinking on: the prompt opened the block, so the reply starts inside it
    const on = try reply_text.splitThinking(a, "Plan.\n</think>\n\nDone.", true, markers);
    try std.testing.expectEqualStrings("Plan.\n", on.reasoning);
    try std.testing.expectEqualStrings("Done.", on.answer);
    // a reply that writes the opener itself loses it
    const own = try reply_text.splitThinking(a, "<think>\nPlan.\n</think>\n\nDone.", true, markers);
    try std.testing.expectEqualStrings("Plan.\n", own.reasoning);
    try std.testing.expectEqualStrings("Done.", own.answer);
    // thinking off: the prompt closed the block, the reply is all answer (the route does not split it)
    const call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>";
    const mid = try reply_text.splitThinking(a, "Look it up.\n</think>\n\nChecking.\n" ++ call, true, markers);
    try std.testing.expectEqualStrings("Look it up.\n", mid.reasoning);
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const parsed = try tool_parse.parse(a, mid.answer, tools, null);
    try std.testing.expectEqualStrings("Checking.", std.mem.trim(u8, parsed.content, " \n"));
    const fun = parsed.calls.?[0].get("function").?;
    try std.testing.expectEqualStrings("get_weather", fun.get("name").?.string);
    try std.testing.expectEqualStrings("{\"city\":\"Rome\",\"days\":2}", fun.get("arguments").?.string);
}
