//! A recipient reply max_tokens cuts inside a call message: its cut call, whole and streamed, as #1299's GLM call.
const std = @import("std");
const harness = @import("recipient_reply_test.zig");
const Text = harness.Text;

const head = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"write_file\",\"parameters\":{\"type\":\"object\"," ++
    "\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"},\"limit\":{\"type\":\"integer\"}}}}}]";
const thought = " to=self<|message|>A file is needed.<|eom|>";
const call = "<|start|>assistant to=write_file<|message|>";
const path = "<atem:function_calls>\n<atem:invoke name=\"write_file\">\n<atem:parameter name=\"path\">notes.txt</atem:parameter>\n";
const opened = path ++ "<atem:parameter name=\"content\">The history of";
const rest = " the bicycle</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|>";
const args = "{\"path\":\"notes.txt\",\"content\":\"The history of";

/// `cut` then `rest`, max_tokens at `cut`'s end: whole and streamed replies agree at every stride (null: unchecked).
fn expectCut(cut: []const u8, calls: usize, arguments: []const u8, content: ?[]const u8, extra: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: Text = .{};
    const n = (try text.tokens(a, cut)).len;
    const body = try std.fmt.allocPrint(a, "{s},\"max_tokens\":{d}{s}", .{ head, n, extra });
    var whole = try harness.send(.{}, try std.mem.concat(a, u8, &.{ cut, rest }), 4096, try std.mem.concat(a, u8, &.{ body, "}" }));
    defer whole.deinit();
    try std.testing.expectEqual(@as(u16, 200), whole.status);
    for ([_]usize{ 1, 3, 7, 4096 }) |stride| {
        var streamed = try harness.send(.{}, try std.mem.concat(a, u8, &.{ cut, rest }), stride, try std.mem.concat(a, u8, &.{ body, ",\"stream\":true}" }));
        defer streamed.deinit();
        for ([_]*const harness.Shown{ &whole, &streamed }) |s| {
            try std.testing.expectEqual(calls, s.calls);
            try std.testing.expectEqualStrings(arguments, s.arguments.items);
            if (content) |c| try std.testing.expectEqualStrings(c, s.content.items);
            for ([_][]const u8{ "atem", "<tool_call", "<function=", "<|" }) |m|
                if (std.mem.indexOf(u8, s.content.items, m) != null) return error.CallMarkupInContent;
            try std.testing.expectEqualStrings("A file is needed.", s.reasoning.items);
            try std.testing.expect(s.length);
        }
    }
}

test "recipient cut: a call max_tokens cuts is the reply's tool call, cut, whole and streamed" {
    try expectCut(thought ++ call ++ opened, 1, args, "", "");
    const said = "<|start|>assistant to=commentary<|message|>Writing it.<|eom|>";
    try expectCut(thought ++ said ++ call ++ opened, 1, args, "Writing it.", "");
    // a whole call first: both calls, the second cut
    const first = call ++ "<atem:function_calls>\n<atem:invoke name=\"write_file\">\n<atem:parameter name=\"path\">a.txt" ++
        "</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eom|>";
    try expectCut(thought ++ first ++ call ++ opened, 2, "{\"path\":\"a.txt\"}" ++ args, "", "");
    // inside a value's closer, between parameters, inside a parameter's name and after the invoke closed
    try expectCut(thought ++ call ++ opened ++ "</atem:para", 1, args, "", "");
    inline for (.{ "", "<atem:parameter name=\"con", "<atem:para" }) |tail|
        try expectCut(thought ++ call ++ path ++ tail, 1, "{\"path\":\"notes.txt\"", "", "");
    try expectCut(thought ++ call ++ path ++ "</atem:invoke>\n</atem:func", 1, "{\"path\":\"notes.txt\"}", "", "");
    // a value its schema types is JSON, cut
    const limit = "<atem:function_calls>\n<atem:invoke name=\"write_file\">\n<atem:parameter name=\"limit\">12";
    try expectCut(thought ++ call ++ limit, 1, "{\"limit\":12", "", "");
}

test "recipient cut: a call cut before its invoke names the tool is the header's call" {
    inline for (.{ "", "<atem:function_ca", "<atem:function_calls>\n<atem:invoke name=\"write_fi" }) |tail|
        try expectCut(thought ++ call ++ tail, 1, "", "", "");
    // inside the call's header no call is named: nothing is shown
    try expectCut(thought ++ "<|start|>assistant to=write_fi", 0, "", "", "");
}

test "recipient cut: the one-call policy hides the cut call" {
    try expectCut(thought ++ call ++ opened, 0, "", null, ",\"parallel_tool_calls\":false");
}
