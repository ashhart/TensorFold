//! Responses' include: OpenAI's values accepted and ignored, anything else refused as the 2.x line refuses it.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const responses_translate = @import("responses_translate.zig");

test "OpenAI's include values are accepted, others refused by name" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    for ([_][]const u8{ "null", "[]", "[\"reasoning.encrypted_content\"]", "[\"message.output_text.logprobs\", \"web_search_call.results\"]" }) |text|
        try responses_translate.include(&cx, (try json.parse(a, text)).ok);
    const refused = [_][2][]const u8{
        .{ "[\"reasoning.bogus\"]", "include has no value 'reasoning.bogus': the values are code_interpreter_call.outputs, computer_call_output.output.image_url, file_search_call.results, message.input_image.image_url, message.output_text.logprobs, reasoning.encrypted_content, web_search_call.action.sources, web_search_call.results" },
        .{ "\"reasoning.encrypted_content\"", "include must be a list of strings" },
        .{ "[1]", "include must be a list of strings" },
    };
    for (refused) |r| {
        try std.testing.expectError(error.Refused, responses_translate.include(&cx, (try json.parse(a, r[0])).ok));
        try std.testing.expectEqual(errors.Kind.request, cx.kind);
        try std.testing.expectEqualStrings(r[1], cx.message);
    }
}
