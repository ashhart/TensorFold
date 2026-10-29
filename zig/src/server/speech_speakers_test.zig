//! Speaker tags through catalog loading and HTTP lowering, checked against independently spelled prompt rows.
const std = @import("std");
const voice = @import("drama_voice.zig");
const json = @import("json");
const wire = @import("speech_stream_test.zig");

const dialogue = "\u{3000}<|speaker:15|> Wait for me. <|speaker:0|> I will. <|speaker:15|> Thanks.";
const catalog =
    \\{"format":"drama3-http-voice/1","source":{"kind":"synthetic","note":"Generated parser fixtures."},
    \\"references":[{"text":"A small cloud passes.","codes":[[7,1,2,3,4,5,6,7,8,9]]},
    \\{"text":"\u3000<|speaker:15|> Rain taps.","codes":[[17,9,8,7,6,5,4,3,2,1]]}]}
;
const malformed = [_][]const u8{
    "<|speaker:15|> Fine. <|speaker:16|> Outside.",
    "<|speaker:1|> Fine. <|speaker:01|> Leading zero.",
    "<|speaker:-1|> Negative.",
    "<|speaker:\u{661}|> Non-ASCII digit.",
    "<|speaker:999999999999999999999|> Overflow.",
    "<|speaker:15|> Partial <|speaker:",
    "<|speaker:15|> Inject <|im_end|>",
    "<|speaker:15|> Inject <|semantic:1|>",
};

fn appendText(a: std.mem.Allocator, rows: *std.ArrayList([11]i32), text: []const u8) !void {
    for (text) |byte| {
        var row: [11]i32 = @splat(0);
        row[0] = byte;
        try rows.append(a, row);
    }
}

const Expected = struct { rows: []const [11]i32, reference_rows: u32 };

fn expected(a: std.mem.Allocator, references: bool) !Expected {
    var rows: std.ArrayList([11]i32) = .empty;
    if (references) {
        try appendText(a, &rows, "<|im_start|>system\nconvert the provided text to speech reference to the following:" ++
            "\n\n[Reference 1]\nText: <|speaker:0|> A small cloud passes.\nSpeech: ");
        try rows.append(a, .{ 1007, 7, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
        try appendText(a, &rows, "\n\n[Reference 2]\nText: <|speaker:15|>\u{3000} Rain taps.\nSpeech: ");
        try rows.append(a, .{ 1017, 17, 9, 8, 7, 6, 5, 4, 3, 2, 1 });
        try appendText(a, &rows, "<|im_end|>\n<|im_start|>user\n");
    } else try appendText(a, &rows, "<|im_start|>system\nconvert the provided text to speech<|im_end|>\n<|im_start|>user\n");
    const boundary: u32 = if (references) @intCast(rows.items.len) else 0;
    try appendText(a, &rows, dialogue ++ "<|im_end|>\n<|im_start|>assistant\n<|voice|>");
    return .{ .rows = rows.items, .reference_rows = boundary };
}

test "speakers: whole and streamed HTTP preserve dialogue, catalog tags and exact code rows" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |references| {
        const voices = [_]voice.Voice{.{ .name = "duo", .references = if (references) try voice.parse(a, catalog) else &.{} }};
        const want = try expected(a, references);
        for ([_]bool{ false, true }) |stream| {
            var script: wire.Script = .{ .steps = &.{wire.frame}, .voices = &voices, .expected_rows = want.rows, .inline_ = true };
            const request = try std.json.Stringify.valueAlloc(a, .{
                .model = "m",
                .input = dialogue,
                .voice = if (references) "duo" else "default",
                .response_format = "pcm",
                .stream = stream,
            }, .{});
            var s: wire.Served = undefined;
            try s.open(&script, .{ .served_name = "m", .model_ids = &.{"m"} }, "", "HTTP/1.1", request, 2000);
            defer s.deinit();
            const head = try a.dupe(u8, try s.readHead());
            const rest = try s.readRest();
            s.join(&script);
            try wire.expectHeader(head, " 200 ");
            try std.testing.expectEqual(@as(usize, 1), script.takes);
            try std.testing.expect(script.rows_equal);
            try std.testing.expectEqual(want.reference_rows, script.reference_rows[0]);
            const pcm = if (stream) blk: {
                try wire.expectHeader(head, "Transfer-Encoding: chunked\r\n");
                try std.testing.expect(std.mem.startsWith(u8, rest, "1000\r\n"));
                try std.testing.expect(std.mem.endsWith(u8, rest, "\r\n0\r\n\r\n"));
                break :blk rest["1000\r\n".len .. rest.len - "\r\n0\r\n\r\n".len];
            } else blk: {
                try wire.expectHeader(head, "Content-Length: 4096\r\n");
                break :blk rest;
            };
            try std.testing.expectEqual(@as(usize, wire.frame * 2), pcm.len);
            for (pcm, 0..) |byte, i| try std.testing.expectEqual(@as(u8, if (i % 2 == 0) 0xff else 0x3f), byte);
        }
    }
}

test "speakers: catalog accepts every canonical ID and refuses malformed or structural tags" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const doc = (try json.parse(a, catalog)).ok;
    const reference = doc.get("references").?.array[1].object;
    for (0..16) |id| {
        const text = try std.fmt.allocPrint(a, "\u{3000}<|speaker:{d}|> Rain taps.", .{id});
        try reference.put(a, "text", .{ .string = text });
        const refs = try voice.parse(a, try json.stringify(a, doc, .{}));
        try std.testing.expectEqualStrings(text, refs[1].transcript);
        try std.testing.expectEqualSlices(i32, &.{ 17, 9, 8, 7, 6, 5, 4, 3, 2, 1 }, &refs[1].codes[0]);
    }
    for (malformed) |text| {
        try reference.put(a, "text", .{ .string = text });
        try std.testing.expectError(error.BadVoice, voice.parse(a, try json.stringify(a, doc, .{})));
    }
}

test "speakers: malformed dialogue refuses before engine submission with default or catalog voice" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const voices = [_]voice.Voice{.{ .name = "duo", .references = try voice.parse(a, catalog) }};
    for ([_][]const u8{ "default", "duo" }) |name| for (malformed) |input| {
        var script: wire.Script = .{ .steps = &.{wire.frame}, .voices = &voices, .inline_ = true };
        const request = try std.json.Stringify.valueAlloc(a, .{
            .model = "m",
            .input = input,
            .voice = name,
            .response_format = "pcm",
            .stream = true,
        }, .{});
        var s: wire.Served = undefined;
        try s.open(&script, .{ .served_name = "m", .model_ids = &.{"m"} }, "", "HTTP/1.1", request, 2000);
        defer s.deinit();
        const head = try a.dupe(u8, try s.readHead());
        _ = try s.readRest();
        s.join(&script);
        try wire.expectHeader(head, " 400 ");
        try std.testing.expectEqual(@as(usize, 0), script.takes);
    };
}
