//! hf_text.zig's pretokenize_regex override, which only a Clef checkpoint (one with joint_head_config.json) takes.
const std = @import("std");
const HfText = @import("hf_text.zig").HfText;

test "pretokenize_regex: a Clef checkpoint's field replaces tokenizer.json's Split, one that fails refuses; others ignore it" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // a BPE over raw characters whose merges span a combining mark: the split shows in the ids
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data =
        \\{"pre_tokenizer":{"type":"Split","pattern":{"Regex":"\\p{L}+"},"behavior":"Isolated","invert":false},
        \\ "model":{"type":"BPE","vocab":{"e":0,"\u0301":1,"t":2,"e\u0301":3,"te\u0301":4,"e\u0301te\u0301":5},
        \\  "merges":["e \u0301","t e\u0301","e\u0301 te\u0301"]}}
    });
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var problem: []const u8 = "";
    const word = "e\u{301}te\u{301}";
    for ([_][]const u8{ "{\"pretokenize_regex\": \"[\\\\p{L}\\\\p{M}]+\"}", "{\"pretokenize_regex\": 3}" }) |field| {
        try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = field });
        const t = try HfText.load(gpa, io, dir, null, a, &problem); // no joint head: the field is not this loader's
        defer t.deinit();
        try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 0, 1 }, try t.text().encode(a, word, false));
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "joint_head_config.json", .data = "{}" });
    const cases = [_]struct { []const u8, []const u32 }{
        .{ "{}", &.{ 0, 1, 2, 0, 1 } }, // no field: tokenizer.json's \p{L}+ splits at each mark
        .{ "{\"pretokenize_regex\": \"[\\\\p{L}\\\\p{M}]+\"}", &.{5} },
    };
    for (cases) |c| {
        try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = c[0] });
        const t = try HfText.load(gpa, io, dir, null, a, &problem);
        defer t.deinit();
        try std.testing.expectEqualSlices(u32, c[1], try t.text().encode(a, word, false));
    }
    for ([_][]const u8{ "{\"pretokenize_regex\": \"[\\\\p{L}\"}", "{\"pretokenize_regex\": 3}", "{\"pretokenize_regex\": \"\"}" }) |bad| {
        try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = bad });
        // a load that wrongly succeeds is closed before the test fails, so nothing it holds outlives the test
        if (HfText.load(gpa, io, dir, null, a, &problem)) |loaded| {
            loaded.deinit();
            return error.TestUnexpectedResult;
        } else |e| try std.testing.expectEqual(error.Load, e);
        try std.testing.expect(std.mem.indexOf(u8, problem, "pretokenize_regex") != null);
    }
}
