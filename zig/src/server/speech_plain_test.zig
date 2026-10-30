const std = @import("std");
const plain = @import("speech_plain.zig");

test "empty redraw list matches reference plain controls synthetic goldens" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    // Expected strings were produced by executing the reference behaviour on synthetic text.
    const cases = [_][2][]const u8{
        .{ "", "" },
        .{ "  ordinary\ttext  ", "  ordinary\ttext  " },
        .{ "[sighs] hello [unknown]", "hello" },
        .{ "a[[sighs]]b", "a ]b" },
        .{ "a[unclosed", "a[unclosed" },
        .{ "<WHISPER>text</WHISPER> [sighs]", "<WHISPER>text</WHISPER>" },
        .{ "<soft>text</soft>.", "text ." },
        .{ "\t[sighs]a\n\n b\u{3000}c ", "a b c" },
        .{ "Direction: warm, then [slower]\nLine: [sighs] Hello <soft>there</soft>.", "Hello there ." },
        .{ " Direction: note\nNotLine: text", " Direction: note\nNotLine: text" },
        .{ "x[pause] [sighs](url) `[sad]`", "x (url) ` `" },
        .{ "你好[unknown]世界", "你好 世界" },
        .{ "end [sighs]", "end" },
        .{ "[sighs][laughs]", "" },
        .{ "[note <soft>x</soft>] tail", "tail" },
    };
    for (cases) |c| try std.testing.expectEqualStrings(c[1], try plain.filter(arena.allocator(), .{}, c[0]));
}
