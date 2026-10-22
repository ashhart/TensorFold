const std = @import("std");
const controls = @import("drama_allow.zig");

test "delivery controls match synthetic goldens" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const policy = try controls.parse("sad,whispering");
    // Expected strings come from running the filter on synthetic text.
    const cases = [_][2][]const u8{
        .{ "", "" },
        .{ "[Pleased!] It works.", "[Pleased!] It works." },
        .{ "It [PLEASED] works.", "It [PLEASED] works." },
        .{ "It [ softly,\tsighing ] works.", "It works." },
        .{ "[Sad] Hi [ WHISPERING! ] there.", "[sad] Hi [whispering] there." },
        .{ "[nervous][sighing] I miss you [a] lot.", "I miss you [a] lot." },
        .{ "<WHISPER>Go on</Whisper> <soft>now</soft>.", "Go on now." },
        .{ "Meet [someone] on [Tuesday].", "Meet [someone] on [Tuesday]." },
        .{ "As [see note 3] says.", "As [see note 3] says." },
        .{ "Read [sighs](http://example.test).", "Read [sighs](http://example.test)." },
        .{ "Use x[i], a[pause] and m[1][sighs].", "Use x[i], a[pause] and m[1][sighs]." },
        .{ "Write `[pleased]` before a line.", "Write `[pleased]` before a line." },
        .{ "A [softly and unclosed", "A [softly and unclosed" },
        .{ "Two  spaces [1]\n\nstay.", "Two  spaces [1]\n\nstay." },
        .{ "[note <SOFT>quietly</SOFT>] stays.", "[note quietly] stays." },
        .{ "[<soft>Whispering</soft>] Hi.", "[whispering] Hi." },
        .{ "[see [sighs] <fast>now</fast>] ok", "[see [sighs] now] ok" },
        .{ "[see [note] [sighs] <soft>x</soft>] stays. [sighs] Go.", "[see [note] [sighs] x] stays. Go." },
        .{ "[see [note] [sighs] Go.", "[see [note] Go." },
        .{ "Hi [sighs] there, [pleased]\n\nnext. ¶ [sighs] On.", "Hi there, [pleased]\n\nnext. ¶ On." },
        .{ "Hi\n[sighs]\nthere word [sighs].", "Hi\nthere word ." },
        .{ " \t [sighs] Hi", "Hi" },
        .{ "你好[sighs]世界", "你好[sighs]世界" },
        .{ "[WHISpering\u{a0}!] café", "[whispering] café" },
        .{ "end [sighs]", "end " },
        .{ "[unknown] [whispering][sighs][sad]", "[unknown] [whispering][sad]" },
        .{ "a]b>c[unfinished", "a]b>c[unfinished" },
        .{ "<unknown>text</unknown>", "<unknown>text</unknown>" },
        .{ "[sighs [laughs] tail", "[sighs [laughs] tail" },
        .{ "[very very very very very very very very very very very very very very very very very very very very very very very very calm] text", "[very very very very very very very very very very very very very very very very very very very very very very very very calm] text" },
        .{ "x[note [sighs] tail]", "x[note [sighs] tail]" },
        .{ "[note note note note note note note note note note note note note note note note note note note note note note note note note note [sighs] tail] text", "[note note note note note note note note note note note note note note note note note note note note note note note note note note [sighs] tail] text" },
    };
    for (cases) |c| try std.testing.expectEqualStrings(c[1], try controls.filter(a, policy, c[0]));
}
