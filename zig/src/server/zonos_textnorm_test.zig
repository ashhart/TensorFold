//! spokenText against the reference's model/tests/test_textnorm_wrapper.py (96d9e0ec), case for case, byte for byte.
const std = @import("std");
const speech = @import("zonos_speech.zig");
const textnorm = @import("zonos_textnorm.zig");
const loadDynLib = @import("jpeg_out.zig").loadDynLib;

const Case = struct { code: []const u8, text: []const u8, want: []const u8 };

/// `text` in `code` through the server's text step, with `normalize` as the request sent it.
fn spoken(a: std.mem.Allocator, text: []const u8, code: []const u8, normalize: ?bool) ![]u8 {
    return speech.spokenText(a, text, textnorm.language(code).?, normalize);
}

fn expectSpoken(c: Case, normalize: ?bool) !void {
    const got = try spoken(std.testing.allocator, c.text, c.code, normalize);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings(c.want, got);
}

/// test_collapse_line_breaks_only_touches_runs_with_a_break
const collapse = [_]Case{
    .{ .code = "en_us", .text = "spend less.\n\nHunt\u{2019}s Tolan", .want = "spend less. Hunt\u{2019}s Tolan" },
    .{ .code = "en_us", .text = "Line one.\r\nLine two.", .want = "Line one. Line two." },
    .{ .code = "en_us", .text = "Tabbed.\tNext", .want = "Tabbed. Next" },
    .{ .code = "en_us", .text = "a b", .want = "a b" },
    .{ .code = "en_us", .text = "Three\n\n\nbreaks", .want = "Three breaks" },
    .{ .code = "en_us", .text = "one line", .want = "one line" },
    .{ .code = "en_us", .text = "two  spaces", .want = "two  spaces" },
    .{ .code = "en_us", .text = "  lead", .want = "  lead" },
    .{ .code = "en_us", .text = "trail  ", .want = "trail  " },
    .{ .code = "en_us", .text = "", .want = "" },
};

/// test_normalization_switch, the English and other-language end-to-end tests, the phoneme-group and marker tests
const normalized = [_]Case{
    .{ .code = "en_us", .text = "It costs $5.\nNext.", .want = "It costs five dollars. Next." },
    .{ .code = "en_us", .text = "spend less.\n\nHunt\u{2019}s Tolan is called Harper.", .want = "spend less. Hunt\u{2019}s Tolan is called Harper." },
    .{ .code = "en_us", .text = "Really?\nYes.", .want = "Really? Yes." },
    .{ .code = "en_us", .text = "It costs $5.\nNext day.", .want = "It costs five dollars. Next day." },
    .{ .code = "en_us", .text = "We are open 24/7.", .want = "We are open twenty four seven." },
    .{ .code = "en_us", .text = "Save $20k.", .want = "Save twenty thousand dollars." },
    .{ .code = "en_us", .text = "In 1990, he moved.", .want = "In nineteen ninety, he moved." },
    .{ .code = "en_us", .text = "Call me at 123 Main Street, please.", .want = "Call me at one twenty three Main Street, please." },
    .{ .code = "ja", .text = "\u{5024}\u{6bb5}\u{306f}5.5\u{30c9}\u{30eb}\u{3067}\u{3059}\u{3002}\n\u{6b21}\u{3002}", .want = "\u{5024}\u{6bb5}\u{306f}\u{4e94}\u{70b9}\u{4e94}\u{30c9}\u{30eb}\u{3067}\u{3059}\u{3002}\u{6b21}\u{3002}" },
    .{ .code = "ja", .text = "\u{5024}\u{6bb5}\u{306f}5.5\u{30c9}\u{30eb}\u{3067}\u{3059}\u{3002} \u{6b21}\u{3002}", .want = "\u{5024}\u{6bb5}\u{306f}\u{4e94}\u{70b9}\u{4e94}\u{30c9}\u{30eb}\u{3067}\u{3059}\u{3002}\u{6b21}\u{3002}" },
    .{ .code = "cmn", .text = "\u{4e00}\u{6708}\u{4e00}\u{65e5}\u{3002} \u{4e8c}\u{6708}\u{4e8c}\u{65e5}\u{3002}", .want = "\u{4e00}\u{6708}\u{4e00}\u{65e5}\u{3002}\u{4e8c}\u{6708}\u{4e8c}\u{65e5}\u{3002}" },
    .{ .code = "pt_br", .text = "Custa 10%. Depois 20%, talvez.", .want = "Custa dez por cento. Depois vinte por cento, talvez." },
    .{ .code = "pt_br", .text = "P\u{e1}gina 3 (de 10).", .want = "P\u{e1}gina tr\u{ea}s (de dez)." },
    .{ .code = "pt_br", .text = "Ele chegou \u{e0}s 9h.", .want = "Ele chegou \u{e0}s nove horas." },
    .{ .code = "de", .text = "Seit 1990. Danach nichts.", .want = "Seit ein tausend neun hundert neunzig. Danach nichts." },
    .{ .code = "nl", .text = "Ik heb 5 katten.\nJa.", .want = "Ik heb vijf katten. Ja." },
    .{ .code = "no", .text = "Det koster 5 kroner.\nNeste.", .want = "Det koster fem kroner. Neste." },
    .{ .code = "ar", .text = "\u{644}\u{62f}\u{64a} 3 \u{643}\u{62a}\u{628}.", .want = "\u{644}\u{62f}\u{64a} \u{62b}\u{644}\u{627}\u{62b}\u{629} \u{643}\u{62a}\u{628}." },
    .{ .code = "hi", .text = "\u{92e}\u{947}\u{930}\u{947} \u{92a}\u{93e}\u{938} 5 \u{915}\u{93f}\u{924}\u{93e}\u{92c}\u{947}\u{902} \u{939}\u{948}\u{902}\u{964}", .want = "\u{92e}\u{947}\u{930}\u{947} \u{92a}\u{93e}\u{938} \u{92a}\u{93e}\u{901}\u{91a} \u{915}\u{93f}\u{924}\u{93e}\u{92c}\u{947}\u{902} \u{939}\u{948}\u{902}\u{964}" },
    .{ .code = "vi", .text = "T\u{f4}i c\u{f3} 3 cu\u{1ed1}n s\u{e1}ch.", .want = "T\u{f4}i c\u{f3} ba cu\u{1ed1}n s\u{e1}ch." },
    .{ .code = "en_us", .text = "I paid 5 \u{e000}k\u{e6}t\u{e001} for\n\u{e000}d\u{254}\u{261}\u{e001} food.", .want = "I paid five \u{e000}k\u{e6}t\u{e001} for \u{e000}d\u{254}\u{261}\u{e001} food." },
    .{ .code = "en_us", .text = "say  \u{e000}h\u{259}\u{2c8}lo\u{28a}\u{e001}  5", .want = "say  \u{e000}h\u{259}\u{2c8}lo\u{28a}\u{e001}  five" },
    .{ .code = "en_us", .text = "\u{e002}It costs $5\u{e004} today.", .want = "\u{e002}It costs five dollars\u{e004} today." },
    .{ .code = "en_us", .text = "say \u{e000}k\u{e6}t\u{e001}\u{e003} 5 times", .want = "say \u{e000}k\u{e6}t\u{e001}\u{e003} five times" },
};

fn needLibrary() !void {
    if (!textnorm.available(std.testing.allocator)) return error.SkipZigTest;
}

test "left out or false, the text is spoken as typed with line breaks collapsed, whether or not the library loads" {
    const typed: Case = .{ .code = "en_us", .text = "It costs $5.\nNext.", .want = "It costs $5. Next." };
    try expectSpoken(typed, null);
    try expectSpoken(typed, false);
    for (collapse) |c| {
        try expectSpoken(c, null);
        try expectSpoken(c, false);
    }
}

test "a grammar name or an unknown code is not a request language (test_unsupported_language_returns_collapsed_text)" {
    // the reference server refuses these codes before normalizing (languages.normalize), as this server does
    try std.testing.expect(textnorm.language("xx") == null);
    try std.testing.expect(textnorm.language("en") == null);
}

test "with text_normalization true, each of the reference's cases gives its spoken form byte for byte" {
    try needLibrary();
    for (normalized) |c| try expectSpoken(c, true);
    // test_german_ordinal_dates_keep_the_ordinal compares a prefix
    const de = try spoken(std.testing.allocator, "Am 3. Oktober war es kalt.", "de", true);
    defer std.testing.allocator.free(de);
    try std.testing.expect(std.mem.startsWith(u8, de, "Am dritter oktober"));
}

test "every request language's grammar is one the library has (test_every_server_language_has_a_grammar)" {
    try needLibrary();
    const Fns = struct { zyphra_textnorm_supported: *const fn ([*]const u8, usize) callconv(.c) c_int };
    const f = loadDynLib(Fns, .textnorm, std.testing.allocator).?;
    var distinct: usize = 0;
    for (textnorm.languages, 0..) |l, i| {
        try std.testing.expectEqual(@as(c_int, 1), f.zyphra_textnorm_supported(l.grammar.ptr, l.grammar.len));
        for (textnorm.languages[0..i]) |e| {
            if (std.mem.eql(u8, e.grammar, l.grammar)) break;
        } else distinct += 1;
    }
    // zyphra_textnorm.LANGUAGES of release 0.2.0 has 43 grammars; with each one supported, the two sets are equal
    try std.testing.expectEqual(@as(usize, 43), distinct);
}
