//! ZONOS3's text normalization: Zyphra's zyphra-textnorm library (its C ABI), loaded on first use.
const std = @import("std");
const sys = @import("sys");
const log = @import("log.zig");
const loadDynLib = @import("jpeg_out.zig").loadDynLib;
const Allocator = std.mem.Allocator;

pub const default_language = "en_us";

/// A request language code and its grammar package: `server` of the reference's registries/languages.json.
pub const Language = struct { code: []const u8, grammar: []const u8 };
pub const languages = [_]Language{
    .{ .code = "en_us", .grammar = "en" }, .{ .code = "en_gb", .grammar = "en" }, .{ .code = "fr_fr", .grammar = "fr" },
    .{ .code = "de", .grammar = "de" },    .{ .code = "es", .grammar = "es" },    .{ .code = "it", .grammar = "it" },
    .{ .code = "pt_br", .grammar = "pt" }, .{ .code = "ja", .grammar = "ja" },    .{ .code = "cmn", .grammar = "zh" },
    .{ .code = "ko", .grammar = "ko" },    .{ .code = "ar", .grammar = "ar" },    .{ .code = "bn", .grammar = "bn" },
    .{ .code = "bg", .grammar = "bg" },    .{ .code = "cs", .grammar = "cs" },    .{ .code = "da", .grammar = "da" },
    .{ .code = "nl", .grammar = "nl" },    .{ .code = "fi", .grammar = "fi" },    .{ .code = "ka", .grammar = "ka" },
    .{ .code = "el", .grammar = "el" },    .{ .code = "gu", .grammar = "gu" },    .{ .code = "he", .grammar = "he" },
    .{ .code = "hi", .grammar = "hi" },    .{ .code = "hu", .grammar = "hu" },    .{ .code = "id", .grammar = "id" },
    .{ .code = "kn", .grammar = "kn" },    .{ .code = "ms", .grammar = "ms" },    .{ .code = "ml", .grammar = "ml" },
    .{ .code = "mr", .grammar = "mr" },    .{ .code = "no", .grammar = "nb" },    .{ .code = "or", .grammar = "or" },
    .{ .code = "pl", .grammar = "pl" },    .{ .code = "pa", .grammar = "pa" },    .{ .code = "ro", .grammar = "ro" },
    .{ .code = "ru", .grammar = "ru" },    .{ .code = "sk", .grammar = "sk" },    .{ .code = "sv", .grammar = "sv" },
    .{ .code = "tl", .grammar = "tl" },    .{ .code = "ta", .grammar = "ta" },    .{ .code = "te", .grammar = "te" },
    .{ .code = "th", .grammar = "th" },    .{ .code = "tr", .grammar = "tr" },    .{ .code = "uk", .grammar = "uk" },
    .{ .code = "ur", .grammar = "ur" },    .{ .code = "vi", .grammar = "vi" },
};

/// The codes, comma separated, for a refusal.
pub const codes = blk: {
    var s: []const u8 = "";
    for (languages, 0..) |l, i| s = s ++ (if (i == 0) "" else ", ") ++ l.code;
    break :blk s;
};

/// languages.normalize: a request's `language` stripped, lowercased, `-` read as `_`; null for an unknown code.
pub fn language(value: []const u8) ?Language {
    const name = std.mem.trim(u8, value, " \t\r\n\x0b\x0c");
    for (languages) |l| {
        if (l.code.len != name.len) continue;
        for (l.code, name) |want, got| {
            if (want != (if (got == '-') '_' else std.ascii.toLower(got))) break;
        } else return l;
    }
    return null;
}

const Handle = ?*anyopaque;
const Fns = struct {
    zyphra_textnorm_new: *const fn (?[*]const u8, usize) callconv(.c) Handle,
    zyphra_textnorm_normalize: *const fn (Handle, [*]const u8, usize, [*]const u8, usize) callconv(.c) ?[*:0]u8,
    zyphra_textnorm_free: *const fn (?[*:0]u8) callconv(.c) void,
};
const Ready = struct { fns: Fns, handle: Handle };

var state: union(enum) { unknown, missing, ready: Ready } = .unknown;
var mutex: std.Io.Mutex = .init;

/// The library and one normalizer that keeps no markup tags (the reference servers' own), made by the first call.
fn get(a: Allocator) ?Ready {
    const io = log.io();
    mutex.lockUncancelable(io);
    defer mutex.unlock(io);
    if (state == .unknown) state = blk: {
        const fns = loadDynLib(Fns, .textnorm, a) orelse break :blk .missing;
        const handle = fns.zyphra_textnorm_new(null, 0) orelse break :blk .missing;
        break :blk .{ .ready = .{ .fns = fns, .handle = handle } };
    };
    return if (state == .ready) state.ready else null;
}

/// Whether the library loads here; the first call tries.
pub fn available(a: Allocator) bool {
    return get(a) != null;
}

/// What a request that asks for normalization is told when the library is missing.
pub fn missing() []const u8 {
    return sys.optional_libraries.spec(.textnorm).message;
}

/// A start or pause marker (U+E002..U+E00A) at byte `i`.
fn markerAt(s: []const u8, i: usize) bool {
    return i + 2 < s.len and s[i] == 0xee and s[i + 1] == 0x80 and s[i + 2] >= 0x82 and s[i + 2] <= 0x8a;
}

const group_open = "\xee\x80\x80";
const group_close = "\xee\x80\x81";

/// Each marker as a one-character pronunciation group (U+E000 marker U+E001), which the grammars keep as typed.
fn wrapMarkers(a: Allocator, s: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    while (i < s.len) {
        if (markerAt(s, i)) {
            try out.appendSlice(a, group_open);
            try out.appendSlice(a, s[i..][0..3]);
            try out.appendSlice(a, group_close);
            i += 3;
        } else {
            try out.append(a, s[i]);
            i += 1;
        }
    }
    return out.toOwnedSlice(a);
}

/// Undoes wrapMarkers: U+E000, a marker, U+E001 gives the marker.
fn unwrapMarkers(a: Allocator, s: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    while (i < s.len) {
        if (std.mem.startsWith(u8, s[i..], group_open) and markerAt(s, i + 3) and std.mem.startsWith(u8, s[i + 6 ..], group_close)) {
            try out.appendSlice(a, s[i + 3 ..][0..3]);
            i += 9;
        } else {
            try out.append(a, s[i]);
            i += 1;
        }
    }
    return out.toOwnedSlice(a);
}

/// The spoken form of `text` in `lang`'s grammar (caller frees), as zonos3.tokenizer.textnorm gives it.
pub fn normalize(a: Allocator, text: []const u8, lang: Language) error{ OutOfMemory, NoLibrary }![]u8 {
    const lib = get(a) orelse return error.NoLibrary;
    const wrapped = try wrapMarkers(a, text);
    defer a.free(wrapped);
    const out = lib.fns.zyphra_textnorm_normalize(lib.handle, wrapped.ptr, wrapped.len, lang.grammar.ptr, lang.grammar.len) orelse return error.OutOfMemory;
    defer lib.fns.zyphra_textnorm_free(out);
    return unwrapMarkers(a, std.mem.span(out));
}

test "a language code is read whatever its case, with - for _ and spaces around it" {
    try std.testing.expectEqualStrings("en", language("en_us").?.grammar);
    try std.testing.expectEqualStrings("pt", language(" PT-BR\n").?.grammar);
    try std.testing.expectEqualStrings("nb", language("no").?.grammar);
    try std.testing.expectEqualStrings("zh", language("cmn").?.grammar);
    try std.testing.expect(language("en") == null);
    try std.testing.expect(language("") == null);
    try std.testing.expectEqual(44, languages.len);
    try std.testing.expect(std.mem.startsWith(u8, codes, "en_us, en_gb, fr_fr, "));
    try std.testing.expect(std.mem.endsWith(u8, codes, ", ur, vi"));
}

test "markers are wrapped as pronunciation groups and unwrapped again" {
    const a = std.testing.allocator;
    const text = "one \xee\x80\x82two\xee\x80\x8a \xee\x80\x8b";
    const wrapped = try wrapMarkers(a, text);
    defer a.free(wrapped);
    try std.testing.expectEqualStrings("one \xee\x80\x80\xee\x80\x82\xee\x80\x81two\xee\x80\x80\xee\x80\x8a\xee\x80\x81 \xee\x80\x8b", wrapped);
    const back = try unwrapMarkers(a, wrapped);
    defer a.free(back);
    try std.testing.expectEqualStrings(text, back);
    // a group the text already held around something else stays
    const kept = try unwrapMarkers(a, "\xee\x80\x80ab\xee\x80\x81 \xee\x80\x80\xee\x80\x82");
    defer a.free(kept);
    try std.testing.expectEqualStrings("\xee\x80\x80ab\xee\x80\x81 \xee\x80\x80\xee\x80\x82", kept);
}
