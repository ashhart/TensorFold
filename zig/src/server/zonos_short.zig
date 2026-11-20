//! ZONOS3's short texts on the speech route (zonos3.tts.short_text.prompt): a few words get a pause and a filler.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const text_mod = @import("model_text.zig");
const req = @import("zonos_request");
const tokens = @import("zonos_tokens.zig");
const Allocator = std.mem.Allocator;

pub const field = "short_text";
pub const Config = req.short_text.Config;

/// --short-text: the reference package's registries/short_text.json.
pub fn load(a: Allocator, io: std.Io, path: []const u8) !*const Config {
    const config = try a.create(Config);
    config.* = try Config.parse(a, try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22)));
    return config;
}

/// The text to speak (the words, the pause marker, the filler) and what the cut needs.
pub const Made = struct { text: []const u8, short: api.SpeechLaci.Short };

/// Python's str.isspace for one character.
fn isSpace(c: u21) bool {
    return switch (c) {
        '\t'...'\r', 0x1c...0x20, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

fn blank(s: []const u8) bool {
    var it = (std.unicode.Utf8View.init(s) catch return false).iterator();
    while (it.nextCodepoint()) |c| if (!isSpace(c)) return false;
    return true;
}

/// len(text.split()): the runs of characters that are not whitespace.
fn words(s: []const u8) usize {
    var n: usize = 0;
    var inside = false;
    var it = (std.unicode.Utf8View.init(s) catch return 0).iterator();
    while (it.nextCodepoint()) |c| {
        const space = isSpace(c);
        n += @intFromBool(!space and !inside);
        inside = !space;
    }
    return n;
}

/// `spoken` (normalized) as a short text, or null; `allowed`: the request may run under LACI alone.
pub fn plan(config: ?*const Config, cx: *errors.Cx, raw: json.Value, text: text_mod.Text, spec: ?tokens.Spec, spoken: []const u8, language: []const u8, allowed: bool) errors.Refused!?Made {
    if (raw.field(field)) |v| {
        const mode = v.str() orelse return cx.refuse("ZONOS3 request: short_text must be \"auto\" or \"off\"");
        if (std.mem.eql(u8, mode, "off")) return null;
        if (!std.mem.eql(u8, mode, "auto")) return cx.refuse("ZONOS3 request: short_text must be \"auto\" or \"off\"");
    }
    const c = config orelse return null;
    const sp = spec orelse return null;
    if (!allowed or tokens.hasMarkers(spoken) or words(spoken) > c.max_words) return null;
    // the pause marker must be one of the release's pause tokens
    const m = c.pause_marker;
    if (m.len != 3 or m[0] != 0xee or m[1] != 0x80 or m[2] < 0x83 or m[2] - 0x83 >= sp.pause_ids.len) return null;
    const filler = c.filler(language) orelse return null;
    const said = try tokens.payload(cx, text, spec, spoken);
    const rest = try tokens.payload(cx, text, spec, filler);
    if (said.len == 0 or said.len * c.max_ratio[1] > rest.len * c.max_ratio[0]) return null;
    // the filler's first token that is not whitespace: text-span index 0 is BOS, then the words, then the pause
    const first = for (rest, 0..) |id, i| {
        const piece = try text.decode(cx.a, &.{id});
        if (!blank(piece)) break i;
    } else return null;
    return .{
        .text = try std.mem.concat(cx.a, u8, &.{ spoken, m, filler }),
        .short = .{ .filler_token = @intCast(1 + said.len + 1 + first), .lookback = @intCast(c.lookback), .min_voiced = @intCast(c.min_voiced), .max_frames = @intCast(c.max_frames), .max_retries = @intCast(c.max_retries), .threshold = c.threshold, .weights = c.weights },
    };
}

test "words are runs of non-whitespace, as Python's split counts them" {
    const t = std.testing;
    try t.expectEqual(@as(usize, 2), words("  Good\tmorning.\n"));
    try t.expectEqual(@as(usize, 1), words("Hello."));
    try t.expectEqual(@as(usize, 0), words(" \u{3000} "));
    try t.expectEqual(@as(usize, 3), words("a\u{a0}b c"));
    try t.expect(blank(" \t") and !blank(" a"));
}

test "a text of a word or two gets the pause marker and its language's filler; others do not qualify" {
    const t = std.testing;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const weights: [req.config.codebooks * req.config.codebook_size]i8 = @splat(0);
    const c: Config = .{ .pause_marker = "\xee\x80\x84", .max_words = 2, .max_ratio = .{ 2, 5 }, .lookback = 2, .min_voiced = 3, .max_frames = 15, .max_retries = 2, .threshold = 3, .weights = &weights, .fillers = &.{.{ .language = "en_us", .text = " The filler sentence goes on for a while." }} };
    const spec: tokens.Spec = .{ .open_id = 8197, .close_id = 8198, .symbols = &.{}, .start_id = 8199, .pause_ids = &.{ 8200, 8201 } };
    const text = tokens.Bytes.text();
    const raw = (try json.parse(a, "{}")).ok;
    const made = (try plan(&c, &cx, raw, text, spec, "Hi.", "en_us", true)).?;
    try t.expectEqualStrings("Hi.\xee\x80\x84 The filler sentence goes on for a while.", made.text);
    // BOS, three bytes, the pause, the filler's space: its first word is the seventh text row
    try t.expectEqual(@as(u32, 6), made.short.filler_token);
    try t.expect(try plan(&c, &cx, raw, text, spec, "One two three.", "en_us", true) == null);
    try t.expect(try plan(&c, &cx, raw, text, spec, "A rather long word.", "en_us", true) == null);
    try t.expect(try plan(&c, &cx, raw, text, spec, "Hi.", "fr_fr", true) == null);
    try t.expect(try plan(&c, &cx, raw, text, spec, "Hi.", "en_us", false) == null);
    try t.expect(try plan(&c, &cx, raw, text, null, "Hi.", "en_us", true) == null);
    try t.expect(try plan(null, &cx, raw, text, spec, "Hi.", "en_us", true) == null);
    try t.expect(try plan(&c, &cx, raw, text, spec, "Hi.\xee\x80\x83", "en_us", true) == null);
    try t.expect(try plan(&c, &cx, (try json.parse(a, "{\"short_text\": \"off\"}")).ok, text, spec, "Hi.", "en_us", true) == null);
    try t.expectError(error.Refused, plan(&c, &cx, (try json.parse(a, "{\"short_text\": \"on\"}")).ok, text, spec, "Hi.", "en_us", true));
}
