//! Audio input (Info.audio): input_audio and audio_url parts to the engine's encoder, each clip's locator to its rows.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const images = @import("images.zig");
const image_key = @import("image_key.zig");
const Value = json.Value;
const Cx = errors.Cx;

/// A clip's encoded bytes at most (a request's body limit bounds them all: about 12 minutes of 16-bit 16 kHz WAV).
pub const max_audio_bytes = 32 << 20;

/// The audio token: the locator the template writes must be one added token, the one a clip's rows take.
pub fn bindLocator(info: *api.Info, text: @import("model_text.zig").Text) error{InvalidAudioLocator}!void {
    const o = if (info.audio) |*a| a else return;
    o.audio_token = text.tokenId(o.locator) orelse {
        @import("log.zig").line("audio input: the tokenizer has no single token {s}; audio cannot be served", .{o.locator});
        return error.InvalidAudioLocator;
    };
}

/// A request's audio prompt: each clip's rows in place of its locator, the clips, and the key the prompt cache matches.
pub const Expanded = struct {
    ids: []const u32,
    history_len: usize,
    audios: []const api.Audio,
    cache_key: []const u32, // one word per row; a clip's rows carry its content hash
};

/// The type of a normalized audio part: {"type": "audio_url", "audio_url": {"url": a data URL}}.
pub const part_type = "audio_url";

/// The audio_url part a user's input_audio or audio_url part becomes; it reads its own output back unchanged.
pub fn part(cx: *Cx, p: Value, kind: []const u8) errors.Refused!Value {
    for ([_][]const u8{ "image", "images", "image_url", "input_image", "video", "video_url" }) |k| if (json.truthyField(p, k)) return cx.refuse("audio parts cannot contain other media");
    const url = if (std.mem.eql(u8, kind, "input_audio")) try inputAudioUrl(cx, p) else try audioUrl(cx, p);
    const inner = try json.newObject(cx.a);
    try inner.put(cx.a, "url", .{ .string = url });
    const out = try json.newObject(cx.a);
    try out.put(cx.a, "type", .{ .string = part_type });
    try out.put(cx.a, part_type, .{ .object = inner });
    return .{ .object = out };
}

/// OpenAI's {"input_audio": {"data": base64, "format": "wav"}} as a base64 WAV data URL.
fn inputAudioUrl(cx: *Cx, p: Value) errors.Refused![]const u8 {
    const ia = p.get("input_audio") orelse return cx.refuse("an input_audio content part must contain input_audio");
    if (ia != .object) return cx.refuse("input_audio must be an object with data and format");
    const data = ia.strField("data") orelse return cx.refuse("input_audio must contain a base64 data string");
    if (data.len == 0) return cx.refuse("input_audio must contain a base64 data string");
    if (ia.get("format")) |f| if (f != .string or !std.ascii.eqlIgnoreCase(f.string, "wav"))
        return cx.refuse("input_audio format must be wav; this server decodes WAV files only");
    return std.mem.concat(cx.a, u8, &.{ "data:audio/wav;base64,", data });
}

/// The model card's {"audio_url": {"url": "data:audio/wav;base64,..."}} (or the URL alone); remote URLs are refused.
fn audioUrl(cx: *Cx, p: Value) errors.Refused![]const u8 {
    const au = p.get("audio_url") orelse return cx.refuse("an audio_url content part must contain audio_url");
    const u = if (au == .object) au.get("url") orelse return cx.refuse("audio_url must contain a url string") else au;
    if (u != .string or u.string.len == 0) return cx.refuse("audio_url must contain a url string");
    if (!std.ascii.startsWithIgnoreCase(u.string, "data:")) return cx.refuse("audio_url must be a data URL holding a WAV file; remote audio URLs are unsupported");
    return u.string;
}

/// A content part this module wrote: an audio clip's data URL, else null.
fn clipUrl(p: Value) ?[]const u8 {
    if (!p.typeIs(part_type)) return null;
    const au = p.get(part_type) orelse return null;
    return au.strField("url");
}

/// The messages as the template reads them: a user turn's clips become `locator`-newline prefixes before its text.
pub fn forTemplate(a: std.mem.Allocator, messages: Value, locator: []const u8) !Value {
    if (messages != .array) return messages;
    var out: ?[]Value = null;
    for (messages.array, 0..) |m, i| {
        const content = m.get("content") orelse continue;
        if (content != .array) continue;
        var clips: usize = 0;
        for (content.array) |p| clips += @intFromBool(clipUrl(p) != null);
        if (clips == 0) continue;
        // the template moves leading "<locator>\n" prefixes behind the text: "<text> <locator> <locator>"
        var text: std.ArrayList(u8) = .empty;
        for (0..clips) |_| try text.print(a, "{s}\n", .{locator});
        for (content.array) |p| if (p.typeIs("text")) try text.appendSlice(a, p.strField("text") orelse "");
        const item = try json.copyObject(a, m.object);
        try item.put(a, "content", .{ .string = text.items });
        const list = out orelse try a.dupe(Value, messages.array);
        list[i] = .{ .object = item };
        out = list;
    }
    return if (out) |list| .{ .array = list } else messages;
}

/// The clips' data URLs in the order the template renders them.
fn clipUrls(cx: *Cx, messages: Value) errors.Refused![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    if (messages != .array) return out.items;
    for (messages.array) |m| {
        const content = m.get("content") orelse continue;
        if (content != .array) continue;
        for (content.array) |p| if (clipUrl(p)) |u| try out.append(cx.a, u);
    }
    return out.items;
}

/// The prompt with each locator token expanded to its clip's rows, the clips at their positions, and the cache key.
pub fn expand(offer: ?api.AudioOffer, cx: *Cx, messages: Value, ids: []const u32, history_len: usize) errors.Refused!?Expanded {
    const o = offer orelse return null;
    const urls = try clipUrls(cx, messages);
    const marks = std.mem.count(u32, ids, &.{o.audio_token});
    if (urls.len == 0 and marks == 0) return null;
    if (marks != urls.len) return cx.refuse(try std.fmt.allocPrint(cx.a, "the prompt holds {d} audio locators for {d} audio parts; message text must not begin with {s} and a newline", .{ marks, urls.len, o.locator }));
    if (urls.len > o.max_audios) return cx.refuse(try std.fmt.allocPrint(cx.a, "a request may hold at most {d} audio clips; it has {d}", .{ o.max_audios, urls.len }));
    const prepared = try cx.a.alloc(api.PreparedAudio, urls.len);
    var left = o.audio_tokens;
    for (urls, prepared, 0..) |u, *clip, i| {
        const bytes = try images.dataBytesOf(cx, u, max_audio_bytes, "audio");
        clip.* = o.prepare(o.ctx, cx.a, bytes, left) catch |e| return refusal(cx, i, e, left);
        if (clip.tokens == 0 or clip.tokens > left) return refusal(cx, i, error.AudioTooLong, left);
        left -= clip.tokens;
    }
    var out_ids: std.ArrayList(u32) = .empty;
    var key: std.ArrayList(u32) = .empty;
    var clips: std.ArrayList(api.Audio) = .empty;
    var hist = history_len;
    var k: usize = 0;
    var close: ?u32 = null; // a one-row clip's identity half that the word after it carries
    for (ids, 0..) |t, src| {
        if (t != o.audio_token) {
            try out_ids.append(cx.a, t);
            try key.append(cx.a, close orelse t);
            close = null;
            continue;
        }
        const clip = prepared[k];
        k += 1;
        try clips.append(cx.a, .{ .samples = clip.samples, .at = @intCast(out_ids.items.len), .tokens = clip.tokens, .hash = clip.hash });
        try out_ids.appendNTimes(cx.a, t, clip.tokens);
        try key.append(cx.a, image_key.open(clip.hash));
        for (1..clip.tokens) |j| try key.append(cx.a, if (j == 1) image_key.close(clip.hash) else image_key.row(clip.hash, j));
        close = if (clip.tokens == 1) image_key.close(clip.hash) else null;
        if (src < history_len) hist += clip.tokens - 1;
    }
    return .{ .ids = out_ids.items, .history_len = hist, .audios = clips.items, .cache_key = key.items };
}

fn refusal(cx: *Cx, i: usize, err: anyerror, budget: u32) errors.Refused {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.AudioTooLong => cx.refuse(try std.fmt.allocPrint(cx.a, "audio {d} is longer than this request's audio budget of {d} rows ({d:.0} seconds)", .{ i + 1, budget, @as(f64, @floatFromInt(budget)) * 0.08 })),
        error.WavNot16k => cx.refuse(try std.fmt.allocPrint(cx.a, "audio {d} must be sampled at 16 kHz", .{i + 1})),
        error.WavNotFinite => cx.refuse(try std.fmt.allocPrint(cx.a, "audio {d} has a sample that is NaN or infinite", .{i + 1})),
        error.WavNotMono => cx.refuse(try std.fmt.allocPrint(cx.a, "audio {d} must be mono", .{i + 1})),
        error.AudioTooShort => cx.refuse(try std.fmt.allocPrint(cx.a, "audio {d} is shorter than 20 ms", .{i + 1})),
        else => cx.refuse(try std.fmt.allocPrint(cx.a, "audio {d} could not be decoded; send a WAV file (16, 24 or 32-bit PCM or 32-bit float, 16 kHz, mono)", .{i + 1})),
    };
}

/// The tests' clips: a data URL's bytes are its samples, one row a byte, and their hash names them.
fn fakePrepare(_: *anyopaque, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedAudio {
    if (bytes.len > max_tokens) return error.AudioTooLong;
    return .{ .samples = try a.alloc(f32, bytes.len), .tokens = @intCast(bytes.len), .hash = std.hash.Wyhash.hash(0, bytes) };
}

var test_ctx: u8 = 0;
const test_offer: api.AudioOffer = .{ .ctx = &test_ctx, .prepare = fakePrepare, .audio_token = 131072, .locator = "<|audio|>", .audio_tokens = 64, .max_audios = 4 };

fn testMessages(a: std.mem.Allocator, text: []const u8) !Value {
    return switch (try json.parseText(a, text)) {
        .ok => |m| m,
        .err => error.TestUnexpectedResult,
    };
}

const card_prompt = "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only.";
// prompt_ls_1272_135031_0009.json's template_ids: the card's request, its locator at 43, its history 47 ids
const dump_ids = [_]u32{ 10, 25708, 1010, 11, 1010, 10, 3263, 1010, 32535, 15661, 1046, 26242, 1058, 1435, 29930, 1046, 1488, 3592, 1058, 10564, 94503, 1044, 3907, 12369, 1428, 1321, 124440, 1046, 6392, 33361, 1058, 34315, 65656, 1321, 8961, 3229, 1046, 27039, 1058, 21482, 2342, 1046, 1032, 131072, 11, 1010, 10, 1503, 19464, 1010, 12, 13 };

test "the card's request: one locator after the text, expanded to the clip's rows as the dump lays them out" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const msgs = try testMessages(a, "[{\"role\": \"system\", \"content\": \"\"}, {\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"" ++ card_prompt ++
        "\"}, {\"type\": \"audio_url\", \"audio_url\": {\"url\": \"data:audio/wav,abcdefghijklmnopqrstuvwx\"}}]}]");
    const view = try forTemplate(a, msgs, "<|audio|>");
    try std.testing.expectEqualStrings("<|audio|>\n" ++ card_prompt, view.array[1].get("content").?.string);
    try std.testing.expectEqualStrings("", view.array[0].get("content").?.string);
    const x = (try expand(test_offer, &cx, msgs, &dump_ids, 47)).?;
    try std.testing.expectEqual(@as(usize, 75), x.ids.len);
    try std.testing.expectEqualSlices(u32, dump_ids[0..43], x.ids[0..43]);
    for (x.ids[43..67]) |t| try std.testing.expectEqual(@as(u32, 131072), t);
    try std.testing.expectEqualSlices(u32, dump_ids[44..], x.ids[67..]);
    try std.testing.expectEqual(@as(usize, 70), x.history_len);
    try std.testing.expectEqual(@as(usize, 1), x.audios.len);
    try std.testing.expectEqual(@as(u32, 43), x.audios[0].at);
    try std.testing.expectEqual(@as(u32, 24), x.audios[0].tokens);
    try std.testing.expectEqual(x.ids.len, x.cache_key.len);
    try std.testing.expectEqual(image_key.open(x.audios[0].hash), x.cache_key[43]);
    try std.testing.expectEqual(image_key.close(x.audios[0].hash), x.cache_key[44]);
    try std.testing.expectEqualSlices(u32, x.ids[67..], x.cache_key[67..]);
}

test "several clips: each locator takes its own clip's rows in order, keyed by the clip's content" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const msgs = try testMessages(a,
        \\[{"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "data:,abc"}}, {"type": "audio_url", "audio_url": {"url": "data:,d"}}]},
        \\ {"role": "assistant", "content": "y"},
        \\ {"role": "user", "content": [{"type": "text", "text": "and"}, {"type": "audio_url", "audio_url": {"url": "data:,abc"}}]}]
    );
    const view = try forTemplate(a, msgs, "<|audio|>");
    try std.testing.expectEqualStrings("<|audio|>\n<|audio|>\n", view.array[0].get("content").?.string);
    try std.testing.expect(view.array[1].object == msgs.array[1].object);
    try std.testing.expectEqualStrings("<|audio|>\nand", view.array[2].get("content").?.string);
    const L = 131072;
    const x = (try expand(test_offer, &cx, msgs, &[_]u32{ 1, L, 2, L, 3, 4, 5, L, 6 }, 5)).?;
    try std.testing.expectEqualSlices(u32, &.{ 1, L, L, L, 2, L, 3, 4, 5, L, L, L, 6 }, x.ids);
    try std.testing.expectEqual(@as(usize, 7), x.history_len); // the first message's clips are history, the last's not
    try std.testing.expectEqual(@as(usize, 3), x.audios.len);
    for ([_]u32{ 1, 5, 9 }, [_]u32{ 3, 1, 3 }, x.audios) |at, n, clip| {
        try std.testing.expectEqual(at, clip.at);
        try std.testing.expectEqual(n, clip.tokens);
    }
    try std.testing.expectEqualSlices(u32, x.cache_key[1..4], x.cache_key[9..12]); // the same clip, the same words
    try std.testing.expect(x.cache_key[1] != x.cache_key[5]);
    try std.testing.expectEqual(image_key.close(x.audios[1].hash), x.cache_key[6]); // a one-row clip's other half
    try std.testing.expectEqual(@as(u32, 2), x.cache_key[4]);
}

test "a clip past the request's remaining rows, or one clip too many, is refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const msgs = try testMessages(a,
        \\[{"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "data:,abcdefgh"}}, {"type": "audio_url", "audio_url": {"url": "data:,abcdefghij"}}]}]
    );
    const L = 131072;
    var small = test_offer;
    small.audio_tokens = 12;
    try std.testing.expectError(error.Refused, expand(small, &cx, msgs, &[_]u32{ 1, L, 2, L }, 0));
    try std.testing.expectEqualStrings("audio 2 is longer than this request's audio budget of 4 rows (0 seconds)", cx.message);
    var one = test_offer;
    one.max_audios = 1;
    try std.testing.expectError(error.Refused, expand(one, &cx, msgs, &[_]u32{ 1, L, 2, L }, 0));
    try std.testing.expectEqualStrings("a request may hold at most 1 audio clips; it has 2", cx.message);
    try std.testing.expectError(error.Refused, expand(test_offer, &cx, msgs, &[_]u32{ 1, L, 2 }, 0));
    try std.testing.expectEqualStrings("the prompt holds 1 audio locators for 2 audio parts; message text must not begin with <|audio|> and a newline", cx.message);
}

test "messages without audio pass through; a locator with no clip is refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const msgs = try testMessages(a,
        \\[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    );
    const view = try forTemplate(a, msgs, "<|audio|>");
    try std.testing.expect(view.array.ptr == msgs.array.ptr);
    try std.testing.expect(try expand(test_offer, &cx, msgs, &[_]u32{ 1, 2, 3 }, 2) == null);
    try std.testing.expect(try expand(null, &cx, msgs, &[_]u32{ 1, 131072 }, 0) == null);
    try std.testing.expectError(error.Refused, expand(test_offer, &cx, msgs, &[_]u32{ 1, 131072, 3 }, 0));
}

test "input_audio and audio_url parts normalize to one data URL part, twice alike; other sources are refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const parts = try testMessages(a,
        \\[{"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
        \\ {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,UklGRg=="}},
        \\ {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "mp3"}},
        \\ {"type": "audio_url", "audio_url": {"url": "https://example.com/a.wav"}},
        \\ {"type": "audio_url", "audio_url": {}},
        \\ {"type": "input_audio", "input_audio": {"data": ""}}]
    );
    for (parts.array[0..2], [_][]const u8{ "input_audio", "audio_url" }) |p, kind| {
        const once = try part(&cx, p, kind);
        const twice = try part(&cx, once, part_type);
        try std.testing.expectEqualStrings("data:audio/wav;base64,UklGRg==", clipUrl(twice).?);
        try std.testing.expectEqualStrings(part_type, twice.strField("type").?);
    }
    for (parts.array[2..], [_][]const u8{ "input_audio", "audio_url", "audio_url", "input_audio" }) |p, kind|
        try std.testing.expectError(error.Refused, part(&cx, p, kind));
    try std.testing.expectEqualStrings("input_audio must contain a base64 data string", cx.message);
}
