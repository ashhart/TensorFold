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

pub const Expanded = struct {
    ids: []const u32,
    history_len: usize,
    audios: []const api.Audio,
    cache_key: []const u32,
};

/// The part type normalization writes: an audio_url part with a data URL (vLLM's, and the model card's).
pub const part_type = "audio_url";

/// An input_audio or data-URL audio_url part as {"type": "audio_url", "audio_url": {"url"}}, as `expand` reads it.
pub fn part(cx: *Cx, p: Value, kind: []const u8) errors.Refused!Value {
    var url: []const u8 = "";
    if (std.mem.eql(u8, kind, "input_audio")) {
        const ia = p.get("input_audio") orelse return cx.refuse("an input_audio content part must contain input_audio");
        const data = if (ia == .object) ia.get("data") else null;
        if (data == null or data.? != .string or data.?.string.len == 0) return cx.refuse("input_audio must contain base64 data");
        const format = ia.get("format");
        if (format) |f| if (f != .string or !std.ascii.eqlIgnoreCase(f.string, "wav")) return cx.refuse("input_audio format must be wav");
        url = try std.mem.concat(cx.a, u8, &.{ "data:audio/wav;base64,", data.?.string });
    } else {
        const au = p.get("audio_url") orelse return cx.refuse("an audio_url content part must contain audio_url");
        const u = if (au == .string) au else if (au == .object) au.get("url") orelse Value.null else Value.null;
        if (u != .string or u.string.len == 0) return cx.refuse("audio_url must contain a url string");
        if (!std.ascii.startsWithIgnoreCase(u.string, "data:")) return cx.refuse("audio requires a data URL (data:audio/wav;base64,...)");
        url = u.string;
    }
    for ([_][]const u8{ "image", "images", "image_url", "input_image", "video", "video_url" }) |k| if (json.truthyField(p, k)) return cx.refuse("audio parts cannot contain other media");
    const inner = try json.newObject(cx.a);
    try inner.put(cx.a, "url", .{ .string = url });
    const out = try json.newObject(cx.a);
    try out.put(cx.a, "type", .{ .string = part_type });
    try out.put(cx.a, "audio_url", .{ .object = inner });
    return .{ .object = out };
}

fn isAudio(p: Value) bool {
    const kind = p.get("type") orelse return false;
    return kind == .string and std.mem.eql(u8, kind.string, part_type);
}

/// `messages` for the template: a user message with audio becomes vLLM's string form, a locator a clip, then its text.
pub fn forTemplate(a: std.mem.Allocator, messages: Value, locator: []const u8) !Value {
    if (messages != .array) return messages;
    const out = try a.dupe(Value, messages.array);
    for (out) |*m| {
        const content = m.get("content") orelse continue;
        if (content != .array) continue;
        var clips: usize = 0;
        var text: std.ArrayList(u8) = .empty;
        for (content.array) |p| {
            if (isAudio(p)) {
                clips += 1;
            } else if (p.get("text")) |t| if (t == .string) try text.appendSlice(a, t.string);
        }
        if (clips == 0) continue;
        var s: std.ArrayList(u8) = .empty;
        for (0..clips) |_| {
            try s.appendSlice(a, locator);
            try s.append(a, '\n');
        }
        try s.appendSlice(a, text.items);
        const copy = try json.copyObject(a, m.object);
        try copy.put(a, "content", .{ .string = s.items });
        m.* = .{ .object = copy };
    }
    return .{ .array = out };
}

/// The prompt with each clip's placeholder expanded to its rows, its clips and its cache ids; null without audio.
pub fn expand(offer: ?api.AudioOffer, cx: *Cx, messages: Value, ids: []const u32, history_len: usize) errors.Refused!?Expanded {
    const o = offer orelse return null;
    var urls: std.ArrayList([]const u8) = .empty;
    if (messages == .array) for (messages.array) |m| {
        const content = m.get("content") orelse continue;
        if (content != .array) continue;
        for (content.array) |p| if (isAudio(p)) try urls.append(cx.a, p.get("audio_url").?.get("url").?.string);
    };
    var marks: usize = 0;
    for (ids) |t| marks += @intFromBool(t == o.audio_token);
    const n = urls.items.len;
    if (n == 0 and marks == 0) return null;
    // the template writes a locator only for the ones content starts with: text that spells one adds a placeholder
    if (marks > n) return cx.refuse("message text cannot spell the audio locator before a message's text");
    if (marks < n) return cx.fail(.server, "the chat template wrote {d} audio placeholders for {d} audio parts", .{ marks, n });
    if (n > o.max_audios) return cx.refuse(try std.fmt.allocPrint(cx.a, "a request may hold at most {d} audio clips; it has {d}", .{ o.max_audios, n }));
    const budget: u32 = o.audio_tokens / @as(u32, @intCast(n));
    const prepared = try cx.a.alloc(api.PreparedAudio, n);
    for (urls.items, prepared, 0..) |url, *p, i| {
        const bytes = try images.dataBytesOf(cx, url, max_audio_bytes, "audio");
        p.* = o.prepare(o.ctx, cx.a, bytes, budget) catch |e| return refusal(cx, i, e, budget);
        if (p.tokens == 0 or p.tokens > budget) return refusal(cx, i, error.AudioTooLong, budget);
    }
    var out_ids: std.ArrayList(u32) = .empty;
    var key: std.ArrayList(u32) = .empty;
    var clips: std.ArrayList(api.Audio) = .empty;
    var hist = history_len;
    var k: usize = 0;
    for (ids, 0..) |t, src| {
        const before = out_ids.items.len;
        if (t == o.audio_token) {
            const c = prepared[k];
            k += 1;
            try clips.append(cx.a, .{ .samples = c.samples, .at = @intCast(out_ids.items.len), .tokens = c.tokens, .hash = c.hash });
            // the first two rows carry the clip's identity whole, the rest a hash of it and the row
            for (0..c.tokens) |j| {
                try out_ids.append(cx.a, t);
                try key.append(cx.a, switch (j) {
                    0 => image_key.open(c.hash),
                    1 => image_key.close(c.hash),
                    else => image_key.row(c.hash, j),
                });
            }
        } else try images.both(cx.a, &out_ids, &key, t);
        if (src < history_len) hist += out_ids.items.len - before - 1;
    }
    return .{ .ids = out_ids.items, .history_len = hist, .audios = clips.items, .cache_key = key.items };
}

fn refusal(cx: *Cx, i: usize, err: anyerror, budget: u32) errors.Refused {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.AudioTooLong => cx.refuse(try std.fmt.allocPrint(cx.a, "audio {d} is longer than this request's audio budget of {d} rows ({d:.0} seconds)", .{ i + 1, budget, @as(f64, @floatFromInt(budget)) * 0.08 })),
        error.WavRate => cx.refuse(try std.fmt.allocPrint(cx.a, "audio {d} must be sampled at 16 kHz", .{i + 1})),
        error.WavNotMono => cx.refuse(try std.fmt.allocPrint(cx.a, "audio {d} must be mono", .{i + 1})),
        error.AudioTooShort => cx.refuse(try std.fmt.allocPrint(cx.a, "audio {d} is shorter than 10 ms", .{i + 1})),
        else => cx.refuse(try std.fmt.allocPrint(cx.a, "audio {d} could not be decoded; send a WAV file (16-bit PCM or 32-bit float, 16 kHz, mono)", .{i + 1})),
    };
}

fn fakePrepare(_: *anyopaque, a: std.mem.Allocator, bytes: []const u8, _: u32) anyerror!api.PreparedAudio {
    // the test's clips: one row a byte, the first byte as their identity
    return .{ .samples = try a.alloc(f32, 0), .tokens = @intCast(bytes.len), .hash = bytes[0] };
}

test "a clip's placeholder becomes its rows, keyed by the clip" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    var dummy: u8 = 0;
    const o: api.AudioOffer = .{ .ctx = &dummy, .prepare = fakePrepare, .audio_token = 9, .locator = "<|audio|>", .audio_tokens = 64, .max_audios = 2 };
    const raw = (try json.parse(a, "[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Transcribe.\"},{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"AQID\",\"format\":\"wav\"}}]}]")).ok;
    const p = try part(&cx, raw.array[0].get("content").?.array[1], "input_audio");
    raw.array[0].get("content").?.array[1] = p;
    // the template's view: the locator first
    const t = try forTemplate(a, raw, "<|audio|>");
    try std.testing.expectEqualStrings("<|audio|>\nTranscribe.", t.array[0].get("content").?.string);
    const x = (try expand(o, &cx, raw, &.{ 1, 2, 9, 3 }, 3)).?;
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 9, 9, 9, 3 }, x.ids);
    try std.testing.expectEqual(@as(usize, 1), x.audios.len);
    try std.testing.expectEqual(@as(u32, 2), x.audios[0].at);
    try std.testing.expectEqual(@as(u32, 3), x.audios[0].tokens);
    try std.testing.expectEqual(@as(usize, 5), x.history_len);
    try std.testing.expectEqual(image_key.open(1), x.cache_key[2]);
    // a placeholder text spells, with no clip for it, is refused
    try std.testing.expectError(error.Refused, expand(o, &cx, raw, &.{ 9, 9, 3 }, 0));
}
