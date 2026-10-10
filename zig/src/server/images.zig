//! Image input (--vision): a request's image_url parts to the engine's tower input, and the rendered prompt's
//! placeholders expanded to each image's tokens, as TensorFold 0.6.6's vision path does for GLM-5.3-Flash.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const errors = @import("errors.zig");
const messages_mod = @import("messages.zig");
const model_text = @import("model_text.zig");
const Value = json.Value;
const Cx = errors.Cx;

/// images.py's limits: an image's and a request's encoded bytes, and the visual tokens of a low-detail image.
pub const max_image_bytes = 10 << 20;
pub const max_request_bytes = 20 << 20;
const low_detail_tokens = 256;

pub const Expanded = struct {
    ids: []const u32,
    history_len: usize,
    images: []const api.Image,
    cache_key: []const u32,
};

const Part = struct { url: []const u8, low: bool };

/// The image parts of normalized `messages`, in the order the template meets them.
fn parts(cx: *Cx, messages: Value) errors.Refused![]const Part {
    var out: std.ArrayList(Part) = .empty;
    if (messages != .array) return out.items;
    for (messages.array) |m| {
        const content = m.get("content") orelse continue;
        if (content != .array) continue;
        for (content.array) |p| {
            const kind = p.get("type") orelse continue;
            if (kind != .string or !std.mem.eql(u8, kind.string, "image_url")) continue;
            const iu = p.get("image_url").?;
            const detail = iu.get("detail");
            try out.append(cx.a, .{ .url = iu.get("url").?.string, .low = detail != null and std.mem.eql(u8, detail.?.string, "low") });
        }
    }
    return out.items;
}

/// A data URL's bytes (base64 or percent-encoded); remote URLs are refused (the native server fetches nothing).
pub fn dataBytes(cx: *Cx, url: []const u8) errors.Refused![]const u8 {
    if (std.ascii.startsWithIgnoreCase(url, "http://") or std.ascii.startsWithIgnoreCase(url, "https://"))
        return cx.refuse("image URLs are not fetched by this server; send the image inline as a data: URL (data:image/png;base64,...)");
    if (!std.ascii.startsWithIgnoreCase(url, "data:")) return cx.refuse("invalid image data URL");
    const comma = std.mem.indexOfScalar(u8, url, ',') orelse return cx.refuse("invalid image data URL");
    const header = url[5..comma];
    const payload = url[comma + 1 ..];
    if (header.len > 256) return cx.refuse("invalid image data URL");
    var it = std.mem.splitScalar(u8, header, ';');
    _ = it.next();
    const enc = it.next();
    if (it.next() != null or (enc != null and !std.ascii.eqlIgnoreCase(enc.?, "base64"))) return cx.refuse("image data URL supports only optional base64 encoding");
    if (enc != null) {
        if (payload.len > 4 * ((max_image_bytes + 2) / 3)) return cx.refuse("image data URL exceeds the encoded byte limit");
        const d = std.base64.standard.Decoder;
        const n = d.calcSizeForSlice(payload) catch return cx.refuse("invalid image data URL encoding");
        const out = try cx.a.alloc(u8, n);
        d.decode(out, payload) catch return cx.refuse("invalid image data URL encoding");
        if (out.len == 0 or out.len > max_image_bytes) return cx.refuse("image is empty or exceeds the encoded byte limit");
        return out;
    }
    if (payload.len > max_image_bytes * 3) return cx.refuse("image data URL exceeds the encoded byte limit");
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < payload.len) : (i += 1) {
        if (payload[i] == '%' and i + 2 < payload.len) {
            const b = std.fmt.parseInt(u8, payload[i + 1 .. i + 3], 16) catch return cx.refuse("invalid image data URL encoding");
            try out.append(cx.a, b);
            i += 2;
        } else try out.append(cx.a, payload[i]);
    }
    if (out.items.len == 0 or out.items.len > max_image_bytes) return cx.refuse("image is empty or exceeds the encoded byte limit");
    return out.items;
}

pub const Format = enum { png, jpeg, gif, webp, other };

/// The image's format from its first bytes, before any decoder sees it.
pub fn sniff(bytes: []const u8) Format {
    if (std.mem.startsWith(u8, bytes, "\x89PNG\r\n\x1a\n")) return .png;
    if (std.mem.startsWith(u8, bytes, "\xff\xd8\xff")) return .jpeg;
    if (std.mem.startsWith(u8, bytes, "GIF87a") or std.mem.startsWith(u8, bytes, "GIF89a")) return .gif;
    if (bytes.len >= 12 and std.mem.eql(u8, bytes[0..4], "RIFF") and std.mem.eql(u8, bytes[8..12], "WEBP")) return .webp;
    return .other;
}

fn refusal(cx: *Cx, i: usize, err: anyerror) errors.Refused {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.ImageTooLarge => cx.refuse(try std.fmt.allocPrint(cx.a, "image {d} exceeds the limits: 10 MB encoded, 8192 pixels a side, 16 megapixels", .{i + 1})),
        error.Animated => cx.refuse(try std.fmt.allocPrint(cx.a, "image {d}: animated and multipage images are unsupported; send a single frame", .{i + 1})),
        error.ImageTooSmall => cx.refuse(try std.fmt.allocPrint(cx.a, "image {d}: the visual-token budget is too small for an image", .{i + 1})),
        else => cx.refuse(try std.fmt.allocPrint(cx.a, "image {d} could not be decoded; send PNG or JPEG", .{i + 1})),
    };
}

/// A quoted marker's stand-in while the template renders: two private-use code points, the second naming the marker.
const sentinel = "\u{F0000}";

/// The image markers' text (Vision.markers), longest first so one never matches inside another.
pub fn markerText(text: model_text.Text, a: std.mem.Allocator, v: api.Vision) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    for (v.markers) |id| {
        const s = try text.tokenString(a, id);
        if (s.len > 0) try out.append(a, s);
    }
    std.mem.sort([]const u8, out.items, {}, struct {
        fn longer(_: void, x: []const u8, y: []const u8) bool {
            return x.len > y.len;
        }
    }.longer);
    return out.items;
}

/// `messages` with each marker the text spells (a user's words, a tool's output) replaced by its sentinel, so only
/// the template's own markers, one wrapper per image part, render as the image tokens (#511).
pub fn shield(a: std.mem.Allocator, v: Value, markers: []const []const u8) !Value {
    return switch (v) {
        .string => |s| .{ .string = try shieldText(a, s, markers) },
        .array => |items| blk: {
            const out = try a.alloc(Value, items.len);
            for (items, out) |x, *o| o.* = try shield(a, x, markers);
            break :blk .{ .array = out };
        },
        .object => |o| blk: {
            const out = try json.newObject(a);
            var it = o.iterator();
            while (it.next()) |kv| try out.put(a, kv.key_ptr.*, try shield(a, kv.value_ptr.*, markers));
            break :blk .{ .object = out };
        },
        else => v,
    };
}

fn shieldText(a: std.mem.Allocator, s: []const u8, markers: []const []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    var changed = false;
    outer: while (i < s.len) {
        for (markers, 0..) |m, k| if (std.mem.startsWith(u8, s[i..], m)) {
            try out.appendSlice(a, sentinel);
            try out.appendSlice(a, try std.fmt.allocPrint(a, "{u}", .{@as(u21, @intCast(0xF0001 + k))}));
            i += m.len;
            changed = true;
            continue :outer;
        };
        try out.append(a, s[i]);
        i += 1;
    }
    return if (changed) out.items else s;
}

/// The rendered prompt's ids: its text as the tokenizer reads it, each sentinel's marker as ordinary text tokens.
/// Added tokens split the text anyway, so a prompt without quoted markers gets exactly `encode`'s ids.
pub fn encodeShielded(text: model_text.Text, a: std.mem.Allocator, rendered: []const u8, markers: []const []const u8) model_text.Error![]u32 {
    var ids: std.ArrayList(u32) = .empty;
    var rest = rendered;
    while (std.mem.indexOf(u8, rest, sentinel)) |at| {
        const after = rest[at + sentinel.len ..];
        const cp: u21 = if (after.len >= 4) std.unicode.utf8Decode(after[0..4]) catch 0 else 0;
        const k: usize = if (cp >= 0xF0001) cp - 0xF0001 else markers.len;
        if (k >= markers.len) { // not one of ours: the code point is ordinary text
            try ids.appendSlice(a, try text.encode(a, rest[0 .. at + sentinel.len], false));
            rest = after;
            continue;
        }
        try ids.appendSlice(a, try text.encode(a, rest[0..at], false));
        try ids.appendSlice(a, try text.encodePlain(a, markers[k]));
        rest = after[4..];
    }
    try ids.appendSlice(a, try text.encode(a, rest, false));
    return ids.items;
}

/// The prompt with each image's placeholder expanded to its tokens, the images for the engine and the prompt as the
/// cache matches it; null when the engine reads no images. Images are counted from the request's image parts; the
/// rendered placeholders are the template's alone (text that spells a marker was shielded), one per part.
pub fn expand(vision: ?api.Vision, cx: *Cx, messages: Value, ids: []const u32, history_len: usize) errors.Refused!?Expanded {
    const v = vision orelse return null;
    const ps = try parts(cx, messages);
    var marks: usize = 0;
    for (ids) |t| marks += @intFromBool(t == v.image_token);
    if (marks != ps.len) return cx.fail(.server, "the chat template wrote {d} image placeholders for {d} image parts", .{ marks, ps.len });
    if (ps.len == 0) return null;
    if (ps.len > v.max_images) return cx.refuse(try std.fmt.allocPrint(cx.a, "a request may hold at most {d} images (--vision-max-images); it has {d}", .{ v.max_images, ps.len }));
    const budget = v.image_tokens / @as(u32, @intCast(ps.len));
    if (budget < 1) return cx.refuse("the image count exceeds the visual-token budget");
    const prepared = try cx.a.alloc(api.PreparedImage, ps.len);
    var bytes_total: usize = 0;
    var extra: usize = 0;
    for (ps, prepared, 0..) |p, *out, i| {
        const bytes = try dataBytes(cx, p.url);
        bytes_total += bytes.len;
        if (bytes_total > max_request_bytes) return cx.refuse("the request's images exceed 20 MB encoded");
        switch (sniff(bytes)) { // the Python frontend's formats (images.py: JPEG, PNG, WebP), WebP not yet decoded here
            .png, .jpeg => {},
            .webp => return cx.refuse(try std.fmt.allocPrint(cx.a, "image {d} is WebP, which this server does not decode yet; send PNG or JPEG", .{i + 1})),
            .gif, .other => return cx.refuse(try std.fmt.allocPrint(cx.a, "image {d} is not PNG or JPEG; send one of those", .{i + 1})),
        }
        const cap = if (p.low) @min(budget, low_detail_tokens) else budget;
        out.* = v.prepare(v.ctx, cx.a, bytes, cap) catch |e| return refusal(cx, i, e);
        if (out.tokens == 0 or out.tokens > cap) return cx.refuse("the image processor exceeded the per-image visual-token budget");
        extra += out.tokens - 1;
    }
    const n = ids.len + extra;
    const out_ids = try cx.a.alloc(u32, n);
    const key = try cx.a.alloc(u32, n);
    const imgs = try cx.a.alloc(api.Image, ps.len);
    var hist = history_len;
    var at: usize = 0;
    var k: usize = 0;
    for (ids, 0..) |t, src| {
        if (t != v.image_token) {
            out_ids[at] = t;
            key[at] = t;
            at += 1;
            continue;
        }
        const img = prepared[k];
        imgs[k] = .{ .pixels = img.pixels, .gh = img.gh, .gw = img.gw, .at = @intCast(at), .tokens = img.tokens };
        if (src < history_len) hist += img.tokens - 1;
        for (0..img.tokens) |j| {
            out_ids[at + j] = t;
            // the cache's ids for this image's rows: past the vocabulary, by content and row
            key[at + j] = 0x8000_0000 | @as(u32, @truncate(std.hash.Wyhash.hash(img.hash, std.mem.asBytes(&j)) & 0x7fff_ffff));
        }
        at += img.tokens;
        k += 1;
    }
    return .{ .ids = out_ids, .history_len = hist, .images = imgs, .cache_key = key };
}

test "data URLs decode; remote URLs are refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    try std.testing.expectEqualStrings("hi!", try dataBytes(&cx, "data:image/png;base64,aGkh"));
    try std.testing.expectEqualStrings("a b", try dataBytes(&cx, "data:text/plain,a%20b"));
    try std.testing.expectError(error.Refused, dataBytes(&cx, "https://example.com/x.png"));
    try std.testing.expectError(error.Refused, dataBytes(&cx, "data:image/png;base64,***"));
}

fn fakePrepare(ctx: *anyopaque, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    _ = ctx;
    _ = max_tokens;
    const tokens: u32 = @intCast(bytes.len - 3); // the test's images: a JPEG's first bytes, then one token a byte
    return .{ .pixels = try a.alloc(f32, 0), .gh = 2, .gw = 2 * tokens, .tokens = tokens, .hash = bytes[3] };
}

test "placeholders expand to each image's tokens, the history and cache key with them" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    var dummy: u8 = 0;
    const v: api.Vision = .{ .ctx = &dummy, .prepare = fakePrepare, .image_token = 9, .image_tokens = 64, .max_images = 4 };
    const text =
        \\[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,%FF%D8%FFab"}}, {"type": "text", "text": "x"}]},
        \\ {"role": "assistant", "content": "y"},
        \\ {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,%FF%D8%FFcde"}}]}]
    ;
    const msgs = switch (try json.parseText(a, text)) {
        .ok => |m| m,
        .err => return error.TestUnexpectedResult,
    };
    const ids = [_]u32{ 1, 9, 2, 3, 9, 4 };
    const x = (try expand(v, &cx, msgs, &ids, 4)).?;
    try std.testing.expectEqualSlices(u32, &.{ 1, 9, 9, 2, 3, 9, 9, 9, 4 }, x.ids);
    try std.testing.expectEqual(@as(usize, 5), x.history_len); // the first image is in the history, the second not
    try std.testing.expectEqual(@as(u32, 1), x.images[0].at);
    try std.testing.expectEqual(@as(u32, 5), x.images[1].at);
    try std.testing.expect(x.cache_key[1] >= 0x8000_0000 and x.cache_key[1] != x.cache_key[2]);
    try std.testing.expectEqual(@as(u32, 2), x.cache_key[3]);
    // a placeholder the images do not account for is refused
    try std.testing.expectError(error.Refused, expand(v, &cx, msgs, &[_]u32{ 9, 9, 9 }, 0));
    _ = messages_mod;
}

/// A test tokenizer: "<|image|>" is token 9 unless read plainly, every other byte its own id plus 100.
const MarkText = struct {
    fn text() model_text.Text {
        return .{ .ctx = undefined, .vtable = &.{ .encode = encode, .decode = undefined, .token_id = undefined, .token_string = tokenString, .vocab_size = undefined, .eos_ids = undefined, .render = undefined, .template_source = undefined, .encode_plain = plain } };
    }
    fn encode(_: *anyopaque, a: std.mem.Allocator, s: []const u8, _: bool) model_text.Error![]u32 {
        var out: std.ArrayList(u32) = .empty;
        var i: usize = 0;
        while (i < s.len) {
            if (std.mem.startsWith(u8, s[i..], "<|image|>")) {
                try out.append(a, 9);
                i += 9;
            } else {
                try out.append(a, @as(u32, s[i]) + 100);
                i += 1;
            }
        }
        return out.items;
    }
    fn plain(_: *anyopaque, a: std.mem.Allocator, s: []const u8) model_text.Error![]u32 {
        const out = try a.alloc(u32, s.len);
        for (out, s) |*o, c| o.* = @as(u32, c) + 100;
        return out;
    }
    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return a.dupe(u8, if (id == 9) "<|image|>" else "");
    }
};

test "a marker the text spells stays text: only the template's markers are image tokens (#511)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    const v: api.Vision = .{ .ctx = &dummy, .prepare = fakePrepare, .image_token = 9, .markers = &.{9}, .image_tokens = 64, .max_images = 4 };
    const markers = try markerText(MarkText.text(), a, v);
    const conversation =
        \\[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,ab"}}, {"type": "text", "text": "what is <|image|>?"}]},
        \\ {"role": "tool", "content": "source: x = '<|image|>'"}]
    ;
    const msgs = switch (try json.parseText(a, conversation)) {
        .ok => |m| m,
        .err => return error.TestUnexpectedResult,
    };
    const shielded = try shield(a, msgs, markers);
    // the "template": one marker per image part, then each message's text
    var rendered: std.ArrayList(u8) = .empty;
    try rendered.appendSlice(a, "<|image|>");
    try rendered.appendSlice(a, shielded.array[0].get("content").?.array[1].get("text").?.string);
    try rendered.appendSlice(a, shielded.array[1].get("content").?.string);
    const ids = try encodeShielded(MarkText.text(), a, rendered.items, markers);
    var marks: usize = 0;
    for (ids) |t| marks += @intFromBool(t == 9);
    try std.testing.expectEqual(@as(usize, 1), marks);
    // the quoted markers read as their characters, and text without one is exactly encode's
    try std.testing.expectEqualSlices(u32, try MarkText.encode(undefined, a, "<|image|>what is ", false), ids[0..9]);
    try std.testing.expectEqual(@as(u32, '<' + 100), ids[9]);
    const plain_text = "no markers here";
    try std.testing.expectEqualSlices(u32, try MarkText.encode(undefined, a, plain_text, false), try encodeShielded(MarkText.text(), a, plain_text, markers));
}

test "formats from their first bytes: WebP is refused by name" {
    try std.testing.expectEqual(Format.png, sniff("\x89PNG\r\n\x1a\nrest"));
    try std.testing.expectEqual(Format.jpeg, sniff("\xff\xd8\xff\xe0"));
    try std.testing.expectEqual(Format.gif, sniff("GIF89a..."));
    try std.testing.expectEqual(Format.webp, sniff("RIFF\x10\x00\x00\x00WEBPVP8 "));
    try std.testing.expectEqual(Format.other, sniff("\x00\x00\x00\x18ftypheic"));
}
