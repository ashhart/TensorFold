//! Lessons accept text, not chat message parts or image buffers that their token-only examples cannot carry.
const std = @import("std");
const json = @import("json");

pub const Refused = error{ ImagesNotSupported, TextRequired };

pub fn message(e: Refused) []const u8 {
    return switch (e) {
        error.ImagesNotSupported => "Living Weights lessons take text only: images and message parts cannot be learned",
        error.TextRequired => "text is required as a string",
    };
}

pub fn read(body: json.Value) Refused![]const u8 {
    for ([_][]const u8{ "images", "image", "image_url", "input_image", "messages", "content" }) |key| {
        if (body.field(key)) |v| if (v != .null and (v.items() == null or v.items().?.len > 0))
            return error.ImagesNotSupported;
    }
    const text = body.field("text") orelse return error.TextRequired;
    if (text.items() != null) return error.ImagesNotSupported;
    return text.str() orelse error.TextRequired;
}

test "slide input: media-bearing lessons refuse before any model or learner call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{
        "{\"text\":\"a fact\",\"images\":[{}]}",
        "{\"text\":\"a fact\",\"image_url\":\"data:image/png;base64,AA==\"}",
        "{\"text\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"x\"}}]}",
        "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\"}]}]}",
    }) |raw| {
        const body = (try json.parse(a, raw)).ok;
        try std.testing.expectError(error.ImagesNotSupported, read(body));
    }
    const text = (try json.parse(a, "{\"text\":\"an image of a bridge\",\"images\":[],\"source\":\"chat\"}")).ok;
    try std.testing.expectEqualStrings("an image of a bridge", try read(text));
    try std.testing.expectError(error.TextRequired, read((try json.parse(a, "{}")).ok));
}
