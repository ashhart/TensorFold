//! A text prompt in ERNIE 5.5's chat encoding, with the spans that came from the request, and its ids and text.
const std = @import("std");
const tokenizer = @import("tokenizer");

pub const Effort = enum { low, medium, high };

/// One user turn, an optional system turn before it, and whether and how hard the model thinks.
pub const Request = struct {
    user: []const u8,
    system: ?[]const u8 = null,
    thinking: bool = true,
    /// The encoding writes it only when thinking; `high` is its default.
    effort: Effort = .high,
};

/// The rendered text and the byte spans of request text in it, which no added token may begin, end or cross.
pub const Rendered = struct { text: []u8, spans: []tokenizer.Span };

const mask = "<|ERNIE_MASK|>";
const start = "<|ernie:im_start|>";
const end = "<|ernie:im_end|>";
const think = "<ernie:think>";
const think_end = "</ernie:think>";

const Builder = struct {
    a: std.mem.Allocator,
    text: std.ArrayList(u8) = .empty,
    spans: std.ArrayList(tokenizer.Span) = .empty,

    fn protocol(b: *Builder, s: []const u8) !void {
        try b.text.appendSlice(b.a, s);
    }

    fn request(b: *Builder, s: []const u8) !void {
        if (s.len == 0) return;
        const at = b.text.items.len;
        try b.text.appendSlice(b.a, s);
        try b.spans.append(b.a, .{ at, at + s.len });
    }
};

/// The prompt for `req` up to the assistant's opening, which holds an open think tag or an empty think pair.
pub fn render(a: std.mem.Allocator, req: Request) !Rendered {
    var b: Builder = .{ .a = a };
    try b.protocol(mask ++ start ++ "system\n");
    if (req.system) |s| {
        try b.protocol("<ernie:system_setting>\n");
        try b.request(s);
        try b.protocol("\n</ernie:system_setting>\n");
    }
    try b.protocol("<ernie:global_setting>");
    if (req.thinking) {
        try b.protocol("\nreasoning_effort=");
        try b.request(@tagName(req.effort));
    }
    try b.protocol("\n</ernie:global_setting>" ++ end ++ "\n" ++ start ++ "user\n");
    try b.request(req.user);
    try b.protocol(end ++ "\n" ++ start ++ "assistant\n");
    try b.protocol(if (req.thinking) think else think ++ think_end);
    return .{ .text = try b.text.toOwnedSlice(a), .spans = try b.spans.toOwnedSlice(a) };
}

/// Token ids of a rendered prompt: added tokens only where the encoding wrote them, never from request text.
pub fn encode(a: std.mem.Allocator, tok: *const tokenizer.Tokenizer, r: Rendered) ![]u32 {
    return tok.encodeMasked(a, r.text, r.spans);
}

/// The text of `ids`, special tokens spelled out.
pub fn decode(a: std.mem.Allocator, tok: *const tokenizer.Tokenizer, ids: []const u32) ![]u8 {
    return tok.decode(a, ids, false);
}
