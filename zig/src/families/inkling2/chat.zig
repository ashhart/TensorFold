//! Inkling 2's chat wiring: the block markers' ids from tokenizer_config.json, the stop ids and the template's checks.
const std = @import("std");
const cfg = @import("shape.zig");
/// The release's block markers (special tokens in tokenizer_config.json) the server's reply split reads.
pub const markers = struct {
    pub const model = "<|message_model|>";
    pub const text = "<|content_text|>";
    pub const thinking = "<|content_thinking|>";
    pub const end = "<|end_message|>";
    pub const end_sampling = "<|content_model_end_sampling|>";
    pub const tool_json = "<|content_invoke_tool_json|>";
    pub const tool_text = "<|content_invoke_tool_text|>";
    pub const tool_error = "<|content_tool_error|>";
};

/// The special tokens the template writes and the reply split reads, by id.
pub const Tokens = struct {
    endoftext: u32 = 0,
    user: u32 = 0,
    model: u32 = 0,
    system: u32 = 0,
    tool: u32 = 0,
    text: u32 = 0,
    image_block: u32 = 0,
    end_sampling: u32 = 0,
    thinking: u32 = 0,
    end: u32 = 0,
    tool_error: u32 = 0,
    xml: u32 = 0,
    tool_json: u32 = 0,
    tool_text: u32 = 0,
    /// The image placeholder the processor expands into the tower's rows.
    image: u32 = 0,
};

const names = [_]struct { field: []const u8, text: []const u8 }{
    .{ .field = "endoftext", .text = "<|endoftext|>" },
    .{ .field = "user", .text = "<|message_user|>" },
    .{ .field = "model", .text = markers.model },
    .{ .field = "system", .text = "<|message_system|>" },
    .{ .field = "tool", .text = "<|message_tool|>" },
    .{ .field = "text", .text = markers.text },
    .{ .field = "image_block", .text = "<|content_image|>" },
    .{ .field = "end_sampling", .text = markers.end_sampling },
    .{ .field = "thinking", .text = markers.thinking },
    .{ .field = "end", .text = markers.end },
    .{ .field = "tool_error", .text = markers.tool_error },
    .{ .field = "xml", .text = "<|content_xml|>" },
    .{ .field = "tool_json", .text = markers.tool_json },
    .{ .field = "tool_text", .text = markers.tool_text },
    .{ .field = "image", .text = "<|unused_200054|>" },
};

pub const Error = error{ BadInkling2Tokenizer, Inkling2TokenMissing, Inkling2StopMismatch, Inkling2TemplateMismatch };

pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Tokens {
    const path = try std.fs.path.join(gpa, &.{ dir, "tokenizer_config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(bytes);
    return parse(gpa, bytes);
}

/// Every marker must be a special added token; its id is the decoder key that spells it.
pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Tokens {
    var doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.BadInkling2Tokenizer;
    const dec = doc.value.object.get("added_tokens_decoder") orelse return error.BadInkling2Tokenizer;
    if (dec != .object) return error.BadInkling2Tokenizer;
    var out: Tokens = .{};
    inline for (names) |n| {
        var found = false;
        var it = dec.object.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* != .object) continue;
            const content = e.value_ptr.object.get("content") orelse continue;
            if (content != .string or !std.mem.eql(u8, content.string, n.text)) continue;
            const special = e.value_ptr.object.get("special") orelse return error.BadInkling2Tokenizer;
            if (special != .bool or !special.bool) return error.BadInkling2Tokenizer;
            @field(out, n.field) = std.fmt.parseInt(u32, e.key_ptr.*, 10) catch return error.BadInkling2Tokenizer;
            found = true;
        }
        if (!found) return error.Inkling2TokenMissing;
    }
    return out;
}

/// The ids a reply stops at: config.json's eos, which must be the template's end of a model turn.
pub fn stops(c: *const cfg.Config, tok: *const Tokens) Error![]const u32 {
    const s = c.stops();
    for (s) |id| if (id == tok.end_sampling) return s;
    return error.Inkling2StopMismatch;
}

/// The template must write the blocks the server splits and take the effort switch; any other format is refused.
pub fn checkTemplate(template: []const u8) Error!void {
    const needed = [_][]const u8{ markers.model, markers.thinking, markers.text, markers.end, markers.tool_json, markers.end_sampling, "reasoning_effort", "tool_declare" };
    for (needed) |n| if (std.mem.indexOf(u8, template, n) == null) return error.Inkling2TemplateMismatch;
}

pub fn readTemplate(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, "chat_template.jinja" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    try checkTemplate(bytes);
}

const t = std.testing;
const fixture = @embedFile("tokenizer_fixture.json");

test "inkling2 chat: the markers' ids, the image token and the stop id from the release's files" {
    const tok = try parse(t.allocator, fixture);
    try t.expectEqual(@as(u32, 200001), tok.model);
    try t.expectEqual(@as(u32, 200006), tok.end_sampling);
    try t.expectEqual(@as(u32, 200008), tok.thinking);
    try t.expectEqual(@as(u32, 200010), tok.end);
    try t.expectEqual(@as(u32, 200049), tok.tool_json);
    try t.expectEqual(@as(u32, 200054), tok.image);
    const c = try cfg.parse(t.allocator, @embedFile("fixtures/config.json"));
    try t.expectEqualSlices(u32, &.{200006}, try stops(&c, &tok));
    var other = c;
    other.eos[0] = tok.end;
    try t.expectError(error.Inkling2StopMismatch, stops(&other, &tok));
}

test "inkling2 chat: a missing or non-special marker is refused" {
    const gone = try std.mem.replaceOwned(u8, t.allocator, fixture, "<|content_thinking|>", "<|content_think|>");
    defer t.allocator.free(gone);
    try t.expectError(error.Inkling2TokenMissing, parse(t.allocator, gone));
    const plain = try std.mem.replaceOwned(u8, t.allocator, fixture, "\"special\": true", "\"special\": false");
    defer t.allocator.free(plain);
    try t.expectError(error.BadInkling2Tokenizer, parse(t.allocator, plain));
}

test "inkling2 chat: the template check takes the release's blocks and refuses another format" {
    try checkTemplate(@embedFile("fixtures/chat_template.jinja"));
    try t.expectError(error.Inkling2TemplateMismatch, checkTemplate("{{ '<think>' }} <tool_call> reasoning_effort"));
}
