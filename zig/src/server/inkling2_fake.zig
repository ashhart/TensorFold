//! A fake Inkling 2 tokenizer for the reply parser's host tests: listed pieces, their ids and added flags.
const std = @import("std");
const json = @import("json");
const model_text = @import("model_text.zig");
const reply = @import("inkling2_reply.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

pub const Piece = struct {
    id: u32,
    text: []const u8,
    added: bool = false,
};

pub fn plain(id: u32, text: []const u8) Piece {
    return .{ .id = id, .text = text };
}

pub const message_model = Piece{ .id = 200001, .text = "<|message_model|>", .added = true };
pub const content_text = Piece{ .id = 200004, .text = "<|content_text|>", .added = true };
pub const end_sampling = Piece{ .id = 200006, .text = "<|content_model_end_sampling|>", .added = true };
pub const content_thinking = Piece{ .id = 200008, .text = "<|content_thinking|>", .added = true };
pub const end_message = Piece{ .id = 200010, .text = "<|end_message|>", .added = true };
pub const content_tool_error = Piece{ .id = 200022, .text = "<|content_tool_error|>", .added = true };
pub const content_tool_json = Piece{ .id = 200049, .text = "<|content_invoke_tool_json|>", .added = true };
pub const content_tool_text = Piece{ .id = 200057, .text = "<|content_invoke_tool_text|>", .added = true };
// The tokenizer's other special tokens, none of them a reply control.
pub const message_user = Piece{ .id = 200000, .text = "<|message_user|>", .added = true };
pub const message_tool = Piece{ .id = 200003, .text = "<|message_tool|>", .added = true };
pub const message_system = Piece{ .id = 200002, .text = "<|message_system|>", .added = true };
pub const end_of_text = Piece{ .id = 199999, .text = "<|endoftext|>", .added = true };
pub const content_xml = Piece{ .id = 200024, .text = "<|content_xml|>", .added = true };
pub const unused = Piece{ .id = 200030, .text = "<|unused_200030|>", .added = true };

/// Token ids the fake has decoded, summed over every call.
pub var decoded_ids: usize = 0;

pub const Fake = struct {
    pieces: []const Piece,

    pub fn text(f: *Fake) model_text.Text {
        return .{
            .ctx = f,
            .vtable = &.{
                .encode = encode,
                .decode = decode,
                .token_id = tokenId,
                .token_string = tokenString,
                .vocab_size = vocabSize,
                .eos_ids = eosIds,
                .render = render,
                .template_source = templateSource,
                .added = added,
                .special = added, // every added Inkling 2 token is special in its tokenizer config
            },
        };
    }

    fn find(f: *const Fake, id: u32) ?Piece {
        for (f.pieces) |piece| if (piece.id == id) return piece;
        return null;
    }

    fn encode(ctx: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        for (f.pieces) |piece| if (std.mem.eql(u8, input, piece.text)) {
            const ids = try a.alloc(u32, 1);
            ids[0] = piece.id;
            return ids;
        };
        return error.Template;
    }

    fn decode(ctx: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        decoded_ids += ids.len;
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(a);
        for (ids) |id| {
            const piece = f.find(id) orelse std.debug.panic("unknown fake token {d}", .{id});
            try out.appendSlice(a, piece.text);
        }
        return out.toOwnedSlice(a);
    }

    fn tokenId(ctx: *anyopaque, spelling: []const u8) ?u32 {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        for (f.pieces) |piece| if (piece.added and std.mem.eql(u8, spelling, piece.text)) return piece.id;
        return null;
    }

    fn tokenString(ctx: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        const piece = f.find(id) orelse return a.dupe(u8, "");
        return a.dupe(u8, piece.text);
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 200058;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{200006};
    }

    fn added(ctx: *anyopaque, id: u32) bool {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        const piece = f.find(id) orelse return false;
        return piece.added;
    }

    fn render(_: *anyopaque, _: Allocator, _: Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return error.Template;
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

pub fn parse(a: Allocator, pieces: []const Piece, tools: []const Value) !reply.Result {
    var fake: Fake = .{ .pieces = pieces };
    const ids = try a.alloc(u32, pieces.len);
    for (pieces, ids) |piece, *id| id.* = piece.id;
    return reply.parse(a, fake.text(), ids, tools);
}

pub fn offeredTools(a: Allocator, names: []const []const u8) ![]Value {
    const tools = try a.alloc(Value, names.len);
    for (names, tools) |name, *tool| {
        const function = try json.newObject(a);
        try function.put(a, "name", .{ .string = name });
        const object = try json.newObject(a);
        try object.put(a, "type", .{ .string = "function" });
        try object.put(a, "function", .{ .object = function });
        tool.* = .{ .object = object };
    }
    return tools;
}

pub fn expectCall(calls: ?[]Value, index: usize, name: []const u8, arguments: []const u8) !void {
    try std.testing.expect(calls != null);
    const all = calls.?;
    try std.testing.expect(all.len > index);
    const call = all[index];
    try std.testing.expect(call.get("id") == null); // the route gives each call the request's id
    try std.testing.expectEqualStrings("function", call.get("type").?.string);
    const function = call.get("function").?;
    try std.testing.expectEqualStrings(name, function.get("name").?.string);
    try std.testing.expectEqualStrings(arguments, function.get("arguments").?.string);
}
