//! Embedding input forms flatten into bounded sequences without inventing a model's tokenizer or task prefixes.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const Text = @import("model_text.zig").Text;
const Cx = errors.Cx;
const Value = json.Value;
pub const Format = enum { float, base64 };
pub const Parsed = struct { tokens: []const u32, ends: []const u32, full_dimensions: u32, dimensions: u32, format: Format };

pub fn parse(cx: *Cx, text: Text, cap: api.EmbeddingInfo, context: u32, body: Value) errors.Refused!Parsed {
    if (body != .object) return cx.refuse("request body must be an object");
    for ([_][]const u8{ "images", "image_url", "audio", "input_audio", "video" }) |key|
        if (body.field(key) != null) return cx.refuse("embedding inputs support text only");
    var format: Format = .float;
    if (body.field("encoding_format")) |value| {
        if (value != .string) return cx.refuse("encoding_format must be float or base64");
        format = std.meta.stringToEnum(Format, value.string) orelse return cx.refuse("encoding_format must be float or base64");
    }
    var requested: ?u32 = null;
    if (body.field("dimensions")) |value| {
        if (value != .int) return cx.refuse("dimensions must be a positive integer");
        requested = std.fmt.parseInt(u32, value.int, 10) catch return cx.refuse("dimensions must be a positive integer");
        if (requested.? == 0) return cx.refuse("dimensions must be a positive integer");
    }
    const dimensions = cap.dimension(requested) catch |err| return cx.refuse(if (err == error.MrlNotDeclared)
        "this model does not declare MRL dimensions"
    else
        "dimensions is not supported by this model");
    const input = body.get("input") orelse return cx.refuse("input is required");
    var tokens: std.ArrayList(u32) = .empty;
    errdefer tokens.deinit(cx.a);
    var ends: std.ArrayList(u32) = .empty;
    errdefer ends.deinit(cx.a);
    if (input == .string) {
        try sequence(cx, text, cap, context, input, &tokens, &ends);
    } else if (input == .array and input.array.len > 0) {
        const first = input.array[0];
        if (first == .int) {
            try sequence(cx, text, cap, context, input, &tokens, &ends);
        } else if (first == .string or first == .array) {
            if (input.array.len > cap.max_inputs) return cx.refuse("too many embedding inputs");
            for (input.array, 0..) |item, i| {
                if (@as(std.meta.Tag(Value), item) != @as(std.meta.Tag(Value), first)) return cx.refuse("embedding input forms cannot be mixed");
                if (item == .string and blank(item.string)) return cx.fail(.request, "embedding input at index {d} must not be empty or whitespace only", .{i});
                try sequence(cx, text, cap, context, item, &tokens, &ends);
            }
        } else return cx.refuse("input must contain strings or token IDs");
    } else return cx.refuse("input must be nonempty text, token IDs, or a batch of either");
    const ids = try tokens.toOwnedSlice(cx.a);
    errdefer cx.a.free(ids);
    return .{ .tokens = ids, .ends = try ends.toOwnedSlice(cx.a), .full_dimensions = cap.dimensions, .dimensions = dimensions, .format = format };
}

fn sequence(cx: *Cx, text: Text, cap: api.EmbeddingInfo, context: u32, input: Value, tokens: *std.ArrayList(u32), ends: *std.ArrayList(u32)) errors.Refused!void {
    if (ends.items.len >= cap.max_inputs) return cx.refuse("too many embedding inputs");
    const at = tokens.items.len;
    if (input == .string) {
        if (blank(input.string)) return cx.refuse("embedding text must not be empty or whitespace only");
        const ids = text.encode(cx.a, input.string, true) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.Template => cx.refuse("the tokenizer cannot encode this embedding input"),
        };
        defer cx.a.free(ids);
        try fits(cx, context, ids.len);
        for (ids) |id| try token(cx, cap, id, tokens);
    } else if (input == .array) {
        try fits(cx, context, input.array.len);
        for (input.array) |item| {
            if (item != .int) return cx.refuse("token IDs must be integers");
            const id = std.fmt.parseInt(u32, item.int, 10) catch return cx.refuse("token ID is outside the vocabulary");
            try token(cx, cap, id, tokens);
        }
    } else return cx.refuse("each embedding input must be text or token IDs");
    const count = tokens.items.len - at;
    if (count == 0) return cx.refuse("embedding input must contain tokens");
    try ends.append(cx.a, @intCast(tokens.items.len));
}

/// One input longer than the context is refused as such, before the batch's own limit.
fn fits(cx: *Cx, context: u32, count: usize) errors.Refused!void {
    if (context > 0 and count > context) return cx.fail(.context_length, "the input ({d} tokens) is longer than the model's context length ({d} tokens)", .{ count, context });
}

/// Empty or whitespace only, by Python's str.isspace (as SGLang strips): nothing to embed; invalid UTF-8 is not blank.
fn blank(s: []const u8) bool {
    const view = std.unicode.Utf8View.init(s) catch return false;
    var it = view.iterator();
    while (it.nextCodepoint()) |c| if (!space(c)) return false;
    return true;
}

fn space(c: u21) bool {
    return switch (c) {
        0x09...0x0d, 0x1c...0x20, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

fn token(cx: *Cx, cap: api.EmbeddingInfo, id: u32, tokens: *std.ArrayList(u32)) errors.Refused!void {
    if (id >= cap.vocabulary) return cx.refuse("token ID is outside the vocabulary");
    // the input's own length was checked before: here only the whole batch can pass the admitted tokens
    if (tokens.items.len >= cap.max_batch_tokens) return cx.fail(.request, "the embedding batch exceeds the admitted {d} tokens", .{cap.max_batch_tokens});
    try tokens.append(cx.a, id);
}
