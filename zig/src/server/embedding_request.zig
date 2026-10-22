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
            for (input.array) |item| {
                if (@as(std.meta.Tag(Value), item) != @as(std.meta.Tag(Value), first)) return cx.refuse("embedding input forms cannot be mixed");
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
        if (input.string.len == 0) return cx.refuse("embedding text must not be empty");
        const ids = text.encode(cx.a, input.string, true) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.Template => cx.refuse("the tokenizer cannot encode this embedding input"),
        };
        defer cx.a.free(ids);
        for (ids) |id| try token(cx, cap, id, tokens);
    } else if (input == .array) {
        for (input.array) |item| {
            if (item != .int) return cx.refuse("token IDs must be integers");
            const id = std.fmt.parseInt(u32, item.int, 10) catch return cx.refuse("token ID is outside the vocabulary");
            try token(cx, cap, id, tokens);
        }
    } else return cx.refuse("each embedding input must be text or token IDs");
    const count = tokens.items.len - at;
    if (count == 0) return cx.refuse("embedding input must contain tokens");
    if (context > 0 and count > context) return cx.fail(.context_length, "embedding input exceeds the {d}-token context", .{context});
    try ends.append(cx.a, @intCast(tokens.items.len));
}

fn token(cx: *Cx, cap: api.EmbeddingInfo, id: u32, tokens: *std.ArrayList(u32)) errors.Refused!void {
    if (id >= cap.vocabulary) return cx.refuse("token ID is outside the vocabulary");
    if (tokens.items.len >= cap.max_batch_tokens) return cx.refuse("embedding batch exceeds the admitted token limit");
    try tokens.append(cx.a, id);
}
