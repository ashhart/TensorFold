//! Completed vectors retain their full-width stride while optional MRL crops and encodings stay row independent.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const Parsed = @import("embedding_request.zig").Parsed;

pub fn encode(cx: *errors.Cx, model: []const u8, parsed: Parsed, output: []f32) errors.Refused!json.Value {
    const full: usize = parsed.full_dimensions;
    const wanted: usize = parsed.dimensions;
    const size = std.math.mul(usize, parsed.ends.len, full) catch return cx.refuse("embedding output size overflows");
    if (full == 0 or wanted == 0 or wanted > full or output.len != size)
        return cx.fail(.server, "invalid embedding output shape", .{});
    for (output) |value| if (!std.math.isFinite(value)) return cx.fail(.server, "embedding output is not finite", .{});
    const data = try cx.a.alloc(json.Value, parsed.ends.len);
    for (data, 0..) |*item, i| {
        const values = output[i * full ..][0..wanted];
        if (wanted < full) api.embedding_vector.normalize(values) catch return cx.fail(.server, "embedding truncation has no finite nonzero norm", .{});
        const encoded: json.Value = switch (parsed.format) {
            .float => blk: {
                const array = try cx.a.alloc(json.Value, wanted);
                for (array, values) |*slot, value| slot.* = .{ .float = value };
                break :blk .{ .array = array };
            },
            .base64 => blk: {
                const byte_count = std.math.mul(usize, wanted, 4) catch return cx.fail(.server, "embedding encoding size overflows", .{});
                const bytes = try cx.a.alloc(u8, byte_count);
                for (values, 0..) |value, col| std.mem.writeInt(u32, bytes[col * 4 ..][0..4], @bitCast(value), .little);
                const encoder = std.base64.standard.Encoder;
                const text = try cx.a.alloc(u8, encoder.calcSize(bytes.len));
                break :blk .{ .string = encoder.encode(text, bytes) };
            },
        };
        const object = try json.newObject(cx.a);
        try object.put(cx.a, "object", .{ .string = "embedding" });
        try object.put(cx.a, "index", try json.intValue(cx.a, i));
        try object.put(cx.a, "embedding", encoded);
        item.* = .{ .object = object };
    }
    const usage = try json.newObject(cx.a);
    try usage.put(cx.a, "prompt_tokens", try json.intValue(cx.a, parsed.tokens.len));
    try usage.put(cx.a, "total_tokens", try json.intValue(cx.a, parsed.tokens.len));
    const result = try json.newObject(cx.a);
    try result.put(cx.a, "object", .{ .string = "list" });
    try result.put(cx.a, "data", .{ .array = data });
    try result.put(cx.a, "model", .{ .string = model });
    try result.put(cx.a, "usage", .{ .object = usage });
    return .{ .object = result };
}
