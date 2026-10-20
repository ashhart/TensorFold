//! Embedding wire forms, limits and MRL encoding use independent fixtures rather than an admitted model.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const request = @import("embedding_request.zig");
const response = @import("embedding_response.zig");
const errors = @import("errors.zig");
const text = @import("model_text.zig");
const t = std.testing;
const cap: api.EmbeddingInfo = .{ .dimensions = 3, .vocabulary = 256, .max_batch_tokens = 8, .max_inputs = 4, .mrl_dimensions = &.{ 1, 2 } };

pub const ByteText = struct {
    pub fn model() text.Text {
        return .{ .ctx = undefined, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eos, .render = render, .template_source = source } };
    }
    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (ids, input) |*id, b| id.* = b;
        return ids;
    }
    fn decode(_: *anyopaque, a: std.mem.Allocator, _: []const u32) ![]u8 {
        return a.dupe(u8, "");
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: std.mem.Allocator, _: u32) ![]u8 {
        return a.dupe(u8, "");
    }
    fn vocab(_: *anyopaque) u32 {
        return 256;
    }
    fn eos(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn render(_: *anyopaque, _: std.mem.Allocator, _: json.Value, _: text.RenderOptions, _: *[]const u8) text.Error!text.Rendered {
        return error.Template;
    }
    fn source(_: *anyopaque) []const u8 {
        return "";
    }
};

fn parse(cx: *errors.Cx, body: []const u8, info: api.EmbeddingInfo) !request.Parsed {
    return request.parse(cx, ByteText.model(), info, 4, (try json.parse(cx.a, body)).ok);
}

test "embeddings protocol: strings and token batches retain exact sequence boundaries and usage" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    for ([_][]const u8{ "{\"input\":\"abc\"}", "{\"input\":[97,98,99]}" }) |body| {
        const got = try parse(&cx, body, cap);
        try t.expectEqualSlices(u32, &.{ 97, 98, 99 }, got.tokens);
        try t.expectEqualSlices(u32, &.{3}, got.ends);
        try t.expectEqual(@as(u32, 3), got.dimensions);
    }
    for ([_][]const u8{ "{\"input\":[\"a\",\"bc\"]}", "{\"input\":[[97],[98,99]]}" }) |body| {
        const got = try parse(&cx, body, cap);
        try t.expectEqualSlices(u32, &.{ 97, 98, 99 }, got.tokens);
        try t.expectEqualSlices(u32, &.{ 1, 3 }, got.ends);
    }
}

test "embeddings protocol: invalid forms limits media and undeclared MRL refuse" {
    const bad = [_][]const u8{
        "{}",                                        "{\"input\":null}",                              "{\"input\":\"\"}",                 "{\"input\":[]}",                        "{\"input\":[[]]}",
        "{\"input\":[true]}",                        "{\"input\":[1.0]}",                             "{\"input\":[-1]}",                 "{\"input\":[256]}",                     "{\"input\":[4294967296]}",
        "{\"input\":[\"a\",1]}",                     "{\"input\":[[1],\"a\"]}",                       "{\"input\":\"abcde\"}",            "{\"input\":[[1,2,3,4],[5,6,7,8],[9]]}", "{\"input\":[\"a\",\"b\",\"c\",\"d\",\"e\"]}",
        "{\"input\":[1],\"images\":[\"x\"]}",        "{\"input\":[1],\"dimensions\":0}",              "{\"input\":[1],\"dimensions\":4}", "{\"input\":[1],\"dimensions\":true}",   "{\"input\":[1],\"dimensions\":1.0}",
        "{\"input\":[1],\"dimensions\":4294967296}", "{\"input\":[1],\"encoding_format\":\"bytes\"}",
    };
    for (bad) |body| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        var cx: errors.Cx = .{ .a = arena.allocator() };
        try t.expectError(error.Refused, parse(&cx, body, cap));
    }
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    var plain = cap;
    plain.mrl_dimensions = &.{};
    try t.expectError(error.Refused, parse(&cx, "{\"input\":[1],\"dimensions\":3}", plain));
}

test "embeddings protocol: float and little-endian base64 preserve MRL rows and full-width strides" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    var got: [2][4]f32 = undefined;
    for ([_][]const u8{ "float", "base64" }, 0..) |format, kind| {
        const body = try std.fmt.allocPrint(cx.a, "{{\"input\":[[3,4],[5,12]],\"dimensions\":2,\"encoding_format\":\"{s}\"}}", .{format});
        const parsed = try parse(&cx, body, cap);
        var vectors = [_]f32{ 3, 4, 12, 5, 12, 0 };
        const value = try response.encode(&cx, "embedding-test", parsed, &vectors);
        try t.expectEqualStrings("4", value.get("usage").?.get("prompt_tokens").?.int);
        try t.expectEqualStrings("embedding-test", value.get("model").?.string);
        for (value.get("data").?.array, 0..) |row, i| {
            const embedding = row.get("embedding").?;
            if (kind == 0) {
                for (embedding.array, 0..) |number, j| got[kind][i * 2 + j] = @floatCast(number.float);
            } else {
                const decoder = std.base64.standard.Decoder;
                var bytes: [8]u8 = undefined;
                try t.expectEqual(@as(usize, 8), try decoder.calcSizeForSlice(embedding.string));
                try decoder.decode(&bytes, embedding.string);
                for (0..2) |j| got[kind][i * 2 + j] = @bitCast(std.mem.readInt(u32, bytes[j * 4 ..][0..4], .little));
            }
        }
    }
    try t.expectEqualSlices(f32, &.{ 0.6, 0.8, 5.0 / 13.0, 12.0 / 13.0 }, &got[0]);
    try t.expectEqualSlices(u8, std.mem.asBytes(&got[0]), std.mem.asBytes(&got[1]));
    const parsed = try parse(&cx, "{\"input\":[1],\"dimensions\":2}", cap);
    var zero = [_]f32{ 0, 0, 1 };
    try t.expectError(error.Refused, response.encode(&cx, "m", parsed, &zero));
    var nan = [_]f32{ 3, 4, std.math.nan(f32) };
    try t.expectError(error.Refused, response.encode(&cx, "m", parsed, &nan));
}
