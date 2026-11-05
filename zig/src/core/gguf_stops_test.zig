//! Serialized stop metadata checks control kinds, deduplication and bounded output storage.
const std = @import("std");
const g = @import("gguf.zig");
const stops = @import("gguf_stops.zig");
const F = @import("gguf_test_fixture.zig").Fixture;
const t = std.testing;
fn image(eos: u32, kind: i32, types: bool) F {
    var f = F.init(0, if (types) 3 else 2);
    f.array("tokenizer.ggml.tokens", .string, 3);
    for ([_][]const u8{ "word", "<|endoftext|>", "<|im_end|>" }) |word| f.string(word);
    if (types) {
        f.array("tokenizer.ggml.token_type", .int32, 3);
        for ([_]i32{ 1, kind, 3 }) |k| f.int(i32, k);
    }
    f.kv("tokenizer.ggml.eos_token_id", .{ .uint32 = eos });
    f.finish(32, 0);
    return f;
}
test "GGUF stops: declared EOS and additional control ids remain ordered and deduplicated" {
    var f = image(2, 3, true);
    var d = try g.Document.parse(t.allocator, f.slice(), .{});
    defer d.deinit();
    var out: [8]u32 = undefined;
    const n = try stops.read(&d, &.{ "<|endoftext|>", "<|im_end|>", "<|endoftext|>" }, &out);
    try t.expectEqualSlices(u32, &.{ 2, 1 }, out[0..n]);
    try t.expectError(error.StopSetCapacity, stops.read(&d, &.{"<|endoftext|>"}, out[0..1]));
}
test "GGUF stops: ordinary end-of-text spellings and missing kinds are explicit refusals" {
    for ([_]i32{ 1, 4, 5 }) |kind| {
        var f = image(2, kind, true);
        var d = try g.Document.parse(t.allocator, f.slice(), .{});
        defer d.deinit();
        var out: [8]u32 = undefined;
        try t.expectError(error.StopTokenIsNotControl, stops.read(&d, &.{"<|endoftext|>"}, &out));
    }
    var f = image(2, 3, false);
    var d = try g.Document.parse(t.allocator, f.slice(), .{});
    defer d.deinit();
    var out: [8]u32 = undefined;
    try t.expectError(error.MissingStopTokenTypes, stops.read(&d, &.{"<|endoftext|>"}, &out));
}
test "GGUF stops: a declared end-of-text EOS is not appended twice" {
    var f = image(1, 3, true);
    var d = try g.Document.parse(t.allocator, f.slice(), .{});
    defer d.deinit();
    var out: [8]u32 = undefined;
    const n = try stops.read(&d, &.{"<|endoftext|>"}, &out);
    try t.expectEqualSlices(u32, &.{1}, out[0..n]);
}
