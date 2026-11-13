//! A prompt's progress on a chat or completions stream (``return_progress``): a chunk with no choices.
const std = @import("std");
const json = @import("json");
const api = @import("engine_api");
const Allocator = std.mem.Allocator;

/// `chunk` with an empty ``choices`` and ``tensorfold.progress``, as the usage chunk has its ``usage``.
pub fn put(a: Allocator, chunk: *json.Object, p: api.Progress) Allocator.Error!void {
    const counts = try json.newObject(a);
    try counts.put(a, "done", try json.intValue(a, p.done));
    try counts.put(a, "total", try json.intValue(a, p.total));
    try counts.put(a, "cached", try json.intValue(a, p.cached));
    const block = try json.newObject(a);
    try block.put(a, "progress", .{ .object = counts });
    try chunk.put(a, "choices", .{ .array = &.{} });
    try chunk.put(a, "tensorfold", .{ .object = block });
}

test "prompt progress: a chunk's choices give way to an empty list and its counts go under tensorfold.progress" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const chunk = (try json.parse(a, "{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0}]}")).ok;
    try put(a, chunk.object, .{ .done = 512, .total = 2048, .cached = 256 });
    const want = "{\"id\": \"c\", \"object\": \"chat.completion.chunk\", \"choices\": [], \"tensorfold\": {\"progress\": {\"done\": 512, \"total\": 2048, \"cached\": 256}}}";
    try std.testing.expectEqualStrings(want, try json.stringify(a, chunk, .{}));
}
