//! A reply's OpenAI usage object, and DeepSeek's own cache keys beside it for its native V4.1 chat.
const std = @import("std");
const json = @import("json");
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const Counts = struct { prompt: usize, cached: usize, completion: usize, reasoning: usize };

/// ``usage`` with OpenAI's details; `deepseek` adds DeepSeek's prompt_cache_hit_tokens and prompt_cache_miss_tokens.
pub fn of(a: Allocator, n: Counts, deepseek: bool) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "prompt_tokens", try json.intValue(a, n.prompt));
    try o.put(a, "completion_tokens", try json.intValue(a, n.completion));
    try o.put(a, "total_tokens", try json.intValue(a, n.prompt + n.completion));
    const cached = try json.newObject(a);
    try cached.put(a, "cached_tokens", try json.intValue(a, n.cached));
    try o.put(a, "prompt_tokens_details", .{ .object = cached });
    const thought = try json.newObject(a);
    try thought.put(a, "reasoning_tokens", try json.intValue(a, n.reasoning));
    try o.put(a, "completion_tokens_details", .{ .object = thought });
    if (deepseek) {
        try o.put(a, "prompt_cache_hit_tokens", try json.intValue(a, n.cached));
        try o.put(a, "prompt_cache_miss_tokens", try json.intValue(a, n.prompt -| n.cached));
    }
    return .{ .object = o };
}

test "DeepSeek-V4.1's chat usage adds its cache hit and miss counts; other usage keeps OpenAI's keys alone" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const n: Counts = .{ .prompt = 40, .cached = 32, .completion = 5, .reasoning = 2 };
    try std.testing.expectEqualStrings(
        "{\"prompt_tokens\": 40, \"completion_tokens\": 5, \"total_tokens\": 45, \"prompt_tokens_details\": {\"cached_tokens\": 32}, \"completion_tokens_details\": {\"reasoning_tokens\": 2}, \"prompt_cache_hit_tokens\": 32, \"prompt_cache_miss_tokens\": 8}",
        try json.stringify(a, try of(a, n, true), .{}),
    );
    try std.testing.expect((try of(a, n, false)).get("prompt_cache_hit_tokens") == null);
}
