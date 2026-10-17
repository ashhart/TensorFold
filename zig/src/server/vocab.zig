//! The vocabulary size a checkpoint declares, for the token masks of structured output.
const std = @import("std");
const json = @import("json");

/// config.json's vocab_size: the text model's (text_config, or an omni checkpoint's llm_config) before the top level's.
pub fn size(cfg: json.Value) ?u32 {
    for ([_]?json.Value{ cfg.get("text_config"), cfg.get("llm_config"), cfg }) |part| if (part) |p| if (p.get("vocab_size")) |v| if (v.int64()) |n| {
        if (n > 0 and n <= std.math.maxInt(u32)) return @intCast(n);
    };
    return null;
}

fn parse(a: std.mem.Allocator, text: []const u8) !json.Value {
    return switch (try json.parse(a, text)) {
        .ok => |v| v,
        .err => error.TestUnexpectedResult,
    };
}

test "the vocabulary is the text model's: text_config, llm_config of an omni wrapper, else the top level" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqual(@as(?u32, 131072), size(try parse(a, "{\"model_type\": \"nemotron_h_omni\", \"llm_config\": {\"vocab_size\": 131072}}")));
    try std.testing.expectEqual(@as(?u32, 151936), size(try parse(a, "{\"text_config\": {\"vocab_size\": 151936}, \"vocab_size\": 1}")));
    try std.testing.expectEqual(@as(?u32, 32000), size(try parse(a, "{\"vocab_size\": 32000}")));
    try std.testing.expectEqual(@as(?u32, null), size(try parse(a, "{\"llm_config\": {\"hidden_size\": 4096}}")));
    try std.testing.expectEqual(@as(?u32, null), size(try parse(a, "{\"vocab_size\": 0}")));
}
