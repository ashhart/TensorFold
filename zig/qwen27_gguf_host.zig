//! Serialized GGUF family checks run the same adapter exported by the native family.
test {
    _ = @import("src/families/qwen27/gguf_test.zig");
}

test "Qwen27 GGUF: actual model resolver refuses GGUF before tokenizer, checkpoint or device access" {
    const std = @import("std");
    var p: []const u8 = "";
    for ([_][]const u8{ "missing.gguf", "missing.GGUF", "/not-a-model/qwen.gguf" }) |path| {
        try std.testing.expectEqual(null, try @import("server_hub").resolve(std.testing.allocator, std.testing.io, null, path, &p));
        try std.testing.expect(std.mem.startsWith(u8, p, "GGUF serving is coming"));
    }
}
