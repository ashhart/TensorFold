//! Serialized GGUF family checks run the same adapter exported by the native family.
test {
    _ = @import("src/families/qwen27/gguf_test.zig");
}

test "Qwen27 GGUF: actual model resolver refuses missing GGUF before tokenizer, checkpoint or device access" {
    const std = @import("std");
    var p: []const u8 = "";
    for ([_][]const u8{ "missing.gguf", "missing.GGUF", "/not-a-model/qwen.gguf" }) |path| {
        try std.testing.expectEqual(null, try @import("server_hub").resolve(std.testing.allocator, std.testing.io, null, path, &p));
        try std.testing.expect(std.mem.startsWith(u8, p, "GGUF file is missing"));
    }
}
test "Qwen27 GGUF: actual resolver accepts a regular file and an unambiguous directory" {
    const std = @import("std");
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    try tmp.dir.writeFile(io, .{ .sub_path = "one.gguf", .data = "GGUF" });
    const file = try tmp.dir.realPathFileAlloc(io, "one.gguf", a);
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    var problem: []const u8 = "";
    try std.testing.expectEqualStrings(file, (try @import("server_hub").resolve(a, io, null, file, &problem)).?);
    try std.testing.expectEqualStrings(file, (try @import("server_hub").resolve(a, io, null, dir, &problem)).?);
}
