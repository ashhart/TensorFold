const std = @import("std");
const st = @import("safetensors.zig");

test "security: distinct safetensors cannot alias the same mutable bytes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const invalid = [_][]const u8{
        "{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}," ++
            "\"b\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}",
        "{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[2,6]}," ++
            "\"b\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}",
    };
    for (invalid) |json| try std.testing.expectError(error.OverlappingTensors, st.parseHeader(arena.allocator(), json, 6));
}

test "security: empty tensors and unsorted adjacent safetensors remain readable" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json = "{\"b\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[2,4]}," ++
        "\"empty\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[2,2]}," ++
        "\"a\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}";
    const parsed = try st.parseHeader(arena.allocator(), json, 4);
    try std.testing.expectEqual(@as(usize, 3), parsed.count());
}

test "security: header lengths are bounded before allocation or addition" {
    try std.testing.expectEqual(@as(usize, 2), try st.headerSize(2, 10));
    try std.testing.expectEqual(@as(usize, st.max_header_bytes), try st.headerSize(st.max_header_bytes, st.max_header_bytes + 8));
    for ([_][2]u64{
        .{ 0, 0 },                                             .{ 0, 7 }, .{ 3, 10 }, .{ std.math.maxInt(u64), std.math.maxInt(u64) },
        .{ st.max_header_bytes + 1, st.max_header_bytes + 9 },
    }) |sizes| try std.testing.expectError(error.BadSafetensors, st.headerSize(sizes[0], sizes[1]));
}

test "multiple text namespaces share validation while unrelated vision layouts are skipped" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json = "{\"vision.x\":{\"shape\":[1,1,1,1,1,1]}," ++
        "\"lm_head.weight\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}," ++
        "\"language_model.x\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[4,6]}}";
    const entries = try st.parseHeaderPrefixes(arena.allocator(), json, 6, &.{ "language_model.", "lm_head." });
    try std.testing.expectEqual(@as(usize, 2), entries.count());
    try std.testing.expect(entries.contains("language_model.x") and entries.contains("lm_head.weight"));
}
