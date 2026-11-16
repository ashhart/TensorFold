//! Safetensors headers: parsing, sorting and what is refused.
const std = @import("std");
const header = @import("header.zig");

const t = std.testing;

const sample =
    \\{"__metadata__":{"format":"pt"},"b.x":{"dtype":"F8_E4M3","shape":[2,3],"data_offsets":[0,6]},
    \\"a.y":{"dtype":"BF16","shape":[4],"data_offsets":[6,14]},"c":{"dtype":"U8","shape":[],"data_offsets":[14,15]}}
;

test "a header's tensors come back sorted with dtype and shape, the metadata block skipped" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const list = try header.parse(arena.allocator(), sample);
    try t.expectEqual(@as(usize, 3), list.len);
    try t.expectEqualStrings("a.y", list[0].name);
    try t.expectEqual(header.Dtype.bf16, list[0].dtype);
    try t.expectEqualSlices(u64, &.{4}, list[0].shape());
    try t.expectEqualSlices(u64, &.{ 2, 3 }, list[1].shape());
    try t.expectEqual(header.Dtype.f8_e4m3, list[1].dtype);
    try t.expectEqual(@as(usize, 0), list[2].shape().len);
}

test "an unknown dtype, a rank past four, a broken length and bad JSON are refused" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try t.expectError(error.UnknownDtype, header.parse(a, "{\"w\":{\"dtype\":\"F16\",\"shape\":[1]}}"));
    try t.expectError(error.BadHeader, header.parse(a, "{\"w\":{\"dtype\":\"U8\",\"shape\":[1,1,1,1,1]}}"));
    try t.expectError(error.BadHeader, header.parse(a, "[1]"));
    try t.expectError(error.BadHeader, header.parse(a, "{\"w\":{\"dtype\":\"U8\",\"shape\":[-1]}}"));
    var file: [12]u8 = @splat(0);
    std.mem.writeInt(u64, file[0..8], 99, .little);
    try t.expectError(error.BadHeader, header.jsonOf(&file));
    std.mem.writeInt(u64, file[0..8], 4, .little);
    try t.expectEqualStrings(&.{ 0, 0, 0, 0 }, try header.jsonOf(&file));
}

test "every dtype's tag reads back as that dtype, and any other tag is not one" {
    for (std.enums.values(header.Dtype)) |d| try t.expectEqual(d, header.Dtype.of(d.tag()).?);
    try t.expect(header.Dtype.of("bf16") == null);
    try t.expect(header.Dtype.of("") == null);
}
