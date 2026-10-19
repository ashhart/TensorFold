const std = @import("std");
const t = std.testing;
const Config = @import("config.zig").Config;
const Pack = @import("pack.zig").Pack;
test "shared config requires checkpoint fields rather than guessing" {
    try t.expectError(error.MissingField, Config.parse(t.allocator, "{}"));
}
test "shared config rejects zero divisor before shape arithmetic" {
    var c: Config = .{};
    c.o_groups = 0;
    try t.expectError(error.BadConfig, c.validate());
}
test "shared config rejects huge numeric integers without a float cast trap" {
    try t.expectError(error.BadConfig, Config.parse(t.allocator, "{\"hidden_size\":1e100}"));
}
test "failed shard admission publishes no partial names or file owner" {
    var p = Pack.init(t.allocator);
    defer p.deinit();
    const good = "{\"a\":{\"dtype\":\"I32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    try p.addHeader("a.safetensors", good, 4);
    const bad = "{\"b\":{\"dtype\":\"I32\",\"shape\":[1],\"data_offsets\":[0,4]},\"a\":{\"dtype\":\"I32\",\"shape\":[1],\"data_offsets\":[4,8]}}";
    try t.expectError(error.DuplicateTensor, p.addHeader("bad.safetensors", bad, 8));
    try t.expectEqual(@as(usize, 1), p.files.items.len);
    try t.expect(!p.has("b"));
}
test "EXL3 shape and rank admission rejects narrowing and zero world" {
    const x = @import("core").exl3_layout;
    try t.expectError(error.BadTrellis, x.Group.fromShape(&.{ std.math.maxInt(u32), 8, 16 }));
    const g = try x.Group.fromShape(&.{ 8, 8, 16 });
    try t.expectError(error.SplitNotOnBlocks, g.part(.row, 0, 0));
}
test "markers are one scalar int32 codebook selector" {
    var p = Pack.init(t.allocator);
    defer p.deinit();
    try p.addHeader("bad", "{\"x.mul1\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[0,4]}}", 4);
    var buf: [128]u8 = undefined;
    try t.expectError(error.BadMarker, p.marker("x", &buf));
    var both = Pack.init(t.allocator);
    defer both.deinit();
    try both.addHeader("both", "{\"x.mul1\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[0,4]},\"x.mcg\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[4,8]}}", 8);
    try t.expectError(error.BadMarker, both.marker("x", &buf));
}
test "span byte multiplication rejects overflow before IO" {
    var p = Pack.init(t.allocator);
    defer p.deinit();
    const info: @import("pack.zig").Info = .{ .file = 0, .dtype = .i32, .rank = 0, .shape = @splat(1), .start = 0, .nbytes = 4 };
    try t.expectError(error.BadSpan, p.read(t.io, info, .{ .offset = 0, .len = std.math.maxInt(u64), .count = 2 }, &.{}));
}
test "index names are safe basenames and an invalid entry publishes no map" {
    var p = Pack.init(t.allocator);
    defer p.deinit();
    try t.expectError(error.BadIndex, p.useIndex("{\"weight_map\":{\"a\":\"ok.safetensors\",\"b\":\"../outside.safetensors\"}}"));
    try t.expectEqual(@as(u32, 0), p.owner.count());
}
