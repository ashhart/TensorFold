const std = @import("std");
const owner = @import("cuda_prompt_wt.zig");
const Store = owner.Store;
const Tensor = @import("core").safetensors.Tensor;
const hostOnly = owner.hostOnly;

test "the n-gram tables, the draft head and the vision tower stay off the device" {
    try std.testing.expect(hostOnly("language_model.model.layers.2.ple.ple_embedding.ngram_embedding.shards.7.weight"));
    try std.testing.expect(hostOnly("language_model.mtp.layers.0.mlp.switch_mlp.up_proj.weight"));
    try std.testing.expect(hostOnly("vision_tower.blocks.0.attn.qkv.weight"));
    try std.testing.expect(!hostOnly("language_model.model.layers.2.ple.ple_embedding.ngram_heads_offsets"));
    try std.testing.expect(!hostOnly("language_model.model.layers.0.mlp.switch_mlp.up_proj.weight"));
    try std.testing.expect(!hostOnly("language_model.lm_head.weight"));
}

test "canonical checkpoint aliases preserve exact stored namespace and prefer a direct key" {
    const a = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, a,
        \\{"weight_map":{"model.language_model.layers.0.x.weight":"a","lm_head.weight":"b","mtp.x":"c","language_model.model.direct":"d","model.language_model.direct":"e"}}
    , .{});
    var store: Store = .{ .gpa = a, .io = std.testing.io, .dir = "", .parsed = parsed };
    defer store.deinit();
    var buf: [512]u8 = undefined;
    try std.testing.expectEqualStrings("model.language_model.layers.0.x.weight", try store.resolveKey("language_model.model.layers.0.x.weight", &buf));
    try std.testing.expectEqualStrings("lm_head.weight", try store.resolveKey("language_model.lm_head.weight", &buf));
    try std.testing.expectEqualStrings("mtp.x", try store.resolveKey("language_model.mtp.x", &buf));
    try std.testing.expectEqualStrings("language_model.model.direct", try store.resolveKey("language_model.model.direct", &buf));
    try std.testing.expectError(error.MissingTensor, store.resolveKey("language_model.model.missing", &buf));
    try std.testing.expect(hostOnly("model.visual.weight"));
    try std.testing.expect(hostOnly("mtp.layers.0.weight"));
}

test "released expert planes keep every shard of a load open without invalidating held peers, until loaded" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const header = "{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}";
    var data: [8 + header.len + 1]u8 = undefined;
    std.mem.writeInt(u64, data[0..8], header.len, .little);
    @memcpy(data[8..][0..header.len], header);
    data[data.len - 1] = 17;
    try tmp.dir.writeFile(io, .{ .sub_path = "a", .data = &data });
    data[data.len - 1] = 29;
    try tmp.dir.writeFile(io, .{ .sub_path = "b", .data = &data });
    try tmp.dir.writeFile(io, .{ .sub_path = "c", .data = &data });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"x\":\"a\"}}" });
    var store = try Store.open(a, io, dir);
    defer store.deinit();
    const first = try store.hold("x");
    const address = first.bytes.ptr;
    store.release();
    try std.testing.expectEqual(@as(usize, 0), store.held.items.len);
    try std.testing.expectEqual(@as(usize, 1), store.released.items.len);
    // Removing the path proves the next borrow uses the existing mapping and parsed header.
    try tmp.dir.deleteFile(io, "a");
    const again = try store.hold("x");
    try std.testing.expectEqual(address, again.bytes.ptr);
    try std.testing.expectEqual(@as(u8, 17), again.bytes[0]);
    const wm = &store.parsed.value.object.getPtr("weight_map").?.object;
    wm.getPtr("x").?.* = .{ .string = "b" };
    const other = try store.hold("x");
    try std.testing.expectEqual(@as(u8, 29), other.bytes[0]);
    try std.testing.expectEqual(@as(u8, 17), again.bytes[0]);
    store.release();
    try std.testing.expectEqual(@as(usize, 2), store.released.items.len);
    try tmp.dir.deleteFile(io, "b");
    wm.getPtr("x").?.* = .{ .string = "a" };
    try std.testing.expectEqual(@as(u8, 17), (try store.hold("x")).bytes[0]);
    wm.getPtr("x").?.* = .{ .string = "b" };
    try std.testing.expectEqual(@as(u8, 29), (try store.hold("x")).bytes[0]);
    store.release();
    store.release();
    try std.testing.expectEqual(@as(usize, 2), store.released.items.len);
    wm.getPtr("x").?.* = .{ .string = "c" };
    _ = try store.hold("x");
    store.release();
    // a third shard joins the other two: none is closed and reparsed while the load goes on
    try std.testing.expectEqual(@as(usize, 3), store.released.items.len);
    try tmp.dir.deleteFile(io, "c");
    try std.testing.expectEqual(@as(u8, 29), (try store.hold("x")).bytes[0]);
    store.loaded();
    try std.testing.expectEqual(@as(usize, 0), store.held.items.len + store.released.items.len);
}

test "tensor() switching across three shards opens and parses each once a load, sharing the store's released shards" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const names = [_][]const u8{ "a", "b", "c" };
    for (names, 0..) |name, i| {
        var header_buf: [80]u8 = undefined;
        const header = try std.fmt.bufPrint(&header_buf, "{{\"x{s}\":{{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}}}", .{name});
        var data: [8 + 80 + 1]u8 = undefined;
        std.mem.writeInt(u64, data[0..8], header.len, .little);
        @memcpy(data[8..][0..header.len], header);
        data[8 + header.len] = @intCast(10 + i);
        try tmp.dir.writeFile(io, .{ .sub_path = name, .data = data[0 .. 8 + header.len + 1] });
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"xa\":\"a\",\"xb\":\"b\",\"xc\":\"c\"}}" });
    var store = try Store.open(a, io, dir);
    defer store.deinit();
    // each shard's file is gone after its first open, so a second open or header parse fails the load
    for (names, 0..) |name, i| {
        var key = [_]u8{ 'x', name[0] };
        try std.testing.expectEqual(@as(u8, @intCast(10 + i)), (try store.tensor(&key)).bytes[0]);
        try tmp.dir.deleteFile(io, name);
    }
    for (0..4) |round| for (names, 0..) |name, i| {
        var key = [_]u8{ 'x', name[0] };
        try std.testing.expectEqual(@as(u8, @intCast(10 + i)), (try store.tensor(&key)).bytes[0]);
        if (round != 1) continue;
        const copy = try store.copyOf(&key);
        defer a.free(copy);
        try std.testing.expectEqual(@as(u8, @intCast(10 + i)), copy[0]);
    };
    // hold() takes a parked shard from the same set, and release() gives it back to tensor()
    try std.testing.expectEqual(@as(u8, 10), (try store.hold("xa")).bytes[0]);
    store.release();
    try std.testing.expectEqual(@as(u8, 10), (try store.tensor("xa")).bytes[0]);
    try std.testing.expectEqual(@as(u8, 11), (try store.tensor("xb")).bytes[0]);
    store.loaded();
    try std.testing.expect(store.file == null and store.held.items.len + store.released.items.len == 0);
}

test "centered scales add one in FP32 while direct scales preserve the original bits" {
    _ = @import("cuda_norms.zig");
    const raw = [_]f32{ -0.25, 0.0001, 2.75 };
    const t: Tensor = .{ .dtype = .f32, .rank = 1, .shape = .{ 3, 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&raw) };
    var store: Store = undefined;
    store.norm_offset = 1;
    const scales = try store.normScale(std.testing.allocator, t);
    defer std.testing.allocator.free(scales);
    for (raw, scales) |v, scale| try std.testing.expectEqual(v + @as(f32, 1), scale);
    store.norm_offset = 0;
    const direct = try store.normScale(std.testing.allocator, t);
    defer std.testing.allocator.free(direct);
    try std.testing.expectEqualSlices(f32, &raw, direct);
}
