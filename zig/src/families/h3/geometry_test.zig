//! Serialized H3 case fixtures run through the reader used by Model.load, without a Metal device.
const std = @import("std");
const st = @import("safetensors");
const g = @import("geometry.zig");
const schedule = @import("schedule.zig");
const testing = std.testing;

const Fixture = struct {
    a: std.mem.Allocator,
    names: st.Header = .empty,
    data: std.ArrayList(u8) = .empty,
    q: st.Entry = weight(&.{ 256, 256 }),
    fc2: st.Entry = weight(&.{ 256, 256 }),
    layers: usize = 1,
    base: bool = false,

    fn add(f: *Fixture, name: []const u8, dtype: st.DType, shape: []const usize) !void {
        var e = weight(shape);
        e.dtype = dtype;
        e.begin = f.data.items.len;
        var n = dtype.size();
        for (shape) |v| n *= v;
        try f.data.appendNTimes(f.a, 0, n);
        e.end = f.data.items.len;
        try f.names.put(f.a, name, e);
    }

    fn set(f: *Fixture, name: []const u8, i: usize, v: i32) void {
        const e = f.names.get(name).?;
        std.mem.writeInt(i32, f.data.items[e.begin + i * 4 ..][0..4], v, .little);
    }

    fn read(f: *Fixture) !g.Geometry {
        var json: []const u8 = "{";
        for (f.names.keys(), f.names.values(), 0..) |name, e, i| {
            const dims = try std.json.Stringify.valueAlloc(f.a, e.shape[0..e.rank], .{});
            const dtype = switch (e.dtype) {
                .i32 => "I32",
                .bf16 => "BF16",
                .f32 => "F32",
                else => "U8",
            };
            json = try std.fmt.allocPrint(f.a, "{s}{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":{s},\"data_offsets\":[{d},{d}]}}", .{ json, if (i == 0) "" else ",", name, dtype, dims, e.begin, e.end });
        }
        json = try std.fmt.allocPrint(f.a, "{s}}}", .{json});
        const names = try st.parseHeader(f.a, json, f.data.items.len);
        return g.read(f.a, .{ .names = names, .data = f.data.items }, f.q, f.fc2, f.layers, f.base);
    }
};

fn weight(shape: []const usize) st.Entry {
    var e = st.Entry{ .dtype = .bf16, .rank = @intCast(shape.len), .shape = @splat(1), .begin = 0, .end = 0 };
    @memcpy(e.shape[0..shape.len], shape);
    return e;
}

fn fixture(a: std.mem.Allocator) !Fixture {
    var f = Fixture{ .a = a };
    try f.add("geometry", .i32, &.{8});
    for ([_]i32{ 1, 1, 1, 1, 2, 1, 1, 1 }, 0..) |n, i| f.set("geometry", i, n);
    try f.add("text", .bf16, &.{ 1, 256 });
    try f.add("final", .bf16, &.{ 2, 512 });
    try f.add("final_norm.weight", .bf16, &.{256});
    inline for (.{ "video", "audio" }) |kind| {
        const n = if (comptime std.mem.eql(u8, kind, "video")) 1 else 2;
        try f.add(kind ++ "_in.weight", .f32, &.{ 256, 4 });
        try f.add(kind ++ "_in.bias", .f32, &.{256});
        try f.add(kind ++ "_out.weight", .f32, &.{ 4, 256 });
        try f.add(kind ++ "_out.bias", .f32, &.{4});
        try f.add(kind, .f32, &.{ n, 4 });
        try f.add(kind ++ "_velocity", .f32, &.{ n, 4 });
        try f.add(kind ++ "_step", .f32, &.{ 2, 2 });
    }
    try f.add("condition", .f32, &.{ 1, 4 });
    try f.add("cos", .f32, &.{ 5, 2 });
    try f.add("sin", .f32, &.{ 5, 2 });
    try f.add("adaln", .i32, &.{ 2, 5 });
    try f.add("times", .i32, &.{ 2, 5 });
    try f.add("tables.0", .bf16, &.{ 6, 6, 256 });
    try f.add("tile_slot", .i32, &.{5});
    for ([_]i32{ 0, 1, 2, 3, 64 }, 0..) |n, i| f.set("tile_slot", i, n);
    try f.add("tile_sizes", .i32, &.{2});
    f.set("tile_sizes", 0, 4);
    f.set("tile_sizes", 1, 1);
    return f;
}

fn arena() std.heap.ArenaAllocator {
    return std.heap.ArenaAllocator.init(testing.allocator);
}

test "H3 counts length and optional first-frame fields" {
    for (0..10) |len| {
        var v: [9]i32 = .{ 1, 1, 1, 0, 1, 1, 0, 0, 0 };
        if (len >= 6 and len <= 8) {
            const r = try g.counts(v[0..len]);
            try testing.expectEqual(@as(usize, 2), r.rows);
        } else try testing.expectError(error.H3BadCounts, g.counts(v[0..len]));
    }
}

test "H3 negative counts every position including held frames and audio" {
    for (0..8) |i| {
        var v = [_]i32{ 1, 1, 1, 1, 2, 1, 1, 1 };
        v[i] = -1;
        try testing.expectError(error.H3NegativeCount, g.counts(&v));
        v[i] = std.math.minInt(i32);
        try testing.expectError(error.H3NegativeCount, g.counts(&v));
    }
}

test "H3 held audio zero equal and one beyond total" {
    var a = arena();
    defer a.deinit();
    var f = try fixture(a.allocator());
    for ([_]i32{ 0, 2 }) |held| {
        f.set("geometry", 7, held);
        _ = try f.read();
    }
    f.set("geometry", 7, 3);
    try testing.expectError(error.H3HeldAudioExceedsTotal, f.read());
}

test "H3 empty media refusal" {
    for ([_]usize{ 4, 5 }) |i| {
        var v = [_]i32{ 1, 1, 1, 0, 1, 1 };
        v[i] = 0;
        try testing.expectError(error.H3EmptyMedia, g.counts(&v));
    }
}

test "H3 tile limits and capacities" {
    try testing.expectError(error.H3BadTiles, g.counts(&.{ 0, 1, 1, 0, 1, 1 }));
    try testing.expectError(error.H3BadTiles, g.counts(&.{ 1, 0, 0, 0, 1, 1 }));
    try testing.expectError(error.H3BadTiles, g.counts(&.{ 1, 32768, 1, 0, 1, 32768 }));
    _ = try g.counts(&.{ 1, 32767, 32767, 0, 64, 32767 * 64 });
    try testing.expectError(error.H3TileCapacity, g.counts(&.{ 2, 1, 1, 0, 1, 1 }));
    try testing.expectError(error.H3TileCapacity, g.counts(&.{ 1, 2, 1, 0, 1, 1 }));
    try testing.expectError(error.H3TileCapacity, g.counts(&.{ 1, 1, 1, 0, 65, 1 }));
    try testing.expectError(error.H3TileCapacity, g.counts(&.{ 1, 1, 1, 0, 1, 65 }));
    _ = try g.counts(&.{ 1, 1, 1, 0, 64, 64 });
}

test "H3 keep zero all and beyond video tiles" {
    _ = try g.counts(&.{ 1, 1, 0, 0, 1, 1 });
    _ = try g.counts(&.{ 1, 1, 1, 0, 1, 1 });
    try testing.expectError(error.H3BadKeep, g.counts(&.{ 1, 1, 2, 0, 1, 1 }));
}

test "H3 row and tile count sums refuse before narrowing" {
    try testing.expectError(error.H3RowCountOverflow, g.counts(&.{ 2147483647, 1, 1, 0, 1, 1 }));
    try testing.expectError(error.H3RowCountOverflow, g.counts(&.{ 1, 1, 1, 2147483647, 1, 1 }));
    try testing.expectError(error.H3RowCountOverflow, g.sum(&.{ std.math.maxInt(usize), 1 }));
    try testing.expectEqual(@as(usize, g.scalar_max), try g.sum(&.{ g.scalar_max - 1, 1 }));
}

test "H3 serialized geometry rank type and length" {
    var a = arena();
    defer a.deinit();
    for ([_][]const usize{ &.{}, &.{ 1, 8 }, &.{5}, &.{9} }) |shape| {
        var f = try fixture(a.allocator());
        try f.add("geometry", .i32, shape);
        try testing.expectError(error.H3BadCounts, f.read());
    }
    var f = try fixture(a.allocator());
    try f.add("geometry", .f32, &.{8});
    try testing.expectError(error.H3BadCounts, f.read());
}

test "H3 checkpoint rank alignment fused layout and layer count" {
    var a = arena();
    defer a.deinit();
    for ([_][]const usize{ &.{256}, &.{ 255, 256 }, &.{ 256, 255 }, &.{ 0, 256 } }) |shape| {
        var f = try fixture(a.allocator());
        f.q = weight(shape);
        try testing.expectError(error.H3BadCheckpointShape, f.read());
    }
    var f = try fixture(a.allocator());
    f.fc2 = weight(&.{ 255, 256 });
    try testing.expectError(error.H3BadCheckpointShape, f.read());
    f.fc2 = weight(&.{ 256, 255 });
    try testing.expectError(error.H3BadCheckpointShape, f.read());
    f.fc2 = weight(&.{ 256, 256 });
    f.layers = 0;
    try testing.expectError(error.H3BadCheckpointShape, f.read());
    f.layers = 1;
    f.base = true;
    try testing.expectError(error.H3BadCheckpointShape, f.read());
    f.q = weight(&.{ 768, 256 });
    _ = try f.read();
}

test "H3 shader dimensions scalar endpoints" {
    try testing.expectError(error.H3BadDimension, g.dimension(0));
    try testing.expectError(error.H3BadDimension, g.dimension(@as(usize, g.scalar_max) + 1));
    try g.dimension(1);
    try g.dimension(g.scalar_max);
}

test "H3 every dimension-bearing tensor has required rank" {
    var a = arena();
    defer a.deinit();
    for ([_][]const u8{ "video_in.weight", "audio_in.weight", "tables.0", "cos", "final", "adaln" }) |name| {
        var f = try fixture(a.allocator());
        try f.add(name, .f32, &.{1});
        try testing.expectError(error.H3BadTensorShape, f.read());
    }
}

test "H3 zero dimensions in widths tables times and steps" {
    var a = arena();
    defer a.deinit();
    const names = [_][]const u8{ "video_in.weight", "audio_in.weight", "tables.0", "final", "adaln" };
    for (names, 0..) |name, i| {
        var f = try fixture(a.allocator());
        const shape: []const usize = if (i < 2) &.{ 256, 0 } else if (i == 2) &.{ 6, 0, 256 } else &.{ 0, 512 };
        try f.add(name, if (i < 2) .f32 else if (i == 4) .i32 else .bf16, shape);
        try testing.expectError(error.H3BadDimension, f.read());
    }
}

test "H3 rotary even nonzero and within head" {
    var a = arena();
    defer a.deinit();
    for ([_]usize{ 0, 1, 127, 130 }) |rot| {
        var f = try fixture(a.allocator());
        try f.add("cos", .f32, &.{ 5, rot });
        try testing.expectError(error.H3BadRotary, f.read());
    }
    for ([_]usize{ 2, 128 }) |rot| {
        var f = try fixture(a.allocator());
        try f.add("cos", .f32, &.{ 5, rot });
        try f.add("sin", .f32, &.{ 5, rot });
        _ = try f.read();
    }
}

test "H3 all consumed case tensors require exact shape" {
    var a = arena();
    defer a.deinit();
    const original = try fixture(a.allocator());
    for (original.names.keys()) |name| {
        if (std.mem.eql(u8, name, "geometry")) continue;
        var f = try fixture(a.allocator());
        const old = f.names.get(name).?;
        var shape = old.shape;
        shape[old.rank] = 1;
        try f.add(name, old.dtype, shape[0 .. old.rank + 1]);
        try testing.expectError(error.H3BadTensorShape, f.read());
    }
}

test "H3 table indices negative upper bound and last valid" {
    var a = arena();
    defer a.deinit();
    for ([_][]const u8{ "adaln", "times" }) |name| {
        const limit: i32 = if (std.mem.eql(u8, name, "adaln")) 6 else 2;
        const err = if (limit == 6) error.H3BadModulationIndex else error.H3BadTimeIndex;
        var f = try fixture(a.allocator());
        f.set(name, 9, limit - 1);
        _ = try f.read();
        for ([_]i32{ -1, limit }) |v| {
            f.set(name, 9, v);
            try testing.expectError(err, f.read());
        }
    }
}

test "H3 tile sizes zero above64 and mismatched sum" {
    var a = arena();
    defer a.deinit();
    var f = try fixture(a.allocator());
    for ([_]i32{ -1, 0, 65 }) |n| {
        f.set("tile_sizes", 0, n);
        try testing.expectError(error.H3BadTileSize, f.read());
    }
    f.set("tile_sizes", 0, 3);
    try testing.expectError(error.H3TileSizeMismatch, f.read());
}

test "H3 tile slots negative capacity padding and duplicate" {
    var a = arena();
    defer a.deinit();
    var f = try fixture(a.allocator());
    for ([_]i32{ -1, 128, 4 }) |n| {
        f.set("tile_slot", 0, n);
        try testing.expectError(error.H3BadTileSlot, f.read());
    }
    f.set("tile_slot", 0, 1);
    try testing.expectError(error.H3DuplicateTileSlot, f.read());
}

test "H3 tile partition separates held frames audio and generated video" {
    var a = arena();
    defer a.deinit();
    var f = try fixture(a.allocator());
    f.set("tile_slot", 0, 64);
    f.set("tile_slot", 4, 0);
    try testing.expectError(error.H3BadTilePartition, f.read());
}

test "H3 buffer arithmetic refuses usize and signed shader offset overflow" {
    const max: usize = std.math.maxInt(isize);
    try testing.expectEqual(max, try g.product(&.{ max, 1 }));
    try testing.expectError(error.H3BufferSizeOverflow, g.product(&.{ max, 2 }));
    try testing.expectError(error.H3BufferSizeOverflow, g.product(&.{ std.math.maxInt(usize), 2 }));
    var a = arena();
    defer a.deinit();
    var f = try fixture(a.allocator());
    var geometry = try f.read();
    geometry.lines = g.scalar_max;
    geometry.hidden = g.scalar_max;
    try testing.expectError(error.H3BufferSizeOverflow, geometry.buffers(false));
}

test "H3 reader checks tensor ranges even for direct callers" {
    var a = arena();
    defer a.deinit();
    var f = try fixture(a.allocator());
    var e = f.names.get("geometry").?;
    e.end = f.data.items.len + 1;
    try f.names.put(f.a, "geometry", e);
    const c = g.Case{ .names = f.names, .data = f.data.items };
    try testing.expectError(error.H3BadTensorRange, c.bytes("geometry"));
}

test "H3 forward checks step and all four latent slices" {
    try g.forward(5, 4, 8, 10, 10, 1, 4, 8, 4, 8);
    try testing.expectError(error.H3BadStep, g.forward(5, 4, 8, 10, 10, 2, 4, 8, 4, 8));
    try testing.expectError(error.H3BadStep, g.forward(5, 4, 8, 10, 9, 0, 4, 8, 4, 8));
    for (0..4) |i| {
        var lens = [_]usize{ 4, 8, 4, 8 };
        lens[i] += 1;
        try testing.expectError(error.H3BadLatentLength, g.forward(5, 4, 8, 10, 10, 0, lens[0], lens[1], lens[2], lens[3]));
    }
    try g.stepRange(0, 1);
    try testing.expectError(error.H3BadStep, g.stepRange(1, 1));
}

test {
    _ = schedule;
    _ = @import("schedule_geometry_test.zig");
}

test "H3 serialized legacy six and seven count cases remain admitted" {
    var a = arena();
    defer a.deinit();
    for ([_]usize{ 6, 7 }) |len| {
        var f = try fixture(a.allocator());
        const v = [_]i32{ 1, 1, 1, 1, 2, 1, 1 };
        try f.add("geometry", .i32, &.{len});
        for (v[0..len], 0..) |n, i| f.set("geometry", i, n);
        if (len == 6) {
            try f.add("cos", .f32, &.{ 4, 2 });
            try f.add("sin", .f32, &.{ 4, 2 });
            try f.add("adaln", .i32, &.{ 2, 4 });
            try f.add("times", .i32, &.{ 2, 4 });
            try f.add("tile_slot", .i32, &.{4});
            for ([_]i32{ 0, 1, 2, 64 }, 0..) |n, i| f.set("tile_slot", i, n);
            f.set("tile_sizes", 0, 3);
        }
        const got = try f.read();
        try testing.expectEqual(@as(usize, 0), got.layout.held_audio);
        try testing.expectEqual(@as(usize, if (len == 6) 0 else 1), got.layout.condition);
    }
}

test "H3 serialized full64 tiles no text or held frame and last valid slots" {
    var a = arena();
    defer a.deinit();
    var f = try fixture(a.allocator());
    const v = [_]i32{ 1, 1, 0, 0, 64, 64, 0, 64 };
    for (v, 0..) |n, i| f.set("geometry", i, n);
    try f.add("text", .bf16, &.{ 0, 256 });
    inline for (.{ "video", "audio" }) |kind| {
        try f.add(kind, .f32, &.{ 64, 4 });
        try f.add(kind ++ "_velocity", .f32, &.{ 64, 4 });
    }
    try f.add("cos", .f32, &.{ 128, 2 });
    try f.add("sin", .f32, &.{ 128, 2 });
    try f.add("adaln", .i32, &.{ 2, 128 });
    try f.add("times", .i32, &.{ 2, 128 });
    try f.add("tile_slot", .i32, &.{128});
    for (0..128) |i| f.set("tile_slot", i, @intCast(i));
    f.set("tile_sizes", 0, 64);
    f.set("tile_sizes", 1, 64);
    _ = try f.read();
}

test {
    _ = @import("schedule_replace_test.zig");
    _ = @import("picture_test.zig");
    _ = @import("picture_vae_test.zig");
}

/// MiniMax-H3-Image's case: text and picture rows only, no audio tensor, two rows in two tiles.
fn picture(a: std.mem.Allocator) !Fixture {
    var f = Fixture{ .a = a, .base = true, .q = weight(&.{ 768, 256 }) };
    try f.add("geometry", .i32, &.{6});
    for ([_]i32{ 1, 1, 1, 1, 0, 1 }, 0..) |n, i| f.set("geometry", i, n);
    try f.add("text", .bf16, &.{ 1, 256 });
    try f.add("final", .bf16, &.{ 2, 512 });
    try f.add("final_norm.weight", .bf16, &.{256});
    try f.add("video_in.weight", .f32, &.{ 256, 96 });
    try f.add("video_in.bias", .f32, &.{256});
    try f.add("video_out.weight", .f32, &.{ 96, 256 });
    try f.add("video_out.bias", .f32, &.{96});
    try f.add("video", .f32, &.{ 1, 96 });
    try f.add("video_velocity", .f32, &.{ 1, 96 });
    try f.add("video_step", .f32, &.{ 2, 2 });
    try f.add("cos", .f32, &.{ 2, 2 });
    try f.add("sin", .f32, &.{ 2, 2 });
    try f.add("adaln", .i32, &.{ 2, 2 });
    try f.add("times", .i32, &.{ 2, 2 });
    try f.add("tables.0", .bf16, &.{ 6, 6, 256 });
    try f.add("tile_slot", .i32, &.{2});
    f.set("tile_slot", 1, 64);
    try f.add("tile_sizes", .i32, &.{2});
    f.set("tile_sizes", 0, 1);
    f.set("tile_sizes", 1, 1);
    return f;
}

test "H3 picture case without audio rows or tensors is admitted" {
    var a = arena();
    defer a.deinit();
    var f = try picture(a.allocator());
    const got = try f.read();
    try testing.expectEqual(@as(usize, 0), got.layout.audio);
    try testing.expectEqual(@as(usize, 0), got.audio_in);
    try testing.expectEqual(@as(usize, 96), got.video_in);
    try testing.expectEqual(@as(usize, 2), got.layout.rows);
    // a first frame's (an edit's reference) rows are admitted as before
    try f.add("geometry", .i32, &.{7});
    for ([_]i32{ 1, 1, 1, 1, 0, 1, 1 }, 0..) |n, i| f.set("geometry", i, n);
    try f.add("condition", .f32, &.{ 1, 96 });
    inline for (.{ "cos", "sin" }) |name| try f.add(name, .f32, &.{ 3, 2 });
    inline for (.{ "adaln", "times" }) |name| try f.add(name, .i32, &.{ 2, 3 });
    try f.add("tile_slot", .i32, &.{3});
    for ([_]i32{ 0, 1, 64 }, 0..) |n, i| f.set("tile_slot", i, n);
    f.set("tile_sizes", 0, 2);
    try testing.expectEqual(@as(usize, 1), (try f.read()).layout.condition);
}

test "H3 picture case refuses audio rows and every audio tensor" {
    var a = arena();
    defer a.deinit();
    var f = try picture(a.allocator());
    f.set("geometry", 4, 1);
    try testing.expectError(error.H3PictureHasAudio, f.read());
    for (g.audio_tensors[1..]) |name| {
        var h = try picture(a.allocator());
        try h.add(name, .f32, &.{ 1, 4 });
        try testing.expectError(error.H3PictureHasAudio, h.read());
    }
    // with audio's input projection it is a clip's case, which needs audio rows and all of audio's tensors as before
    var h = try picture(a.allocator());
    try h.add("audio_in.weight", .f32, &.{ 256, 4 });
    try testing.expectError(error.H3EmptyMedia, h.read());
    h.set("geometry", 4, 1);
    try testing.expectError(error.MissingTensor, h.read());
}

test "H3 picture counts: no audio, held or generated, and a picture still needs its rows" {
    _ = try g.countsFor(&.{ 1, 1, 1, 1, 0, 1 }, false);
    try testing.expectError(error.H3PictureHasAudio, g.countsFor(&.{ 1, 1, 1, 1, 1, 1 }, false));
    try testing.expectError(error.H3HeldAudioExceedsTotal, g.countsFor(&.{ 1, 1, 1, 1, 0, 1, 0, 1 }, false));
    try testing.expectError(error.H3EmptyMedia, g.countsFor(&.{ 1, 1, 1, 1, 0, 0 }, false));
    // a clip's counts are unchanged: audio rows required
    try testing.expectError(error.H3EmptyMedia, g.countsFor(&.{ 1, 1, 1, 1, 0, 1 }, true));
    try testing.expectError(error.H3EmptyMedia, g.counts(&.{ 1, 1, 1, 1, 0, 1 }));
}

test "H3 picture step heads: one per step, read at its step, never in a clip" {
    var a = arena();
    defer a.deinit();
    var f = try picture(a.allocator());
    try testing.expect(!(try f.read()).step_heads);
    try f.add("video_out.weight", .f32, &.{ 2, 96, 256 });
    try testing.expectError(error.H3BadTensorShape, f.read());
    try f.add("video_out.bias", .f32, &.{ 2, 96 });
    try testing.expect((try f.read()).step_heads);
    // a head count other than the case's steps
    try f.add("video_out.weight", .f32, &.{ 3, 96, 256 });
    try f.add("video_out.bias", .f32, &.{ 3, 96 });
    try testing.expectError(error.H3StepHeadCount, f.read());
    // a clip's case keeps its single head
    var clip = try fixture(a.allocator());
    try clip.add("video_out.weight", .f32, &.{ 2, 4, 256 });
    try clip.add("video_out.bias", .f32, &.{ 2, 4 });
    try testing.expectError(error.H3BadTensorShape, clip.read());
    var plain = try fixture(a.allocator());
    try testing.expect(!(try plain.read()).step_heads);
}

test "H3 step head offsets: step i reads head i, one head reads 0, a rescheduled run is refused" {
    try testing.expectEqual([2]usize{ 0, 0 }, try g.headOffsets(0, 5, 8, 96, 5376));
    try testing.expectEqual([2]usize{ 0, 0 }, try g.headOffsets(6, 0, 6, 96, 5376));
    try testing.expectEqual([2]usize{ 5 * 96 * 5376 * 4, 5 * 96 * 4 }, try g.headOffsets(6, 5, 6, 96, 5376));
    try testing.expectError(error.H3BadStep, g.headOffsets(6, 6, 6, 96, 5376));
    try testing.expectError(error.H3StepHeadCount, g.headOffsets(6, 0, 8, 96, 5376));
    try testing.expectError(error.H3StepHeadCount, g.headOffsets(6, 0, 4, 96, 5376));
}
