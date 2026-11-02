//! Exact pack bytes against independent mapped tensors, including TP strides, tails and shard counts.
const std = @import("std");
const core = @import("core");
const pack = @import("pack.zig");
const reads = @import("pack_read.zig");
const a = std.testing.allocator;
const io = std.testing.io;
const t = std.testing;

test "DeepSeek pack reads: owned bytes equal the mapping across chunks, strides and EOF tails" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const n = 2 * reads.limit + 137;
    const image = try pack.image(a, &.{.{ .name = "w", .dtype = "U8", .shape = &.{n} }});
    defer a.free(image);
    const data_at = 8 + std.mem.readInt(u64, image[0..8], .little);
    for (image[data_at..], 0..) |*b, i| b.* = @truncate(i *% 37 +% i / 251);
    try tmp.dir.writeFile(io, .{ .sub_path = "a.safetensors", .data = image });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const path = try tmp.dir.realPathFileAlloc(io, "a.safetensors", a);
    defer a.free(path);
    var mapped = try core.safetensors.File.open(a, io, path);
    defer mapped.close(io);
    const want = mapped.get("w").?.bytes;
    var p = try pack.Pack.open(a, io, dir);
    defer p.deinit();
    const spans = [_]core.exl3_layout.Span{
        .{ .offset = 0, .len = n },
        .{ .offset = 1, .len = reads.limit + 17 },
        .{ .offset = 4095, .len = 2 },
        .{ .offset = n - 1, .len = 1 },
        .{ .offset = 3, .len = 4099, .stride = reads.limit / 2 + 17, .count = 4 },
        .{ .offset = 7, .len = 11, .stride = 3, .count = 37 },
        .{ .offset = 13, .len = 17, .stride = 0, .count = 3 },
        .{ .offset = 0, .len = 0, .count = 0 },
    };
    for (spans) |s| {
        const guarded = try a.alloc(u8, @intCast(s.bytes() + 2));
        defer a.free(guarded);
        @memset(guarded, 0xad);
        const out = guarded[1 .. guarded.len - 1];
        try p.read(io, p.get("w").?, s, out);
        try t.expectEqual(@as(u8, 0xad), guarded[0]);
        try t.expectEqual(@as(u8, 0xad), guarded[guarded.len - 1]);
        for (0..@intCast(s.count)) |i| {
            const at = s.offset + i * s.stride;
            const len: usize = @intCast(s.len);
            try t.expectEqualSlices(u8, want[@intCast(at)..][0..len], out[i * len ..][0..len]);
        }
    }
    var byte: [1]u8 = undefined;
    try t.expectError(error.BadSpan, p.read(io, p.get("w").?, .{ .offset = n, .len = 1 }, &byte));
}

test "DeepSeek pack reads: 65 shards load without a gather source cap" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    for (0..65) |i| {
        var name: [32]u8 = undefined;
        const tensor = try std.fmt.bufPrint(&name, "w{d}", .{i});
        const image = try pack.image(a, &.{.{ .name = tensor, .dtype = "U8", .shape = &.{3}, .fill = @intCast(i) }});
        defer a.free(image);
        var file: [32]u8 = undefined;
        try tmp.dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&file, "{d}.safetensors", .{i}), .data = image });
    }
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var p = try pack.Pack.open(a, io, dir);
    defer p.deinit();
    try t.expectEqual(@as(usize, 65), p.files.items.len);
    for (0..65) |i| {
        var name: [32]u8 = undefined;
        const info = p.get(try std.fmt.bufPrint(&name, "w{d}", .{i})).?;
        var out: [3]u8 = undefined;
        try p.read(io, info, .{ .offset = 0, .len = 3 }, &out);
        try t.expectEqualSlices(u8, &@as([3]u8, @splat(@intCast(i))), &out);
    }
}

test "DeepSeek pack reads: bounded scratch, repeated offsets, allocation refusal and truncation" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const bytes = try a.alloc(u8, reads.limit + 123);
    defer a.free(bytes);
    for (bytes, 0..) |*b, i| b.* = @truncate(i *% 13 +% i / 251);
    try tmp.dir.writeFile(io, .{ .sub_path = "raw.bin", .data = bytes });
    const path = try tmp.dir.realPathFileAlloc(io, "raw.bin", a);
    defer a.free(path);
    const z = try a.dupeSentinel(u8, path, 0);
    defer a.free(z);
    var r = try reads.Reader.open(a, io, z, true);
    defer r.close();
    const out = try a.alloc(u8, bytes.len - 1);
    defer a.free(out);
    try r.read(out, 1);
    try t.expect(r.scratch.len <= reads.limit);
    try t.expectEqualSlices(u8, bytes[1..], out);
    try r.read(out[0..4095], 123);
    try t.expectEqualSlices(u8, bytes[123..][0..4095], out[0..4095]);
    try t.expectError(error.ShortRead, r.read(out[0..2], bytes.len - 1));
    var failing = t.FailingAllocator.init(a, .{ .fail_index = 0 });
    var refused = try reads.Reader.open(failing.allocator(), io, z, true);
    defer refused.close();
    @memset(out[0..2], 0xad);
    try t.expectError(error.OutOfMemory, refused.read(out[0..2], 0));
    try t.expectEqualSlices(u8, &.{ 0xad, 0xad }, out[0..2]);
    var ordinary = try reads.Reader.open(failing.allocator(), io, z, false);
    defer ordinary.close();
    try ordinary.read(out[0..2], 1);
    try t.expectEqualSlices(u8, bytes[1..3], out[0..2]);
    try t.expectEqual(@as(usize, 0), ordinary.scratch.len);
}
