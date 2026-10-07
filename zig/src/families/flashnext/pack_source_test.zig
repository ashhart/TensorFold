//! A reader accepts a pack from the checkpoint beside it and refuses a different or unmarked pack.
const std = @import("std");
const Io = std.Io;
const pack = @import("pack.zig");
const pio = @import("pack_io.zig");
const fixture = @import("pack_test.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;

fn buildPacks(a: std.mem.Allocator, tmp: std.testing.TmpDir) !void {
    try fixture.writeCheckpoint(tmp, a);
    try tmp.dir.createDirPath(io, "out");
    const model_dir = try fixture.tmpPath(a, tmp, ".");
    const out_dir = try fixture.tmpPath(a, tmp, "out");
    _ = try pack.build(a, io, model_dir, out_dir, null);
}

test "the reader accepts a pack whose identity matches the checkpoint beside it" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try buildPacks(a, tmp);
    const model_dir = try fixture.tmpPath(a, tmp, ".");
    const pack_path = try fixture.tmpPath(a, tmp, "out/pack.safetensors");
    try pio.checkSource(a, io, pack_path, model_dir);
    // the mapped-bytes form the engine uses agrees
    const image = try tmp.dir.readFileAlloc(io, "out/pack.safetensors", a, .limited(1 << 30));
    const identity = try pio.sourceIdentity(a, io, model_dir);
    try pio.checkSourceMapped(a, image, identity, "pack.safetensors");
}

test "the reader refuses a pack whose checkpoint changed after the build" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try buildPacks(a, tmp);
    // a shard that grows after the build changes its size and header hash
    const image = try tmp.dir.readFileAlloc(io, "model.safetensors", a, .limited(1 << 30));
    const grown = try a.alloc(u8, image.len + 1);
    @memcpy(grown[0..image.len], image);
    grown[image.len] = 'x';
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = grown });
    const model_dir = try fixture.tmpPath(a, tmp, ".");
    const pack_path = try fixture.tmpPath(a, tmp, "out/pack.safetensors");
    try std.testing.expectError(error.PackSourceMismatch, pio.checkSource(a, io, pack_path, model_dir));
}

test "the reader loads a pack with no recorded source and refuses a mismatching one" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // every dump the Python tool makes has no __metadata__, so an unmarked pack loads with a warning
    var header: std.Io.Writer.Allocating = .init(a);
    try header.writer.writeAll("{\"w\":{\"dtype\":\"U32\",\"shape\":[1,1],\"data_offsets\":[0,4]}}");
    const head_bytes = header.written();
    var image = try a.alloc(u8, 8 + head_bytes.len + 4);
    std.mem.writeInt(u64, image[0..8], head_bytes.len, .little);
    @memcpy(image[8..][0..head_bytes.len], head_bytes);
    @memset(image[8 + head_bytes.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = "pack.safetensors", .data = image });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{}" });
    const pack_path = try fixture.tmpPath(a, tmp, "pack.safetensors");
    const model_dir = try fixture.tmpPath(a, tmp, ".");
    try pio.checkSource(a, io, pack_path, model_dir);
    // the mapped-bytes form agrees, whatever identity the checkpoint claims
    const read = try tmp.dir.readFileAlloc(io, "pack.safetensors", a, .limited(1 << 30));
    try pio.checkSourceMapped(a, read, "index absent\n", "pack.safetensors");
    // a pack whose recorded source names another checkpoint still refuses
    const marked_json = try std.fmt.allocPrint(a, "{{\"__metadata__\":{{\"tf_source\":\"index {s}\"}},\"w\":{{\"dtype\":\"U32\",\"shape\":[1,1],\"data_offsets\":[0,4]}}}}", .{"0000000000000000000000000000000000000000000000000000000000000000"});
    const marked = try a.alloc(u8, 8 + marked_json.len + 4);
    std.mem.writeInt(u64, marked[0..8], marked_json.len, .little);
    @memcpy(marked[8..][0..marked_json.len], marked_json);
    @memset(marked[8 + marked_json.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = "marked.safetensors", .data = marked });
    try std.testing.expectError(error.PackSourceMismatch, pio.checkSource(a, io, try fixture.tmpPath(a, tmp, "marked.safetensors"), model_dir));
}

test "the io and mapped identity paths produce the same text" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture.writeCheckpoint(tmp, a);
    const model_dir = try fixture.tmpPath(a, tmp, ".");
    const through_io = try pio.sourceIdentity(a, io, model_dir);
    // the engine's path: mapped bytes for the index and every weight_map shard
    const index_bytes = try tmp.dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(1 << 30));
    var parsed = try std.json.parseFromSlice(std.json.Value, a, index_bytes, .{});
    defer parsed.deinit();
    var shards: std.ArrayList(pio.MappedShard) = .empty;
    var seen: std.ArrayList([]const u8) = .empty;
    var it = parsed.value.object.get("weight_map").?.object.iterator();
    while (it.next()) |kv| {
        var dup = false;
        for (seen.items) |n| if (std.mem.eql(u8, n, kv.value_ptr.string)) {
            dup = true;
            break;
        };
        if (dup) continue;
        try seen.append(a, kv.value_ptr.string);
        const bytes = try tmp.dir.readFileAlloc(io, kv.value_ptr.string, a, .limited(1 << 30));
        try shards.append(a, .{ .name = kv.value_ptr.string, .size = bytes.len, .bytes = bytes });
    }
    const through_maps = try pio.identityFromMapped(a, index_bytes, shards.items);
    try std.testing.expectEqualStrings(through_io, through_maps);
    try std.testing.expect(std.mem.startsWith(u8, through_maps, "index "));
}

test "the identity string does not depend on the shard list's order" {
    const a = gpa;
    const h1 = pio.hashBytes("a");
    const h2 = pio.hashBytes("b");
    const one = [_]pio.ShardId{
        .{ .name = "model-00001.safetensors", .size = 10, .header_sha256 = h1 },
        .{ .name = "model-00002.safetensors", .size = 20, .header_sha256 = h2 },
    };
    const two = [_]pio.ShardId{ one[1], one[0] };
    const s1 = try pio.identityString(a, h1, one[0..]);
    defer a.free(s1);
    const s2 = try pio.identityString(a, h1, two[0..]);
    defer a.free(s2);
    try std.testing.expectEqualStrings(s1, s2);
    try std.testing.expect(std.mem.indexOf(u8, s1, "shard model-00001.safetensors 10 ") != null);
    try std.testing.expect(std.mem.indexOf(u8, s1, "shard model-00002.safetensors 20 ") != null);
}

test "the no-dump pack cache is ready only complete and matching, resolved from any working directory" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture.writeCheckpoint(tmp, a);
    try tmp.dir.createDirPath(io, "cache");
    const model_dir = try fixture.tmpPath(a, tmp, ".");
    const cache_dir = try fixture.tmpPath(a, tmp, "cache");
    const identity = try pio.sourceIdentity(a, io, model_dir);
    // an empty cache: not ready, the loader builds
    try std.testing.expect(!try pio.packsReady(a, io, cache_dir, identity));
    // a real build into the cache: ready
    _ = try pack.build(a, io, model_dir, cache_dir, null);
    try std.testing.expect(try pio.packsReady(a, io, cache_dir, identity));
    // a cache interrupted between the two pack writes: not ready, the loader rebuilds
    try tmp.dir.deleteFile(io, "cache/pack_mlx.safetensors");
    try std.testing.expect(!try pio.packsReady(a, io, cache_dir, identity));
    // an empty pack_mlx.safetensors is as bad as a missing one: Prompt.init would index nothing
    try tmp.dir.writeFile(io, .{ .sub_path = "cache/pack_mlx.safetensors", .data = "" });
    try std.testing.expect(!try pio.packsReady(a, io, cache_dir, identity));
    try tmp.dir.deleteFile(io, "cache/pack_mlx.safetensors");
    // a pack built before the checkpoint changed: not ready
    _ = try pack.build(a, io, model_dir, cache_dir, null);
    const image = try tmp.dir.readFileAlloc(io, "model.safetensors", a, .limited(1 << 30));
    const grown = try a.alloc(u8, image.len + 1);
    @memcpy(grown[0..image.len], image);
    grown[image.len] = 'x';
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = grown });
    const changed = try pio.sourceIdentity(a, io, model_dir);
    try std.testing.expect(!try pio.packsReady(a, io, cache_dir, changed));
}
