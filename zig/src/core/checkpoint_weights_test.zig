//! Header-only admission counts what the mapped checkpoint counted, reads no tensor byte and refuses what it refused.
const std = @import("std");
const st = @import("safetensors.zig");
const ck = @import("checkpoint.zig");
const weights = @import("checkpoint_weights.zig");

fn image(dir: std.Io.Dir, name: []const u8, header: []const u8, payload: usize) !void {
    const io = std.testing.io;
    const file = try dir.createFile(io, name, .{});
    defer file.close(io);
    var prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &prefix, header.len, .little);
    try file.writePositionalAll(io, &prefix, 0);
    try file.writePositionalAll(io, header, 8);
    try file.setLength(io, 8 + header.len + payload);
}

/// The count as it was: every shard mapped through the checkpoint, every entry summed.
fn mappedCount(a: std.mem.Allocator, io: std.Io, dir: []const u8, vision: bool) !u64 {
    var model = try ck.Checkpoint.openModel(a, io, dir);
    defer model.close();
    var total: u64 = 0;
    for (model.files.items) |f| for (f.names.keys(), f.names.values()) |name, entry| {
        if (!vision and weights.visionTensor(name)) continue;
        total += entry.end - entry.begin;
    };
    return total;
}

test "checkpoint admission: a header read alone agrees with the mapped file and balances every allocation failure" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const json = "{\"x\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]}," ++
        "\"empty\":{\"dtype\":\"BF16\",\"shape\":[0,4],\"data_offsets\":[24,24]}}";
    try image(tmp.dir, "model.safetensors", json, 24);
    const path = try tmp.dir.realPathFileAlloc(io, "model.safetensors", a);
    defer a.free(path);
    var mapped = try st.File.open(a, io, path);
    defer mapped.close(io);
    var metadata = try st.Metadata.read(a, io, path);
    defer metadata.deinit();
    try std.testing.expectEqual(mapped.data, metadata.data);
    try std.testing.expectEqual(mapped.map.memory.len, metadata.file_size);
    try std.testing.expectEqual(mapped.names.count(), metadata.names.count());
    for (mapped.names.keys(), mapped.names.values(), metadata.names.keys(), metadata.names.values()) |wk, wv, gk, gv| {
        try std.testing.expectEqualStrings(wk, gk);
        try std.testing.expectEqualDeep(wv, gv);
    }
    // resizes refused, so the arena's growth sites stay the same in every optimize mode
    var growth = std.testing.FailingAllocator.init(a, .{ .resize_fail_index = 0 });
    try std.testing.checkAllAllocationFailures(growth.allocator(), struct {
        fn run(allocator: std.mem.Allocator, file: []const u8) !void {
            var value = try st.Metadata.read(allocator, std.testing.io, file);
            defer value.deinit();
            try std.testing.expectEqual(@as(usize, 2), value.names.count());
        }
    }.run, .{path});
    try std.testing.expectEqual(growth.allocated_bytes, growth.freed_bytes);
}

test "checkpoint admission: mixed shards count what the mapped checkpoint counted, a repeated name each time" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const mixed = "{\"language_model.w\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[0,8]}," ++
        "\"vision_model.w\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[8,20]}}";
    const tower = "{\"mlp1.w\":{\"dtype\":\"U8\",\"shape\":[5],\"data_offsets\":[0,5]}," ++
        "\"language_model.v\":{\"dtype\":\"U8\",\"shape\":[7],\"data_offsets\":[5,12]}}";
    try image(tmp.dir, "model-1.safetensors", mixed, 20);
    try image(tmp.dir, "model-2.safetensors", tower, 12);
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    try std.testing.expectEqual(@as(u64, 15), try weights.count(a, io, dir, false));
    try std.testing.expectEqual(@as(u64, 32), try weights.count(a, io, dir, true));
    try image(tmp.dir, "model-3.safetensors", mixed, 20);
    for ([_]bool{ false, true }) |vision| {
        try std.testing.expectEqual(try mappedCount(a, io, dir, vision), try weights.count(a, io, dir, vision));
    }
    try std.testing.expectEqual(@as(u64, 23), try weights.count(a, io, dir, false));
    try tmp.dir.writeFile(io, .{ .sub_path = ".tensorfold-save-pending", .data = "pending" });
    try std.testing.expectError(error.CheckpointSaveUncertain, weights.count(a, io, dir, false));
    try tmp.dir.deleteFile(io, ".tensorfold-save-pending");
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"a\":\"model-9.safetensors\"}}" });
    try std.testing.expectError(error.FileNotFound, weights.count(a, io, dir, false));
}

test "checkpoint admission: a short file, an oversized header, a short payload and overlapping ranges refuse" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const file_path = try std.fs.path.join(a, &.{ dir, "model.safetensors" });
    defer a.free(file_path);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = "short" });
    try std.testing.expectError(error.BadSafetensors, st.Metadata.read(a, io, file_path));
    try std.testing.expectError(error.BadSafetensors, weights.count(a, io, dir, false));
    var too_long: [8]u8 = undefined;
    std.mem.writeInt(u64, &too_long, st.max_header_bytes + 1, .little);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &too_long });
    try std.testing.expectError(error.BadSafetensors, st.Metadata.read(a, io, file_path));
    try image(tmp.dir, "model.safetensors", "{\"x\":{\"dtype\":\"U8\",\"shape\":[9],\"data_offsets\":[0,9]}}", 8);
    try std.testing.expectError(error.BadSafetensors, weights.count(a, io, dir, false));
    const overlap = "{\"x\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]}," ++
        "\"y\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[4,12]}}";
    try image(tmp.dir, "model.safetensors", overlap, 12);
    try std.testing.expectError(error.OverlappingTensors, weights.count(a, io, dir, false));
}

const ReadProbe = struct { base: std.Io, bytes: usize = 0, beyond_header: bool = false, header_end: usize };
threadlocal var read_probe: ?*ReadProbe = null;

fn observedRead(ctx: ?*anyopaque, file: std.Io.File, buffers: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
    const p = read_probe.?;
    var requested: usize = 0;
    for (buffers) |buffer| requested += buffer.len;
    if (offset > p.header_end or requested > p.header_end - @min(offset, p.header_end)) p.beyond_header = true;
    const n = try p.base.vtable.fileReadPositional(ctx, file, buffers, offset);
    p.bytes += n;
    return n;
}

test "checkpoint admission: the only reads are the length prefix and the header" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const json = "{\"x\":{\"dtype\":\"U8\",\"shape\":[67108864],\"data_offsets\":[0,67108864]}}";
    try image(tmp.dir, "model.safetensors", json, 64 << 20);
    const path = try tmp.dir.realPathFileAlloc(io, "model.safetensors", a);
    defer a.free(path);
    var probe: ReadProbe = .{ .base = io, .header_end = 8 + json.len };
    var vtable = io.vtable.*;
    vtable.fileReadPositional = observedRead;
    read_probe = &probe;
    defer read_probe = null;
    var metadata = try st.Metadata.read(a, .{ .userdata = io.userdata, .vtable = &vtable }, path);
    defer metadata.deinit();
    try std.testing.expectEqual(@as(usize, 8 + json.len), probe.bytes);
    try std.testing.expect(!probe.beyond_header);
    try std.testing.expectEqual(@as(usize, 64 << 20), metadata.names.get("x").?.end);
}

test "checkpoint admission: a 64 MiB shard counts within 128 KiB when mmap falls back to reads" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const header = "{\"language_model.w\":{\"dtype\":\"U8\",\"shape\":[67108864],\"data_offsets\":[0,67108864]}}";
    try image(tmp.dir, "model.safetensors", header, 64 << 20);
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    defer a.free(dir);
    var storage: [128 << 10]u8 = undefined;
    var bounded: std.heap.FixedBufferAllocator = .init(&storage);
    var threaded: std.Io.Threaded = .init(bounded.allocator(), .{ .disable_memory_mapping = true });
    defer threaded.deinit();
    try std.testing.expectEqual(@as(u64, 64 << 20), try weights.count(bounded.allocator(), threaded.io(), dir, false));
}
