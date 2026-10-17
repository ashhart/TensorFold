//! Content identity of the exact checkpoint shard set and index, independent of its local folder name.
const std = @import("std");
const checkpoint = @import("core").checkpoint;
const Sha256 = std.crypto.hash.sha2.Sha256;
fn add(h: *Sha256, io: std.Io, path: []const u8, name: []const u8) !void {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var length: [8]u8 = undefined;
    std.mem.writeInt(u64, &length, name.len, .little);
    h.update(&length);
    h.update(name);
    std.mem.writeInt(u64, &length, try file.length(io), .little);
    h.update(&length);
    var buffer: [64 << 10]u8 = undefined;
    var chunk: [64 << 10]u8 = undefined;
    var reader = file.reader(io, &buffer);
    while (true) {
        const n = try reader.interface.readSliceShort(&chunk);
        if (n == 0) break;
        h.update(chunk[0..n]);
    }
}
pub fn digest(a: std.mem.Allocator, io: std.Io, path: []const u8) ![32]u8 {
    const files = try checkpoint.shardFiles(a, io, path);
    defer checkpoint.freeShardFiles(a, files);
    const index = try std.fs.path.join(a, &.{ path, "model.safetensors.index.json" });
    defer a.free(index);
    var h = Sha256.init(.{});
    add(&h, io, index, "model.safetensors.index.json") catch |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    };
    // shardFiles is the loader's source of truth, including shards named by an index outside this directory.
    for (files) |file| try add(&h, io, file, file[path.len + 1 ..]);
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}
