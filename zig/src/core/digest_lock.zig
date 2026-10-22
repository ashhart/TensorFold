//! Bounded file-work stripes coordinate duplicate hashes; the index lock protects only metadata transactions.
const std = @import("std");
const descriptor = @import("digest_key.zig");
const Mutex = if (descriptor.persistent) std.c.pthread_mutex_t else u8;
pub const stripes = 64;
var index_mutex: Mutex = if (descriptor.persistent) .{} else 0;
var work_mutexes: [stripes]Mutex = @splat(if (descriptor.persistent) .{} else 0);

pub fn stripe(key: descriptor.Key) usize {
    var bytes: [24]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], key.device, .little);
    std.mem.writeInt(u128, bytes[8..24], key.inode, .little);
    return @intCast(std.hash.Wyhash.hash(0, &bytes) % stripes);
}

pub const Guard = struct {
    dir: std.Io.Dir,
    file: std.Io.File,
    mutex: *Mutex,

    pub fn open(io: std.Io, root: []const u8, slot: ?usize) !Guard {
        const mutex = if (slot) |i| &work_mutexes[i] else &index_mutex;
        if (std.c.pthread_mutex_lock(mutex) != .SUCCESS) return error.DigestLockFailed;
        errdefer _ = std.c.pthread_mutex_unlock(mutex);
        try std.Io.Dir.cwd().createDirPath(io, root);
        var dir = try std.Io.Dir.cwd().openDir(io, root, .{});
        errdefer dir.close(io);
        var name: [32]u8 = undefined;
        const path = if (slot) |i| try std.fmt.bufPrint(&name, "hash-{d:0>2}.lock", .{i}) else "index.lock";
        const file = dir.createFile(io, path, .{ .truncate = false }) catch |err| switch (err) {
            error.FileNotFound => try dir.openFile(io, path, .{ .mode = .read_write }),
            else => return err,
        };
        errdefer file.close(io);
        try file.lock(io, .exclusive);
        return .{ .dir = dir, .file = file, .mutex = mutex };
    }

    pub fn close(g: *Guard, io: std.Io) void {
        g.file.close(io);
        g.dir.close(io);
        _ = std.c.pthread_mutex_unlock(g.mutex);
    }
};
