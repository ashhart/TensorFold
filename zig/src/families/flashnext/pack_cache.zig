//! Complete no-dump packs are immutable cache entries, published by one directory rename.
const std = @import("std");
const Io = std.Io;
const pio = @import("pack_io.zig");
const vocab = @embedFile("draft_vocab_default.txt");
const pack_names = [_][]const u8{ "pack.safetensors", "pack_mlx.safetensors", "pack_mtp_mlx.safetensors" };

pub fn defaultRoot(gpa: std.mem.Allocator, home: []const u8) ![]u8 {
    if (home.len == 0 or !std.fs.path.isAbsolute(home)) return error.CacheHomeMissing;
    return std.fs.path.join(gpa, &.{ home, ".cache", "tensorfold", "packs" });
}

pub fn directory(gpa: std.mem.Allocator, root: []const u8, identity: []const u8) ![]u8 {
    return std.fs.path.join(gpa, &.{ root, &std.fmt.bytesToHex(pio.hashBytes(identity), .lower) });
}

pub fn ready(gpa: std.mem.Allocator, io: Io, dir: []const u8, identity: []const u8) !bool {
    for (pack_names) |name| {
        const path = try std.fs.path.join(gpa, &.{ dir, name });
        defer gpa.free(path);
        if (!try pio.packReady(gpa, io, path, identity)) return false;
    }
    return true;
}

pub fn ensure(gpa: std.mem.Allocator, io: Io, model_dir: []const u8, cache_root: []const u8, identity: []const u8, builder: anytype) ![]u8 {
    const legacy = try std.fs.path.join(gpa, &.{ model_dir, "zig-pack" });
    defer gpa.free(legacy);
    if (try ready(gpa, io, legacy, identity)) return gpa.dupe(u8, legacy);
    const dir = try directory(gpa, cache_root, identity);
    errdefer gpa.free(dir);
    if (try ready(gpa, io, dir, identity)) return dir;
    try Io.Dir.cwd().createDirPath(io, cache_root);
    const lock_path = try std.fmt.allocPrint(gpa, "{s}.lock", .{dir});
    defer gpa.free(lock_path);
    const lock = try Io.Dir.cwd().createFile(io, lock_path, .{ .truncate = false, .lock = .exclusive });
    defer lock.close(io);
    // Every writer for this identity holds the persistent lock before touching its stale or final entries.
    try cleanStale(gpa, io, cache_root, std.fs.path.basename(dir));
    if (try ready(gpa, io, dir, identity)) return dir;
    var random: [16]u8 = undefined;
    var temp: []u8 = undefined;
    while (true) {
        io.random(&random);
        temp = try std.fmt.allocPrint(gpa, "{s}.part-{s}", .{ dir, &std.fmt.bytesToHex(random, .lower) });
        Io.Dir.cwd().createDir(io, temp, .default_dir) catch |err| {
            gpa.free(temp);
            if (err == error.PathAlreadyExists) continue;
            return err;
        };
        break;
    }
    defer gpa.free(temp);
    defer Io.Dir.cwd().deleteTree(io, temp) catch {};
    const path = try std.fs.path.join(gpa, &.{ temp, "draft_vocab.txt" });
    defer gpa.free(path);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = vocab });
    try builder.build(gpa, io, model_dir, temp, path);
    if (!try ready(gpa, io, temp, identity)) return error.IncompletePackBuild;
    for (pack_names ++ .{"draft_vocab.txt"}) |name| {
        const file_path = try std.fs.path.join(gpa, &.{ temp, name });
        defer gpa.free(file_path);
        const file = try Io.Dir.cwd().openFile(io, file_path, .{});
        defer file.close(io);
        try file.sync(io);
    }
    try Io.Dir.cwd().deleteTree(io, dir);
    try Io.Dir.cwd().renamePreserve(temp, .cwd(), dir, io);
    return dir;
}

fn cleanStale(gpa: std.mem.Allocator, io: Io, root: []const u8, key: []const u8) !void {
    var dir = try Io.Dir.cwd().openDir(io, root, .{ .iterate = true });
    defer dir.close(io);
    const prefix = try std.fmt.allocPrint(gpa, "{s}.part-", .{key});
    defer gpa.free(prefix);
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .directory or entry.name.len != prefix.len + 32 or !std.mem.startsWith(u8, entry.name, prefix)) continue;
        var valid = true;
        for (entry.name[prefix.len..]) |c| if (!std.ascii.isHex(c)) {
            valid = false;
            break;
        };
        if (valid) try dir.deleteTree(io, entry.name);
    }
}

test {
    _ = @import("pack_cache_test.zig");
}
