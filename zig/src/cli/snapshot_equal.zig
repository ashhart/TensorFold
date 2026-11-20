//! Byte-for-byte snapshot reuse without following directory links or loading weight files into memory.
const std = @import("std");
const Io = std.Io;
const Dir = Io.Dir;

pub fn equal(io: Io, staged_path: []const u8, cached_path: []const u8) !bool {
    const cwd = Dir.cwd();
    var staged = try cwd.openDir(io, staged_path, .{ .iterate = true, .follow_symlinks = false });
    defer staged.close(io);
    var cached = try cwd.openDir(io, cached_path, .{ .iterate = true, .follow_symlinks = false });
    defer cached.close(io);
    return sameDir(io, staged, cached);
}

fn sameDir(io: Io, staged: Dir, cached: Dir) !bool {
    var entries: usize = 0;
    var it = staged.iterate();
    while (try it.next(io)) |entry| {
        entries += 1;
        const other = cached.statFile(io, entry.name, .{ .follow_symlinks = false }) catch |e| switch (e) {
            error.FileNotFound => return false,
            else => return e,
        };
        const stat = try staged.statFile(io, entry.name, .{ .follow_symlinks = false });
        if (stat.kind != other.kind) return false;
        switch (stat.kind) {
            .directory => {
                var left = try staged.openDir(io, entry.name, .{ .iterate = true, .follow_symlinks = false });
                defer left.close(io);
                var right = try cached.openDir(io, entry.name, .{ .iterate = true, .follow_symlinks = false });
                defer right.close(io);
                if (!try sameDir(io, left, right)) return false;
            },
            .file => if (!try sameFile(io, staged, cached, entry.name)) return false,
            .sym_link => if (!try sameLink(io, staged, cached, entry.name)) return false,
            else => return false,
        }
    }
    var other_it = cached.iterate();
    while (try other_it.next(io)) |_| {
        if (entries == 0) return false;
        entries -= 1;
    }
    return entries == 0;
}

fn sameFile(io: Io, staged: Dir, cached: Dir, name: []const u8) !bool {
    const left = try staged.openFile(io, name, .{});
    defer left.close(io);
    const right = try cached.openFile(io, name, .{});
    defer right.close(io);
    const length = try left.length(io);
    if (length != try right.length(io)) return false;
    var left_buffer: [16 * 1024]u8 = undefined;
    var right_buffer: [16 * 1024]u8 = undefined;
    var left_reader = left.reader(io, &left_buffer);
    var right_reader = right.reader(io, &right_buffer);
    var remaining = length;
    while (remaining != 0) {
        const amount: usize = @intCast(@min(remaining, left_buffer.len));
        const left_bytes = left_buffer[0..amount];
        const right_bytes = right_buffer[0..amount];
        try left_reader.interface.readSliceAll(left_bytes);
        try right_reader.interface.readSliceAll(right_bytes);
        if (!std.mem.eql(u8, left_bytes, right_bytes)) return false;
        remaining -= amount;
    }
    return true;
}

fn sameLink(io: Io, staged: Dir, cached: Dir, name: []const u8) !bool {
    var left_buffer: [std.fs.max_path_bytes]u8 = undefined;
    var right_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const left_len = try staged.readLink(io, name, &left_buffer);
    const right_len = try cached.readLink(io, name, &right_buffer);
    if (!std.mem.eql(u8, left_buffer[0..left_len], right_buffer[0..right_len])) return false;
    return sameFile(io, staged, cached, name) catch |err| switch (err) {
        // Download staging can omit blobs until its links are relocated into the cache layout.
        error.FileNotFound => true,
        else => err,
    };
}
