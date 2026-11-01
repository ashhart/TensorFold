//! An exclusively owned compatibility/rank directory; disposable boots purge it before tier workers start.

const std = @import("std");
const Io = std.Io;
const linux = std.os.linux;

pub fn prepare(dir: Io.Dir, io: Io, fresh: bool) !void {
    // The directory descriptor holds this advisory exclusive lock until Disk.close closes it.
    if (linux.errno(linux.flock(dir.handle, 2 | 4)) != .SUCCESS) return error.SessionNamespaceBusy;
    if (!fresh) return;
    var it = dir.iterate();
    while (try it.next(io)) |entry| try discard(dir, io, entry);
}

pub fn discard(dir: Io.Dir, io: Io, entry: Io.Dir.Entry) !void {
    if (!std.mem.endsWith(u8, entry.name, ".tfs") and !std.mem.endsWith(u8, entry.name, ".tmp")) return;
    const kind = if (entry.kind == .unknown)
        (try dir.statFile(io, entry.name, .{ .follow_symlinks = false })).kind
    else
        entry.kind;
    if (kind == .file) try dir.deleteFile(io, entry.name);
}
