//! What `tensorfold rm` takes from a repo's cache folder: snapshots, the blobs only they reach, refs and bytes freed.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const Dir = std.Io.Dir;
const Stat = std.Io.File.Stat;
const hub = @import("hub.zig");

/// One entry below the repo's folder: its path from there, '/'-joined, and its own metadata, a link not followed.
const Entry = struct { rel: []const u8, stat: Stat };

/// A snapshot folder by name (a commit): the blobs its links and hard links reach, its files and its links out.
pub const Snapshot = struct { name: []const u8, blobs: []const []const u8, files: []const Stat, outside: []const []const u8 };

/// A file under refs/ by name ("main", "pr/1"), with the commit it names when it holds one.
pub const Ref = struct { name: []const u8, commit: ?[]const u8, stat: Stat };

/// A file directly in blobs/.
pub const Blob = struct { name: []const u8, stat: Stat };

/// A repo's cache folder as read: its snapshots, blobs and refs by name, and every file anywhere below it.
pub const Repo = struct {
    snapshots: []const Snapshot,
    blobs: []const Blob,
    refs: []const Ref,
    files: []const Stat,

    /// The snapshot `revision` names: a 40-hex commit with a snapshot folder, or a ref whose commit has one.
    pub fn find(r: Repo, revision: []const u8) ?usize {
        const commit = if (hub.hexSha(revision)) revision else for (r.refs) |ref| {
            if (std.mem.eql(u8, ref.name, revision)) break ref.commit orelse return null;
        } else return null;
        for (r.snapshots, 0..) |s, i| if (std.ascii.eqlIgnoreCase(s.name, commit)) return i;
        return null;
    }

    /// The names of the refs that name snapshot `name`, sorted.
    pub fn refsOf(r: Repo, a: Allocator, name: []const u8) ![]const []const u8 {
        var out: std.ArrayList([]const u8) = .empty;
        for (r.refs) |ref| if (ref.commit) |c| if (std.ascii.eqlIgnoreCase(c, name)) try out.append(a, ref.name);
        return out.items;
    }
};

/// A snapshot that goes, with the refs that name it.
pub const Gone = struct { name: []const u8, refs: []const []const u8 };

/// What a removal takes by name, the links out of the folder (only the links go) and the bytes that frees.
pub const Plan = struct {
    whole: bool,
    snapshots: []const Gone,
    blobs: []const []const u8,
    refs: []const []const u8,
    outside: []const []const u8,
    bytes: u64,
    kept_snapshots: usize = 0,
    kept_blobs: usize = 0,
};

/// Reads the repo's folder at `folder`; nothing below it is followed, and links are resolved only to classify them.
pub fn scan(a: Allocator, io: Io, folder: []const u8) !Repo {
    var dir = try Dir.cwd().openDir(io, folder, .{ .iterate = true, .follow_symlinks = false });
    defer dir.close(io);
    var all: std.ArrayList(Entry) = .empty;
    try walk(a, io, dir, "", &all);
    const real = try Dir.cwd().realPathFileAlloc(io, folder, a);
    var blobs: std.ArrayList(Blob) = .empty;
    var refs: std.ArrayList(Ref) = .empty;
    var names: std.ArrayList([]const u8) = .empty;
    var files: std.ArrayList(Stat) = .empty;
    for (all.items) |e| switch (e.stat.kind) {
        .file => {
            try files.append(a, e.stat);
            if (child(e.rel, "blobs")) |name| try blobs.append(a, .{ .name = name, .stat = e.stat });
            if (below(e.rel, "refs")) |name| try refs.append(a, .{ .name = name, .commit = try commitIn(a, io, dir, e.rel), .stat = e.stat });
        },
        .directory => if (child(e.rel, "snapshots")) |name| try names.append(a, name),
        else => {},
    };
    sortNames(names.items);
    std.mem.sort(Ref, refs.items, {}, struct {
        fn lessThan(_: void, l: Ref, r: Ref) bool {
            return std.mem.lessThan(u8, l.name, r.name);
        }
    }.lessThan);
    const snapshots = try a.alloc(Snapshot, names.items.len);
    for (snapshots, names.items) |*s, name| s.* = try snapshot(a, io, folder, real, name, all.items, blobs.items);
    return .{ .snapshots = snapshots, .blobs = blobs.items, .refs = refs.items, .files = files.items };
}

/// The removal of the whole folder (`only` null), or of snapshot `only` with the blobs no other snapshot reaches.
pub fn plan(a: Allocator, r: Repo, only: ?usize) !Plan {
    var gone: std.ArrayList(Gone) = .empty;
    var outside: std.ArrayList([]const u8) = .empty;
    for (r.snapshots, 0..) |s, i| if (only == null or only.? == i) {
        try gone.append(a, .{ .name = s.name, .refs = try r.refsOf(a, s.name) });
        try outside.appendSlice(a, s.outside);
    };
    const i = only orelse {
        const blobs = try a.alloc([]const u8, r.blobs.len);
        for (blobs, r.blobs) |*name, b| name.* = b.name;
        const refs = try a.alloc([]const u8, r.refs.len);
        for (refs, r.refs) |*name, ref| name.* = ref.name;
        return .{ .whole = true, .snapshots = gone.items, .blobs = blobs, .refs = refs, .outside = outside.items, .bytes = try freed(a, r.files) };
    };
    const s = r.snapshots[i];
    var kept: std.StringHashMapUnmanaged(void) = .empty;
    for (r.snapshots, 0..) |other, j| if (j != i) for (other.blobs) |b| try kept.put(a, b, {});
    var files: std.ArrayList(Stat) = .empty;
    try files.appendSlice(a, s.files);
    var blobs: std.ArrayList([]const u8) = .empty;
    var kept_blobs: usize = 0;
    for (s.blobs) |name| {
        if (kept.contains(name)) {
            kept_blobs += 1;
            continue;
        }
        try blobs.append(a, name);
        for (r.blobs) |b| if (std.mem.eql(u8, b.name, name)) try files.append(a, b.stat);
    }
    var refs: std.ArrayList([]const u8) = .empty;
    for (r.refs) |ref| if (ref.commit) |c| if (std.ascii.eqlIgnoreCase(c, s.name)) {
        try refs.append(a, ref.name);
        try files.append(a, ref.stat);
    };
    return .{
        .whole = false,
        .snapshots = gone.items,
        .blobs = blobs.items,
        .refs = refs.items,
        .outside = outside.items,
        .bytes = try freed(a, files.items),
        .kept_snapshots = r.snapshots.len - 1,
        .kept_blobs = kept_blobs,
    };
}

/// Every entry below `dir`, each before its contents; only real folders are entered, never a link to one.
fn walk(a: Allocator, io: Io, dir: Dir, prefix: []const u8, out: *std.ArrayList(Entry)) !void {
    var it = dir.iterate();
    while (try it.next(io)) |e| {
        const st = dir.statFile(io, e.name, .{ .follow_symlinks = false }) catch |err| switch (err) {
            error.FileNotFound => continue, // gone since it was listed
            else => return err,
        };
        const rel = if (prefix.len == 0) try a.dupe(u8, e.name) else try std.fmt.allocPrint(a, "{s}/{s}", .{ prefix, e.name });
        try out.append(a, .{ .rel = rel, .stat = st });
        if (st.kind != .directory) continue;
        var sub = try dir.openDir(io, e.name, .{ .iterate = true, .follow_symlinks = false });
        defer sub.close(io);
        try walk(a, io, sub, rel, out);
    }
}

/// One snapshot's blobs, reached by its links into blobs/ or its hard links to them, its files and its links out.
fn snapshot(a: Allocator, io: Io, folder: []const u8, real: []const u8, name: []const u8, all: []const Entry, blobs: []const Blob) !Snapshot {
    const prefix = try std.fmt.allocPrint(a, "snapshots/{s}/", .{name});
    var reached: std.StringArrayHashMapUnmanaged(void) = .empty;
    var files: std.ArrayList(Stat) = .empty;
    var outside: std.ArrayList([]const u8) = .empty;
    for (all) |e| {
        if (!std.mem.startsWith(u8, e.rel, prefix)) continue;
        switch (e.stat.kind) {
            .file => {
                try files.append(a, e.stat);
                for (blobs) |b| if (b.stat.inode == e.stat.inode) try reached.put(a, b.name, {});
            },
            .sym_link => switch (try reach(a, io, real, try std.fs.path.join(a, &.{ folder, e.rel }))) {
                .blob => |target| for (blobs) |b| if (std.mem.eql(u8, b.name, target)) try reached.put(a, b.name, {}),
                .outside => try outside.append(a, e.rel),
                .none => {},
            },
            else => {},
        }
    }
    const linked = try a.dupe([]const u8, reached.keys());
    sortNames(linked);
    sortNames(outside.items);
    return .{ .name = name, .blobs = linked, .files = files.items, .outside = outside.items };
}

/// Where a link lands with every link on the way followed: a file in blobs/, outside the folder, or neither.
const Reach = union(enum) { blob: []const u8, outside, none };

fn reach(a: Allocator, io: Io, real: []const u8, link: []const u8) !Reach {
    const target = Dir.cwd().realPathFileAlloc(io, link, a) catch |e| switch (e) {
        error.OutOfMemory => return e,
        else => return .none, // dangling or unreadable: it reaches nothing to count
    };
    const rest = inside(real, target) orelse return .outside;
    const name = inside("blobs", rest) orelse return .none;
    return if (std.mem.indexOfScalar(u8, name, std.fs.path.sep) == null) .{ .blob = name } else .none;
}

/// `path`'s part below `root` when `path` lies inside it, in the platform's separators.
fn inside(root: []const u8, path: []const u8) ?[]const u8 {
    if (path.len <= root.len + 1 or !std.mem.startsWith(u8, path, root) or path[root.len] != std.fs.path.sep) return null;
    return path[root.len + 1 ..];
}

/// The name of `rel` when it is an entry directly in the folder `top`.
fn child(rel: []const u8, top: []const u8) ?[]const u8 {
    const name = below(rel, top) orelse return null;
    return if (std.mem.indexOfScalar(u8, name, '/') == null) name else null;
}

/// `rel`'s path below the folder `top`.
fn below(rel: []const u8, top: []const u8) ?[]const u8 {
    if (rel.len <= top.len + 1 or !std.mem.startsWith(u8, rel, top) or rel[top.len] != '/') return null;
    return rel[top.len + 1 ..];
}

/// The commit a ref file names, or null when it holds anything but one 40-hex sha.
fn commitIn(a: Allocator, io: Io, dir: Dir, rel: []const u8) !?[]const u8 {
    const text = dir.readFileAlloc(io, rel, a, .limited(256)) catch |e| switch (e) {
        error.OutOfMemory => return e,
        else => return null,
    };
    const sha = std.mem.trim(u8, text, " \r\n");
    return if (hub.hexSha(sha)) sha else null;
}

/// Bytes freed when the names in `gone` are deleted: once per file, and only when no name outside them keeps it.
fn freed(a: Allocator, gone: []const Stat) !u64 {
    const Names = struct { count: u64, links: u64, size: u64 };
    var files: std.AutoHashMapUnmanaged(std.Io.File.INode, Names) = .empty;
    for (gone) |st| {
        const slot = try files.getOrPut(a, st.inode);
        if (!slot.found_existing) slot.value_ptr.* = .{ .count = 0, .links = @intCast(st.nlink), .size = st.size };
        slot.value_ptr.count += 1;
    }
    var total: u64 = 0;
    var it = files.valueIterator();
    while (it.next()) |f| {
        if (f.count >= f.links) total += f.size;
    }
    return total;
}

fn sortNames(names: [][]const u8) void {
    std.mem.sort([]const u8, names, {}, struct {
        fn lessThan(_: void, l: []const u8, r: []const u8) bool {
            return std.mem.lessThan(u8, l, r);
        }
    }.lessThan);
}
