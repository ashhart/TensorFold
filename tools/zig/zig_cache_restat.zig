//! `zig_cache_restat CACHE_DIR ZIG_LIB_DIR [--prune]`: a record whose file still matches its digest gets today's id.
const std = @import("std");
const Io = std.Io;
const Cache = std.Build.Cache;
const File = Cache.Manifest.File;

const Seen = struct { size: u64, inode: Io.File.INode, mtime: i64, digest: Cache.BinDigest };
const Totals = struct {
    manifests: usize = 0,
    rewritten: usize = 0,
    updated: usize = 0,
    changed: usize = 0,
    unreadable: usize = 0,
    pruned: usize = 0,
    outputs: usize = 0,
};
/// A manifest, the output folder Zig names after it, and the output folders its files are in.
const Entry = struct { name: []const u8, out: Cache.HexDigest, dead: bool, uses: []const Cache.HexDigest };

const Restat = struct {
    gpa: std.mem.Allocator,
    io: Io,
    /// Zig's path prefixes in order: cwd, zig lib, local cache, global cache, build root.
    roots: [5]Io.Dir,
    seen: std.StringHashMapUnmanaged(?Seen) = .empty,
    key: std.ArrayList(u8) = .empty,
    buffer: []u8,
    totals: Totals = .{},
    entries: std.ArrayList(Entry) = .empty,

    /// What `path` under prefix `prefix` holds today, read once per run; null when it can't be read.
    fn look(r: *Restat, prefix: usize, path: []const u8) !?Seen {
        r.key.clearRetainingCapacity();
        try r.key.append(r.gpa, @intCast(prefix));
        try r.key.appendSlice(r.gpa, path);
        const entry = try r.seen.getOrPut(r.gpa, r.key.items);
        if (entry.found_existing) return entry.value_ptr.*;
        entry.key_ptr.* = try r.gpa.dupe(u8, r.key.items);
        entry.value_ptr.* = r.read(r.roots[prefix], path);
        return entry.value_ptr.*;
    }

    fn read(r: *Restat, root: Io.Dir, path: []const u8) ?Seen {
        const file = root.openFile(r.io, path, .{}) catch return null;
        defer file.close(r.io);
        const stat = file.stat(r.io) catch return null;
        if (stat.kind != .file) return null;
        var hasher = Cache.hasher_init;
        var offset: u64 = 0;
        while (true) {
            const n = file.readPositional(r.io, &.{r.buffer}, offset) catch return null;
            if (n == 0) break;
            hasher.update(r.buffer[0..n]);
            offset += n;
        }
        const mtime: i64 = @intCast(stat.mtime.toNanoseconds());
        var seen: Seen = .{ .size = stat.size, .inode = stat.inode, .mtime = mtime, .digest = undefined };
        hasher.final(&seen.digest);
        return seen;
    }

    /// Returns how many records now carry today's id, or null when `contents` isn't a complete manifest.
    fn update(r: *Restat, name: []const u8, contents: []u8) !?usize {
        var updated: usize = 0;
        var off: usize = 0;
        var input: Cache.BinDigest = @splat(0);
        const named = name.len == Cache.hex_digest_len and (std.fmt.hexToBytes(&input, name) catch null) != null;
        // Zig names an entry's output folder by hashing its manifest's name, then each record's digest in order.
        var out = Cache.hasher_init;
        out.update(&input);
        var dead = false;
        var uses: std.ArrayList(Cache.HexDigest) = .empty;
        while (off + 1 < contents.len) {
            const file_off: File.Offset = @fromBackingInt(@as(u32, @intCast(off)));
            const file = file_off.getFallible(contents) catch return null;
            const path = file_off.pathFallible(contents) catch return null;
            if (file.flags.prefix >= r.roots.len or path.len == 0) return null;
            off += File.sizeOf(path.len);
            out.update(&file.digest);
            if ((file.flags.prefix == 2 or file.flags.prefix == 3) and path.len > 2 + Cache.hex_digest_len and
                path[0] == 'o' and (path[1] == '/' or path[1] == '\\'))
                try uses.append(r.gpa, path[2..][0..Cache.hex_digest_len].*);
            if (file.flags.is_directory or file.flags.metadata_only) continue;
            const seen = try r.look(file.flags.prefix, path) orelse {
                dead = true;
                continue;
            };
            if (seen.size != file.size or !std.mem.eql(u8, &seen.digest, &file.digest)) {
                r.totals.changed += 1;
                dead = true;
                continue;
            }
            if (seen.inode == file.inode and seen.mtime == file.mtime) continue;
            file.inode = seen.inode;
            file.mtime = seen.mtime;
            updated += 1;
        }
        if (off + 1 != contents.len or contents[off] != 0) return null;
        if (named) {
            var digest: Cache.BinDigest = undefined;
            out.final(&digest);
            try r.entries.append(r.gpa, .{
                .name = try r.gpa.dupe(u8, name),
                .out = std.fmt.bytesToHex(digest, .lower),
                .dead = dead,
                .uses = uses.items,
            });
        }
        return updated;
    }

    fn manifest(r: *Restat, dir: Io.Dir, name: []const u8) !void {
        const file = dir.openFile(r.io, name, .{ .mode = .read_write }) catch {
            r.totals.unreadable += 1;
            return;
        };
        defer file.close(r.io);
        const size = (file.stat(r.io) catch {
            r.totals.unreadable += 1;
            return;
        }).size;
        const contents = try r.gpa.alloc(u8, size);
        defer r.gpa.free(contents);
        const got = file.readPositionalAll(r.io, contents, 0) catch {
            r.totals.unreadable += 1;
            return;
        };
        r.totals.manifests += 1;
        const updated = (if (got == size) try r.update(name, contents) else null) orelse {
            r.totals.unreadable += 1;
            return;
        };
        if (updated == 0) return;
        file.writePositionalAll(r.io, contents, 0) catch {
            r.totals.unreadable += 1;
            return;
        };
        r.totals.rewritten += 1;
        r.totals.updated += updated;
    }

    /// `--prune`: an entry whose files changed can't hit again, and nor can one that reads its outputs, so both go.
    fn prune(r: *Restat, manifests: Io.Dir, outputs: Io.Dir) !void {
        var gone: std.AutoHashMapUnmanaged(Cache.HexDigest, void) = .empty;
        for (r.entries.items) |e| if (e.dead) try gone.put(r.gpa, e.out, {});
        var more = true;
        while (more) {
            more = false;
            for (r.entries.items) |*e| if (!e.dead) for (e.uses) |used| if (gone.contains(used)) {
                e.dead = true;
                try gone.put(r.gpa, e.out, {});
                more = true;
                break;
            };
        }
        for (r.entries.items) |e| if (e.dead) {
            manifests.deleteFile(r.io, e.name) catch continue;
            r.totals.pruned += 1;
            outputs.access(r.io, &e.out, .{}) catch continue;
            outputs.deleteTree(r.io, &e.out) catch continue;
            r.totals.outputs += 1;
        };
    }
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const pruning = args.len == 4 and std.mem.eql(u8, args[3], "--prune");
    if (args.len != 3 and !pruning) {
        std.debug.print("usage: zig_cache_restat CACHE_DIR ZIG_LIB_DIR [--prune]\n", .{});
        std.process.exit(2);
    }
    const cwd = Io.Dir.cwd();
    const cache = try cwd.createDirPathOpen(io, args[1], .{});
    const lib = try cwd.openDir(io, args[2], .{});
    const manifests = cache.openDir(io, "h", .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("zig_cache_restat: no manifests\n", .{});
            return;
        },
        else => return err,
    };
    const arena = init.arena.allocator();
    const buffer = try arena.alloc(u8, 1 << 20);
    var r: Restat = .{ .gpa = arena, .io = io, .roots = .{ cwd, lib, cache, cache, cwd }, .buffer = buffer };
    var it = manifests.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file or std.mem.eql(u8, entry.name, "timestamp")) continue;
        try r.manifest(manifests, entry.name);
    }
    if (pruning) try r.prune(manifests, try cache.createDirPathOpen(io, "o", .{}));
    const t = r.totals;
    std.debug.print("zig_cache_restat: {d} manifests, {d} rewritten, {d} records updated, {d} changed, {d} unreadable", .{
        t.manifests, t.rewritten, t.updated, t.changed, t.unreadable,
    });
    if (pruning) std.debug.print(", {d} pruned with {d} output folders", .{ t.pruned, t.outputs });
    std.debug.print("\n", .{});
}
