//! A model folder's original files, cloned once before --slide first teaches it, and put back by `slide rollback`.
const std = @import("std");
const disk = @import("lanes").learned_disk;
const files = @import("originals_files.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;

/// The snapshot's folder inside the model's: loaders read only the model folder's own files, so none reads it.
pub const folder_name = ".tensorfold-original";
const partial_name = folder_name ++ ".partial"; // a snapshot being made; the next attempt starts it again
const manifest_name = "SHA256SUMS"; // `shasum -a 256 -c SHA256SUMS` inside the snapshot checks it by hand
const graphs_name = "graphs"; // the fact graph files --slide servers kept for the folder, one absolute path a line
const put_back_suffix = ".tensorfold-rollback"; // a file being put back, renamed over its name once whole

/// A file a lesson may write: a safetensors file, the shard index or config.json.
pub fn learnable(name: []const u8) bool {
    if (std.mem.endsWith(u8, name, ".safetensors")) return true;
    return std.mem.eql(u8, name, "model.safetensors.index.json") or std.mem.eql(u8, name, "config.json");
}

/// How files are kept: clones where the file system makes them (tests turn them off), free space as statvfs reads it.
pub const Options = struct {
    clones: bool = true,
    free: *const fn ([:0]const u8) ?u64 = disk.available,
    tamper: ?*const fn ([:0]const u8) void = null, // tests: a kept or put-back file changed before it is checked
    uncloned: ?*const fn ([:0]const u8) bool = null, // tests: the files this file system will not clone
    checked: bool = false, // this process made or checked the snapshot already: a later lesson reuses it unread
};

/// A snapshot made now (`made`) or found (`checked`: hashed now), its files, bytes, clones and seconds; `taught` too.
pub const Kept = struct {
    made: bool = false,
    checked: bool = false,
    files: usize = 0,
    bytes: u64 = 0,
    cloned: usize = 0,
    seconds: f64 = 0,
    taught: bool = false, // lessons reached the folder before its snapshot, which keeps them
};

/// A rollback: the files put back (bytes, clones), the lesson files and fact graphs it removed, and its seconds.
pub const Restored = struct {
    files: usize = 0,
    bytes: u64 = 0,
    cloned: usize = 0,
    removed: []const []const u8 = &.{},
    graphs: []const []const u8 = &.{},
    seconds: f64 = 0,
};

/// A process's hold on a model folder, gone when it is released or the process ends.
pub const Hold = struct {
    fd: std.c.fd_t,

    pub fn release(h: Hold) void {
        _ = std.c.close(h.fd);
    }
};

/// The folder held without waiting: shared to serve it, exclusive to teach it or roll it back; Busy when held against.
pub fn hold(a: Allocator, dir: []const u8, exclusive: bool) error{ Busy, Unlockable, OutOfMemory }!Hold {
    const fd = files.openFolder(try a.dupeSentinel(u8, dir, 0));
    if (fd < 0) return error.Unlockable;
    const kind: c_int = if (exclusive) std.c.LOCK.EX else std.c.LOCK.SH;
    const rc = std.c.flock(fd, kind | std.c.LOCK.NB);
    if (rc == 0) return .{ .fd = fd };
    const busy = std.c.errno(rc) == .AGAIN;
    _ = std.c.close(fd);
    return if (busy) error.Busy else error.Unlockable;
}

/// A server's hold (exclusive to teach the folder); null: it serves unheld, `why` the line to log; else refused, `why`.
pub fn serving(a: Allocator, dir: []const u8, teach: bool, why: *[]const u8) !?Hold {
    return hold(a, dir, teach) catch |e| {
        why.* = switch (e) {
            error.OutOfMemory => return e,
            error.Busy => if (teach) try say(a, busy_teach, dir) else try say(a, busy_serve, dir),
            error.Unlockable => if (teach) try say(a, unlocked_teach, dir) else try say(a, unlocked_serve, dir),
        };
        if (e == error.Unlockable and !teach) return null;
        return e;
    };
}

const busy_teach = "--slide rewrites the files in {s}, and another TensorFold process has that folder open: " ++
    "stop it, or teach a copy";
const busy_serve = "{s} is held by a --slide server or a `tensorfold slide rollback`, either of which may be " ++
    "rewriting its files: stop that server or let the rollback finish, then serve it";
const unlocked_teach = "{s} cannot be locked, so a rollback could rewrite it while --slide teaches it: teach a copy " ++
    "on a disk that locks (APFS, ext4, Btrfs, XFS)";
const rollback_busy = "a TensorFold server or another rollback has {s} open: stop it, then roll back";
const unlocked_serve = "note: {s} cannot be locked, so `tensorfold slide rollback` cannot tell that this server " ++
    "has it open";

fn say(a: Allocator, comptime text: []const u8, dir: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, text, .{dir});
}

/// The folder's snapshot: made from its learnable files the first time and checked against its manifest, else reused.
pub fn keep(io: Io, dir: []const u8, graph: ?[]const u8, o: Options, a: Allocator, why: *[]const u8) !Kept {
    return keepFiles(io, dir, graph, o, a, why) catch |e| {
        const text = "{s}'s original files could not be kept ({s}), so nothing was taught";
        if (why.len == 0) why.* = std.fmt.allocPrint(a, text, .{ dir, @errorName(e) }) catch "nothing was kept";
        return e;
    };
}

fn keepFiles(io: Io, dir: []const u8, graph: ?[]const u8, o: Options, a: Allocator, why: *[]const u8) !Kept {
    const t0 = Io.Clock.awake.now(io).toNanoseconds();
    const snap = try std.fs.path.join(a, &.{ dir, folder_name });
    if (try isDir(io, snap)) {
        const entries = try readManifest(a, io, snap, why);
        if (try unmatched(a, io, snap, entries, o.checked)) |name| {
            const text = "{s}/{s} is missing or does not match its line in {s}, so the way back is broken: move " ++
                "that folder aside to keep {s} as it is now, then teach again";
            why.* = try std.fmt.allocPrint(a, text, .{ snap, name, manifest_name, dir });
            return error.Damaged;
        }
        if (graph) |g| try noteGraph(a, io, snap, g);
        var out: Kept = .{ .checked = !o.checked, .files = entries.len };
        for (entries) |e| out.bytes += (try Io.Dir.cwd().statFile(io, try join(a, snap, e.name), .{})).size;
        out.seconds = since(io, t0);
        return out;
    }
    const part = try std.fs.path.join(a, &.{ dir, partial_name });
    Io.Dir.cwd().deleteTree(io, part) catch {};
    if (std.c.mkdir(try a.dupeSentinel(u8, part, 0), 0o755) != 0) return error.SnapshotFolder;
    errdefer Io.Dir.cwd().deleteTree(io, part) catch {};
    const names = try learnables(a, io, dir);
    if (names.len == 0) {
        const text = "{s} has no config.json or safetensors files to keep, so nothing was taught";
        why.* = try std.fmt.allocPrint(a, text, .{dir});
        return error.NothingToKeep;
    }
    const jobs = try a.alloc(Job, names.len);
    var copies: u64 = 0;
    var largest: u64 = 0;
    var out: Kept = .{ .made = true, .files = names.len, .taught = taughtBefore(a, io, dir, names) };
    for (names, jobs) |name, *j| {
        j.* = .{ .src = try join(a, dir, name), .dst = try join(a, part, name), .tamper = o.tamper };
        const size = (try Io.Dir.cwd().statFile(io, j.src, .{})).size;
        out.bytes += size;
        j.copy = !try cloneIf(o, j.src, j.dst);
        if (!j.copy) out.cloned += 1 else {
            copies += size;
            largest = @max(largest, size);
        }
    }
    if (copies > 0) try room(a, o, try a.dupeSentinel(u8, part, 0), copies + largest, dir, teach_fix, why);
    Pool(Job).run(jobs);
    for (names, jobs) |name, j| {
        if (j.failed) |e| {
            const text = "{s} could not be kept ({s}), so nothing was taught";
            why.* = try std.fmt.allocPrint(a, text, .{ name, @errorName(e) });
            return e;
        }
        if (!std.mem.eql(u8, &j.orig, &j.kept)) {
            const text = "the copy of {s} kept in {s} does not match the original, so it was removed and nothing " ++
                "was taught";
            why.* = try std.fmt.allocPrint(a, text, .{ name, snap });
            return error.Mismatch;
        }
    }
    var text: std.ArrayList(u8) = .empty;
    for (names, jobs) |name, j| {
        const hex = std.fmt.bytesToHex(j.orig, .lower);
        try text.print(a, "{s}  {s}\n", .{ &hex, name });
    }
    try files.writeWhole(a, try join(a, part, manifest_name), text.items);
    if (graph) |g| try noteGraph(a, io, part, g);
    files.rename(try a.dupeSentinel(u8, part, 0), try a.dupeSentinel(u8, snap, 0)) catch return error.SnapshotFolder;
    files.syncDir(a, dir);
    out.seconds = since(io, t0);
    return out;
}

/// The folder put back as its snapshot holds it, byte for byte, with what lessons added removed; refused while held.
pub fn rollback(io: Io, dir: []const u8, o: Options, a: Allocator, why: *[]const u8) !Restored {
    return rollbackFiles(io, dir, o, a, why) catch |e| {
        const text = "{s} could not be rolled back ({s}): fix that, then roll back again";
        if (why.len == 0) why.* = std.fmt.allocPrint(a, text, .{ dir, @errorName(e) }) catch "the rollback failed";
        return e;
    };
}

fn rollbackFiles(io: Io, dir: []const u8, o: Options, a: Allocator, why: *[]const u8) !Restored {
    const t0 = Io.Clock.awake.now(io).toNanoseconds();
    const h = hold(a, dir, true) catch |e| {
        why.* = switch (e) {
            error.Busy => try say(a, rollback_busy, dir),
            else => try std.fmt.allocPrint(a, "{s} cannot be opened and locked ({s})", .{ dir, @errorName(e) }),
        };
        return e;
    };
    defer h.release();
    const snap = try std.fs.path.join(a, &.{ dir, folder_name });
    if (!try isDir(io, snap)) {
        const text = "{s} has no {s} snapshot (--slide makes one before a folder's first lesson), so there is " ++
            "nothing to roll back to";
        why.* = try std.fmt.allocPrint(a, text, .{ dir, folder_name });
        return error.NoSnapshot;
    }
    const entries = try readManifest(a, io, snap, why);
    if (try unmatched(a, io, snap, entries, false)) |name| {
        const text = "{s}/{s} does not match its line in {s}, so {s} was left as it is";
        why.* = try std.fmt.allocPrint(a, text, .{ snap, name, manifest_name, dir });
        return error.Damaged;
    }
    var out: Restored = .{ .files = entries.len };
    const backs = try a.alloc(Back, entries.len);
    for (entries, backs) |e, *b| {
        const tmp = try std.fmt.allocPrintSentinel(a, "{s}/{s}{s}", .{ dir, e.name, put_back_suffix }, 0);
        b.* = .{ .src = try join(a, snap, e.name), .tmp = tmp, .live = try join(a, dir, e.name) };
        b.size = (try Io.Dir.cwd().statFile(io, b.src, .{})).size;
        out.bytes += b.size;
    }
    // nothing is put back before every clone is made and the copies fit; a stop at any point leaves no temp behind
    errdefer for (backs) |b| {
        _ = std.c.unlink(b.tmp);
    };
    var largest: u64 = 0;
    for (backs) |*b| {
        _ = std.c.unlink(b.tmp); // an interrupted rollback's
        b.cloned = try cloneIf(o, b.src, b.tmp);
        if (b.cloned) out.cloned += 1 else largest = @max(largest, b.size);
    }
    if (out.cloned < backs.len) try room(a, o, try a.dupeSentinel(u8, dir, 0), largest, dir, rollback_fix, why);
    for (entries, backs) |e, b| {
        if (!b.cloned and !std.mem.eql(u8, &try files.copyHashed(b.src, b.tmp), &e.digest)) {
            why.* = try std.fmt.allocPrint(a, "{s} changed while it was copied back: roll back again", .{b.src});
            return error.Damaged;
        }
        try files.rename(b.tmp, b.live);
    }
    files.syncDir(a, dir);
    if (o.tamper) |t| for (backs) |b| t(b.live);
    if (try unmatched(a, io, dir, entries, false)) |name| {
        const text = "{s} in {s} does not match the snapshot after it was put back: roll back again";
        why.* = try std.fmt.allocPrint(a, text, .{ name, dir });
        return error.NotRestored;
    }
    out.removed = try removeAdded(a, io, dir, entries);
    out.graphs = try removeGraphs(a, io, snap);
    files.syncDir(a, dir);
    out.seconds = since(io, t0);
    return out;
}

/// One file put back: its kept copy, the temp beside it (a clone, else a checked copy) and the name it is renamed to.
const Back = struct { src: [:0]const u8, tmp: [:0]const u8, live: [:0]const u8, size: u64 = 0, cloned: bool = false };

/// A clone of `src` at the new path `dst` where the options and the file system allow one, else false.
fn cloneIf(o: Options, src: [:0]const u8, dst: [:0]const u8) !bool {
    if (!o.clones) return false;
    if (o.uncloned) |refuses| if (refuses(src)) return false;
    return files.clone(src, dst);
}

/// The first of `entries` missing from `at` or (unless `named_only`) not its SHA-256 there, else null.
fn unmatched(a: Allocator, io: Io, at: []const u8, entries: []const Entry, named_only: bool) !?[]const u8 {
    if (named_only) {
        for (entries) |e| _ = Io.Dir.cwd().statFile(io, try join(a, at, e.name), .{}) catch return e.name;
        return null;
    }
    const jobs = try a.alloc(Job, entries.len);
    for (entries, jobs) |e, *j| j.* = .{ .src = try join(a, at, e.name), .hash_only = true };
    Pool(Job).run(jobs);
    for (entries, jobs) |e, j| if (j.failed != null or !std.mem.eql(u8, &j.orig, &e.digest)) return e.name;
    return null;
}

/// One file's hashing, and its keeping: `src` hashed (copied to `dst` first when `copy`), then the kept `dst` hashed.
const Job = struct {
    src: [:0]const u8,
    dst: [:0]const u8 = "",
    copy: bool = false,
    hash_only: bool = false,
    tamper: ?*const fn ([:0]const u8) void = null,
    orig: [32]u8 = undefined,
    kept: [32]u8 = undefined,
    failed: ?anyerror = null,

    fn run(j: *Job) void {
        j.orig = (if (j.copy) files.copyHashed(j.src, j.dst) else files.hashFile(j.src)) catch |e| {
            j.failed = e;
            return;
        };
        if (j.hash_only) return;
        if (j.tamper) |t| t(j.dst);
        j.kept = files.hashFile(j.dst) catch |e| {
            j.failed = e;
            return;
        };
    }
};

/// Every job run once, on up to eight threads (this one too, so a failed spawn only slows it).
fn Pool(comptime J: type) type {
    return struct {
        fn run(jobs: []J) void {
            var next: std.atomic.Value(usize) = .init(0);
            var threads: [7]?std.Thread = @splat(null);
            const n = @min(jobs.len -| 1, threads.len);
            for (threads[0..n]) |*t| t.* = std.Thread.spawn(.{}, work, .{ jobs, &next }) catch null;
            work(jobs, &next);
            for (threads[0..n]) |t| if (t) |x| x.join();
        }

        fn work(jobs: []J, next: *std.atomic.Value(usize)) void {
            while (true) {
                const i = next.fetchAdd(1, .monotonic);
                if (i >= jobs.len) return;
                jobs[i].run();
            }
        }
    };
}

/// A kept file's name and SHA-256, as the snapshot's manifest lists it.
const Entry = struct { name: []const u8, digest: [32]u8 };

/// The snapshot's manifest; a missing or malformed one, or a name no snapshot keeps, is Damaged.
fn readManifest(a: Allocator, io: Io, snap: []const u8, why: *[]const u8) ![]Entry {
    errdefer if (why.len == 0) {
        const text = "{s} has no readable {s}, so the way back is broken: move that folder aside, then teach again";
        why.* = std.fmt.allocPrint(a, text, .{ snap, manifest_name }) catch "the snapshot's manifest is unreadable";
    };
    const at = try join(a, snap, manifest_name);
    const text = Io.Dir.cwd().readFileAlloc(io, at, a, .limited(1 << 20)) catch return error.Damaged;
    var out: std.ArrayList(Entry) = .empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        if (line.len <= 66 or !std.mem.eql(u8, line[64..66], "  ")) return error.Damaged;
        var e: Entry = .{ .name = line[66..], .digest = undefined };
        _ = std.fmt.hexToBytes(&e.digest, line[0..64]) catch return error.Damaged;
        const plain = e.name[0] != '.' and std.mem.indexOfAny(u8, e.name, "/\\") == null;
        if (!plain or !learnable(e.name)) return error.Damaged;
        try out.append(a, e);
    }
    if (out.items.len == 0) return error.Damaged;
    return out.items;
}

/// The folder's learnable regular files, in name order.
fn learnables(a: Allocator, io: Io, dir: []const u8) ![]const []const u8 {
    var d = try Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    var it = d.iterate();
    while (try it.next(io)) |e| {
        if (!learnable(e.name) or e.name[0] == '.') continue;
        if ((try d.statFile(io, e.name, .{ .follow_symlinks = false })).kind != .file) continue;
        try names.append(a, try a.dupe(u8, e.name));
    }
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    return names.items;
}

/// Whether lessons reached the folder before its snapshot: config.json names learned neurons, or a lesson file exists.
fn taughtBefore(a: Allocator, io: Io, dir: []const u8, names: []const []const u8) bool {
    for (names) |name| if (std.mem.startsWith(u8, name, "living-weights")) return true;
    const at = join(a, dir, "config.json") catch return false;
    const config = Io.Dir.cwd().readFileAlloc(io, at, a, .limited(1 << 24)) catch return false;
    return std.mem.indexOf(u8, config, "\"tensorfold_learned_neurons\"") != null;
}

/// What only TensorFold writes in the folder and the snapshot does not list (`added`); every other file stays.
fn removeAdded(a: Allocator, io: Io, dir: []const u8, entries: []const Entry) ![]const []const u8 {
    var d = try Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var gone: std.ArrayList([]const u8) = .empty;
    var it = d.iterate();
    while (try it.next(io)) |e| {
        if (e.name[0] == '.' or !added(e.name, entries)) continue;
        if ((try d.statFile(io, e.name, .{ .follow_symlinks = false })).kind == .directory) continue;
        try gone.append(a, try a.dupe(u8, e.name));
    }
    for (gone.items) |name| if (std.c.unlink(try join(a, dir, name)) != 0) return error.RemoveFailed;
    return gone.items;
}

/// A lesson's file (living.zig), a write cut short (shard_edit's `NAME.tmp`) or a put-back temp, none of them listed.
fn added(name: []const u8, entries: []const Entry) bool {
    if (listed(name, entries)) return false;
    if (lessonFile(name)) return true;
    if (std.mem.endsWith(u8, name, ".tmp")) {
        const of = name[0 .. name.len - ".tmp".len];
        return listed(of, entries) or lessonFile(of);
    }
    return std.mem.endsWith(u8, name, put_back_suffix) and listed(name[0 .. name.len - put_back_suffix.len], entries);
}

fn listed(name: []const u8, entries: []const Entry) bool {
    for (entries) |e| if (std.mem.eql(u8, e.name, name)) return true;
    return false;
}

/// The names living.zig gives lessons: `living-weights-<digits>.safetensors`, and every lesson in one older file.
fn lessonFile(name: []const u8) bool {
    if (std.mem.eql(u8, name, "living-weights.safetensors")) return true;
    const head = "living-weights-";
    const tail = ".safetensors";
    if (name.len <= head.len + tail.len or !std.mem.startsWith(u8, name, head) or !std.mem.endsWith(u8, name, tail)) {
        return false;
    }
    for (name[head.len .. name.len - tail.len]) |c| if (!std.ascii.isDigit(c)) return false;
    return true;
}

/// The fact graph files the snapshot lists, removed where they still read as one (a JSON object with facts and next).
fn removeGraphs(a: Allocator, io: Io, snap: []const u8) ![]const []const u8 {
    const at = try join(a, snap, graphs_name);
    const list = Io.Dir.cwd().readFileAlloc(io, at, a, .limited(1 << 20)) catch |e| switch (e) {
        error.FileNotFound => return &.{},
        else => return e,
    };
    var gone: std.ArrayList([]const u8) = .empty;
    var lines = std.mem.splitScalar(u8, list, '\n');
    while (lines.next()) |path| {
        if (!std.fs.path.isAbsolute(path)) continue;
        const text = Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 26)) catch continue;
        const v = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch continue;
        if (v != .object or v.object.get("facts") == null or v.object.get("next") == null) continue;
        if (std.c.unlink(try a.dupeSentinel(u8, path, 0)) != 0) return error.RemoveFailed;
        try gone.append(a, path);
    }
    return gone.items;
}

/// `graph`, made absolute, listed once in the snapshot's graphs file.
fn noteGraph(a: Allocator, io: Io, snap: []const u8, graph: []const u8) !void {
    const path = if (std.fs.path.isAbsolute(graph)) graph else blk: {
        const here = try Io.Dir.cwd().realPathFileAlloc(io, ".", a);
        break :blk try std.fs.path.join(a, &.{ here, graph });
    };
    const file = try join(a, snap, graphs_name);
    const old = Io.Dir.cwd().readFileAlloc(io, file, a, .limited(1 << 20)) catch |e| switch (e) {
        error.FileNotFound => "",
        else => return e,
    };
    var lines = std.mem.splitScalar(u8, old, '\n');
    while (lines.next()) |line| if (std.mem.eql(u8, line, path)) return;
    try files.writeWhole(a, file, try std.mem.concat(a, u8, &.{ old, path, "\n" }));
}

const teach_fix = "free some space, or teach a copy on APFS, Btrfs or XFS";
const rollback_fix = "free some space, then roll back again";

/// Refused by name unless the file system under `at` has `need` bytes free; `fix` is what the user can do.
fn room(
    a: Allocator,
    o: Options,
    at: [:0]const u8,
    need: u64,
    dir: []const u8,
    fix: []const u8,
    why: *[]const u8,
) !void {
    const free = o.free(at);
    if (free != null and free.? >= need) return;
    var have: []const u8 = "its free space cannot be read";
    if (free) |f| have = try std.fmt.allocPrint(a, "{d:.1} GB are", .{gigabytes(f)});
    const text = "{s} is on a file system without clones, and copying its original files needs {d:.1} GB free " ++
        "where {s}: {s}";
    why.* = try std.fmt.allocPrint(a, text, .{ dir, gigabytes(need), have, fix });
    return error.NoSpace;
}

fn isDir(io: Io, path: []const u8) !bool {
    const st = Io.Dir.cwd().statFile(io, path, .{ .follow_symlinks = false }) catch |e| switch (e) {
        error.FileNotFound => return false,
        else => return e,
    };
    if (st.kind != .directory) return error.NotASnapshot;
    return true;
}

fn join(a: Allocator, dir: []const u8, name: []const u8) ![:0]const u8 {
    return std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, name }, 0);
}

fn gigabytes(n: u64) f64 {
    return @as(f64, @floatFromInt(n)) / 1e9;
}

fn since(io: Io, t0: i96) f64 {
    return @as(f64, @floatFromInt(Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e9;
}
