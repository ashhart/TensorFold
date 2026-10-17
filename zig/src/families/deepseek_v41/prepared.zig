//! Prepared weight folders (TF_DSV41_PREPARED, _WRITE, _VERIFY): built layer images cached on disk for a fast boot.
const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const named = @import("named.zig");
const pack_mod = @import("pack.zig");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const magic = "TFDSV41P";
const version: u32 = 1;
const page: u64 = 4096;

/// What the images depend on besides the pack's bytes.
pub const Key = struct {
    rank: u32,
    world: u32,
    o_groups: u32,
    blocks: []const u32,
    dspark: bool,
};

/// The builders' source, so a change to how images are made never reads an old folder.
const sources = [_][]const u8{ @embedFile("named.zig"), @embedFile("plan.zig"), @embedFile("stage.zig"), @embedFile("pack.zig"), @embedFile("exl3.zig") };

pub const Stats = struct { hits: u32 = 0, misses: u32 = 0, written: u32 = 0, read_bytes: u64 = 0, read_ns: u64 = 0, write_ns: u64 = 0 };

pub const Cache = struct {
    gpa: Allocator,
    io: Io,
    /// ``<dir>/<key>/rank<r>``
    root: []u8,
    write: bool,
    verify: bool,
    stats: Stats = .{},

    /// The cache TF_DSV41_PREPARED names, or null when unset.
    pub fn fromEnv(gpa: Allocator, io: Io, p: *const pack_mod.Pack, key: Key) !?Cache {
        const dir = std.mem.span(std.c.getenv("TF_DSV41_PREPARED") orelse return null);
        if (std.mem.trim(u8, dir, " ").len == 0) return null;
        const flag = struct {
            fn on(name: [*:0]const u8) bool {
                const v = std.c.getenv(name) orelse return false;
                return std.mem.eql(u8, std.mem.span(v), "1");
            }
        }.on;
        return try open(gpa, io, dir, try packDigest(io, p), key, flag("TF_DSV41_PREPARED_WRITE"), flag("TF_DSV41_PREPARED_VERIFY"));
    }

    pub fn open(gpa: Allocator, io: Io, dir: []const u8, pack_digest: [32]u8, key: Key, write: bool, verify: bool) !Cache {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update(magic);
        h.update(std.mem.asBytes(&version));
        for (sources) |s| h.update(s);
        h.update(&pack_digest);
        h.update(std.mem.asBytes(&[_]u32{ key.rank, key.world, key.o_groups, @intFromBool(key.dspark), @intCast(key.blocks.len) }));
        h.update(std.mem.sliceAsBytes(key.blocks));
        var d: [32]u8 = undefined;
        h.final(&d);
        const hex = std.fmt.bytesToHex(d[0..8].*, .lower);
        const root = try std.fmt.allocPrint(gpa, "{s}/{s}/rank{d}", .{ dir, &hex, key.rank });
        errdefer gpa.free(root);
        if (write) try Io.Dir.cwd().createDirPath(io, root);
        return .{ .gpa = gpa, .io = io, .root = root, .write = write, .verify = verify };
    }

    pub fn deinit(c: *Cache) void {
        c.gpa.free(c.root);
    }

    fn path(c: *const Cache, buf: []u8, tag: []const u8, suffix: []const u8) ![]const u8 {
        return std.fmt.bufPrint(buf, "{s}/{s}.bin{s}", .{ c.root, tag, suffix });
    }

    /// Asks the kernel to read `<tag>.bin` ahead (POSIX_FADV_WILLNEED); a missing file is ignored, nothing waits.
    pub fn willNeed(c: *const Cache, tag: []const u8) void {
        var pbuf: [std.fs.max_path_bytes]u8 = undefined;
        const p = c.path(pbuf[0 .. pbuf.len - 1], tag, "") catch return;
        pbuf[p.len] = 0;
        const linux = std.os.linux;
        const rc = linux.openat(linux.AT.FDCWD, pbuf[0..p.len :0], .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
        if (linux.errno(rc) != .SUCCESS) return;
        const fd: linux.fd_t = @intCast(rc);
        defer _ = linux.close(fd);
        _ = linux.fadvise(fd, 0, 0, linux.POSIX_FADV.WILLNEED);
    }

    /// Fills `b` from ``<tag>.bin``: true when read, false when there is no usable file (`b` is then untouched).
    pub fn get(c: *Cache, b: *named.Builder, tag: []const u8) !bool {
        var pbuf: [std.fs.max_path_bytes]u8 = undefined;
        const t0 = Io.Clock.awake.now(c.io).toNanoseconds();
        const file = Io.Dir.cwd().openFile(c.io, try c.path(&pbuf, tag, ""), .{}) catch {
            c.stats.misses += 1;
            return false;
        };
        defer file.close(c.io);
        const got = readInto(c.gpa, c.io, file, b, c.verify) catch |e| {
            if (!quiet) std.log.warn("dsv41 prepared: {s} unusable ({t}); building it from the pack", .{ pbuf[0 .. c.root.len + tag.len + 5], e });
            c.stats.misses += 1;
            return false;
        };
        c.stats.hits += 1;
        c.stats.read_bytes += got;
        c.stats.read_ns += @intCast(Io.Clock.awake.now(c.io).toNanoseconds() - t0);
        return true;
    }

    /// Writes ``<tag>.bin`` from `b` when writing is on and the file is missing.
    pub fn put(c: *Cache, b: *const named.Builder, tag: []const u8) !void {
        if (!c.write) return;
        var pbuf: [std.fs.max_path_bytes]u8 = undefined;
        var tbuf: [std.fs.max_path_bytes]u8 = undefined;
        const final_path = try c.path(&pbuf, tag, "");
        const tmp_path = try c.path(&tbuf, tag, ".tmp");
        const t0 = Io.Clock.awake.now(c.io).toNanoseconds();
        const dir = Io.Dir.cwd();
        {
            const file = try dir.createFile(c.io, tmp_path, .{ .truncate = true });
            defer file.close(c.io);
            try writeOut(c.gpa, c.io, file, b);
        }
        try dir.rename(tmp_path, dir, final_path, c.io);
        c.stats.written += 1;
        c.stats.write_ns += @intCast(Io.Clock.awake.now(c.io).toNanoseconds() - t0);
    }
};

/// The pack's shards as they are on disk: each file's name, size and modification time.
pub fn packDigest(io: Io, p: *const pack_mod.Pack) ![32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    for (p.files.items) |f| {
        h.update(f);
        const st = Io.Dir.cwd().statFile(io, try std.fmt.bufPrint(&buf, "{s}/{s}", .{ p.dir, f }), .{}) catch |e| switch (e) {
            error.FileNotFound => {
                h.update("missing");
                continue;
            },
            else => return e,
        };
        h.update(std.mem.asBytes(&[_]i128{ @intCast(st.size), st.mtime.toNanoseconds() }));
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    return d;
}

const Entry = extern struct {
    kind: u8,
    rank: u8,
    name_len: u16,
    pad: u32 = 0,
    shape: [4]u64,
    offset: u64,
    len: u64,
    sha256: [32]u8,
};

fn writeOut(gpa: Allocator, io: Io, file: Io.File, b: *const named.Builder) !void {
    var head: std.ArrayList(u8) = .empty;
    defer head.deinit(gpa);
    try head.appendSlice(gpa, magic);
    try head.appendSlice(gpa, std.mem.asBytes(&version));
    try head.appendSlice(gpa, std.mem.asBytes(&@as(u32, @intCast(b.out.items.len))));
    var size: u64 = 16;
    for (b.out.items) |n| size += @sizeOf(Entry) + n.name.len;
    var at = std.mem.alignForward(u64, size, page);
    for (b.out.items) |n| {
        var e: Entry = .{ .kind = @backingInt(n.kind), .rank = n.rank, .name_len = @intCast(n.name.len), .shape = undefined, .offset = at, .len = n.bytes.len, .sha256 = undefined };
        for (&e.shape, n.shape) |*d, s| d.* = s;
        std.crypto.hash.sha2.Sha256.hash(n.bytes, &e.sha256, .{});
        try head.appendSlice(gpa, std.mem.asBytes(&e));
        try head.appendSlice(gpa, n.name);
        at = std.mem.alignForward(u64, at + n.bytes.len, page);
    }
    try file.writePositionalAll(io, head.items, 0);
    var cursor = std.mem.alignForward(u64, size, page);
    for (b.out.items) |n| {
        try file.writePositionalAll(io, n.bytes, cursor);
        cursor = std.mem.alignForward(u64, cursor + n.bytes.len, page);
    }
    try file.setLength(io, cursor);
}

/// Reads every tensor of a file into `b` (each into its own buffer); returns payload bytes. On error `b` gets nothing.
fn readInto(gpa: Allocator, io: Io, file: Io.File, b: *named.Builder, verify: bool) !u64 {
    var fixed: [16]u8 = undefined;
    if (try file.readPositionalAll(io, &fixed, 0) != 16) return error.Truncated;
    if (!std.mem.eql(u8, fixed[0..8], magic)) return error.BadMagic;
    if (std.mem.readInt(u32, fixed[8..12], .little) != version) return error.BadVersion;
    const count = std.mem.readInt(u32, fixed[12..16], .little);
    const total = try file.length(io);
    var got: std.ArrayList(named.Named) = .empty;
    errdefer {
        for (got.items) |n| {
            gpa.free(n.name);
            gpa.free(n.bytes);
        }
        got.deinit(gpa);
    }
    var off: u64 = 16;
    var bytes: u64 = 0;
    for (0..count) |_| {
        var e: Entry = undefined;
        if (try file.readPositionalAll(io, std.mem.asBytes(&e), off) != @sizeOf(Entry)) return error.Truncated;
        off += @sizeOf(Entry);
        if (e.offset + e.len > total or e.rank > 4 or e.kind > @backingInt(named.Kind.i16)) return error.BadEntry;
        const name = try gpa.alloc(u8, e.name_len);
        errdefer gpa.free(name);
        if (try file.readPositionalAll(io, name, off) != name.len) return error.Truncated;
        off += e.name_len;
        const data = try gpa.alignedAlloc(u8, .@"16", e.len);
        errdefer gpa.free(data);
        if (try file.readPositionalAll(io, data, e.offset) != data.len) return error.Truncated;
        if (verify) {
            var sha: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(data, &sha, .{});
            if (!std.mem.eql(u8, &sha, &e.sha256)) return error.DigestMismatch;
        }
        var n: named.Named = .{ .name = name, .kind = @fromBackingInt(@intCast(e.kind)), .shape = undefined, .rank = e.rank, .bytes = data };
        for (&n.shape, e.shape) |*d, s| d.* = @intCast(s);
        try got.append(gpa, n);
        bytes += e.len;
    }
    try b.out.appendSlice(b.gpa, got.items);
    got.deinit(gpa);
    return bytes;
}

test "a layer's images round-trip through its prepared file, and another key misses" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{&tmp.sub_path});
    defer gpa.free(root);
    var p = pack_mod.Pack.init(gpa);
    defer p.deinit();
    const key: Key = .{ .rank = 1, .world = 2, .o_groups = 8, .blocks = &.{ 0, 1, 2 }, .dspark = true };
    var c = try Cache.open(gpa, io, root, try packDigest(io, &p), key, true, true);
    defer c.deinit();
    var b: named.Builder = .{ .gpa = gpa, .io = io, .pack = &p };
    defer b.deinit();
    const sizes = [_]usize{ 4096, 5000, 16, 1 };
    for (sizes, 0..) |n, i| {
        const bytes = try gpa.alignedAlloc(u8, .@"16", n);
        for (bytes, 0..) |*x, j| x.* = @truncate(j *% 31 +% i);
        try b.out.append(gpa, .{ .name = try std.fmt.allocPrint(gpa, "L1.t{d}.trellis", .{i}), .kind = @fromBackingInt(@intCast(i % 4)), .shape = .{ n, 1, 1, 1 }, .rank = @intCast(1 + i % 4), .bytes = bytes });
    }
    var miss: named.Builder = .{ .gpa = gpa, .io = io, .pack = &p };
    defer miss.deinit();
    try std.testing.expect(!try c.get(&miss, "L1"));
    try c.put(&b, "L1");
    var back: named.Builder = .{ .gpa = gpa, .io = io, .pack = &p };
    defer back.deinit();
    try std.testing.expect(try c.get(&back, "L1"));
    try std.testing.expectEqual(b.out.items.len, back.out.items.len);
    for (b.out.items, back.out.items) |x, y| {
        try std.testing.expectEqualStrings(x.name, y.name);
        try std.testing.expectEqual(x.kind, y.kind);
        try std.testing.expectEqual(x.rank, y.rank);
        try std.testing.expectEqualSlices(usize, &x.shape, &y.shape);
        try std.testing.expectEqualSlices(u8, x.bytes, y.bytes);
        try std.testing.expect(std.mem.isAligned(@intFromPtr(y.bytes.ptr), 16));
    }
    try std.testing.expectEqual(@as(u32, 1), c.stats.hits);
    // another rank (or blocks, pack, builder source) is another folder
    var other = try Cache.open(gpa, io, root, try packDigest(io, &p), .{ .rank = 0, .world = 2, .o_groups = 8, .blocks = &.{ 0, 1, 2 }, .dspark = true }, false, false);
    defer other.deinit();
    var none: named.Builder = .{ .gpa = gpa, .io = io, .pack = &p };
    defer none.deinit();
    try std.testing.expect(!try other.get(&none, "L1"));
    // a damaged payload is refused with VERIFY and the builder stays empty
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const f = try Io.Dir.cwd().openFile(io, try c.path(&pbuf, "L1", ""), .{ .mode = .read_write });
    try f.writePositionalAll(io, "x", page);
    f.close(io);
    var bad: named.Builder = .{ .gpa = gpa, .io = io, .pack = &p };
    defer bad.deinit();
    try std.testing.expect(!try c.get(&bad, "L1"));
    try std.testing.expectEqual(@as(usize, 0), bad.out.items.len);
}
