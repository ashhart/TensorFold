//! Prepared weight folders (TF_DSV41_PREPARED, _WRITE, _VERIFY): built layer images cached on disk for a fast boot.
const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const named = @import("named.zig");
const pack_mod = @import("pack.zig");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const magic = "TFDSV41P";
const version: u32 = 2;
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
const sources = [_][]const u8{ @embedFile("named.zig"), @embedFile("plan.zig"), @embedFile("stage.zig"), @embedFile("pack.zig"), @embedFile("projection.zig"), @embedFile("names.zig"), @import("core").exl3_layout.source_bytes, @import("prepare.zig").source_bytes };

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
        const hex = std.fmt.bytesToHex(d, .lower);
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
    pub fn get(c: *Cache, b: *named.Builder, tag: []const u8, expected: []const named.Named) !bool {
        var pbuf: [std.fs.max_path_bytes]u8 = undefined;
        const t0 = Io.Clock.awake.now(c.io).toNanoseconds();
        const file = Io.Dir.cwd().openFile(c.io, try c.path(&pbuf, tag, ""), .{}) catch {
            c.stats.misses += 1;
            return false;
        };
        defer file.close(c.io);
        const got = readInto(b.gpa, c.io, file, b, c.verify, expected) catch |e| {
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

/// Content identity includes shard names, lengths, headers and all tensor bytes. Stat equality is insufficient.
pub fn packDigest(io: Io, p: *const pack_mod.Pack) ![32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var path: [std.fs.max_path_bytes]u8 = undefined;
    var chunk: [65536]u8 = undefined;
    for (p.files.items) |name| {
        const file = try Io.Dir.cwd().openFile(io, try std.fmt.bufPrint(&path, "{s}/{s}", .{ p.dir, name }), .{});
        defer file.close(io);
        const stat = try file.stat(io);
        h.update(std.mem.asBytes(&@as(u64, name.len)));
        h.update(name);
        h.update(std.mem.asBytes(&stat.size));
        var off: u64 = 0;
        while (off < stat.size) {
            const want: usize = @intCast(@min(chunk.len, stat.size - off));
            if (try file.readPositionalAll(io, chunk[0..want], off) != want) return error.ShortRead;
            h.update(chunk[0..want]);
            off += want;
        }
        const after = try file.stat(io);
        if (after.size != stat.size or !std.meta.eql(after.mtime, stat.mtime)) return error.CheckpointChanged;
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return digest;
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

fn aligned(n: u64) !u64 {
    return (try std.math.add(u64, n, page - 1)) & ~(page - 1);
}

fn writeOut(gpa: Allocator, io: Io, file: Io.File, b: *const named.Builder) !void {
    var head: std.ArrayList(u8) = .empty;
    defer head.deinit(gpa);
    try head.appendSlice(gpa, magic);
    try head.appendSlice(gpa, std.mem.asBytes(&version));
    try head.appendSlice(gpa, std.mem.asBytes(&@as(u32, @intCast(b.out.items.len))));
    var size: u64 = 16;
    for (b.out.items) |n| {
        if (n.name.len > std.math.maxInt(u16)) return error.BadEntry;
        size = try std.math.add(u64, size, try std.math.add(u64, @sizeOf(Entry), n.name.len));
    }
    var at = try aligned(size);
    for (b.out.items) |n| {
        var e: Entry = .{ .kind = @backingInt(n.kind), .rank = n.rank, .name_len = @intCast(n.name.len), .shape = undefined, .offset = at, .len = n.bytes.len, .sha256 = undefined };
        for (&e.shape, n.shape) |*d, s| d.* = s;
        std.crypto.hash.sha2.Sha256.hash(n.bytes, &e.sha256, .{});
        try head.appendSlice(gpa, std.mem.asBytes(&e));
        try head.appendSlice(gpa, n.name);
        at = try aligned(try std.math.add(u64, at, n.bytes.len));
    }
    try file.writePositionalAll(io, head.items, 0);
    var cursor = try aligned(size);
    for (b.out.items) |n| {
        try file.writePositionalAll(io, n.bytes, cursor);
        cursor = try aligned(try std.math.add(u64, cursor, n.bytes.len));
    }
    try file.setLength(io, cursor);
}

/// Reads every tensor of a file into `b` (each into its own buffer); returns payload bytes. On error `b` gets nothing.
fn readInto(gpa: Allocator, io: Io, file: Io.File, b: *named.Builder, verify: bool, expected: []const named.Named) !u64 {
    var fixed: [16]u8 = undefined;
    if (try file.readPositionalAll(io, &fixed, 0) != 16) return error.Truncated;
    if (!std.mem.eql(u8, fixed[0..8], magic)) return error.BadMagic;
    if (std.mem.readInt(u32, fixed[8..12], .little) != version) return error.BadVersion;
    const count = std.mem.readInt(u32, fixed[12..16], .little);
    const total = try file.length(io);
    if (count != expected.len or count > 1_000_000 or total < 16 or count > (total - 16) / (@sizeOf(Entry) + 1)) return error.BadEntry;
    var got: std.ArrayList(named.Named) = .empty;
    errdefer {
        for (got.items) |n| {
            gpa.free(n.name);
            gpa.free(n.bytes);
        }
        got.deinit(gpa);
    }
    var off: u64 = 16;
    var first_payload: u64 = total;
    var payload_end: u64 = 0;
    var bytes: u64 = 0;
    for (0..count) |_| {
        var e: Entry = undefined;
        if (try file.readPositionalAll(io, std.mem.asBytes(&e), off) != @sizeOf(Entry)) return error.Truncated;
        off = std.math.add(u64, off, @sizeOf(Entry)) catch return error.BadEntry;
        const end = std.math.add(u64, e.offset, e.len) catch return error.BadEntry;
        const name_end = std.math.add(u64, off, e.name_len) catch return error.BadEntry;
        if (e.offset < payload_end) return error.BadEntry;
        first_payload = @min(first_payload, e.offset);
        payload_end = end;
        if (name_end > first_payload) return error.BadEntry;
        if (end > total or e.offset % page != 0 or name_end > e.offset or e.name_len == 0 or
            e.rank == 0 or e.rank > 4 or e.pad != 0 or e.kind > @backingInt(named.Kind.i16)) return error.BadEntry;
        var elements: u64 = 1;
        for (e.shape[0..e.rank]) |d| {
            if (d == 0) return error.BadEntry;
            elements = std.math.mul(u64, elements, d) catch return error.BadEntry;
        }
        for (e.shape[e.rank..]) |d| if (d != 1) return error.BadEntry;
        const item_size: u64 = if (e.kind == @backingInt(named.Kind.f32)) 4 else 2;
        if ((std.math.mul(u64, elements, item_size) catch return error.BadEntry) != e.len) return error.BadEntry;
        const name = try gpa.alloc(u8, e.name_len);
        errdefer gpa.free(name);
        if (try file.readPositionalAll(io, name, off) != name.len) return error.Truncated;
        off = name_end;
        const want = expected[got.items.len];
        if (!std.mem.eql(u8, name, want.name) or e.kind != @backingInt(want.kind) or e.rank != want.rank) return error.BadEntry;
        for (e.shape, want.shape) |stored, planned| if (stored != planned) return error.BadEntry;
        for (got.items) |prior| {
            if (std.mem.eql(u8, name, prior.name)) return error.BadEntry;
        }
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
        bytes = std.math.add(u64, bytes, e.len) catch return error.BadEntry;
        try got.append(gpa, n);
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
    const sizes = [_]usize{ 4096, 5000, 16, 2 };
    for (sizes, 0..) |n, i| {
        const bytes = try gpa.alignedAlloc(u8, .@"16", n);
        for (bytes, 0..) |*x, j| x.* = @truncate(j *% 31 +% i);
        try b.out.append(gpa, .{ .name = try std.fmt.allocPrint(gpa, "L1.t{d}.trellis", .{i}), .kind = @fromBackingInt(@intCast(i % 4)), .shape = .{ n / (if (i % 4 == 0) @as(usize, 4) else 2), 1, 1, 1 }, .rank = 1, .bytes = bytes });
    }
    var miss: named.Builder = .{ .gpa = gpa, .io = io, .pack = &p };
    defer miss.deinit();
    try std.testing.expect(!try c.get(&miss, "L1", b.out.items));
    try c.put(&b, "L1");
    var back: named.Builder = .{ .gpa = gpa, .io = io, .pack = &p };
    defer back.deinit();
    try std.testing.expect(try c.get(&back, "L1", b.out.items));
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
    try std.testing.expect(!try other.get(&none, "L1", b.out.items));
    // a damaged payload is refused with VERIFY and the builder stays empty
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const f = try Io.Dir.cwd().openFile(io, try c.path(&pbuf, "L1", ""), .{ .mode = .read_write });
    try f.writePositionalAll(io, "x", page);
    f.close(io);
    var bad: named.Builder = .{ .gpa = gpa, .io = io, .pack = &p };
    defer bad.deinit();
    try std.testing.expect(!try c.get(&bad, "L1", b.out.items));
    try std.testing.expectEqual(@as(usize, 0), bad.out.items.len);
}

test "real cache separates equal-stat checkpoints by payload and shares identical content" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(io, "a", .default_dir);
    try tmp.dir.createDir(io, "b", .default_dir);
    const image = try pack_mod.image(a, &.{.{ .name = "q", .dtype = "BF16", .shape = &.{1} }});
    defer a.free(image);
    try tmp.dir.writeFile(io, .{ .sub_path = "a/model.safetensors", .data = image });
    image[image.len - 1] ^= 1;
    try tmp.dir.writeFile(io, .{ .sub_path = "b/model.safetensors", .data = image });
    const fa = try tmp.dir.openFile(io, "a/model.safetensors", .{});
    defer fa.close(io);
    const fb = try tmp.dir.openFile(io, "b/model.safetensors", .{ .mode = .read_write });
    defer fb.close(io);
    const stat = try fa.stat(io);
    try fb.setTimestamps(io, .{ .access_timestamp = .unchanged, .modify_timestamp = .{ .new = stat.mtime } });
    try std.testing.expectEqual(stat.size, (try fb.stat(io)).size);
    try std.testing.expectEqual(stat.mtime, (try fb.stat(io)).mtime);
    const dir_a = try tmp.dir.realPathFileAlloc(io, "a", a);
    defer a.free(dir_a);
    const dir_b = try tmp.dir.realPathFileAlloc(io, "b", a);
    defer a.free(dir_b);
    var pa = try pack_mod.Pack.open(a, io, dir_a);
    defer pa.deinit();
    var pb = try pack_mod.Pack.open(a, io, dir_b);
    defer pb.deinit();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const key: Key = .{ .rank = 0, .world = 1, .o_groups = 1, .blocks = &.{}, .dspark = false };
    var ca = try Cache.open(a, io, root, try packDigest(io, &pa), key, true, true);
    defer ca.deinit();
    var cb = try Cache.open(a, io, root, try packDigest(io, &pb), key, true, true);
    defer cb.deinit();
    var original: named.Builder = .{ .gpa = a, .io = io, .pack = &pa };
    defer original.deinit();
    const bytes = try a.alignedAlloc(u8, .@"16", 2);
    @memset(bytes, 1);
    try original.out.append(a, .{ .name = try a.dupe(u8, "q"), .kind = .bf16, .shape = .{ 1, 1, 1, 1 }, .rank = 1, .bytes = bytes });
    try ca.put(&original, "L0");
    var back: named.Builder = .{ .gpa = a, .io = io, .pack = &pb };
    defer back.deinit();
    try std.testing.expect(!try cb.get(&back, "L0", original.out.items));
    try std.testing.expectEqual(@as(usize, 0), back.out.items.len);
    image[image.len - 1] ^= 1;
    try fb.writePositionalAll(io, image, 0);
    var same = try Cache.open(a, io, root, try packDigest(io, &pb), key, false, true);
    defer same.deinit();
    try std.testing.expectEqualStrings(ca.root, same.root);
    try std.testing.expect(try same.get(&back, "L0", original.out.items));
}

test "prepared metadata corruption is a clean miss with no partial publication" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    var p = pack_mod.Pack.init(a);
    defer p.deinit();
    const key: Key = .{ .rank = 0, .world = 1, .o_groups = 1, .blocks = &.{}, .dspark = false };
    var c = try Cache.open(a, io, root, @splat(1), key, true, false);
    defer c.deinit();
    var b: named.Builder = .{ .gpa = a, .io = io, .pack = &p };
    defer b.deinit();
    const bytes = try a.alignedAlloc(u8, .@"16", 2);
    @memset(bytes, 7);
    try b.out.append(a, .{ .name = try a.dupe(u8, "q"), .kind = .bf16, .shape = .{ 1, 1, 1, 1 }, .rank = 1, .bytes = bytes });
    for ([_]bool{ false, true }) |verify| {
        c.verify = verify;
        try c.put(&b, "L0");
        var valid: named.Builder = .{ .gpa = a, .io = io, .pack = &p };
        defer valid.deinit();
        try std.testing.expect(try c.get(&valid, "L0", b.out.items));
        for (0..10) |bad_case| {
            try c.put(&b, "L0");
            var buf: [std.fs.max_path_bytes]u8 = undefined;
            const file = try Io.Dir.cwd().openFile(io, try c.path(&buf, "L0", ""), .{ .mode = .read_write });
            defer file.close(io);
            var entry: Entry = undefined;
            _ = try file.readPositionalAll(io, std.mem.asBytes(&entry), 16);
            switch (bad_case) {
                0 => entry.shape[0] = 5120,
                1 => entry.kind = @backingInt(named.Kind.f32),
                2 => entry.offset = std.math.maxInt(u64),
                3 => entry.rank = 0,
                4 => entry.name_len = std.math.maxInt(u16),
                5 => entry.shape[1] = std.math.maxInt(u64),
                7 => {
                    entry.shape[0] = 2;
                    entry.len = 4;
                },
                8 => entry.kind = @backingInt(named.Kind.f16),
                9 => {
                    entry.shape[0] = std.math.maxInt(u64);
                    entry.shape[1] = 2;
                    entry.rank = 2;
                },
                6 => {
                    var count: u32 = std.math.maxInt(u32);
                    try file.writePositionalAll(io, std.mem.asBytes(&count), 12);
                },
                else => unreachable,
            }
            try file.writePositionalAll(io, std.mem.asBytes(&entry), 16);
            var back: named.Builder = .{ .gpa = a, .io = io, .pack = &p };
            defer back.deinit();
            try std.testing.expect(!try c.get(&back, "L0", b.out.items));
            try std.testing.expectEqual(@as(usize, 0), back.out.items.len);
        }
    }
}
