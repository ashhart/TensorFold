//! Row block and state files: payload, a state's tokens, then a trailer; read past the page cache and checked first.
const std = @import("std");
const platform = @import("sys");
const builtin = @import("builtin");
const Name = @import("rows_chain.zig").Name;

const MAGIC: u32 = 0x42524654; // "TFRB"
const VERSION: u32 = 1;

/// Offsets, lengths and buffers of uncached reads align to this (a Mac's VM page; Linux O_DIRECT needs 4 KiB of it).
pub const alignment = 16384;
pub const Stage = []align(alignment) u8;

pub const Kind = enum(u32) { block = 1, state = 2 };

/// Every file's last 96 bytes: what it is, its sizes and hashes, its name and (a state) its position.
pub const Trailer = extern struct {
    magic: u32 = MAGIC,
    version: u32 = VERSION,
    kind: u32,
    flags: u32 = 0,
    payload: u64,
    payload_hash: u64,
    meta: u64 = 0,
    meta_hash: u64 = 0,
    name: Name,
    at: u32 = 0,
    block: u32 = 0,
    reserved: u64 = 0,
};

comptime {
    std.debug.assert(@sizeOf(Trailer) == 96);
}

extern "c" fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) c_int;

/// A file's path: `dir`/`stem``ext` (and `.part` while it is written).
pub fn path(buf: []u8, dir: []const u8, stem: []const u8, ext: []const u8, part: bool) ![:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "{s}/{s}{s}{s}", .{ dir, stem, ext, if (part) ".part" else "" }, 0);
}

/// One file written front to back through a `.part` name (0600, past the page cache), renamed into place last.
pub const Writer = struct {
    fd: c_int,
    tmp: [1200]u8 = undefined,
    tmp_len: usize = 0,
    hash: std.hash.Wyhash = .init(0),
    payload: u64 = 0,

    pub fn open(dir: []const u8, stem: []const u8, ext: []const u8) !Writer {
        var w: Writer = .{ .fd = -1 };
        const t = try path(&w.tmp, dir, stem, ext, true);
        w.tmp_len = t.len;
        w.fd = platform.c.open(t, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0o600));
        if (w.fd < 0) return error.RowsWrite;
        if (builtin.os.tag == .macos) _ = platform.c.fcntl(w.fd, platform.c.F.NOCACHE, @as(c_int, 1));
        return w;
    }

    fn tmpPath(w: *const Writer) [:0]const u8 {
        return w.tmp[0..w.tmp_len :0];
    }

    /// Payload bytes, appended.
    pub fn append(w: *Writer, bytes: []const u8) !void {
        w.hash.update(bytes);
        try put(w.fd, bytes);
        w.payload += bytes.len;
    }

    /// The metadata pieces and the trailer, then (a state) fsync, then the rename that makes it a file of the store.
    pub fn finish(w: *Writer, t: Trailer, pieces: []const []const u8, sync: bool) !u64 {
        var trailer = t;
        trailer.payload = w.payload;
        trailer.payload_hash = w.hash.final();
        var mh = std.hash.Wyhash.init(1);
        for (pieces) |m| {
            mh.update(m);
            try put(w.fd, m);
            trailer.meta += m.len;
        }
        trailer.meta_hash = mh.final();
        try put(w.fd, std.mem.asBytes(&trailer));
        if (sync and platform.c.fsync(w.fd) != 0) return error.RowsWrite;
        if (builtin.os.tag == .linux) {
            _ = platform.c.fdatasync(w.fd);
            _ = posix_fadvise(w.fd, 0, 0, 4); // POSIX_FADV_DONTNEED: written pages leave the cache
        }
        _ = platform.c.close(w.fd);
        w.fd = -1;
        const t_path = w.tmpPath();
        var dest: [1200]u8 = undefined;
        const d = try std.fmt.bufPrintSentinel(&dest, "{s}", .{t_path[0 .. t_path.len - ".part".len]}, 0);
        if (platform.c.rename(t_path, d) != 0) {
            _ = platform.c.unlink(t_path);
            return error.RowsWrite;
        }
        return trailer.payload + trailer.meta + @sizeOf(Trailer);
    }

    /// A write given up: the `.part` file goes.
    pub fn abort(w: *Writer) void {
        if (w.fd >= 0) _ = platform.c.close(w.fd);
        w.fd = -1;
        _ = platform.c.unlink(w.tmpPath());
    }
};

fn put(fd: c_int, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = platform.c.write(fd, bytes.ptr + done, @min(bytes.len - done, 1 << 30));
        if (n <= 0) {
            if (n < 0 and platform.c.errno(n) == .INTR) continue;
            return error.RowsWrite;
        }
        done += @intCast(n);
    }
}

fn get(fd: c_int, dest: []u8, at: u64) bool {
    var done: usize = 0;
    while (done < dest.len) {
        const n = platform.c.pread(fd, dest.ptr + done, @min(dest.len - done, 1 << 30), @intCast(at + done));
        if (n < 0 and platform.c.errno(n) == .INTR) continue;
        if (n <= 0) return false;
        done += @intCast(n);
    }
    return true;
}

/// An open file of the store and its checked trailer.
pub const Opened = struct {
    fd: c_int,
    trailer: Trailer,

    pub fn close(o: *Opened) void {
        _ = platform.c.close(o.fd);
        o.fd = -1;
    }
};

/// `file` opened past the page cache, its trailer read and checked against its size, kind and (when given) its name.
pub fn open(file: [:0]const u8, kind: Kind, name: ?Name) !Opened {
    var fd: c_int = -1;
    if (builtin.os.tag == .linux) fd = platform.c.open(file, .{ .ACCMODE = .RDONLY, .CLOEXEC = true, .NOFOLLOW = true, .DIRECT = true }, @as(platform.c.mode_t, 0));
    if (fd < 0) fd = platform.c.open(file, .{ .ACCMODE = .RDONLY, .CLOEXEC = true, .NOFOLLOW = true }, @as(platform.c.mode_t, 0));
    if (fd < 0) return error.RowsMissing;
    errdefer _ = platform.c.close(fd);
    if (builtin.os.tag == .macos) _ = platform.c.fcntl(fd, platform.c.F.NOCACHE, @as(c_int, 1));
    const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
    if (end < @sizeOf(Trailer)) return error.RowsRead;
    var t: Trailer = undefined;
    var raw: [@sizeOf(Trailer) + 2 * 4096]u8 align(4096) = undefined; // O_DIRECT: an aligned read around the trailer
    const at: u64 = @as(u64, @intCast(end)) - @sizeOf(Trailer);
    const lo = std.mem.alignBackward(u64, at, 4096);
    const span = std.mem.alignForward(u64, @as(u64, @intCast(end)) - lo, 4096);
    const n = platform.c.pread(fd, &raw, @intCast(span), @intCast(lo));
    if (n < @as(isize, @intCast(@as(u64, @intCast(end)) - lo))) return error.RowsRead;
    @memcpy(std.mem.asBytes(&t), raw[@intCast(at - lo)..][0..@sizeOf(Trailer)]);
    if (t.magic != MAGIC or t.version != VERSION or t.kind != @backingInt(kind)) return error.RowsRead;
    if (t.payload + t.meta + @sizeOf(Trailer) != @as(u64, @intCast(end))) return error.RowsRead;
    if (name) |want| if (!std.mem.eql(u8, &want, &t.name)) return error.RowsRead;
    return .{ .fd = fd, .trailer = t };
}

/// The payload into `into` (aligned reads from offset 0), checked against its hash; the bytes are its first `payload`.
pub fn payload(o: *const Opened, into: Stage) ![]u8 {
    const n = o.trailer.payload;
    const span = std.mem.alignForward(u64, n, alignment);
    if (span > into.len) return error.RowsStage;
    var done: u64 = 0;
    while (done < n) { // past the payload the read may run into the metadata or come back short at the file's end
        const from = std.mem.alignBackward(u64, done, 4096); // a short O_DIRECT read resumes on an aligned boundary
        const r = platform.c.pread(o.fd, into.ptr + from, @intCast(@min(span - from, 1 << 30)), @intCast(from));
        if (r < 0 and platform.c.errno(r) == .INTR) continue;
        if (r <= 0) return error.RowsRead;
        const next = from + @as(u64, @intCast(r));
        if (next <= done) return error.RowsRead;
        done = next;
    }
    if (std.hash.Wyhash.hash(0, into[0..@intCast(n)]) != o.trailer.payload_hash) return error.RowsHash;
    return into[0..@intCast(n)];
}

/// The metadata (a state's tokens, spans and starts) into `a`'s memory, checked against its hash.
pub fn meta(o: *const Opened, a: std.mem.Allocator) ![]u8 {
    const m = try a.alloc(u8, @intCast(o.trailer.meta));
    errdefer a.free(m);
    if (!readUnaligned(o.fd, m, o.trailer.payload)) return error.RowsRead;
    var mh = std.hash.Wyhash.init(1);
    mh.update(m);
    if (mh.final() != o.trailer.meta_hash) return error.RowsHash;
    return m;
}

/// `dest.len` bytes at `at` of `fd`, through an aligned bounce buffer (O_DIRECT allows no other).
fn readUnaligned(fd: c_int, dest: []u8, at: u64) bool {
    var bounce: [64 * 1024]u8 align(4096) = undefined;
    var done: usize = 0;
    while (done < dest.len) {
        const pos = at + done;
        const lo = std.mem.alignBackward(u64, pos, 4096);
        const skip: usize = @intCast(pos - lo);
        const want = @min(dest.len - done, bounce.len - skip);
        const span = std.mem.alignForward(usize, skip + want, 4096);
        var got: usize = 0;
        while (got < skip + want) {
            const n = platform.c.pread(fd, bounce[got..].ptr, span - got, @intCast(lo + got));
            if (n < 0 and platform.c.errno(n) == .INTR) continue;
            if (n <= 0) return false;
            got += @intCast(n);
        }
        @memcpy(dest[done..][0..want], bounce[skip..][0..want]);
        done += want;
    }
    return true;
}

/// A staging buffer that holds `bytes` read in aligned spans.
pub fn stage(a: std.mem.Allocator, bytes: u64) !Stage {
    return a.alignedAlloc(u8, .fromByteUnits(alignment), @intCast(std.mem.alignForward(u64, @max(bytes, 1), alignment)));
}

test "a block file reads back byte for byte, and a flipped byte, a cut tail or another name is refused" {
    const a = std.testing.allocator;
    var dir_buf: [64]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&dir_buf, "/tmp/tf-rows-file-{d}", .{platform.c.getpid()}, 0);
    _ = platform.c.mkdir(dir, 0o700);
    defer _ = platform.c.rmdir(dir);
    const body = try a.alloc(u8, 70_000);
    defer a.free(body);
    for (body, 0..) |*b, i| b.* = @truncate(i *% 31 +% 7);
    const name: Name = @splat(5);
    var w = try Writer.open(dir, "b", ".blk");
    try w.append(body[0..1000]);
    try w.append(body[1000..]);
    const bytes = try w.finish(.{ .kind = @backingInt(Kind.block), .payload = 0, .payload_hash = 0, .name = name }, &.{}, false);
    try std.testing.expectEqual(@as(u64, body.len + 96), bytes);
    var buf: [128]u8 = undefined;
    const file = try path(&buf, dir, "b", ".blk", false);
    defer _ = platform.c.unlink(file);
    const s = try stage(a, body.len);
    defer a.free(s);
    {
        var o = try open(file, .block, name);
        defer o.close();
        try std.testing.expectEqualSlices(u8, body, try payload(&o, s));
    }
    try std.testing.expectError(error.RowsRead, open(file, .block, @splat(6)));
    try std.testing.expectError(error.RowsRead, open(file, .state, null));
    const fd = platform.c.open(file, .{ .ACCMODE = .RDWR }, @as(platform.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    defer _ = platform.c.close(fd);
    try std.testing.expectEqual(@as(isize, 1), platform.c.pwrite(fd, "X", 1, 12345));
    {
        var o = try open(file, .block, name);
        defer o.close();
        try std.testing.expectError(error.RowsHash, payload(&o, s));
    }
    try std.testing.expectEqual(@as(c_int, 0), platform.c.ftruncate(fd, 5000));
    try std.testing.expectError(error.RowsRead, open(file, .block, name));
}

test "a state file's metadata reads back checked, and an aborted write leaves nothing" {
    const a = std.testing.allocator;
    var dir_buf: [64]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&dir_buf, "/tmp/tf-rows-meta-{d}", .{platform.c.getpid()}, 0);
    _ = platform.c.mkdir(dir, 0o700);
    defer _ = platform.c.rmdir(dir);
    const tokens = [_]u32{ 1, 2, 3, 4, 5 };
    var w = try Writer.open(dir, "s", ".st");
    try w.append("own bytes");
    _ = try w.finish(.{ .kind = @backingInt(Kind.state), .payload = 0, .payload_hash = 0, .name = @splat(1), .at = 4 }, &.{std.mem.sliceAsBytes(&tokens)}, true);
    var buf: [128]u8 = undefined;
    const file = try path(&buf, dir, "s", ".st", false);
    defer _ = platform.c.unlink(file);
    var o = try open(file, .state, null);
    defer o.close();
    try std.testing.expectEqual(@as(u32, 4), o.trailer.at);
    const m = try meta(&o, a);
    defer a.free(m);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&tokens), m);
    var gone = try Writer.open(dir, "g", ".st");
    try gone.append("half");
    gone.abort();
    var part: [128]u8 = undefined;
    try std.testing.expectError(error.RowsMissing, open(try path(&part, dir, "g", ".st", true), .state, null));
}
