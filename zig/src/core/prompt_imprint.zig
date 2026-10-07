//! Prompt-cache states on disk (Imprint): a learned harness state outlives the server and serves later fresh sessions.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// One state on disk: the tokens it read (its position plus the family's lookahead) and the chunk starts below it.
pub const Meta = struct { key: u64, at: u32, tokens: []u32, starts: []u32 };

const MAGIC: u32 = 0x494d5052; // "IMPR"
const VERSION: u32 = 1;

pub const Imprint = struct {
    gpa: Allocator,
    dir: [:0]u8, // root/<identity>: the index, and each state's files (the family's)
    metas: std.ArrayList(Meta) = .empty,

    /// The states learned under `identity` (model, build, layout, rules) in `root`, which is made when missing.
    pub fn open(gpa: Allocator, root: []const u8, identity: u64) !Imprint {
        const dir = try std.fmt.allocPrintSentinel(gpa, "{s}/{x:0>16}", .{ root, identity }, 0);
        errdefer gpa.free(dir);
        try makePath(gpa, dir);
        var m: Imprint = .{ .gpa = gpa, .dir = dir };
        errdefer m.deinit();
        try m.load();
        return m;
    }

    pub fn deinit(m: *Imprint) void {
        for (m.metas.items) |x| free(m.gpa, x);
        m.metas.deinit(m.gpa);
        m.gpa.free(m.dir);
    }

    fn free(gpa: Allocator, x: Meta) void {
        gpa.free(x.tokens);
        gpa.free(x.starts);
    }

    pub fn keyOf(tokens: []const u32) u64 {
        return std.hash.Wyhash.hash(0x1e47, std.mem.sliceAsBytes(tokens));
    }

    pub fn has(m: *const Imprint, key: u64) bool {
        for (m.metas.items) |x| if (x.key == key) return true;
        return false;
    }

    /// The longest learned state `prompt` resumes: its tokens a prefix, a row left, `starts` below it the prompt's own.
    pub fn best(m: *const Imprint, prompt: []const u32, starts: []const u32, planned: bool, longer_than: u32) ?*const Meta {
        var out: ?*const Meta = null;
        for (m.metas.items) |*x| {
            if (x.at <= longer_than or x.tokens.len > prompt.len or x.at >= prompt.len) continue;
            if (out != null and x.at <= out.?.at) continue;
            if (!std.mem.eql(u32, prompt[0..x.tokens.len], x.tokens)) continue;
            if (planned and !sameBelow(starts, x.starts, x.at)) continue;
            out = x;
        }
        return out;
    }

    /// Record a state its family has written: one index record (fsynced), then the in-memory entry.
    pub fn add(m: *Imprint, key: u64, at: u32, tokens: []const u32, starts: []const u32) !void {
        if (m.has(key)) return;
        const below = cut(starts, at);
        var head = [_]u32{ MAGIC, VERSION, @truncate(key), @truncate(key >> 32), at, @intCast(tokens.len), @intCast(below.len) };
        var path: [1024]u8 = undefined;
        const fd = std.c.open(try std.fmt.bufPrintSentinel(&path, "{s}/index", .{m.dir}, 0), .{ .ACCMODE = .WRONLY, .APPEND = true, .CREAT = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return error.ImprintWrite;
        defer _ = std.c.close(fd);
        try writeAll(fd, std.mem.sliceAsBytes(&head));
        try writeAll(fd, std.mem.sliceAsBytes(tokens));
        try writeAll(fd, std.mem.sliceAsBytes(below));
        if (std.c.fsync(fd) != 0) return error.ImprintWrite;
        const t = try m.gpa.dupe(u32, tokens);
        errdefer m.gpa.free(t);
        const s = try m.gpa.dupe(u32, below);
        errdefer m.gpa.free(s);
        try m.metas.append(m.gpa, .{ .key = key, .at = at, .tokens = t, .starts = s });
    }

    /// The index's records; a torn last record (a crash mid-append) ends the read.
    fn load(m: *Imprint) !void {
        var path: [1024]u8 = undefined;
        const fd = std.c.open(try std.fmt.bufPrintSentinel(&path, "{s}/index", .{m.dir}, 0), .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return; // nothing learned yet
        defer _ = std.c.close(fd);
        var off: u64 = 0;
        while (true) {
            var head: [7]u32 = undefined;
            if (!readAt(fd, std.mem.sliceAsBytes(&head), off) or head[0] != MAGIC or head[1] != VERSION) return;
            const n: usize = head[5];
            const k: usize = head[6];
            if (n > 1 << 22 or k > n) return;
            const t = try m.gpa.alloc(u32, n);
            const s = m.gpa.alloc(u32, k) catch |err| {
                m.gpa.free(t);
                return err;
            };
            const at = off + @sizeOf(@TypeOf(head));
            if (!readAt(fd, std.mem.sliceAsBytes(t), at) or !readAt(fd, std.mem.sliceAsBytes(s), at + 4 * n)) {
                free(m.gpa, .{ .key = 0, .at = 0, .tokens = t, .starts = s });
                return;
            }
            const key = @as(u64, head[2]) | @as(u64, head[3]) << 32;
            if (m.has(key)) free(m.gpa, .{ .key = 0, .at = 0, .tokens = t, .starts = s }) else try m.metas.append(m.gpa, .{ .key = key, .at = head[4], .tokens = t, .starts = s });
            off = at + 4 * (n + k);
        }
    }
};

/// The chunk starts below `at`.
fn cut(starts: []const u32, at: u32) []const u32 {
    var n: usize = 0;
    while (n < starts.len and starts[n] < at) n += 1;
    return starts[0..n];
}

/// Whether a prompt's chunk starts below `at` are the learned pass's (a planned family's state depends on them).
fn sameBelow(starts: []const u32, learned: []const u32, at: u32) bool {
    return std.mem.eql(u32, cut(starts, at), learned) and std.mem.indexOfScalar(u32, starts, at) != null;
}

/// This binary's bytes, hashed: a learned state serves only the build (its kernels and arithmetic) that made it.
pub fn buildHash(io: std.Io) !u64 {
    var buf: [std.Io.Dir.max_path_bytes + 1]u8 = undefined;
    const n = try std.process.executablePath(io, buf[0 .. buf.len - 1]);
    buf[n] = 0;
    const fd = std.c.open(buf[0..n :0], .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.ImprintBuild;
    defer _ = std.c.close(fd);
    var h = std.hash.Wyhash.init(0xb1d);
    var chunk: [1 << 16]u8 = undefined;
    while (true) {
        const got = std.c.read(fd, &chunk, chunk.len);
        if (got < 0) return error.ImprintBuild;
        if (got == 0) return h.final();
        h.update(chunk[0..@intCast(got)]);
    }
}

/// Where learned states live unless --learn-dir says: $HOME/.cache/tensorfold/learned.
pub fn defaultRoot(a: Allocator) ![]const u8 {
    const home = std.c.getenv("HOME") orelse return error.ImprintDir;
    return std.fmt.allocPrint(a, "{s}/.cache/tensorfold/learned", .{std.mem.span(home)});
}

pub fn writeAll(fd: c_int, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.write(fd, bytes.ptr + done, bytes.len - done);
        if (n <= 0) return error.ImprintWrite;
        done += @intCast(n);
    }
}

pub fn readAt(fd: c_int, dest: []u8, at: u64) bool {
    var done: usize = 0;
    while (done < dest.len) {
        const n = std.c.pread(fd, dest.ptr + done, dest.len - done, @intCast(at + done));
        if (n <= 0) return false;
        done += @intCast(n);
    }
    return true;
}

/// `dir` and its parents (mode 0700).
fn makePath(gpa: Allocator, dir: [:0]const u8) !void {
    const tmp = try gpa.dupeSentinel(u8, dir, 0);
    defer gpa.free(tmp);
    for (tmp[1..], 1..) |ch, i| if (ch == '/') {
        tmp[i] = 0;
        _ = std.c.mkdir(tmp.ptr, 0o700);
        tmp[i] = '/';
    };
    const rc = std.c.mkdir(tmp.ptr, 0o700);
    if (rc != 0 and std.c.errno(rc) != .EXIST) return error.ImprintDir;
}

test "learned states survive a reopen; the longest usable prefix wins" {
    const gpa = std.testing.allocator;
    var tmp_buf: [256]u8 = undefined;
    const root = try std.fmt.bufPrint(&tmp_buf, "/tmp/tf-imprint-test-{d}", .{std.c.getpid()});
    defer for ([_][]const u8{ "/0000000000000007/index", "/0000000000000007", "/0000000000000008", "" }) |tail| {
        var p: [300]u8 = undefined;
        const z = std.fmt.bufPrintSentinel(&p, "{s}{s}", .{ root, tail }, 0) catch continue;
        if (std.c.unlink(z) != 0) _ = std.c.rmdir(z);
    };
    {
        var m = try Imprint.open(gpa, root, 7);
        defer m.deinit();
        try m.add(Imprint.keyOf(&.{ 1, 2, 3 }), 2, &.{ 1, 2, 3 }, &.{ 1, 2, 5 });
        try m.add(Imprint.keyOf(&.{ 1, 2, 3, 4, 5 }), 4, &.{ 1, 2, 3, 4, 5 }, &.{ 1, 4 });
        try std.testing.expect(m.has(Imprint.keyOf(&.{ 1, 2, 3 })));
    }
    var m = try Imprint.open(gpa, root, 7);
    defer m.deinit();
    try std.testing.expectEqual(@as(usize, 2), m.metas.items.len);
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7 };
    try std.testing.expectEqual(@as(u32, 4), m.best(&prompt, &.{ 1, 4, 6 }, true, 0).?.at);
    try std.testing.expectEqual(@as(u32, 2), m.best(&prompt, &.{ 1, 2, 6 }, true, 0).?.at); // other starts below 4: only 2 fits
    try std.testing.expect(m.best(&prompt, &.{ 1, 4, 6 }, true, 4) == null); // memory already holds 4
    try std.testing.expect(m.best(&.{ 1, 2, 9, 9 }, &.{ 1, 2 }, true, 0) == null); // token 2 differs: 3 was read
    var other = try Imprint.open(gpa, root, 8); // another identity sees nothing
    defer other.deinit();
    try std.testing.expectEqual(@as(usize, 0), other.metas.items.len);
}
