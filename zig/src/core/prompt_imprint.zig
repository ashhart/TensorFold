//! Prompt-cache states on disk (Imprint): a learned harness state outlives the server and serves later fresh sessions.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// One state on disk: the tokens it read (position plus lookahead), the chunk starts below it, and its bytes.
pub const Meta = struct { key: u64, at: u32, tokens: []u32, starts: []u32, bytes: u64, used: u64 = 0 };

const MAGIC: u32 = 0x494d5052; // "IMPR"
const VERSION: u32 = 2;
const HEAD = 9; // a record's words before its tokens: magic, version, key (2), at, token and start counts, bytes (2)

pub const Imprint = struct {
    gpa: Allocator,
    root: []u8,
    dir: [:0]u8, // root/<identity>: the index, the last-open stamp and each state's files (the family's)
    cap: u64, // bytes of learned states this Mac keeps under `root`, every identity together
    others: u64 = 0, // bytes other identities under `root` still hold
    metas: std.ArrayList(Meta) = .empty,
    clock: u64 = 0, // a meta's `used`: the clock at its last write or read

    /// The states learned under `identity` in `root` (made when missing); other identities past `cap` go, oldest first.
    pub fn open(gpa: Allocator, root: []const u8, identity: u64, cap: u64) !Imprint {
        const r = try gpa.dupe(u8, root);
        errdefer gpa.free(r);
        const dir = try std.fmt.allocPrintSentinel(gpa, "{s}/{x:0>16}", .{ root, identity }, 0);
        errdefer gpa.free(dir);
        try makePath(gpa, dir);
        var m: Imprint = .{ .gpa = gpa, .root = r, .dir = dir, .cap = cap };
        errdefer m.deinit();
        try m.load();
        m.stamp();
        m.others = sweep(gpa, root, dir, cap -| @max(m.total(), files(dir, false))); // a peer's halves have no index
        return m;
    }

    pub fn deinit(m: *Imprint) void {
        for (m.metas.items) |x| free(m.gpa, x);
        m.metas.deinit(m.gpa);
        m.gpa.free(m.dir);
        m.gpa.free(m.root);
    }

    fn free(gpa: Allocator, x: Meta) void {
        gpa.free(x.tokens);
        gpa.free(x.starts);
    }

    pub fn keyOf(tokens: []const u32) u64 {
        return std.hash.Wyhash.hash(0x1e47, std.mem.sliceAsBytes(tokens));
    }

    fn find(m: *const Imprint, key: u64) ?usize {
        for (m.metas.items, 0..) |x, i| if (x.key == key) return i;
        return null;
    }

    pub fn has(m: *const Imprint, key: u64) bool {
        return m.find(key) != null;
    }

    /// The bytes of every state learned under this identity.
    pub fn total(m: *const Imprint) u64 {
        var n: u64 = 0;
        for (m.metas.items) |x| n += x.bytes;
        return n;
    }

    /// Whether `bytes` more fit under the cap; other identities give room first, least recently opened first.
    pub fn fits(m: *Imprint, bytes: u64) bool {
        if (m.others + m.total() + bytes > m.cap) m.others = sweep(m.gpa, m.root, m.dir, m.cap -| (m.total() + bytes));
        return m.others + m.total() + bytes <= m.cap;
    }

    /// The least recently used state learned here: the next to go when the cap needs room.
    pub fn victim(m: *const Imprint) ?u64 {
        var out: ?*const Meta = null;
        for (m.metas.items) |*x| if (out == null or x.used < out.?.used) {
            out = x;
        };
        return if (out) |x| x.key else null;
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

    /// State `key` was read: the most recently used now.
    pub fn touch(m: *Imprint, key: u64) void {
        const i = m.find(key) orelse return;
        m.clock += 1;
        m.metas.items[i].used = m.clock;
    }

    /// Record a state its family has written (`bytes` on this Mac): an index record, fsynced, then the entry.
    pub fn add(m: *Imprint, key: u64, at: u32, tokens: []const u32, starts: []const u32, bytes: u64) !void {
        if (m.has(key)) return;
        const below = cut(starts, at);
        var path: [1100]u8 = undefined;
        const fd = std.c.open(try std.fmt.bufPrintSentinel(&path, "{s}/index", .{m.dir}, 0), .{ .ACCMODE = .WRONLY, .APPEND = true, .CREAT = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return error.ImprintWrite;
        defer _ = std.c.close(fd);
        try record(fd, key, at, tokens, below, bytes);
        if (std.c.fsync(fd) != 0) return error.ImprintWrite;
        const t = try m.gpa.dupe(u32, tokens);
        errdefer m.gpa.free(t);
        const s = try m.gpa.dupe(u32, below);
        errdefer m.gpa.free(s);
        m.clock += 1;
        try m.metas.append(m.gpa, .{ .key = key, .at = at, .tokens = t, .starts = s, .bytes = bytes, .used = m.clock });
    }

    /// Forget state `key`: the index is rewritten without it (its family removes the files).
    pub fn remove(m: *Imprint, key: u64) !void {
        const i = m.find(key) orelse return;
        free(m.gpa, m.metas.orderedRemove(i));
        var path: [1100]u8 = undefined;
        var part: [1100]u8 = undefined;
        const tmp = try std.fmt.bufPrintSentinel(&part, "{s}/index.part", .{m.dir}, 0);
        const fd = std.c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return error.ImprintWrite;
        {
            defer _ = std.c.close(fd);
            for (m.metas.items) |x| try record(fd, x.key, x.at, x.tokens, x.starts, x.bytes);
            if (std.c.fsync(fd) != 0) return error.ImprintWrite;
        }
        if (std.c.rename(tmp, try std.fmt.bufPrintSentinel(&path, "{s}/index", .{m.dir}, 0)) != 0) return error.ImprintWrite;
    }

    /// The index's records, oldest first; a torn last record (a crash mid-append) ends the read.
    fn load(m: *Imprint) !void {
        var path: [1100]u8 = undefined;
        const fd = std.c.open(try std.fmt.bufPrintSentinel(&path, "{s}/index", .{m.dir}, 0), .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return; // nothing learned yet
        defer _ = std.c.close(fd);
        var off: u64 = 0;
        while (true) {
            var head: [HEAD]u32 = undefined;
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
            const x: Meta = .{ .key = word64(head[2..4]), .at = head[4], .tokens = t, .starts = s, .bytes = word64(head[7..9]) };
            if (!readAt(fd, std.mem.sliceAsBytes(t), at) or !readAt(fd, std.mem.sliceAsBytes(s), at + 4 * n)) return free(m.gpa, x);
            if (m.has(x.key)) free(m.gpa, x) else {
                m.clock += 1;
                var y = x;
                y.used = m.clock;
                m.metas.append(m.gpa, y) catch |err| {
                    free(m.gpa, x);
                    return err;
                };
            }
            off = at + 4 * (n + k);
        }
    }

    /// This identity was opened: the newest under `root` for the sweep (root/clock counts opens).
    fn stamp(m: *const Imprint) void {
        var path: [1100]u8 = undefined;
        const clock = std.fmt.bufPrintSentinel(&path, "{s}/clock", .{m.root}, 0) catch return;
        const now = readWord(clock) + 1;
        writeWord(clock, now);
        var used: [1100]u8 = undefined;
        writeWord(std.fmt.bufPrintSentinel(&used, "{s}/used", .{m.dir}, 0) catch return, now);
    }
};

/// One index record: its head, its tokens, then the chunk starts below its position.
fn record(fd: c_int, key: u64, at: u32, tokens: []const u32, starts: []const u32, bytes: u64) !void {
    const head = [HEAD]u32{ MAGIC, VERSION, @truncate(key), @truncate(key >> 32), at, @intCast(tokens.len), @intCast(starts.len), @truncate(bytes), @truncate(bytes >> 32) };
    try writeAll(fd, std.mem.sliceAsBytes(&head));
    try writeAll(fd, std.mem.sliceAsBytes(tokens));
    try writeAll(fd, std.mem.sliceAsBytes(starts));
}

fn word64(w: []const u32) u64 {
    return @as(u64, w[0]) | @as(u64, w[1]) << 32;
}

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

/// An identity directory's name: 16 hex digits.
fn identityName(name: []const u8) bool {
    if (name.len != 16) return false;
    for (name) |ch| if (!std.ascii.isHex(ch)) return false;
    return true;
}

/// Other identities' directories (16 hex digits, their files only) deleted, oldest first, down to `room` bytes.
fn sweep(gpa: Allocator, root: []const u8, keep: []const u8, room: u64) u64 {
    const Dir = struct { name: [16]u8, bytes: u64, used: u64 };
    var dirs: std.ArrayList(Dir) = .empty;
    defer dirs.deinit(gpa);
    var buf: [1100]u8 = undefined;
    const d = std.c.opendir(std.fmt.bufPrintSentinel(&buf, "{s}", .{root}, 0) catch return 0) orelse return 0;
    var held: u64 = 0;
    while (std.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (!identityName(name) or std.mem.endsWith(u8, keep, name)) continue;
        var sub: [1100]u8 = undefined;
        const path = std.fmt.bufPrint(&sub, "{s}/{s}", .{ root, name }) catch continue;
        const used = readWord(std.fmt.bufPrintSentinel(&buf, "{s}/used", .{path}, 0) catch continue);
        const x: Dir = .{ .name = name[0..16].*, .bytes = files(path, false), .used = used };
        held += x.bytes;
        dirs.append(gpa, x) catch break;
    }
    _ = std.c.closedir(d);
    std.mem.sort(Dir, dirs.items, {}, struct {
        fn older(_: void, a: Dir, b: Dir) bool {
            return a.used < b.used;
        }
    }.older);
    for (dirs.items) |x| {
        if (held <= room) break;
        var sub: [1100]u8 = undefined;
        const path = std.fmt.bufPrintSentinel(&sub, "{s}/{s}", .{ root, &x.name }, 0) catch continue;
        _ = files(path, true);
        if (std.c.rmdir(path) == 0) held -|= x.bytes;
    }
    return held;
}

/// The bytes of the regular files directly in `dir`, each unlinked when `remove`.
fn files(dir: []const u8, remove: bool) u64 {
    var buf: [1100]u8 = undefined;
    const d = std.c.opendir(std.fmt.bufPrintSentinel(&buf, "{s}", .{dir}, 0) catch return 0) orelse return 0;
    defer _ = std.c.closedir(d);
    var n: u64 = 0;
    while (std.c.readdir(d)) |ent| {
        if (ent.type != std.c.DT.REG) continue;
        var p: [1400]u8 = undefined;
        const f = std.fmt.bufPrintSentinel(&p, "{s}/{s}", .{ dir, std.mem.sliceTo(&ent.name, 0) }, 0) catch continue;
        const fd = std.c.open(f, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) continue;
        const end = std.c.lseek(fd, 0, std.c.SEEK.END);
        _ = std.c.close(fd);
        if (end > 0) n += @intCast(end);
        if (remove) _ = std.c.unlink(f);
    }
    return n;
}

fn readWord(path: [:0]const u8) u64 {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return 0;
    defer _ = std.c.close(fd);
    var w: u64 = 0;
    return if (readAt(fd, std.mem.asBytes(&w), 0)) w else 0;
}

fn writeWord(path: [:0]const u8, w: u64) void {
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return;
    defer _ = std.c.close(fd);
    writeAll(fd, std.mem.asBytes(&w)) catch {};
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

/// This Mac's OS build (sysctl kern.osversion): its Metal compiler can change a kernel's bits.
pub fn osBuild(buf: []u8) []const u8 {
    var n: usize = buf.len;
    if (std.c.sysctlbyname("kern.osversion", buf.ptr, &n, null, 0) != 0) return "";
    return std.mem.sliceTo(buf[0..n], 0);
}

/// Where learned states live unless --learn-dir says: $HOME/.cache/tensorfold/learned.
pub fn defaultRoot(a: Allocator) ![]const u8 {
    const home = std.c.getenv("HOME") orelse return error.ImprintDir;
    return std.fmt.allocPrint(a, "{s}/.cache/tensorfold/learned", .{std.mem.span(home)});
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

test {
    _ = @import("prompt_imprint_test.zig");
}
