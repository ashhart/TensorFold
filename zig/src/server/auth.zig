//! API keys held as SHA-256 digests with labels; a restricted key file reread on change or SIGHUP.
const std = @import("std");
const posix = std.posix;
const builtin = @import("builtin");
const log = @import("log.zig");
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

const Key = struct { digest: [32]u8, label: []const u8 };

pub var reload_requested: std.atomic.Value(bool) = .init(false);

fn onHangup(_: posix.SIG) callconv(.c) void {
    reload_requested.store(true, .release);
}

/// Hears SIGHUP as "reread the key file".
pub fn installHangup() void {
    const action: posix.Sigaction = .{ .handler = .{ .handler = onHangup }, .mask = posix.sigemptyset(), .flags = 0 };
    posix.sigaction(.HUP, &action, null);
}

pub const Store = struct {
    gpa: Allocator,
    static: []Key,
    path: ?[]const u8,
    metrics_open: bool,
    mutex: std.Io.Mutex = .init,
    file: []Key = &.{},
    file_arena: ?std.heap.ArenaAllocator = null,
    valid: bool = true,
    stamp: [5]i128 = .{ 0, 0, 0, 0, 0 },
    checked_ns: ?i96 = null,

    /// ``keys`` from --api-key, ``environment`` from TENSORFOLD_API_KEY; errors carry Python's message.
    pub fn init(gpa: Allocator, keys: []const []const u8, environment: []const u8, path: ?[]const u8, metrics_open: bool, problem: *[]const u8) !Store {
        var static: std.ArrayList(Key) = .empty;
        for (keys, 1..) |key, i| {
            const d = digest(key) orelse return fail(problem, "API keys must be nonempty text without whitespace");
            try static.append(gpa, .{ .digest = d, .label = try std.fmt.allocPrint(gpa, "cli-{d}", .{i}) });
        }
        var it = std.mem.splitScalar(u8, environment, ',');
        var i: usize = 0;
        while (it.next()) |raw| {
            i += 1;
            const key = std.mem.trim(u8, raw, " \t\r\n\x0b\x0c");
            if (key.len == 0) continue;
            const d = digest(key) orelse return fail(problem, "API keys must be nonempty text without whitespace");
            try static.append(gpa, .{ .digest = d, .label = try std.fmt.allocPrint(gpa, "env-{d}", .{i}) });
        }
        var store: Store = .{ .gpa = gpa, .static = static.items, .path = path, .metrics_open = metrics_open };
        if (path != null) {
            store.readFile(problem) catch |e| switch (e) {
                error.Unsafe => {
                    if (problem.len == 0) problem.* = "API key file cannot be opened safely";
                    return error.KeyFile;
                },
                error.Invalid => return error.KeyFile,
                error.OutOfMemory => return error.OutOfMemory,
            };
        }
        return store;
    }

    fn fail(problem: *[]const u8, message: []const u8) error{KeyFile} {
        problem.* = message;
        return error.KeyFile;
    }

    pub fn enabled(s: *const Store) bool {
        return s.static.len > 0 or s.path != null;
    }

    /// Reads the key file, replacing the file's keys; Unsafe for an OS error, Invalid with ``problem`` set.
    fn readFile(s: *Store, problem: *[]const u8) error{ Unsafe, Invalid, OutOfMemory }!void {
        const path = s.path.?;
        const pathz = try s.gpa.dupeSentinel(u8, path, 0);
        defer s.gpa.free(pathz);
        const fd = posix.openatZ(posix.AT.FDCWD, pathz, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .NONBLOCK = true }, 0) catch return error.Unsafe;
        defer _ = posix.system.close(fd);
        const st = meta(fd, null) orelse return error.Unsafe;
        if ((st.mode & posix.S.IFMT) != posix.S.IFREG or st.mode & 0o044 != 0) {
            problem.* = "API key file must be a regular file unreadable by other users; use chmod 600";
            return error.Invalid;
        }
        if (st.size > 1 << 20) {
            problem.* = "API key file exceeds 1 MiB";
            return error.Invalid;
        }
        var arena: std.heap.ArenaAllocator = .init(s.gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const text = try a.alloc(u8, (1 << 20) + 1);
        var got: usize = 0;
        while (got < text.len) {
            const n = posix.read(fd, text[got..]) catch return error.Unsafe;
            if (n == 0) break;
            got += n;
        }
        if (got > 1 << 20) {
            problem.* = "API key file exceeds 1 MiB";
            return error.Invalid;
        }
        if (!std.unicode.utf8ValidateSlice(text[0..got])) return error.Unsafe;
        var keys: std.ArrayList(Key) = .empty;
        var lines = std.mem.splitAny(u8, text[0..got], "\n\r");
        while (lines.next()) |raw| {
            const line = std.mem.trim(u8, raw, " \t\x0b\x0c\x1c\x1d\x1e\x1f");
            if (line.len == 0 or line[0] == '#') continue;
            var label: []const u8 = undefined;
            var key: []const u8 = line;
            if (std.mem.indexOfScalar(u8, line, ':')) |colon| {
                label = std.mem.trim(u8, line[0..colon], " \t");
                key = std.mem.trim(u8, line[colon + 1 ..], " \t");
                if (!validLabel(label)) {
                    problem.* = "API key labels need 1-64 letters, digits, dots, underscores or hyphens";
                    return error.Invalid;
                }
            } else label = try std.fmt.allocPrint(a, "file-{d}", .{keys.items.len + 1});
            const d = digest(key) orelse {
                problem.* = "API keys must be nonempty text without whitespace";
                return error.Invalid;
            };
            try keys.append(a, .{ .digest = d, .label = try a.dupe(u8, label) });
        }
        if (keys.items.len == 0) {
            problem.* = "API key file contains no keys";
            return error.Invalid;
        }
        if (s.file_arena) |*old| old.deinit();
        s.file_arena = arena;
        s.file = keys.items;
        s.stamp = .{ st.dev, st.ino, st.mtime_ns, st.size, st.mode };
    }

    fn statStamp(s: *Store) ?[5]i128 {
        const pathz = s.gpa.dupeSentinel(u8, s.path.?, 0) catch return null;
        defer s.gpa.free(pathz);
        const st = meta(-1, pathz) orelse return null;
        return .{ st.dev, st.ino, st.mtime_ns, st.size, st.mode };
    }

    /// Rereads the key file when SIGHUP asked or, at most once a second, when its stamp changed.
    fn refresh(s: *Store, now_ns: i96) void {
        if (s.path == null) return;
        const forced = reload_requested.swap(false, .acq_rel);
        if (!forced) if (s.checked_ns) |last| if (now_ns - last < std.time.ns_per_s) return;
        s.checked_ns = now_ns;
        var problem: []const u8 = "";
        const stamp = s.statStamp();
        if (stamp == null or forced or !s.valid or !std.mem.eql(i128, &stamp.?, &s.stamp)) {
            const ok = if (stamp == null) false else if (s.readFile(&problem)) |_| true else |_| false;
            if (ok) {
                s.valid = true;
            } else {
                if (s.valid) log.line("API key file reload refused; authenticated routes remain closed", .{});
                if (s.file_arena) |*old| old.deinit();
                s.file_arena = null;
                s.file = &.{};
                s.valid = false;
            }
        }
    }

    /// The label of the key a request carries (Authorization: Bearer, or x-api-key), or null.
    pub fn match(s: *Store, a: Allocator, authorization: ?[]const u8, x_api_key: ?[]const u8) ?[]const u8 {
        const m = log.io();
        s.mutex.lockUncancelable(m);
        defer s.mutex.unlock(m);
        s.refresh(std.Io.Clock.awake.now(m).toNanoseconds());
        var candidates: [2][32]u8 = undefined;
        candidates[0] = emptyDigest();
        if (authorization) |value| {
            if (std.mem.indexOfScalar(u8, value, ' ')) |space| {
                if (std.ascii.eqlIgnoreCase(value[0..space], "bearer")) candidates[0] = headerDigest(a, std.mem.trim(u8, value[space + 1 ..], latin1_space));
            }
        }
        candidates[1] = if (x_api_key) |value| headerDigest(a, value) else emptyDigest();
        var label: ?[]const u8 = null;
        for ([_][]Key{ s.static, s.file }) |keys| for (keys) |key| {
            var matched = false;
            for (candidates) |c| matched = std.crypto.timing_safe.eql([32]u8, key.digest, c) or matched;
            if (matched and label == null) label = key.label;
        };
        return if (s.valid) label else null;
    }
};

const latin1_space = " \t\r\n\x0b\x0c\x1c\x1d\x1e\x1f\x85\xa0";

/// A header value is latin-1 text whose UTF-8 is hashed, as Python decodes and encodes it.
fn headerDigest(a: Allocator, value: []const u8) [32]u8 {
    const text = @import("http_conn.zig").latin1(a, value) catch return emptyDigest();
    return digest(text) orelse emptyDigest();
}

fn validLabel(label: []const u8) bool {
    if (label.len == 0 or label.len > 64) return false;
    for (label) |ch| if (!(std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '.' or ch == '-')) return false;
    return true;
}

fn emptyDigest() [32]u8 {
    var out: [32]u8 = undefined;
    Sha256.hash("", &out, .{});
    return out;
}

/// SHA-256 of a key, or null for empty text or text with whitespace or control characters.
fn digest(key: []const u8) ?[32]u8 {
    if (key.len == 0) return null;
    var view = std.unicode.Utf8View.init(key) catch return null;
    var it = view.iterator();
    while (it.nextCodepoint()) |cp| if (cp < 32 or isSpace(cp)) return null;
    var out: [32]u8 = undefined;
    Sha256.hash(key, &out, .{});
    return out;
}

fn isSpace(cp: u21) bool {
    return switch (cp) {
        ' ', 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

/// The paths only a key opens (``auth_http.gated``).
pub fn gated(raw_path: []const u8, metrics_open: bool) bool {
    const path = routePath(raw_path);
    if (metrics_open and (std.mem.eql(u8, path, "/metrics") or std.mem.eql(u8, path, "/v1/metrics"))) return false;
    if (std.mem.eql(u8, path, "/v1") or std.mem.startsWith(u8, path, "/v1/")) return true;
    for ([_][]const u8{ "/metrics", "/dashboard", "/stats", "/tokenize", "/detokenize", "/models", "/messages", "/messages/count_tokens", "/chat/completions", "/completions", "/decisions", "/responses" }) |p| if (std.mem.eql(u8, path, p)) return true;
    if (std.mem.startsWith(u8, path, "/responses/")) return true;
    for ([_][]const u8{ "/chat/completions", "/completions", "/decisions", "/models" }) |end| if (std.mem.endsWith(u8, path, end)) return true;
    return false;
}

/// A request path without its query and trailing slashes.
pub fn routePath(raw: []const u8) []const u8 {
    const q = std.mem.indexOfScalar(u8, raw, '?') orelse raw.len;
    return std.mem.trimEnd(u8, raw[0..q], "/");
}

/// Whether ``host`` is a loopback address (a server without keys warns otherwise).
pub fn loopback(host: []const u8) bool {
    if (std.ascii.eqlIgnoreCase(host, "localhost")) return true;
    const ip = std.Io.net.IpAddress.parse(host, 0) catch return false;
    return switch (ip) {
        .ip4 => |v4| v4.bytes[0] == 127,
        .ip6 => |v6| std.mem.eql(u8, &v6.bytes, &[16]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }),
    };
}

test "gates and digests" {
    try std.testing.expect(gated("/v1/models", false));
    try std.testing.expect(!gated("/health", false));
    try std.testing.expect(!gated("/metrics/", true));
    try std.testing.expect(gated("/alternative/chat/completions", false));
    try std.testing.expect(digest("has space") == null);
    try std.testing.expect(loopback("127.0.0.1") and loopback("::1") and !loopback("0.0.0.0"));
}

/// A file's type and permission bits, size, identity and mtime: statx on Linux (its libc has no stat), stat elsewhere.
const Meta = struct { mode: u32, size: u64, dev: u64, ino: u64, mtime_ns: i128 };

/// `path` at the working directory, or the open `fd` itself when `path` is null.
fn meta(fd: posix.fd_t, path: ?[*:0]const u8) ?Meta {
    if (builtin.os.tag == .linux) {
        const linux = std.os.linux;
        var sx: linux.Statx = undefined;
        const want: linux.STATX = .{ .TYPE = true, .MODE = true, .INO = true, .SIZE = true, .MTIME = true };
        const rc = if (path) |p| std.c.statx(posix.AT.FDCWD, p, 0, want, &sx) else std.c.statx(fd, "", 0x1000, want, &sx); // AT_EMPTY_PATH
        if (rc != 0) return null;
        const dev = (@as(u64, sx.dev_major) << 32) | sx.dev_minor;
        return .{ .mode = sx.mode, .size = sx.size, .dev = dev, .ino = sx.ino, .mtime_ns = @as(i128, sx.mtime.sec) * std.time.ns_per_s + sx.mtime.nsec };
    }
    var st: posix.Stat = undefined;
    const rc = if (path) |p| posix.system.stat(p, &st) else posix.system.fstat(fd, &st);
    if (posix.errno(rc) != .SUCCESS) return null;
    const t = st.mtime();
    return .{ .mode = @intCast(st.mode), .size = @intCast(st.size), .dev = @intCast(st.dev), .ino = @intCast(st.ino), .mtime_ns = std.math.lossyCast(i128, t.nsec) + @as(i128, t.sec) * std.time.ns_per_s };
}

test "the key file: refused while others can read it, read at 0600, reread after an edit" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "keys", .data = "ops: sk-one\n" });
    const path = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}/keys", .{tmp.sub_path}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.fchmodat(posix.AT.FDCWD, path, 0o644, 0));
    var problem: []const u8 = "";
    try std.testing.expectError(error.KeyFile, Store.init(a, &.{}, "", path, false, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "chmod 600") != null);
    try std.testing.expectEqual(@as(c_int, 0), std.c.fchmodat(posix.AT.FDCWD, path, 0o600, 0));
    var s = try Store.init(a, &.{}, "", path, false, &problem);
    try std.testing.expectEqualStrings("ops", s.match(a, "Bearer sk-one", null).?);
    const seen = meta(-1, path).?;
    try std.testing.expectEqual(@as(u64, 12), seen.size);
    try std.testing.expectEqual(@as(u32, 0o600), seen.mode & 0o777);
    try tmp.dir.writeFile(io, .{ .sub_path = "keys", .data = "ops: sk-one\nci: sk-two\n" });
    try std.testing.expect(!std.mem.eql(i128, &s.stamp, &s.statStamp().?));
    s.checked_ns = null;
    try std.testing.expectEqualStrings("ci", s.match(a, null, "sk-two").?);
}
