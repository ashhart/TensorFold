//! TENSORFOLD_TOOL_ID_KEY_FILE: an operator's tool-call id key from an owner-only file, so ids outlive one process.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const auth = @import("auth.zig");
const ids = @import("ids.zig");
const meta = @import("file_meta.zig").meta;
const Allocator = std.mem.Allocator;

pub const name = "TENSORFOLD_TOOL_ID_KEY_FILE";
/// The fewest key bytes a file holds once one trailing line ending is dropped: 256 bits.
pub const least_bytes = 32;
/// The most bytes a key file holds.
pub const most_bytes = 4096;

pub const Error = error{ Refused, OutOfMemory };

/// The key ``raw`` names, null when blank; a refusal names the setting, never its text or the file's bytes.
pub fn load(a: Allocator, raw: ?[]const u8, problem: *[]const u8) Error!?ids.Key {
    const path = (platform.environment.path(name, raw) catch return refuse(a, problem, "is not a path: it holds control characters", .{})) orelse return null;
    const fd = auth.openKeys(try a.dupeSentinel(u8, path, 0)) catch |e|
        return refuse(a, problem, "names a file that cannot be opened ({s}); name an existing file, not a link", .{@errorName(e)});
    defer _ = platform.c.close(fd);
    const st = meta(fd, null) orelse return refuse(a, problem, "names a file whose mode cannot be read", .{});
    if (st.mode & platform.c.S.IFMT != platform.c.S.IFREG) return refuse(a, problem, "names something that is not a regular file", .{});
    if (st.mode & 0o077 != 0) return refuse(a, problem, "names a file other users can read or write (mode 0{o}); use chmod 600", .{st.mode & 0o7777});
    var secret: [most_bytes + 1]u8 = undefined;
    defer std.crypto.secureZero(u8, &secret);
    var got: usize = 0;
    while (got < secret.len) {
        const n = auth.readKeys(fd, secret[got..]) catch |e| return refuse(a, problem, "names a file that cannot be read ({s})", .{@errorName(e)});
        if (n == 0) break;
        got += n;
    }
    if (got > most_bytes) return refuse(a, problem, "names a file of more than {d} bytes", .{most_bytes});
    const ending: usize = if (std.mem.endsWith(u8, secret[0..got], "\r\n")) 2 else if (std.mem.endsWith(u8, secret[0..got], "\n")) 1 else 0;
    const key = secret[0 .. got - ending];
    if (key.len < least_bytes) return refuse(a, problem, "names a file of fewer than {d} key bytes; write one with openssl rand -hex 32", .{least_bytes});
    return ids.Key.of(key);
}

fn refuse(a: Allocator, problem: *[]const u8, comptime why: []const u8, args: anytype) Error {
    problem.* = try std.fmt.allocPrint(a, name ++ " " ++ why, args);
    return error.Refused;
}

/// Every log line written while a test runs, to show that no key's bytes reach one.
const Capture = struct {
    bytes: std.ArrayList(u8) = .empty,
    fn line(ctx: *anyopaque, text: []const u8) void {
        const self: *@This() = @ptrCast(@alignCast(ctx));
        self.bytes.appendSlice(std.testing.allocator, text) catch @panic("log capture");
    }
};

const test_secret = "SECRET-tool-id-key-0123456789abcdef-0123456789";

fn put(a: Allocator, dir: std.testing.TmpDir, file: []const u8, data: []const u8, mode: platform.c.mode_t) ![:0]const u8 {
    try dir.dir.writeFile(std.testing.io, .{ .sub_path = file, .data = data });
    const path = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}/{s}", .{ dir.sub_path, file }, 0);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.fchmodat(platform.posix.AT.FDCWD, path, mode, 0));
    return path;
}

/// What ``load`` says for ``path``: it must refuse, name the setting first and echo neither the path nor a key.
fn refusal(a: Allocator, path: []const u8) ![]const u8 {
    var problem: []const u8 = "";
    try std.testing.expectError(error.Refused, load(a, path, &problem));
    try std.testing.expect(std.mem.startsWith(u8, problem, name ++ " "));
    try std.testing.expect(std.mem.indexOf(u8, problem, "SECRET") == null);
    try std.testing.expect(std.mem.indexOf(u8, problem, path) == null);
    return problem;
}

test "a key file that is missing, locked, a link, a folder, open to others, short or long is refused by name, unechoed" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // mode bits and links; NTFS keeps access in ACLs
    try @import("test_permissions.zig").skipIfRoot();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var capture: Capture = .{};
    defer capture.bytes.deinit(std.testing.allocator);
    @import("log.zig").Test.sink = .{ .ctx = &capture, .call = Capture.line };
    defer @import("log.zig").Test.sink = null;
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const has = struct {
        fn has(text: []const u8, part: []const u8) !void {
            try std.testing.expect(std.mem.indexOf(u8, text, part) != null);
        }
    }.has;
    try has(try refusal(a, try std.fmt.allocPrint(a, "{s}/missing", .{dir})), "cannot be opened (FileNotFound)");
    for ([_]platform.c.mode_t{ 0o644, 0o640, 0o604, 0o620, 0o602, 0o610 }) |mode| {
        const why = try refusal(a, try put(a, tmp, "shared", test_secret, mode));
        try has(why, try std.fmt.allocPrint(a, "other users can read or write (mode 0{o}); use chmod 600", .{mode}));
    }
    try has(try refusal(a, try put(a, tmp, "short", test_secret[0 .. least_bytes - 1] ++ "\n", 0o600)), "fewer than 32 key bytes");
    try has(try refusal(a, try put(a, tmp, "crlf-short", test_secret[0 .. least_bytes - 1] ++ "\r\n", 0o600)), "fewer than 32 key bytes");
    try has(try refusal(a, try put(a, tmp, "long", test_secret ++ @as([most_bytes + 1 - test_secret.len]u8, @splat('k')), 0o600)), "more than 4096 bytes");
    const folder = try std.fmt.allocPrintSentinel(a, "{s}/folder", .{dir}, 0);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.mkdir(folder, 0o700));
    try has(try refusal(a, folder), "not a regular file");
    _ = try put(a, tmp, "good", test_secret, 0o600);
    const link = try std.fmt.allocPrintSentinel(a, "{s}/link", .{dir}, 0);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.symlink("good", link));
    try has(try refusal(a, link), "cannot be opened");
    try has(try refusal(a, try put(a, tmp, "locked", test_secret, 0o000)), "cannot be opened (AccessDenied)");
    var problem: []const u8 = "";
    try std.testing.expectError(error.Refused, load(a, test_secret ++ "\x1b", &problem)); // a key pasted into the setting
    try std.testing.expect(std.mem.startsWith(u8, problem, name ++ " ") and std.mem.indexOf(u8, problem, "SECRET") == null);
    try std.testing.expectError(error.Refused, load(a, test_secret, &problem));
    try std.testing.expect(std.mem.startsWith(u8, problem, name ++ " ") and std.mem.indexOf(u8, problem, "SECRET") == null);
    try std.testing.expectEqual(@as(usize, 0), capture.bytes.items.len); // and nothing logged at all
}

test "the same key file gives the same key on every load, one line ending dropped, and a blank setting gives none" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the files are made owner-only by mode bits
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var problem: []const u8 = "";
    const want = try ids.derived(a, &.of(test_secret), "t_", 32, "x");
    for ([_][]const u8{ test_secret, test_secret ++ "\n", test_secret ++ "\r\n" }, 0..) |data, i| for (0..2) |_| {
        const key = (try load(a, try put(a, tmp, try std.fmt.allocPrint(a, "k{d}", .{i}), data, 0o600), &problem)).?;
        try std.testing.expectEqualStrings(want, try ids.derived(a, &key, "t_", 32, "x"));
    };
    const other = (try load(a, try put(a, tmp, "other", test_secret ++ "\n\n", 0o400), &problem)).?;
    try std.testing.expect(!std.mem.eql(u8, want, try ids.derived(a, &other, "t_", 32, "x")));
    _ = (try load(a, try put(a, tmp, "least", test_secret[0..least_bytes], 0o600), &problem)).?;
    _ = (try load(a, try put(a, tmp, "most", test_secret ++ @as([most_bytes - test_secret.len]u8, @splat('k')), 0o600), &problem)).?;
    for ([_]?[]const u8{ null, "", " \t" }) |raw| try std.testing.expect(try load(a, raw, &problem) == null);
}
