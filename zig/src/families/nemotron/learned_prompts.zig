//! Nemotron's learned prompt states (--learn): a kept state's file on disk, and the probe that keys this build's bits.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const snapshot = @import("snapshot.zig");
const Metal = @import("backend.zig").Metal;
const Snap = snapshot.Snap;

const FILE_MAGIC: u32 = 0x4e454d53; // "NEMS"
const HEAD = 5; // magic, position, head flag, bytes (2)
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// A learned state's file: <dir>/<key>.bin.
pub fn path(buf: []u8, dir: []const u8, key: u64) ![:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}.bin", .{ dir, key }, 0);
}

/// `snap` to `file` (its position, layout and size, then its bytes), through a temporary file renamed into place.
pub fn writeFile(snap: *const Snap, file: [:0]const u8) !void {
    var tmp_buf: [1100]u8 = undefined;
    const tmp = try std.fmt.bufPrintSentinel(&tmp_buf, "{s}.part", .{file}, 0);
    const fd = std.c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.SnapshotWrite;
    errdefer _ = std.c.unlink(tmp);
    {
        defer _ = std.c.close(fd);
        const head = [HEAD]u32{ FILE_MAGIC, snap.at, @intFromBool(snap.head), @truncate(snap.bytes), @truncate(snap.bytes >> 32) };
        try put(fd, std.mem.sliceAsBytes(&head));
        try put(fd, snap.buf.contents()[0..snap.bytes]);
        if (std.c.fsync(fd) != 0) return error.SnapshotWrite;
    }
    if (std.c.rename(tmp, file) != 0) return error.SnapshotWrite;
}

/// `file`'s state into `snap`, whose buffer holds its bytes; a file of another position, layout or size is refused.
pub fn readFile(snap: *Snap, file: [:0]const u8) !void {
    const fd = std.c.open(file, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.SnapshotRead;
    defer _ = std.c.close(fd);
    var head: [HEAD]u32 = undefined;
    try get(fd, std.mem.sliceAsBytes(&head), 0);
    const bytes = @as(u64, head[3]) | @as(u64, head[4]) << 32;
    if (head[0] != FILE_MAGIC or head[1] != snap.at or head[2] != @intFromBool(snap.head) or bytes != snap.bytes) return error.SnapshotRead;
    try get(fd, snap.buf.contents()[0..snap.bytes], @sizeOf(@TypeOf(head)));
}

fn put(fd: c_int, b: []const u8) !void {
    var done: usize = 0;
    while (done < b.len) {
        const n = std.c.write(fd, b.ptr + done, @min(b.len - done, 1 << 30));
        if (n <= 0) return error.SnapshotWrite;
        done += @intCast(n);
    }
}

fn get(fd: c_int, b: []u8, at: u64) !void {
    var done: usize = 0;
    while (done < b.len) {
        const n = std.c.pread(fd, b.ptr + done, @min(b.len - done, 1 << 30), @intCast(at + done));
        if (n <= 0) return error.SnapshotRead;
        done += @intCast(n);
    }
}

/// This build's prompt-pass bits: a fixed prompt's state at a mark, hashed (a kernel or compiler change moves it).
pub fn probe(b: *Metal) !u64 {
    var toks: [4096 + 40 + 7]u32 = undefined; // a whole chunk, a middle one, then a short one on the decode kernels
    for (&toks, 0..) |*t, i| t.* = @intCast(1000 + i * 7919 % 50000);
    const Hook = struct {
        b: *Metal,
        snap: ?*Snap = null,
        failed: ?anyerror = null,
        fn at(ptr: *anyopaque, s: *lanes.Stream, mark: u32) void {
            const p: *@This() = @ptrCast(@alignCast(ptr));
            const c = p.b.cacheOf(s) catch |e| {
                p.failed = e;
                return;
            };
            p.snap = snapshot.save(p.b, c, mark) catch |e| {
                p.failed = e;
                return;
            };
        }
    };
    var hook: Hook = .{ .b = b };
    const marks = [_]u32{4136};
    var s = try lanes.Stream.init(b.gpa, .{ .id = "probe", .prompt = &toks, .max_new = 1, .drafts = b.head != null, .chunks = &.{ 4096, 4136 }, .reuse = .{ .marks = &marks, .hook = .{ .ptr = &hook, .at = Hook.at } } });
    defer s.deinit(b.gpa);
    const be = b.backend();
    defer be.release(&s);
    try be.prefill(&s);
    const snap = hook.snap orelse return hook.failed orelse error.SnapshotOutOfStep;
    defer snapshot.drop(b, snap);
    try b.drain();
    return std.hash.Wyhash.hash(0x4e45, snap.buf.contents()[0..snap.bytes]);
}

fn metal(ptr: *anyopaque) *Metal {
    return @ptrCast(@alignCast(ptr));
}

/// --learn: a kept state to its file, once the copy into it has landed.
pub fn snapWrite(ptr: *anyopaque, saved: *anyopaque, dir: [:0]const u8, key: u64) anyerror!void {
    const b = metal(ptr);
    const snap: *Snap = @ptrCast(@alignCast(saved));
    var buf: [1100]u8 = undefined;
    const file = try path(&buf, dir, key);
    try b.drain();
    try writeFile(snap, file);
}

/// --learn: learned state `key` of `at` tokens read back into a new buffer, restored like any kept state.
pub fn snapRead(ptr: *anyopaque, dir: [:0]const u8, key: u64, at: u32) anyerror!*anyopaque {
    const b = metal(ptr);
    var buf: [1100]u8 = undefined;
    const file = try path(&buf, dir, key);
    const head = b.head != null;
    const n = snapshot.bytes(b.m.config, at, head);
    const snap = try b.gpa.create(Snap);
    errdefer b.gpa.destroy(snap);
    snap.* = .{ .at = at, .head = head, .buf = try b.m.device.buffer(n, opts), .bytes = n };
    errdefer snap.buf.deinit();
    try readFile(snap, file);
    return @ptrCast(snap);
}

/// --learn: learned state `key`'s file removed.
pub fn snapForget(_: *anyopaque, dir: [:0]const u8, key: u64) void {
    var buf: [1100]u8 = undefined;
    _ = std.c.unlink(path(&buf, dir, key) catch return);
}

test "a learned state's file name is its key in hex" {
    var buf: [1100]u8 = undefined;
    try std.testing.expectEqualStrings("/x/00000000000004d2.bin", try path(&buf, "/x", 1234));
}
