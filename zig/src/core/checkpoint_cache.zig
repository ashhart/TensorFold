//! Load-only checkpoint pages leave Linux's cache after their last read; runtime sources keep their own policy.
const std = @import("std");
const Io = std.Io;

/// Deferred in the checkpoint's owning load scope, after all users of its borrowed bytes.
pub fn close(ck: anytype) void {
    ck.release();
    ck.close();
}

/// A completed positional read needs no mapping release. Advice is best effort and never changes file bytes.
pub fn drop(file: Io.File) void {
    if (@import("builtin").os.tag != .linux) return;
    _ = std.os.linux.fadvise(file.handle, 0, 0, std.os.linux.POSIX_FADV.DONTNEED);
}

/// Only the pack's explicit load sources, including nested index paths; no recursive directory sweep.
pub fn files(io: Io, dir: []const u8, names: []const []const u8) void {
    if (@import("builtin").os.tag != .linux) return;
    visit(io, dir, names, drop);
}

fn visit(io: Io, dir: []const u8, names: []const []const u8, comptime advise: anytype) void {
    for (names) |name| {
        var buf: [Io.Dir.max_path_bytes]u8 = undefined;
        const path = std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, name }) catch continue;
        const f = Io.Dir.cwd().openFile(io, path, .{}) catch continue;
        defer f.close(io);
        advise(f);
    }
}

/// Flash Next also closes shards while loading; its live PLE mappings must be excluded from the final sweep.
pub fn directory(io: Io, dir: []const u8, keep: anytype) void {
    if (@import("builtin").os.tag != .linux) return;
    walk(io, dir, keep, drop);
}

fn walk(io: Io, dir: []const u8, keep: anytype, comptime advise: anytype) void {
    var d = Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return;
    defer d.close(io);
    var it = d.iterate();
    entries: while (it.next(io) catch null) |e| {
        if (e.kind != .file and e.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, e.name, ".safetensors")) continue;
        const f = d.openFile(io, e.name, .{}) catch continue;
        defer f.close(io);
        const stat = f.stat(io) catch continue;
        for (keep) |k| {
            const held = k.file.file.stat(io) catch continue :entries;
            if (held.inode == stat.inode) continue :entries;
        }
        advise(f);
    }
}

const t = std.testing;
const Fake = struct {
    reads: usize = 0,
    releases: usize = 0,
    closed: bool = false,
    pub fn release(f: *@This()) void {
        std.debug.assert(f.reads == 3 and !f.closed);
        f.releases += 1;
    }
    pub fn close(f: *@This()) void {
        std.debug.assert(f.releases == 1 and !f.closed);
        f.closed = true;
    }
    fn load(f: *@This(), fail: bool) !void {
        defer closeCheckpoint(f);
        f.reads += 1; // backbone
        f.reads += 1; // output head
        f.reads += 1; // separate drafter
        if (fail) return error.CopyRefused;
    }
};
fn closeCheckpoint(f: *Fake) void {
    close(f);
}

test "checkpoint cache: last read precedes release and close on success and refusal" {
    for ([_]bool{ false, true }) |fail| {
        var f: Fake = .{};
        if (fail) try t.expectError(error.CopyRefused, f.load(true)) else try f.load(false);
        try t.expectEqual(@as(usize, 1), f.releases);
        try t.expect(f.closed);
    }
}

test "checkpoint cache: final sweep excludes live PLE, subdirectories and unrelated files" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    for ([_][]const u8{ "model.safetensors", "mtp.safetensors", "ple.safetensors", "config.json" }) |name|
        try tmp.dir.writeFile(t.io, .{ .sub_path = name, .data = name });
    try tmp.dir.createDirPath(t.io, "ssd.safetensors");
    try tmp.dir.writeFile(t.io, .{ .sub_path = "ssd.safetensors/expert.safetensors", .data = "runtime" });
    if (@import("builtin").os.tag != .windows) try tmp.dir.symLink(t.io, "ple.safetensors", "alias.safetensors", .{});
    const ple = try tmp.dir.openFile(t.io, "ple.safetensors", .{});
    defer ple.close(t.io);
    const keep = [_]struct { file: struct { file: Io.File } }{.{ .file = .{ .file = ple } }};
    const path = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(path);
    const Spy = struct {
        var count: usize = 0;
        fn advise(f: Io.File) void {
            var buf: [64]u8 = undefined;
            const n = f.readPositionalAll(t.io, &buf, 0) catch unreachable;
            std.debug.assert(std.mem.eql(u8, buf[0..n], "model.safetensors") or std.mem.eql(u8, buf[0..n], "mtp.safetensors"));
            count += 1;
        }
    };
    Spy.count = 0;
    walk(t.io, path, &keep, Spy.advise);
    try t.expectEqual(@as(usize, 2), Spy.count);
}

test "checkpoint cache: pack sweep names nested shards and skips missing and unrelated sources" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(t.io, "tensors");
    for ([_][]const u8{ "a.safetensors", "tensors/b.safetensors", "runtime.safetensors" }) |name|
        try tmp.dir.writeFile(t.io, .{ .sub_path = name, .data = name });
    const path = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(path);
    const Spy = struct {
        var count: usize = 0;
        fn advise(f: Io.File) void {
            var buf: [64]u8 = undefined;
            const n = f.readPositionalAll(t.io, &buf, 0) catch unreachable;
            std.debug.assert(std.mem.eql(u8, buf[0..n], "a.safetensors") or std.mem.eql(u8, buf[0..n], "tensors/b.safetensors"));
            count += 1;
        }
    };
    Spy.count = 0;
    visit(t.io, path, &.{ "a.safetensors", "missing.safetensors", "tensors/b.safetensors" }, Spy.advise);
    try t.expectEqual(@as(usize, 2), Spy.count);
}
