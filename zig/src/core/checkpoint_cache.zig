//! Load-only checkpoint pages leave Linux's cache after their last read; runtime sources keep their own policy.
const std = @import("std");
const Io = std.Io;
pub const Load = @import("checkpoint_drop.zig").Load;
pub const range = @import("checkpoint_drop.zig").range;

/// Deferred in the checkpoint's owning load scope, after all users of its borrowed bytes.
pub fn close(ck: anytype) void {
    ck.release();
    ck.close();
}

/// A completed positional read needs no mapping release. Advice is best effort and never changes file bytes.
pub fn drop(file: Io.File) void {
    @import("checkpoint_drop.zig").drop(file);
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
pub fn shards(io: Io, paths: []const [:0]u8, keep: anytype) void {
    if (@import("builtin").os.tag != .linux) return;
    walk(io, paths, keep, drop);
}

/// `paths` from the index (checkpoint.shardFiles), so shards nested under tensors/ are reached; held inodes skipped.
fn walk(io: Io, paths: []const [:0]u8, keep: anytype, comptime advise: anytype) void {
    entries: for (paths) |path| {
        const f = Io.Dir.cwd().openFile(io, path, .{}) catch continue;
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

test "checkpoint cache: final sweep reaches the index's nested shards and excludes live PLE and aliases" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(t.io, "tensors");
    for ([_][]const u8{ "tensors/model-1.safetensors", "tensors/model-2.safetensors", "tensors/ple.safetensors" }) |name|
        try tmp.dir.writeFile(t.io, .{ .sub_path = name, .data = name });
    if (@import("builtin").os.tag != .windows) try tmp.dir.symLink(t.io, "ple.safetensors", "tensors/alias.safetensors", .{});
    const ple = try tmp.dir.openFile(t.io, "tensors/ple.safetensors", .{});
    defer ple.close(t.io);
    const keep = [_]struct { file: struct { file: Io.File } }{.{ .file = .{ .file = ple } }};
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(root);
    var paths: [5][:0]u8 = undefined;
    for (&paths, [_][]const u8{ "model-1", "model-2", "ple", "alias", "missing" }) |*p, name|
        p.* = try std.fmt.allocPrintSentinel(t.allocator, "{s}/tensors/{s}.safetensors", .{ root, name }, 0);
    defer for (paths) |p| t.allocator.free(p);
    const Spy = struct {
        var count: usize = 0;
        fn advise(f: Io.File) void {
            var buf: [64]u8 = undefined;
            const n = f.readPositionalAll(t.io, &buf, 0) catch unreachable;
            std.debug.assert(std.mem.startsWith(u8, buf[0..n], "tensors/model-"));
            count += 1;
        }
    };
    Spy.count = 0;
    walk(t.io, &paths, &keep, Spy.advise);
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
