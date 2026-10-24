//! A journal-backed storage fixture without an index or inference engine.
const std = @import("std");
const Storage = @import("notes/storage.zig").Storage;
const Bytes = @import("notes_storage_bytes_test.zig").Bytes;
const a = std.testing.allocator;

pub const Rig = struct {
    tmp: std.testing.TmpDir,
    arena: std.heap.ArenaAllocator,
    bytes: Bytes = .{},
    owned: ?*Storage = null,
    snap: []const u8 = "",
    path: []const u8 = "",

    pub fn init(_: std.mem.Allocator) !*Rig {
        const r = try a.create(Rig);
        r.* = .{ .tmp = std.testing.tmpDir(.{}), .arena = .init(a) };
        errdefer r.deinit();
        const temp = r.arena.allocator();
        const dir = try r.tmp.dir.realPathFileAlloc(std.testing.io, ".", temp);
        r.snap = try std.fs.path.join(temp, &.{ dir, "notes.json" });
        r.path = try std.mem.concat(temp, u8, &.{ r.snap, ".journal" });
        try r.restart();
        return r;
    }

    pub fn store(r: *Rig) *Storage {
        return r.owned.?;
    }

    pub fn restart(r: *Rig) !void {
        if (r.owned) |s| s.deinit();
        r.owned = null;
        var why: []const u8 = "";
        r.owned = try Storage.load(a, @embedFile("notes/testdata/l6_bytes_snapshot.json"), r.bytes.text(), r.path, &why);
        try r.store().beginRun(r.arena.allocator(), .{});
    }

    pub fn journal(r: *Rig, temp: std.mem.Allocator) ![]const u8 {
        return std.Io.Dir.cwd().readFileAlloc(std.testing.io, r.path, temp, .unlimited);
    }

    pub fn deinit(r: *Rig) void {
        if (r.owned) |s| s.deinit();
        r.tmp.cleanup();
        r.arena.deinit();
        a.destroy(r);
    }
};
