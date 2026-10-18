//! Whole-pack bytes and requested host allocation peaks against the frozen builder on the same synthetic fixture.
const std = @import("std");
const fixture = @import("pack_test.zig");
const pack = @import("pack.zig");
const previous_peaks = [_]usize{ 636288, 628370 };
const expected_hashes = [2][3][]const u8{
    .{ "bc88b43fa3b82ecf971cc41dcce78118fdffcb72a54987621fbb17d9189d7f3b", "ff6342967562d9a73c82756e197659fcb180af2173a7baef39dc64334e1e5eca", "bcdd04bf3a681fda5d0b9529fdecd328d054d88e0848799afd36c66b898612e0" },
    .{ "eaf35876b18a5471b1921e3b6cfa78af3bb9d3ce1f08d999687afc162f062b95", "ff6342967562d9a73c82756e197659fcb180af2173a7baef39dc64334e1e5eca", "bcdd04bf3a681fda5d0b9529fdecd328d054d88e0848799afd36c66b898612e0" },
};

const PackCounter = struct {
    child: std.mem.Allocator,
    live: usize = 0,
    peak: usize = 0,
    fn self(ctx: *anyopaque) *PackCounter {
        return @ptrCast(@alignCast(ctx));
    }
    fn changed(c: *PackCounter, old: usize, new: usize) void {
        c.live = c.live - old + new;
        c.peak = @max(c.peak, c.live);
    }
    fn alloc(ctx: *anyopaque, n: usize, align_: std.mem.Alignment, ra: usize) ?[*]u8 {
        const c = self(ctx);
        const p = c.child.rawAlloc(n, align_, ra) orelse return null;
        c.changed(0, n);
        return p;
    }
    fn resize(ctx: *anyopaque, bytes: []u8, align_: std.mem.Alignment, n: usize, ra: usize) bool {
        const c = self(ctx);
        if (!c.child.rawResize(bytes, align_, n, ra)) return false;
        c.changed(bytes.len, n);
        return true;
    }
    fn remap(ctx: *anyopaque, bytes: []u8, align_: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        const c = self(ctx);
        const p = c.child.rawRemap(bytes, align_, n, ra) orelse return null;
        c.changed(bytes.len, n);
        return p;
    }
    fn free(ctx: *anyopaque, bytes: []u8, align_: std.mem.Alignment, ra: usize) void {
        const c = self(ctx);
        c.child.rawFree(bytes, align_, ra);
        c.changed(bytes.len, 0);
    }
    fn allocator(c: *PackCounter) std.mem.Allocator {
        return .{ .ptr = c, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
};

test "Flash Next memory: pack phases retain whole-file bytes and fit the predecessor host allocation peak" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture.writeCheckpoint(tmp, arena.allocator());
    const dir = try fixture.tmpPath(arena.allocator(), tmp, ".");
    const out = try fixture.tmpPath(arena.allocator(), tmp, "out");
    try tmp.dir.createDirPath(io, "out");
    try tmp.dir.writeFile(io, .{ .sub_path = "draft.txt", .data = "0 1 2 3" });
    const vocab = try fixture.tmpPath(arena.allocator(), tmp, "draft.txt");
    for ([_]?[]const u8{ null, vocab }, 0..) |draft, mode| {
        var counter = PackCounter{ .child = a };
        _ = try pack.build(counter.allocator(), io, dir, out, draft);
        try std.testing.expectEqual(@as(usize, 0), counter.live);
        if (@import("builtin").os.tag == .macos) try std.testing.expect(counter.peak <= previous_peaks[mode]);
        std.debug.print("PACK_PHASE_PEAK {d} {d}\n", .{ mode, counter.peak });
        for ([_][]const u8{ "pack.safetensors", "pack_mlx.safetensors", "pack_mtp_mlx.safetensors" }, 0..) |name, file_index| {
            const path = try std.fs.path.join(a, &.{ out, name });
            defer a.free(path);
            const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
            defer a.free(bytes);
            var hash: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
            try std.testing.expectEqualStrings(expected_hashes[mode][file_index], &std.fmt.bytesToHex(hash, .lower));
        }
    }
}
