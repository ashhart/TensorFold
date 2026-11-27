//! Canonical identity metadata, independent main payload bytes and unchanged predecessor allocation ceilings.
const std = @import("std");
const fixture = @import("pack_test.zig");
const pack = @import("pack.zig");
/// The frozen builder's peaks on this fixture with its relative paths; the pack's own peaks stay at or under them.
const previous_peaks = [_]usize{ 636195, 628339 };
const expected_hashes = [2][3][]const u8{
    .{ "e65960962b48470959293a1ddaba29130d3bb05602f4660cb527ccfba64007b8", "30ef606b3f65f4ea6683f495fe20c12bc95d1713fc948720c235f0a32e05276b", "8e63cedb06a1870a9c985d41bfac2aa83f89f564ed4dfea0f8e1d2d06683c206" },
    .{ "fca0d68664069d63514f9b6390e24136fa98e00bdd9ed0f7cdda1ca688f6b416", "30ef606b3f65f4ea6683f495fe20c12bc95d1713fc948720c235f0a32e05276b", "8e63cedb06a1870a9c985d41bfac2aa83f89f564ed4dfea0f8e1d2d06683c206" },
};
/// Complete payload-region SHA-256 values frozen from main ac4cb4f538, independently of identity metadata.
const main_payload_hashes = [2][3][]const u8{
    .{ "5c1f504914d9bcbf07d36d209bf3ca5d999e81ebd8e0452a296d583a1c4cea14", "50704748745904236dd5a8ce0127c7a1d5b33ab1990eee9b7edd4a725d4547ac", "233fa80d521e6572777c520fd86baa5f6c09d10af9b2d3a8280424fbf6c9260b" },
    .{ "e761d1217af487d9905ceb88e98d7eedd4eb213d643d0e871edd35f8f9611cd7", "50704748745904236dd5a8ce0127c7a1d5b33ab1990eee9b7edd4a725d4547ac", "233fa80d521e6572777c520fd86baa5f6c09d10af9b2d3a8280424fbf6c9260b" },
};
const main_linux_peaks = [_]usize{ 601023, 599747 };

fn canonicalHash(a: std.mem.Allocator, bytes: []const u8) ![32]u8 {
    const data = 8 + std.mem.readInt(u64, bytes[0..8], .little);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, bytes[8..@intCast(data)], .{});
    defer parsed.deinit();
    const source = parsed.value.object.getPtr("__metadata__").?.object.getPtr("tf_source").?;
    try std.testing.expect(std.mem.startsWith(u8, source.string, "flashnext pack 4\n"));
    var stable: std.Io.Writer.Allocating = .init(a);
    defer stable.deinit();
    var lines = std.mem.splitScalar(u8, source.string, '\n');
    while (lines.next()) |line| {
        if (line.len == 0 or std.mem.startsWith(u8, line, "stamp ") or std.mem.startsWith(u8, line, "payload ")) continue;
        try stable.writer.print("{s}\n", .{line});
    }
    source.* = .{ .string = stable.written() };
    const header = try std.json.Stringify.valueAlloc(a, parsed.value, .{});
    defer a.free(header);
    var length: [8]u8 = undefined;
    std.mem.writeInt(u64, &length, header.len, .little);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(&length);
    hash.update(header);
    hash.update(bytes[@intCast(data)..]);
    return hash.finalResult();
}

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

test "Flash Next memory: pack phases retain main payload bytes and fit the predecessor host allocation peak" {
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
        // The fixture's paths are relative, so this peak is the same wherever the checkout lives.
        if (@import("builtin").os.tag == .macos) try std.testing.expect(counter.peak <= previous_peaks[mode]);
        std.debug.print("PACK_PHASE_PEAK {d} {d}\n", .{ mode, counter.peak });
        if (@import("builtin").os.tag == .linux) try std.testing.expect(counter.peak <= main_linux_peaks[mode]);
        for ([_][]const u8{ "pack.safetensors", "pack_mlx.safetensors", "pack_mtp_mlx.safetensors" }, 0..) |name, file_index| {
            const path = try std.fs.path.join(a, &.{ out, name });
            defer a.free(path);
            const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
            defer a.free(bytes);
            var hash: [32]u8 = undefined;
            const data = 8 + std.mem.readInt(u64, bytes[0..8], .little);
            std.crypto.hash.sha2.Sha256.hash(bytes[@intCast(data)..], &hash, .{});
            try std.testing.expectEqualStrings(main_payload_hashes[mode][file_index], &std.fmt.bytesToHex(hash, .lower));
            hash = try canonicalHash(a, bytes);
            try std.testing.expectEqualStrings(expected_hashes[mode][file_index], &std.fmt.bytesToHex(hash, .lower));
        }
    }
}
