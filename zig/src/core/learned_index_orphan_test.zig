const std = @import("std");
const imprint = @import("prompt_imprint.zig");
const Imprint = imprint.Imprint;
const Fake = @import("prompt_cache_fake.zig").Fake;
const pc = @import("prompt_cache.zig");

fn learn(im: *Imprint, prompt: []const u32) !void {
    const Disk = struct {
        fn free(_: [:0]const u8) ?u64 {
            return 1 << 30;
        }
        fn clock() ?u64 {
            return 100;
        }
    };
    const a = std.testing.allocator;
    im.admission = .{ .floor = 0, .free = Disk.free, .clock = Disk.clock };
    var f: Fake = .{ .gpa = a, .at = 5 };
    var s = pc.Store.init(a, f.learned(), .{ .min_prompt = 0, .min_gap = 1, .lookahead = 1 }, 1 << 20);
    defer s.deinit();
    s.imprint = im;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    _ = try s.lookup(arena.allocator(), prompt, 7, &.{5}, &.{}, &.{});
    try std.testing.expect(s.keep(prompt, 5, null, &.{}, &.{}));
    try std.testing.expect(im.has(Imprint.keyOf(prompt[0..6])));
}
fn exists(file: [:0]const u8) bool {
    const fd = std.c.open(file, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return false;
    _ = std.c.close(fd);
    return true;
}
fn put(file: [:0]const u8, bytes: []const u8) !void {
    const fd = std.c.open(file, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.TestFile;
    defer _ = std.c.close(fd);
    try imprint.writeAll(fd, bytes);
}

test "reopen sweeps only unindexed owned payloads left by old post-tear learning" {
    const a = std.testing.allocator;
    for (0..3) |kind| {
        var buf: [128]u8 = undefined;
        const root = try std.fmt.bufPrint(&buf, "/tmp/tf-learn-lost-{d}-{d}", .{ std.c.getpid(), kind });
        defer @import("prompt_imprint_test.zig").rmTree(root);
        const first = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
        var later = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18 };
        var lost: [3]u64 = undefined;
        {
            var im = try Imprint.open(a, root, 1, 1 << 20);
            defer im.deinit();
            try learn(&im, &first);
            for (&lost) |*key| {
                try learn(&im, &later);
                key.* = Imprint.keyOf(later[0..6]);
                later[0] += 10;
            }
            var path: [1200]u8 = undefined;
            const fd = std.c.open(try std.fmt.bufPrintSentinel(&path, "{s}/index", .{im.dir}, 0), .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
            if (fd < 0) return error.TestFile;
            {
                defer _ = std.c.close(fd);
                var original: [240]u8 = undefined;
                try std.testing.expect(imprint.readAt(fd, &original, 0));
                try std.testing.expectEqual(@as(c_int, 0), std.c.ftruncate(fd, 60));
                if (std.c.lseek(fd, 60, std.c.SEEK.SET) != 60) return error.TestFile;
                const key = Imprint.keyOf(&.{ 31, 32 });
                const torn = [_]u32{ 0x494d5052, 2, @truncate(key), @truncate(key >> 32), 1, if (kind == 0) 512 else 2, 0, 100, 0 };
                if (kind != 1) try imprint.writeAll(fd, std.mem.sliceAsBytes(&torn));
                try imprint.writeAll(fd, "x");
                try imprint.writeAll(fd, original[60..]);
            }
            try put(try std.fmt.bufPrintSentinel(&path, "{s}/00000000000000af.r0.bin", .{im.dir}, 0), "lost");
            try put(try std.fmt.bufPrintSentinel(&path, "{s}/00000000000000af.r1.bin", .{im.dir}, 0), "lost");
            try put(try std.fmt.bufPrintSentinel(&path, "{s}/{x:0>16}.r0.bin", .{ im.dir, Imprint.keyOf(first[0..6]) }, 0), "kept");
            for ([_][]const u8{ "foreign.bin", "00000000000000AF.bin", "00000000000000af.r2.bin" }) |name| {
                try put(try std.fmt.bufPrintSentinel(&path, "{s}/{s}", .{ im.dir, name }, 0), "foreign");
            }
        }
        var im = try Imprint.open(a, root, 1, 1 << 20);
        defer im.deinit();
        try std.testing.expectEqual(@as(usize, 1), im.metas.items.len);
        try std.testing.expectEqual(@as(u64, 60), try im.indexScratchBytes());
        var path: [1200]u8 = undefined;
        for (lost) |key| try std.testing.expect(!exists(try Fake.file(&path, im.dir, key)));
        try std.testing.expect(exists(try Fake.file(&path, im.dir, Imprint.keyOf(first[0..6]))));
        try std.testing.expect(exists(try std.fmt.bufPrintSentinel(&path, "{s}/{x:0>16}.r0.bin", .{ im.dir, Imprint.keyOf(first[0..6]) }, 0)));
        for ([_][]const u8{ "00000000000000af.r0.bin", "00000000000000af.r1.bin" }) |name| {
            try std.testing.expect(!exists(try std.fmt.bufPrintSentinel(&path, "{s}/{s}", .{ im.dir, name }, 0)));
        }
        for ([_][]const u8{ "foreign.bin", "00000000000000AF.bin", "00000000000000af.r2.bin" }) |name| {
            try std.testing.expect(exists(try std.fmt.bufPrintSentinel(&path, "{s}/{s}", .{ im.dir, name }, 0)));
        }
        try std.testing.expectEqual(@as(u64, 109), try @import("lanes").learned_dirs.bytes(im.dir));
    }
}
