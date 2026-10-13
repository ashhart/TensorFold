const std = @import("std");
const pc = @import("prompt_cache.zig");
const Imprint = @import("prompt_imprint.zig").Imprint;
const Fake = @import("prompt_cache_fake.zig").Fake;
const dirs = @import("lanes").learned_dirs;
const Admission = @import("lanes").learned_disk.Admission;
const Disk = struct {
    var tick: ?u64 = 100;
    fn clock() ?u64 {
        return tick;
    }
    fn free(_: [:0]const u8) ?u64 {
        return 1 << 30;
    }
};
fn root(buf: []u8, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-learn-fault-{s}-{d}", .{ suffix, std.c.getpid() });
}
fn put(dir: []const u8, name: []const u8, count: usize) !void {
    var buf: [1200]u8 = undefined;
    const fd = std.c.open(try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0), .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.Create;
    defer _ = std.c.close(fd);
    if (std.c.ftruncate(fd, @intCast(count)) != 0) return error.Create;
}

pub fn learnOne(im: *Imprint, prompt: []const u32) !void {
    const a = std.testing.allocator;
    im.admission = .{ .floor = 0, .free = Disk.free, .clock = Disk.clock };
    var f: Fake = .{ .gpa = a, .at = 5, .sum = prefixSum(prompt[0..5]) };
    var s = pc.Store.init(a, f.learned(), .{ .min_prompt = 0, .min_gap = 1, .lookahead = 1 }, 1 << 20);
    defer s.deinit();
    s.imprint = im;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    _ = try s.lookup(arena.allocator(), prompt, 7, &.{5}, &.{}, &.{});
    try std.testing.expect(s.keep(prompt, 5, null, &.{}, &.{}));
    try std.testing.expect(im.has(Imprint.keyOf(prompt[0..6])));
    try std.testing.expectEqual(@as(usize, 1), f.writes);
}

fn prefixSum(tokens: []const u32) u64 {
    var sum: u64 = 0;
    for (tokens) |t| sum = sum *% 31 +% t;
    return sum;
}

pub fn recallOne(im: *Imprint, prompt: []const u32) !void {
    const a = std.testing.allocator;
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.learned(), .{ .min_prompt = 0, .min_gap = 1, .lookahead = 1 }, 1 << 20);
    defer s.deinit();
    s.imprint = im;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const plan = try s.begin(arena.allocator(), prompt, 7, &.{5}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 5), plan.from);
    try std.testing.expectEqual(@as(u32, 5), f.at);
    try std.testing.expectEqual(prefixSum(prompt[0..5]), f.sum);
}

fn appendIndex(dir: []const u8, bytes: []const u8) !void {
    var buf: [1200]u8 = undefined;
    const fd = std.c.open(try std.fmt.bufPrintSentinel(&buf, "{s}/index", .{dir}, 0), .{ .ACCMODE = .WRONLY, .APPEND = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.TestFile;
    defer _ = std.c.close(fd);
    try @import("prompt_imprint.zig").writeAll(fd, bytes);
}

test "learning after a torn index header or payload survives another reopen with both states" {
    const a = std.testing.allocator;
    const first = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const later = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18 };
    const record = [_]u32{ 0x494d5052, 2, 99, 0, 1, 2, 0, 100, 0, 31, 32 };
    for ([_]usize{ 1, 35, 36, 37, 43 }) |cut| {
        var tag: [32]u8 = undefined;
        var buf: [128]u8 = undefined;
        const r = try root(&buf, try std.fmt.bufPrint(&tag, "torn-{d}", .{cut}));
        defer @import("prompt_imprint_test.zig").rmTree(r);
        {
            var im = try Imprint.open(a, r, 1, 1 << 20);
            defer im.deinit();
            try learnOne(&im, &first);
            try appendIndex(im.dir, std.mem.sliceAsBytes(&record)[0..cut]);
        }
        {
            var im = try Imprint.open(a, r, 1, 1 << 20);
            defer im.deinit();
            try learnOne(&im, &later);
        }
        var im = try Imprint.open(a, r, 1, 1 << 20);
        defer im.deinit();
        try std.testing.expectEqual(@as(usize, 2), im.metas.items.len);
        try recallOne(&im, &first);
        try recallOne(&im, &later);
    }
}

test "invalid complete records salvage the valid prefix but unknown versions remain untouched" {
    const a = std.testing.allocator;
    const first = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    for ([_]u32{ 0, 3, (1 << 22) + 1 }) |bad| {
        var tag: [32]u8 = undefined;
        var buf: [128]u8 = undefined;
        const r = try root(&buf, try std.fmt.bufPrint(&tag, "corrupt-{d}", .{bad}));
        defer @import("prompt_imprint_test.zig").rmTree(r);
        var im = try Imprint.open(a, r, 1, 1 << 20);
        defer im.deinit();
        try learnOne(&im, &first);
        var record = [_]u32{ 0x494d5052, 2, 99, 0, 1, 2, 0, 100, 0, 31, 32 };
        if (bad == 0) record[0] = 0 else if (bad == 3) record[1] = 3 else record[5] = bad;
        try appendIndex(im.dir, std.mem.sliceAsBytes(&record));
        if (bad != 3) {
            var repaired = try Imprint.open(a, r, 1, 1 << 20);
            defer repaired.deinit();
            try std.testing.expectEqual(@as(usize, 1), repaired.metas.items.len);
            try std.testing.expectEqual(@as(u64, 60), try repaired.indexScratchBytes());
            try recallOne(&repaired, &first);
            continue;
        }
        if (Imprint.open(a, r, 1, 1 << 20)) |value| {
            var unexpected = value;
            unexpected.deinit();
            return error.FutureIndexAccepted;
        } else |err| try std.testing.expectEqual(error.ImprintRead, err);
        try std.testing.expectEqual(@as(u64, 104), try im.indexScratchBytes());
        var path: [1200]u8 = undefined;
        const fd = std.c.open(try std.fmt.bufPrintSentinel(&path, "{s}/index", .{im.dir}, 0), .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.TestFile;
        defer _ = std.c.close(fd);
        var actual: [44]u8 = undefined;
        try std.testing.expect(@import("prompt_imprint.zig").readAt(fd, &actual, 60));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&record), &actual);
    }
}

test "add repairs a live torn index before appending without requiring a caller reopen" {
    const a = std.testing.allocator;
    const first = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const later = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18 };
    const record = [_]u32{ 0x494d5052, 2, 99, 0, 1, 2, 0, 100, 0, 31, 32 };
    for ([_]usize{ 1, 35, 36, 37, 43 }) |cut| {
        var tag: [32]u8 = undefined;
        var buf: [128]u8 = undefined;
        const r = try root(&buf, try std.fmt.bufPrint(&tag, "live-{d}", .{cut}));
        defer @import("prompt_imprint_test.zig").rmTree(r);
        {
            var im = try Imprint.open(a, r, 1, 1 << 20);
            defer im.deinit();
            try learnOne(&im, &first);
            try appendIndex(im.dir, std.mem.sliceAsBytes(&record)[0..cut]);
            try learnOne(&im, &later);
            try std.testing.expectEqual(@as(u64, 120), try im.indexScratchBytes());
        }
        var im = try Imprint.open(a, r, 1, 1 << 20);
        defer im.deinit();
        try std.testing.expectEqual(@as(usize, 2), im.metas.items.len);
        try recallOne(&im, &first);
        try recallOne(&im, &later);
    }
}

test "add rejects invalid token keys and geometry before writing an index" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const r = try root(&buf, "invalid-add");
    defer @import("prompt_imprint_test.zig").rmTree(r);
    var im = try Imprint.open(a, r, 1, 1 << 20);
    defer im.deinit();
    const key = Imprint.keyOf(&.{ 1, 2 });
    try std.testing.expectError(error.ImprintWrite, im.add(key ^ 1, 1, &.{ 1, 2 }, &.{}, 100));
    try std.testing.expectError(error.ImprintWrite, im.add(key, 0, &.{ 1, 2 }, &.{}, 100));
    try std.testing.expectError(error.ImprintWrite, im.add(key, 3, &.{ 1, 2 }, &.{}, 100));
    try std.testing.expectError(error.ImprintWrite, im.add(key, 1, &.{ 1, 2 }, &.{ 2, 1 }, 100));
    try std.testing.expectEqual(@as(u64, 0), try im.indexScratchBytes());
    try std.testing.expectEqual(@as(usize, 0), im.metas.items.len);
}

test "a live unknown-version suffix refuses add without publishing a new record" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const r = try root(&buf, "live-version");
    defer @import("prompt_imprint_test.zig").rmTree(r);
    var im = try Imprint.open(a, r, 1, 1 << 20);
    defer im.deinit();
    const first = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try learnOne(&im, &first);
    const future = [_]u32{ 0x494d5052, 3, 99, 0, 1, 2, 0, 100, 0 };
    try appendIndex(im.dir, std.mem.sliceAsBytes(&future));
    try std.testing.expectError(error.ImprintRead, im.add(Imprint.keyOf(&.{ 31, 32 }), 1, &.{ 31, 32 }, &.{}, 100));
    try std.testing.expectEqual(@as(u64, 96), try im.indexScratchBytes());
    try std.testing.expectEqual(@as(usize, 1), im.metas.items.len);
}

test "short future-version prefixes preserve index bytes and payloads on open and live add" {
    const a = std.testing.allocator;
    const first = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const future = [_]u32{ 0x494d5052, 3, 99, 0, 1, 2, 0, 100, 0 };
    for ([_]bool{ false, true }) |prefix| for ([_]bool{ false, true }) |live| {
        for (8..36) |cut| {
            var tag: [40]u8 = undefined;
            var buf: [160]u8 = undefined;
            const r = try root(&buf, try std.fmt.bufPrint(&tag, "future-{any}-{any}-{d}", .{ prefix, live, cut }));
            defer @import("prompt_imprint_test.zig").rmTree(r);
            var im = try Imprint.open(a, r, 1, 1 << 20);
            defer im.deinit();
            if (prefix) try learnOne(&im, &first);
            var path: [1200]u8 = undefined;
            const fd = std.c.open(try std.fmt.bufPrintSentinel(&path, "{s}/index", .{im.dir}, 0), .{ .ACCMODE = .RDWR, .APPEND = true, .CREAT = true }, @as(std.c.mode_t, 0o600));
            if (fd < 0) return error.TestFile;
            defer _ = std.c.close(fd);
            try @import("prompt_imprint.zig").writeAll(fd, std.mem.sliceAsBytes(&future)[0..cut]);
            var expected: [96]u8 = undefined;
            const len = (if (prefix) @as(usize, 60) else 0) + cut;
            try std.testing.expect(@import("prompt_imprint.zig").readAt(fd, expected[0..len], 0));
            try put(im.dir, "00000000000000af.bin", 7);
            if (live) {
                try std.testing.expectError(error.ImprintRead, im.add(Imprint.keyOf(&.{ 31, 32 }), 1, &.{ 31, 32 }, &.{}, 100));
            } else if (Imprint.open(a, r, 1, 1 << 20)) |value| {
                var unexpected = value;
                unexpected.deinit();
                return error.FuturePrefixAccepted;
            } else |err| try std.testing.expectEqual(error.ImprintRead, err);
            try std.testing.expectEqual(@as(u64, @intCast(len)), try im.indexScratchBytes());
            var actual: [96]u8 = undefined;
            try std.testing.expect(@import("prompt_imprint.zig").readAt(fd, actual[0..len], 0));
            try std.testing.expectEqualSlices(u8, expected[0..len], actual[0..len]);
            const payload = std.c.open(try std.fmt.bufPrintSentinel(&path, "{s}/00000000000000af.bin", .{im.dir}, 0), .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
            if (payload < 0) return error.FuturePayloadRemoved;
            _ = std.c.close(payload);
            if (prefix) {
                const kept = std.c.open(try Fake.file(&path, im.dir, Imprint.keyOf(first[0..6])), .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
                if (kept < 0) return error.ValidPrefixPayloadRemoved;
                _ = std.c.close(kept);
            }
        }
    };
}
test "a null clock cannot authorize eviction or either half of a write" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const r = try root(&buf, "clock");
    defer @import("prompt_imprint_test.zig").rmTree(r);
    var im = try Imprint.open(a, r, 1, 440);
    defer im.deinit();
    try im.add(Imprint.keyOf(&.{ 1, 2 }), 1, &.{ 1, 2 }, &.{}, 105);
    im.admission = .{ .floor = 0, .free = Disk.free, .clock = Disk.clock };
    Disk.tick = null;
    defer Disk.tick = 100;
    var f: Fake = .{ .gpa = a, .at = 5 };
    var s = pc.Store.init(a, f.paired(), .{ .min_prompt = 0, .min_gap = 1, .lookahead = 1 }, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const prompt = [_]u32{ 7, 7, 7, 7, 7, 7, 9, 10 };
    _ = try s.lookup(arena.allocator(), &prompt, 7, &.{5}, &.{}, &.{});
    try std.testing.expect(s.keep(&prompt, 5, null, &.{}, &.{}));
    try std.testing.expect(im.has(Imprint.keyOf(&.{ 1, 2 })));
    try std.testing.expectEqual(@as(usize, 0), f.disk_forgets);
    try std.testing.expectEqual(@as(usize, 0), f.writes);
}
test "an index replacement failure retains its metadata and removes its partial file" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const r = try root(&buf, "index");
    defer @import("prompt_imprint_test.zig").rmTree(r);
    var im = try Imprint.open(a, r, 1, 1 << 20);
    defer im.deinit();
    try im.add(Imprint.keyOf(&.{ 1, 2 }), 1, &.{ 1, 2 }, &.{}, 105);
    var path: [1200]u8 = undefined;
    var idx_buf: [1200]u8 = undefined;
    const idx = try std.fmt.bufPrintSentinel(&idx_buf, "{s}/index", .{im.dir}, 0);
    try dirs.unlink(idx);
    try std.testing.expectEqual(@as(c_int, 0), std.c.mkdir(idx, 0o700));
    defer _ = std.c.rmdir(idx); // the tree removal goes two levels down
    try std.testing.expectError(error.ImprintWrite, im.remove(Imprint.keyOf(&.{ 1, 2 })));
    try std.testing.expect(im.has(Imprint.keyOf(&.{ 1, 2 })));
    const part = try std.fmt.bufPrintSentinel(&path, "{s}/index.part", .{im.dir}, 0);
    const fd = std.c.open(part, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd >= 0) {
        _ = std.c.close(fd);
        return error.PartRemains;
    }
}
test "other-identity cap pressure is planned before a current-identity write" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const r = try root(&buf, "others");
    defer @import("prompt_imprint_test.zig").rmTree(r);
    {
        var old = try Imprint.open(a, r, 1, 1 << 20);
        defer old.deinit();
        try put(old.dir, "state.bin", 300);
    }
    var im = try Imprint.open(a, r, 2, 440);
    defer im.deinit();
    try std.testing.expectEqual(@as(u64, 308), im.others);
    im.admission = .{ .floor = 0, .free = Disk.free, .clock = Disk.clock };
    var f: Fake = .{ .gpa = a, .at = 5 };
    var s = pc.Store.init(a, f.learned(), .{ .min_prompt = 0, .min_gap = 1, .lookahead = 1 }, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const prompt = [_]u32{ 7, 7, 7, 7, 7, 7, 9, 10 };
    _ = try s.lookup(arena.allocator(), &prompt, 7, &.{5}, &.{}, &.{});
    try std.testing.expect(s.keep(&prompt, 5, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(u64, 0), try dirs.otherBytes(r, im.dir, 1));
    try std.testing.expectEqual(@as(usize, 1), f.writes);
    try std.testing.expect(im.has(Imprint.keyOf(prompt[0..6])));
}
test "unlink failure is observable and missing files are idempotently removed" {
    var buf: [128]u8 = undefined;
    const r = try root(&buf, "unlink");
    var path: [150]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&path, "{s}", .{r}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.mkdir(dir, 0o700));
    defer _ = std.c.rmdir(dir);
    try std.testing.expectError(error.LearnedUnlink, dirs.unlink(dir));
    var absent: [180]u8 = undefined;
    try dirs.unlink(try std.fmt.bufPrintSentinel(&absent, "{s}/absent", .{dir}, 0));
}
test "disk admission releases reservations and bounds repeated failure retries" {
    var admission: Admission = .{ .floor = 0, .free = Disk.free, .clock = Disk.clock };
    try std.testing.expectEqual(Admission.Result.ready, admission.reserve(".", 100));
    admission.finish(".", 100, false);
    try std.testing.expectEqual(Admission.Result.quiet, admission.reserve(".", 100));
    for (0..20) |_| {
        Disk.tick.? += 100;
        _ = admission.reserve(".", std.math.maxInt(u64));
    }
    try std.testing.expectEqual(Disk.tick.? + 60, admission.retry_at);
    try std.testing.expectEqual(@as(u64, 0), admission.reserved);
    Disk.tick = 100;
}

test "a missing indexed half cannot promise physical reclaim or destroy a valid victim" {
    const Space = struct {
        var available: u64 = 0;
        fn free(_: [:0]const u8) ?u64 {
            return available;
        }
    };
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const r = try root(&buf, "missing");
    defer @import("prompt_imprint_test.zig").rmTree(r);
    var im = try Imprint.open(a, r, 1, 1 << 20);
    defer im.deinit();
    try im.add(Imprint.keyOf(&.{ 1, 2 }), 1, &.{ 1, 2 }, &.{}, 100);
    try im.add(Imprint.keyOf(&.{ 3, 4 }), 1, &.{ 3, 4 }, &.{}, 100);
    var filename: [32]u8 = undefined;
    try put(im.dir, try std.fmt.bufPrint(&filename, "{x:0>16}.bin", .{Imprint.keyOf(&.{ 3, 4 })}), 100);
    im.admission = .{ .floor = 100, .free = Space.free, .clock = Disk.clock };
    Space.available = 100 + 105 + 256 + 24 + (try im.indexScratchBytes()) - 150;
    var f: Fake = .{ .gpa = a, .at = 5 };
    var s = pc.Store.init(a, f.paired(), .{ .min_prompt = 0, .min_gap = 1, .lookahead = 1 }, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const prompt = [_]u32{ 7, 7, 7, 7, 7, 7, 9, 10 };
    _ = try s.lookup(arena.allocator(), &prompt, 7, &.{5}, &.{}, &.{});
    try std.testing.expect(s.keep(&prompt, 5, null, &.{}, &.{}));
    try std.testing.expect(im.has(Imprint.keyOf(&.{ 1, 2 })) and im.has(Imprint.keyOf(&.{ 3, 4 })));
    try std.testing.expectEqual(@as(usize, 0), f.disk_forgets);
    try std.testing.expectEqual(@as(usize, 0), f.writes);
    try std.testing.expectEqual(@as(u64, 100), try pc.singleReclaim(&f, im.dir, Imprint.keyOf(&.{ 3, 4 })));
}
test "uncertain reserve ACK reaches Store rollback before any intent or half write" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const r = try root(&buf, "ack");
    defer @import("prompt_imprint_test.zig").rmTree(r);
    var im = try Imprint.open(a, r, 1, 1 << 20);
    defer im.deinit();
    im.admission = .{ .floor = 0, .free = Disk.free, .clock = Disk.clock };
    var f: Fake = .{ .gpa = a, .at = 5, .disk_fail_reserve_ack = true };
    var s = pc.Store.init(a, f.paired(), .{ .min_prompt = 0, .min_gap = 1, .lookahead = 1 }, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const prompt = [_]u32{ 7, 7, 7, 7, 7, 7, 9, 10 };
    _ = try s.lookup(arena.allocator(), &prompt, 7, &.{5}, &.{}, &.{});
    try std.testing.expect(s.keep(&prompt, 5, null, &.{}, &.{}));
    try std.testing.expect(!f.disk_reserved);
    try std.testing.expectEqual(@as(usize, 1), f.disk_finishes);
    try std.testing.expectEqual(@as(usize, 0), f.writes);
    try std.testing.expect((try im.pendingWrite()) == null);
}
test "failed rollback deletion survives restart and recovers before cap admission" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const r = try root(&buf, "pending");
    defer @import("prompt_imprint_test.zig").rmTree(r);
    const prompt = [_]u32{ 7, 7, 7, 7, 7, 7, 9, 10 };
    for (0..2) |round| {
        var im = try Imprint.open(a, r, 1, 400);
        defer im.deinit();
        im.admission = .{ .floor = 0, .free = Disk.free, .clock = Disk.clock };
        var f: Fake = .{ .gpa = a, .at = 5, .disk_fail_peer = round == 0, .disk_fail_delete = round == 0 };
        var s = pc.Store.init(a, f.paired(), .{ .min_prompt = 0, .min_gap = 1, .lookahead = 1 }, 1 << 20);
        defer s.deinit();
        s.imprint = &im;
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        _ = try s.lookup(arena.allocator(), &prompt, 7, &.{5}, &.{}, &.{});
        try std.testing.expect(s.keep(&prompt, 5, null, &.{}, &.{}));
        const key = Imprint.keyOf(prompt[0..6]);
        try std.testing.expectEqual(@as(usize, 1), f.writes);
        if (round == 0) {
            try std.testing.expectEqual(key, (try im.pendingWrite()).?);
            try std.testing.expect(!im.has(key));
        } else {
            try std.testing.expect((try im.pendingWrite()) == null);
            try std.testing.expect(im.has(key));
        }
        try std.testing.expect(!f.disk_reserved);
        try std.testing.expectEqual(@as(u64, 0), im.admission.reserved);
    }
}

test "Store learns after old zero-byte and torn used stamps under cap pressure" {
    const a = std.testing.allocator;
    for (0..8) |length| {
        var tag: [32]u8 = undefined;
        var buf: [128]u8 = undefined;
        const r = try root(&buf, try std.fmt.bufPrint(&tag, "used-{d}", .{length}));
        defer @import("prompt_imprint_test.zig").rmTree(r);
        {
            var old = try Imprint.open(a, r, 1, 1 << 20);
            defer old.deinit();
            try put(old.dir, "state.bin", 300);
            try put(old.dir, "used", length);
        }
        var im = try Imprint.open(a, r, 2, 440);
        defer im.deinit();
        im.admission = .{ .floor = 0, .free = Disk.free, .clock = Disk.clock };
        var f: Fake = .{ .gpa = a, .at = 5 };
        var s = pc.Store.init(a, f.learned(), .{ .min_prompt = 0, .min_gap = 1, .lookahead = 1 }, 1 << 20);
        defer s.deinit();
        s.imprint = &im;
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const prompt = [_]u32{ 7, 7, 7, 7, 7, 7, 9, 10 };
        _ = try s.lookup(arena.allocator(), &prompt, 7, &.{5}, &.{}, &.{});
        try std.testing.expect(s.keep(&prompt, 5, null, &.{}, &.{}));
        try std.testing.expectEqual(@as(usize, 1), f.writes);
        try std.testing.expect(im.has(Imprint.keyOf(prompt[0..6])));
        try std.testing.expectEqual(@as(u64, 0), try dirs.otherBytes(r, im.dir, 1));
    }
}
