//! The prompt cache's learned-state tests: harness states kept on disk, the cap, the disk floor and peers' halves.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const imprint = @import("prompt_imprint.zig");
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const Store = pc.Store;
const Allocator = std.mem.Allocator;
const Fake = @import("prompt_cache_fake.zig").Fake;
const fresh = @import("prompt_cache_test.zig").fresh;

test "a learned harness state outlives its store: a fresh session on a new one resumes it from disk, equal to a fresh pass" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var root_buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&root_buf, "/tmp/tf-learn-test-{d}", .{std.c.getpid()});
    const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };
    const harness = [_]u32{ 7, 7, 7, 7, 7, 7 }; // shared by every session: a state at 5 reads it all (lookahead 1)
    const key = imprint.Imprint.keyOf(&harness);
    defer rmTree(root);
    {
        var im = try imprint.Imprint.open(gpa, root, 3, 1 << 30);
        defer im.deinit();
        {
            var f: Fake = .{ .gpa = gpa };
            var s = Store.init(gpa, f.learned(), rules, 1 << 20);
            defer s.deinit();
            s.imprint = &im;
            const first = harness ++ [_]u32{ 1, 2, 9 };
            try std.testing.expectEqual(fresh(&first), f.pass(&s, &first, try s.begin(a, &first, 8, &.{5}, &.{}, null, &.{})));
            try std.testing.expect(im.has(key));
        }
        var again = try imprint.Imprint.open(gpa, root, 3, 1 << 30); // a new server reads the index back
        defer again.deinit();
        var f: Fake = .{ .gpa = gpa };
        var s = Store.init(gpa, f.learned(), rules, 1 << 20);
        defer s.deinit();
        s.imprint = &again;
        const second = harness ++ [_]u32{ 3, 4, 9 };
        const p = try s.begin(a, &second, 8, &.{5}, &.{}, null, &.{});
        try std.testing.expectEqual(@as(u32, 5), p.from);
        try std.testing.expectEqual(fresh(&second), f.pass(&s, &second, p));
    }
}

test "past the learned-state cap the least recently used state is forgotten, and a file that no longer reads is too" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var root_buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&root_buf, "/tmp/tf-learn-cap-{d}", .{std.c.getpid()});
    defer rmTree(root);
    const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };
    const one = [_]u32{ 7, 7, 7, 7, 7, 7 };
    const two = [_]u32{ 8, 8, 8, 8, 8, 8 };
    var im = try imprint.Imprint.open(gpa, root, 3, 440); // One state plus conservative index/header allowance fits.
    defer im.deinit();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.learned(), rules, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    const p1 = one ++ [_]u32{ 1, 9 };
    try std.testing.expectEqual(fresh(&p1), f.pass(&s, &p1, try s.begin(a, &p1, 7, &.{5}, &.{}, null, &.{})));
    const p2 = two ++ [_]u32{ 1, 9 };
    try std.testing.expectEqual(fresh(&p2), f.pass(&s, &p2, try s.begin(a, &p2, 7, &.{5}, &.{}, null, &.{})));
    try std.testing.expect(im.has(imprint.Imprint.keyOf(&two)) and !im.has(imprint.Imprint.keyOf(&one)));
    var path: [512]u8 = undefined;
    try std.testing.expect(std.c.unlink(try Fake.file(&path, im.dir, imprint.Imprint.keyOf(&one))) != 0); // its file went too
    _ = std.c.unlink(try Fake.file(&path, im.dir, imprint.Imprint.keyOf(&two))); // a file lost behind the store's back
    var g: Fake = .{ .gpa = gpa };
    var t = Store.init(gpa, g.learned(), rules, 1 << 20);
    defer t.deinit();
    t.imprint = &im;
    const p3 = two ++ [_]u32{ 2, 9 };
    const plan = try t.begin(a, &p3, 7, &.{5}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 0), plan.from); // the read failed: a fresh pass, and the state is forgotten
    try std.testing.expect(!im.has(imprint.Imprint.keyOf(&two)));
    try std.testing.expectEqual(fresh(&p3), g.pass(&t, &p3, plan));
    try std.testing.expect(im.has(imprint.Imprint.keyOf(&two))); // learned again by that pass
}

test "learned states are prompt arithmetic: a request that decodes rows below one prefills instead, and a decoded state is never learned" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var root_buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&root_buf, "/tmp/tf-learn-spans-{d}", .{std.c.getpid()});
    defer rmTree(root);
    const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };
    const harness = [_]u32{ 7, 7, 7, 7, 7, 7 };
    var im = try imprint.Imprint.open(gpa, root, 3, 1 << 30);
    defer im.deinit();
    {
        var f: Fake = .{ .gpa = gpa };
        var s = Store.init(gpa, f.learned(), rules, 1 << 20);
        defer s.deinit();
        s.imprint = &im;
        const first = harness ++ [_]u32{ 1, 2, 9 };
        _ = f.pass(&s, &first, try s.begin(a, &first, 8, &.{5}, &.{}, null, &.{}));
        try std.testing.expect(im.has(imprint.Imprint.keyOf(&harness)));
    }
    const second = harness ++ [_]u32{ 3, 4, 9 };
    for ([_]struct { spans: []const [2]u32, from: u32 }{ .{ .spans = &.{.{ 2, 4 }}, .from = 0 }, .{ .spans = &.{.{ 6, 8 }}, .from = 5 } }) |c| {
        var f: Fake = .{ .gpa = gpa };
        var s = Store.init(gpa, f.learned(), rules, 1 << 20);
        defer s.deinit();
        s.imprint = &im;
        const p = try s.begin(a, &second, 8, &.{5}, &.{}, null, c.spans);
        try std.testing.expectEqual(c.from, p.from);
    }
    const other = [_]u32{ 8, 8, 8, 8, 8, 8, 1, 9 };
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.learned(), rules, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    const p = try s.begin(a, &other, 7, &.{5}, &.{}, null, &.{.{ 1, 3 }});
    try std.testing.expectEqual(@as(u32, 0), p.from);
    f.at = 5;
    try std.testing.expect(s.keep(&other, 5, null, &.{}, &.{.{ 1, 3 }}));
    try std.testing.expect(!im.has(imprint.Imprint.keyOf(other[0..6])));
}

test "Store learning preserves the disk floor, backs off writes across keys, and recovers without retained failures" {
    const Disk = struct {
        var space: ?u64 = 0;
        var tick: u64 = 100;
        fn free(_: [:0]const u8) ?u64 {
            return space;
        }
        fn clock() ?u64 {
            return tick;
        }
    };
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var path: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&path, "/tmp/tf-learn-backoff-{d}", .{std.c.getpid()});
    defer rmTree(root);
    var im = try imprint.Imprint.open(a, root, 11, 1 << 20);
    defer im.deinit();
    im.admission = .{ .floor = 100, .free = Disk.free, .clock = Disk.clock };
    var fake: Fake = .{ .gpa = a };
    var store = Store.init(a, fake.learned(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 }, 1 << 20);
    defer store.deinit();
    store.imprint = &im;
    for (0..6) |i| {
        const token: u32 = @intCast(i + 1);
        const prompt = [_]u32{ token, token, token, token, token, token, 9, 10 };
        if (i == 2) {
            Disk.space = 10000;
            fake.fail_write = true;
        }
        if (i == 5) {
            Disk.tick += 100;
            fake.fail_write = false;
        }
        const plan = try store.begin(arena.allocator(), &prompt, 7, &.{5}, &.{}, null, &.{});
        try std.testing.expectEqual(fresh(&prompt), fake.pass(&store, &prompt, plan));
        try std.testing.expectEqual(@as(usize, if (i < 2) 0 else if (i < 5) 1 else 2), fake.writes);
    }
    try std.testing.expectEqual(@as(usize, 1), im.metas.items.len);
    try std.testing.expectEqual(@as(u64, 0), im.admission.reserved);
}

test "learning reserves the existing index rewrite above the disk floor and resumes an unchanged kept key" {
    const Disk = struct {
        var space: u64 = 485;
        fn free(_: [:0]const u8) ?u64 {
            return space;
        }
        fn clock() ?u64 {
            return 100;
        }
    };
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var path: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&path, "/tmp/tf-learn-index-floor-{d}", .{std.c.getpid()});
    defer rmTree(root);
    var im = try imprint.Imprint.open(a, root, 12, 1 << 20);
    defer im.deinit();
    try im.add(imprint.Imprint.keyOf(&.{ 1, 2 }), 1, &.{ 1, 2 }, &.{}, 1);
    im.admission = .{ .floor = 100, .free = Disk.free, .clock = Disk.clock };
    var fake: Fake = .{ .gpa = a };
    var store = Store.init(a, fake.learned(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 }, 1 << 20);
    defer store.deinit();
    store.imprint = &im;
    const prompt = [_]u32{ 7, 7, 7, 7, 7, 7, 9, 10 };
    const plan = try store.begin(arena.allocator(), &prompt, 7, &.{5}, &.{}, null, &.{});
    _ = fake.pass(&store, &prompt, plan);
    try std.testing.expectEqual(@as(usize, 0), fake.writes);
    Disk.space += try im.indexScratchBytes();
    fake.at = 5;
    try std.testing.expect(store.keep(&prompt, 5, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 1), fake.writes);
    try std.testing.expectEqual(@as(u64, 0), im.admission.reserved);
}

test "actual Store learn writes neither half on peer refusal, backs off, and removes both halves after a peer fault" {
    const Disk = struct {
        var tick: u64 = 100;
        fn free(_: [:0]const u8) ?u64 {
            return 10000;
        }
        fn clock() ?u64 {
            return tick;
        }
    };
    const a = std.testing.allocator;
    var path: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&path, "/tmp/tf-pair-refuse-{d}", .{std.c.getpid()});
    defer rmTree(root);
    var im = try imprint.Imprint.open(a, root, 51, 1 << 20);
    defer im.deinit();
    im.admission = .{ .floor = 100, .free = Disk.free, .clock = Disk.clock };
    var fake: Fake = .{ .gpa = a, .at = 5, .disk_need_bytes = 1 };
    var store = Store.init(a, fake.paired(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 }, 1 << 20);
    defer store.deinit();
    store.imprint = &im;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    _ = try store.lookup(arena.allocator(), &prompt, 7, &.{5}, &.{}, &.{});
    for (0..32) |_| try std.testing.expect(store.keep(&prompt, 5, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 0), fake.writes);
    try std.testing.expectEqual(@as(usize, 0), fake.disk_peer_writes);
    try std.testing.expectEqual(@as(usize, 1), fake.disk_queries);
    Disk.tick += 100;
    fake.disk_need_bytes = 0;
    fake.disk_fail_peer = true;
    try std.testing.expect(store.keep(&prompt, 5, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 1), fake.writes);
    try std.testing.expectEqual(@as(usize, 1), fake.disk_peer_writes);
    try std.testing.expect(!fake.disk_reserved);
    try std.testing.expectEqual(@as(usize, 1), fake.disk_finishes);
    try std.testing.expect(!im.has(imprint.Imprint.keyOf(prompt[0..6])));
    try std.testing.expectEqual(@as(u64, 0), im.admission.reserved);
    var file_path: [512]u8 = undefined;
    const file = try Fake.file(&file_path, im.dir, imprint.Imprint.keyOf(prompt[0..6]));
    const fd = std.c.open(file, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd >= 0) {
        _ = std.c.close(fd);
        return error.OrphanedHalf;
    }
}
test "under-cap disk pressure selects the same LRU victim before either half is persisted" {
    const Disk = struct {
        var space: u64 = 0;
        fn free(_: [:0]const u8) ?u64 {
            return space;
        }
        fn clock() ?u64 {
            return 100;
        }
        fn release() void {
            space += 16;
        }
    };
    const a = std.testing.allocator;
    var path: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&path, "/tmp/tf-pair-victim-{d}", .{std.c.getpid()});
    defer rmTree(root);
    var im = try imprint.Imprint.open(a, root, 52, 1 << 20);
    defer im.deinit();
    try im.add(imprint.Imprint.keyOf(&.{ 1, 2 }), 1, &.{ 1, 2 }, &.{}, 16);
    var victim_path: [512]u8 = undefined;
    const victim_fd = std.c.open(try Fake.file(&victim_path, im.dir, imprint.Imprint.keyOf(&.{ 1, 2 })), .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(std.c.mode_t, 0o600));
    if (victim_fd < 0) return error.Create;
    try std.testing.expectEqual(@as(c_int, 0), std.c.ftruncate(victim_fd, 16));
    _ = std.c.close(victim_fd);
    im.admission = .{ .floor = 100, .free = Disk.free, .clock = Disk.clock };
    const prompt = [_]u32{ 7, 7, 7, 7, 7, 7, 9, 10 };
    Disk.space = 100 + 105 + 256 + (6 * 4) + (try im.indexScratchBytes()) - 8;
    var fake: Fake = .{ .gpa = a, .at = 5, .disk_need_bytes = 8, .disk_release = Disk.release };
    var store = Store.init(a, fake.paired(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 }, 1 << 20);
    defer store.deinit();
    store.imprint = &im;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    _ = try store.lookup(arena.allocator(), &prompt, 7, &.{5}, &.{}, &.{});
    try std.testing.expect(store.keep(&prompt, 5, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 1), fake.disk_forgets);
    try std.testing.expect(!im.has(imprint.Imprint.keyOf(&.{ 1, 2 })));
    try std.testing.expectEqual(@as(usize, 1), fake.writes);
    try std.testing.expectEqual(@as(usize, 1), fake.disk_peer_writes);
    try std.testing.expect(!fake.disk_reserved);
}
