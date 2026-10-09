const std = @import("std");
const Imprint = @import("prompt_imprint.zig").Imprint;
const imp = @import("prompt_imprint.zig");
const dirs = @import("lanes").learned_dirs;
const Allocator = std.mem.Allocator;

const Tracked = struct {
    storage: [16384]u8 = undefined,
    backing: std.heap.FixedBufferAllocator = undefined,
    records: [64]struct { address: usize, live: bool } = undefined,
    count: usize = 0,
    duplicate_frees: usize = 0,

    fn allocator(t: *Tracked) Allocator {
        t.backing = .init(&t.storage);
        return .{ .ptr = t, .vtable = &.{ .alloc = alloc, .resize = Allocator.noResize, .remap = Allocator.noRemap, .free = free } };
    }
    fn alloc(ptr: *anyopaque, len: usize, alignment: std.mem.Alignment, ret: usize) ?[*]u8 {
        const t: *Tracked = @ptrCast(@alignCast(ptr));
        if (t.count == t.records.len) return null;
        const memory = t.backing.allocator().rawAlloc(len, alignment, ret) orelse return null;
        t.records[t.count] = .{ .address = @intFromPtr(memory), .live = true };
        t.count += 1;
        return memory;
    }
    fn free(ptr: *anyopaque, memory: []u8, _: std.mem.Alignment, _: usize) void {
        const t: *Tracked = @ptrCast(@alignCast(ptr));
        for (t.records[0..t.count]) |*record| if (record.address == @intFromPtr(memory.ptr)) {
            if (!record.live) t.duplicate_frees += 1;
            record.live = false;
            return;
        };
        unreachable;
    }
    fn outstanding(t: *const Tracked) usize {
        var n: usize = 0;
        for (t.records[0..t.count]) |record| if (record.live) {
            n += 1;
        };
        return n;
    }
};
fn scratch(buf: []u8, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-imprint-fault-{s}-{d}", .{ suffix, std.c.getpid() });
}
fn word(path: [:0]const u8, n: u64) !void {
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.TestFile;
    defer _ = std.c.close(fd);
    try imp.writeAll(fd, std.mem.asBytes(&n));
}
test "Imprint failed open releases each allocation once after another identity scan" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "ownership");
    defer @import("prompt_imprint_test.zig").rmTree(root);
    var old = try Imprint.open(a, root, 1, 1 << 20);
    defer old.deinit();
    var path: [256]u8 = undefined;
    const unreadable = try std.fmt.bufPrintSentinel(&path, "{s}/unreadable", .{old.dir}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.mkdir(unreadable, 0o700));
    defer _ = std.c.rmdir(unreadable);
    var tracked: Tracked = .{};
    try std.testing.expectError(error.ImprintRead, Imprint.open(tracked.allocator(), root, 2, 1 << 20));
    try std.testing.expectEqual(@as(usize, 0), tracked.duplicate_frees);
    try std.testing.expectEqual(@as(usize, 0), tracked.outstanding());
}
test "Imprint allocation failures keep transferred path ownership singular" {
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "allocations");
    defer @import("prompt_imprint_test.zig").rmTree(root);
    {
        var seeded = try Imprint.open(std.testing.allocator, root, 1, 1 << 20);
        defer seeded.deinit();
        try seeded.add(Imprint.keyOf(&.{ 1, 2 }), 1, &.{ 1, 2 }, &.{}, 100);
    }
    const Trial = struct {
        fn run(a: Allocator, r: []const u8) !void {
            var m = try Imprint.open(a, r, 1, 1 << 20);
            defer m.deinit();
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Trial.run, .{root});
}
test "failed atomic stamp replacement preserves the previous complete used word" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "stamp");
    defer @import("prompt_imprint_test.zig").rmTree(root);
    var m = try Imprint.open(a, root, 1, 1 << 20);
    defer m.deinit();
    var path: [256]u8 = undefined;
    try word(try std.fmt.bufPrintSentinel(&path, "{s}/used", .{m.dir}, 0), 42);
    const part = try std.fmt.bufPrintSentinel(&path, "{s}/used.part", .{m.dir}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.mkdir(part, 0o700));
    defer _ = std.c.rmdir(part);
    var reopened = try Imprint.open(a, root, 1, 1 << 20);
    defer reopened.deinit();
    try std.testing.expectEqual(@as(u64, 42), try dirs.used(root, 1));
}
test "used stamp reads still refuse an actual I/O error" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "io");
    defer @import("prompt_imprint_test.zig").rmTree(root);
    var m = try Imprint.open(a, root, 1, 1 << 20);
    defer m.deinit();
    var path: [256]u8 = undefined;
    const used = try std.fmt.bufPrintSentinel(&path, "{s}/used", .{m.dir}, 0);
    try dirs.unlink(used);
    try std.testing.expectEqual(@as(c_int, 0), std.c.mkdir(used, 0o700));
    defer _ = std.c.rmdir(used);
    try std.testing.expectError(error.LearnedDirectoryRead, dirs.used(root, 1));
}
