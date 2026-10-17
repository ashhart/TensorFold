//! Checkpoint reads that skip the page cache and pace on the memory the kernel can give: a load never outruns reclaim.
const std = @import("std");
const platform = @import("sys");

/// Offsets, lengths and buffers of uncached reads align to the VM page (F_NOCACHE only skips the cache for aligned reads).
pub const alignment = 16384;

/// How long a read waits for room before the load is refused; memory freed within it lets the load go on.
pub const wait_ms: u32 = 30_000;

/// A reading thread's aligned staging buffer.
pub const Stage = []align(alignment) u8;

/// A thread's staging buffer of `bytes` (a multiple of `alignment`).
pub fn stage(gpa: std.mem.Allocator, bytes: usize) !Stage {
    return gpa.alignedAlloc(u8, .fromByteUnits(alignment), bytes);
}

/// Bytes as the paced reads count them: free pages, and pages the kernel drops unwritten (file cache, purgeable).
pub const Memory = struct {
    free: u64,
    reclaimable: u64,

    /// Room for the next read: with the page cache full, free pages sit near the kernel's target, far under the floor.
    pub fn room(m: Memory) u64 {
        return m.free +| m.reclaimable;
    }
};

/// The room kept before a read goes ahead on a Mac with `ram` bytes: 3% of RAM, at least 8 GiB.
pub fn floorFor(ram: u64) u64 {
    return @max(8 << 30, ram / 100 * 3);
}

/// This Mac's floor.
pub fn floor() u64 {
    return floorFor(sysctl("hw.memsize") orelse 0);
}

/// Free pages alone, in bytes (the page-cache drop's measure); unreadable: the most.
pub fn freeBytes() u64 {
    const pages = sysctl("vm.page_free_count") orelse return std.math.maxInt(u64);
    return pages * pageBytes();
}

/// This Mac's counters now; null where they can't be read (the reads then go on unpaced).
pub fn memory() ?Memory {
    const page = pageBytes();
    const free = sysctl("vm.page_free_count") orelse return null;
    const file = sysctl("vm.page_pageable_external_count") orelse return null;
    const purgeable = sysctl("vm.page_purgeable_count") orelse 0;
    return .{ .free = free * page, .reclaimable = (file + purgeable) * page };
}

fn pageBytes() u64 {
    return sysctl("hw.pagesize") orelse alignment;
}

/// A counter of 4 or 8 bytes (macOS 27 keeps vm.page_purgeable_count in 8, the page counts in 4).
fn sysctl(comptime name: [:0]const u8) ?u64 {
    var v: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (platform.c.sysctlbyname(name, &v, &len, null, 0) != 0) return null;
    return switch (len) {
        8 => v,
        4 => @as(u32, @truncate(v)),
        else => null,
    };
}

/// Waits while `gauge`'s room is under `want`; still short after `limit_ms`: error.MemoryShort, its reading in `short`.
pub fn waitFor(gauge: anytype, want: u64, limit_ms: u32, short: *Memory) error{MemoryShort}!void {
    var waited: u32 = 0;
    while (gauge.reading()) |m| : (waited += 2) {
        if (m.room() >= want) return;
        if (waited >= limit_ms) {
            short.* = m;
            return error.MemoryShort;
        }
        gauge.pause(2);
    }
}

/// The refusal's sentence for a reading that stayed short: its numbers, the floor and the wait.
pub fn refusal(buf: []u8, m: Memory, want: u64) []const u8 {
    const gb = struct {
        fn of(bytes: u64) f64 {
            return @as(f64, @floatFromInt(bytes)) / 1e9;
        }
    }.of;
    return std.fmt.bufPrint(buf, "the load stopped after {d} s short of memory: {d:.1} GB free and {d:.1} GB of file cache the kernel can drop, under the {d:.1} GB a load leaves the system (3% of RAM, at least 8 GiB). Stop what else holds memory, then load again", .{ wait_ms / 1000, gb(m.free), gb(m.reclaimable), gb(want) }) catch "the load stopped short of memory";
}

/// The paced reads' gauge: this Mac's counters, and a real sleep.
const Mac = struct {
    fn reading(_: Mac) ?Memory {
        return memory();
    }

    fn pause(_: Mac, ms: u32) void {
        const ts: platform.c.timespec = .{ .sec = 0, .nsec = @intCast(@as(u64, ms) * std.time.ns_per_ms) };
        _ = platform.c.nanosleep(&ts, null);
    }
};

/// `dest.len` bytes at `at` of `fd` (F_NOCACHE) through page-aligned reads into `buf`, each once there is room for it.
pub fn read(fd: platform.c.fd_t, dest: []u8, at: u64, buf: Stage, short: *Memory) !void {
    const want = floor();
    var done: usize = 0;
    while (done < dest.len) {
        try waitFor(Mac{}, want, wait_ms, short);
        const pos = at + done;
        const lo = std.mem.alignBackward(u64, pos, alignment);
        const skip: usize = @intCast(pos - lo);
        const take = @min(dest.len - done, buf.len - skip);
        const span = std.mem.alignForward(usize, skip + take, alignment);
        var got: usize = 0;
        while (got < skip + take) { // a read past the file's end comes back short once the bytes asked for are in
            const n = platform.c.pread(fd, buf.ptr + got, span - got, @intCast(lo + got));
            if (n <= 0) return error.ShortRead;
            got += @intCast(n);
        }
        @memcpy(dest[done..][0..take], buf[skip..][0..take]);
        done += take;
    }
}

/// A Mac's counters played back one reading a poll (the last one repeats), on a clock that sleeps nothing.
const Played = struct {
    readings: []const ?Memory,
    polls: usize = 0,
    slept_ms: u64 = 0,

    fn reading(p: *Played) ?Memory {
        defer p.polls += 1;
        return p.readings[@min(p.polls, p.readings.len - 1)];
    }

    fn pause(p: *Played, ms: u32) void {
        p.slept_ms += ms;
    }
};

test "a warm page cache is room to read into; memory short past the wait refuses the load with its reading" {
    const gib: u64 = 1 << 30;
    const want = floorFor(256 * gib); // 8 GiB on a 256 GiB Mac, where 3% of RAM is less
    try std.testing.expectEqual(@as(u64, 8 << 30), want);
    var short: Memory = .{ .free = 0, .reclaimable = 0 };
    var warm: Played = .{ .readings = &.{.{ .free = 6 * gib, .reclaimable = 150 * gib }} }; // the stalled whole-GLM load
    try waitFor(&warm, want, wait_ms, &short);
    try std.testing.expectEqual(@as(u64, 0), warm.slept_ms);
    var freed: Played = .{ .readings = &.{ .{ .free = gib, .reclaimable = gib }, .{ .free = gib, .reclaimable = gib }, .{ .free = 9 * gib, .reclaimable = 0 } } };
    try waitFor(&freed, want, wait_ms, &short); // another process let go within the wait
    try std.testing.expectEqual(@as(u64, 4), freed.slept_ms);
    var full: Played = .{ .readings = &.{.{ .free = gib, .reclaimable = 2 * gib }} };
    try std.testing.expectError(error.MemoryShort, waitFor(&full, want, wait_ms, &short));
    try std.testing.expectEqual(@as(u64, wait_ms), full.slept_ms);
    try std.testing.expectEqual(Memory{ .free = gib, .reclaimable = 2 * gib }, short);
    var buf: [400]u8 = undefined;
    try std.testing.expectEqualStrings("the load stopped after 30 s short of memory: 1.1 GB free and 2.1 GB of file cache the kernel can drop, under the 8.6 GB a load leaves the system (3% of RAM, at least 8 GiB). Stop what else holds memory, then load again", refusal(&buf, short, want));
    var unread: Played = .{ .readings = &.{null} };
    try waitFor(&unread, want, wait_ms, &short); // counters unreadable: unpaced, as before
    try std.testing.expectEqual(@as(u64, 0), unread.slept_ms);
}

test "this Mac's counters: free and reclaimable pages within its RAM" {
    const m = memory() orelse return error.SkipZigTest;
    const ram = sysctl("hw.memsize") orelse return error.SkipZigTest;
    try std.testing.expect(m.free > 0 and m.room() <= ram);
    try std.testing.expectEqual(floorFor(ram), floor());
}

test "aligned spans copy the bytes asked for, at any offset and length" {
    const gpa = std.testing.allocator;
    var name_buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrintSentinel(&name_buf, "/tmp/tf-paced-read-{d}", .{platform.c.getpid()}, 0);
    const fd = platform.c.open(name, .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true }, @as(platform.c.mode_t, 0o600));
    try std.testing.expect(fd >= 0);
    defer {
        _ = platform.c.close(fd);
        _ = platform.c.unlink(name);
    }
    const data = try gpa.alloc(u8, 3 * alignment + 777);
    defer gpa.free(data);
    for (data, 0..) |*b, i| b.* = @truncate(i *% 131 +% 7);
    try std.testing.expectEqual(@as(isize, @intCast(data.len)), platform.c.write(fd, data.ptr, data.len));
    const buf = try stage(gpa, 2 * alignment);
    defer gpa.free(buf);
    var short: Memory = undefined;
    for ([_][2]usize{ .{ 0, 1 }, .{ 5, 100 }, .{ alignment - 3, 10 }, .{ 1, 2 * alignment + 50 }, .{ 3 * alignment, 777 }, .{ 9, data.len - 9 } }) |c| {
        const out = try gpa.alloc(u8, c[1]);
        defer gpa.free(out);
        try read(fd, out, c[0], buf, &short);
        try std.testing.expectEqualSlices(u8, data[c[0]..][0..c[1]], out);
    }
}
