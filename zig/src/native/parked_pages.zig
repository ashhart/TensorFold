//! Free pages parked on the kernel's per-CPU lists, read from /proc/zoneinfo: free, but not in MemFree or MemAvailable.
const std = @import("std");

/// What the per-CPU lists hold at rest when /proc/zoneinfo prints no high_min (kernels before 6.7).
pub const resting: u64 = 1 << 29;

/// Bytes of the `count:` lines under every `pagesets` less their `high_min:` (else `resting`); 0 on a bad field.
pub fn bytes(text: []const u8, page: u64) u64 {
    var count: u64 = 0;
    var rest: u64 = 0;
    var listed = false;
    var rows = std.mem.splitScalar(u8, text, '\n');
    while (rows.next()) |row| {
        var words = std.mem.tokenizeAny(u8, row, " \t\r");
        const first = words.next() orelse "";
        if (std.mem.eql(u8, std.mem.trim(u8, row, " \t\r"), "pagesets")) {
            listed = true;
        } else if (listed and !std.mem.startsWith(u8, row, "    ") and !std.mem.startsWith(u8, row, "  vm stats")) {
            listed = false; // the next zone, or this zone's next field
        } else if (listed and (std.mem.eql(u8, first, "count:") or std.mem.eql(u8, first, "high_min:"))) {
            const n = std.fmt.parseInt(u64, words.next() orelse return 0, 10) catch return 0;
            const sum = if (first[0] == 'c') &count else &rest;
            sum.* = std.math.add(u64, sum.*, n) catch return 0;
        }
    }
    if (count == 0) return 0;
    const held = std.math.mul(u64, count, page) catch return 0;
    const rests = if (rest > 0) std.math.mul(u64, rest, page) catch return 0 else resting;
    return held -| rests;
}

/// One zone's /proc/zoneinfo block as kernel 6.7+ prints it, with page lists when `counts` is given.
fn zone(a: std.mem.Allocator, out: *std.ArrayList(u8), node: u32, name: []const u8, counts: ?[]const u64, high_min: ?u64) !void {
    try out.print(a, "Node {d}, zone {s:>8}\n  pages free     10\n        managed  1000\n      nr_free_cma  0\n", .{ node, name });
    if (counts) |cs| {
        try out.appendSlice(a, "  pagesets\n");
        for (cs, 0..) |c, n| {
            try out.print(a, "    cpu: {d}\n              count:    {d}\n              high:     5000\n              batch:    63\n", .{ n, c });
            if (high_min) |h| try out.print(a, "              high_min: {d}\n", .{h});
            try out.appendSlice(a, "              high_max: 198283\n  vm stats threshold: 110\n");
        }
    }
    try out.appendSlice(a, "  node_unreclaimable:  0\n  start_pfn:           4096\n");
}

test "parked pages, kernel 6.7+: every CPU, zone and node, less each list's high_min" {
    const a = std.testing.allocator;
    const page: u64 = 4096;
    const hm: u64 = 2146;
    var t: std.ArrayList(u8) = .empty;
    defer t.deinit(a);
    const ca = [_]u64{ 200000, 3143, 2332, 17 };
    const cb = [_]u64{ 100000, 100000, 100000 };
    const cc = [_]u64{400000};
    try zone(a, &t, 0, "DMA", &ca, hm);
    try zone(a, &t, 0, "DMA32", null, hm);
    try zone(a, &t, 0, "Normal", &cb, hm);
    try zone(a, &t, 1, "Normal", &cc, hm);
    try zone(a, &t, 1, "Movable", null, hm);
    var sum: u64 = 0;
    for (ca ++ cb ++ cc) |c| sum += c;
    try std.testing.expectEqual((sum - 8 * hm) * page, bytes(t.items, page));
    t.clearRetainingCapacity();
    try zone(a, &t, 0, "Normal", &@as([20]u64, @splat(100)), hm);
    try std.testing.expectEqual(@as(u64, 0), bytes(t.items, page)); // under the resting level: nothing
}

test "parked pages, before 6.7: no high_min rests the first 512 MiB" {
    const a = std.testing.allocator;
    const page: u64 = 4096;
    var t: std.ArrayList(u8) = .empty;
    defer t.deinit(a);
    const half = ((1 << 30) + resting) / page / 2;
    try zone(a, &t, 0, "Normal", &.{ half, half }, null);
    try std.testing.expectEqual(@as(u64, 1 << 30), bytes(t.items, page));
    t.clearRetainingCapacity();
    try zone(a, &t, 0, "Normal", &.{ 1000, 1000, 1000, 1000 }, null);
    try std.testing.expectEqual(@as(u64, 0), bytes(t.items, page));
}

test "parked pages: only the count: and high_min: lines under pagesets" {
    const a = std.testing.allocator;
    const page: u64 = 4096;
    var t: std.ArrayList(u8) = .empty;
    defer t.deinit(a);
    try t.appendSlice(a, "  count: 5\n      count: 6\n");
    try zone(a, &t, 0, "Normal", &.{2146 + 10}, 2146);
    try t.appendSlice(a, "      count:    99999\n      high_min: 7\n"); // after start_pfn: not a list
    try t.appendSlice(a, "Node 0, zone Other\n      count:    77777\n  pagesets\n    cpu: 0\n              high:     1\n");
    try std.testing.expectEqual(10 * page, bytes(t.items, page));
    const next_field = "Node 0, zone Normal\n  pagesets\n    cpu: 0\n              count: 2200\n              high: 9\n" ++
        "              high_min: 2000\n  vm stats threshold: 8\n    cpu: 1\n              count: 2100\n" ++
        "              high_min: 2000\n  all_unreclaimable: 0\n      count: 9999\n";
    try std.testing.expectEqual(300 * page, bytes(next_field, page));
}

test "parked pages: no lists, empty, garbled, binary or overflowing text counts nothing" {
    const a = std.testing.allocator;
    const page: u64 = 4096;
    var t: std.ArrayList(u8) = .empty;
    defer t.deinit(a);
    try zone(a, &t, 0, "DMA", null, 2146);
    try zone(a, &t, 0, "Normal", null, 2146);
    try std.testing.expectEqual(@as(u64, 0), bytes(t.items, page));
    try std.testing.expectEqual(@as(u64, 0), bytes("", page));
    var good: std.ArrayList(u8) = .empty;
    defer good.deinit(a);
    try zone(a, &good, 0, "Normal", &.{1000000}, 2146);
    try std.testing.expect(bytes(good.items, page) > 0);
    const swaps = [_][2][]const u8{
        .{ "count:    1000000", "count:    x" },
        .{ "count:    1000000", "count:" },
        .{ "count:    1000000", "count:    -5" },
        .{ "high_min: 2146", "high_min: x" },
        .{ "count:    1000000", "count:    \xff\xfe" },
    };
    for (swaps) |swap| {
        const garbled = try std.mem.replaceOwned(u8, a, good.items, swap[0], swap[1]);
        defer a.free(garbled);
        try std.testing.expectEqual(@as(u64, 0), bytes(garbled, page));
    }
    try std.testing.expectEqual(@as(u64, 0), bytes("\x00\xff\xfe\x00\xff\xfe\n  pagesets\x00", page));
    const wraps = "Node 0, zone Normal\n  pagesets\n    cpu: 0\n      count: 18446744073709551615\n      count: 1\n";
    try std.testing.expectEqual(@as(u64, 0), bytes(wraps, page));
    try std.testing.expectEqual(@as(u64, 0), bytes(good.items, std.math.maxInt(u64))); // pages times size overflows
}

test "this host's /proc/zoneinfo parses to less than its memory (Linux)" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const a = std.testing.allocator;
    const procText = @import("cuda_memory.zig").procText;
    const zones = procText(a, std.testing.io, "/proc/zoneinfo") orelse return error.Unreadable;
    defer a.free(zones);
    const mem = procText(a, std.testing.io, "/proc/meminfo") orelse return error.Unreadable;
    defer a.free(mem);
    try std.testing.expect(std.mem.indexOf(u8, zones, "pagesets") != null);
    try std.testing.expect(bytes(zones, std.heap.pageSize()) < @import("cuda_memory.zig").meminfo(mem).?.total);
}

test "a GB10's /proc/zoneinfo: lists still draining after a server exit count, lists at rest count nothing" {
    const page = 4096; // GB10's page size
    try std.testing.expectEqual(@as(u64, 1027231744), bytes(@embedFile("fixtures/gb10_parked/zoneinfo_draining"), page));
    try std.testing.expectEqual(@as(u64, 0), bytes(@embedFile("fixtures/gb10_parked/zoneinfo_at_budget"), page));
}
