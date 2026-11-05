const std = @import("std");
const builtin = @import("builtin");
const g = @import("benchmark_group.zig");
const expect = std.testing.expect;

pub fn fixture(rank: u32, world: u32, node: u8, device: u8) g.Wire {
    var w: g.Wire = .{ .rank = rank, .world = world, .memory = 128 << 30, .gpu_memory = 120 << 30, .device_known = 1 };
    w.node[0] = node;
    w.device[0] = device;
    @memcpy(w.cpu[0..10], "ARM Cortex");
    @memcpy(w.gpu[0..11], "NVIDIA GB10");
    return w;
}

test "benchmark group: 1/2/4/8 loaded ranks, node identity and loaded-device deduplication" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]u32{ 1, 2, 4, 8 }) |world| for ([_]bool{ false, true }) |multi| {
        var all: [8]g.Wire = undefined;
        for (all[0..world], 0..) |*w, i| w.* = fixture(@intCast(i), world, if (multi) @intCast(i + 1) else 1, 1);
        const bytes = (try g.assemble(a, all[0..world])).?;
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
        try expect(doc.object.get("rank_count").?.integer == world);
        const nodes = doc.object.get("nodes").?.array.items;
        try expect(nodes.len == (if (multi) @as(usize, world) else 1));
        try expect(nodes[0].object.get("hardware").?.object.get("gpu_count").?.integer == 1);
        try std.testing.expectEqualStrings(if (multi and world > 1) "multi-node" else "single-node", doc.object.get("category").?.string);
        for ([_][]const u8{ "uuid", "device", "hostname", "boot" }) |private| try expect(std.mem.indexOf(u8, bytes, private) == null);
    };
    var local = [_]g.Wire{ fixture(0, 2, 1, 1), fixture(1, 2, 1, 2) };
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, (try g.assemble(a, &local)).?, .{});
    try expect(doc.object.get("nodes").?.array.items[0].object.get("hardware").?.object.get("gpu_count").?.integer == 2);
}

test "benchmark group: incomplete identity, wrong rank cover and conflicting hardware fail closed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const good = [_]g.Wire{ fixture(0, 2, 1, 1), fixture(1, 2, 1, 2) };
    for (0..7) |fault| {
        var bad = good;
        switch (fault) {
            0 => bad[1].rank = 0,
            1 => bad[1].world = 4,
            2 => bad[1].node = @splat(0),
            3 => bad[1].device_known = 0,
            4 => bad[1].memory += 1,
            5 => bad[1].gpu[0] = '/',
            6 => {
                bad[1].device = bad[0].device;
                bad[1].gpu_memory += 1;
            },
            else => unreachable,
        }
        try expect((try g.assemble(a, &bad)) == null);
    }
    try expect((try g.assemble(a, &.{})) == null);
}

test "benchmark group: the loaded communicator supplies count and every peer before a snapshot is published" {
    const Ctx = struct {
        d: *u8,
        pub fn name(_: @This(), _: []u8) ![]const u8 {
            return "NVIDIA GB10";
        }
        pub fn memInfo(_: @This()) !struct { total: usize } {
            return .{ .total = 120 << 30 };
        }
        pub fn uuid(_: @This()) ![16]u8 {
            return @splat(1);
        }
    };
    const Group = struct {
        calls: u32 = 0,
        pub fn benchmarkShape(_: *@This()) !struct { world: u32, rank: u32 } {
            return .{ .world = 4, .rank = 0 };
        }
        pub fn benchmarkGather(self: *@This(), comptime T: type, _: *u8, _: u8, mine: T, all: []T) !void {
            try expect(mine.world == 4 and mine.rank == 0 and all.len == 4);
            self.calls += 1;
            for (all, 0..) |*w, i| w.* = fixture(@intCast(i), 4, @intCast(i + 1), 1);
        }
    };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var byte: u8 = 0;
    var group: Group = .{};
    const bytes = (try g.cuda(arena.allocator(), std.testing.io, Ctx{ .d = &byte }, &group, @as(u8, 0))).?;
    try expect(group.calls == 1 and std.mem.indexOf(u8, bytes, "multi-node") != null);
}

/// `probe` names a node only on macOS and Linux, so a Metal handshake's own inventory is unranked anywhere else.
fn probeNamesNode() bool {
    return builtin.os.tag == .macos or builtin.os.tag == .linux;
}

test "benchmark group: Metal endpoint inventories and legacy HELLO peers preserve the serving handshake" {
    if (!probeNamesNode()) return error.SkipZigTest;
    const Device = struct {
        pub fn name(_: @This()) [*:0]const u8 {
            return "Apple Test GPU";
        }
    };
    const Link = struct {
        bytes: [8192]u8 align(8) = @splat(0),
        peer_ready: u64,
        sent: ?g.Wire = null,
        flushed: bool = false,
        pub fn size(_: *@This()) u32 {
            return 2;
        }
        pub fn rank(_: *@This()) u32 {
            return 0;
        }
        pub fn window(x: *@This()) []u8 {
            return &x.bytes;
        }
        pub fn write2Signal(x: *@This(), peer: u32, offset: usize, head: []const u8, body: []const u8, hello: usize, value: u64) !void {
            try expect(peer == 1 and offset == 4096 and hello == 64 and body.len == 0 and value >= 2);
            x.sent = std.mem.bytesToValue(g.Wire, head[0..@sizeOf(g.Wire)]);
            @as(*u64, @ptrCast(@alignCast(x.bytes[hello..][0..8]))).* = x.peer_ready;
        }
        pub fn flush(x: *@This()) !void {
            x.flushed = true;
        }
    };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    for ([_]u64{ 2, 3 }) |ready| for ([_]bool{ false, true }) |old| {
        var link: Link = .{ .peer_ready = if (old) 1 else ready };
        const peer = fixture(1, 2, 2, 1);
        @memcpy(link.bytes[4096..][0..@sizeOf(g.Wire)], std.mem.asBytes(&peer));
        const bytes = try g.metal(arena.allocator(), std.testing.io, Device{}, &link, 4096, 64, ready);
        try expect(link.flushed and link.sent.?.rank == 0 and link.sent.?.world == 2);
        try expect((bytes == null) == old);
        if (bytes) |b| try expect(std.mem.indexOf(u8, b, "multi-node") != null);
    };
}

test "benchmark group: a delayed Metal peer retains the startup wait without a new deadline" {
    if (!probeNamesNode()) return error.SkipZigTest;
    const Clock = struct {
        var calls: std.atomic.Value(u32) = .init(0);
        var table: std.Io.VTable = undefined;
        fn now(ctx: ?*anyopaque, kind: std.Io.Clock) std.Io.Timestamp {
            if (kind == .awake) return .{ .nanoseconds = @as(i96, calls.fetchAdd(1, .monotonic)) * 61 * std.time.ns_per_s };
            return std.testing.io.vtable.now(ctx, kind);
        }
        fn io() std.Io {
            calls.store(0, .monotonic);
            table = std.testing.io.vtable.*;
            table.now = now;
            return .{ .userdata = std.testing.io.userdata, .vtable = &table };
        }
    };
    const Device = struct {
        pub fn name(_: @This()) [*:0]const u8 {
            return "Apple Test GPU";
        }
    };
    const Link = struct {
        bytes: [8192]u8 align(8) = @splat(0),
        sent: ?g.Wire = null,
        flushed: std.atomic.Value(bool) = .init(false),
        ready: u64,
        pub fn size(_: *@This()) u32 {
            return 2;
        }
        pub fn rank(_: *@This()) u32 {
            return 0;
        }
        pub fn window(x: *@This()) []u8 {
            return &x.bytes;
        }
        pub fn write2Signal(x: *@This(), _: u32, _: usize, head: []const u8, _: []const u8, _: usize, _: u64) !void {
            x.sent = std.mem.bytesToValue(g.Wire, head[0..@sizeOf(g.Wire)]);
        }
        pub fn flush(x: *@This()) !void {
            x.flushed.store(true, .release);
        }
        fn peer(x: *@This()) void {
            while (!x.flushed.load(.acquire)) std.atomic.spinLoopHint();
            for (0..4096) |_| std.atomic.spinLoopHint();
            const word: *u64 = @ptrCast(@alignCast(x.bytes[64..][0..8]));
            @atomicStore(u64, word, x.ready, .release);
        }
    };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    for ([_]u64{ 2, 3 }) |ready| {
        var link: Link = .{ .ready = ready };
        const peer = fixture(1, 2, 2, 1);
        @memcpy(link.bytes[4096..][0..@sizeOf(g.Wire)], std.mem.asBytes(&peer));
        const thread = try std.Thread.spawn(.{}, Link.peer, .{&link});
        defer thread.join();
        const bytes = (try g.metal(arena.allocator(), Clock.io(), Device{}, &link, 4096, 64, ready)).?;
        try expect(link.flushed.load(.acquire) and link.sent.?.rank == 0);
        try expect(std.mem.indexOf(u8, bytes, "multi-node") != null);
        try std.testing.expectEqual(@as(u32, 0), Clock.calls.load(.monotonic));
    }
}
