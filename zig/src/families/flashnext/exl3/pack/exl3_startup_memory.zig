const std = @import("std");

pub const Reservation = struct {
    stream_bytes: u64,
    streams: u64,

    pub fn total(self: Reservation) !u64 {
        return std.math.mul(u64, self.stream_bytes, self.streams);
    }
};

pub const Ledger = struct {
    persistent: u64,
    scratch: u64,
    deep_prefill: u64,
    pool_scratch: u64,
    expert_pools: u64,
    reservation: Reservation,

    pub fn total(self: Ledger) !u64 {
        var bytes = try self.reservation.total();
        inline for (.{ self.persistent, self.scratch, self.deep_prefill, self.pool_scratch, self.expert_pools }) |part| bytes = try std.math.add(u64, bytes, part);
        return bytes;
    }
};

pub fn mebibytes(text: []const u8) !u64 {
    return std.math.mul(u64, try std.fmt.parseInt(u64, text, 10), 1 << 20);
}

pub fn available(free: u64, reserve: u64, future: u64) !u64 {
    const held = try std.math.add(u64, reserve, future);
    if (held > free) return error.InsufficientStartupMemory;
    return free - held;
}

pub fn poolBytes(lines: u64, line_bytes: u64, layers: u64, fixed: u64) !u64 {
    return std.math.add(u64, try std.math.mul(u64, try std.math.mul(u64, lines, line_bytes), layers), fixed);
}

test "BF16 c1 c8 preserve the requested stream bytes" {
    const bf16_stream: u64 = 262144 * 256 * 2;
    for ([_]u64{ 1, 8 }) |streams| {
        const reservation: Reservation = .{ .stream_bytes = bf16_stream, .streams = streams };
        try std.testing.expectEqual(bf16_stream * streams, try reservation.total());
        const free = try available(16 << 30, 1 << 30, try reservation.total());
        try std.testing.expectEqual(@as(u64, 16 << 30), free + (1 << 30) + try reservation.total());
    }
}

test "startup accounting fails closed on overflow and underfunded explicit pools" {
    try std.testing.expectError(error.Overflow, (Reservation{ .stream_bytes = std.math.maxInt(u64), .streams = 8 }).total());
    try std.testing.expectError(error.Overflow, mebibytes("18446744073709551615"));
    try std.testing.expectError(error.Overflow, poolBytes(std.math.maxInt(u64), 2, 49, 0));
    try std.testing.expectError(error.InsufficientStartupMemory, available(100, 20, try poolBytes(10, 1, 8, 1)));
    try std.testing.expectError(error.Overflow, available(100, std.math.maxInt(u64), 1));
}

test "ledger includes all allocations after the pool checkpoint" {
    const ledger: Ledger = .{ .persistent = 100, .scratch = 20, .deep_prefill = 30, .pool_scratch = 40, .expert_pools = 50, .reservation = .{ .stream_bytes = 10, .streams = 8 } };
    try std.testing.expectEqual(@as(u64, 320), try ledger.total());
}

test "pool setup may consume its allocator margin but not lane or server reservations" {
    const server: u64 = 100;
    const margin: u64 = 25;
    const future: u64 = 80;
    const planner: u64 = 10;
    const planned_free = server + margin + future + planner;
    try std.testing.expectEqual(margin, try available(planned_free, server, future + planner));
    try std.testing.expectEqual(@as(u64, 0), try available(planned_free - margin, server, future + planner));
    try std.testing.expectError(error.InsufficientStartupMemory, available(planned_free - margin - 1, server, future + planner));
}
