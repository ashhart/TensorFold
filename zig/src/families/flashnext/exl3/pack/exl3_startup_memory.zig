const std = @import("std");
const planner_layout = @import("../dplan/exl3_dplan_layout.zig");
const demand_trace = @import("../dplan/exl3_dplan_trace.zig");

pub const Reservation = struct {
    stream_bytes: u64,
    streams: u64,
    graph_bytes: u64,
    after_pool_bytes: u64 = 0,

    pub fn total(self: Reservation) !u64 {
        return std.math.add(u64, try std.math.add(u64, try std.math.mul(u64, self.stream_bytes, self.streams), self.graph_bytes), self.after_pool_bytes);
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

pub fn graphReservation(enabled: bool, text: ?[]const u8) !u64 {
    if (!enabled) return 0;
    const bytes = try mebibytes(text orelse return error.GraphReservationRequired);
    if (bytes == 0) return error.GraphReservationRequired;
    return bytes;
}

pub fn available(free: u64, reserve: u64, future: u64) !u64 {
    const held = try std.math.add(u64, reserve, future);
    if (held > free) return error.InsufficientStartupMemory;
    return free - held;
}

pub fn laneRoom(local: u64, graph_reservation: u64) u64 {
    return local -| graph_reservation;
}

pub fn poolBytes(lines: u64, line_bytes: u64, layers: u64, fixed: u64) !u64 {
    return std.math.add(u64, try std.math.mul(u64, try std.math.mul(u64, lines, line_bytes), layers), fixed);
}

pub fn plannerBytes(layers: u64, lines: u64, rows: u64) !u64 {
    const state = try planner_layout.deviceBytes(layers, try std.math.mul(u64, layers, lines), rows);
    return std.math.add(u64, state, try plannerTraceBytes(layers, traceEnabled()));
}

pub fn plannerBytesForPools(lines: []const usize, rows: u64) !u64 {
    var total_lines: u64 = 0;
    for (lines) |count| total_lines = try std.math.add(u64, total_lines, count);
    const state = try planner_layout.deviceBytes(lines.len, total_lines, rows);
    return std.math.add(u64, state, try plannerTraceBytes(lines.len, traceEnabled()));
}

pub fn plannerTraceBytes(pools: u64, enabled: bool) !u64 {
    return if (enabled) std.math.mul(u64, pools, demand_trace.pool_bytes) else 0;
}

fn traceEnabled() bool {
    const value = std.c.getenv("TF_FLASHNEXT_EXL3_DPLAN_TRACE") orelse return false;
    return demand_trace.enabledValue(std.mem.span(value));
}

test "BF16 c1 c8 and graph on off preserve the requested stream bytes" {
    const bf16_stream: u64 = 262144 * 256 * 2;
    for ([_]u64{ 1, 8 }) |streams| {
        for ([_]bool{ false, true }) |graphs| {
            const graph = try graphReservation(graphs, "512");
            const reservation: Reservation = .{ .stream_bytes = bf16_stream, .streams = streams, .graph_bytes = graph, .after_pool_bytes = 4096 };
            try std.testing.expectEqual(bf16_stream * streams + graph + 4096, try reservation.total());
            const free = try available(16 << 30, 1 << 30, try reservation.total());
            try std.testing.expectEqual(@as(u64, 16 << 30), free + (1 << 30) + try reservation.total());
        }
    }
}

test "startup accounting fails closed on overflow and underfunded explicit pools" {
    try std.testing.expectError(error.Overflow, (Reservation{ .stream_bytes = std.math.maxInt(u64), .streams = 8, .graph_bytes = 0 }).total());
    try std.testing.expectError(error.Overflow, mebibytes("18446744073709551615"));
    try std.testing.expectError(error.GraphReservationRequired, graphReservation(true, null));
    try std.testing.expectError(error.GraphReservationRequired, graphReservation(true, "0"));
    try std.testing.expectEqual(@as(u64, 0), try graphReservation(false, null));
    try std.testing.expectError(error.Overflow, poolBytes(std.math.maxInt(u64), 2, 49, 0));
    try std.testing.expectError(error.InsufficientStartupMemory, available(100, 20, try poolBytes(10, 1, 8, 1)));
    try std.testing.expectError(error.Overflow, available(100, std.math.maxInt(u64), 1));
}

test "ledger includes all allocations after the pool checkpoint" {
    const ledger: Ledger = .{ .persistent = 100, .scratch = 20, .deep_prefill = 30, .pool_scratch = 40, .expert_pools = 50, .reservation = .{ .stream_bytes = 10, .streams = 8, .graph_bytes = 60, .after_pool_bytes = 70 } };
    try std.testing.expectEqual(@as(u64, 450), try ledger.total());
}

test "pool setup may consume its allocator margin but not lane graph or server reservations" {
    const server: u64 = 100;
    const margin: u64 = 25;
    const future: u64 = 80;
    const planner: u64 = 10;
    const planned_free = server + margin + future + planner;
    try std.testing.expectEqual(margin, try available(planned_free, server, future + planner));
    try std.testing.expectEqual(@as(u64, 0), try available(planned_free - margin, server, future + planner));
    try std.testing.expectError(error.InsufficientStartupMemory, available(planned_free - margin - 1, server, future + planner));
}

test "planner allocation ledger agrees with device state statics work and split buffers" {
    const layers: u64 = 49;
    const lines: u64 = 10;
    const rows: u64 = 16;
    const expected = layers * planner_layout.stateWords(lines) * 4 + layers * 512 * 3 * (8 + 8 + 4) + rows * 10 * 9 * 8 + 256 + (512 + 64) * 4 + 2 * rows * 11 * 4;
    try std.testing.expectEqual(expected, try plannerBytes(layers, lines, rows));
    try std.testing.expectEqual(try plannerBytes(layers, 0, rows) + layers * lines * 8, try plannerBytes(layers, lines, rows));
    try std.testing.expectError(error.Overflow, plannerBytes(layers, std.math.maxInt(u64), rows));
}

test "opt-in demand traces are included in the startup planner reservation" {
    try std.testing.expectEqual(@as(u64, 0), try plannerTraceBytes(48, false));
    try std.testing.expectEqual(@as(u64, 48 << 20), try plannerTraceBytes(48, true));
    const base = try planner_layout.deviceBytes(48, 0, 16);
    try std.testing.expectEqual(base + (48 << 20), try std.math.add(u64, base, try plannerTraceBytes(48, true)));
}

test "exl3: ragged planner redistribution conserves each rank's exact device budget" {
    const uniform = [_]usize{ 32, 32, 32, 32 };
    const ragged = [_]usize{ 10, 73, 11, 34 };
    const other_rank = [_]usize{ 12, 28, 52, 36 };
    for ([_][]const usize{ &uniform, &ragged, &other_rank }) |counts| {
        var state_words: usize = 0;
        for (counts) |lines| {
            const entry = try planner_layout.PoolState.init(lines, state_words);
            state_words = entry.endWords();
        }
        const rows = 16;
        const work = rows * 10 * 9 * 8 + 256 + (512 + 64) * 4;
        const split = 2 * rows * 11 * 4;
        const allocated = state_words * 4 + counts.len * planner_layout.static_bytes + work + split;
        try std.testing.expectEqual(allocated, try plannerBytesForPools(counts, rows));
        try std.testing.expectEqual(try plannerBytes(uniform.len, uniform[0], rows), try plannerBytesForPools(counts, rows));
        const line_bytes = 3 << 20;
        const fixed = try plannerBytesForPools(counts, rows);
        var total = fixed;
        for (counts) |lines| total += lines * line_bytes;
        try std.testing.expectEqual(try poolBytes(32, line_bytes, 4, fixed), total);
    }
    try std.testing.expectError(error.Overflow, plannerBytesForPools(&.{ std.math.maxInt(usize), std.math.maxInt(usize) }, 16));
    try std.testing.expectError(error.Overflow, plannerBytesForPools(&ragged, std.math.maxInt(u64)));
}

test "lane admission protects the full graph reservation without changing BF16 stream sizes" {
    const stream_bytes: u64 = 262144 * 256 * 2;
    const graph_bytes: u64 = 512 << 20;
    for ([_]u64{ 1, 8 }) |streams| {
        const requested = try std.math.mul(u64, streams, stream_bytes);
        try std.testing.expectEqual(requested, laneRoom(requested, 0));
        try std.testing.expectEqual(requested, laneRoom(try std.math.add(u64, requested, graph_bytes), graph_bytes));
        try std.testing.expect(laneRoom(requested, graph_bytes) < requested);
    }
    try std.testing.expectEqual(@as(u64, 0), laneRoom(graph_bytes - 1, graph_bytes));
    try std.testing.expectEqual(@as(u64, 0), laneRoom(graph_bytes, graph_bytes));
    try std.testing.expectEqual(@as(u64, 0), laneRoom(0, graph_bytes));
}

test "lane room subtraction cannot overflow at integer boundaries" {
    const maximum = std.math.maxInt(u64);
    try std.testing.expectEqual(maximum, laneRoom(maximum, 0));
    try std.testing.expectEqual(@as(u64, 0), laneRoom(maximum, maximum));
    try std.testing.expectEqual(@as(u64, 0), laneRoom(1, maximum));
    try std.testing.expectEqual(maximum - 1, laneRoom(maximum, 1));
    try std.testing.expectError(error.Overflow, (Reservation{ .stream_bytes = maximum, .streams = 1, .graph_bytes = 1 }).total());
}

test "per rank protection precedes agreement of lane counts" {
    const graph_bytes: u64 = 512 << 20;
    const rank_zero_stream: u64 = 128 << 20;
    const rank_one_stream: u64 = 64 << 20;
    const rank_zero_room = laneRoom(8 * rank_zero_stream + graph_bytes, graph_bytes);
    const rank_one_room = laneRoom(8 * rank_one_stream + graph_bytes, graph_bytes);
    try std.testing.expectEqual(@as(u64, 8), @min(rank_zero_room / rank_zero_stream, rank_one_room / rank_one_stream));
    try std.testing.expectEqual(@as(u64, 7), @min(rank_zero_room / rank_zero_stream, laneRoom(8 * rank_one_stream + graph_bytes - 1, graph_bytes) / rank_one_stream));
}
