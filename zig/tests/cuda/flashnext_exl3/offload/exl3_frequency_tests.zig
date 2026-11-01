const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("../../check.zig");
const profile = flash.exl3.frequency;
const Pool = flash.exl3_pool.Pool;

fn verify(gpu: check.Gpu, pool: *Pool, expected: usize) !void {
    const tables = try gpu.gpa.alloc(u8, pool.lines.tables.len);
    defer gpu.gpa.free(tables);
    try pool.lines.tables.download(0, tables);
    var resident: usize = 0;
    var bytes: [16]u8 = undefined;
    for (0..512) |expert| {
        var found: ?usize = null;
        for (pool.slots_tab, pool.line_tab, 0..) |held, state, line| {
            if (held == expert and state == .resident) found = line;
        }
        if (found != null) resident += 1;
        for (0..3) |projection| {
            const pointer = std.mem.readInt(u64, tables[pool.lines.tlay.ptr[projection] + expert * 8 ..][0..8], .little);
            const width = std.mem.readInt(i32, tables[pool.lines.tlay.k2[projection] + expert * 4 ..][0..4], .little);
            if (found) |line| {
                const offset = line * pool.lines.layout.line_bytes + pool.lines.layout.off[projection];
                try check.expect(pointer == pool.lines.mem.ptr + offset and width == 1, "frequency warm table expert {d} projection {d}", .{ expert, projection });
                try pool.lines.mem.download(offset, &bytes);
                try check.sameBytes("frequency warm expert bytes", &bytes, pool.src.routed[expert][projection]);
            } else try check.expect(pointer == 0 and width == 0, "unloaded frequency expert has no device entry", .{});
        }
    }
    try check.expect(resident == expected, "frequency resident count {d} expected {d}", .{ resident, expected });
}

pub fn run(gpu: check.Gpu, arguments: []const [:0]const u8) !void {
    if (arguments.len != 2) return error.MissingArgument;
    const rank = try std.fmt.parseInt(u32, arguments[0], 10);
    if (rank >= 2) return error.BadRank;
    const path = arguments[1];
    const identity: profile.Identity = .{ .model = @splat(0x57), .world = 2, .rank = rank };
    const start = @as(usize, rank) * 256;
    var bank = try cuda.HostBuffer.alloc(gpu.d, 512 * 3 * 16);
    defer bank.free();
    var source: [512][3][]const u8 = @splat(.{ &.{}, &.{}, &.{} });
    for (start..start + 256) |expert| for (0..3) |projection| {
        const offset = (expert * 3 + projection) * 16;
        const bytes = bank.bytes[offset..][0..16];
        @memset(bytes, @truncate(expert + projection));
        source[expert][projection] = bytes;
    };
    var saved: profile.Counts = .{};
    for (0..20) |index| saved.decoder[7][start + index] = @intCast(100 + index);
    saved.decoder[7][(start + 256) % 512] = std.math.maxInt(u32);
    try profile.write(gpu.gpa, gpu.io, path, identity, &saved);
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const scales: flash.exl3_pool.Scales = .{ .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0 };
    for ([_]usize{ 10, 23 }, 0..) |capacity, restart| {
        const restored = try profile.read(gpu.gpa, gpu.io, path, identity);
        var pool = try Pool.init(gpu.gpa, gpu.io, gpu.d, .{ .routed = &source }, scales, .{
            .layer = 7,
            .lines = capacity,
            .dims = 16,
            .width = 16,
            .max_rows = 1,
            .direct = true,
        }, null);
        defer pool.deinit();
        @memcpy(&pool.counts[7], &restored.decoder[7]);
        const before = pool.counts[7];
        const heated = try pool.warmProfile();
        try check.expect(heated == @min(capacity, 20), "frequency restart fills known local experts only", .{});
        try check.expect(std.mem.eql(u32, &before, &pool.counts[7]), "warming preserves observation counts", .{});
        try verify(gpu, &pool, heated);
        var picks: [11]i32 = @splat(-1);
        for (0..10) |index| picks[index] = @intCast(start + 19 - index);
        var device_picks = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&picks));
        defer device_picks.free();
        var planner = try flash.exl3.dplan.DevPlan.init(gpu.gpa, gpu.d, &.{&pool});
        defer planner.deinit();
        try planner.toDevice(stream);
        try planner.plan(stream, 0, device_picks.ptr, 1);
        try planner.join(stream);
        try planner.restore(stream, 0);
        try planner.toHost();
        for (0..20) |index| {
            const expected = before[start + index] + @as(u32, @intFromBool(index >= 10));
            try check.expect(pool.counts[7][start + index] == expected, "device learned count survives profile restart", .{});
        }
        @memcpy(&saved.decoder[7], &pool.counts[7]);
        try profile.write(gpu.gpa, gpu.io, path, identity, &saved);
        std.debug.print("FREQUENCY rank={d} restart={d} capacity={d} warmed={d}\n", .{ rank, restart, capacity, heated });
    }
    var other_rank = identity;
    other_rank.rank = 1 - rank;
    if (profile.read(gpu.gpa, gpu.io, path, other_rank)) |_| return error.AcceptedForeignProfile else |failure| {
        try check.expect(failure == error.IdentityMismatch, "frequency rejects other TP rank", .{});
    }
    check.pass("frequency disk restart, live device counts, rank isolation and resized warm pools", .{});
}
