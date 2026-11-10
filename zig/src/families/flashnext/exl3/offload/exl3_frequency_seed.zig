const std = @import("std");

pub fn select(counts: []const u32, widths: []const [3]i32, output: []u16) !usize {
    if (counts.len != 512 or widths.len != counts.len or output.len > counts.len) return error.InvalidProfileShape;
    var order: [512]u16 = undefined;
    var used: usize = 0;
    for (counts, widths, 0..) |count, width, expert| {
        if (count == 0 or width[0] <= 0 or width[1] <= 0 or width[2] <= 0) continue;
        order[used] = @intCast(expert);
        used += 1;
    }
    const Ranking = struct {
        fn hotter(histogram: []const u32, left: u16, right: u16) bool {
            if (histogram[left] != histogram[right]) return histogram[left] > histogram[right];
            return left < right;
        }
    };
    std.mem.sort(u16, order[0..used], counts, Ranking.hotter);
    const selected = @min(output.len, used);
    @memcpy(output[0..selected], order[0..selected]);
    return selected;
}

pub fn warm(pool: anytype) !usize {
    const policy = @import("exl3_offload.zig");
    for (pool.line_tab, pool.slots_tab) |state, expert| {
        if (state != .empty or expert != policy.Cache.empty_id) return error.ProfilePoolNotEmpty;
    }
    if (pool.round_no != 0) return error.ProfilePoolNotEmpty;
    var selected: [512]u16 = undefined;
    const count = try select(&pool.counts[pool.opt.layer], pool.k2, selected[0..@min(pool.opt.lines, selected.len)]);
    for (selected[0..count], 0..) |expert, held| {
        const line = try policy.stageUsedAt(&pool.v, pool.opt.layer, expert, selected[0..held], pool.counts);
        try pool.lines.load(line, expert, pool.src.routed[expert], pool.k2[expert], null, pool.opt.direct, pool.io);
    }
    try pool.lines.drain();
    for (selected[0..count]) |expert| try policy.reapAt(&pool.v, pool.opt.layer, expert, pool.now);
    return count;
}

test "exl3: frequency seed ranks local positive experts without heating observations" {
    var histogram: [512]u32 = @splat(0);
    var widths: [512][3]i32 = @splat(.{ 0, 0, 0 });
    histogram[2] = 19;
    histogram[3] = 19;
    histogram[4] = 20;
    histogram[511] = std.math.maxInt(u32);
    widths[2] = .{ 1, 1, 1 };
    widths[3] = .{ 1, 1, 1 };
    widths[4] = .{ 1, 0, 1 };
    var output: [2]u16 = undefined;
    const before = histogram;
    try std.testing.expectEqual(@as(usize, 2), try select(&histogram, &widths, &output));
    try std.testing.expectEqualSlices(u16, &.{ 2, 3 }, &output);
    try std.testing.expectEqualSlices(u32, &before, &histogram);
}

test "exl3: frequency seed leaves spare lines for unknown and unobserved experts" {
    var histogram: [512]u32 = @splat(0);
    const widths: [512][3]i32 = @splat(.{ 1, 1, 1 });
    var output: [512]u16 = undefined;
    try std.testing.expectEqual(@as(usize, 0), try select(&histogram, &widths, &output));
    histogram[511] = 1;
    try std.testing.expectEqual(@as(usize, 1), try select(&histogram, &widths, &output));
    try std.testing.expectEqual(@as(u16, 511), output[0]);
    try std.testing.expectEqual(@as(usize, 0), try select(&histogram, &widths, output[0..0]));
    try std.testing.expectError(error.InvalidProfileShape, select(histogram[0..511], &widths, &output));
}

test "exl3: frequency warm publishes residency only after copies drain and never heats counts" {
    const policy = @import("exl3_offload.zig");
    const Copies = struct {
        ids: [2]u16 = undefined,
        used: usize = 0,
        drained: bool = false,
        states: []policy.Line,
        fail: bool = false,

        pub fn load(copies: *@This(), line: usize, expert: usize, _: [3][]const u8, _: [3]i32, evicted: ?usize, _: bool, _: void) !void {
            try std.testing.expect(evicted == null);
            try std.testing.expectEqual(policy.Line.inflight, copies.states[line]);
            if (copies.fail) return error.CopyFailed;
            copies.ids[copies.used] = @intCast(expert);
            copies.used += 1;
        }

        pub fn drain(copies: *@This()) !void {
            for (copies.states) |state| try std.testing.expectEqual(policy.Line.inflight, state);
            copies.drained = true;
        }
    };
    const Fake = struct {
        line_tab: []policy.Line,
        slots_tab: []u16,
        round_no: u32 = 0,
        counts: *[48][512]u32,
        opt: struct { layer: u8 = 7, lines: usize = 2, direct: bool = true } = .{},
        k2: []const [3]i32,
        v: policy.Vram,
        lines: Copies,
        src: struct { routed: []const [3][]const u8 },
        io: void = {},
        now: u32 = 0,
    };
    const allocator = std.testing.allocator;
    const counts = try allocator.create([48][512]u32);
    defer allocator.destroy(counts);
    counts.* = std.mem.zeroes([48][512]u32);
    counts[7][2] = 100;
    counts[7][4] = 200;
    const before = counts[7];
    const widths: [512][3]i32 = @splat(.{ 1, 1, 1 });
    const sources: [512][3][]const u8 = @splat(.{ &.{}, &.{}, &.{} });
    var ids: [2]u16 = undefined;
    var states: [2]policy.Line = undefined;
    var stamps: [2]u32 = undefined;
    var layers: [2]u8 = undefined;
    var fake: Fake = .{
        .line_tab = &states,
        .slots_tab = &ids,
        .counts = counts,
        .k2 = &widths,
        .v = policy.Vram.init(&ids, &states, &stamps, &layers),
        .lines = .{ .states = &states },
        .src = .{ .routed = &sources },
    };
    try std.testing.expectEqual(@as(usize, 2), try warm(&fake));
    try std.testing.expect(fake.lines.drained);
    try std.testing.expectEqualSlices(u16, &.{ 4, 2 }, &ids);
    try std.testing.expectEqualSlices(u32, &before, &counts[7]);
    for (states) |state| try std.testing.expectEqual(policy.Line.resident, state);
    try std.testing.expectError(error.ProfilePoolNotEmpty, warm(&fake));
    fake.v = policy.Vram.init(&ids, &states, &stamps, &layers);
    fake.lines = .{ .states = &states, .fail = true };
    try std.testing.expectError(error.CopyFailed, warm(&fake));
    try std.testing.expect(!fake.lines.drained);
    try std.testing.expectEqual(policy.Line.inflight, states[0]);
    try std.testing.expectEqualSlices(u32, &before, &counts[7]);
}
