//! One NVFP4 decode grid over compatible projections, retaining each member's existing tile and K-slice arithmetic.
const std = @import("std");
const q = @import("qmmf.zig");
const launch = @import("launch.zig");
const Stream = @import("stream.zig").Stream;
pub const Many = struct { w: q.Weight, out: u64 };
pub const Part = extern struct { codes: u64, scales: u64, out: u64, scale: f32, n: i32, npad: i32, tiles: i32 };
pub const Parts = extern struct { p: [4]Part, count: i32 };
pub const symbols = [6][:0]const u8{ "tf_qmmf_many16", "tf_qmmf_many16c", "tf_qmmf_many32", "tf_qmmf_many32c", "tf_qmmf_many64", "tf_qmmf_many64c" };
comptime {
    std.debug.assert(@sizeOf(Part) == 40 and @sizeOf(Parts) == 168);
}

pub fn tiles(rows: usize, n: usize) usize {
    const bm = q.bucket(rows);
    return (rows + bm - 1) / bm * ((n + 63) / 64);
}

/// The kernel's running-prefix map, with no ownership assigned beyond the final member tile.
pub fn locate(counts: []const usize, block: usize) ?struct { part: usize, tile: usize } {
    var local = block;
    for (counts, 0..) |count, i| {
        if (local < count) return .{ .part = i, .tile = local };
        local -= count;
    }
    return null;
}

pub fn eligible(l: q.Lane, rows: usize, parts: []const Many) bool {
    if (rows == 0 or rows > 128 or parts.len < 2 or parts.len > 4) return false;
    const first = parts[0].w;
    const sk = q.slices(first);
    if (first.mode != .fp4 or first.k == 0 or first.k % 64 != 0) return false;
    if (sk == 0 or sk > 8 or sk & (sk - 1) != 0 or first.k / 64 % sk != 0 or (sk > 1 and !l.clusters)) return false;
    for (parts, 0..) |part, i| {
        const w = part.w;
        if (w.mode != first.mode or w.k != first.k or q.slices(w) != sk or w.n == 0 or w.npad < w.n or w.npad % 128 != 0) return false;
        for (parts[0..i]) |before| {
            const end = @as(u128, part.out) + @as(u128, rows) * w.n * 2;
            const before_end = @as(u128, before.out) + @as(u128, rows) * before.w.n * 2;
            if (part.out < before_end and before.out < end) return false;
        }
    }
    return true;
}

/// Decode only; incompatible members keep their separate launches and reuse scratch in stream order.
pub fn matmulMany(l: q.Lane, s: Stream, x: u64, ldx: usize, rows: usize, parts: []const Many, scratch: u64) !void {
    if (!eligible(l, rows, parts)) {
        for (parts) |part| try l.matmulOut(s, x, ldx, rows, part.w, part.out, .bf16, scratch);
        return;
    }
    const w = parts[0].w;
    const p = q.planFor(rows, w, l.clusters);
    const bm: usize = p.bm;
    const row_tiles = (rows + bm - 1) / bm;
    var descriptors: Parts = std.mem.zeroes(Parts);
    var total: usize = 0;
    for (parts, 0..) |part, i| {
        const count = tiles(rows, part.w.n);
        descriptors.p[i] = .{ .codes = part.w.codes, .scales = part.w.scales, .out = part.out, .scale = part.w.scale, .n = @intCast(part.w.n), .npad = @intCast(part.w.npad), .tiles = @intCast(count) };
        total += count;
    }
    descriptors.count = @intCast(parts.len);
    const band = @max(1, @min(row_tiles, (12 << 20) / (bm * w.k * 2)));
    var args: launch.Args = .{};
    args.add(x);
    args.add(descriptors);
    for ([_]usize{ rows, w.k, p.sk, if (rows == 1) w.k else ldx, band }) |v| args.add(@as(i32, @intCast(v)));
    const tile: usize = if (bm == 16) 0 else if (bm == 32) 1 else 2;
    try launch.launch(l.many[tile * 2 + @intFromBool(p.cluster)], .{
        .grid = .{ .x = @intCast(total), .z = p.sk },
        .block = .{ .x = 128 },
        .shared = q.sharedBytes(.fp4, p.bm),
        .cluster = if (p.cluster) .{ .x = 1, .y = 1, .z = p.sk } else null,
    }, s, &args);
}

test "qmmf many: every uneven member tile maps exactly once at every decode row tile" {
    for ([_]usize{ 1, 2, 8, 16, 17, 32, 64, 65, 128 }) |rows| {
        const ns = [_]usize{ 1, 65, 127, 2047 };
        var counts: [4]usize = undefined;
        for (&counts, ns) |*count, n| count.* = tiles(rows, n);
        for (1..5) |parts| {
            var seen: [4][64]bool = @splat(@splat(false));
            var total: usize = 0;
            for (counts[0..parts]) |count| total += count;
            for (0..total) |block| {
                const mapped = locate(counts[0..parts], block).?;
                try std.testing.expect(mapped.part < parts and mapped.tile < counts[mapped.part]);
                try std.testing.expect(!seen[mapped.part][mapped.tile]);
                seen[mapped.part][mapped.tile] = true;
            }
            for (counts[0..parts], 0..) |count, part| for (seen[part][0..count]) |visited| try std.testing.expect(visited);
            try std.testing.expect(locate(counts[0..parts], total) == null);
        }
    }
}

test "qmmf many: incompatible modes K slices and output aliases retain separate-launch eligibility" {
    var lane: q.Lane = undefined;
    lane.clusters = true;
    const w: q.Weight = .{ .mode = .fp4, .codes = 1, .scales = 2, .n = 129, .k = 4096, .npad = 256, .sk = 4 };
    var parts = [_]Many{ .{ .w = w, .out = 0x10000 }, .{ .w = w, .out = 0x20000 } };
    try std.testing.expect(eligible(lane, 16, &parts));
    parts[1].w.mode = .bf16;
    try std.testing.expect(!eligible(lane, 16, &parts));
    parts[1].w = w;
    parts[1].w.k = 2048;
    try std.testing.expect(!eligible(lane, 16, &parts));
    parts[1].w = w;
    parts[1].w.sk = 2;
    try std.testing.expect(!eligible(lane, 16, &parts));
    parts[1].w = w;
    parts[1].out = parts[0].out + 2;
    try std.testing.expect(!eligible(lane, 16, &parts));
    parts[1].out = 0x20000;
    lane.clusters = false;
    try std.testing.expect(!eligible(lane, 16, &parts));
    try std.testing.expect(!eligible(lane, 129, &parts));
}

test "qmmf many: compatible members launch once and incompatible members call matmulOut separately" {
    const abi = @import("abi.zig");
    const Mock = struct {
        var calls: usize = 0;
        var grouped: usize = 0;
        var count: i32 = 0;
        var grid: u32 = 0;
        fn launch(f: abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
            calls += 1;
            grid = gx;
            if (@intFromPtr(f.?) >= 100) {
                grouped += 1;
                count = @as(*const Parts, @ptrCast(@alignCast(args.?[1].?))).*.count;
            }
            return 0;
        }
    };
    var driver: @import("driver.zig").Driver = undefined;
    driver.api.cuLaunchKernel = Mock.launch;
    const stream: Stream = .{ .d = &driver, .handle = null };
    const old: @import("module.zig").Function = .{ .d = &driver, .handle = @ptrFromInt(7) };
    var lane: q.Lane = undefined;
    lane.clusters = false;
    lane.half_below = 0;
    for (&lane.fns) |*set| set.* = @splat(old);
    for (&lane.many, 0..) |*function, i| function.* = .{ .d = &driver, .handle = @ptrFromInt(100 + i) };
    const w: q.Weight = .{ .mode = .fp4, .codes = 1, .scales = 2, .n = 129, .k = 512, .npad = 256, .sk = 1 };
    var parts = [_]Many{ .{ .w = w, .out = 0x10000 }, .{ .w = w, .out = 0x20000 } };
    parts[1].w.n = 65;
    Mock.calls = 0;
    Mock.grouped = 0;
    try lane.matmulMany(stream, 0x1000, 512, 8, &parts, 0);
    try std.testing.expectEqual(@as(usize, 1), Mock.calls);
    try std.testing.expectEqual(@as(usize, 1), Mock.grouped);
    try std.testing.expectEqual(@as(i32, 2), Mock.count);
    try std.testing.expectEqual(@as(u32, 5), Mock.grid);
    parts[1].w.k = 1024;
    Mock.calls = 0;
    Mock.grouped = 0;
    try lane.matmulMany(stream, 0x1000, 512, 8, &parts, 0);
    try std.testing.expectEqual(@as(usize, 2), Mock.calls);
    try std.testing.expectEqual(@as(usize, 0), Mock.grouped);
}
