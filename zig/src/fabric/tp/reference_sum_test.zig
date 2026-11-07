//! Literal three/four-rank witnesses for intermediate rounding and fp32 rank order, including the host transport.
const std = @import("std");
const reference = @import("reference.zig");
const host = @import("host.zig");
const collective = @import("collective.zig");

const Fixture = struct {
    inputs: [4][4]u8,
    want: [4]u8,
    reordered: [4]u8,

    fn pair(a: u16, b: u16) [4]u8 {
        var out: [4]u8 = undefined;
        std.mem.writeInt(u16, out[0..2], a, .little);
        std.mem.writeInt(u16, out[2..4], b, .little);
        return out;
    }

    fn of(world: usize, t: collective.DType) Fixture {
        // Column 0 loses half-ulps if rounded per rank; column 1 loses tiny addends in fp32 until large cancels.
        const bf = t == .bf16;
        const one: u16 = if (bf) 0x3f80 else 0x3c00;
        const half: u16 = if (bf) 0x3b80 else 0x1000;
        const large: u16 = if (bf) 0x4c00 else 0x7800;
        const tiny: u16 = if (bf) 0x3f80 else 0x1400;
        const twice: u16 = if (bf) 0x4000 else 0x1800;
        const thrice: u16 = if (bf) 0x4040 else 0x1a00;
        return .{
            .inputs = .{ pair(one, large), pair(half, tiny), pair(half, large | 0x8000), pair(half, twice) },
            .want = pair(one + @as(u16, if (world == 3) 1 else 2), if (world == 3) 0 else twice),
            .reordered = pair(one + @as(u16, if (world == 3) 1 else 2), if (world == 3) tiny else thrice),
        };
    }
};

fn checkReference(world: usize) !void {
    for ([_]collective.DType{ .bf16, .f16 }) |t| {
        const f = Fixture.of(world, t);
        var views: [4][]const u8 = undefined;
        for (0..world) |r| views[r] = &f.inputs[r];
        var got: [4]u8 = undefined;
        reference.reduce(&got, views[0..world], t, .sum);
        try std.testing.expectEqualSlices(u8, &f.want, &got);
        var per_rank = f.inputs[0];
        for (views[1..world]) |input| reference.accumulate(&per_rank, input, t, .sum);
        try std.testing.expect(!std.mem.eql(u8, got[0..2], per_rank[0..2]));
        std.mem.swap([]const u8, &views[1], &views[2]);
        reference.reduce(&got, views[0..world], t, .sum);
        try std.testing.expectEqualSlices(u8, &f.reordered, &got);
        try std.testing.expect(!std.mem.eql(u8, f.want[2..4], got[2..4]));
    }
}

const HostCase = struct {
    f: Fixture,
    t: collective.DType,
    fn go(c: collective.Collective, rank: u32, self: *@This()) !void {
        var input = self.f.inputs[rank];
        var out: [4]u8 = undefined;
        try c.allReduce(@intFromPtr(&input), @intFromPtr(&out), 2, self.t, .sum, null);
        try std.testing.expectEqualSlices(u8, &self.f.want, &out);
        try c.allReduce(@intFromPtr(&input), @intFromPtr(&input), 2, self.t, .sum, null);
        try std.testing.expectEqualSlices(u8, &self.f.want, &input);
    }
};

fn checkHost(world: u32) !void {
    for ([_]collective.DType{ .bf16, .f16 }) |t| {
        var c: HostCase = .{ .f = Fixture.of(world, t), .t = t };
        try host.run(world, &c, HostCase.go);
    }
}

test "half sum: three-rank reference rounds once in fp32 rank order" {
    try checkReference(3);
}
test "half sum: four-rank reference rounds once in fp32 rank order" {
    try checkReference(4);
}
test "half sum: three-rank host transport matches literal bits out of place and in place" {
    try checkHost(3);
}
test "half sum: four-rank host transport matches literal bits out of place and in place" {
    try checkHost(4);
}

test "half sum: one rank preserves every bit and two ranks keep the binary operation's bits" {
    const patterns = [_]u16{ 0, 0x8000, 1, 0x8001, 0x3f80, 0x7c00, 0x7f80, 0x7fff, 0xffff };
    for ([_]collective.DType{ .bf16, .f16 }) |t| for (patterns) |left| for (patterns) |right| {
        const a = Fixture.pair(left, right);
        const b = Fixture.pair(right, left);
        var got: [4]u8 = undefined;
        reference.reduce(&got, &.{&a}, t, .sum);
        try std.testing.expectEqualSlices(u8, &a, &got);
        var binary = a;
        reference.accumulate(&binary, &b, t, .sum);
        reference.reduce(&got, &.{ &a, &b }, t, .sum);
        try std.testing.expectEqualSlices(u8, &binary, &got);
    };
}
