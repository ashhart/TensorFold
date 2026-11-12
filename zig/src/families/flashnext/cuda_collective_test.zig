//! The family collective's FP64 all-gather over a fake two-rank in-place NCCL that moves `count` elements of the type it is
//! given, as NCCL does: every double must arrive bit for bit (an FP32-typed copy would deliver half their bytes).
const std = @import("std");
const cuda = @import("cuda");
const Collective = @import("cuda_collective.zig").Collective;
const nccl = cuda.nccl;

const Fake = struct {
    var rank: usize = 0;
    var peer: []const u8 = &.{};

    fn elementBytes(t: nccl.DataType) usize {
        return switch (t) {
            .i8, .u8 => 1,
            .f16, .bf16 => 2,
            .i32, .u32, .f32 => 4,
            .i64, .u64, .f64 => 8,
        };
    }

    /// An in-place all-gather of two ranks: this rank's block already sits in its slot, the peer's lands in the other.
    fn allGather(send: cuda.abi.DevicePtr, recv: cuda.abi.DevicePtr, count: usize, t: nccl.DataType, _: nccl.Comm, _: cuda.abi.Stream) callconv(.c) nccl.Result {
        const block = count * elementBytes(t);
        if (send != recv + rank * block or block > peer.len) return 1;
        const to: [*]u8 = @ptrFromInt(recv + (1 - rank) * block);
        @memcpy(to[0..block], peer[0..block]);
        return 0;
    }

    fn asyncError(_: nccl.Comm, state: *nccl.Result) callconv(.c) nccl.Result {
        state.* = 0;
        return 0;
    }

    fn abort(_: nccl.Comm) callconv(.c) nccl.Result {
        return 0;
    }
};

fn bits(x: f64) u64 {
    return @bitCast(x);
}

/// Doubles an FP32 round trip changes: low bits below its 24, a subnormal and an overflow for FP32, a NaN with a payload,
/// a signed zero; the two ranks differ.
const partials: [2][8]u64 = .{
    .{ bits(1.0 + 0x1p-40), bits(0x1p-1000), bits(0x1p200), bits(-0.0), 0x7ff8_dead_beef_0001, bits(1.0e9 + 0x1p-20), bits(0x1p-150), bits(std.math.floatMax(f64)) },
    .{ bits(-1.0e9 - 0x1p-20), bits(0x1p-1074), bits(-0x1p300), bits(0.0), 0xfff0_0000_0000_0001, bits(3.0 + 0x1p-51), bits(0x1p127 * (1.0 + 0x1p-52)), bits(-std.math.inf(f64)) },
};

test "the FP64 all-gather delivers every rank's doubles bit for bit" {
    for (0..2) |rank| {
        var pair: [2]c_int = undefined;
        if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
        defer _ = std.c.close(pair[0]);
        defer _ = std.c.close(pair[1]);
        var c: Collective = undefined;
        c.failed = false;
        c.comm = @ptrFromInt(0x1000);
        c.link = .{ .fd = pair[0] };
        c.timeout_ms = 1000;
        c.rank = rank;
        c.spin_ns = 0;
        c.lib.api.ncclAllGather = Fake.allGather;
        c.api.ncclCommGetAsyncError = Fake.asyncError;
        c.api.ncclCommAbort = Fake.abort;
        Fake.rank = rank;
        Fake.peer = std.mem.sliceAsBytes(&partials[1 - rank]);
        var d: cuda.Driver = undefined;
        const stream: cuda.Stream = .{ .d = &d, .handle = null };
        var gathered: [2][8]u64 = undefined;
        @memset(std.mem.sliceAsBytes(&gathered), 0);
        gathered[rank] = partials[rank];
        try c.gatherF64(stream, @intFromPtr(&gathered[rank]), @intFromPtr(&gathered), partials[0].len);
        try std.testing.expectEqualSlices(u64, &partials[0], &gathered[0]);
        try std.testing.expectEqualSlices(u64, &partials[1], &gathered[1]);
    }
}
