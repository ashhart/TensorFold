//! The actual hybrid dispatch keeps unknown multirank arithmetic and pairwise exchange on NCCL.
const std = @import("std");
const cuda = @import("cuda");
const hybrid = @import("hybrid.zig");
const Nccl = @import("nccl.zig").Nccl;
const Roce = @import("roce_multi.zig").Roce;
const Fabric = @import("roce_multi_fake.zig").Fabric;
const t = std.testing;
const Seen = struct {
    var reduce: usize = 0;
    var send: usize = 0;
    var recv: usize = 0;
    var gather: usize = 0;
    var launches: usize = 0;
    var peer: c_int = -1;
    fn allReduce(_: cuda.abi.DevicePtr, _: cuda.abi.DevicePtr, _: usize, _: cuda.nccl.DataType, _: cuda.nccl.RedOp, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        reduce += 1;
        return 0;
    }
    fn allGather(_: cuda.abi.DevicePtr, _: cuda.abi.DevicePtr, _: usize, _: cuda.nccl.DataType, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        gather += 1;
        return 0;
    }
    fn group() callconv(.c) cuda.nccl.Result {
        return 0;
    }
    fn sendFn(_: cuda.abi.DevicePtr, _: usize, _: cuda.nccl.DataType, p: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        send += 1;
        peer = p;
        return 0;
    }
    fn recvFn(_: cuda.abi.DevicePtr, _: usize, _: cuda.nccl.DataType, p: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        recv += 1;
        peer = p;
        return 0;
    }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const A = extern struct { in: u64, out: u64, send_slots: u64, recv_slots: u64, flags: u64, ctrl: u64, state: u64, nbytes: u64, slot_bytes: u64, timeout: u64, rank: u32, mode: u32, nh: u32, opts: u32, lens: u64, chunk: u64, world: u32, pad: u32 };
        const a: *const A = @ptrCast(@alignCast(args.?[0].?));
        if (bx != 512 or a.mode != 0 or a.world < 3 or a.world > 4 or a.in != 0x1000 or a.out != 0x2000 or a.nh != 1 or a.nbytes > 4096) return 1;
        launches += 1;
        return 0;
    }
};

test "multirank route: every floating reduction and ring/all-to-all pair goes to the original NCCL call" {
    for ([_]u32{ 3, 4 }) |world| {
        var f: Fabric = undefined;
        try f.init(world, 1, 4096);
        defer f.deinit();
        var d: cuda.Driver = undefined;
        d.api.cuLaunchKernel = Seen.launch;
        var lib: cuda.nccl.Library = undefined;
        lib.api.ncclAllReduce = Seen.allReduce;
        lib.api.ncclAllGather = Seen.allGather;
        lib.api.ncclSend = Seen.sendFn;
        lib.api.ncclRecv = Seen.recvFn;
        lib.api.ncclGroupStart = Seen.group;
        lib.api.ncclGroupEnd = Seen.group;
        Seen.reduce = 0;
        Seen.send = 0;
        Seen.recv = 0;
        Seen.gather = 0;
        Seen.launches = 0;
        for (0..world) |rank| {
            var nccl: Nccl = .{ .d = &d, .lib = &lib, .comm = null, .me = @intCast(rank), .size = world, .scratch = undefined, .stream = undefined };
            var r: Roce = .{ .me = @intCast(rank), .world = world, .layout = f.layout, .slot_bytes = 4096, .timeout_ns = 1000, .region = f.regions[rank], .mapping = f.regions[rank], .wire = .{ .verbs = f.wires[rank] }, .cpu = null, .d = &d, .dev = 0x3000, .state = .{ .d = &d, .ptr = 0x4000, .len = 16 }, .func = .{ .d = &d, .handle = @ptrFromInt(1) }, .chunk_abi = true };
            var route: hybrid.Hybrid = .{ .small = .{ .roce_multi = &r }, .nccl = &nccl, .max_bytes = 4096 };
            const c = route.iface();
            for ([_]@import("collective.zig").DType{ .f16, .bf16, .f32, .f64 }) |dt| try c.allReduce(0x1000, 0x2000, 16, dt, .sum, null);
            // Every directed peer appears, including the cyclic next-rank edge; the kernel must never broadcast it.
            for (0..world) |peer| if (peer != rank) {
                try c.exchange(0x1000, 0x2000, 16, .u8, @intCast(peer), null);
                try t.expectEqual(@as(c_int, @intCast(peer)), Seen.peer);
            };
            try t.expectEqual(@as(usize, 2 * rank), Seen.launches);
            try c.allGather(0x1000, 0x2000, 128, .u8, null);
            try c.allGatherV(0x1000, 0x2000, 128, 0x5000, .u8, null);
            try t.expectEqual(@as(usize, 2 * (rank + 1)), Seen.launches);
        }
        try t.expectEqual(@as(usize, world * 4), Seen.reduce);
        try t.expectEqual(@as(usize, world * (world - 1)), Seen.send);
        try t.expectEqual(Seen.send, Seen.recv);
        try t.expectEqual(@as(usize, 0), Seen.gather);
    }
}
