//! Actual hybrid routing through a deferred fake ring and an independent rank-ordered NCCL byte oracle.
const std = @import("std");
const cuda = @import("cuda");
const tp = @import("hybrid.zig");
const Roce = @import("roce_multi.zig").Roce;
const Nccl = @import("nccl.zig").Nccl;
const fixture = @import("roce_relay_fixture.zig");
const consumers = @import("roce_relay_collectives.zig");
const t = std.testing;

const Fake = struct {
    var h: *fixture.Harness = undefined;
    var rank: u32 = 0;
    var seq: u32 = 0;
    var wires: usize = 0;
    var nccls: usize = 0;
    var source: []u8 = undefined;
    fn oracle(out: []u8, n: usize) void {
        for (0..4) |r| for (0..n / 4) |i|
            std.mem.writeInt(u32, out[r * n + i * 4 ..][0..4], fixture.pattern(seq, @intCast(r), @intCast(i)), .little);
    }
    fn nccl(src: u64, dst: u64, n: usize, _: cuda.nccl.DataType, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        if (src != @intFromPtr(source.ptr)) return 1;
        nccls += 1;
        const out: [*]u8 = @ptrFromInt(dst);
        oracle(out[0 .. 4 * n], n);
        return 0;
    }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const Head = extern struct { src: u64, dst: u64, send: u64, recv: u64, flags: u64, ctrl: u64, state: u64, n: u64 };
        const a: *const Head = @ptrCast(@alignCast(args.?[0].?));
        if (a.src != @intFromPtr(source.ptr)) return 1;
        run(a.dst, @intCast(a.n)) catch return 1;
        wires += 1;
        return 0;
    }
    fn run(dst: u64, n: u32) !void {
        for (0..4) |r| try h.stage(@intCast(r), seq, n);
        const at = h.fabric.layout.sendAt(seq & 1);
        try t.expectEqualSlices(u8, source[0..n], h.fabric.regions[rank][at..][0..n]);
        var done: [4]bool = @splat(false);
        var rounds: u32 = 0;
        while (!std.mem.allEqual(bool, &done, true)) : (rounds += 1) {
            if (rounds > 128) return error.Stalled;
            for (0..4) |r| {
                try h.tick(true, @intCast(r));
                if (!done[r] and h.ready(@intCast(r), seq)) {
                    if (r == rank) {
                        const out: [*]u8 = @ptrFromInt(dst);
                        try consumers.consume(h.fabric.regions[r], h.fabric.layout, @intCast(r), seq, n, .gather, out[0 .. 4 * n]);
                    }
                    try h.check(@intCast(r), seq);
                    done[r] = true;
                }
            }
        }
    }
};

test "relay ceiling: hybrid changes paths at the ceiling and both return every byte unchanged" {
    var h: fixture.Harness = undefined;
    try h.init(256 * 1024);
    defer h.deinit();
    Fake.h = &h;
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    var lib: cuda.nccl.Library = undefined;
    lib.api.ncclAllGather = Fake.nccl;
    const a = t.allocator;
    const source = try a.alloc(u8, 256 * 1024);
    defer a.free(source);
    Fake.source = source;
    const result = try a.alloc(u8, 4 * source.len);
    defer a.free(result);
    const want = try a.alloc(u8, result.len);
    defer a.free(want);
    var seq: u32 = 0;
    for (0..4) |rank| for ([_]usize{ 65520, 65536, 65552, 128 * 1024, 256 * 1024 }) |n| {
        seq += 1;
        Fake.rank = @intCast(rank);
        Fake.seq = seq;
        Fake.wires = 0;
        Fake.nccls = 0;
        for (0..n / 4) |i| std.mem.writeInt(u32, source[i * 4 ..][0..4], fixture.pattern(seq, @intCast(rank), @intCast(i)), .little);
        Fake.oracle(want[0 .. 4 * n], n);
        var nccl: Nccl = .{ .d = &d, .lib = &lib, .comm = null, .me = @intCast(rank), .size = 4, .scratch = undefined, .stream = undefined };
        var r: Roce = .{ .me = @intCast(rank), .world = 4, .relay = true, .layout = h.fabric.layout, .slot_bytes = source.len, .timeout_ns = 1000, .region = h.fabric.regions[rank], .mapping = h.fabric.regions[rank], .wire = .{ .verbs = h.fabric.wires[rank] }, .cpu = null, .d = &d, .dev = 0x3000, .state = .{ .d = &d, .ptr = 0x4000, .len = 16 }, .func = .{ .d = &d, .handle = @ptrFromInt(1) }, .chunk_abi = true };
        var route: tp.Hybrid = .{ .small = .{ .roce_multi = &r }, .nccl = &nccl, .max_bytes = source.len, .chunked_bytes = 1 << 20 };
        try route.iface().allGather(@intFromPtr(source.ptr), @intFromPtr(result.ptr), n, .u8, null);
        try t.expectEqualSlices(u8, want[0 .. 4 * n], result[0 .. 4 * n]);
        try t.expectEqual(@as(usize, @intFromBool(n <= 65536)), Fake.wires);
        try t.expectEqual(@as(usize, @intFromBool(n > 65536)), Fake.nccls);
        if (n > 65536) {
            try t.expect(!route.iface().varFits(n));
            r.relay_max_bytes = source.len;
            try route.iface().allGather(@intFromPtr(source.ptr), @intFromPtr(result.ptr), n, .u8, null);
            try t.expectEqualSlices(u8, want[0 .. 4 * n], result[0 .. 4 * n]);
        }
        route.max_bytes = 0;
        try route.iface().allGather(@intFromPtr(source.ptr), @intFromPtr(result.ptr), n, .u8, null);
        try t.expectEqualSlices(u8, want[0 .. 4 * n], result[0 .. 4 * n]);
        route.max_bytes = source.len;
        r.relay_max_bytes = 128 * 1024;
        try t.expect(r.fits(128 * 1024));
        try t.expect(!r.fits(128 * 1024 + 16));
        r.relay = false;
        try t.expect(r.fits(n));
        try t.expect(route.iface().varFits(source.len + 16));
        r.relay = true;
        r.relay_max_bytes = 64 * 1024;
        try t.expect(!route.iface().varFits(source.len + 16));
    };
}
