//! The `Collective` contract on the in-process ranks: every call's bits against the reference, desyncs and aborts.
const std = @import("std");
const collective = @import("collective.zig");
const reference = @import("reference.zig");
const host = @import("host.zig");
const Collective = collective.Collective;
const DType = collective.DType;

fn addr(b: []u8) collective.DevicePtr {
    return @intFromPtr(b.ptr);
}

const n = 96;

const Calls = struct {
    fn go(c: Collective, rank: u32, _: void) !void {
        const world = c.world();
        inline for (.{ DType.f32, DType.bf16, DType.i32, DType.f16 }) |t| {
            const bytes = n * t.size();
            var mine: [n * 8]u8 = undefined;
            var out: [n * 8 * host.max_ranks]u8 = undefined;
            reference.fill(mine[0..bytes], t, rank, 7);
            // all-reduce against the reference sum of every rank's input, in rank order
            var ins: [host.max_ranks][n * 8]u8 = undefined;
            var views: [host.max_ranks][]const u8 = undefined;
            for (0..world) |r| {
                reference.fill(ins[r][0..bytes], t, @intCast(r), 7);
                views[r] = ins[r][0..bytes];
            }
            var want: [n * 8]u8 = undefined;
            reference.reduce(want[0..bytes], views[0..world], t, .sum);
            try c.allReduce(addr(&mine), addr(&out), n, t, .sum, null);
            try std.testing.expectEqualSlices(u8, want[0..bytes], out[0..bytes]);
            // in place
            try c.allReduce(addr(&mine), addr(&mine), n, t, .sum, null);
            try std.testing.expectEqualSlices(u8, want[0..bytes], mine[0..bytes]);
            // all-gather: rank order
            reference.fill(mine[0..bytes], t, rank, 7);
            try c.allGather(addr(&mine), addr(&out), n, t, null);
            for (0..world) |r| try std.testing.expectEqualSlices(u8, views[r], out[r * bytes ..][0..bytes]);
        }
        try c.barrier();
    }
};

test "every call on two and three in-process ranks equals the reference" {
    try host.run(2, {}, Calls.go);
    try host.run(3, {}, Calls.go);
}

const Pairwise = struct {
    fn go(c: Collective, rank: u32, _: void) !void {
        var mine: [64]u8 = @splat(@intCast(rank + 1));
        var got: [64]u8 = undefined;
        const peer = 1 - rank;
        try c.exchange(addr(&mine), addr(&got), 64, .u8, peer, null);
        try std.testing.expectEqualSlices(u8, &@as([64]u8, @splat(@intCast(peer + 1))), &got);
        // send / recv, ordered so neither waits forever
        if (rank == 0) {
            try c.send(addr(&mine), 16, .f32, 1, null);
        } else {
            var buf: [64]u8 = undefined;
            try c.recv(addr(&buf), 16, .f32, 0, null);
            try std.testing.expectEqualSlices(u8, &@as([64]u8, @splat(1)), &buf);
        }
        try std.testing.expectError(error.Invalid, c.exchange(addr(&mine), addr(&got), 64, .u8, rank, null));
    }
};

test "exchange, send and recv between two ranks" {
    try host.run(2, {}, Pairwise.go);
}

const Desync = struct {
    fn go(c: Collective, rank: u32, _: void) !void {
        var a: [64]u8 = @splat(0);
        var b: [128]u8 = undefined;
        // the ranks disagree on the count: both see it instead of hanging
        const r = c.allGather(addr(&a), addr(&b), if (rank == 0) 16 else 32, .u8, null);
        if (r) |_| return error.ShouldHaveFailed else |e| if (e != error.Invalid and e != error.Aborted) return e;
    }
};

test "a desynchronised call is refused on every rank" {
    try host.run(2, {}, Desync.go);
}

const Aborts = struct {
    fn go(c: Collective, rank: u32, _: void) !void {
        var a: [16]u8 = @splat(0);
        var b: [32]u8 = undefined;
        if (rank == 1) {
            // rank 1 never calls: it aborts after a while, which must free rank 0's waiting call
            @import("sock.zig").sleepNs(50 * std.time.ns_per_ms);
            c.abort();
            return;
        }
        try std.testing.expectError(error.Aborted, c.allGather(addr(&a), addr(&b), 16, .u8, null));
        try std.testing.expectError(error.Aborted, c.check());
        try std.testing.expectError(error.Aborted, c.barrier());
    }
};

test "abort frees a rank waiting in a collective" {
    try host.run(2, {}, Aborts.go);
}

const VarGather = struct {
    fn go(c: Collective, rank: u32, refuse: u32) !void {
        const world = c.world();
        const stride = 200;
        const whole = refuse & (@as(u32, 1) << @intCast(rank)) != 0;
        try std.testing.expectEqual(!whole, c.varFits(stride));
        try std.testing.expectEqual(refuse == 0, try c.varAgreed());
        try std.testing.expectEqual(refuse == 0, try c.varAgreed()); // kept: no second collective
        var ins: [host.max_ranks][stride]u8 = undefined;
        for (0..world) |r| reference.fill(&ins[r], .u8, @intCast(r), 11);
        // lengths of 0, odd, 16 and the whole stride, in bytes; .f32 strides count elements
        const cases = [_][3]i32{ .{ 0, 17, 200 }, .{ 200, 0, 3 }, .{ 16, 16, 16 }, .{ 199, 200, 1 } };
        inline for (.{ DType.u8, DType.f32 }) |t| for (cases) |lens| {
            var out: [stride * host.max_ranks]u8 = @splat(0xEE);
            try c.allGatherV(addr(&ins[rank]), addr(&out), stride / t.size(), @intFromPtr(&lens), t, null);
            for (0..world) |r| {
                const len: usize = @intCast(lens[r]);
                try std.testing.expectEqualSlices(u8, ins[r][0..len], out[r * stride ..][0..len]);
                // past a length: the transport's choice (untouched here, or the whole stride where lengths are refused)
                const past = out[r * stride + len .. (r + 1) * stride];
                if (whole) try std.testing.expectEqualSlices(u8, ins[r][len..], past) else for (past) |x| try std.testing.expectEqual(@as(u8, 0xEE), x);
            }
        };
        try c.barrier();
    }
};

test "a variable-length all-gather moves each rank's length, and the ranks agree on device lengths once" {
    for ([_]u32{ 2, 3 }) |w| {
        try host.runWith(w, .{}, @as(u32, 0), VarGather.go);
        try host.runWith(w, .{ .var_refuse = 2 }, @as(u32, 2), VarGather.go);
    }
}

const VarDesync = struct {
    fn go(c: Collective, rank: u32, _: void) !void {
        var a: [64]u8 = @splat(0);
        var b: [128]u8 = undefined;
        const lens = [2]i32{ 8, if (rank == 0) 8 else 16 };
        const r = c.allGatherV(addr(&a), addr(&b), 64, @intFromPtr(&lens), .u8, null);
        if (r) |_| return error.ShouldHaveFailed else |e| if (e != error.Invalid and e != error.Aborted) return e;
    }
};

test "lengths that differ between ranks are refused on every rank" {
    try host.run(2, {}, VarDesync.go);
}
