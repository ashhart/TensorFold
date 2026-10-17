//! Every collective on real GPUs against the host reference, eager and replayed from a graph.
const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const Collective = tp.Collective;
const DType = tp.DType;
const Op = tp.Op;
const reference = tp.reference;

pub const Ctx = struct {
    d: *const cuda.Driver,
    gpa: std.mem.Allocator,
    stream: cuda.Stream,
    rank: u32,
    /// Device buffers sized for the largest case: input, output (world x), a spare.
    in: cuda.DeviceBuffer,
    out: cuda.DeviceBuffer,
    /// Host scratch: this rank's input, the peer's, the expected and the downloaded output.
    mine: []u8,
    peer: []u8,
    want: []u8,
    got: []u8,
    failures: u32 = 0,
    passes: u32 = 0,

    pub fn init(d: *const cuda.Driver, gpa: std.mem.Allocator, rank: u32, max_bytes: usize) !Ctx {
        return .{
            .d = d,
            .gpa = gpa,
            .stream = try cuda.Stream.init(d, true),
            .rank = rank,
            .in = try cuda.DeviceBuffer.alloc(d, 2 * max_bytes + 256),
            .out = try cuda.DeviceBuffer.alloc(d, 2 * max_bytes + 256),
            .mine = try gpa.alloc(u8, max_bytes),
            .peer = try gpa.alloc(u8, max_bytes),
            .want = try gpa.alloc(u8, 2 * max_bytes),
            .got = try gpa.alloc(u8, 2 * max_bytes),
        };
    }

    pub fn deinit(c: *Ctx) void {
        c.stream.deinit();
        c.in.free();
        c.out.free();
        for ([_][]u8{ c.mine, c.peer, c.want, c.got }) |b| c.gpa.free(b);
    }

    fn verdict(c: *Ctx, ok: bool, comptime fmt: []const u8, args: anytype) void {
        if (ok) c.passes += 1 else c.failures += 1;
        std.debug.print("{s} " ++ fmt ++ "\n", .{if (ok) "PASS" else "FAIL"} ++ args);
    }

    /// Both ranks' inputs for case `seed`: this rank's into `mine`, the peer's into `peer` (n bytes each).
    fn inputs(c: *Ctx, n: usize, t: DType, seed: u64) void {
        reference.fill(c.mine[0..n], t, c.rank, seed);
        reference.fill(c.peer[0..n], t, 1 - c.rank, seed);
    }

    fn rankInput(c: *Ctx, r: u32, n: usize) []const u8 {
        return if (r == c.rank) c.mine[0..n] else c.peer[0..n];
    }

    fn compare(c: *Ctx, comm: Collective, label: []const u8, what: []const u8, t: DType, n: usize, out_off: usize, want: []const u8) !void {
        try c.stream.synchronize();
        comm.check() catch |e| {
            c.verdict(false, "{s} {s} {t} {d} B: transport error {t}", .{ label, what, t, n, e });
            return;
        };
        try c.out.download(out_off, c.got[0..want.len]);
        const ok = std.mem.eql(u8, c.got[0..want.len], want);
        var first: usize = 0;
        if (!ok) first = std.mem.indexOfDiff(u8, c.got[0..want.len], want) orelse 0;
        c.verdict(ok, "{s} {s} {t} {d} B a rank{s}", .{ label, what, t, n, if (ok) "" else " (first difference at a byte shown below)" });
        if (!ok) std.debug.print("     byte {d}: got {x} want {x}\n", .{ first, c.got[first], want[first] });
    }

    /// all-reduce (out of place, in place, misaligned by `skew` bytes), all-gather (out of place, in place), exchange.
    pub fn collectives(c: *Ctx, comm: Collective, label: []const u8, n_bytes: usize, seed: u64) !void {
        const reduce_types = [_]DType{ .f32, .bf16, .f16, .i32 };
        for (reduce_types, 0..) |t, ti| {
            const n = n_bytes - n_bytes % t.size();
            const count = n / t.size();
            c.inputs(n, t, seed + ti);
            reference.reduce(c.want[0..n], &.{ c.rankInput(0, n), c.rankInput(1, n) }, t, .sum);
            try c.in.uploadAsync(0, c.mine[0..n], c.stream.handle);
            try comm.allReduce(c.in.ptr, c.out.ptr, count, t, .sum, c.stream.handle);
            try c.compare(comm, label, "all-reduce sum", t, n, 0, c.want[0..n]);
            // in place, at an offset of one element (not 16-byte aligned for 2-byte types)
            const skew = t.size();
            try c.out.uploadAsync(skew, c.mine[0..n], c.stream.handle);
            try comm.allReduce(c.out.ptr + skew, c.out.ptr + skew, count, t, .sum, c.stream.handle);
            try c.compare(comm, label, "all-reduce sum in place, skewed", t, n, skew, c.want[0..n]);
        }
        for ([_]Op{ .max, .min }) |op| {
            const n = n_bytes - n_bytes % 4;
            c.inputs(n, .f32, seed + 11);
            reference.reduce(c.want[0..n], &.{ c.rankInput(0, n), c.rankInput(1, n) }, .f32, op);
            try c.in.uploadAsync(0, c.mine[0..n], c.stream.handle);
            try comm.allReduce(c.in.ptr, c.out.ptr, n / 4, .f32, op, c.stream.handle);
            try c.compare(comm, label, if (op == .max) "all-reduce max" else "all-reduce min", .f32, n, 0, c.want[0..n]);
        }
        for ([_]DType{ .u8, .bf16, .f32 }, 0..) |t, ti| {
            const n = n_bytes - n_bytes % t.size();
            c.inputs(n, t, seed + 20 + ti);
            @memcpy(c.want[0..n], c.rankInput(0, n));
            @memcpy(c.want[n..][0..n], c.rankInput(1, n));
            try c.in.uploadAsync(0, c.mine[0..n], c.stream.handle);
            try comm.allGather(c.in.ptr, c.out.ptr, n / t.size(), t, c.stream.handle);
            try c.compare(comm, label, "all-gather", t, n, 0, c.want[0 .. 2 * n]);
            try c.out.fill8(0, c.stream.handle);
            try c.out.uploadAsync(c.rank * n, c.mine[0..n], c.stream.handle);
            try comm.allGather(c.out.ptr + c.rank * n, c.out.ptr, n / t.size(), t, c.stream.handle);
            try c.compare(comm, label, "all-gather in place", t, n, 0, c.want[0 .. 2 * n]);
            try c.in.uploadAsync(0, c.mine[0..n], c.stream.handle);
            try comm.exchange(c.in.ptr, c.out.ptr, n / t.size(), t, 1 - c.rank, c.stream.handle);
            try c.compare(comm, label, "exchange", t, n, 0, c.peer[0..n]);
        }
    }

    /// send then recv one way, then the other way (NCCL behind every backend).
    pub fn pointToPoint(c: *Ctx, comm: Collective, label: []const u8, n_bytes: usize, seed: u64) !void {
        const n = n_bytes - n_bytes % 4;
        c.inputs(n, .f32, seed);
        try c.in.uploadAsync(0, c.mine[0..n], c.stream.handle);
        for (0..2) |dir| {
            const sender: u32 = @intCast(dir);
            if (c.rank == sender) {
                try comm.send(c.in.ptr, n / 4, .f32, 1 - c.rank, c.stream.handle);
                try c.stream.synchronize();
            } else {
                try comm.recv(c.out.ptr, n / 4, .f32, sender, c.stream.handle);
                try c.compare(comm, label, "send/recv", .f32, n, 0, c.peer[0..n]);
            }
        }
    }

    /// A captured chain replayed with fresh inputs, eager calls between replays (the mailbox epoch spans both).
    pub fn graphs(c: *Ctx, comm: Collective, label: []const u8, n_bytes: usize, replays: u32) !void {
        const n = n_bytes - n_bytes % 16;
        const half = n / 2; // bf16 elements
        try cuda.graph.beginCapture(c.stream, .thread_local);
        // x1 = sum(x0) [bf16], x2 = gather(x1), x3 = sum(x2) [f32 view], x4 = exchange(x3)
        const x0 = c.in.ptr;
        const x1 = c.out.ptr;
        const x2 = c.in.ptr + n;
        const x3 = c.out.ptr + n;
        const x4 = c.out.ptr + 3 * n;
        const ops: ?tp.collective.Error = blk: {
            comm.allReduce(x0, x1, half, .bf16, .sum, c.stream.handle) catch |e| break :blk e;
            comm.allGather(x1, x2, half, .bf16, c.stream.handle) catch |e| break :blk e;
            comm.allReduce(x2, x3, n / 2, .f32, .sum, c.stream.handle) catch |e| break :blk e;
            comm.exchange(x3, x4, n / 2, .f32, 1 - c.rank, c.stream.handle) catch |e| break :blk e;
            break :blk null;
        };
        var g = try cuda.graph.endCapture(c.stream);
        defer g.deinit();
        if (ops) |e| return e;
        var exec = try g.instantiate();
        defer exec.deinit();
        const nodes = try g.nodeCount();
        const want1 = try c.gpa.alloc(u8, 2 * n);
        defer c.gpa.free(want1);
        const want3 = try c.gpa.alloc(u8, 2 * n);
        defer c.gpa.free(want3);
        for (0..replays) |k| {
            c.inputs(n, .bf16, 1000 + k);
            try c.in.uploadAsync(0, c.mine[0..n], c.stream.handle);
            reference.reduce(want1[0..n], &.{ c.rankInput(0, n), c.rankInput(1, n) }, .bf16, .sum);
            @memcpy(want1[n..][0..n], want1[0..n]); // the gather of equal sums: twice the same
            reference.reduce(want3[0 .. 2 * n], &.{ want1[0 .. 2 * n], want1[0 .. 2 * n] }, .f32, .sum);
            try exec.launchOn(c.stream);
            try c.compare(comm, label, "graph replay: sum, gather, sum, exchange", .bf16, n, n, want3[0 .. 2 * n]);
            try c.compare(comm, label, "graph replay: exchange of the sum", .f32, 2 * n, 3 * n, want3[0 .. 2 * n]);
            // an eager call between replays
            try c.collectivesQuick(comm, label, k);
        }
        std.debug.print("     ({s}: {d} graph nodes, {d} replays)\n", .{ label, nodes, replays });
    }

    fn collectivesQuick(c: *Ctx, comm: Collective, label: []const u8, k: usize) !void {
        const n = 4096;
        c.inputs(n, .bf16, 5000 + k);
        reference.reduce(c.want[0..n], &.{ c.rankInput(0, n), c.rankInput(1, n) }, .bf16, .sum);
        try c.in.uploadAsync(5 * n, c.mine[0..n], c.stream.handle);
        try comm.allReduce(c.in.ptr + 5 * n, c.out.ptr + 5 * n, n / 2, .bf16, .sum, c.stream.handle);
        try c.compare(comm, label, "eager between replays", .bf16, n, 5 * n, c.want[0..n]);
    }
};
