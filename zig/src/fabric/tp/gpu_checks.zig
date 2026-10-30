//! Every collective on real GPUs against the host reference, eager and replayed from a graph; multiple ranks.
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
    world: u32,
    /// Device buffers sized for the largest case: input, output, a spare ((world + 2) x the largest shard each).
    in: cuda.DeviceBuffer,
    out: cuda.DeviceBuffer,
    /// Host scratch: every rank's input (`ins[r]`), the expected and the downloaded output.
    ins: [][]u8,
    want: []u8,
    got: []u8,
    failures: u32 = 0,
    passes: u32 = 0,
    baseline: ?Collective = null,
    oracle: cuda.DeviceBuffer,

    pub fn init(d: *const cuda.Driver, gpa: std.mem.Allocator, rank: u32, world: u32, max_bytes: usize) !Ctx {
        const ins = try gpa.alloc([]u8, world);
        for (ins) |*b| b.* = try gpa.alloc(u8, max_bytes);
        return .{
            .d = d,
            .gpa = gpa,
            .stream = try cuda.Stream.init(d, true),
            .rank = rank,
            .world = world,
            .in = try cuda.DeviceBuffer.alloc(d, @max(world + 2, 7) * max_bytes + 256),
            .out = try cuda.DeviceBuffer.alloc(d, @max(world + 2, 7) * max_bytes + 256),
            .oracle = try cuda.DeviceBuffer.alloc(d, @max(world + 2, 7) * max_bytes + 256),
            .ins = ins,
            .want = try gpa.alloc(u8, (world + 2) * max_bytes),
            .got = try gpa.alloc(u8, (world + 2) * max_bytes),
        };
    }

    pub fn deinit(c: *Ctx) void {
        c.stream.deinit();
        c.in.free();
        c.out.free();
        c.oracle.free();
        for (c.ins) |b| c.gpa.free(b);
        c.gpa.free(c.ins);
        for ([_][]u8{ c.want, c.got }) |b| c.gpa.free(b);
    }

    fn verdict(c: *Ctx, ok: bool, comptime fmt: []const u8, args: anytype) void {
        if (ok) c.passes += 1 else c.failures += 1;
        std.debug.print("{s} " ++ fmt ++ "\n", .{if (ok) "PASS" else "FAIL"} ++ args);
    }

    /// Every rank's input for case `seed` (n bytes each), this rank's included.
    fn inputs(c: *Ctx, n: usize, t: DType, seed: u64) void {
        for (c.ins, 0..) |b, r| reference.fill(b[0..n], t, @intCast(r), seed);
    }

    fn mine(c: *Ctx, n: usize) []const u8 {
        return c.ins[c.rank][0..n];
    }

    /// Every rank's first n bytes, rank order (`reference.reduce`'s inputs).
    fn all(c: *Ctx, n: usize, buf: *[tp.bootstrap.max_ranks][]const u8) []const []const u8 {
        for (c.ins, 0..) |b, r| buf[r] = b[0..n];
        return buf[0..c.world];
    }

    /// This rank's exchange partner: ranks pair up as (0, 1), (2, 3), ...; the last rank of an odd world sits out.
    fn partner(c: *const Ctx) ?u32 {
        const p = c.rank ^ 1;
        return if (p < c.world) p else null;
    }

    /// Above two ranks, the actual NCCL result is the reference, with the same in-place geometry.
    fn sumReference(c: *Ctx, n: usize, t: DType, offset: ?usize, input: cuda.abi.DevicePtr) !void {
        if (c.world == 2 or t == .u8 or t == .i32 or t == .i64) return;
        const slow = c.baseline orelse return error.MissingNcclReference;
        const out = c.oracle.ptr + (offset orelse 0);
        if (offset) |skew| try c.oracle.uploadAsync(skew, c.mine(n), c.stream.handle);
        try slow.allReduce(if (offset != null) out else input, out, n / t.size(), t, .sum, c.stream.handle);
        try c.stream.synchronize();
        try slow.check();
        try c.oracle.download(offset orelse 0, c.want[0..n]);
    }

    fn compare(c: *Ctx, comm: Collective, label: []const u8, what: []const u8, t: DType, n: usize, out_off: usize, want: []const u8) !void {
        return c.compareWith(comm, label, what, t, n, out_off, want);
    }

    /// Every comparison is byte equality; no tolerance can qualify a transport.
    fn compareWith(c: *Ctx, comm: Collective, label: []const u8, what: []const u8, t: DType, n: usize, out_off: usize, want: []const u8) !void {
        try c.stream.synchronize();
        comm.check() catch |e| {
            c.verdict(false, "{s} {s} {t} {d} B: transport error {t}", .{ label, what, t, n, e });
            return;
        };
        try c.out.download(out_off, c.got[0..want.len]);
        const got = c.got[0..want.len];
        const ok = std.mem.eql(u8, got, want);
        var first: usize = 0;
        if (!ok) first = std.mem.indexOfDiff(u8, got, want) orelse 0;
        c.verdict(ok, "{s} {s} {t} {d} B a rank{s}{s}", .{ label, what, t, n, " (exact)", if (ok) "" else " (first difference at a byte shown below)" });
        if (!ok) std.debug.print("     byte {d}: got {x} want {x}\n", .{ first, got[first], want[first] });
    }

    /// all-reduce (out of place, in place, misaligned by `skew` bytes), all-gather (out of place, in place), exchange.
    pub fn collectives(c: *Ctx, comm: Collective, label: []const u8, n_bytes: usize, seed: u64) !void {
        const w: usize = c.world;
        var views: [tp.bootstrap.max_ranks][]const u8 = undefined;
        const reduce_types = [_]DType{ .f32, .bf16, .f16, .i32 };
        for (reduce_types, 0..) |t, ti| {
            const n = n_bytes - n_bytes % t.size();
            const count = n / t.size();
            c.inputs(n, t, seed + ti);
            reference.reduce(c.want[0..n], c.all(n, &views), t, .sum);
            try c.in.uploadAsync(0, c.mine(n), c.stream.handle);
            try c.sumReference(n, t, null, c.in.ptr);
            try c.out.fill8(0xa5, c.stream.handle);
            try comm.allReduce(c.in.ptr, c.out.ptr, count, t, .sum, c.stream.handle);
            try c.compareWith(comm, label, "all-reduce sum", t, n, 0, c.want[0..n]);
            // in place, at an offset of one element (not 16-byte aligned for 2-byte types)
            const skew = t.size();
            try c.out.uploadAsync(skew, c.mine(n), c.stream.handle);
            try c.sumReference(n, t, skew, c.out.ptr + skew);
            try comm.allReduce(c.out.ptr + skew, c.out.ptr + skew, count, t, .sum, c.stream.handle);
            try c.compareWith(comm, label, "all-reduce sum in place, skewed", t, n, skew, c.want[0..n]);
        }
        for ([_]Op{ .max, .min }) |op| {
            const n = n_bytes - n_bytes % 4;
            c.inputs(n, .f32, seed + 11);
            reference.reduce(c.want[0..n], c.all(n, &views), .f32, op);
            try c.in.uploadAsync(0, c.mine(n), c.stream.handle);
            try comm.allReduce(c.in.ptr, c.out.ptr, n / 4, .f32, op, c.stream.handle);
            try c.compare(comm, label, if (op == .max) "all-reduce max" else "all-reduce min", .f32, n, 0, c.want[0..n]);
        }
        for ([_]DType{ .u8, .bf16, .f32 }, 0..) |t, ti| {
            const n = n_bytes - n_bytes % t.size();
            c.inputs(n, t, seed + 20 + ti);
            for (0..w) |r| @memcpy(c.want[r * n ..][0..n], c.ins[r][0..n]);
            try c.in.uploadAsync(0, c.mine(n), c.stream.handle);
            try comm.allGather(c.in.ptr, c.out.ptr, n / t.size(), t, c.stream.handle);
            try c.compare(comm, label, "all-gather", t, n, 0, c.want[0 .. w * n]);
            try c.out.fill8(0, c.stream.handle);
            try c.out.uploadAsync(c.rank * n, c.mine(n), c.stream.handle);
            try comm.allGather(c.out.ptr + c.rank * n, c.out.ptr, n / t.size(), t, c.stream.handle);
            try c.compare(comm, label, "all-gather in place", t, n, 0, c.want[0 .. w * n]);
            // the exchange in pairs (above two ranks the route sends it to NCCL); every rank meets the next collective
            if (c.partner()) |p| {
                try c.in.uploadAsync(0, c.mine(n), c.stream.handle);
                try comm.exchange(c.in.ptr, c.out.ptr, n / t.size(), t, p, c.stream.handle);
                try c.compare(comm, label, "exchange", t, n, 0, c.ins[p][0..n]);
            }
        }
    }

    /// send then recv one way, then the other way (NCCL behind every backend), between ranks 0 and 1; the others wait.
    pub fn pointToPoint(c: *Ctx, comm: Collective, label: []const u8, n_bytes: usize, seed: u64) !void {
        const n = n_bytes - n_bytes % 4;
        c.inputs(n, .f32, seed);
        if (c.rank > 1) return;
        try c.in.uploadAsync(0, c.mine(n), c.stream.handle);
        for (0..2) |dir| {
            const sender: u32 = @intCast(dir);
            if (c.rank == sender) {
                try comm.send(c.in.ptr, n / 4, .f32, 1 - c.rank, c.stream.handle);
                try c.stream.synchronize();
            } else {
                try comm.recv(c.out.ptr, n / 4, .f32, sender, c.stream.handle);
                try c.compare(comm, label, "send/recv", .f32, n, 0, c.ins[sender][0..n]);
            }
        }
    }

    /// A captured chain replayed with fresh inputs, eager calls between replays (the mailbox epoch spans both).
    pub fn graphs(c: *Ctx, comm: Collective, label: []const u8, n_bytes: usize, replays: u32) !void {
        const w: usize = c.world;
        const n = n_bytes - n_bytes % 16;
        const half = n / 2; // bf16 elements
        const peer = c.partner();
        try cuda.graph.beginCapture(c.stream, .thread_local);
        // x1 = sum(x0) [bf16], x2 = gather(x1) [w n], x3 = sum(x2) [f32 view], x4 = exchange(x3) (in pairs)
        const x0 = c.in.ptr;
        const x1 = c.out.ptr;
        const x2 = c.in.ptr + n;
        const x3 = c.out.ptr + n;
        const x4 = c.out.ptr + (1 + w) * n;
        const ops: ?tp.collective.Error = blk: {
            comm.allReduce(x0, x1, half, .bf16, .sum, c.stream.handle) catch |e| break :blk e;
            comm.allGather(x1, x2, half, .bf16, c.stream.handle) catch |e| break :blk e;
            comm.allReduce(x2, x3, w * n / 4, .f32, .sum, c.stream.handle) catch |e| break :blk e;
            if (peer) |p| comm.exchange(x3, x4, w * n / 4, .f32, p, c.stream.handle) catch |e| break :blk e;
            break :blk null;
        };
        var g = try cuda.graph.endCapture(c.stream);
        defer g.deinit();
        if (ops) |e| return e;
        var exec = try g.instantiate();
        defer exec.deinit();
        const nodes = try g.nodeCount();
        const want1 = try c.gpa.alloc(u8, w * n);
        defer c.gpa.free(want1);
        const want3 = try c.gpa.alloc(u8, w * n);
        defer c.gpa.free(want3);
        var views: [tp.bootstrap.max_ranks][]const u8 = undefined;
        const want4 = try c.gpa.alloc(u8, w * n);
        defer c.gpa.free(want4);
        for (0..replays) |k| {
            c.inputs(n, .bf16, 1000 + k);
            try c.in.uploadAsync(0, c.mine(n), c.stream.handle);
            reference.reduce(want1[0..n], c.all(n, &views), .bf16, .sum);
            for (1..w) |r| @memcpy(want1[r * n ..][0..n], want1[0..n]); // the gather of equal sums: w times the same
            for (views[0..w]) |*v| v.* = want1[0 .. w * n];
            reference.reduce(want3[0 .. w * n], views[0..w], .f32, .sum);
            if (c.world > 2) {
                const slow = c.baseline orelse return error.MissingNcclReference;
                try slow.allReduce(x0, x1, half, .bf16, .sum, c.stream.handle);
                try slow.allGather(x1, x2, half, .bf16, c.stream.handle);
                try slow.allReduce(x2, x3, w * n / 4, .f32, .sum, c.stream.handle);
                if (peer) |p| try slow.exchange(x3, x4, w * n / 4, .f32, p, c.stream.handle);
                try c.stream.synchronize();
                try c.out.download(0, want1[0..n]);
                try c.out.download(n, want3);
                if (peer != null) try c.out.download((1 + w) * n, want4);
            } else @memcpy(want4, want3);
            @memset(c.want[0 .. w * n], 0xa5);
            try c.in.uploadAsync(n, c.want[0 .. w * n], c.stream.handle);
            try c.out.fill8(0xa5, c.stream.handle);
            try exec.launchOn(c.stream);
            try c.compare(comm, label, "graph replay: first sum", .bf16, n, 0, want1[0..n]);
            try c.compare(comm, label, "graph replay: sum, gather, sum", .f32, w * n, n, want3);
            if (peer != null) try c.compare(comm, label, "graph replay: exchange of the sum", .f32, w * n, (1 + w) * n, want4);
            // an eager call between replays
            try c.collectivesQuick(comm, label, k);
        }
        std.debug.print("     ({s}: {d} graph nodes, {d} replays)\n", .{ label, nodes, replays });
    }

    fn collectivesQuick(c: *Ctx, comm: Collective, label: []const u8, k: usize) !void {
        const n = 4096;
        var views: [tp.bootstrap.max_ranks][]const u8 = undefined;
        c.inputs(n, .bf16, 5000 + k);
        reference.reduce(c.want[0..n], c.all(n, &views), .bf16, .sum);
        try c.in.uploadAsync(5 * n, c.mine(n), c.stream.handle);
        try c.sumReference(n, .bf16, null, c.in.ptr + 5 * n);
        try comm.allReduce(c.in.ptr + 5 * n, c.out.ptr + 5 * n, n / 2, .bf16, .sum, c.stream.handle);
        try c.compareWith(comm, label, "eager between replays", .bf16, n, 5 * n, c.want[0..n]);
    }
};
