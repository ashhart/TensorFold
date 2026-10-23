//! Every collective on real GPUs against the host reference, eager and replayed from a graph; 2 to 16 ranks.
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

    pub fn init(d: *const cuda.Driver, gpa: std.mem.Allocator, rank: u32, world: u32, max_bytes: usize) !Ctx {
        const ins = try gpa.alloc([]u8, world);
        for (ins) |*b| b.* = try gpa.alloc(u8, max_bytes);
        return .{
            .d = d,
            .gpa = gpa,
            .stream = try cuda.Stream.init(d, true),
            .rank = rank,
            .world = world,
            .in = try cuda.DeviceBuffer.alloc(d, (world + 2) * max_bytes + 256),
            .out = try cuda.DeviceBuffer.alloc(d, (world + 2) * max_bytes + 256),
            .ins = ins,
            .want = try gpa.alloc(u8, (world + 2) * max_bytes),
            .got = try gpa.alloc(u8, (world + 2) * max_bytes),
        };
    }

    pub fn deinit(c: *Ctx) void {
        c.stream.deinit();
        c.in.free();
        c.out.free();
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

    /// Whether a sum is promised in rank order (bit-exact): integers, two ranks, or the route's own f32/bf16 kernel.
    fn sumExact(c: *const Ctx, comm: Collective, t: DType, bytes: usize) bool {
        if (t == .i32 or t == .i64 or t == .u8 or c.world == 2) return true;
        return comm.kind() != .nccl and (t == .f32 or t == .bf16) and comm.varFits(bytes);
    }

    fn compare(c: *Ctx, comm: Collective, label: []const u8, what: []const u8, t: DType, n: usize, out_off: usize, want: []const u8) !void {
        return c.compareWith(comm, label, what, t, n, out_off, want, true);
    }

    /// `exact`: bit for bit; else float elements within the rounding of adds in another order (NCCL above two ranks).
    fn compareWith(c: *Ctx, comm: Collective, label: []const u8, what: []const u8, t: DType, n: usize, out_off: usize, want: []const u8, exact: bool) !void {
        try c.stream.synchronize();
        comm.check() catch |e| {
            c.verdict(false, "{s} {s} {t} {d} B: transport error {t}", .{ label, what, t, n, e });
            return;
        };
        try c.out.download(out_off, c.got[0..want.len]);
        const got = c.got[0..want.len];
        const ok = if (exact) std.mem.eql(u8, got, want) else close(got, want, t, c.world);
        var first: usize = 0;
        if (!ok) first = std.mem.indexOfDiff(u8, got, want) orelse 0;
        c.verdict(ok, "{s} {s} {t} {d} B a rank{s}{s}", .{ label, what, t, n, if (exact) "" else " (within rounding: NCCL's ring order is not rank order)", if (ok) "" else " (first difference at a byte shown below)" });
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
            const exact = c.sumExact(comm, t, n);
            reference.reduce(c.want[0..n], c.all(n, &views), t, .sum);
            try c.in.uploadAsync(0, c.mine(n), c.stream.handle);
            try comm.allReduce(c.in.ptr, c.out.ptr, count, t, .sum, c.stream.handle);
            try c.compareWith(comm, label, "all-reduce sum", t, n, 0, c.want[0..n], exact);
            // in place, at an offset of one element (not 16-byte aligned for 2-byte types)
            const skew = t.size();
            try c.out.uploadAsync(skew, c.mine(n), c.stream.handle);
            try comm.allReduce(c.out.ptr + skew, c.out.ptr + skew, count, t, .sum, c.stream.handle);
            try c.compareWith(comm, label, "all-reduce sum in place, skewed", t, n, skew, c.want[0..n], exact);
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
        // each sum is rank order only where the route's kernel adds it (the second is w times larger)
        const exact1 = c.sumExact(comm, .bf16, n);
        const exact3 = c.sumExact(comm, .f32, w * n);
        for (0..replays) |k| {
            c.inputs(n, .bf16, 1000 + k);
            try c.in.uploadAsync(0, c.mine(n), c.stream.handle);
            reference.reduce(want1[0..n], c.all(n, &views), .bf16, .sum);
            for (1..w) |r| @memcpy(want1[r * n ..][0..n], want1[0..n]); // the gather of equal sums: w times the same
            for (views[0..w]) |*v| v.* = want1[0 .. w * n];
            reference.reduce(want3[0 .. w * n], views[0..w], .f32, .sum);
            try exec.launchOn(c.stream);
            if (!exact1) {
                // NCCL's ring above two ranks: its bf16 sum within rounding, then the rest from its own bits
                try c.compareWith(comm, label, "graph replay: sum", .bf16, n, 0, want1[0..n], false);
                @memcpy(want1[0..n], c.got[0..n]);
                for (1..w) |r| @memcpy(want1[r * n ..][0..n], want1[0..n]);
                reference.reduce(want3[0 .. w * n], views[0..w], .f32, .sum);
            }
            try c.compareWith(comm, label, "graph replay: sum, gather, sum", .f32, w * n, n, want3[0 .. w * n], exact3);
            if (peer != null) try c.compareWith(comm, label, "graph replay: exchange of the sum", .f32, w * n, (1 + w) * n, want3[0 .. w * n], exact3);
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
        try comm.allReduce(c.in.ptr + 5 * n, c.out.ptr + 5 * n, n / 2, .bf16, .sum, c.stream.handle);
        try c.compareWith(comm, label, "eager between replays", .bf16, n, 5 * n, c.want[0..n], c.sumExact(comm, .bf16, n));
    }
};

/// Whether float elements of `t` are within a few roundings of the rank-order sum of `world` operands.
fn close(got: []const u8, want: []const u8, t: DType, world: u32) bool {
    const steps: f32 = @floatFromInt(world);
    switch (t) {
        .bf16, .f16 => {
            const eps: f32 = if (t == .bf16) 1.0 / 128.0 else 1.0 / 1024.0;
            for (0..got.len / 2) |i| {
                const a = halfToF32(std.mem.readInt(u16, got[2 * i ..][0..2], .little), t);
                const b = halfToF32(std.mem.readInt(u16, want[2 * i ..][0..2], .little), t);
                if (@abs(a - b) > steps * eps * @max(@abs(b), 4.0 * steps)) return false;
            }
            return true;
        },
        .f32 => {
            for (0..got.len / 4) |i| {
                const a: f32 = @bitCast(std.mem.readInt(u32, got[4 * i ..][0..4], .little));
                const b: f32 = @bitCast(std.mem.readInt(u32, want[4 * i ..][0..4], .little));
                if (@abs(a - b) > steps * 1e-6 * @max(@abs(b), 4.0 * steps)) return false;
            }
            return true;
        },
        else => return std.mem.eql(u8, got, want),
    }
}

fn halfToF32(bits: u16, t: DType) f32 {
    return if (t == .bf16) reference.bf16ToF32(bits) else @floatCast(@as(f16, @bitCast(bits)));
}
