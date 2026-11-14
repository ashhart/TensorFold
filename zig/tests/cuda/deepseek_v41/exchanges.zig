//! `tf-dsv41-test exchanges`: the family's exchanges at worlds 1 to 4 as threads on one GPU (tp.device), bit for bit.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const tp = @import("tp");
const partition = @import("dsv41_partition");
const check = @import("../check.zig");

const Gpu = check.Gpu;
const Collective = tp.Collective;
const max = tp.device.max_ranks;

const D = 5120; // config.zig's hidden
const vocab = 1280; // ten 128-row blocks: 4/3/3 at world 3
const n = 7;
/// The vocabulary's ends and both sides of every share edge at worlds 2 to 4, where `partition.share` puts them.
const edges = edges: {
    var ids: []const i64 = &.{ 0, vocab - 1 };
    for (worlds[1..]) |w| for (1..w) |r| {
        const lo: i64 = (partition.share(vocab, r, w) catch unreachable).lo;
        ids = ids ++ &[_]i64{ lo - 1, lo };
    };
    break :edges ids[0..ids.len].*;
};
const ne = edges.len + 1; // and one id at random
const worlds = [_]u32{ 1, 2, 3, 4 };

fn bf16(f: f32) u16 {
    const u: u32 = @bitCast(f);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// Finite values in [-amp, amp], never +-0 (embed_sum's +0 from the other ranks would flip a -0).
fn fill(rnd: std.Random, out: []u16, amp: f32) void {
    for (out) |*v| {
        v.* = bf16((rnd.float(f32) * 2 - 1) * amp);
        if (v.* & 0x7fff == 0) v.* = 0x3f80;
    }
}

fn fillF32(rnd: std.Random, out: []f32, lo: f32, hi: f32) void {
    for (out) |*v| v.* = lo + rnd.float(f32) * (hi - lo);
}

fn bytes(s: anytype) []const u8 {
    return std.mem.sliceAsBytes(s);
}

/// One device buffer per input, freed together.
const Bufs = struct {
    d: *const cuda.Driver,
    list: [24]cuda.DeviceBuffer = undefined,
    len: usize = 0,

    fn keep(b: *Bufs, buf: cuda.DeviceBuffer) cuda.DeviceBuffer {
        std.debug.assert(b.len < b.list.len);
        b.list[b.len] = buf;
        b.len += 1;
        return buf;
    }
    fn up(b: *Bufs, host: []const u8) !u64 {
        return b.keep(try cuda.DeviceBuffer.fromHost(b.d, host)).ptr;
    }
    fn zeros(b: *Bufs, len: usize) !cuda.DeviceBuffer {
        const buf = b.keep(try cuda.DeviceBuffer.alloc(b.d, @max(len, 4)));
        try buf.fill8(0, null);
        return buf;
    }
    fn deinit(b: *Bufs) void {
        for (b.list[0..b.len]) |*x| x.free();
    }
};

const Embed = struct {
    gpu: Gpu,
    k: *const dsv41.Kernels,
    table: []const u16,
    ids: []const i64,
    got: [max][]u8,

    fn go(c: Collective, rank: u32, e: *Embed) anyerror!void {
        const W = c.world();
        const part = try partition.share(vocab, rank, W);
        var stream = try cuda.Stream.init(e.gpu.d, false);
        defer stream.deinit();
        var b: Bufs = .{ .d = e.gpu.d };
        defer b.deinit();
        const tab = try b.up(bytes(e.table[part.lo * D ..][0 .. part.len * D]));
        const ids = try b.up(bytes(e.ids));
        const rows = (try b.zeros(ne * D * 2)).ptr;
        const recv = (try b.zeros(W * ne * D * 2)).ptr;
        const out = try b.zeros(ne * 4 * D * 2);
        const o = e.k.others(stream);
        try o.embedSend(ids, ne, tab, @intCast(part.lo), part.len, D, rows);
        try c.allGather(rows, recv, ne * D, .bf16, stream.handle);
        try o.embedSum(recv, W, ne, D, out.ptr);
        try stream.synchronize();
        try out.download(0, e.got[rank]);
    }
};

/// Embedding rows from uneven vocabulary shares, gathered and summed: every rank holds each id's row x4.
fn embed(gpu: Gpu, k: *const dsv41.Kernels, rnd: std.Random) !void {
    const table = try gpu.gpa.alloc(u16, vocab * D);
    defer gpu.gpa.free(table);
    fill(rnd, table, 2.0);
    const ids = edges ++ [_]i64{@intCast(rnd.uintLessThan(u32, vocab))};
    const want = try gpu.gpa.alloc(u16, ne * 4 * D);
    defer gpu.gpa.free(want);
    for (ids, 0..) |id, i| for (0..4) |s| @memcpy(want[(i * 4 + s) * D ..][0..D], table[@as(usize, @intCast(id)) * D ..][0..D]);
    var e: Embed = .{ .gpu = gpu, .k = k, .table = table, .ids = &ids, .got = undefined };
    const store = try gpu.gpa.alloc(u8, max * ne * 4 * D * 2);
    defer gpu.gpa.free(store);
    for (&e.got, 0..) |*g, r| g.* = store[r * ne * 4 * D * 2 ..][0 .. ne * 4 * D * 2];
    var what: [64]u8 = undefined;
    for (worlds) |w| {
        try tp.device.run(gpu.ctx, std.heap.smp_allocator, w, .{}, &e, Embed.go);
        for (0..w) |r| try check.sameBytes(try std.fmt.bufPrint(&what, "embed world {d} rank {d}", .{ w, r }), e.got[r], bytes(want));
    }
    check.pass("BITEXACT exchanges embed: {d} ids, every share edge, over vocab {d} at worlds 1-4 (4/3/3 at 3) == each row x4", .{ ne, vocab });
}

const Cols = struct {
    gpu: Gpu,
    k: *const dsv41.Kernels,
    cols: usize,
    send: []const u16, // rank r's [n, cols] at r * n * cols
    got: [max][]u8,

    fn go(c: Collective, rank: u32, e: *Cols) anyerror!void {
        const W = c.world();
        var stream = try cuda.Stream.init(e.gpu.d, false);
        defer stream.deinit();
        var b: Bufs = .{ .d = e.gpu.d };
        defer b.deinit();
        const send = try b.up(bytes(e.send[rank * n * e.cols ..][0 .. n * e.cols]));
        const recv = (try b.zeros(W * n * e.cols * 2)).ptr;
        const out = try b.zeros(n * W * e.cols * 2);
        try c.allGather(send, recv, n * e.cols, .bf16, stream.handle);
        try e.k.others(stream).gatherCols(recv, W, n, e.cols, out.ptr);
        try stream.synchronize();
        try out.download(0, e.got[rank][0..out.len]);
    }
};

/// Engram's row exchange: each rank's columns gathered, then laid side by side in rank order.
fn cols(gpu: Gpu, k: *const dsv41.Kernels, rnd: std.Random) !void {
    const cw = 8 * 64; // eight heads of a 64-wide share each
    const send = try gpu.gpa.alloc(u16, max * n * cw);
    defer gpu.gpa.free(send);
    fill(rnd, send, 4.0);
    var e: Cols = .{ .gpu = gpu, .k = k, .cols = cw, .send = send, .got = undefined };
    const store = try gpu.gpa.alloc(u8, max * 4 * n * cw * 2);
    defer gpu.gpa.free(store);
    for (&e.got, 0..) |*g, r| g.* = store[r * 4 * n * cw * 2 ..][0 .. 4 * n * cw * 2];
    const want = try gpu.gpa.alloc(u16, 4 * n * cw);
    defer gpu.gpa.free(want);
    var what: [64]u8 = undefined;
    for (worlds) |w| {
        for (0..n) |i| for (0..w) |q| @memcpy(want[(i * w + q) * cw ..][0..cw], send[(q * n + i) * cw ..][0..cw]);
        try tp.device.run(gpu.ctx, std.heap.smp_allocator, w, .{}, &e, Cols.go);
        const len = w * n * cw * 2;
        for (0..w) |r| try check.sameBytes(try std.fmt.bufPrint(&what, "engram cols world {d} rank {d}", .{ w, r }), e.got[r][0..len], bytes(want)[0..len]);
    }
    check.pass("BITEXACT exchanges engram: {d} rows of {d} columns a rank at worlds 1-4 == the rank-order layout", .{ n, cw });
}

/// One mHC site over gathered partials: `pf` mhc_pf's site (PFDEC, past 16 rows), else mhc_cuda's boundary.
const Fold = struct {
    gpu: Gpu,
    k: *const dsv41.Kernels,
    pf: bool,
    R: usize,
    x: []const u16, // [R, 4 D] streams
    parts: [max][]const u16, // rank r's [R, D] partial at this world
    fnw: []const u16, // [24, 4 D]
    base: []const f32,
    scale: []const f32,
    nw: []const f32,
    coef: []const f32, // pre [R, 4], post [R, 4], comb [R, 16]
    got: [max][]u8,

    fn go(c: Collective, rank: u32, e: *Fold) anyerror!void {
        const W = c.world();
        const R = e.R;
        var stream = try cuda.Stream.init(e.gpu.d, false);
        defer stream.deinit();
        var b: Bufs = .{ .d = e.gpu.d };
        defer b.deinit();
        const x = try b.up(bytes(e.x));
        const send = try b.up(bytes(e.parts[rank]));
        const recv = (try b.zeros(W * R * D * 2)).ptr;
        const coef = try b.up(bytes(e.coef));
        const outs = [_]cuda.DeviceBuffer{
            try b.zeros(R * 4 * D * 2), // new streams
            try b.zeros(R * 4 * 40 * 32 * 4), // partials
            try b.zeros(R * D * 2), // collapsed row
            try b.zeros(R * D * 2), // normed input (boundary)
            try b.zeros(R * 24 * 4), // next coefficients (boundary)
        };
        try c.allGather(send, recv, R * D, .bf16, stream.handle);
        const o = e.k.others(stream);
        if (e.pf) {
            try o.mhcSite(.{ .x = x, .xs = 4 * D, .xout = outs[0].ptr, .g = recv, .gr = @intCast(R * D), .world = @intCast(W), .gbf16 = 1, .post = coef + R * 16, .comb = coef + R * 32, .pre = coef, .@"fn" = try b.up(bytes(e.fnw)), .part = outs[1].ptr, .c = outs[2].ptr, .R = @intCast(R) }, 0, false, false);
        } else {
            const nc = outs[4].ptr;
            try o.mhcBoundary(.{
                .x = x,
                .xs = 4 * D,
                .xout = outs[0].ptr,
                .g = recv,
                .gr = @intCast(R * D),
                .world = @intCast(W),
                .gbf16 = 1,
                .post = coef + R * 16,
                .comb = coef + R * 32,
                .pre = coef,
                .@"fn" = try b.up(bytes(e.fnw)),
                .base = try b.up(bytes(e.base)),
                .scale = try b.up(bytes(e.scale)),
                .part = outs[1].ptr,
                .c = outs[2].ptr,
                .nw = try b.up(bytes(e.nw)),
                .out = outs[3].ptr,
                .opre = nc,
                .opost = nc + R * 16,
                .ocomb = nc + R * 32,
                .cnt = (try b.zeros(256)).ptr,
                .R = @intCast(R),
                .eps = 1e-6,
                .hc_eps = 1e-6,
                .post_alpha = 2.0,
                .iters = 20,
            }, 0, false);
        }
        try stream.synchronize();
        var at: usize = 0;
        for (outs) |buf| {
            try buf.download(0, e.got[rank][at..][0..buf.len]);
            at += buf.len;
        }
    }

    fn outBytes(R: usize) usize {
        return R * 4 * D * 2 + R * 4 * 40 * 32 * 4 + 2 * R * D * 2 + R * 24 * 4;
    }
};

/// Worlds 2-4 against world 1 holding the sum: multiples of 1/16 under 16 in size are exact in bf16 and in fp32 sums.
fn fold(gpu: Gpu, k: *const dsv41.Kernels, rnd: std.Random, pf: bool, R: usize) !void {
    const gpa = gpu.gpa;
    const x = try gpa.alloc(u16, R * 4 * D);
    defer gpa.free(x);
    fill(rnd, x, 1.0);
    const fnw = try gpa.alloc(u16, 24 * 4 * D);
    defer gpa.free(fnw);
    fill(rnd, fnw, 0.02);
    var base: [24]f32 = undefined;
    fillF32(rnd, &base, -0.5, 0.5);
    var scale: [3]f32 = undefined;
    fillF32(rnd, &scale, 0.5, 1.5);
    const nw = try gpa.alloc(f32, D);
    defer gpa.free(nw);
    fillF32(rnd, nw, 0.8, 1.2);
    const coef = try gpa.alloc(f32, R * 24);
    defer gpa.free(coef);
    fillF32(rnd, coef, 0.0, 0.5);
    // sums in [-96, 96] sixteenths; ranks before the last [-32, 32], the last the rest
    const sum = try gpa.alloc(i32, R * D);
    defer gpa.free(sum);
    for (sum) |*s| s.* = rnd.intRangeAtMost(i32, -96, 96);
    const parts = try gpa.alloc(u16, max * R * D);
    defer gpa.free(parts);
    const len = Fold.outBytes(R);
    const want = try gpa.alloc(u8, len);
    defer gpa.free(want);
    const store = try gpa.alloc(u8, max * len);
    defer gpa.free(store);
    var e: Fold = .{ .gpu = gpu, .k = k, .pf = pf, .R = R, .x = x, .parts = undefined, .fnw = fnw, .base = &base, .scale = &scale, .nw = nw, .coef = coef, .got = undefined };
    for (&e.got, 0..) |*g, r| g.* = store[r * len ..][0..len];
    var what: [64]u8 = undefined;
    const name = if (pf) "mhc_pf site" else "mhc boundary";
    for ([_]u32{ 1, 1, 2, 3, 4 }, 0..) |w, round| {
        for (0..R * D) |i| {
            var left = sum[i];
            for (0..w) |r| {
                const v = if (r + 1 == w) left else rnd.intRangeAtMost(i32, -32, 32);
                left -= v;
                parts[r * R * D + i] = bf16(@as(f32, @floatFromInt(v)) / 16.0);
            }
        }
        for (&e.parts, 0..) |*p, r| p.* = parts[r * R * D ..][0 .. R * D];
        try tp.device.run(gpu.ctx, std.heap.smp_allocator, w, .{}, &e, Fold.go);
        // the first round is the reference; the second shows the kernel repeats its bytes
        if (round == 0) {
            @memcpy(want, e.got[0]);
            continue;
        }
        for (0..w) |r| try check.sameBytes(try std.fmt.bufPrint(&what, "{s} R {d} world {d} rank {d}", .{ name, R, w, r }), e.got[r], want);
    }
    check.pass("BITEXACT exchanges {s}: R {d}, partials of worlds 2-4 == world 1 holding their exact sum, every output", .{ name, R });
}

pub fn run(gpu: Gpu, k: *const dsv41.Kernels) !void {
    var prng = std.Random.DefaultPrng.init(2270);
    const rnd = prng.random();
    try embed(gpu, k, rnd);
    try cols(gpu, k, rnd);
    try fold(gpu, k, rnd, false, 5);
    try fold(gpu, k, rnd, true, 20);
}
