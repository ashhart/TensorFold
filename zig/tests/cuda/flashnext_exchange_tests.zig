//! Two ranks' prompt-pass expert exchange against the all-reduce it replaces: the written rows, sums and norms' bytes.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");

const Collective = flash.collective.Collective;
const dims = 2560;
const slots = 11;
const streams = 4;
const wide = dims * streams;
const pss_n = dims / 256 * streams;
const eps: f32 = 1e-6;
const neg_zero: u32 = 0x8000_0000;

/// -0, +0, the least subnormals, the largest subnormal and finite values, the infinities, a NaN and two plain values.
const specials = [_]u32{ neg_zero, 0, 1, 0x8000_0001, 0x007f_ffff, 0x7f7f_ffff, 0xff7f_ffff, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0x3f80_0000, 0xbe80_0000 };

pub fn run(gpu: check.Gpu, c: *Collective, stream: cuda.Stream, rank: usize) !void {
    try blocks(gpu, c, stream, rank);
    try planted(gpu, c, stream, rank);
    var glue = try flash.Glue.load(gpu.d);
    defer glue.deinit();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_batch);
    defer mod.unload();
    const k: Kernels = .{ .moe_sum = try mod.function("fn_moe_sum") };
    const tri: flash.Tri = .{ .s = stream, .own = &glue };
    for ([_]usize{ 65, 66, 67, 255, 1536, 1537, 8192 }) |n| try pass(gpu, c, stream, tri, k, rank, n);
    check.pass("rank {d}: the exchange's written rows, square sums and normalized rows equal the all-reduce's for 65 to 8,192 rows, odd and even; a changed sum fails", .{rank});
}

/// sumBlock leaves block `rank` of the two ranks' values summed exactly, from one value to a vocabulary half.
fn blocks(gpu: check.Gpu, c: *Collective, stream: cuda.Stream, rank: usize) !void {
    for ([_]usize{ 1, 17, 4099, 124_160 }) |count| {
        const host = try gpu.gpa.alloc(f32, 2 * count);
        defer gpu.gpa.free(host);
        for (host, 0..) |*v, i| v.* = @floatFromInt(i % 127 + rank);
        var values = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
        defer values.free();
        try c.sumBlock(stream, values.ptr, count);
        try c.finish(stream);
        try values.download(0, std.mem.sliceAsBytes(host));
        for (host[rank * count ..][0..count], rank * count..) |v, at| try check.expect(v == @as(f32, @floatFromInt(2 * (at % 127) + 1)), "rank {d}: reduce-scatter block at {d} of {d}", .{ rank, at, count });
    }
    check.pass("rank {d}: in-place FP32 reduce-scatter exact for a scalar, tail and vocabulary-half blocks", .{rank});
}

/// One rank's specials beside the other's +0, then both ranks' values: the block's bytes equal the all-reduce's.
fn planted(gpu: check.Gpu, c: *Collective, stream: cuda.Stream, rank: usize) !void {
    const m = specials.len * 3;
    var host: [2 * m]u32 = undefined;
    for (&host, 0..) |*v, i| {
        const j = i % specials.len;
        const case = (i / specials.len) % 3;
        v.* = if (case == 2) specials[(j + rank * 5) % specials.len] else if (case == rank) specials[j] else 0;
    }
    var all = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&host));
    defer all.free();
    var mine = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&host));
    defer mine.free();
    try c.sum(stream, all.ptr, 2 * m);
    try c.sumBlock(stream, mine.ptr, m);
    try c.finish(stream);
    var got_all: [2 * m]u32 = undefined;
    var got: [2 * m]u32 = undefined;
    try all.download(0, std.mem.asBytes(&got_all));
    try mine.download(0, std.mem.asBytes(&got));
    try check.sameBytes("reduce-scatter block against the all-reduce, planted", std.mem.sliceAsBytes(got[rank * m ..][0..m]), std.mem.sliceAsBytes(got_all[rank * m ..][0..m]));
    for (rank * m..rank * m + m) |i| {
        const j = i % specials.len;
        if ((i / specials.len) % 3 == 2) continue;
        const v: f32 = @bitCast(specials[j]);
        const want: u32 = if (specials[j] == neg_zero) 0 else specials[j];
        try check.expect(if (std.math.isNan(v)) std.math.isNan(@as(f32, @bitCast(got[i]))) else got[i] == want, "rank {d}: x + 0 keeps x (-0 gives +0) at {d}: {x:0>8} from {x:0>8}", .{ rank, i, got[i], specials[j] });
    }
    check.pass("rank {d}: reduce-scatter equals the all-reduce on planted -0, subnormals, extremes, infinities, NaN and two-rank values", .{rank});
}

const Kernels = struct { moe_sum: cuda.Function };
const Bufs = struct { y: u64, wts: u64, moe_f32: u64 };
const Pass = struct { s: *const Bufs, k: Kernels, stream: cuda.Stream, pending_f32: bool = false };

/// A writeback's outputs over the pass's rows and the spare one, poisoned so a row no launch writes shows.
const Out = struct {
    hout: cuda.DeviceBuffer,
    pss: cuda.DeviceBuffer,
    norm: cuda.DeviceBuffer,

    fn init(d: *const cuda.Driver, stream: cuda.Stream, rows: usize) !Out {
        var o: Out = .{ .hout = try cuda.DeviceBuffer.alloc(d, rows * wide * 2), .pss = undefined, .norm = undefined };
        errdefer o.hout.free();
        o.pss = try cuda.DeviceBuffer.alloc(d, rows * pss_n * 4);
        errdefer o.pss.free();
        o.norm = try cuda.DeviceBuffer.alloc(d, rows * wide * 2);
        for ([_]cuda.DeviceBuffer{ o.hout, o.pss, o.norm }) |b| try b.fill8(0x5a, stream.handle);
        return o;
    }

    fn deinit(o: *Out) void {
        o.hout.free();
        o.pss.free();
        o.norm.free();
    }

    fn same(o: Out, gpu: check.Gpu, want: Out, what: []const u8) !void {
        inline for (.{ "hout", "pss", "norm" }) |name| {
            const got = try check.download(gpu, @field(o, name));
            defer gpu.gpa.free(got);
            const ref = try check.download(gpu, @field(want, name));
            defer gpu.gpa.free(ref);
            try check.sameBytes(what, got, ref);
        }
    }
};

fn mix(x: u64) u64 {
    var z = x +% 0x9e37_79b9_7f4a_7c15;
    z = (z ^ (z >> 30)) *% 0xbf58_476d_1ce4_e5b9;
    z = (z ^ (z >> 27)) *% 0x94d0_49bb_1331_11eb;
    return z ^ (z >> 31);
}

/// A value in [-1, 1) times 2^-16 to 2^7, or now and then one of the specials.
fn value(h: u64) f32 {
    if (h % 4093 < 4) return @bitCast(specials[@intCast((h >> 20) % specials.len)]);
    const unit = @as(f32, @floatFromInt(h >> 40)) / (1 << 24);
    return (unit * 2 - 1) * std.math.ldexp(@as(f32, 1), @as(i32, @intCast((h >> 8) % 24)) - 16);
}

fn bf16(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

/// Routed pairs owned by a rank drawn per pair, the shared slot's columns in halves; a few elements on both ranks.
fn fill(y: []f32, wts: []f32, n: usize, rank: usize) void {
    @memset(y, 0);
    @memset(wts, 0);
    for (0..n) |r| for (0..slots) |k| {
        const pair = r * slots + k;
        const w = mix(pair ^ 0x77);
        wts[pair] = if (w % 211 == 5) @bitCast(neg_zero) else if (w % 211 == 6) @bitCast(@as(u32, 3)) else @as(f32, @floatFromInt(w >> 40)) / (1 << 26);
        if (r == 3 and k == 4) wts[pair] = std.math.nan(f32);
        if (r == 5 and k == 2) wts[pair] = std.math.inf(f32);
        const owner = mix(pair ^ 0x3a) & 1;
        for (0..dims) |d| {
            const at = pair * dims + d;
            const h = mix(at);
            const own = if (k == slots - 1) (d >= dims / 2) == (rank == 1) else owner == rank;
            if (own) y[at] = value(h) else if (h % 50_021 == 7) y[at] = value(mix(h)) * 0.75;
        }
    };
}

/// The all-reduce and both mode-2 writebacks against the exchange and both mode-4 writebacks, on the same inputs.
fn pass(gpu: check.Gpu, c: *Collective, stream: cuda.Stream, tri: flash.Tri, k: Kernels, rank: usize, n: usize) !void {
    const a = gpu.gpa;
    const rows = n + 1;
    const y = try a.alloc(f32, rows * slots * dims);
    defer a.free(y);
    const wts = try a.alloc(f32, rows * slots);
    defer a.free(wts);
    fill(y, wts, n, rank);
    const hs = try a.alloc(u16, n * wide + n * streams);
    defer a.free(hs);
    for (hs, 0..) |*v, i| v.* = bf16(@as(f32, @floatFromInt(mix(i ^ 0x51) >> 40)) / (1 << 23) - 1);
    for (hs[n * wide ..], 0..) |*v, i| v.* = bf16(0.5 + @as(f32, @floatFromInt(mix(i ^ 0x1f) >> 40)) / (1 << 24));
    var scale: [wide]f32 = undefined;
    for (&scale, 0..) |*v, i| v.* = 0.5 + @as(f32, @floatFromInt(mix(i ^ 0x2b) >> 40)) / (1 << 24);
    var y_a = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(y));
    defer y_a.free();
    // the exchange's y and moe_f32 carry a poisoned guard row past the spare one
    const guarded = try a.alloc(f32, (rows + 1) * slots * dims);
    defer a.free(guarded);
    @memcpy(guarded[0..y.len], y);
    @memset(std.mem.sliceAsBytes(guarded[y.len..]), 0x5a);
    var y_b = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(guarded));
    defer y_b.free();
    var wd = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(wts));
    defer wd.free();
    var hd = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(hs));
    defer hd.free();
    var sd = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&scale));
    defer sd.free();
    var moe = try cuda.DeviceBuffer.alloc(gpu.d, (rows + 1) * dims * 4);
    defer moe.free();
    try moe.fill8(0x5a, stream.handle);
    var dummy = try cuda.DeviceBuffer.alloc(gpu.d, 256);
    defer dummy.free();
    const h = hd.ptr;
    const inj = hd.ptr + n * wide * 2;
    const sc = sd.ptr;
    var outs: [5]Out = undefined;
    var made: usize = 0;
    defer for (outs[0..made]) |*o| o.deinit();
    while (made < outs.len) : (made += 1) outs[made] = try Out.init(gpu.d, stream, rows);
    const rs = n * dims;

    try c.sum(stream, y_a.ptr, n * slots * dims);
    try tri.hcWriteback(h, outs[0].hout.ptr, outs[0].pss.ptr, dummy.ptr, "*bf16", inj, y_a.ptr, "*fp32", wd.ptr, "*fp32", rs, n, dims, streams, 2, 10, 11, 1);
    try tri.hcWritebackNorm(h, outs[1].hout.ptr, outs[1].pss.ptr, dummy.ptr, inj, y_a.ptr, wd.ptr, sc, outs[1].norm.ptr, eps, rs, n, dims, streams, 2, 10, 11, 1);

    const bufs: Bufs = .{ .y = y_b.ptr, .wts = wd.ptr, .moe_f32 = moe.ptr };
    var p: Pass = .{ .s = &bufs, .k = k, .stream = stream };
    try flash.moe_exchange.run(&p, c, n);
    try check.expect(p.pending_f32, "rank {d}: the exchange leaves its sum pending in fp32", .{rank});
    try tri.hcWriteback(h, outs[2].hout.ptr, outs[2].pss.ptr, moe.ptr, "*fp32", inj, dummy.ptr, "*bf16", dummy.ptr, "*bf16", rs, n, dims, streams, 4, 1, 1, 1);
    try tri.hcWritebackNorm(h, outs[3].hout.ptr, outs[3].pss.ptr, moe.ptr, inj, dummy.ptr, dummy.ptr, sc, outs[3].norm.ptr, eps, rs, n, dims, streams, 4, 1, 1, 1);
    try c.finish(stream);
    try outs[2].same(gpu, outs[0], "exchange then writeback against all-reduce then writeback");
    try outs[3].same(gpu, outs[1], "exchange then writeback-norm against all-reduce then writeback-norm");
    const tail = try check.download(gpu, outs[3].hout);
    defer a.free(tail);
    for (tail[n * wide * 2 ..]) |b| try check.expect(b == 0x5a, "rank {d}: the spare row past {d} rows is not written", .{ rank, n });
    const y_after = try check.download(gpu, y_b);
    defer a.free(y_after);
    for (y_after[y.len * 4 ..]) |b| try check.expect(b == 0x5a, "rank {d}: {d} rows: the exchange writes no y row past its spare one", .{ rank, n });
    const moe_after = try check.download(gpu, moe);
    defer a.free(moe_after);
    for (moe_after[rows * dims * 4 ..]) |b| try check.expect(b == 0x5a, "rank {d}: {d} rows: the exchange writes no moe_f32 row past its spare one", .{ rank, n });

    // a control on the compared path: one changed sum must change the written rows
    const sums = try a.alloc(f32, n * dims);
    defer a.free(sums);
    try moe.download(0, std.mem.sliceAsBytes(sums));
    var at: usize = 0;
    for (sums, 0..) |v, i| {
        if (std.math.isFinite(v) and (!std.math.isFinite(sums[at]) or @abs(v) > @abs(sums[at]))) at = i;
    }
    const flipped: u32 = @as(u32, @bitCast(sums[at])) ^ (1 << 22);
    try moe.upload(at * 4, std.mem.asBytes(&flipped));
    try tri.hcWriteback(h, outs[4].hout.ptr, outs[4].pss.ptr, moe.ptr, "*fp32", inj, dummy.ptr, "*bf16", dummy.ptr, "*bf16", rs, n, dims, streams, 4, 1, 1, 1);
    try c.finish(stream);
    const changed = try check.download(gpu, outs[4].hout);
    defer a.free(changed);
    const ref = try check.download(gpu, outs[0].hout);
    defer a.free(ref);
    try check.expect(!std.mem.eql(u8, changed, ref), "rank {d}: a flipped sum bit at {d} changes the written rows", .{ rank, at });
    check.pass("rank {d}: {d} rows: exchange bytes equal the all-reduce's; nothing written past the spare row; the changed-sum control fails as planted", .{ rank, n });
}
