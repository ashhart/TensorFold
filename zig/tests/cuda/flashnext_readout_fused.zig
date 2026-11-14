//! The fused BF16 read-out (flashnext_hcn.cu) against the five launches it replaces: every output byte, then times.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig_ = @import("flashnext_rig.zig");

const tiles = flash.tiles;
const tk = flash.tile_kernels;

const dims = 2560;
const streams = 4;
const wide = streams * dims;
const low = 320;
const n_down = low + streams;
const nc = dims / 256;
const eps: f32 = 1e-6;
const most = 3072;

const Chain = struct {
    r: *rig_.Rig,
    g: *const flash.Glue,
    lane: *const cuda.qmmf.Lane,
    t: *const tk.TileKernels,
    wdown: cuda.qmmf.Weight,
    wup: cuda.qmmf.Weight,
    h: u64,
    pss: u64,
    scale: u64,
    norm: u64,
    part: u64,
    act: u64,
    xs: u64,
    inj: u64,
    up: u64,
    mixed: u64,
    xsm: u64,
    spart: u64,
    rows: usize,
    sms: usize,

    /// normalize, the lane's fp32 down, reduce_act, the lane's bf16 up and the dense mix, as readoutNative runs them
    fn five(c: *Chain) !void {
        const s = c.r.s;
        try c.g.hcNormalize(s, c.h, c.pss, c.scale, c.norm, eps, c.rows, dims, streams);
        try c.lane.matmulF32(s, c.norm, wide, c.rows, c.wdown, c.part, c.spart);
        try c.g.hcReduceAct(s, c.part, c.act, c.xs, c.inj, 1, c.rows, streams, low, n_down, 1);
        try c.lane.matmul(s, c.act, low, c.rows, c.wup, c.up, c.spart);
        try c.g.hcDenseMix(s, c.up, c.norm, c.mixed, c.xsm, c.rows, dims, streams);
    }

    /// the fused down, reduce_act and the fused up+mix, as readoutFused runs them
    fn fused(c: *Chain) !void {
        const s = c.r.s;
        try tiles.fusedDown(c.t, c.sms, s, c.h, c.pss, c.scale, c.wdown, c.part, c.rows, eps);
        try c.g.hcReduceAct(s, c.part, c.act, c.xs, c.inj, 1, c.rows, streams, low, n_down, 1);
        try tiles.fusedUpMix(c.t, s, c.act, c.wup, c.h, c.pss, c.scale, c.mixed, c.xsm, c.rows, eps);
    }

    fn poison(c: *Chain) !void {
        const m = c.rows;
        const bufs = [_][2]u64{ .{ c.part, m * n_down * 4 }, .{ c.act, m * low * 2 }, .{ c.xs, m * (low / 32) * 4 }, .{ c.inj, m * streams * 2 }, .{ c.mixed, m * dims * 2 }, .{ c.xsm, m * (dims / 32) * 4 } };
        for (bufs) |b| try (cuda.DeviceBuffer{ .d = c.r.gpu.d, .ptr = b[0], .len = b[1] }).fill8(0xff, c.r.s.handle);
    }

    /// The read-out's outputs back to back: the down's fp32 rows, act, its sums, the gates, mixed and its sums.
    fn outputs(c: *Chain, a: std.mem.Allocator) ![]u8 {
        const m = c.rows;
        var all: std.ArrayList(u8) = .empty;
        try all.appendSlice(a, std.mem.sliceAsBytes(try c.r.back(f32, c.part, m * n_down)));
        try all.appendSlice(a, std.mem.sliceAsBytes(try c.r.back(u16, c.act, m * low)));
        try all.appendSlice(a, std.mem.sliceAsBytes(try c.r.back(f32, c.xs, m * (low / 32))));
        try all.appendSlice(a, std.mem.sliceAsBytes(try c.r.back(u16, c.inj, m * streams)));
        try all.appendSlice(a, std.mem.sliceAsBytes(try c.r.back(u16, c.mixed, m * dims)));
        try all.appendSlice(a, std.mem.sliceAsBytes(try c.r.back(f32, c.xsm, m * (dims / 32))));
        return all.items;
    }
};

fn median(c: *Chain, comptime f: fn (*Chain) anyerror!void, start: cuda.Event, end: cuda.Event) !f64 {
    var t: [9]f64 = undefined;
    try f(c);
    for (&t) |*x| {
        try start.record(c.r.s);
        for (0..10) |_| try f(c);
        try end.record(c.r.s);
        try c.r.s.synchronize();
        x.* = @as(f64, try start.elapsedMs(end)) * 100;
    }
    std.mem.sort(f64, &t, {}, std.sort.asc(f64));
    return t[t.len / 2];
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x4c2);
    var r = try rig_.Rig.init(gpu, a, prng.random());
    defer r.deinit();
    var g = try flash.Glue.load(gpu.d);
    defer g.deinit();
    var lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    var t = try tk.TileKernels.load(gpu.d);
    defer t.unload();

    // bf16 weights in the lane's order, rows padded to 128 with zeros
    const pd = try a.alloc(u16, cuda.nvfp4.padded(n_down) * wide);
    cuda.qmmf.packBf16(pd, r.bfs(n_down * wide, 0.04), n_down, wide);
    const pu = try a.alloc(u16, cuda.nvfp4.padded(wide) * low);
    cuda.qmmf.packBf16(pu, r.bfs(wide * low, 0.4), wide, low);
    const dh = try r.dev(std.mem.sliceAsBytes(r.bfs(most * wide, 3)));
    const dpss = try r.out(most * nc * streams * 4);
    var c: Chain = .{
        .r = &r,
        .g = &g,
        .lane = &lane,
        .t = &t,
        .wdown = cuda.modelopt.bf16Weight(try r.dev(std.mem.sliceAsBytes(pd)), n_down, wide),
        .wup = cuda.modelopt.bf16Weight(try r.dev(std.mem.sliceAsBytes(pu)), wide, low),
        .h = dh,
        .pss = dpss,
        .scale = try r.dev(std.mem.sliceAsBytes(r.f32s(wide, 0.5, 1.5))),
        .norm = try r.out(most * wide * 2),
        .part = try r.out(most * n_down * 4),
        .act = try r.out(most * low * 2),
        .xs = try r.out(most * (low / 32) * 4),
        .inj = try r.out(most * streams * 2),
        .up = try r.out(most * wide * 2),
        .mixed = try r.out(most * dims * 2),
        .xsm = try r.out(most * (dims / 32) * 4),
        .spart = try r.out(8 * most * n_down * 4),
        .rows = most,
        .sms = lane.prompt_sms,
    };
    // the streams' 256-column square sums, as the writeback leaves them for the read-out
    try g.hcWriteback(r.s, dh, 0, dpss, 0, 0, 0, 0, most * dims, most, dims, streams, 0, 1, 1, 1);

    // every down tile at every row count: SM counts of 1, the device's and 4,096 pick the widest, the waves' and the least
    const counts = [_]usize{ 64, 65, 193, 255, 256, 767, 768, 769, 1000, 1024, 1536, 2053, 3072 };
    for (counts) |m| {
        c.rows = m;
        try c.poison();
        try c.five();
        const want = try c.outputs(a);
        for ([_]usize{ 1, lane.prompt_sms, 4096 }) |sms| {
            c.sms = sms;
            try c.poison();
            try c.fused();
            try check.sameBytes("fused read-out", try c.outputs(a), want);
        }
        check.pass("readout-fused: {d} rows, down tiles of {d}, {d} and {d} rows: every output byte equals the five launches'", .{ m, 16 * tk.hcn_down_mt[tiles.pickDown(m, 1)], 16 * tk.hcn_down_mt[tiles.pickDown(m, lane.prompt_sms)], 16 * tk.hcn_down_mt[tiles.pickDown(m, 4096)] });
    }

    c.sms = lane.prompt_sms;
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    for ([_]usize{ 256, 512, 768, 1024, 1536, 2048, 3072 }) |m| {
        c.rows = m;
        const a1 = try median(&c, Chain.five, start, end);
        const b1 = try median(&c, Chain.fused, start, end);
        const a2 = try median(&c, Chain.five, start, end);
        const b2 = try median(&c, Chain.fused, start, end);
        std.debug.print("RESULT readout-fused rows {d}: five launches {d:.1} us, fused {d:.1} us (medians of 9 x 10, alternated)\n", .{ m, (a1 + a2) / 2, (b1 + b2) / 2 });
    }
    r.report("readout-fused");
}
