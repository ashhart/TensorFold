//! Matrix-engine prompt attention (nem_attn_pfs.cl) against the per-row kernels: difference, chunk invariance, time.

const std = @import("std");
const rt = @import("rig.zig");
const stop = @import("xpu").stop;

const spv_pfs = @import("xpu").kernels.nem_attn_pfs;
const spv_rows = @import("xpu").kernels.nem_rows;
const gpa = std.heap.page_allocator;
const Buf = rt.Buffer;
const NH: u32 = 32;
const NKV: u32 = 2;
const HD: u32 = 128;
const NT: u32 = 4; // N tiles of the build (nem_attn_pfs.cl default)
const NHG: u32 = 16 / (2 * NT);

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn bf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

fn randBf(buf: []u16, rnd: std.Random) void {
    for (buf) |*v| v.* = @intCast(@as(u32, @bitCast(rnd.floatNorm(f32))) >> 16);
}

fn at(b: Buf, off: usize) Buf {
    return .{ .rt = b.rt, .ptr = @ptrFromInt(@intFromPtr(b.ptr.?) + off), .len = b.len - off };
}

fn run(k: *rt.Kernel, groups: [3]u32, a: anytype) !void {
    inline for (a, 0..) |v, i| {
        const T = @TypeOf(v);
        if (T == Buf) {
            try k.setBuffer(i, v);
        } else if (T == f32) {
            try k.setF32(i, v);
        } else try k.setU32(i, @as(u32, v));
    }
    try k.launch(groups);
}

const Ctx = struct {
    prep: rt.Kernel,
    pfs: rt.Kernel,
    part: rt.Kernel,
    merge: rt.Kernel,
    qb: Buf,
    po: Buf,
    pm: Buf,
    pl: Buf,
    maxc: u32,

    /// The earlier path (Win.attention): blocks of rb rows against the whole prefix, split-K chunks of 512 keys, merge.
    fn old(c: *Ctx, q: Buf, kc: Buf, vc: Buf, out: Buf, pos0: u32, n: u32) !void {
        const chunk: u32 = 512;
        const q_dim = NH * HD;
        const scale: f32 = 1.0 / @sqrt(@as(f32, HD));
        const nch_all = @max(1, (pos0 + n + chunk - 1) / chunk);
        const blk: u32 = @max(16, @min(n, (16 * c.maxc) / nch_all));
        var off: u32 = 0;
        while (off < n) : (off += blk) {
            const rb = @min(blk, n - off);
            const nch = (pos0 + off + rb + chunk - 1) / chunk;
            const qo = @as(usize, off) * q_dim * 2;
            try run(&c.part, .{ NKV, nch, rb }, .{ at(q, qo), kc, vc, c.po, c.pm, c.pl, pos0 + off + 1, chunk, NKV, scale, nch_all });
            try run(&c.merge, .{ NH, rb, 1 }, .{ c.po, c.pm, c.pl, at(out, qo), pos0 + off + 1, chunk, NH, nch_all });
        }
    }

    /// The matrix-engine kernel: query tiles first, then the attention.
    fn new(c: *Ctx, q: Buf, kc: Buf, vc: Buf, out: Buf, pos0: u32, n: u32) !void {
        const rt8 = (n + 7) / 8;
        try run(&c.prep, .{ rt8 * NKV * NHG * NT * 1024 / 64, 1, 1 }, .{ q, c.qb, n });
        try run(&c.pfs, .{ rt8, NKV * NHG, 1 }, .{ c.qb, kc, vc, out, pos0, n });
    }
};

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    stop.install();
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var time_old = true;
    for (args[1..]) |a| if (std.mem.eql(u8, a, "--no-old-time")) {
        time_old = false;
    };
    var r = try rt.open();
    defer r.deinit();
    var pm = try r.moduleWith(spv_pfs, "-cl-intel-256-GRF-per-thread");
    var rm = try r.module(spv_rows);
    const maxc: u32 = 256 + 16; // 135K keys
    var c: Ctx = .{
        .prep = try pm.kernel("nem_attn_pfs_prep", .{ 64, 1, 1 }),
        .pfs = try pm.kernel("nem_attn_prefill_s", .{ 128, 1, 1 }),
        .part = try rm.kernel("attn_partial_r", .{ 256, 1, 1 }),
        .merge = try rm.kernel("attn_merge_r", .{ 16, 1, 1 }),
        .qb = try r.alloc(512 / 8 * NKV * NHG * NT * 1024 * 4),
        .po = try r.alloc(16 * maxc * NH * HD * 4),
        .pm = try r.alloc(16 * maxc * NH * 4),
        .pl = try r.alloc(16 * maxc * NH * 4),
        .maxc = maxc,
    };
    const max_n: u32 = 512;
    const max_len: u32 = 131072 + max_n;
    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();
    const qh = try gpa.alloc(u16, @as(usize, max_n) * NH * HD);
    const kh = try gpa.alloc(u16, @as(usize, max_len) * NKV * HD);
    const vh = try gpa.alloc(u16, @as(usize, max_len) * NKV * HD);
    randBf(qh, rnd);
    randBf(kh, rnd);
    randBf(vh, rnd);
    const q = try r.alloc(qh.len * 2);
    const kc = try r.alloc(kh.len * 2);
    const vc = try r.alloc(vh.len * 2);
    try r.upload(q, std.mem.sliceAsBytes(qh));
    try r.upload(kc, std.mem.sliceAsBytes(kh));
    try r.upload(vc, std.mem.sliceAsBytes(vh));
    const oa = try r.alloc(qh.len * 2);
    const ob = try r.alloc(qh.len * 2);
    try r.sync();
    const ha = try gpa.alloc(u16, qh.len);
    const hb = try gpa.alloc(u16, qh.len);
    std.debug.print("accuracy and chunk invariance (n rows at offset pos0; bf16 outputs; diff = matrix-engine kernel against the per-row kernels)\n", .{});
    const cases = [_][2]u32{ .{ 0, 256 }, .{ 100, 200 }, .{ 3000, 256 }, .{ 30000, 512 }, .{ 100000, 128 } };
    var bad = false;
    for (cases) |cs| {
        const pos0 = cs[0];
        const n = cs[1];
        try c.old(q, kc, vc, oa, pos0, n);
        try c.new(q, kc, vc, ob, pos0, n);
        try r.sync();
        const cnt = @as(usize, n) * NH * HD;
        try r.download(std.mem.sliceAsBytes(ha[0..cnt]), oa);
        try r.download(std.mem.sliceAsBytes(hb[0..cnt]), ob);
        try r.sync();
        var maxd: f64 = 0;
        var maxo: f64 = 0;
        var sd: f64 = 0;
        var so: f64 = 0;
        for (ha[0..cnt], hb[0..cnt]) |a, b| {
            maxd = @max(maxd, @abs(@as(f64, bf(a)) - bf(b)));
            maxo = @max(maxo, @abs(@as(f64, bf(a))));
            sd += (@as(f64, bf(a)) - bf(b)) * (@as(f64, bf(a)) - bf(b));
            so += @as(f64, bf(a)) * bf(a);
        }
        // chunk invariance: the same rows as 2 and 4 launches (q rows offset, pos0 advanced)
        var inv = true;
        for ([_]u32{ 2, 4 }) |parts| {
            const w = n / parts;
            if (w == 0) continue;
            const oc = try r.alloc(cnt * 2);
            var p: u32 = 0;
            while (p < parts) : (p += 1) try c.new(at(q, @as(usize, p) * w * NH * HD * 2), kc, vc, at(oc, @as(usize, p) * w * NH * HD * 2), pos0 + p * w, w);
            try r.sync();
            const hc = try gpa.alloc(u16, cnt);
            defer gpa.free(hc);
            try r.download(std.mem.sliceAsBytes(hc), oc);
            try r.sync();
            if (!std.mem.eql(u16, hc, hb[0..cnt])) inv = false;
        }
        if (maxd / maxo > 0.02 or !inv) bad = true;
        std.debug.print("  pos0 {d:>6} n {d:>3}: max|diff| / max|o| {e:.2}, rms diff / rms o {e:.2}, chunk invariant (1 vs 2 vs 4 launches) {s}\n", .{ pos0, n, maxd / maxo, @sqrt(sd / so), if (inv) "yes" else "NO" });
    }
    std.debug.print("time of one layer, 512-row window at offset pos0 (best of 3), earlier per-row path against the matrix-engine kernel (4 * 128 * 32 flop a key-row pair)\n", .{});
    for ([_]u32{ 0, 8192, 32768, 65536, 131072 - 512 }) |pos0| {
        var best_new: u64 = std.math.maxInt(u64);
        var best_old: u64 = std.math.maxInt(u64);
        for (0..3) |_| {
            const t0 = nowNs();
            try c.new(q, kc, vc, ob, pos0, max_n);
            try r.sync();
            best_new = @min(best_new, nowNs() - t0);
        }
        if (time_old and pos0 <= 65536) for (0..2) |_| {
            const t0 = nowNs();
            try c.old(q, kc, vc, oa, pos0, max_n);
            try r.sync();
            best_old = @min(best_old, nowNs() - t0);
        };
        const flop = 4.0 * 128.0 * 32.0 * (@as(f64, max_n) * @as(f64, pos0) + @as(f64, max_n) * @as(f64, max_n) / 2.0);
        std.debug.print("  pos0 {d:>6}: new {d:>8.2} ms ({d:>5.1} TFLOP/s)  earlier ", .{ pos0, @as(f64, @floatFromInt(best_new)) / 1e6, flop / @as(f64, @floatFromInt(best_new)) / 1e3 });
        if (best_old == std.math.maxInt(u64)) std.debug.print("-\n", .{}) else std.debug.print("{d:.2} ms\n", .{@as(f64, @floatFromInt(best_old)) / 1e6});
    }
    std.debug.print("{s}\n", .{if (bad) "nem_attn_pf: FAILED" else "nem_attn_pf: ok"});
}
