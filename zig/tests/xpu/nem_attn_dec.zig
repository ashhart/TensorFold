//! Matrix-engine Nemotron decode attention against the per-row kernel and an FP64 host reference, on random data.

const std = @import("std");
const rt = @import("rig.zig");
const stop = @import("xpu").stop;

const gpa = std.heap.page_allocator;
const Buf = rt.Buffer;
const NH: u32 = 32;
const NKV: u32 = 2;
const HD: u32 = 128;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn bf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}

fn randBf(buf: []u16, rnd: std.Random, sd: f32) void {
    for (buf) |*v| v.* = @intCast(@as(u32, @bitCast(rnd.floatNorm(f32) * sd)) >> 16);
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
    dec: rt.Kernel,
    part: rt.Kernel,
    merge: rt.Kernel,
    merge_r: rt.Kernel,
    dmerge: rt.Kernel,
    po: Buf,
    pm: Buf,
    pl: Buf,
    scale: f32 = 1.0 / @sqrt(@as(f32, HD)),

    fn old(c: *Ctx, q: Buf, kc: Buf, vc: Buf, out: Buf, len: u32) !void {
        const nch = (len + 511) / 512;
        try run(&c.part, .{ NKV, nch, 1 }, .{ q, kc, vc, c.po, c.pm, c.pl, len, @as(u32, 512), NKV, @as(u32, 0), c.scale });
        try run(&c.merge, .{ NH, 1, 1 }, .{ c.po, c.pm, c.pl, out, len, @as(u32, 512), NH, @as(u32, 0) });
    }

    fn new(c: *Ctx, q: Buf, kc: Buf, vc: Buf, out: Buf, len: u32) !void {
        const nch = (len + 511) / 512;
        try run(&c.dec, .{ NKV * 2, nch, 1 }, .{ q, kc, vc, c.po, c.pm, c.pl, len, @as(u32, 0), c.scale });
        try run(&c.dmerge, .{ NH, 1, 4 }, .{ c.po, c.pm, c.pl, out, len, @as(u32, 0) });
    }
};

fn refHead(q: []const u16, kh: []const u16, vh: []const u16, head: usize, len: usize, scale: f64, out: []f64) !void {
    const hk = head / 16;
    const s = try gpa.alloc(f64, len);
    defer gpa.free(s);
    var mx: f64 = -std.math.inf(f64);
    for (0..len) |k| {
        var a: f64 = 0;
        for (0..HD) |d| a += bf(q[head * HD + d]) * bf(kh[(k * NKV + hk) * HD + d]);
        s[k] = a * scale;
        mx = @max(mx, s[k]);
    }
    var den: f64 = 0;
    @memset(out, 0);
    for (0..len) |k| {
        const p = @exp(s[k] - mx);
        den += p;
        for (0..HD) |d| out[d] += p * bf(vh[(k * NKV + hk) * HD + d]);
    }
    for (out) |*o| o.* /= den;
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    stop.install();
    var r = try rt.open();
    defer r.deinit();
    var am = try r.module(@import("xpu").kernels.attn);
    var dm = try r.moduleWith(@import("xpu").kernels.nem_attn_dec, "-cl-intel-256-GRF-per-thread");
    var rm = try r.module(@import("xpu").kernels.nem_rows);
    const max_len: u32 = 131072 + 16;
    const maxc: u32 = max_len / 512 + 1;
    var c: Ctx = .{
        .dec = try dm.kernel("nem_attn_dec_partial", .{ 128, 1, 1 }),
        .part = try am.kernel("attn_partial", .{ 256, 1, 1 }),
        .merge = try am.kernel("attn_merge", .{ 16, 1, 1 }),
        .merge_r = try rm.kernel("attn_merge_r", .{ 16, 1, 1 }),
        .dmerge = try dm.kernel("nem_attn_dec_merge", .{ 256, 1, 1 }),
        .po = try r.alloc(@as(usize, 16) * maxc * NH * HD * 4),
        .pm = try r.alloc(@as(usize, 16) * maxc * NH * 4),
        .pl = try r.alloc(@as(usize, 16) * maxc * NH * 4),
    };
    var prng = std.Random.DefaultPrng.init(9);
    const rnd = prng.random();
    const kh = try gpa.alloc(u16, @as(usize, max_len) * NKV * HD);
    const vh = try gpa.alloc(u16, kh.len);
    randBf(kh, rnd, 1.0);
    randBf(vh, rnd, 1.0);
    const kc = try r.alloc(kh.len * 2);
    const vc = try r.alloc(kh.len * 2);
    try r.upload(kc, std.mem.sliceAsBytes(kh));
    try r.upload(vc, std.mem.sliceAsBytes(vh));
    const qh = try gpa.alloc(u16, 16 * NH * HD);
    const q = try r.alloc(qh.len * 2);
    const oa = try r.alloc(qh.len * 2);
    const ob = try r.alloc(qh.len * 2);
    const ha = try gpa.alloc(u16, NH * HD);
    const hb = try gpa.alloc(u16, NH * HD);
    var bad = false;
    std.debug.print("decode attention, one row: error of the per-row kernel (old) and of the matrix-engine kernel (new) against FP64 over 4 heads (max|diff| / max|o|, rms diff / rms o)\n", .{});
    for ([_]f32{ 1.0, 4.0 }) |qsd| {
        randBf(qh, rnd, qsd);
        try r.upload(q, std.mem.sliceAsBytes(qh));
        for ([_]u32{ 1, 100, 513, 4000, 30000, 131000 }) |len| {
            try c.old(q, kc, vc, oa, len);
            try c.new(q, kc, vc, ob, len);
            try r.sync();
            try r.download(std.mem.sliceAsBytes(ha), oa);
            try r.download(std.mem.sliceAsBytes(hb), ob);
            try r.sync();
            var e_old: f64 = 0;
            var e_new: f64 = 0;
            var e_nvo: f64 = 0;
            var s_old: f64 = 0;
            var s_new: f64 = 0;
            var s_ref: f64 = 0;
            var mo: f64 = 0;
            const ref = try gpa.alloc(f64, HD);
            defer gpa.free(ref);
            for ([_]usize{ 0, 9, 17, 31 }) |head| {
                try refHead(qh, kh, vh, head, len, 1.0 / @sqrt(@as(f64, HD)), ref);
                for (0..HD) |d| {
                    const a = bf(ha[head * HD + d]);
                    const b = bf(hb[head * HD + d]);
                    e_old = @max(e_old, @abs(a - ref[d]));
                    e_new = @max(e_new, @abs(b - ref[d]));
                    e_nvo = @max(e_nvo, @abs(a - b));
                    s_old += (a - ref[d]) * (a - ref[d]);
                    s_new += (b - ref[d]) * (b - ref[d]);
                    s_ref += ref[d] * ref[d];
                    mo = @max(mo, @abs(ref[d]));
                }
            }
            if (e_new / mo > 0.02 or @sqrt(s_new / s_ref) > 1.5 * @sqrt(s_old / s_ref) + 1e-3) bad = true;
            std.debug.print("  q sd {d:.0} len {d:>6}: old {e:.2} ({e:.2})  new {e:.2} ({e:.2})  new vs old {e:.2}\n", .{ qsd, len, e_old / mo, @sqrt(s_old / s_ref), e_new / mo, @sqrt(s_new / s_ref), e_nvo / mo });
        }
    }
    // a window of rows against the same tokens decoded alone: bitwise
    {
        randBf(qh, rnd, 2.0);
        try r.upload(q, std.mem.sliceAsBytes(qh));
        const pos0: u32 = 20000;
        const n: u32 = 16;
        const nch = (pos0 + n + 511) / 512;
        try run(&c.dec, .{ NKV * 2, nch, n }, .{ q, kc, vc, c.po, c.pm, c.pl, pos0 + 1, nch, c.scale });
        try run(&c.dmerge, .{ NH, n, 4 }, .{ c.po, c.pm, c.pl, oa, pos0 + 1, nch });
        try r.sync();
        const wh = try gpa.alloc(u16, n * NH * HD);
        try r.download(std.mem.sliceAsBytes(wh), oa);
        try r.sync();
        var same = true;
        for (0..n) |z| {
            const qz: Buf = .{ .rt = q.rt, .ptr = @ptrFromInt(@intFromPtr(q.ptr.?) + z * NH * HD * 2), .len = q.len - z * NH * HD * 2 };
            try c.new(qz, kc, vc, ob, pos0 + 1 + @as(u32, @intCast(z)));
            try r.sync();
            try r.download(std.mem.sliceAsBytes(hb), ob);
            try r.sync();
            if (!std.mem.eql(u16, hb, wh[z * NH * HD ..][0 .. NH * HD])) same = false;
        }
        if (!same) bad = true;
        std.debug.print("window of {d} rows at {d} against the same tokens decoded alone: {s}\n", .{ n, pos0, if (same) "bitwise identical" else "DIFFERENT" });
    }
    std.debug.print("time of one layer, one row (best of 5), old against new\n", .{});
    randBf(qh, rnd, 1.0);
    try r.upload(q, std.mem.sliceAsBytes(qh));
    for ([_]u32{ 1024, 8192, 32768, 65536, 131000 }) |len| {
        var bo: u64 = std.math.maxInt(u64);
        var bn: u64 = std.math.maxInt(u64);
        for (0..5) |_| {
            var t0 = nowNs();
            try c.old(q, kc, vc, oa, len);
            try r.sync();
            bo = @min(bo, nowNs() - t0);
            t0 = nowNs();
            try c.new(q, kc, vc, ob, len);
            try r.sync();
            bn = @min(bn, nowNs() - t0);
        }
        var bp: u64 = std.math.maxInt(u64);
        for (0..5) |_| { // the new partial kernel alone (the merge of the chunks is the rest)
            const nch = (len + 511) / 512;
            const t0 = nowNs();
            try run(&c.dec, .{ NKV * 2, nch, 1 }, .{ q, kc, vc, c.po, c.pm, c.pl, len, @as(u32, 0), c.scale });
            try r.sync();
            bp = @min(bp, nowNs() - t0);
        }
        std.debug.print("      new partial alone {d:.1} us\n", .{@as(f64, @floatFromInt(bp)) / 1e3});
        const bytes = @as(f64, @floatFromInt(len)) * NKV * HD * 2 * 2;
        std.debug.print("  len {d:>6}: old {d:>7.1} us ({d:>5.0} GB/s)  new {d:>7.1} us ({d:>5.0} GB/s)\n", .{ len, @as(f64, @floatFromInt(bo)) / 1e3, bytes / @as(f64, @floatFromInt(bo)), @as(f64, @floatFromInt(bn)) / 1e3, bytes / @as(f64, @floatFromInt(bn)) });
    }
    std.debug.print("{s}\n", .{if (bad) "nem_attn_dec: FAILED" else "nem_attn_dec: ok"});
}
