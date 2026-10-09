//! Prompt-window attention over the q8 / q4 KV cache vs the per-row kernel. usage: xpu-qwen_attnpfq-test q8|q4

const std = @import("std");
const rt = @import("rig.zig");
const stop = @import("xpu").stop;
const al = @import("qwen_xpu").attn_long;
const exl3 = @import("xpu").exl3;

const gpa = std.heap.page_allocator;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn bf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

fn randBf(buf: []u16, rnd: std.Random, sd: f32) void {
    for (buf) |*v| v.* = @intCast(@as(u32, @bitCast(rnd.floatNorm(f32) * sd)) >> 16);
}


fn half(v: u16) f64 {
    return @as(f16, @bitCast(v));
}

/// Dequantized values (fp64) of one record of mode m into out[256] (as the decode kernel reads them).
fn dequant(m: al.Mode, rec: []const u8, out: []f64) void {
    switch (m) {
        .bf16 => for (0..256) |i| {
            out[i] = bf(std.mem.readInt(u16, rec[2 * i ..][0..2], .little));
        },
        .q8 => for (0..256) |i| {
            const d = half(std.mem.readInt(u16, rec[256 + 2 * (i / 32) ..][0..2], .little));
            out[i] = @as(f64, @floatFromInt(@as(i8, @bitCast(rec[i])))) * d;
        },
        .q4 => for (0..256) |i| {
            const b = i / 32;
            const j = i % 32; // dword j / 8 of the block, nibble p (dim 2p) or p + 4 (dim 2p + 1)
            const w = std.mem.readInt(u32, rec[b * 16 + (j / 8) * 4 ..][0..4], .little);
            const p = (j % 8) / 2;
            const sh: u5 = @intCast(4 * p + (if (j % 2 == 1) @as(usize, 16) else 0));
            const n: f64 = @floatFromInt((w >> sh) & 0xF);
            const d = half(std.mem.readInt(u16, rec[128 + 2 * b ..][0..2], .little));
            out[i] = (n - 8.0) * d;
        },
    }
}

/// fp64 attention (pre-gate) of one query row (24 heads, bf16 q) over `len` keys of dequantized records kd / vd.
fn refRow(len: usize, q: []const u16, kd: []const f64, vd: []const f64, o: []f64) void {
    const sc = gpa.alloc(f64, len) catch unreachable;
    defer gpa.free(sc);
    for (0..24) |h| {
        const hk = h / 6;
        var mx: f64 = -1e300;
        for (0..len) |t| {
            var s: f64 = 0;
            for (0..256) |d| s += @as(f64, bf(q[h * 256 + d])) * kd[(t * 4 + hk) * 256 + d];
            sc[t] = s * 0.0625;
            mx = @max(mx, sc[t]);
        }
        var den: f64 = 0;
        for (0..len) |t| {
            sc[t] = @exp(sc[t] - mx);
            den += sc[t];
        }
        for (0..256) |d| {
            var a: f64 = 0;
            for (0..len) |t| a += sc[t] * vd[(t * 4 + hk) * 256 + d];
            o[h * 256 + d] = a / den;
        }
    }
}

const spv_pf = @import("xpu").kernels.qwen_attn_pf;

/// The earlier one-head, 16-row kernel (qwen_attn_pf.cl), for the bf16 comparison.
fn runOld(k: *rt.Kernel, q: rt.Buffer, kc: rt.Buffer, vc: rt.Buffer, qg: rt.Buffer, out: rt.Buffer, pos0: u32, rows: u32) !void {
    try k.setBuffer(0, q);
    try k.setBuffer(1, kc);
    try k.setBuffer(2, vc);
    try k.setBuffer(3, qg);
    try k.setBuffer(4, out);
    try k.setU32(5, pos0);
    try k.setU32(6, rows);
    try k.setF32(7, 0.0625);
    try k.launch(.{ (rows + 15) / 16, 24, 1 });
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    stop.install();
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const ms: []const u8 = if (args.len > 1) args[1] else "q8";
    const mode: al.Mode = if (std.mem.eql(u8, ms, "q4")) .q4 else if (std.mem.eql(u8, ms, "bf16")) .bf16 else .q8;
    var r = try rt.open();
    defer r.deinit();
    var pf = try al.Pfs.init(&r, mode, 2048);
    var oldpf: ?rt.Kernel = null;
    if (mode == .bf16) {
        var om = try r.moduleWith(spv_pf, "-cl-intel-256-GRF-per-thread");
        oldpf = try om.kernel("attn_prefill", .{ 64, 1, 1 });
    }
    var long = try al.Long.init(&r, mode);
    var kvq = try al.Kvq.init(&r);
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();

    const max_len: usize = 2048 + 122880;
    const kv = try gpa.alloc(u16, max_len * 1024);
    const stage = try r.alloc(kv.len * 2);
    const kc = try r.alloc(al.cacheBytesFor(mode, @intCast(max_len)));
    const vc = try r.alloc(al.cacheBytesFor(mode, @intCast(max_len)));
    for ([_]rt.Buffer{ kc, vc }, 0..) |dst, idx| {
        randBf(kv, rnd, 1.0);
        if (idx == 1) for (kv, 0..) |*v, i| { // V with a spread of magnitudes per 32-dim block, like real activations
            const blk: f32 = @floatFromInt(1 + (i / 32) % 5);
            v.* = @intCast(@as(u32, @bitCast(bf(v.*) * blk)) >> 16);
        };
        try r.upload(stage, std.mem.sliceAsBytes(kv));
        if (mode == .bf16) {
            try r.upload(dst, std.mem.sliceAsBytes(kv));
            continue;
        }
        var p: usize = 0;
        while (p < max_len) : (p += 8192) {
            const n: u32 = @intCast(@min(8192, max_len - p));
            try kvq.append(mode, exl3.at(stage, p * 1024 * 2), dst, @intCast(p), n);
        }
    }
    const qrows: usize = 2048;
    const qh = try gpa.alloc(u16, qrows * 24 * 512);
    randBf(qh, rnd, 0.7);
    const qg = try r.alloc(qh.len * 2);
    try r.upload(qg, std.mem.sliceAsBytes(qh));
    const qd = try gpa.alloc(u16, qrows * 24 * 256);
    for (0..qrows * 24) |i| @memcpy(qd[i * 256 ..][0..256], qh[i * 512 ..][0..256]);
    const q = try r.alloc(qd.len * 2);
    try r.upload(q, std.mem.sliceAsBytes(qd));
    const out_new = try r.alloc(qd.len * 2);
    const out_old = try r.alloc(qd.len * 2);
    const po = try r.alloc(@as(usize, 64) * 128 * 24 * 256 * 4);
    const pmx = try r.alloc(@as(usize, 64) * 128 * 24 * 4);
    const plx = try r.alloc(@as(usize, 64) * 128 * 24 * 4);
    try r.sync();

    for ([_][2]u32{ .{ 0, 60 }, .{ 200, 60 }, .{ 3990, 50 }, .{ 8000, 40 }, .{ 30000, 37 }, .{ 100000, 33 } }) |c| {
        const pos0 = c[0];
        const rows = c[1];
        try pf.run(q, kc, vc, qg, out_new, pos0, rows);
        try long.run(q, kc, vc, po, pmx, plx, qg, out_old, pos0 + 1, rows);
        const a = try gpa.alloc(u16, rows * 24 * 256);
        const b = try gpa.alloc(u16, rows * 24 * 256);
        try r.download(std.mem.sliceAsBytes(a), out_new);
        try r.download(std.mem.sliceAsBytes(b), out_old);
        try r.sync();
        var mx: f32 = 0;
        var worst: f32 = 0;
        var sum2: f64 = 0;
        var den2: f64 = 0;
        var differ: usize = 0;
        for (a, b) |x, y| {
            mx = @max(mx, @abs(bf(y)));
            worst = @max(worst, @abs(bf(x) - bf(y)));
            sum2 += @as(f64, bf(x) - bf(y)) * (bf(x) - bf(y));
            den2 += @as(f64, bf(y)) * bf(y);
            differ += @intFromBool(x != y);
        }
        std.debug.print("{s} pos0 {d:>6} rows {d:>3}: {d} of {d} values differ from the per-row kernel, worst {e:.2} of max|o| {d:.3}, rms rel {e:.2}\n", .{ ms, pos0, rows, differ, a.len, worst / mx, mx, @sqrt(sum2 / den2) });
        gpa.free(a);
        gpa.free(b);
    }


    // accuracy vs fp64 over the dequantized records (pre-gate o = out / sigmoid(gate)), 30000 + 37 case; output is bf16
    {
        const pos0: u32 = 30000;
        const rows: u32 = 37;
        const len = pos0 + rows;
        const rb = al.recBytes(mode);
        const kh = try gpa.alloc(u8, @as(usize, len) * 4 * rb);
        const vh = try gpa.alloc(u8, @as(usize, len) * 4 * rb);
        try r.download(kh, kc);
        try r.download(vh, vc);
        const kd = try gpa.alloc(f64, @as(usize, len) * 1024);
        const vd = try gpa.alloc(f64, @as(usize, len) * 1024);
        for (0..@as(usize, len) * 4) |i| {
            dequant(mode, kh[i * rb ..][0..rb], kd[i * 256 ..][0..256]);
            dequant(mode, vh[i * rb ..][0..rb], vd[i * 256 ..][0..256]);
        }
        try pf.run(q, kc, vc, qg, out_new, pos0, rows);
        try long.run(q, kc, vc, po, pmx, plx, qg, out_old, pos0 + 1, rows);
        const a = try gpa.alloc(u16, rows * 24 * 256);
        const b = try gpa.alloc(u16, rows * 24 * 256);
        try r.download(std.mem.sliceAsBytes(a), out_new);
        try r.download(std.mem.sliceAsBytes(b), out_old);
        try r.sync();
        var e_new: f64 = 0;
        var e_old: f64 = 0;
        var den: f64 = 0;
        const o = try gpa.alloc(f64, 24 * 256);
        for ([_]u32{ 0, 17, 36 }) |z| {
            refRow(pos0 + z + 1, qd[@as(usize, z) * 24 * 256 ..][0 .. 24 * 256], kd, vd, o);
            for (0..24 * 256) |i| {
                const g = 1.0 / (1.0 + @exp(-@as(f64, bf(qh[(@as(usize, z) * 24 * 512) + (i / 256) * 512 + 256 + i % 256]))));
                const rv = o[i] * g;
                e_new += (bf(a[@as(usize, z) * 24 * 256 + i]) - rv) * (bf(a[@as(usize, z) * 24 * 256 + i]) - rv);
                e_old += (bf(b[@as(usize, z) * 24 * 256 + i]) - rv) * (bf(b[@as(usize, z) * 24 * 256 + i]) - rv);
                den += rv * rv;
            }
        }
        std.debug.print("{s} vs fp64 over the dequantized records (3 rows x 24 heads at 30000+): rms rel error new {e:.3} per-row kernel {e:.3}\n", .{ ms, @sqrt(e_new / den), @sqrt(e_old / den) });
        gpa.free(kh);
        gpa.free(vh);
        gpa.free(kd);
        gpa.free(vd);
    }

    // key-range invariance: same rows with scratch in ranges of 4096 / 128 keys vs the default (bit-identical expected)
    if (mode != .bf16) {
        for ([_]u32{ 4096, 128 }) |rg| {
            try pf.run(q, kc, vc, qg, out_new, 30000, 37);
            var pf2 = try al.Pfs.init(&r, mode, 2048);
            pf2.range = rg;
            try pf2.run(q, kc, vc, qg, out_old, 30000, 37);
            const a = try gpa.alloc(u16, 37 * 24 * 256);
            const b = try gpa.alloc(u16, 37 * 24 * 256);
            try r.download(std.mem.sliceAsBytes(a), out_new);
            try r.download(std.mem.sliceAsBytes(b), out_old);
            try r.sync();
            std.debug.print("key-range invariance (30037 keys, ranges of {d} against {d}): {s}\n", .{ rg, pf.range, if (std.mem.eql(u16, a, b)) "bit-identical" else "DIFFERENT" });
        }
    }

    // chunk invariance: 100 rows at 5000 in one launch vs 3 launches of other widths (first not a multiple of 16)
    {
        const pos0: u32 = 5000;
        try pf.run(q, kc, vc, qg, out_new, pos0, 100);
        const splits = [_]u32{ 7, 48, 45 };
        var off: u32 = 0;
        for (splits) |n| {
            const qo = exl3.at(q, @as(usize, off) * 24 * 256 * 2);
            const go = exl3.at(qg, @as(usize, off) * 24 * 512 * 2);
            const oo = exl3.at(out_old, @as(usize, off) * 24 * 256 * 2);
            try pf.run(qo, kc, vc, go, oo, pos0 + off, n);
            off += n;
        }
        const a = try gpa.alloc(u16, 100 * 24 * 256);
        const b = try gpa.alloc(u16, 100 * 24 * 256);
        try r.download(std.mem.sliceAsBytes(a), out_new);
        try r.download(std.mem.sliceAsBytes(b), out_old);
        try r.sync();
        std.debug.print("chunk invariance (100 rows | 7 + 48 + 45): {s}\n", .{if (std.mem.eql(u16, a[0 .. 100 * 24 * 256], b[0 .. 100 * 24 * 256])) "bit-identical" else "DIFFERENT"});
    }

    for ([_]u32{ 0, 16384, 65536, 120832 }) |pos0| {
        if (stop.requested()) {
            try r.sync();
            return error.Interrupted;
        }
        const rows: u32 = 2048;
        var best: u64 = std.math.maxInt(u64);
        for (0..3) |_| {
            const t0 = nowNs();
            try pf.run(q, kc, vc, qg, out_new, pos0, rows);
            try r.sync();
            best = @min(best, nowNs() - t0);
        }
        const msec = @as(f64, @floatFromInt(best)) / 1e6;
        const keys = @as(f64, @floatFromInt(rows)) * (@as(f64, @floatFromInt(pos0)) + @as(f64, @floatFromInt(rows)) / 2);
        const tf = keys * 24 * 256 * 4 / (msec / 1e3) / 1e12;
        var old_ms: f64 = 0;
        if (oldpf) |*ok| { // bf16: the earlier one-head kernel (qwen_attn_pf.cl), 2048 rows
            var ob: u64 = std.math.maxInt(u64);
            for (0..3) |_| {
                const t0 = nowNs();
                try runOld(ok, q, kc, vc, qg, out_old, pos0, rows);
                try r.sync();
                ob = @min(ob, nowNs() - t0);
            }
            old_ms = @as(f64, @floatFromInt(ob)) / 1e6;
        } else if (std.c.getenv("NOLONG") == null) {
            var ob: u64 = std.math.maxInt(u64);
            for (0..2) |_| {
                const t0 = nowNs();
                try long.run(q, kc, vc, po, pmx, plx, qg, out_old, pos0 + 1, 32);
                try r.sync();
                ob = @min(ob, nowNs() - t0);
            }
            old_ms = @as(f64, @floatFromInt(ob)) / 1e6 * 2048.0 / 32.0;
        }
        std.debug.print("{s} context offset {d:>6}: prefill kernel {d:>8.2} ms for 2048 rows ({d:>5.1} TFLOP/s-equivalent); {s} {d:>9.1} ms\n", .{ ms, pos0, msec, tf, if (oldpf != null) "earlier one-head kernel (2048 rows)" else "per-row long kernel (32 rows scaled)", old_ms });
    }
}
