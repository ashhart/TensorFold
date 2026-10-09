//! Matrix-engine prompt-window attention vs the per-row kernels, plus timing. usage: xpu-qwen_attnpf-test

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const al = @import("qwen_xpu").attn_long;

const spv_pf = @import("xpu").kernels.qwen_attn_pf;
const spv_rows = @import("xpu").kernels.qwen_rows;
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

pub fn main() !void {
    var r = try rt.open();
    var pm = try r.moduleWith(spv_pf, "-cl-intel-256-GRF-per-thread");
    var pf = try pm.kernel("attn_prefill", .{ 64, 1, 1 });
    var rm = try r.module(spv_rows);
    var part = try rm.kernel("attn_partial_r", .{ 128, 1, 1 });
    var merge = try rm.kernel("attn_merge_r", .{ 256, 1, 1 });
    var long = try al.Long.init(&r, .bf16);
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();

    const max_len: usize = 2048 + 122880;
    const kv = try gpa.alloc(u16, max_len * 1024);
    randBf(kv, rnd, 1.0);
    const kc = try r.alloc(kv.len * 2);
    try r.upload(kc, std.mem.sliceAsBytes(kv));
    randBf(kv, rnd, 1.0);
    const vc = try r.alloc(kv.len * 2);
    try r.upload(vc, std.mem.sliceAsBytes(kv));
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

    // correctness: rows 60 at pos0 200 (below 4096 keys), rows 40 at pos0 8000 (long kernel)
    for ([_][2]u32{ .{ 200, 60 }, .{ 3990, 50 }, .{ 8000, 40 } }) |c| {
        const pos0 = c[0];
        const rows = c[1];
        try pf.setBuffer(0, q);
        try pf.setBuffer(1, kc);
        try pf.setBuffer(2, vc);
        try pf.setBuffer(3, qg);
        try pf.setBuffer(4, out_new);
        try pf.setU32(5, pos0);
        try pf.setU32(6, rows);
        try pf.setF32(7, 0.0625);
        try pf.launch(.{ (rows + 31) / 32, 24, 1 });
        if (pos0 + rows <= 4096) {
            try part.setBuffer(0, q);
            try part.setBuffer(1, kc);
            try part.setBuffer(2, vc);
            try part.setBuffer(3, po);
            try part.setBuffer(4, pmx);
            try part.setBuffer(5, plx);
            try part.setU32(6, pos0);
            try part.setF32(7, 0.0625);
            try part.setU32(8, 128);
            try part.launch(.{ 4, (pos0 + rows + 63) / 64, rows });
            try merge.setBuffer(0, po);
            try merge.setBuffer(1, pmx);
            try merge.setBuffer(2, plx);
            try merge.setBuffer(3, qg);
            try merge.setBuffer(4, out_old);
            try merge.setU32(5, pos0);
            try merge.setU32(6, 128);
            try merge.launch(.{ 24, rows, 1 });
        } else try long.run(q, kc, vc, po, pmx, plx, qg, out_old, pos0 + 1, rows);
        const a = try gpa.alloc(u16, rows * 24 * 256);
        const b = try gpa.alloc(u16, rows * 24 * 256);
        try r.download(std.mem.sliceAsBytes(a), out_new);
        try r.download(std.mem.sliceAsBytes(b), out_old);
        try r.sync();
        var mx: f32 = 0;
        var worst: f32 = 0;
        var differ: usize = 0;
        for (a, b) |x, y| {
            mx = @max(mx, @abs(bf(y)));
            worst = @max(worst, @abs(bf(x) - bf(y)));
            differ += @intFromBool(x != y);
        }
        std.debug.print("pos0 {d:>5} rows {d:>3}: {d} of {d} values differ from the per-row kernels, worst {e:.2} of max|o| {d:.3}\n", .{ pos0, rows, differ, a.len, worst / mx, mx });
    }

    // speed: 2048 rows (the new kernel) and 32 rows of the per-row/long kernels, per layer
    for ([_]u32{ 0, 16384, 65536, 120832 }) |pos0| {
        const rows: u32 = 2048;
        var best: u64 = std.math.maxInt(u64);
        for (0..3) |_| {
            const t0 = nowNs();
            try pf.setBuffer(0, q);
            try pf.setBuffer(1, kc);
            try pf.setBuffer(2, vc);
            try pf.setBuffer(3, qg);
            try pf.setBuffer(4, out_new);
            try pf.setU32(5, pos0);
            try pf.setU32(6, rows);
            try pf.setF32(7, 0.0625);
            try pf.launch(.{ (rows + 31) / 32, 24, 1 });
            try r.sync();
            best = @min(best, nowNs() - t0);
        }
        const ms = @as(f64, @floatFromInt(best)) / 1e6;
        // causal: sum over rows of (pos0 + z + 1) keys x 24 heads x 256 dims x 4 flop
        const keys = @as(f64, @floatFromInt(rows)) * (@as(f64, @floatFromInt(pos0)) + @as(f64, @floatFromInt(rows)) / 2);
        const tf = keys * 24 * 256 * 4 / (ms / 1e3) / 1e12;
        var old_ms: f64 = 0;
        if (pos0 + 32 > 4096) {
            var ob: u64 = std.math.maxInt(u64);
            for (0..2) |_| {
                const t0 = nowNs();
                try long.run(q, kc, vc, po, pmx, plx, qg, out_old, pos0 + 1, 32);
                try r.sync();
                ob = @min(ob, nowNs() - t0);
            }
            old_ms = @as(f64, @floatFromInt(ob)) / 1e6 * 2048.0 / 32.0;
        }
        std.debug.print("context offset {d:>6}: prefill kernel {d:>8.2} ms for 2048 rows ({d:>5.1} TFLOP/s-equivalent); per-row long kernel (32 rows scaled) {d:>9.1} ms\n", .{ pos0, ms, tf, old_ms });
    }
}
