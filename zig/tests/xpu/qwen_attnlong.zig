//! Long-context decode attention vs split-K kernels: agreement, invariance, timing. usage: xpu-qwen_attnlong-test [N]

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");

const spv_old = @import("xpu").kernels.qwen_attn;
const spv_new = @import("xpu").kernels.qwen_attn_long;
const kv_dim: usize = 4 * 256;
const LC: u32 = 1024;
const MAXC: u32 = 128;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn bf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const max_len: usize = if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else 131072;
    var r = try rt.open();
    defer r.deinit();
    var mo = try r.module(spv_old);
    var mn = try r.moduleWith(spv_new, if (std.c.getenv("ATTN_NOGRF") == null) "-cl-intel-256-GRF-per-thread" else null);
    var part = try mo.kernel("attn_partial", .{ 128, 1, 1 });
    var merge = try mo.kernel("attn_merge", .{ 256, 1, 1 });
    var lpart = try mn.kernel("attn_long_partial", .{ 128, 1, 1 });
    var lmerge = try mn.kernel("attn_long_merge", .{ 256, 1, 1 });
    const gpa = std.heap.page_allocator;
    const host = try gpa.alloc(u16, max_len * kv_dim);
    var prng = std.Random.DefaultPrng.init(7);
    for (host) |*v| {
        const f: f32 = (prng.random().float(f32) - 0.5) * 4.0; // K/V values
        v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
    }
    const kc = try r.alloc(max_len * kv_dim * 2);
    const vc = try r.alloc(max_len * kv_dim * 2);
    try r.upload(kc, std.mem.sliceAsBytes(host));
    for (host) |*v| v.* = v.* ^ 0x0100;
    try r.upload(vc, std.mem.sliceAsBytes(host));
    try r.sync();
    const rows: u32 = 4;
    const qh = try gpa.alloc(u16, rows * 24 * 512);
    for (qh) |*v| v.* = @truncate(@as(u32, @bitCast((prng.random().float(f32) - 0.5) * 2.0)) >> 16);
    const qg = try r.alloc(rows * 24 * 512 * 2);
    try r.upload(qg, std.mem.sliceAsBytes(qh));
    // q of row z: the first 256 values of each head's 512 (query, gate)
    const qs = try gpa.alloc(u16, rows * 24 * 256);
    for (0..rows * 24) |i| @memcpy(qs[i * 256 ..][0..256], qh[i * 512 ..][0..256]);
    const q = try r.alloc(rows * 24 * 256 * 2);
    try r.upload(q, std.mem.sliceAsBytes(qs));
    const po = try r.alloc(2048 * 24 * 256 * 4);
    const pm = try r.alloc(2048 * 24 * 4);
    const pl = try r.alloc(2048 * 24 * 4);
    const lpo = try r.alloc(rows * MAXC * 24 * 256 * 4);
    const lpm = try r.alloc(rows * MAXC * 24 * 4);
    const lpl = try r.alloc(rows * MAXC * 24 * 4);
    const att = try r.alloc(24 * 256 * 2);
    const latt = try r.alloc(rows * 24 * 256 * 2);
    const att1 = try r.alloc(24 * 256 * 2);
    try r.sync();
    const oh = try gpa.alloc(u16, 24 * 256);
    const nh = try gpa.alloc(u16, rows * 24 * 256);
    const n1 = try gpa.alloc(u16, 24 * 256);
    var fail = false;
    var lens = std.ArrayList(usize).empty;
    for ([_]usize{ 1, 17, 1000, 1024, 1025, 4096, 8191, 16384, 32768, 65536, 131072 }) |l| if (l + rows <= max_len) try lens.append(gpa, l);
    for (lens.items) |len| {
        const nch_old: u32 = @intCast((len + 63) / 64);
        // old kernel, one row (len keys)
        const run_old = struct {
            fn go(p: *rt.Kernel, mg: *rt.Kernel, qb: rt.Buffer, kb: rt.Buffer, vb: rt.Buffer, o: rt.Buffer, m: rt.Buffer, l: rt.Buffer, qgb: rt.Buffer, out: rt.Buffer, ln: u32, nch: u32) !void {
                try p.setBuffer(0, qb);
                try p.setBuffer(1, kb);
                try p.setBuffer(2, vb);
                try p.setBuffer(3, o);
                try p.setBuffer(4, m);
                try p.setBuffer(5, l);
                try p.setU32(6, ln);
                try p.setF32(7, 0.0625);
                try p.launch(.{ 4, nch, 1 });
                try mg.setBuffer(0, o);
                try mg.setBuffer(1, m);
                try mg.setBuffer(2, l);
                try mg.setBuffer(3, qgb);
                try mg.setBuffer(4, out);
                try mg.setU32(5, ln);
                try mg.launch(.{ 24, 1, 1 });
            }
        }.go;
        try run_old(&part, &merge, q, kc, vc, po, pm, pl, qg, att, @intCast(len), nch_old);
        // new kernel, rows rows at len .. len + rows - 1
        const run_new = struct {
            fn go(p: *rt.Kernel, mg: *rt.Kernel, qb: rt.Buffer, kb: rt.Buffer, vb: rt.Buffer, o: rt.Buffer, m: rt.Buffer, l: rt.Buffer, qgb: rt.Buffer, out: rt.Buffer, len0: u32, nrows: u32) !void {
                const nch: u32 = (len0 + nrows - 1 + LC - 1) / LC;
                try p.setBuffer(0, qb);
                try p.setBuffer(1, kb);
                try p.setBuffer(2, vb);
                try p.setBuffer(3, o);
                try p.setBuffer(4, m);
                try p.setBuffer(5, l);
                try p.setU32(6, len0);
                try p.setF32(7, 0.0625);
                try p.setU32(8, MAXC);
                try p.launch(.{ 4 * nrows, nch, 1 });
                try mg.setBuffer(0, o);
                try mg.setBuffer(1, m);
                try mg.setBuffer(2, l);
                try mg.setBuffer(3, qgb);
                try mg.setBuffer(4, out);
                try mg.setU32(5, len0);
                try mg.setU32(6, MAXC);
                try mg.launch(.{ 24, nrows, 1 });
            }
        }.go;
        try run_new(&lpart, &lmerge, q, kc, vc, lpo, lpm, lpl, qg, latt, @intCast(len), rows);
        // the same row 0 alone, and row 2 alone at len + 2 (qs / qg row 2 are shifted views of a 1-row launch)
        try run_new(&lpart, &lmerge, q, kc, vc, lpo, lpm, lpl, qg, att1, @intCast(len), 1);
        try r.sync();
        try r.download(std.mem.sliceAsBytes(oh), att);
        try r.download(std.mem.sliceAsBytes(nh), latt);
        try r.download(std.mem.sliceAsBytes(n1), att1);
        try r.sync();
        var maxd: f32 = 0;
        var maxv: f32 = 0;
        for (0..24 * 256) |i| {
            maxd = @max(maxd, @abs(bf(oh[i]) - bf(nh[i])));
            maxv = @max(maxv, @abs(bf(oh[i])));
        }
        const same1 = std.mem.eql(u16, n1, nh[0 .. 24 * 256]);
        std.debug.print("len {d:>6}: new vs old max abs diff {e:.2} (max |o| {d:.3}); row 0 of 4-row launch == 1-row launch: {s}\n", .{ len, maxd, maxv, if (same1) "yes" else "NO" });
        if (!same1) fail = true;
    }
    // timing, one row
    for ([_]u32{ 1024, 4096, 8192, 16384, 32768, 65536, 131072 }) |len| {
        if (len > max_len) break;
        const nch_old: u32 = (len + 63) / 64;
        var bo: u64 = std.math.maxInt(u64);
        var bn: u64 = std.math.maxInt(u64);
        for (0..5) |_| {
            const t0 = nowNs();
            for (0..8) |_| {
                try part.setBuffer(0, q);
                try part.setBuffer(1, kc);
                try part.setBuffer(2, vc);
                try part.setBuffer(3, po);
                try part.setBuffer(4, pm);
                try part.setBuffer(5, pl);
                try part.setU32(6, len);
                try part.setF32(7, 0.0625);
                try part.launch(.{ 4, nch_old, 1 });
                try merge.setBuffer(0, po);
                try merge.setBuffer(1, pm);
                try merge.setBuffer(2, pl);
                try merge.setBuffer(3, qg);
                try merge.setBuffer(4, att);
                try merge.setU32(5, len);
                try merge.launch(.{ 24, 1, 1 });
            }
            try r.sync();
            bo = @min(bo, nowNs() - t0);
            const t1 = nowNs();
            for (0..8) |_| {
                try lpart.setBuffer(0, q);
                try lpart.setBuffer(1, kc);
                try lpart.setBuffer(2, vc);
                try lpart.setBuffer(3, lpo);
                try lpart.setBuffer(4, lpm);
                try lpart.setBuffer(5, lpl);
                try lpart.setU32(6, len);
                try lpart.setF32(7, 0.0625);
                try lpart.setU32(8, MAXC);
                try lpart.launch(.{ 4, (len + LC - 1) / LC, 1 });
                try lmerge.setBuffer(0, lpo);
                try lmerge.setBuffer(1, lpm);
                try lmerge.setBuffer(2, lpl);
                try lmerge.setBuffer(3, qg);
                try lmerge.setBuffer(4, latt);
                try lmerge.setU32(5, len);
                try lmerge.setU32(6, MAXC);
                try lmerge.launch(.{ 24, 1, 1 });
            }
            try r.sync();
            bn = @min(bn, nowNs() - t1);
        }
        const bytes = @as(f64, @floatFromInt(len)) * kv_dim * 2 * 2;
        const uo = @as(f64, @floatFromInt(bo)) / 1e3 / 8.0;
        const un = @as(f64, @floatFromInt(bn)) / 1e3 / 8.0;
        std.debug.print("len {d:>6}: old {d:>8.1} us ({d:>5.0} GB/s)   new {d:>8.1} us ({d:>5.0} GB/s, {d:.2} ms per token over 16 layers)\n", .{ len, uo, bytes / uo / 1e3, un, bytes / un / 1e3, un * 16 / 1e3 });
    }
    // rows of a window at 128K keys: K/V re-reads
    for ([_]u32{ 1, 2, 4 }) |nr| {
        const len: u32 = @intCast(max_len - 8);
        var best: u64 = std.math.maxInt(u64);
        for (0..5) |_| {
            const t1 = nowNs();
            for (0..4) |_| {
                try lpart.setBuffer(0, q);
                try lpart.setBuffer(1, kc);
                try lpart.setBuffer(2, vc);
                try lpart.setBuffer(3, lpo);
                try lpart.setBuffer(4, lpm);
                try lpart.setBuffer(5, lpl);
                try lpart.setU32(6, len);
                try lpart.setF32(7, 0.0625);
                try lpart.setU32(8, MAXC);
                try lpart.launch(.{ 4 * nr, (len + nr + LC - 1) / LC, 1 });
                try lmerge.setBuffer(0, lpo);
                try lmerge.setBuffer(1, lpm);
                try lmerge.setBuffer(2, lpl);
                try lmerge.setBuffer(3, qg);
                try lmerge.setBuffer(4, latt);
                try lmerge.setU32(5, len);
                try lmerge.setU32(6, MAXC);
                try lmerge.launch(.{ 24, nr, 1 });
            }
            try r.sync();
            best = @min(best, nowNs() - t1);
        }
        std.debug.print("{d} rows at {d} keys: {d:.1} us per layer\n", .{ nr, len, @as(f64, @floatFromInt(best)) / 1e3 / 4.0 });
    }
    if (fail) return error.RowInvariance;
}
