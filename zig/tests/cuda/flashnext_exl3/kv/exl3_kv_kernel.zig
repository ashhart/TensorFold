//! exl3-kv kernel: the int8 KV attention kernels against upstream's and a float64 oracle.

const std = @import("std");
const flash = @import("flashnext");
const check = @import("../../check.zig");
const rig = @import("../../flashnext_rig.zig");
const Kv = flash.exl3.kv;
const Rig = rig.Rig;
const bf = rig.bf;
const bf16 = rig.bf16;
const kvcodes = @import("exl3_kv_codes.zig");

const q_heads = 24;
const kv_heads = 2;
const head_dim = 256;
const index_heads = 4;
const index_dim = 128;
const half = 32;
const proj_n = 13952;
const eps: f32 = 1e-6;
const idw = 2052;
const top = 512;
const ratio = 4;
const nch = 5;
const group = 32;
const ng = head_dim / group;

const packRow = kvcodes.packRow;
const tileRow = kvcodes.tileRow;
const valueRow = kvcodes.valueRow;
const wht32 = kvcodes.wht32;

/// Keys with a few loud channels, as k_norm + rope leaves them (a lognormal scale a channel).
fn loudRows(r: *Rig, n: usize, width: usize) ![]u16 {
    const scale = try r.a.alloc(f32, width);
    for (scale) |*s| s.* = @exp(r.rng.floatNorm(f32) * 0.8);
    for (0..4) |_| scale[r.rng.uintLessThan(usize, width)] *= 8;
    const v = try r.a.alloc(u16, n * width);
    for (v, 0..) |*x, i| x.* = bf16(r.rng.floatNorm(f32) * scale[i % width] * 0.5);
    return v;
}

/// Sorted distinct blocks: `count` of `complete`, then each block's 4 keys, then the tail (fn_qsa_select's layout).
fn sparseIds(r: *Rig, end: usize, out: []i32) usize {
    const complete = end / ratio;
    var chosen = std.DynamicBitSetUnmanaged.initEmpty(r.a, complete) catch unreachable;
    var got: usize = 0;
    while (got < top) {
        const b = r.rng.uintLessThan(usize, complete);
        if (chosen.isSet(b)) continue;
        chosen.set(b);
        got += 1;
    }
    var k: usize = 0;
    var it = chosen.iterator(.{});
    while (it.next()) |b| {
        for (0..ratio) |j| out[k * ratio + j] = @intCast(b * ratio + j);
        k += 1;
    }
    var n: usize = top * ratio;
    for (ratio * complete..end) |t| {
        out[n] = @intCast(t);
        n += 1;
    }
    return n;
}

/// One row's attention in float64 over value rows `kf`/`vf` [key][head][256] (f64) and bf16 queries.
fn oracleRow(q: []const u16, kf: []const f64, vf: []const f64, keys: []const i32, h: usize, out: []f64, sbuf: []f64) void {
    const kvh = h / (q_heads / kv_heads);
    var best = -std.math.inf(f64);
    for (keys, 0..) |key, j| {
        const k: usize = @intCast(key);
        var dot: f64 = 0;
        for (0..head_dim) |d| dot += bf(q[d]) * kf[(k * kv_heads + kvh) * head_dim + d];
        sbuf[j] = dot * 0.0625;
        best = @max(best, sbuf[j]);
    }
    var l: f64 = 0;
    for (sbuf[0..keys.len]) |*v| {
        v.* = @exp(v.* - best);
        l += v.*;
    }
    for (0..head_dim) |d| {
        var o: f64 = 0;
        for (keys, 0..) |key, j| o += sbuf[j] * vf[(@as(usize, @intCast(key)) * kv_heads + kvh) * head_dim + d];
        out[d] = o / l;
    }
}

pub const Err = struct { max_abs: f64 = 0, peak: f64 = 0, ss: f64 = 0, ss_ref: f64 = 0 };

fn addErr(e: *Err, got: f64, want: f64) void {
    e.max_abs = @max(e.max_abs, @abs(got - want));
    e.peak = @max(e.peak, @abs(want));
    e.ss += (got - want) * (got - want);
    e.ss_ref += want * want;
}

pub fn rel(e: Err) f64 {
    return e.max_abs / e.peak;
}

pub fn rms(e: Err) f64 {
    return @sqrt(e.ss / e.ss_ref);
}

pub fn kernelMode(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x6b76_38);
    var r = try Rig.init(gpu, arena.allocator(), prng.random());
    defer r.deinit();
    var g = try flash.Glue.load(gpu.d);
    defer g.deinit();
    var k8 = try Kv.Kv8.load(gpu.d);
    defer k8.deinit();
    const a = r.a;
    const cap = 4096;
    const kc = try loudRows(&r, cap * kv_heads, head_dim);
    const vc = try loudRows(&r, cap * kv_heads, head_dim);
    const nrow = cap * kv_heads;
    var failed = false;

    // 1. fn_kv8_pack == the host's codes and scales, both domains
    for ([_]bool{ false, true }) |rot| {
        const codes = try a.alloc(i8, nrow * head_dim);
        const scales = try a.alloc(f16, nrow * ng);
        for (0..nrow) |i| packRow(kc[i * head_dim ..][0..head_dim], rot, codes[i * head_dim ..][0..head_dim], scales[i * ng ..][0..ng]);
        const dx = try r.dev(std.mem.sliceAsBytes(kc));
        const dc = try r.out(nrow * head_dim);
        const ds = try r.out(nrow * ng * 2);
        try k8.packRows(r.s, dx, dc, ds, nrow, rot);
        try check.sameBytes(if (rot) "pack codes (rotated)" else "pack codes", std.mem.sliceAsBytes(try r.back(i8, dc, codes.len)), std.mem.sliceAsBytes(codes));
        try check.sameBytes(if (rot) "pack scales (rotated)" else "pack scales", std.mem.sliceAsBytes(try r.back(f16, ds, scales.len)), std.mem.sliceAsBytes(scales));
    }
    check.pass("kv8 pack: fn_kv8_pack's codes and fp16 scales equal the host's bit for bit ({d} rows of 256, plain and rotated)", .{nrow});

    // 2. prep: upstream's Q, indexer rows, and its K/V cache rows packed == fn_kv8_prep's (plain); rotated K/V too
    {
        const rows = 17;
        const pos0 = 700;
        const p = r.bfs(rows * proj_n, 2);
        const qw = r.f32s(head_dim, 0.5, 1.5);
        const kw = r.f32s(head_dim, 0.5, 1.5);
        const iw = r.f32s(index_dim, 0.5, 1.5);
        const inv = try a.alloc(f32, half);
        for (inv, 0..) |*o, i| o.* = @floatCast(std.math.pow(f64, 10_000_000.0, -@as(f64, @floatFromInt(i)) / half));
        const dp = try r.dev(std.mem.sliceAsBytes(p));
        const dpos = try r.dev(std.mem.sliceAsBytes(&[_]i32{ pos0, 0, 0, 0 }));
        const dqw = try r.dev(std.mem.sliceAsBytes(qw));
        const dkw = try r.dev(std.mem.sliceAsBytes(kw));
        const diw = try r.dev(std.mem.sliceAsBytes(iw));
        const dinv = try r.dev(std.mem.sliceAsBytes(inv));
        const end = pos0 + rows;
        const uq = try r.out(rows * q_heads * head_dim * 2);
        const ukc = try r.out(end * kv_heads * head_dim * 2);
        const uvc = try r.out(end * kv_heads * head_dim * 2);
        const uiq = try r.out(rows * index_heads * index_dim * 2);
        const uik = try r.out(end * index_dim * 2);
        try g.attnPrep(r.s, dp, dpos, dqw, dkw, diw, dinv, uq, ukc, uvc, uiq, uik, eps, rows, .{});
        const ukc_h = try r.back(u16, ukc, end * kv_heads * head_dim);
        const uvc_h = try r.back(u16, uvc, end * kv_heads * head_dim);
        for ([_]bool{ false, true }) |rot| {
            const oq = try r.out(rows * q_heads * head_dim * 2);
            const c: Kv.Kv8.Cache = .{ .k8 = try r.out(end * kv_heads * head_dim), .ks = try r.out(end * kv_heads * ng * 2), .v8 = try r.out(end * kv_heads * head_dim), .vs = try r.out(end * kv_heads * ng * 2) };
            const oiq = try r.out(rows * index_heads * index_dim * 2);
            const oik = try r.out(end * index_dim * 2);
            try k8.prepRows(r.s, dp, dpos, dqw, dkw, diw, dinv, oq, c, oiq, oik, eps, rows, proj_n, q_heads, kv_heads, rot);
            if (!rot) try check.sameBytes("prep q", std.mem.sliceAsBytes(try r.back(u16, oq, rows * q_heads * head_dim)), std.mem.sliceAsBytes(try r.back(u16, uq, rows * q_heads * head_dim)));
            try check.sameBytes("prep iq", std.mem.sliceAsBytes(try r.back(u16, oiq, rows * index_heads * index_dim)), std.mem.sliceAsBytes(try r.back(u16, uiq, rows * index_heads * index_dim)));
            try check.sameBytes("prep ik", std.mem.sliceAsBytes((try r.back(u16, oik, end * index_dim))[pos0 * index_dim ..]), std.mem.sliceAsBytes((try r.back(u16, uik, end * index_dim))[pos0 * index_dim ..]));
            const n = rows * kv_heads;
            const want_c = try a.alloc(i8, n * head_dim);
            const want_s = try a.alloc(f16, n * ng);
            for ([_]u64{ c.k8, c.v8 }, [_]u64{ c.ks, c.vs }, [_][]const u16{ ukc_h, uvc_h }) |dc, ds, src| {
                for (0..n) |i| packRow(src[(pos0 * kv_heads + i) * head_dim ..][0..head_dim], rot, want_c[i * head_dim ..][0..head_dim], want_s[i * ng ..][0..ng]);
                try check.sameBytes("prep k/v codes", std.mem.sliceAsBytes((try r.back(i8, dc, end * kv_heads * head_dim))[pos0 * kv_heads * head_dim ..]), std.mem.sliceAsBytes(want_c));
                try check.sameBytes("prep k/v scales", std.mem.sliceAsBytes((try r.back(f16, ds, end * kv_heads * ng))[pos0 * kv_heads * ng ..]), std.mem.sliceAsBytes(want_s));
            }
            if (rot) {
                // a rotated query, rotated back on the host, is upstream's query within one bf16 rounding of each side
                const got = try r.back(u16, oq, rows * q_heads * head_dim);
                const want = try r.back(u16, uq, rows * q_heads * head_dim);
                var worst: f64 = 0;
                for (0..rows * q_heads * ng) |gi| {
                    var x: [group]f64 = undefined;
                    for (0..group) |i| {
                        var acc: f64 = 0;
                        for (0..group) |j| acc += (if (@popCount(i & j) % 2 == 0) @as(f64, 1) else -1) * bf(got[gi * group + j]);
                        x[i] = acc / @sqrt(32.0);
                    }
                    var peak: f64 = 0;
                    for (0..group) |i| peak = @max(peak, @abs(bf(want[gi * group + i])));
                    for (0..group) |i| worst = @max(worst, @abs(x[i] - bf(want[gi * group + i])) / (peak + 1e-30));
                }
                std.debug.print("  rotated query rotated back vs upstream's query: worst {e:.3} of the group's peak\n", .{worst});
                try check.expect(worst < 1.6e-2, "rotated query off by {e:.3}", .{worst});
            }
        }
        check.pass("kv8 prep: upstream's q, index q/k bit-equal; K/V codes and scales == the host's packing of upstream's bf16 cache rows (plain and rotated)", .{});
    }

    // 3. attention: int8 kernels vs upstream's kernels on the dequantised bf16 cache (bits, plain), and vs float64
    for ([_]bool{ false, true }) |sparse| {
        const rows = 20; // up to 16 rows take the read-ahead kernel; 20 one tile at a time (both run below)
        const pos0: usize = if (sparse) 3000 else 1100;
        const q = r.bfs(rows * q_heads * head_dim, 1);
        const ids = try a.alloc(i32, rows * idw);
        const nk = try a.alloc(i32, rows);
        const spr = try a.alloc(i32, rows);
        @memset(ids, 0);
        for (0..rows) |row| {
            const end = pos0 + row + 1;
            spr[row] = @intFromBool(sparse);
            nk[row] = @intCast(if (sparse) sparseIds(&r, end, ids[row * idw ..][0..idw]) else end);
            if (!sparse) for (0..end) |i| {
                ids[row * idw + i] = @intCast(@min(i, idw - 1));
            };
        }
        const dpos = try r.dev(std.mem.sliceAsBytes(&[_]i32{ @intCast(pos0), 0, 0, 0 }));
        const dids = try r.dev(std.mem.sliceAsBytes(ids));
        const dnk = try r.dev(std.mem.sliceAsBytes(nk));
        const dspr = try r.dev(std.mem.sliceAsBytes(spr));
        const po = try r.out(rows * nch * q_heads * head_dim * 4);
        const pm = try r.out(rows * nch * q_heads * 4);
        const pl = try r.out(rows * nch * q_heads * 4);
        const chunks = if (sparse) nch else (pos0 + rows + 511) / 512;
        // the float64 reference over the ORIGINAL bf16 cache, shared by every variant
        const kf0 = try a.alloc(f64, nrow * head_dim);
        const vf0 = try a.alloc(f64, nrow * head_dim);
        for (kf0, kc) |*o, x| o.* = bf(x);
        for (vf0, vc) |*o, x| o.* = bf(x);
        const want = try a.alloc(f64, rows * q_heads * head_dim);
        const sbuf = try a.alloc(f64, pos0 + rows + idw);
        const keys_of = struct {
            fn get(ids_: []const i32, nk_: []const i32, row: usize, buf: []i32, sp: bool) []const i32 {
                const n: usize = @intCast(nk_[row]);
                if (sp) return ids_[row * idw ..][0..n];
                for (0..n) |i| buf[i] = @intCast(i);
                return buf[0..n];
            }
        };
        const kbuf = try a.alloc(i32, pos0 + rows);
        for (0..rows) |row| for (0..q_heads) |h| oracleRow(q[(row * q_heads + h) * head_dim ..][0..head_dim], kf0, vf0, keys_of.get(ids, nk, row, kbuf, sparse), h, want[(row * q_heads + h) * head_dim ..][0..head_dim], sbuf);
        // upstream on the original bf16 cache: its error is the baseline
        const dkc = try r.dev(std.mem.sliceAsBytes(kc));
        const dvc = try r.dev(std.mem.sliceAsBytes(vc));
        const base_out = try r.out(rows * q_heads * head_dim * 2);
        try g.attnChunks(r.s, try r.dev(std.mem.sliceAsBytes(q)), dkc, dvc, dpos, po, pm, pl, dids, dnk, dspr, rows, kv_heads, chunks, true, .{});
        try g.attnMerge(r.s, po, pm, pl, dpos, base_out, dnk, dspr, rows, kv_heads, true);
        var e_base: Err = .{};
        for (try r.back(u16, base_out, want.len), want) |x, w| addErr(&e_base, bf(x), w);
        for ([_]bool{ false, true }) |rot| {
            const codes_k = try a.alloc(i8, nrow * head_dim);
            const codes_v = try a.alloc(i8, nrow * head_dim);
            const sc_k = try a.alloc(f16, nrow * ng);
            const sc_v = try a.alloc(f16, nrow * ng);
            for (0..nrow) |i| {
                packRow(kc[i * head_dim ..][0..head_dim], rot, codes_k[i * head_dim ..][0..head_dim], sc_k[i * ng ..][0..ng]);
                packRow(vc[i * head_dim ..][0..head_dim], rot, codes_v[i * head_dim ..][0..head_dim], sc_v[i * ng ..][0..ng]);
            }
            const c: Kv.Kv8.Cache = .{ .k8 = try r.dev(std.mem.sliceAsBytes(codes_k)), .ks = try r.dev(std.mem.sliceAsBytes(sc_k)), .v8 = try r.dev(std.mem.sliceAsBytes(codes_v)), .vs = try r.dev(std.mem.sliceAsBytes(sc_v)) };
            // queries as the prep stores them: rotated (fp32) then rounded once
            const qr = try a.alloc(u16, q.len);
            for (0..q.len / group) |gi| {
                var x: [group]f32 = undefined;
                for (0..group) |j| x[j] = @floatCast(bf(q[gi * group + j]));
                if (rot) wht32(&x);
                for (0..group) |j| qr[gi * group + j] = bf16(x[j]);
            }
            const dq = try r.dev(std.mem.sliceAsBytes(qr));
            var outs: [2]u64 = undefined;
            for ([_]usize{ 16, rows }, 0..) |m, ci| {
                outs[ci] = try r.out(m * q_heads * head_dim * 2);
                try k8.attnChunks(r.s, dq, c, dpos, po, pm, pl, dids, dnk, dspr, m, q_heads, kv_heads, chunks, true);
                try k8.attnMerge(r.s, po, pm, pl, dpos, outs[ci], dnk, dspr, m, q_heads, kv_heads, true, rot);
            }
            const got = try r.back(u16, outs[1], want.len);
            try check.sameBytes("kv8 attention: 16 rows (read-ahead kernel) == the same rows of 20 (one tile at a time)", std.mem.sliceAsBytes(try r.back(u16, outs[0], 16 * q_heads * head_dim)), std.mem.sliceAsBytes(got[0 .. 16 * q_heads * head_dim]));
            if (!rot) {
                // upstream's kernels on a bf16 cache holding the tile values: the same bits
                const tk = try a.alloc(u16, nrow * head_dim);
                const tv = try a.alloc(u16, nrow * head_dim);
                for (0..nrow) |i| {
                    tileRow(codes_k[i * head_dim ..][0..head_dim], sc_k[i * ng ..][0..ng], tk[i * head_dim ..][0..head_dim]);
                    tileRow(codes_v[i * head_dim ..][0..head_dim], sc_v[i * ng ..][0..ng], tv[i * head_dim ..][0..head_dim]);
                }
                const up = try r.out(rows * q_heads * head_dim * 2);
                try g.attnChunks(r.s, dq, try r.dev(std.mem.sliceAsBytes(tk)), try r.dev(std.mem.sliceAsBytes(tv)), dpos, po, pm, pl, dids, dnk, dspr, rows, kv_heads, chunks, true, .{});
                try g.attnMerge(r.s, po, pm, pl, dpos, up, dnk, dspr, rows, kv_heads, true);
                try check.sameBytes("kv8 attention == upstream's chunk/merge on the dequantised bf16 cache", std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(try r.back(u16, up, want.len)));
            }
            // vs the float64 of the original cache, and vs the float64 of what the int8 cache holds.
            var e_q: Err = .{};
            for (got, want) |x, w| addErr(&e_q, bf(x), w);
            const kf = try a.alloc(f64, nrow * head_dim);
            const vf = try a.alloc(f64, nrow * head_dim);
            for (0..nrow) |i| {
                valueRow(codes_k[i * head_dim ..][0..head_dim], sc_k[i * ng ..][0..ng], rot, kf[i * head_dim ..][0..head_dim]);
                valueRow(codes_v[i * head_dim ..][0..head_dim], sc_v[i * ng ..][0..ng], rot, vf[i * head_dim ..][0..head_dim]);
            }
            const own = try a.alloc(f64, want.len);
            for (0..rows) |row| for (0..q_heads) |h| oracleRow(q[(row * q_heads + h) * head_dim ..][0..head_dim], kf, vf, keys_of.get(ids, nk, row, kbuf, sparse), h, own[(row * q_heads + h) * head_dim ..][0..head_dim], sbuf);
            var e_own: Err = .{};
            for (got, own) |x, w| addErr(&e_own, bf(x), w);
            var e_cache: Err = .{};
            for (own, want) |x, w| addErr(&e_cache, x, w);
            std.debug.print("  {s} {s}: vs float64 of the bf16 cache max {e:.3} rms {e:.3} of peak (upstream bf16: {e:.3} / {e:.3}); int8 cache alone (f64) {e:.3} / {e:.3}; kernel vs f64 of its own cache {e:.3} / {e:.3}\n", .{ if (sparse) "sparse" else "dense", if (rot) "int8 rotated" else "int8 plain", rel(e_q), rms(e_q), rel(e_base), rms(e_base), rel(e_cache), rms(e_cache), rel(e_own), rms(e_own) });
            // the kernel's own error stays at upstream's (bf16 tiles, bf16 p): within 2x of the baseline's
            if (rel(e_own) > 2 * rel(e_base) + 1e-3) {
                std.debug.print("FAIL kv8 attention: kernel error {e:.3} vs its own cache's float64 is more than twice upstream's {e:.3}\n", .{ rel(e_own), rel(e_base) });
                failed = true;
            }
        }
    }
    try check.expect(!failed, "exl3-kv kernel: a check failed", .{});
    check.pass("kv8 attention: plain int8 == upstream's kernels on the dequantised cache bit for bit (dense, sparse, read-ahead and one-tile); rotated within upstream's own error of its cache's float64", .{});
}
