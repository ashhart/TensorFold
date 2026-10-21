//! exl3-kv: the KV-memory levers for 200k contexts (docs/plans/exl3-kv-memory.md), on one card.
//! Usage: tf-cuda-test exl3-kv kernel          int8 KV kernels (flashnext_exl3_kv.cu) vs the host codes, vs upstream's
//!                                              chunk/merge kernels on the dequantised bf16 cache (bits), vs float64
//!        tf-cuda-test exl3-kv bench [CAP]     decode and prompt attention over a CAP-token cache (200,016): bf16
//!                                              (upstream) and int8 (ours), device or mapped pinned host memory; the
//!                                              copy-engine and gather alternatives; host round trip of the ids
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");

const Kv = flash.exl3.kv;
const Rig = rig.Rig;
const bf = rig.bf;
const bf16 = rig.bf16;

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

// -- the host's codes, bit for bit what fn_kv8_pack / fn_kv8_prep write -----------------------------------------------

const inv_sqrt32: f32 = 0.17677669529663688;

/// The normalized Sylvester Hadamard of 32 values in fp32, the kernel's butterfly order.
pub fn wht32(x: *[group]f32) void {
    var m: usize = 1;
    while (m < group) : (m <<= 1) {
        var y: [group]f32 = undefined;
        for (0..group) |i| y[i] = if (i & m != 0) x[i ^ m] - x[i] else x[i] + x[i ^ m];
        x.* = y;
    }
    for (x) |*v| v.* *= inv_sqrt32;
}

/// One group of 32 values (already rotated if wanted) into its codes and fp16 scale.
pub fn quant8(x: *const [group]f32, codes: *[group]i8) f16 {
    var peak: f32 = 0;
    for (x) |v| peak = @max(peak, @abs(v));
    const sh: f16 = @floatCast(@max(peak, 6.103515625e-05));
    const s: f32 = @floatCast(sh);
    for (x, codes) |v, *c| {
        const q = std.math.clamp(@floor(v / s * 128.0) + 128.0, 0.0, 255.0);
        c.* = @intCast(@as(i32, @intFromFloat(q)) - 128);
    }
    return sh;
}

pub fn dequant8(c: i8, s: f16) f32 {
    return (@as(f32, @floatFromInt(c)) + 0.5) * (@as(f32, @floatCast(s)) * 0.0078125);
}

/// int4 (host only, for the quality comparison): exl3_cache.zig quantize4's grid, the scale rounded to fp16 first.
pub fn roundTrip4(x: *const [group]f32, out: *[group]f32) void {
    var peak: f32 = 0;
    for (x) |v| peak = @max(peak, @abs(v));
    const s: f32 = @floatCast(@as(f16, @floatCast(@max(peak, 6.103515625e-05))));
    for (x, out) |v, *o| {
        const q = std.math.clamp(@floor(v / s * 8.0) + 8.0, 0.0, 15.0);
        o.* = (q - 7.5) * s * 0.125;
    }
}

/// A row of 256 bf16 into codes and scales (rot: each group rotated first).
pub fn packRow(x: []const u16, rot: bool, codes: []i8, scales: []f16) void {
    for (0..ng) |gi| {
        var g: [group]f32 = undefined;
        for (0..group) |j| g[j] = @floatCast(bf(x[gi * group + j]));
        if (rot) wht32(&g);
        scales[gi] = quant8(&g, codes[gi * group ..][0..group]);
    }
}

/// The dequantised row as the chunk kernel's bf16 tile holds it (still in the rotated domain when rot).
pub fn tileRow(codes: []const i8, scales: []const f16, out: []u16) void {
    for (0..head_dim) |d| out[d] = bf16(dequant8(codes[d], scales[d / group]));
}

/// The dequantised row back in the model's domain, in float64 (the oracle's view of what the cache holds).
pub fn valueRow(codes: []const i8, scales: []const f16, rot: bool, out: []f64) void {
    for (0..ng) |gi| {
        var g: [group]f64 = undefined;
        for (0..group) |j| g[j] = dequant8(codes[gi * group + j], scales[gi]);
        if (rot) {
            // H is symmetric and orthonormal: its own inverse
            var y: [group]f64 = undefined;
            for (0..group) |i| {
                var acc: f64 = 0;
                for (0..group) |j| acc += (if (@popCount(i & j) % 2 == 0) @as(f64, 1) else -1) * g[j];
                y[i] = acc / @sqrt(32.0);
            }
            g = y;
        }
        @memcpy(out[gi * group ..][0..group], &g);
    }
}

// -- kernel mode -------------------------------------------------------------------------------------------------------

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

fn kernelMode(gpu: check.Gpu) !void {
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
        try g.attnPrep(r.s, dp, dpos, dqw, dkw, diw, dinv, uq, ukc, uvc, uiq, uik, eps, rows);
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
        try g.attnChunks(r.s, try r.dev(std.mem.sliceAsBytes(q)), dkc, dvc, dpos, po, pm, pl, dids, dnk, dspr, rows, kv_heads, chunks, true);
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
                try g.attnChunks(r.s, dq, try r.dev(std.mem.sliceAsBytes(tk)), try r.dev(std.mem.sliceAsBytes(tv)), dpos, po, pm, pl, dids, dnk, dspr, rows, kv_heads, chunks, true);
                try g.attnMerge(r.s, po, pm, pl, dpos, up, dnk, dspr, rows, kv_heads, true);
                try check.sameBytes("kv8 attention == upstream's chunk/merge on the dequantised bf16 cache", std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(try r.back(u16, up, want.len)));
            }
            // vs the float64 of the original cache, and vs the float64 of what the int8 cache holds (its own kernel error)
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

// -- bench mode ---------------------------------------------------------------------------------------------------------

const Bench = struct {
    s: cuda.Stream,
    ev0: cuda.Event,
    ev1: cuda.Event,
    reps: usize = 20,

    fn time(b: *Bench, ctx: anytype, comptime f: fn (@TypeOf(ctx)) anyerror!void) !f64 {
        for (0..3) |_| try f(ctx);
        try b.s.synchronize();
        var best: f64 = std.math.inf(f64);
        for (0..5) |_| {
            try b.ev0.record(b.s);
            for (0..b.reps) |_| try f(ctx);
            try b.ev1.record(b.s);
            try b.ev1.synchronize();
            best = @min(best, @as(f64, try b.ev0.elapsedMs(b.ev1)) / @as(f64, @floatFromInt(b.reps)));
        }
        return best;
    }
};

/// A cache of `cap` tokens in one place: bf16 K/V and its int8 codes + scales (same values, packed on the device).
const Place = struct {
    kc: u64,
    vc: u64,
    c: Kv.Kv8.Cache,
};

const Attn = struct {
    g: *const flash.Glue,
    k8: *const Kv.Kv8,
    s: cuda.Stream,
    p: Place,
    int8: bool,
    q: u64,
    pos: u64,
    po: u64,
    pm: u64,
    pl: u64,
    ids: u64,
    nk: u64,
    spr: u64,
    out: u64,
    rows: usize,
    chunks: usize,

    fn go(a: *const Attn) anyerror!void {
        if (a.int8) {
            try a.k8.attnChunks(a.s, a.q, a.p.c, a.pos, a.po, a.pm, a.pl, a.ids, a.nk, a.spr, a.rows, q_heads, kv_heads, a.chunks, true);
            try a.k8.attnMerge(a.s, a.po, a.pm, a.pl, a.pos, a.out, a.nk, a.spr, a.rows, q_heads, kv_heads, true, true);
        } else {
            try a.g.attnChunks(a.s, a.q, a.p.kc, a.p.vc, a.pos, a.po, a.pm, a.pl, a.ids, a.nk, a.spr, a.rows, kv_heads, a.chunks, true);
            try a.g.attnMerge(a.s, a.po, a.pm, a.pl, a.pos, a.out, a.nk, a.spr, a.rows, kv_heads, true);
        }
    }
};

/// `rows` consecutive sparse rows ending at `end0 + rows`: row 0 picks `top` random blocks, each next row keeps a
/// fraction `keep` of the previous row's blocks and draws the rest anew (the selection overlap a real prompt shows).
fn overlapIds(rng: std.Random, a: std.mem.Allocator, end0: usize, rows: usize, keep: f64, ids: []i32, nk: []i32) !void {
    const complete0 = (end0 + 1) / ratio;
    var cur = try std.DynamicBitSetUnmanaged.initEmpty(a, complete0 + rows);
    var got: usize = 0;
    while (got < top) {
        const b = rng.uintLessThan(usize, complete0);
        if (cur.isSet(b)) continue;
        cur.set(b);
        got += 1;
    }
    for (0..rows) |row| {
        const end = end0 + row + 1;
        const complete = end / ratio;
        if (row > 0) {
            // drop (1 - keep) of the set, refill from all complete blocks
            var drop: usize = @intFromFloat(@round((1 - keep) * @as(f64, top)));
            while (drop > 0) {
                const b = rng.uintLessThan(usize, complete);
                if (!cur.isSet(b)) continue;
                cur.unset(b);
                drop -= 1;
            }
            while (cur.count() < top) {
                const b = rng.uintLessThan(usize, complete);
                if (!cur.isSet(b)) cur.set(b);
            }
        }
        var k: usize = 0;
        var it = cur.iterator(.{});
        while (it.next()) |b| {
            for (0..ratio) |j| ids[row * idw + k * ratio + j] = @intCast(b * ratio + j);
            k += 1;
        }
        var n: usize = top * ratio;
        for (ratio * complete..end) |t| {
            ids[row * idw + n] = @intCast(t);
            n += 1;
        }
        nk[row] = @intCast(n);
    }
}

const Copy = struct {
    s: cuda.Stream,
    dst: u64,
    src: [*]const u8,
    bytes: usize,
    pieces: usize,
    fn go(c: *const Copy) anyerror!void {
        const d = c.s.d;
        const each = c.bytes / c.pieces;
        for (0..c.pieces) |i| try d.check(d.api.cuMemcpyHtoDAsync_v2(c.dst + i * each, c.src + i * each, each, c.s.handle), "cuMemcpyHtoDAsync");
    }
};

const Gather = struct {
    k8: *const Kv.Kv8,
    s: cuda.Stream,
    src_k: u64,
    src_v: u64,
    dst_k: u64,
    dst_v: u64,
    ids: u64,
    n: usize,
    row: usize,
    fn go(g: *const Gather) anyerror!void {
        try g.k8.gatherRows(g.s, g.src_k, g.dst_k, g.ids, g.n, g.row);
        try g.k8.gatherRows(g.s, g.src_v, g.dst_v, g.ids, g.n, g.row);
    }
};

fn benchMode(gpu: check.Gpu, cap: usize) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0xbe7c4);
    const rng = prng.random();
    var r = try Rig.init(gpu, a, rng);
    defer r.deinit();
    var g = try flash.Glue.load(gpu.d);
    defer g.deinit();
    var k8 = try Kv.Kv8.load(gpu.d);
    defer k8.deinit();
    var b: Bench = .{ .s = r.s, .ev0 = try cuda.Event.init(gpu.d, true), .ev1 = try cuda.Event.init(gpu.d, true) };
    {
        var free_b: usize = 0;
        var total_b: usize = 0;
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free_b, &total_b), "cuMemGetInfo");
        std.debug.print("  device after context + the attention modules: free {d:.3} of {d:.3} GiB (context and modules {d:.3} GiB)\n", .{ @as(f64, @floatFromInt(free_b)) / Kv.gib, @as(f64, @floatFromInt(total_b)) / Kv.gib, @as(f64, @floatFromInt(total_b - free_b)) / Kv.gib });
    }
    const nrow = cap * kv_heads;
    const kv_bytes = nrow * head_dim * 2;
    const cb = Kv.Kv8.cacheBytes(cap, kv_heads);
    std.debug.print("== exl3-kv bench: cache of {d} tokens, one attention layer: bf16 K+V {d:.1} MiB, int8 codes+scales {d:.1} MiB\n", .{ cap, @as(f64, @floatFromInt(2 * kv_bytes)) / 1048576.0, @as(f64, @floatFromInt(2 * (cb.codes + cb.scales))) / 1048576.0 });

    // the cache's values: random bf16 made on the device side once, then packed; the host copies get the same bytes
    var dev_k = try cuda.DeviceBuffer.alloc(gpu.d, kv_bytes);
    defer dev_k.free();
    var dev_v = try cuda.DeviceBuffer.alloc(gpu.d, kv_bytes);
    defer dev_v.free();
    var host_k = try cuda.HostBuffer.allocMapped(gpu.d, kv_bytes);
    defer host_k.free();
    var host_v = try cuda.HostBuffer.allocMapped(gpu.d, kv_bytes);
    defer host_v.free();
    {
        const hk = host_k.slice(u16);
        const hv = host_v.slice(u16);
        for (hk) |*x| x.* = bf16(rng.floatNorm(f32) * 0.5);
        for (hv) |*x| x.* = bf16(rng.floatNorm(f32) * 0.5);
        try dev_k.upload(0, host_k.bytes);
        try dev_v.upload(0, host_v.bytes);
    }
    var dev_c: [4]cuda.DeviceBuffer = undefined;
    var host_c: [4]cuda.HostBuffer = undefined;
    for (0..4) |i| {
        const n = if (i % 2 == 0) cb.codes else cb.scales;
        dev_c[i] = try cuda.DeviceBuffer.alloc(gpu.d, n);
        host_c[i] = try cuda.HostBuffer.allocMapped(gpu.d, n);
    }
    defer for (&dev_c, &host_c) |*x, *y| {
        x.free();
        y.free();
    };
    try k8.packRows(r.s, dev_k.ptr, dev_c[0].ptr, dev_c[1].ptr, nrow, true);
    try k8.packRows(r.s, dev_v.ptr, dev_c[2].ptr, dev_c[3].ptr, nrow, true);
    try r.s.synchronize();
    for (0..4) |i| try dev_c[i].download(0, host_c[i].bytes);
    const on_dev: Place = .{ .kc = dev_k.ptr, .vc = dev_v.ptr, .c = .{ .k8 = dev_c[0].ptr, .ks = dev_c[1].ptr, .v8 = dev_c[2].ptr, .vs = dev_c[3].ptr } };
    const on_host: Place = .{ .kc = try host_k.device(), .vc = try host_v.device(), .c = .{ .k8 = try host_c[0].device(), .ks = try host_c[1].device(), .v8 = try host_c[2].device(), .vs = try host_c[3].device() } };

    const max_rows = 256;
    const q = r.bfs(max_rows * q_heads * head_dim, 1);
    const dq = try r.dev(std.mem.sliceAsBytes(q));
    const po = try r.out(max_rows * nch * q_heads * head_dim * 4);
    const pm = try r.out(max_rows * nch * q_heads * 4);
    const pl = try r.out(max_rows * nch * q_heads * 4);
    const out = try r.out(max_rows * q_heads * head_dim * 2);
    const ids = try a.alloc(i32, max_rows * idw);
    const nk = try a.alloc(i32, max_rows);
    const spr = try a.alloc(i32, max_rows);

    const Case = struct { name: []const u8, rows: usize, end0: usize, sparse: bool, keep: f64 };
    const cases = [_]Case{
        .{ .name = "decode 1 row, sparse at the cache end", .rows = 1, .end0 = cap - 2, .sparse = true, .keep = 0 },
        .{ .name = "decode window 4 rows (MTP verify), keep 0.95", .rows = 4, .end0 = cap - 5, .sparse = true, .keep = 0.95 },
        .{ .name = "prompt 256 rows, sparse, keep 0.95", .rows = 256, .end0 = cap - 257, .sparse = true, .keep = 0.95 },
        .{ .name = "prompt 256 rows, sparse, keep 0.80", .rows = 256, .end0 = cap - 257, .sparse = true, .keep = 0.80 },
        .{ .name = "prompt 256 rows, sparse, keep 0 (worst case)", .rows = 256, .end0 = cap - 257, .sparse = true, .keep = 0 },
        .{ .name = "prompt 256 rows, dense (keys 1,795..2,050)", .rows = 256, .end0 = 1794, .sparse = false, .keep = 0 },
    };
    for (cases) |c| {
        @memset(ids, 0);
        if (c.sparse) {
            try overlapIds(rng, a, c.end0, c.rows, c.keep, ids, nk);
        } else for (0..c.rows) |row| {
            nk[row] = @intCast(c.end0 + row + 1);
        }
        for (spr[0..c.rows]) |*x| x.* = @intFromBool(c.sparse);
        // distinct keys these rows read (what crosses PCIe at least once)
        var seen = try std.DynamicBitSetUnmanaged.initEmpty(a, cap);
        var reads: usize = 0;
        for (0..c.rows) |row| {
            const n: usize = @intCast(nk[row]);
            reads += n;
            for (0..n) |i| seen.set(if (c.sparse) @intCast(ids[row * idw + i]) else i);
        }
        const distinct = seen.count();
        const pos0: i32 = @intCast(c.end0 + 1);
        const dpos = try r.dev(std.mem.sliceAsBytes(&[_]i32{ pos0, 0, 0, 0 }));
        const dids = try r.dev(std.mem.sliceAsBytes(ids[0 .. c.rows * idw]));
        const dnk = try r.dev(std.mem.sliceAsBytes(nk[0..c.rows]));
        const dspr = try r.dev(std.mem.sliceAsBytes(spr[0..c.rows]));
        const chunks = if (c.sparse) nch else (c.end0 + c.rows + 1 + 511) / 512;
        var ms: [4]f64 = undefined;
        for ([_]bool{ false, true }, 0..) |int8, ii| for ([_]Place{ on_dev, on_host }, 0..) |p, pi| {
            const at: Attn = .{ .g = &g, .k8 = &k8, .s = r.s, .p = p, .int8 = int8, .q = dq, .pos = dpos, .po = po, .pm = pm, .pl = pl, .ids = dids, .nk = dnk, .spr = dspr, .out = out, .rows = c.rows, .chunks = chunks };
            b.reps = if (pi == 1 and c.rows > 4) 3 else 20;
            ms[ii * 2 + pi] = try b.time(&at, Attn.go);
        };
        // the int8 kernel with the next tile in flight at every row count (Kv.ahead_rows raised)
        var ms_ahead: f64 = ms[2];
        if (c.rows > Kv.ahead_rows) {
            const keep_rows = Kv.ahead_rows;
            Kv.ahead_rows = 1 << 20;
            defer Kv.ahead_rows = keep_rows;
            const at: Attn = .{ .g = &g, .k8 = &k8, .s = r.s, .p = on_dev, .int8 = true, .q = dq, .pos = dpos, .po = po, .pm = pm, .pl = pl, .ids = dids, .nk = dnk, .spr = dspr, .out = out, .rows = c.rows, .chunks = chunks };
            b.reps = 20;
            ms_ahead = try b.time(&at, Attn.go);
        }
        const key_bf: f64 = 2 * kv_heads * head_dim * 2; // one key's K and V, bf16, both heads
        const key_i8: f64 = @floatFromInt(Kv.kvRowBytes(.int8, kv_heads));
        const once_bf = @as(f64, @floatFromInt(distinct)) * key_bf / 1e6;
        const once_i8 = @as(f64, @floatFromInt(distinct)) * key_i8 / 1e6;
        std.debug.print("RESULT {s}: keys read {d}, distinct {d} ({d:.2} MB bf16 / {d:.2} MB int8 once)\n", .{ c.name, reads, distinct, once_bf, once_i8 });
        std.debug.print("  chunks+merge ms: bf16 device {d:.4}  bf16 host-mapped {d:.4} ({d:.1} GB/s of distinct)  int8 device {d:.4} (read-ahead kernel {d:.4})  int8 host-mapped {d:.4} ({d:.1} GB/s of distinct)\n", .{ ms[0], ms[1], once_bf / ms[1], ms[2], ms_ahead, ms[3], once_i8 / ms[3] });
    }

    // explicit staging alternatives for one decode row's 2,051 keys
    {
        const n = 2051;
        const row_bytes = kv_heads * head_dim * 2; // one key, both heads, K (V the same)
        const bytes = 2 * n * row_bytes;
        var pinned = try cuda.HostBuffer.alloc(gpu.d, bytes);
        defer pinned.free();
        var staging = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
        defer staging.free();
        const one: Copy = .{ .s = r.s, .dst = staging.ptr, .src = pinned.bytes.ptr, .bytes = bytes, .pieces = 1 };
        b.reps = 20;
        const t_one = try b.time(&one, Copy.go);
        const blocks: Copy = .{ .s = r.s, .dst = staging.ptr, .src = pinned.bytes.ptr, .bytes = 1024 * (bytes / 1024), .pieces = 1024 };
        const t_blocks = try b.time(&blocks, Copy.go);
        try overlapIds(rng, a, cap - 2, 1, 0, ids, nk);
        const dids = try r.dev(std.mem.sliceAsBytes(ids[0..n]));
        const gk: Gather = .{ .k8 = &k8, .s = r.s, .src_k = on_host.kc, .src_v = on_host.vc, .dst_k = staging.ptr, .dst_v = staging.ptr + n * row_bytes, .ids = dids, .n = n, .row = row_bytes };
        const t_gather = try b.time(&gk, Gather.go);
        // the selection's ids to the host and back to a waiting thread: one D2H and a stream synchronize
        var hid = try cuda.HostBuffer.alloc(gpu.d, n * 4);
        defer hid.free();
        var best: f64 = std.math.inf(f64);
        for (0..50) |_| {
            const t0 = check.now(gpu.io);
            try gpu.d.check(gpu.d.api.cuMemcpyDtoHAsync_v2(hid.bytes.ptr, dids, n * 4, r.s.handle), "cuMemcpyDtoHAsync");
            try r.s.synchronize();
            best = @min(best, @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e6);
        }
        const mb = @as(f64, @floatFromInt(bytes)) / 1e6;
        std.debug.print("RESULT staging one decode row's 2,051 keys (K+V bf16, {d:.2} MB): one pinned H2D {d:.4} ms ({d:.1} GB/s); 1,024 block copies {d:.4} ms; gather kernel from mapped host {d:.4} ms ({d:.1} GB/s); ids D2H + stream sync {d:.4} ms\n", .{ mb, t_one, mb / t_one, t_blocks, t_gather, mb / t_gather, best });
    }
    // the indexer's raw keys in mapped host memory: upstream's pooling reads them only when a block of 4 completes
    {
        const ikc_bytes = cap * 128 * 2;
        var ikc_dev = try cuda.DeviceBuffer.alloc(gpu.d, ikc_bytes);
        defer ikc_dev.free();
        var ikc_host = try cuda.HostBuffer.allocMapped(gpu.d, ikc_bytes);
        defer ikc_host.free();
        for (ikc_host.slice(u16)) |*x| x.* = bf16(rng.floatNorm(f32));
        try ikc_dev.upload(0, ikc_host.bytes);
        const kw = try r.dev(std.mem.sliceAsBytes(r.f32s(128, 0.5, 1.5)));
        const inv = try a.alloc(f32, half);
        for (inv, 0..) |*o, i| o.* = @floatCast(std.math.pow(f64, 10_000_000.0, -@as(f64, @floatFromInt(i)) / half));
        const dinv = try r.dev(std.mem.sliceAsBytes(inv));
        const Pool = struct {
            g: *const flash.Glue,
            s: cuda.Stream,
            ikc: u64,
            pooled: u64,
            pos: u64,
            kw: u64,
            inv: u64,
            rows: usize,
            fn go(p: *const @This()) anyerror!void {
                try p.g.qsaPool(p.s, p.ikc, p.pooled, p.pos, p.kw, p.inv, 1e-6, p.rows);
            }
        };
        for ([_]usize{ 1, 256 }) |rows| {
            const pos0: i32 = @intCast(cap - 256);
            const dpos = try r.dev(std.mem.sliceAsBytes(&[_]i32{ if (rows == 1) pos0 + 3 else pos0, 0, 0, 0 }));
            const pd = try r.out((cap / 4 + 1) * 128 * 2);
            const ph = try r.out((cap / 4 + 1) * 128 * 2);
            const on_d: Pool = .{ .g = &g, .s = r.s, .ikc = ikc_dev.ptr, .pooled = pd, .pos = dpos, .kw = kw, .inv = dinv, .rows = rows };
            const on_h: Pool = .{ .g = &g, .s = r.s, .ikc = try ikc_host.device(), .pooled = ph, .pos = dpos, .kw = kw, .inv = dinv, .rows = rows };
            b.reps = 20;
            const td = try b.time(&on_d, Pool.go);
            const th = try b.time(&on_h, Pool.go);
            const n = (cap / 4 + 1) * 128;
            const same = std.mem.eql(u16, try r.back(u16, pd, n), try r.back(u16, ph, n));
            std.debug.print("RESULT indexer pooling of {d} row(s) at the cache end: raw keys on the device {d:.4} ms, in mapped host memory {d:.4} ms; pooled rows {s}\n", .{ rows, td, th, if (same) "bit-equal" else "DIFFER" });
            try check.expect(same, "pooling from mapped host keys differs", .{});
        }
    }
    // a prompt-cache snapshot (cuda_snap.zig copies with cuMemcpyDtoDAsync) into mapped pinned host memory and back
    {
        const bytes: usize = 256 << 20;
        var dev = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
        defer dev.free();
        var host = try cuda.HostBuffer.allocMapped(gpu.d, bytes);
        defer host.free();
        const hp = try host.device();
        const Dtod = struct {
            s: cuda.Stream,
            dst: u64,
            src: u64,
            n: usize,
            fn go(c: *const @This()) anyerror!void {
                try c.s.d.check(c.s.d.api.cuMemcpyDtoDAsync_v2(c.dst, c.src, c.n, c.s.handle), "cuMemcpyDtoDAsync");
            }
        };
        b.reps = 3;
        const save: Dtod = .{ .s = r.s, .dst = hp, .src = dev.ptr, .n = bytes };
        const t_save = try b.time(&save, Dtod.go);
        const back: Dtod = .{ .s = r.s, .dst = dev.ptr, .src = hp, .n = bytes };
        const t_back = try b.time(&back, Dtod.go);
        const gb = @as(f64, @floatFromInt(bytes)) / 1e9;
        std.debug.print("RESULT a 256 MiB snapshot through cuMemcpyDtoDAsync to/from mapped pinned host memory: save {d:.2} ms ({d:.1} GB/s), restore {d:.2} ms ({d:.1} GB/s)\n", .{ t_save, gb / t_save * 1e3, t_back, gb / t_back * 1e3 });
    }
    check.pass("exl3-kv bench: timings above (an idle card under gpurun)", .{});
}

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    const mode = rest[0];
    if (std.mem.eql(u8, mode, "kernel")) return kernelMode(gpu);
    if (std.mem.eql(u8, mode, "full")) return @import("exl3_kv_full.zig").run(gpu, rest[1..]);
    if (std.mem.eql(u8, mode, "real")) return @import("exl3_kv_real.zig").run(gpu, rest[1..]);
    if (std.mem.eql(u8, mode, "bench")) return benchMode(gpu, if (rest.len > 1) try std.fmt.parseInt(usize, rest[1], 10) else 200_016);
    return error.UnknownMode;
}
