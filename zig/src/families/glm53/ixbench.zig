//! tf-glm53-ixbench [rows] [kv16 0|1] [n,n,...]: the length-dependent part of one full-indexer GLM-5.3 layer at real
//! shapes (32 x 128 indexer, 576-wide latent rows, index_topk 2,048), synthetic data, no weights.
//! For each key count n it times the old indexer scores (g53_idx_scores), the new ones (g53_idx_scores2), top-k
//! (g53_topk) and the sparse latent attention (g53_attn_block + g53_attn_join) for `rows` rows ending at position n,
//! and checks that the new scores are bit-identical to the old and that top-k picks the same keys from both.
const std = @import("std");
const mtl = @import("metal");
const m = @import("model.zig");
const fwd = @import("forward.zig");

const source = @embedFile("kernels.metal");

const IArgs = extern struct { p0: i32, wscale: f32, cap: i32, keys: i32 };
const TArgs = extern struct { p0: i32, top: i32, cap: i32 };
const AArgs = extern struct { p0: i32, sparse: i32, scale: f32, heads: i32, keys: i32, qp_stride: i32 };

const Rng = struct {
    s: u64,
    fn next(r: *Rng) u64 {
        r.s ^= r.s << 13;
        r.s ^= r.s >> 7;
        r.s ^= r.s << 17;
        return r.s;
    }
    fn unit(r: *Rng) f32 { // [-1, 1)
        return @as(f32, @floatFromInt(r.next() >> 40)) / @as(f32, 1 << 23) - 1.0;
    }
};

fn newBuf(device: mtl.Device, len: usize) !mtl.Buffer {
    const b = try device.buffer(@max(len, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    @memset(b.contents()[0..len], 0);
    return b;
}

fn fillF32(b: mtl.Buffer, n: usize, r: *Rng, scale: f32) void {
    for (b.slice(f32, n)) |*x| x.* = r.unit() * scale;
}

fn fillKV(b: mtl.Buffer, n: usize, r: *Rng, half: bool) void {
    if (half) {
        for (b.slice(f16, n)) |*x| x.* = @floatCast(r.unit());
    } else for (b.slice(f32, n)) |*x| x.* = r.unit();
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    const rows: usize = if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else 256;
    const kv16 = if (args.len > 2) args[2][0] != '0' else true;
    var ns: std.ArrayList(usize) = .empty;
    if (args.len > 3) {
        var it = std.mem.splitScalar(u8, args[3], ',');
        while (it.next()) |t| try ns.append(a, try std.fmt.parseInt(usize, t, 10));
    } else for ([_]usize{ 65536, 131072, 262144, 393216, 524288 }) |n| try ns.append(a, n);
    const reps: usize = if (std.c.getenv("IX_REPS")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 3;
    const heads: usize = 16; // one rank's share of the 64 MLA heads on 4 Macs
    var maxn: usize = 0;
    for (ns.items) |n| maxn = @max(maxn, n);
    const cap = maxn + 64;

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    const src = if (kv16) "#define KVT half\n" ++ source else source;
    const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const p_old = try mtl.Pipeline.init(device, lib, "g53_idx_scores", false);
    const p_new = try mtl.Pipeline.init(device, lib, "g53_idx_scores2", false);
    const p_v3 = try mtl.Pipeline.init(device, lib, "g53_idx_scores3", false);
    const p_topk = try mtl.Pipeline.init(device, lib, "g53_topk", false);
    const p_attn = try mtl.Pipeline.init(device, lib, "g53_attn_block", false);
    const p_join = try mtl.Pipeline.init(device, lib, "g53_attn_join", false);
    const tkp: fwd.TkPipes = .{ .init = try mtl.Pipeline.init(device, lib, "g53_tk_init", false), .hist = try mtl.Pipeline.init(device, lib, "g53_tk_hist", false), .digit = try mtl.Pipeline.init(device, lib, "g53_tk_digit", false), .count = try mtl.Pipeline.init(device, lib, "g53_tk_count", false), .scan = try mtl.Pipeline.init(device, lib, "g53_tk_scan", false), .write = try mtl.Pipeline.init(device, lib, "g53_tk_write", false) };
    const ties = std.c.getenv("IX_TIES") != null;
    const kvb: usize = if (kv16) 2 else 4;
    std.debug.print("ixbench: {s}, rows {d}, KV {s}, cap {d}, reps {d} (min of reps; GPU time)\n", .{ device.name(), rows, if (kv16) "fp16" else "fp32", cap, reps });

    var rng: Rng = .{ .s = 0x9E3779B97F4A7C15 };
    const iq = try newBuf(device, rows * m.IH * m.ID * 4);
    const iw = try newBuf(device, rows * m.IH * 4);
    const ic = try newBuf(device, cap * m.ID * kvb);
    const sc_a = try newBuf(device, rows * cap * 4);
    const sc_b = try newBuf(device, rows * cap * 4);
    const sc_c = try newBuf(device, rows * cap * 4);
    const idx_a = try newBuf(device, rows * m.KEYS * 4);
    const idx_b = try newBuf(device, rows * m.KEYS * 4);
    const kvc = try newBuf(device, cap * m.CROW * kvb);
    const ql = try newBuf(device, rows * heads * m.KVL * 4);
    const qp = try newBuf(device, rows * heads * m.QH * 4);
    const po = try newBuf(device, rows * 64 * heads * m.KVL * 4 + 4096);
    const pm = try newBuf(device, rows * 64 * heads * 4 + 256);
    const pl = try newBuf(device, rows * 64 * heads * 4 + 256);
    const att = try newBuf(device, rows * heads * m.KVL * 4);
    const idx_c = try newBuf(device, rows * m.KEYS * 4);
    const tkb: fwd.TkBufs = .{ .scores = .{ .buf = sc_a }, .idx = .{ .buf = idx_c }, .state = .{ .buf = try newBuf(device, fwd.tk_rows_max * 16) }, .hist = .{ .buf = try newBuf(device, fwd.tk_rows_max * 256 * 4) }, .cnt = .{ .buf = try newBuf(device, fwd.tk_rows_max * 1024 * 8) }, .base = .{ .buf = try newBuf(device, fwd.tk_rows_max * 1024 * 8) } };
    fillF32(iq, rows * m.IH * m.ID, &rng, 1.0);
    fillF32(iw, rows * m.IH, &rng, 1.0);
    fillKV(ic, cap * m.ID, &rng, kv16);
    fillKV(kvc, cap * m.CROW, &rng, kv16);
    fillF32(ql, rows * heads * m.KVL, &rng, 0.1);
    fillF32(qp, rows * heads * m.QH, &rng, 0.1);
    const wscale: f32 = 1.0 / (@sqrt(@as(f32, m.IH)) * @sqrt(@as(f32, m.ID)));

    std.debug.print("| n keys | rows | old scores ms | new scores ms | x | top-k ms | attention ms | old ns/(row*key) | new ns/(row*key) | scores bit-equal | top-k equal | few-rows top-k ms | x | few == top-k |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n", .{});
    var all_ok = true;
    for (ns.items) |n| {
        const p0: usize = n - rows; // rows at positions p0 .. n-1: the last row has n keys
        const last_n = n;
        const span = fwd.Engine.idxSpan(last_n, rows);
        var t_old: f64 = 1e9;
        var t_new: f64 = 1e9;
        var t_v3: f64 = 1e9;
        var t_topk: f64 = 1e9;
        var t_attn: f64 = 1e9;
        for (0..reps) |_| {
            // A: old scores
            {
                const cb = queue.commandBufferUnretained();
                const enc = cb.compute(.serial);
                enc.setPipeline(p_old);
                enc.setBuffer(iq, 0, 0);
                enc.setBuffer(iw, 0, 1);
                enc.setBuffer(ic, 0, 2);
                enc.setBuffer(sc_a, 0, 3);
                enc.setValue(IArgs{ .p0 = @intCast(p0), .wscale = wscale, .cap = @intCast(cap), .keys = m.KEYS }, 4);
                enc.dispatchGroups(mtl.Size.of((last_n + 7) / 8, rows, 1), mtl.Size.of(256, 1, 1));
                enc.end();
                cb.commit();
                cb.wait();
                if (cb.failure()) |msg| {
                    std.debug.print("old scores failed: {s}\n", .{msg});
                    return error.GpuFailed;
                }
                t_old = @min(t_old, cb.gpuSeconds());
            }
            // B: new scores
            {
                const cb = queue.commandBufferUnretained();
                const enc = cb.compute(.serial);
                enc.setPipeline(p_new);
                enc.setBuffer(iq, 0, 0);
                enc.setBuffer(iw, 0, 1);
                enc.setBuffer(ic, 0, 2);
                enc.setBuffer(sc_b, 0, 3);
                enc.setValue(fwd.IArgs2{ .p0 = @intCast(p0), .wscale = wscale, .cap = @intCast(cap), .keys = m.KEYS, .span = @intCast(span) }, 4);
                enc.dispatchGroups(mtl.Size.of(rows, (last_n + span - 1) / span, 1), mtl.Size.of(256, 1, 1));
                enc.end();
                cb.commit();
                cb.wait();
                if (cb.failure()) |msg| {
                    std.debug.print("new scores failed: {s}\n", .{msg});
                    return error.GpuFailed;
                }
                t_new = @min(t_new, cb.gpuSeconds());
            }
            // C: two rows a threadgroup
            if (rows > 1) {
                const cb = queue.commandBufferUnretained();
                const enc = cb.compute(.serial);
                enc.setPipeline(p_v3);
                enc.setBuffer(iq, 0, 0);
                enc.setBuffer(iw, 0, 1);
                enc.setBuffer(ic, 0, 2);
                enc.setBuffer(sc_c, 0, 3);
                enc.setValue(fwd.IArgs2{ .p0 = @intCast(p0), .wscale = wscale, .cap = @intCast(cap), .keys = m.KEYS, .span = @intCast(span), .rows = @intCast(rows) }, 4);
                enc.dispatchGroups(mtl.Size.of((rows + 1) / 2, (last_n + span - 1) / span, 1), mtl.Size.of(256, 1, 1));
                enc.end();
                cb.commit();
                cb.wait();
                if (cb.failure()) |msg| {
                    std.debug.print("v3 scores failed: {s}\n", .{msg});
                    return error.GpuFailed;
                }
                t_v3 = @min(t_v3, cb.gpuSeconds());
            }
            // top-k on both score sets (timed on the old one), then the sparse attention over the picks
            inline for (.{ .{ sc_a, idx_a, true }, .{ sc_b, idx_b, false } }) |pair| {
                const cb = queue.commandBufferUnretained();
                const enc = cb.compute(.serial);
                enc.setPipeline(p_topk);
                enc.setBuffer(pair[0], 0, 0);
                enc.setBuffer(pair[1], 0, 1);
                enc.setValue(TArgs{ .p0 = @intCast(p0), .top = m.KEYS, .cap = @intCast(cap) }, 2);
                enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(1024, 1, 1));
                enc.end();
                cb.commit();
                cb.wait();
                if (pair[2]) t_topk = @min(t_topk, cb.gpuSeconds());
            }
            {
                const aa: AArgs = .{ .p0 = @intCast(p0), .sparse = 1, .scale = 1.0 / 16.0, .heads = @intCast(heads), .keys = m.KEYS, .qp_stride = m.QH };
                const cb = queue.commandBufferUnretained();
                const enc = cb.compute(.serial);
                enc.setPipeline(p_attn);
                for ([_]mtl.Buffer{ ql, qp, kvc, idx_a, po, pm, pl }, 0..) |b, bi| enc.setBuffer(b, if (bi == 1) m.NOPE * 4 else 0, bi);
                enc.setValue(aa, 7);
                enc.dispatchGroups(mtl.Size.of((@min(last_n, m.KEYS) + 31) / 32, (heads + 7) / 8, rows), mtl.Size.of(256, 1, 1));
                enc.setPipeline(p_join);
                for ([_]mtl.Buffer{ po, pm, pl, att }, 0..) |b, bi| enc.setBuffer(b, 0, bi);
                enc.setValue(aa, 4);
                enc.dispatchGroups(mtl.Size.of(heads, rows, 1), mtl.Size.of(512, 1, 1));
                enc.end();
                cb.commit();
                cb.wait();
                t_attn = @min(t_attn, cb.gpuSeconds());
            }
        }
        // few-rows top-k (decode) on the old scores: the same picks as g53_topk, timed (rows <= tk_rows_max)
        var t_few: f64 = 1e9;
        var few_ok = true;
        if (rows <= fwd.tk_rows_max) {
            if (ties) { // coarse scores: many equal keys at the threshold (the tie rule must match), both kernels on them
                for (0..rows) |r| for (sc_a.slice(f32, rows * cap)[r * cap ..][0 .. p0 + r + 1]) |*x| {
                    x.* = @round(x.* * 4.0) / 4.0;
                };
                const cb = queue.commandBufferUnretained();
                const enc = cb.compute(.serial);
                enc.setPipeline(p_topk);
                enc.setBuffer(sc_a, 0, 0);
                enc.setBuffer(idx_a, 0, 1);
                enc.setValue(TArgs{ .p0 = @intCast(p0), .top = m.KEYS, .cap = @intCast(cap) }, 2);
                enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(1024, 1, 1));
                enc.end();
                cb.commit();
                cb.wait();
            }
            for (0..reps) |_| {
                const cb = queue.commandBufferUnretained();
                const enc = cb.compute(.serial);
                fwd.encodeTopkFew(enc, tkp, tkb, @intCast(p0), rows, last_n, @intCast(cap), false);
                enc.end();
                cb.commit();
                cb.wait();
                if (cb.failure()) |msg| {
                    std.debug.print("few-rows top-k failed: {s}\n", .{msg});
                    return error.GpuFailed;
                }
                t_few = @min(t_few, cb.gpuSeconds());
            }
            few_ok = std.mem.eql(u32, idx_a.slice(u32, rows * m.KEYS), idx_c.slice(u32, rows * m.KEYS));
            if (!few_ok) {
                const x = idx_a.slice(u32, rows * m.KEYS);
                const y = idx_c.slice(u32, rows * m.KEYS);
                for (0..rows * m.KEYS) |j| if (x[j] != y[j]) {
                    std.debug.print("   few-rows top-k first difference at {d}: g53_topk {d} few {d}\n", .{ j, x[j], y[j] });
                    break;
                };
            }
        }
        // exactness: every row's scores over its own keys, bit for bit; the picked key lists equal
        var bit_ok = true;
        var first_bad: ?[2]usize = null;
        const A = sc_a.slice(u32, rows * cap);
        const B = sc_b.slice(u32, rows * cap);
        for (0..rows) |r| {
            const nr = p0 + r + 1;
            if (nr <= m.KEYS) continue;
            if (!std.mem.eql(u32, A[r * cap ..][0..nr], B[r * cap ..][0..nr])) {
                bit_ok = false;
                if (first_bad == null) {
                    for (0..nr) |t| {
                        if (A[r * cap + t] != B[r * cap + t]) {
                            first_bad = .{ r, t };
                            break;
                        }
                    }
                }
            }
        }
        const idx_ok = std.mem.eql(u32, idx_a.slice(u32, rows * m.KEYS), idx_b.slice(u32, rows * m.KEYS));
        var v3_ok = true;
        if (rows > 1) {
            const Cc = sc_c.slice(u32, rows * cap);
            for (0..rows) |r| {
                const nr = p0 + r + 1;
                if (nr <= m.KEYS) continue;
                if (!std.mem.eql(u32, A[r * cap ..][0..nr], Cc[r * cap ..][0..nr])) v3_ok = false;
            }
        }
        // the picks are sorted, distinct and in range (a sanity check on top-k itself)
        var sane = true;
        for (0..rows) |r| {
            const nr = p0 + r + 1;
            if (nr <= m.KEYS) continue;
            const l = idx_a.slice(u32, rows * m.KEYS)[r * m.KEYS ..][0..m.KEYS];
            for (l, 0..) |v, j| {
                if (v >= nr or (j > 0 and v <= l[j - 1])) sane = false;
            }
        }
        all_ok = all_ok and (ties or (bit_ok and idx_ok and v3_ok)) and sane and few_ok;
        if (rows > 1) std.debug.print("   two rows a threadgroup (g53_idx_scores3): {d:.2} ms, {d:.2}x old, {d:.2}x new, bit-equal {s}\n", .{ t_v3 * 1e3, t_old / t_v3, t_new / t_v3, if (v3_ok) "YES" else "NO" });
        const work: f64 = @floatFromInt(rows * n); // (row, key) pairs, near enough (rows << n)
        std.debug.print("| {d} | {d} | {d:.2} | {d:.2} | {d:.2} | {d:.2} | {d:.3} | {d:.3} | {d:.3} | {s} | {s}{s} | {d:.3} | {d:.2} | {s} |\n", .{ n, rows, t_old * 1e3, t_new * 1e3, t_old / t_new, t_topk * 1e3, t_attn * 1e3, t_old * 1e9 / work, t_new * 1e9 / work, if (bit_ok) "YES" else "NO", if (idx_ok) "YES" else "NO", if (sane) "" else " (top-k list NOT sorted/in range)", if (rows <= fwd.tk_rows_max) t_few * 1e3 else 0, if (rows <= fwd.tk_rows_max) t_topk / t_few else 0, if (rows > fwd.tk_rows_max) "-" else if (few_ok) (if (ties) "YES (ties)" else "YES") else "NO" });
        if (first_bad) |fb| std.debug.print("   first differing score: row {d} key {d}: old {x} new {x}\n", .{ fb[0], fb[1], A[fb[0] * cap + fb[1]], B[fb[0] * cap + fb[1]] });
    }
    std.debug.print("ixbench: {s}\n", .{if (all_ok) "ALL EXACT (scores bit-identical, top-k identical)" else "MISMATCH"});
    if (!all_ok) std.process.exit(1);
}
