//! Quantized KV cache (q8 / q4): attention vs fp64 reference, row invariance, timing. usage: xpu-qwen_kvq-test

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const al = @import("qwen_xpu").attn_long;

const gpa = std.heap.page_allocator;
const max_len: usize = 131072;
const kv_dim: usize = 1024;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn bf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}

fn half(v: u16) f64 {
    return @as(f16, @bitCast(v));
}

fn bfBits(f: f32) u16 {
    const u: u32 = @bitCast(f);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// Dequantized values (fp64) of one record of mode m into out[256].
fn dequant(m: al.Mode, rec: []const u8, out: []f64) void {
    const rb = al.recBytes(m);
    _ = rb;
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
            const j = i % 32; // dword m = j / 8 of the block, nibble p (dim 2p) or p + 4 (dim 2p + 1)
            const w = std.mem.readInt(u32, rec[b * 16 + (j / 8) * 4 ..][0..4], .little);
            const p = (j % 8) / 2;
            const sh: u5 = @intCast(4 * p + (if (j % 2 == 1) @as(usize, 16) else 0));
            const n: f64 = @floatFromInt((w >> sh) & 0xF);
            const d = half(std.mem.readInt(u16, rec[128 + 2 * b ..][0..2], .little));
            out[i] = (n - 8.0) * d;
        },
    }
}

/// fp64 attention of query rows qrow [24][256] over keys [len][4][256]: o[24][256] after the output gate (qg).
fn refAttn(len: usize, q: []const f64, k: []const f64, v: []const f64, o: []f64) void {
    var sc = gpa.alloc(f64, len) catch unreachable;
    defer gpa.free(sc);
    for (0..24) |h| {
        const hk = h / 6;
        var mx: f64 = -1e300;
        for (0..len) |t| {
            var s: f64 = 0;
            for (0..256) |d| s += q[h * 256 + d] * k[(t * 4 + hk) * 256 + d];
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
            for (0..len) |t| a += sc[t] * v[(t * 4 + hk) * 256 + d];
            o[h * 256 + d] = a / den;
        }
    }
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    var r = try rt.open();
    defer r.deinit();
    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();
    const host = try gpa.alloc(u16, max_len * kv_dim);
    defer gpa.free(host);
    // K with a few outlier channels (as real keys have), V plain; both bf16
    const kh = try gpa.alloc(u16, max_len * kv_dim);
    const vh = try gpa.alloc(u16, max_len * kv_dim);
    for (0..max_len * 4) |i| for (0..256) |d| {
        const outl: f32 = if (d % 64 == 7) 6.0 else 1.0;
        kh[i * 256 + d] = bfBits((rnd.floatNorm(f32)) * outl * 0.7);
        vh[i * 256 + d] = bfBits(rnd.floatNorm(f32) * 0.5);
    };
    const kbf = try r.alloc(max_len * kv_dim * 2);
    const vbf = try r.alloc(max_len * kv_dim * 2);
    try r.upload(kbf, std.mem.sliceAsBytes(kh));
    try r.upload(vbf, std.mem.sliceAsBytes(vh));
    const qh = try gpa.alloc(u16, 24 * 512);
    for (qh) |*x| x.* = bfBits(rnd.floatNorm(f32));
    const qg = try r.alloc(24 * 512 * 2);
    try r.upload(qg, std.mem.sliceAsBytes(qh));
    const qs = try gpa.alloc(u16, 24 * 256);
    for (0..24) |i| @memcpy(qs[i * 256 ..][0..256], qh[i * 512 ..][0..256]);
    const q = try r.alloc(24 * 256 * 2);
    try r.upload(q, std.mem.sliceAsBytes(qs));
    const po = try r.alloc(4 * 128 * 24 * 256 * 4);
    const pm = try r.alloc(4 * 128 * 24 * 4);
    const pl = try r.alloc(4 * 128 * 24 * 4);
    const att = try r.alloc(4 * 24 * 256 * 2);
    var kvq = try al.Kvq.init(&r);
    try r.sync();
    const nh = try gpa.alloc(u16, 24 * 256);
    const bfo = try gpa.alloc(u16, 24 * 256);
    // gate of the output: out = bf16(o / l) * sigmoid(gate); divide the gate out on the host to compare the pre-gate o
    var gates: [24 * 256]f64 = undefined;
    for (0..24) |h| for (0..256) |d| {
        const g = bf(qh[h * 512 + 256 + d]);
        gates[h * 256 + d] = 1.0 / (1.0 + @exp(-g));
    };
    var bf16_out: [24 * 256]f64 = undefined;
    for ([_]al.Mode{ .bf16, .q8, .q4 }) |mode| {
        const rec = al.recBytes(mode);
        var lg = try al.Long.init(&r, mode);
        const kc = if (mode == .bf16) kbf else try r.alloc(max_len * 4 * rec);
        const vc = if (mode == .bf16) vbf else try r.alloc(max_len * 4 * rec);
        if (mode != .bf16) {
            var off: usize = 0;
            while (off < max_len) { // 16384 rows a launch (the source buffers are the whole bf16 cache)
                const n: u32 = @intCast(@min(max_len - off, 16384));
                const sk: rt.Buffer = .{ .rt = kbf.rt, .ptr = @ptrFromInt(@intFromPtr(kbf.ptr.?) + off * kv_dim * 2), .len = n * kv_dim * 2 };
                const sv: rt.Buffer = .{ .rt = vbf.rt, .ptr = @ptrFromInt(@intFromPtr(vbf.ptr.?) + off * kv_dim * 2), .len = n * kv_dim * 2 };
                try kvq.append(mode, sk, kc, @intCast(off), n);
                try kvq.append(mode, sv, vc, @intCast(off), n);
                off += n;
            }
            try r.sync();
        }
        // accuracy at a few lengths
        for ([_]usize{ 40, 1000, 3000 }) |len| {
            try lg.run(q, kc, vc, po, pm, pl, qg, att, @intCast(len), 1);
            try r.sync();
            try r.download(std.mem.sliceAsBytes(nh), att);
            try r.sync();
            // host references: over the stored cache (dequantized) and over the original bf16
            const kd = try gpa.alloc(f64, len * 1024);
            const vd = try gpa.alloc(f64, len * 1024);
            const ko = try gpa.alloc(f64, len * 1024);
            const vo = try gpa.alloc(f64, len * 1024);
            defer for ([_][]f64{ kd, vd, ko, vo }) |b| gpa.free(b);
            if (mode != .bf16) {
                const kb = try gpa.alloc(u8, len * 4 * rec);
                const vb = try gpa.alloc(u8, len * 4 * rec);
                defer gpa.free(kb);
                defer gpa.free(vb);
                const ksub: rt.Buffer = .{ .rt = kc.rt, .ptr = kc.ptr, .len = len * 4 * rec };
                const vsub: rt.Buffer = .{ .rt = vc.rt, .ptr = vc.ptr, .len = len * 4 * rec };
                try r.download(kb, ksub);
                try r.download(vb, vsub);
                try r.sync();
                for (0..len * 4) |i| {
                    dequant(mode, kb[i * rec ..][0..rec], kd[i * 256 ..][0..256]);
                    dequant(mode, vb[i * rec ..][0..rec], vd[i * 256 ..][0..256]);
                }
            }
            for (0..len * 1024) |i| {
                ko[i] = bf(kh[i]);
                vo[i] = bf(vh[i]);
            }
            if (mode == .bf16) {
                @memcpy(kd, ko);
                @memcpy(vd, vo);
            }
            const qd = try gpa.alloc(f64, 24 * 256);
            defer gpa.free(qd);
            for (0..24 * 256) |i| qd[i] = bf(qs[i]);
            var o_dq: [24 * 256]f64 = undefined;
            var o_orig: [24 * 256]f64 = undefined;
            refAttn(len, qd, kd, vd, &o_dq);
            refAttn(len, qd, ko, vo, &o_orig);
            var e_kernel: f64 = 0;
            var e_quant: f64 = 0;
            var mag: f64 = 0;
            for (0..24 * 256) |i| {
                const kern = bf(nh[i]) / gates[i]; // undo the gate (bf16 rounding of the product included in the tolerance)
                e_kernel = @max(e_kernel, @abs(kern - o_dq[i]));
                e_quant = @max(e_quant, @abs(o_dq[i] - o_orig[i]));
                mag = @max(mag, @abs(o_orig[i]));
            }
            std.debug.print("{s} len {d:>5}: kernel vs fp64 over the stored cache: max abs {e:.2}; stored cache vs original bf16 (quantization error): max abs {e:.2} (max |o| {d:.3})\n", .{ @tagName(mode), len, e_kernel, e_quant, mag });
            if (mode == .bf16 and len == 1000) for (0..24 * 256) |i| {
                bf16_out[i] = bf(nh[i]);
            };
            _ = bfo;
        }
        // timing over the cache bytes
        for ([_]u32{ 16384, 65536, 131072 }) |len| {
            var best: u64 = std.math.maxInt(u64);
            for (0..5) |_| {
                const t0 = nowNs();
                for (0..8) |_| try lg.run(q, kc, vc, po, pm, pl, qg, att, len, 1);
                try r.sync();
                best = @min(best, nowNs() - t0);
            }
            const us = @as(f64, @floatFromInt(best)) / 1e3 / 8.0;
            const bytes = @as(f64, @floatFromInt(len)) * 4 * @as(f64, @floatFromInt(rec)) * 2;
            std.debug.print("{s} {d:>6} keys: {d:>7.1} us per layer, {d:>5.0} GB/s of cache bytes, {d:.2} ms per token over 16 layers\n", .{ @tagName(mode), len, us, bytes / us / 1e3, us * 16 / 1e3 });
        }
        // rows: 4 rows at 100000 keys equal four single launches
        {
            var one: [4][24 * 256]u16 = undefined;
            for (0..4) |z| {
                try lg.run(q, kc, vc, po, pm, pl, qg, att, @intCast(100000 + z), 1);
                try r.sync();
                try r.download(std.mem.sliceAsBytes(&one[z]), .{ .rt = att.rt, .ptr = att.ptr, .len = 24 * 256 * 2 });
                try r.sync();
            }
            // 4-row launch: rows share q here (one q row), so row z must equal the single launch at len 100000 + z
            const q4 = try r.alloc(4 * 24 * 256 * 2);
            const qg4 = try r.alloc(4 * 24 * 512 * 2);
            for (0..4) |z| {
                try r.upload(.{ .rt = q4.rt, .ptr = @ptrFromInt(@intFromPtr(q4.ptr.?) + z * 24 * 256 * 2), .len = 24 * 256 * 2 }, std.mem.sliceAsBytes(qs));
                try r.upload(.{ .rt = qg4.rt, .ptr = @ptrFromInt(@intFromPtr(qg4.ptr.?) + z * 24 * 512 * 2), .len = 24 * 512 * 2 }, std.mem.sliceAsBytes(qh));
            }
            try r.sync();
            try lg.run(q4, kc, vc, po, pm, pl, qg4, att, 100000, 4);
            try r.sync();
            var four: [4 * 24 * 256]u16 = undefined;
            try r.download(std.mem.sliceAsBytes(&four), att);
            try r.sync();
            var same = true;
            for (0..4) |z| if (!std.mem.eql(u16, &one[z], four[z * 24 * 256 ..][0 .. 24 * 256])) {
                same = false;
            };
            std.debug.print("{s}: rows 0..3 of a 4-row launch equal the single-row launches at 100000..100003 keys: {s}\n", .{ @tagName(mode), if (same) "yes" else "NO" });
            if (!same) return error.RowInvariance;
        }
    }
}
