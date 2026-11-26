//! Synthetic Metal GPU parity tests for Qwen27 TFQ3 (3/4/5-bit decode, split-K, prefill) and BF16 kernels.
const std = @import("std");
const mtl = @import("metal");
const qwen27 = @import("qwen27");
const tfq3 = qwen27.tfq3;
const dispatch = qwen27.tfq3_dispatch;
const glue_gpu = qwen27.glue_gpu;
const projection = qwen27.projection;

const Q27PwV3Tables = extern struct {
    magic: u32,
    version: u32,
    a3: f32,
    b3: f32,
    a4: f32,
    b4: f32,
    a5: f32,
    b5: f32,
    pad0: [8]u32,
    lut3: [256]u32,
    lut4: [1024]u32,
    lut5: [1024]u32,
    c64_3: [64][4]u16,
    c64_4: [64][4]u16,
    c64_5: [64][4]u16,
};

fn f32ToBf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    const lsb = (bits >> 16) & 1;
    return @truncate((bits +% 0x7fff +% lsb) >> 16);
}

fn bf16ToF32(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn f32ToF16Bits(x: f32) u16 {
    const h: f16 = @floatCast(x);
    return @bitCast(h);
}

fn f16BitsToF32(b: u16) f32 {
    const h: f16 = @bitCast(b);
    return @floatCast(h);
}

fn packHalf2(a: f32, b: f32) u32 {
    return @as(u32, f32ToF16Bits(a)) | (@as(u32, f32ToF16Bits(b)) << 16);
}

fn unpackHalf2(u: u32) [2]f32 {
    return .{ f16BitsToF32(@truncate(u)), f16BitsToF32(@truncate(u >> 16)) };
}

fn pwSgn(seed: u32, i: u32) f64 {
    const h = (seed ^ i) *% 2654435761;
    return if (((h >> 31) & 1) != 0) -1.0 else 1.0;
}

fn fwht128(v: *[128]f64) void {
    var step: usize = 1;
    while (step < 128) : (step <<= 1) {
        var i: usize = 0;
        while (i < 128) : (i += step * 2) {
            for (0..step) |j| {
                const a = v[i + j];
                const b = v[i + j + step];
                v[i + j] = a + b;
                v[i + j + step] = a - b;
            }
        }
    }
    const s = 1.0 / @sqrt(128.0);
    for (v) |*e| e.* *= s;
}

fn makeTables(rng: std.Random) Q27PwV3Tables {
    var t: Q27PwV3Tables = std.mem.zeroes(Q27PwV3Tables);
    t.magic = tfq3.check.magic_v3_tables;
    t.version = tfq3.check.format_version;
    for (&t.lut3) |*e| e.* = packHalf2((rng.float(f32) - 0.5) * 2.0, (rng.float(f32) - 0.5) * 2.0);
    for (&t.lut4) |*e| e.* = packHalf2((rng.float(f32) - 0.5) * 2.0, (rng.float(f32) - 0.5) * 2.0);
    for (&t.lut5) |*e| e.* = packHalf2((rng.float(f32) - 0.5) * 2.0, (rng.float(f32) - 0.5) * 2.0);
    for (0..64) |i| {
        const s0 = 0.8 + rng.float(f32) * 0.4;
        const s1 = 0.8 + rng.float(f32) * 0.4;
        const b0 = (rng.float(f32) - 0.5) * 0.2;
        const b1 = (rng.float(f32) - 0.5) * 0.2;
        const row = [4]u16{ f32ToF16Bits(s0), f32ToF16Bits(s1), f32ToF16Bits(b0), f32ToF16Bits(b1) };
        t.c64_3[i] = row;
        t.c64_4[i] = row;
        t.c64_5[i] = row;
    }
    return t;
}

fn decodeQ8(bits: u8, st: *u32, c0: u32, c1: u32, t: *const Q27PwV3Tables) [8]f64 {
    const s0 = st.*;
    var q: [8]f64 = undefined;
    var cs: [4]f32 = undefined;
    var cs2: [4]f32 = undefined;
    var p0: [2]f32 = undefined;
    var p1: [2]f32 = undefined;
    var p2: [2]f32 = undefined;
    var p3: [2]f32 = undefined;
    if (bits == 4) {
        const wc = c0;
        const h0 = s0 *% 0x9E3779B1;
        const h1 = ((s0 << 8) | (wc >> 24)) *% 0x9E3779B1;
        const h2 = ((s0 << 16) | (wc >> 16)) *% 0x9E3779B1;
        const h3 = ((s0 << 24) | (wc >> 8)) *% 0x9E3779B1;
        st.* = wc;
        for (0..4) |j| {
            cs[j] = f16BitsToF32(t.c64_4[(h0 >> 16) & 63][j]);
            cs2[j] = f16BitsToF32(t.c64_4[(h2 >> 16) & 63][j]);
        }
        p0 = unpackHalf2(t.lut4[(((wc >> 24) << 2) ^ (h0 >> 22)) & 1023]);
        p1 = unpackHalf2(t.lut4[((((wc >> 16) & 255) << 2) ^ (h1 >> 22)) & 1023]);
        p2 = unpackHalf2(t.lut4[((((wc >> 8) & 255) << 2) ^ (h2 >> 22)) & 1023]);
        p3 = unpackHalf2(t.lut4[(((wc & 255) << 2) ^ (h3 >> 22)) & 1023]);
    } else if (bits == 3) {
        const m24 = c0;
        const h0 = s0 *% 0x9E3779B1;
        const h1 = ((s0 << 6) | (m24 >> 18)) *% 0x9E3779B1;
        const h2 = ((s0 << 12) | (m24 >> 12)) *% 0x9E3779B1;
        const h3 = ((s0 << 18) | (m24 >> 6)) *% 0x9E3779B1;
        st.* = (s0 << 24) | m24;
        for (0..4) |j| {
            cs[j] = f16BitsToF32(t.c64_3[(h0 >> 16) & 63][j]);
            cs2[j] = f16BitsToF32(t.c64_3[(h2 >> 16) & 63][j]);
        }
        p0 = unpackHalf2(t.lut3[(((m24 >> 16) & 0xFC) ^ (h0 >> 24)) & 255]);
        p1 = unpackHalf2(t.lut3[(((m24 >> 10) & 0xFC) ^ (h1 >> 24)) & 255]);
        p2 = unpackHalf2(t.lut3[(((m24 >> 4) & 0xFC) ^ (h2 >> 24)) & 255]);
        p3 = unpackHalf2(t.lut3[(((m24 << 2) & 0xFC) ^ (h3 >> 24)) & 255]);
    } else {
        const hi8 = c0;
        const lo32 = c1;
        const b0 = ((hi8 << 2) | (lo32 >> 30)) & 1023;
        const b1 = (lo32 >> 20) & 1023;
        const b2 = (lo32 >> 10) & 1023;
        const b3 = lo32 & 1023;
        const h0 = s0 *% 0x9E3779B1;
        const h1 = ((s0 << 10) | b0) *% 0x9E3779B1;
        const h2 = ((s0 << 20) | (hi8 << 12) | (lo32 >> 20)) *% 0x9E3779B1;
        const h3 = ((s0 << 30) | (hi8 << 22) | (lo32 >> 10)) *% 0x9E3779B1;
        st.* = lo32;
        for (0..4) |j| {
            cs[j] = f16BitsToF32(t.c64_5[(h0 >> 16) & 63][j]);
            cs2[j] = f16BitsToF32(t.c64_5[(h2 >> 16) & 63][j]);
        }
        p0 = unpackHalf2(t.lut5[(b0 ^ (h0 >> 22)) & 1023]);
        p1 = unpackHalf2(t.lut5[(b1 ^ (h1 >> 22)) & 1023]);
        p2 = unpackHalf2(t.lut5[(b2 ^ (h2 >> 22)) & 1023]);
        p3 = unpackHalf2(t.lut5[(b3 ^ (h3 >> 22)) & 1023]);
    }
    q[0] = @as(f64, p0[0]) * cs[0] + cs[2];
    q[1] = @as(f64, p0[1]) * cs[1] + cs[3];
    q[2] = @as(f64, p1[0]) * cs[0] + cs[2];
    q[3] = @as(f64, p1[1]) * cs[1] + cs[3];
    q[4] = @as(f64, p2[0]) * cs2[0] + cs2[2];
    q[5] = @as(f64, p2[1]) * cs2[1] + cs2[3];
    q[6] = @as(f64, p3[0]) * cs2[0] + cs2[2];
    q[7] = @as(f64, p3[1]) * cs2[1] + cs2[3];
    return q;
}

/// CPU fp64 reference matching TFQ3 rotated decode + inverse FWHT-128 + `sr_split` sign unflip.
fn cpuTfq3Ref(
    a: std.mem.Allocator,
    bits: u8,
    n: usize,
    k: usize,
    sr_split: usize,
    words: []const u32,
    scales: []const u16,
    t: *const Q27PwV3Tables,
    x_bf16: []const u16,
    rows: usize,
) ![]f64 {
    const n_bc = k / 128;
    const wpb: usize = @as(usize, bits) * 2;
    const w_rot = try a.alloc(f64, n * k);
    defer a.free(w_rot);
    @memset(w_rot, 0.0);

    for (0..n / 16) |strip| {
        const w_base = words[strip * n_bc * wpb * 32 ..];
        const s_base = scales[(strip / 8) * n_bc ..];
        for (0..32) |lane| {
            const hx: usize = (lane & 8) + ((lane & 1) << 2);
            const hy: usize = ((lane & 16) >> 2) | ((lane >> 1) & 3);
            var st: u32 = 0x13579BDF ^ (@as(u32, @intCast(lane)) *% 0x9E3779B1);
            for (0..n_bc) |bc| {
                const sc: f64 = f16BitsToF32(s_base[bc]);
                for (0..2) |h| {
                    var q_sets: [4][8]f64 = undefined;
                    if (bits == 4) {
                        const off = (bc * 2 + h) * 32 * 4 + lane * 4;
                        for (0..4) |j| q_sets[j] = decodeQ8(4, &st, w_base[off + j], 0, t);
                    } else if (bits == 3) {
                        const blk = w_base[(bc * 6 + h * 3) * 32 + lane ..];
                        const a0 = blk[0];
                        const a1 = blk[32];
                        const a2 = blk[64];
                        q_sets[0] = decodeQ8(3, &st, a0 >> 8, 0, t);
                        q_sets[1] = decodeQ8(3, &st, ((a0 << 16) | (a1 >> 16)) & 0xFFFFFF, 0, t);
                        q_sets[2] = decodeQ8(3, &st, ((a1 << 8) | (a2 >> 24)) & 0xFFFFFF, 0, t);
                        q_sets[3] = decodeQ8(3, &st, a2 & 0xFFFFFF, 0, t);
                    } else {
                        const blk = w_base[(bc * 10 + h * 5) * 32 + lane ..];
                        const w0 = blk[0];
                        const w1 = blk[32];
                        const w2 = blk[64];
                        const w3 = blk[96];
                        const w4 = blk[128];
                        q_sets[0] = decodeQ8(5, &st, w0 >> 24, (w0 << 8) | (w1 >> 24), t);
                        q_sets[1] = decodeQ8(5, &st, (w1 >> 16) & 255, (w1 << 16) | (w2 >> 16), t);
                        q_sets[2] = decodeQ8(5, &st, (w2 >> 8) & 255, (w2 << 24) | (w3 >> 8), t);
                        q_sets[3] = decodeQ8(5, &st, w3 & 255, w4, t);
                    }
                    for (0..4) |j| {
                        const cb = bc * 128 + h * 64 + j * 16 + hx;
                        const r0 = strip * 16 + hy;
                        const r1 = r0 + 8;
                        for (0..4) |e| {
                            w_rot[r0 * k + cb + e] = q_sets[j][e] * sc;
                            w_rot[r1 * k + cb + e] = q_sets[j][4 + e] * sc;
                        }
                    }
                }
            }
        }
    }

    const out = try a.alloc(f64, rows * n);
    const x_rot = try a.alloc(f64, k);
    defer a.free(x_rot);
    for (0..rows) |m| {
        for (0..n_bc) |bc| {
            var blk: [128]f64 = undefined;
            for (0..128) |i| {
                const c = bc * 128 + i;
                blk[i] = @as(f64, bf16ToF32(x_bf16[m * k + c])) * pwSgn(0x12345678, @intCast(c));
            }
            fwht128(&blk);
            for (0..128) |i| x_rot[bc * 128 + i] = blk[i];
        }
        for (0..n / 128) |br| {
            var yb: [128]f64 = undefined;
            for (0..128) |i| {
                const r = br * 128 + i;
                var acc: f64 = 0.0;
                for (0..k) |c| acc += w_rot[r * k + c] * x_rot[c];
                yb[i] = acc;
            }
            fwht128(&yb);
            const r0 = br * 128;
            const sr0 = if (sr_split != 0 and r0 >= sr_split) r0 - sr_split else r0;
            for (0..128) |i| out[m * n + r0 + i] = yb[i] * pwSgn(0x9E3779B9, @intCast(sr0 + i));
        }
    }
    return out;
}

fn runAndCompare(
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    pipes: *const dispatch.Pipes,
    bits: u8,
    n: u32,
    k: u32,
    sr_split: u32,
    ldo: u32,
    rng: std.Random,
) !void {
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const tables = makeTables(rng);
    const t_buf = try device.buffer(@sizeOf(Q27PwV3Tables), opts);
    defer t_buf.deinit();
    @memcpy(t_buf.contents()[0..@sizeOf(Q27PwV3Tables)], std.mem.asBytes(&tables));

    const w_cnt = try tfq3.check.matWords(n, k, bits);
    const s_cnt = try tfq3.check.matScales(n, k);
    const w_buf = try device.buffer(w_cnt * 4, opts);
    defer w_buf.deinit();
    const s_buf = try device.buffer(s_cnt * 2, opts);
    defer s_buf.deinit();
    const words = w_buf.slice(u32, w_cnt);
    for (words) |*w| w.* = rng.int(u32);
    const scales = s_buf.slice(u16, s_cnt);
    for (scales) |*s| s.* = f32ToF16Bits(0.02 + rng.float(f32) * 0.03);

    const max_rows: u32 = 33;
    const x_buf = try device.buffer(@as(usize, max_rows) * k * 2, opts);
    defer x_buf.deinit();
    const x_u16 = x_buf.slice(u16, @as(usize, max_rows) * k);
    for (x_u16) |*v| v.* = f32ToBf16((rng.float(f32) - 0.5) * 0.6);

    const ref = try cpuTfq3Ref(a, bits, n, k, sr_split, words, scales, &tables, x_u16, max_rows);
    defer a.free(ref);

    const t_handle: projection.Tfq3 = .{
        .words = .{ .buffer = w_buf },
        .scales = .{ .buffer = s_buf },
        .tables = .{ .buffer = t_buf },
        .sr_split = sr_split,
    };

    const y_buf = try device.buffer(@as(usize, max_rows) * ldo * 2, opts);
    defer y_buf.deinit();
    const row0_b1 = try a.alloc(u16, n);
    defer a.free(row0_b1);
    const prefill_first = try a.alloc(u16, @as(usize, max_rows) * n);
    defer a.free(prefill_first);

    for ([_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 33, 33 }) |rows| {
        const second_prefill = (rows == 33 and prefill_first[0] != 0xFFFF);
        @memset(y_buf.slice(u16, @as(usize, max_rows) * ldo), 0xBEEF);
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        try dispatch.encodeTfq3(pipes, e, t_handle, bits, n, k, .{ .buffer = x_buf }, k, .{ .buffer = y_buf }, ldo, rows);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("GPU failure: {s}\n", .{msg});
            return error.GpuFailure;
        }
        const got = y_buf.slice(u16, @as(usize, max_rows) * ldo);
        var max_err: f64 = 0.0;
        var worst_m: usize = 0;
        var worst_c: usize = 0;
        var worst_got: f64 = 0.0;
        var worst_ref: f64 = 0.0;
        for (0..rows) |m| {
            for (0..n) |c| {
                const gv: f64 = bf16ToF32(got[m * ldo + c]);
                const diff = @abs(gv - ref[m * n + c]);
                if (diff > max_err) {
                    max_err = diff;
                    worst_m = m;
                    worst_c = c;
                    worst_got = gv;
                    worst_ref = ref[m * n + c];
                }
            }
            if (ldo > n) {
                for (n..ldo) |c| if (got[m * ldo + c] != 0xBEEF) return error.StrideClobber;
            }
        }
        if (max_err > 0.025) {
            std.debug.print("TFQ3 {d}b {d}x{d} rows={d} max_err={d:.5} at m={d} c={d} got={d:.5} ref={d:.5}\n", .{ bits, n, k, rows, max_err, worst_m, worst_c, worst_got, worst_ref });
            return error.ParityMismatch;
        }
        if (rows == 1) {
            @memcpy(row0_b1, got[0..n]);
            prefill_first[0] = 0xFFFF;
        } else if (rows <= 16) {
            if (!std.mem.eql(u16, row0_b1, got[0..n])) return error.DecodeBatchParity;
        } else if (!second_prefill) {
            for (0..rows) |m| @memcpy(prefill_first[m * n ..][0..n], got[m * ldo ..][0..n]);
        } else {
            for (0..rows) |m| if (!std.mem.eql(u16, prefill_first[m * n ..][0..n], got[m * ldo ..][0..n])) return error.PrefillRepeatParity;
        }
    }
}

fn checkBf16Ops(device: mtl.Device, queue: mtl.Queue, pipes: *const dispatch.Pipes, rng: std.Random) !void {
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const glue = try glue_gpu.Kernels.init(device);
    defer glue.deinit();

    const vocab: u32 = 16;
    const width: u32 = 256;
    const rows: u32 = 5;
    const table_buf = try device.buffer(@as(usize, vocab) * width * 2, opts);
    defer table_buf.deinit();
    const table = table_buf.slice(u16, @as(usize, vocab) * width);
    for (table) |*v| v.* = f32ToBf16((rng.float(f32) - 0.5) * 4.0);
    const ids_buf = try device.buffer(rows * 4, opts);
    defer ids_buf.deinit();
    const ids = ids_buf.slice(u32, rows);
    for (ids, [_]u32{ 3, 0, 15, 7, 3 }) |*d, v| d.* = v;
    const emb_out = try device.buffer(@as(usize, rows) * width * 2, opts);
    defer emb_out.deinit();
    const emb_t = qwen27.gpu_weights.Tensor{ .buffer = table_buf, .offset = 0, .bytes = table_buf.length(), .dtype = .bf16, .rank = 2, .shape = .{ vocab, width, 0, 0, 0 } };
    const emb_triple: projection.Triple = .{ .weight = emb_t, .scales = emb_t, .biases = emb_t, .bits = 16, .group = 0 };
    {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        try glue.embeddingShape(e, width, vocab, emb_triple, .{ .buffer = ids_buf }, .{ .buffer = emb_out }, rows);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailure;
    }
    const got_emb = emb_out.slice(u16, @as(usize, rows) * width);
    for (0..rows) |r| {
        const want = table[ids[r] * width ..][0..width];
        if (!std.mem.eql(u16, want, got_emb[r * width ..][0..width])) return error.EmbedBf16Mismatch;
    }

    const n: u32 = 96;
    const k: u32 = 512;
    const ldo: u32 = 128;
    const m_rows: u32 = 7;
    const w_buf = try device.buffer(@as(usize, n) * k * 2, opts);
    defer w_buf.deinit();
    const x_buf = try device.buffer(@as(usize, m_rows) * k * 2, opts);
    defer x_buf.deinit();
    const y1_buf = try device.buffer(@as(usize, ldo) * 2, opts);
    defer y1_buf.deinit();
    const ym_buf = try device.buffer(@as(usize, m_rows) * ldo * 2, opts);
    defer ym_buf.deinit();
    const w_u16 = w_buf.slice(u16, @as(usize, n) * k);
    const x_u16 = x_buf.slice(u16, @as(usize, m_rows) * k);
    for (w_u16) |*v| v.* = f32ToBf16((rng.float(f32) - 0.5) * 0.4);
    for (x_u16) |*v| v.* = f32ToBf16((rng.float(f32) - 0.5) * 0.4);
    @memset(y1_buf.slice(u16, ldo), 0xCAFE);
    @memset(ym_buf.slice(u16, @as(usize, m_rows) * ldo), 0xCAFE);

    {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        try dispatch.encodeBf16(pipes, e, .{ .buffer = w_buf }, n, k, .{ .buffer = x_buf }, k, .{ .buffer = y1_buf }, ldo, 1);
        try dispatch.encodeBf16(pipes, e, .{ .buffer = w_buf }, n, k, .{ .buffer = x_buf }, k, .{ .buffer = ym_buf }, ldo, m_rows);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailure;
    }
    const y1 = y1_buf.slice(u16, ldo);
    const ym = ym_buf.slice(u16, @as(usize, m_rows) * ldo);
    if (!std.mem.eql(u16, y1[0..n], ym[0..n])) return error.Bf16MvMmParity;
    for (0..m_rows) |m| {
        for (0..n) |r| {
            var acc: f64 = 0.0;
            for (0..k) |c| acc += @as(f64, bf16ToF32(w_u16[r * k + c])) * bf16ToF32(x_u16[m * k + c]);
            if (@abs(@as(f64, bf16ToF32(ym[m * ldo + r])) - acc) > 0.02) return error.Bf16ParityMismatch;
        }
        for (n..ldo) |c| if (ym[m * ldo + c] != 0xCAFE) return error.StrideClobber;
    }
}

fn checkShardedStore(a: std.mem.Allocator, io: std.Io, device: mtl.Device, queue: mtl.Queue, pipes: *const dispatch.Pipes, rng: std.Random) !void {
    const dir = try std.fmt.allocPrint(a, ".tf-qwen27-tfq3-check-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer a.free(dir);
    try std.Io.Dir.cwd().createDirPath(io, dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};
    var tmp_dir = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer tmp_dir.close(io);

    var tables = makeTables(rng);
    var tbl_page: [tfq3.check.section_align]u8 = @splat(0);
    @memcpy(tbl_page[0..@sizeOf(Q27PwV3Tables)], std.mem.asBytes(&tables));
    try tmp_dir.writeFile(io, .{ .sub_path = tfq3.check.tables_file, .data = &tbl_page });

    const n: u32 = 128;
    const k: u32 = 256;
    const bits: u32 = 3;
    const w_one = try tfq3.check.matWords(n, k, bits);
    const s_one = try tfq3.check.matScales(n, k);
    const off_meta: u64 = 64;
    const off_sc: u64 = tfq3.check.section_align;
    const off_wt: u64 = 2 * tfq3.check.section_align;
    const total = off_wt + try tfq3.check.mappedLen(w_one * 2 * 4);
    const shard_buf = try a.alloc(u8, total);
    defer a.free(shard_buf);
    @memset(shard_buf, 0);
    const hdr: tfq3.check.Header = .{
        .magic = tfq3.check.magic_nonmoe,
        .version = tfq3.check.format_version,
        .num_tensors = 2,
        .pad0 = 0,
        .off_meta = off_meta,
        .off_scales = off_sc,
        .off_weights = off_wt,
        .total_scale_halfs = s_one * 2,
        .total_weight_words = w_one * 2,
        .pad1 = 0,
    };
    @memcpy(shard_buf[0..@sizeOf(tfq3.check.Header)], std.mem.asBytes(&hdr));
    const names = [2][]const u8{ "model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj" };
    for (names, 0..) |nm, i| {
        var meta: tfq3.check.TensorMeta = std.mem.zeroes(tfq3.check.TensorMeta);
        @memcpy(meta.name[0..nm.len], nm);
        meta.R = n;
        meta.C = k;
        meta.b_bits = bits;
        meta.sc_off = @intCast(i * s_one);
        meta.word_off = i * w_one;
        meta.word_count = w_one;
        @memcpy(shard_buf[off_meta + i * @sizeOf(tfq3.check.TensorMeta) ..][0..@sizeOf(tfq3.check.TensorMeta)], std.mem.asBytes(&meta));
    }
    const sc_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, shard_buf[off_sc..][0 .. s_one * 2 * 2]));
    for (sc_u16) |*s| s.* = f32ToF16Bits(0.025);
    const wt_u32: []u32 = @alignCast(std.mem.bytesAsSlice(u32, shard_buf[off_wt..][0 .. w_one * 2 * 4]));
    for (wt_u32) |*w| w.* = rng.int(u32);
    try tmp_dir.writeFile(io, .{ .sub_path = "qwen27_tfq3-00001-of-00001.tffpw", .data = shard_buf });

    var store = (try tfq3.Store.open(a, io, device, dir)) orelse return error.MissingStore;
    defer store.deinit();
    const f0 = store.find("model.layers.0.mlp.gate_proj", n, k) orelse return error.MissingTensor;
    const f1 = store.find("model.layers.0.mlp.up_proj", n, k) orelse return error.MissingTensor;
    if (!tfq3.check.canFusePair(f0.part, f1.part)) return error.ExpectedPairFusion;
    var fused = f0.tfq3;
    fused.sr_split = n;
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const x_buf = try device.buffer(@as(usize, k) * 2, opts);
    defer x_buf.deinit();
    for (x_buf.slice(u16, k)) |*v| v.* = f32ToBf16((rng.float(f32) - 0.5) * 0.5);
    const y_buf = try device.buffer(@as(usize, n * 2) * 2, opts);
    defer y_buf.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    try dispatch.encodeTfq3(pipes, e, fused, @intCast(bits), n * 2, k, .{ .buffer = x_buf }, k, .{ .buffer = y_buf }, n * 2, 1);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailure;
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var prng = std.Random.DefaultPrng.init(0x51772703);
    const rng = prng.random();

    // Verify both pre-M5 (TF_SIMD_FRAGS) and M5 (NAX) shader variants compile cleanly on this device.
    for ([_]bool{ false, true }) |units| {
        const src = try dispatch.buildSource(a, units);
        defer a.free(src);
        const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        lib.deinit();
    }
    var pipes = try dispatch.Pipes.init(a, device);
    defer pipes.deinit();
    for ([_]u8{ 3, 4, 5 }) |bits| {
        try runAndCompare(a, device, queue, &pipes, bits, 256, 512, 0, 256, rng);
        try runAndCompare(a, device, queue, &pipes, bits, 256, 1024, 128, 320, rng);
    }
    try checkBf16Ops(device, queue, &pipes, rng);
    try checkShardedStore(a, io, device, queue, &pipes, rng);
    std.debug.print("PASS qwen27_tfq3_test (3b/4b/5b fused + 2dsk8 + prefill + sr_split + strided + bf16 embed/linear + sharded store)\n", .{});
}
