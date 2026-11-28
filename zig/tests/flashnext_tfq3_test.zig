//! Flash Next TFQ3 format, host sidecar scan, Metal compilation, decode parity, and prefill accuracy tests.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const fz = tf.flashnext_replay;
const tfq3 = fz.tfq3;
const rp = @import("flashnext_tfq3_report.zig");

const D = tfq3.hidden_size;
const NI = tfq3.moe_inter;
const NE = tfq3.num_experts;
const TOPK = tfq3.topk;

fn f32ToF16(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}

fn f32ToBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    const rounded = bits +% 0x7FFF +% ((bits >> 16) & 1);
    return @truncate(rounded >> 16);
}

fn bf16ToF32(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
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
    const sc = 1.0 / @sqrt(128.0);
    for (v) |*x| x.* *= sc;
}

fn fragHome(lane: u32) struct { x: u32, y: u32 } {
    return .{
        .x = (lane & 8) + ((lane & 1) << 2),
        .y = ((lane & 16) >> 2) | ((lane >> 1) & 3),
    };
}

fn fillSyntheticTables(buf: *[16384]u8) void {
    @memset(buf, 0);
    const words: [*]u32 = @ptrCast(@alignCast(buf.ptr));
    words[0] = tfq3.magic_v3_tables;
    words[1] = tfq3.format_version;
    var rng = std.Random.DefaultPrng.init(42);
    const rand = rng.random();
    const packH2 = struct {
        fn f(a: f32, b: f32) u32 {
            return @as(u32, f32ToF16(a)) | (@as(u32, f32ToF16(b)) << 16);
        }
    }.f;
    for (0..256) |i| words[16 + i] = packH2((rand.float(f32) - 0.5) * 1.4, (rand.float(f32) - 0.5) * 1.4);
    for (0..1024) |i| words[272 + i] = packH2((rand.float(f32) - 0.5) * 1.4, (rand.float(f32) - 0.5) * 1.4);
    for (0..1024) |i| words[1296 + i] = packH2((rand.float(f32) - 0.5) * 1.4, (rand.float(f32) - 0.5) * 1.4);
    const halfs: [*]u16 = @ptrCast(@alignCast(buf.ptr));
    for ([_]usize{ 4640, 4896, 5152 }) |base| {
        for (0..64) |i| {
            const sx: f32 = if ((i & 1) != 0) -1.0 else 1.0;
            const sy: f32 = if ((i & 2) != 0) -1.0 else 1.0;
            halfs[base + i * 4 + 0] = f32ToF16(sx);
            halfs[base + i * 4 + 1] = f32ToF16(sy);
            halfs[base + i * 4 + 2] = f32ToF16(0.05);
            halfs[base + i * 4 + 3] = f32ToF16(-0.05);
        }
    }
}

fn decodeHalf4(tbl: *const [16384]u8, bits: u32, st_ptr: *u32, syms: [4]u32) [8]f64 {
    const words: [*]const u32 = @ptrCast(@alignCast(tbl.ptr));
    const halfs: [*]const u16 = @ptrCast(@alignCast(tbl.ptr));
    var st = st_ptr.*;
    const sh: u5 = @intCast(bits * 2);
    const h0 = st *% 0x9E3779B1;
    const st1 = (st << sh) | syms[0];
    const h1 = st1 *% 0x9E3779B1;
    const st2 = (st1 << sh) | syms[1];
    const h2 = st2 *% 0x9E3779B1;
    const st3 = (st2 << sh) | syms[2];
    const h3 = st3 *% 0x9E3779B1;
    st = (st3 << sh) | syms[3];
    st_ptr.* = st;

    const c_base: usize = if (bits == 3) 4640 else if (bits == 4) 4896 else 5152;
    const c0_i: usize = @intCast((h0 >> 16) & 63);
    const c2_i: usize = @intCast((h2 >> 16) & 63);
    const f16f = struct {
        fn g(u: u16) f64 {
            return @floatCast(@as(f16, @bitCast(u)));
        }
    }.g;
    const lut_off: usize = if (bits == 3) 16 else if (bits == 4) 272 else 1296;
    const idxs: [4]usize = if (bits == 3) .{
        ((syms[0] << 2) ^ (h0 >> 24)) & 255,
        ((syms[1] << 2) ^ (h1 >> 24)) & 255,
        ((syms[2] << 2) ^ (h2 >> 24)) & 255,
        ((syms[3] << 2) ^ (h3 >> 24)) & 255,
    } else if (bits == 4) .{
        ((syms[0] << 2) ^ (h0 >> 22)) & 1023,
        ((syms[1] << 2) ^ (h1 >> 22)) & 1023,
        ((syms[2] << 2) ^ (h2 >> 22)) & 1023,
        ((syms[3] << 2) ^ (h3 >> 22)) & 1023,
    } else .{
        (syms[0] ^ (h0 >> 22)) & 1023,
        (syms[1] ^ (h1 >> 22)) & 1023,
        (syms[2] ^ (h2 >> 22)) & 1023,
        (syms[3] ^ (h3 >> 22)) & 1023,
    };
    var out: [8]f64 = undefined;
    for (0..4) |p| {
        const cb = if (p < 2) c0_i else c2_i;
        const lv = words[lut_off + idxs[p]];
        const lx = f16f(@truncate(lv & 0xFFFF));
        const ly = f16f(@truncate(lv >> 16));
        const qx: f16 = @floatCast(lx * f16f(halfs[c_base + cb * 4 + 0]) + f16f(halfs[c_base + cb * 4 + 2]));
        const qy: f16 = @floatCast(ly * f16f(halfs[c_base + cb * 4 + 1]) + f16f(halfs[c_base + cb * 4 + 3]));
        out[p * 2 + 0] = @floatCast(qx);
        out[p * 2 + 1] = @floatCast(qy);
    }
    return out;
}

fn matvecCpuRot(
    tbl: *const [16384]u8,
    w_base: []const u32,
    sc_base: []const u16,
    bits: u32,
    R: usize,
    C: usize,
    x_rot: []const f64,
    y_rot: []f64,
) void {
    @memset(y_rot[0..R], 0.0);
    const n_bc = C / 128;
    const wpb: usize = bits * 2;
    for (0..R / 16) |strip| {
        const s_off = strip * n_bc * wpb * 32;
        const sc_row = (strip >> 3) * n_bc;
        for (0..32) |lane| {
            const h = fragHome(@intCast(lane));
            var st: u32 = 0x13579BDF ^ (@as(u32, @intCast(lane)) *% 0x9E3779B1);
            var a0: f64 = 0.0;
            var a1: f64 = 0.0;
            for (0..n_bc) |bc| {
                const sc: f64 = @floatCast(@as(f16, @bitCast(sc_base[sc_row + bc])));
                const blk = w_base[s_off + bc * wpb * 32 ..];
                var ws: [10]u32 = @splat(0);
                if (bits == 4) {
                    for (0..4) |k| ws[k] = blk[lane * 4 + k];
                    for (0..4) |k| ws[k + 4] = blk[128 + lane * 4 + k];
                } else {
                    for (0..wpb) |w| ws[w] = blk[w * 32 + lane];
                }
                var bit_pos: usize = 0;
                var ba0: f64 = 0.0;
                var ba1: f64 = 0.0;
                for (0..8) |kt| {
                    var syms: [4]u32 = undefined;
                    for (0..4) |p| {
                        var sym: u32 = 0;
                        for (0..bits * 2) |_| {
                            const bit = (ws[bit_pos >> 5] >> @intCast(31 - (bit_pos & 31))) & 1;
                            sym = (sym << 1) | bit;
                            bit_pos += 1;
                        }
                        syms[p] = sym;
                    }
                    const q = decodeHalf4(tbl, bits, &st, syms);
                    const cb = bc * 128 + kt * 16 + h.x;
                    for (0..4) |k| {
                        ba0 += q[k] * x_rot[cb + k];
                        ba1 += q[k + 4] * x_rot[cb + k];
                    }
                }
                a0 += ba0 * sc;
                a1 += ba1 * sc;
            }
            y_rot[strip * 16 + h.y] += a0;
            y_rot[strip * 16 + h.y + 8] += a1;
        }
    }
}

test "Flash Next TFQ3: header sizes, sidecar scan, and shader compilation" {
    try std.testing.expectEqual(@as(usize, 144), @sizeOf(tfq3.LayerHeader));
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(tfq3.ExpertMeta));
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(tfq3.Dims));

    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(dir);
    try std.testing.expect(!tfq3.files.hasLayer(io, dir, 0));
    try std.testing.expect(!tfq3.files.hasMtp(io, dir));

    var tbl: [16384]u8 = undefined;
    fillSyntheticTables(&tbl);
    try tmp.dir.writeFile(io, .{ .sub_path = "pw_v3_tables.bin", .data = &tbl });
    try tmp.dir.writeFile(io, .{ .sub_path = "layer_00.tffpw", .data = "0123456789abcdef" });
    try tmp.dir.writeFile(io, .{ .sub_path = "mtp_00.tffpw", .data = "0123456789abcdef" });
    // present but 16 bytes: refused by name before admission (E332), never counted by presence
    try std.testing.expect(tfq3.files.hasLayer(io, dir, 0));
    try std.testing.expectError(error.Tfq3Truncated, tfq3.files.scan(io, dir, 48));

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const dev = try mtl.Device.init();
    defer dev.deinit();
    const src_simd = try tfq3.buildSource(a, false);
    defer a.free(src_simd);
    const lib_simd = try mtl.Library.fromSource(dev, src_simd, mtl.CompileOptions.mlx());
    lib_simd.deinit();
    var pipes = try tfq3.Pipes.init(a, dev, null);
    defer pipes.deinit();
    try std.testing.expect(pipes.hash != 0);
}

test "Flash Next TFQ3: decode B=1 vs CPU reference, B=1==B=4 and a repeat bit-exact, both modes, prefill accuracy" {
    const dev = try mtl.Device.init();
    const units = dev.tensorUnits();
    dev.deinit();
    // the device's own mode, then on an M5 the simdgroup source too: its decode only (a slice: no u1 mode count)
    const modes: []const bool = if (units) &.{ true, false } else &.{false};
    for (modes) |mode| parity(mode) catch |e| {
        std.debug.print("TFQ3 parity, {s} mode: {s}\n", .{ (rp.Run{ .units = mode }).mode(), @errorName(e) });
        return e;
    };
}

/// Decode B=1 against B=4 and a repeated B=4 exactly and against the CPU reference; own-chip prefill within 0.02.
fn parity(units: bool) !void {
    const a = std.testing.allocator;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const dev = try mtl.Device.init();
    defer dev.deinit();
    const queue = try dev.queue();
    defer queue.deinit();
    var pipes = try tfq3.Pipes.initUnits(a, dev, null, units);
    defer pipes.deinit();
    var ck: rp.Run = .{ .units = units };

    const shared_opt = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    var tbl_bytes: [16384]u8 = undefined;
    fillSyntheticTables(&tbl_bytes);
    const b_tbl = try dev.buffer(16384, shared_opt);
    defer b_tbl.deinit();
    @memcpy(b_tbl.contents()[0..16384], &tbl_bytes);

    // 3 distinct expert configs (3b, 4b, 5b) replicated across 512 metadata entries.
    const wpm3: usize = 51200 * 3;
    const wpm4: usize = 51200 * 4;
    const wpm5: usize = 51200 * 5;
    const total_words = (wpm3 + wpm4 + wpm5) * 3;
    const total_sc = 3 * 3 * 100;
    const b_w = try dev.buffer(total_words * 4, shared_opt);
    defer b_w.deinit();
    const b_s = try dev.buffer(total_sc * 2, shared_opt);
    defer b_s.deinit();
    const b_m = try dev.buffer(NE * @sizeOf(tfq3.ExpertMeta), shared_opt);
    defer b_m.deinit();

    var rng = std.Random.DefaultPrng.init(12345);
    const rand = rng.random();
    const w_sl = b_w.slice(u32, total_words);
    for (w_sl) |*v| v.* = rand.int(u32);
    const s_sl = b_s.slice(u16, total_sc);
    for (s_sl) |*v| v.* = f32ToF16(0.02 + rand.float(f32) * 0.03);

    const metas = b_m.slice(tfq3.ExpertMeta, NE);
    const offs_w = [3]u64{ 0, wpm3 * 3, (wpm3 + wpm4) * 3 };
    const wpms = [3]u64{ wpm3, wpm4, wpm5 };
    const bits_arr = [3]u32{ 3, 4, 5 };
    for (0..NE) |ei| {
        const cls = ei % 3;
        const wo = offs_w[cls];
        const wm = wpms[cls];
        const so: u32 = @intCast(cls * 300);
        metas[ei] = .{
            .b_bits = bits_arr[cls],
            .heat_rank = @intCast(ei),
            .gate_word_off = wo,
            .up_word_off = wo + wm,
            .down_word_off = wo + wm * 2,
            .gate_sc_off = so,
            .up_sc_off = so + 100,
            .down_sc_off = so + 200,
            .gate_snr_db = 20.0,
            .up_snr_db = 20.0,
            .down_snr_db = 20.0,
            .reserved = .{ 0, 0 },
        };
    }

    const tl: tfq3.Layer = .{
        .layer_idx = 0,
        .bytes = 0,
        .tables = b_tbl,
        .meta = b_m,
        .scales = b_s,
        .weights = b_w,
        .has_5b = true,
    };

    // Shared expert 8-bit/32 affine weights.
    const sh_gu_w = try dev.buffer(NI * D, shared_opt);
    defer sh_gu_w.deinit();
    const sh_gu_s = try dev.buffer(NI * (D / 32) * 2, shared_opt);
    defer sh_gu_s.deinit();
    const sh_gu_b = try dev.buffer(NI * (D / 32) * 2, shared_opt);
    defer sh_gu_b.deinit();
    const sh_dn_w = try dev.buffer(D * NI, shared_opt);
    defer sh_dn_w.deinit();
    const sh_dn_s = try dev.buffer(D * (NI / 32) * 2, shared_opt);
    defer sh_dn_s.deinit();
    const sh_dn_b = try dev.buffer(D * (NI / 32) * 2, shared_opt);
    defer sh_dn_b.deinit();
    for (sh_gu_w.contents()[0 .. NI * D]) |*v| v.* = rand.int(u8);
    for (sh_dn_w.contents()[0 .. D * NI]) |*v| v.* = rand.int(u8);
    for (sh_gu_s.slice(u16, NI * (D / 32))) |*v| v.* = f32ToBf16(0.0004);
    for (sh_gu_b.slice(u16, NI * (D / 32))) |*v| v.* = f32ToBf16(-0.05);
    for (sh_dn_s.slice(u16, D * (NI / 32))) |*v| v.* = f32ToBf16(0.0004);
    for (sh_dn_b.slice(u16, D * (NI / 32))) |*v| v.* = f32ToBf16(-0.05);

    var ex: [18]fz.Buf = @splat(.{ .b = sh_gu_w });
    ex[6] = .{ .b = sh_gu_w };
    ex[7] = .{ .b = sh_gu_s };
    ex[8] = .{ .b = sh_gu_b };
    ex[9] = .{ .b = sh_gu_w };
    ex[10] = .{ .b = sh_gu_s };
    ex[11] = .{ .b = sh_gu_b };
    ex[15] = .{ .b = sh_dn_w };
    ex[16] = .{ .b = sh_dn_s };
    ex[17] = .{ .b = sh_dn_b };
    const xw: fz.variants.Experts = .{ .gu = .{ 6, 32, 8, 32 }, .down = .{ 6, 32, 8, 32 } };

    // Test inputs for 4 rows (rows 0..3).
    const rows: usize = 4;
    const b_x = try dev.buffer(rows * D * 2, shared_opt);
    defer b_x.deinit();
    const b_lg = try dev.buffer(rows * 513 * 4, shared_opt);
    defer b_lg.deinit();
    const b_act = try dev.buffer(rows * 11 * NI * 2, shared_opt);
    defer b_act.deinit();
    const b_pick1 = try dev.buffer(rows * TOPK * 4, shared_opt);
    defer b_pick1.deinit();
    const b_wts1 = try dev.buffer(rows * TOPK * 4, shared_opt);
    defer b_wts1.deinit();
    const b_y1 = try dev.buffer(rows * 11 * D * 2, shared_opt);
    defer b_y1.deinit();
    const b_pick4 = try dev.buffer(rows * TOPK * 4, shared_opt);
    defer b_pick4.deinit();
    const b_wts4 = try dev.buffer(rows * TOPK * 4, shared_opt);
    defer b_wts4.deinit();
    const b_y4 = try dev.buffer(rows * 11 * D * 2, shared_opt);
    defer b_y4.deinit();
    const b_r1 = try dev.buffer(16, shared_opt);
    defer b_r1.deinit();
    const b_r4 = try dev.buffer(16, shared_opt);
    defer b_r4.deinit();
    b_r1.slice(i32, 1)[0] = 1;
    b_r4.slice(i32, 1)[0] = @intCast(rows);

    for (b_x.slice(u16, rows * D)) |*v| v.* = f32ToBf16((rand.float(f32) - 0.5) * 0.8);
    const lg_sl = b_lg.slice(f32, rows * 513);
    for (lg_sl) |*v| v.* = (rand.float(f32) - 0.5) * 2.0;
    // Force overlap of experts 0 (3b), 1 (4b), 2 (5b) across all 4 rows.
    for (0..rows) |r| {
        lg_sl[r * 513 + 0] = 9.0;
        lg_sl[r * 513 + 1] = 8.5;
        lg_sl[r * 513 + 2] = 8.0;
    }

    var run: fz.Run = .{ .arena = a, .device = dev, .queue = queue, .serial = true };
    // Run B=1 independently for each row 0..3.
    for (0..rows) |r| {
        const cb = queue.commandBuffer();
        run.enc = cb.compute(.serial);
        tfq3.dispatch.moeDecode(
            &pipes,
            &run,
            &tl,
            .{ .b = b_x, .off = r * D * 2 },
            .{ .b = b_lg, .off = r * 513 * 4 },
            ex,
            .{ .b = b_act, .off = r * 11 * NI * 2 },
            .{ .b = b_pick1, .off = r * TOPK * 4 },
            .{ .b = b_wts1, .off = r * TOPK * 4 },
            .{ .b = b_r1 },
            .{ .b = b_y1, .off = r * 11 * D * 2 },
            1,
            xw,
        );
        run.enc.end();
        cb.commit();
        cb.wait();
        try ck.ran("decode B=1, row", r, cb.failure());
    }
    for (0..rows) |r| {
        for (0..TOPK) |k| {
            const eid = b_pick1.slice(u32, rows * TOPK)[r * TOPK + k];
            ck.widths[r * 11 + k] = if (eid < NE) metas[eid].b_bits else 0;
        }
        ck.widths[r * 11 + 10] = 8;
    }

    // Run B=4 deduplicated in one pass, three times: route_dedup, then the default twice, all the same output bytes.
    var first: ?[]u16 = null;
    defer if (first) |f| a.free(f);
    for (0..3) |pass| {
        pipes.serial_route = pass == 0; // pass 0: main's route_dedup; 1, 2: the default, the same bytes, last
        const cb = queue.commandBuffer();
        run.enc = cb.compute(.serial);
        tfq3.dispatch.moeDecode(
            &pipes,
            &run,
            &tl,
            .{ .b = b_x },
            .{ .b = b_lg },
            ex,
            .{ .b = b_act },
            .{ .b = b_pick4 },
            .{ .b = b_wts4 },
            .{ .b = b_r4 },
            .{ .b = b_y4 },
            rows,
            xw,
        );
        run.enc.end();
        cb.commit();
        cb.wait();
        try ck.ran("decode B=4, pass", pass, cb.failure());
        const now = b_y4.slice(u16, rows * 11 * D);
        if (pass == 0) first = try a.dupe(u16, now) else try ck.same(u16, .bf16, "B=4 outputs and their repeat", D, 11, first.?, now);
    }

    // Verify 100% bit-exact B=1 == B=4 parity for picks, weights, routed outputs, and shared expert!
    try ck.same(u32, .id, "B=1 and B=4 picks", 1, TOPK, b_pick1.slice(u32, rows * TOPK), b_pick4.slice(u32, rows * TOPK));
    try ck.same(u32, .f32, "B=1 and B=4 routing weights", 1, TOPK, b_wts1.slice(u32, rows * TOPK), b_wts4.slice(u32, rows * TOPK));
    try ck.same(u16, .bf16, "B=1 and B=4 outputs", D, 11, b_y1.slice(u16, rows * 11 * D), b_y4.slice(u16, rows * 11 * D));

    // Verify row 0 routed experts 0 (3b), 1 (4b), 2 (5b) against CPU fp64 reference.
    var x_rot: [D]f64 = undefined;
    const x0 = b_x.slice(u16, D);
    for (0..D / 128) |bc| {
        var blk: [128]f64 = undefined;
        for (0..128) |j| blk[j] = @as(f64, bf16ToF32(x0[bc * 128 + j])) * pwSgn(0x12345678, @intCast(bc * 128 + j));
        fwht128(&blk);
        for (0..128) |j| x_rot[bc * 128 + j] = @floatCast(@as(f16, @floatCast(blk[j])));
    }
    const y1_sl = b_y1.slice(u16, 11 * D);
    for (0..3) |slot| {
        const eid = b_pick1.slice(u32, TOPK)[slot];
        const em = metas[eid];
        var g_rot: [NI]f64 = undefined;
        var u_rot: [NI]f64 = undefined;
        matvecCpuRot(&tbl_bytes, w_sl[@intCast(em.gate_word_off)..], s_sl[em.gate_sc_off..], em.b_bits, NI, D, &x_rot, &g_rot);
        matvecCpuRot(&tbl_bytes, w_sl[@intCast(em.up_word_off)..], s_sl[em.up_sc_off..], em.b_bits, NI, D, &x_rot, &u_rot);
        var act_rot: [NI]f64 = undefined;
        for (0..NI / 128) |br| {
            var gb: [128]f64 = undefined;
            var ub: [128]f64 = undefined;
            for (0..128) |j| {
                gb[j] = g_rot[br * 128 + j];
                ub[j] = u_rot[br * 128 + j];
            }
            fwht128(&gb);
            fwht128(&ub);
            var zb: [128]f64 = undefined;
            for (0..128) |j| {
                const idx: u32 = @intCast(br * 128 + j);
                const gv: f32 = bf16ToF32(f32ToBf16(@floatCast(gb[j] * pwSgn(0x9E3779B9, idx))));
                const uv: f32 = bf16ToF32(f32ToBf16(@floatCast(ub[j] * pwSgn(0x9E3779B9, idx))));
                const silu = bf16ToF32(f32ToBf16(gv / (1.0 + @exp(-gv))));
                const act = bf16ToF32(f32ToBf16(silu * uv));
                zb[j] = @as(f64, act) * pwSgn(0x12345678, idx);
            }
            fwht128(&zb);
            for (0..128) |j| act_rot[br * 128 + j] = @floatCast(@as(f16, @floatCast(zb[j])));
        }
        var yr: [D]f64 = undefined;
        matvecCpuRot(&tbl_bytes, w_sl[@intCast(em.down_word_off)..], s_sl[em.down_sc_off..], em.b_bits, D, NI, &act_rot, &yr);
        var near: rp.Near = .{ .bound = 0.02 };
        for (0..D / 128) |br| {
            var yb: [128]f64 = undefined;
            for (0..128) |j| yb[j] = yr[br * 128 + j];
            fwht128(&yb);
            for (0..128) |j| {
                const idx = br * 128 + j;
                const ref = yb[j] * pwSgn(0x9E3779B9, @intCast(idx));
                near.see(idx, ref, bf16ToF32(y1_sl[slot * D + idx]));
            }
        }
        try ck.near("routed expert vs CPU fp64 reference", slot, em.b_bits, near);
    }

    // Verify shared expert (slot 10, 8-bit/32) against CPU fp64 reference.
    var sh_act: [NI]f64 = undefined;
    const sgu_w = sh_gu_w.contents()[0 .. NI * D];
    const sgu_s = sh_gu_s.slice(u16, NI * (D / 32));
    const sgu_b = sh_gu_b.slice(u16, NI * (D / 32));
    for (0..NI) |r| {
        var dot: f64 = 0.0;
        for (0..D) |c| {
            const g = r * (D / 32) + c / 32;
            const wv = @as(f64, @floatFromInt(sgu_w[r * D + c])) * bf16ToF32(sgu_s[g]) + bf16ToF32(sgu_b[g]);
            dot += wv * bf16ToF32(x0[c]);
        }
        const gv = bf16ToF32(f32ToBf16(@floatCast(dot)));
        const silu = bf16ToF32(f32ToBf16(gv / (1.0 + @exp(-gv))));
        sh_act[r] = bf16ToF32(f32ToBf16(silu * gv));
    }
    const sdn_w = sh_dn_w.contents()[0 .. D * NI];
    const sdn_s = sh_dn_s.slice(u16, D * (NI / 32));
    const sdn_b = sh_dn_b.slice(u16, D * (NI / 32));
    var sh_near: rp.Near = .{ .bound = 0.02 };
    for (0..D) |r| {
        var dot: f64 = 0.0;
        for (0..NI) |c| {
            const g = r * (NI / 32) + c / 32;
            const wv = @as(f64, @floatFromInt(sdn_w[r * NI + c])) * bf16ToF32(sdn_s[g]) + bf16ToF32(sdn_b[g]);
            dot += wv * sh_act[c];
        }
        sh_near.see(r, dot, bf16ToF32(y1_sl[10 * D + r]));
    }
    try ck.near("shared expert vs CPU fp64 reference", 10, 8, sh_near);

    // Prefill MMA slots follow one chip's matmul2d layout (nax.h): the simdgroup source's prefill is pre-M5 only
    if (units != dev.tensorUnits()) return;
    // Prefill MoE against decode B=1 across 3 sorted pairs: within 0.02, as the format's prefill may round apart.
    const b_xs = try dev.buffer(3 * D * 2, shared_opt);
    defer b_xs.deinit();
    const b_off = try dev.buffer((NE + 1) * 4, shared_opt);
    defer b_off.deinit();
    const b_pa = try dev.buffer(3 * NI * 2, shared_opt);
    defer b_pa.deinit();
    const b_ds = try dev.buffer(3 * D * 2, shared_opt);
    defer b_ds.deinit();
    const b_tm = try dev.buffer(@as(usize, tfq3.max_prefill_tiles) * 4, shared_opt);
    defer b_tm.deinit();
    const b_tc = try dev.buffer(16, shared_opt);
    defer b_tc.deinit();
    for (0..3) |p| @memcpy(b_xs.contents()[p * D * 2 .. (p + 1) * D * 2], b_x.contents()[0 .. D * 2]);
    const off_sl = b_off.slice(i32, NE + 1);
    off_sl[0] = 0;
    off_sl[1] = 1;
    off_sl[2] = 2;
    for (3..NE + 1) |k| off_sl[k] = 3;
    {
        const cb = queue.commandBuffer();
        run.enc = cb.compute(.serial);
        tfq3.dispatch.moePrefill(&pipes, &run, &tl, .{ .b = b_xs }, .{ .b = b_off }, .{ .b = b_pa }, .{ .b = b_ds }, .{ .b = b_tm }, .{ .b = b_tc }, 3);
        run.enc.end();
        cb.commit();
        cb.wait();
        try ck.ran("prefill, sorted pairs", 3, cb.failure());
    }
    const ds_sl = b_ds.slice(u16, 3 * D);
    for (0..3) |slot| {
        var near: rp.Near = .{ .bound = 0.02 };
        for (0..D) |j| near.see(j, bf16ToF32(y1_sl[slot * D + j]), bf16ToF32(ds_sl[slot * D + j]));
        try ck.near("prefill vs decode B=1", slot, ck.widths[slot], near);
    }
}
