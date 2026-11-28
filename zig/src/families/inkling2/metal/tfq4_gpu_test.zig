//! Apple Silicon Metal GPU fp64 parity & width-invariance tests for Inkling 2 TFQ4 dense and hybrid TFPW MoE.
const std = @import("std");
const mtl = @import("metal");
const tfq3 = @import("tfq3.zig");
const tf4 = tfq3.tfq4;

fn sgn(seed: u32, idx: u32) f64 {
    const h: u32 = (seed ^ idx) *% 2654435761;
    return if ((h >> 31) & 1 != 0) -1.0 else 1.0;
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
    const inv: f64 = 1.0 / @sqrt(128.0);
    for (v) |*x| x.* *= inv;
}

fn bf16ToF64(b: u16) f64 {
    return @floatCast(@as(f32, @bitCast(@as(u32, b) << 16)));
}

fn f32ToBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u +% 0x7FFF +% ((u >> 16) & 1)) >> 16);
}

fn cosine(a: []const f64, b: []const f64) f64 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    return if (na == 0 or nb == 0) 0 else dot / (@sqrt(na) * @sqrt(nb));
}

fn uploadBytes(device: mtl.Device, bytes: []const u8) !mtl.Buffer {
    const buf = try device.buffer(@max(bytes.len, 16), mtl.ResourceOptions.shared);
    @memcpy(buf.contents()[0..bytes.len], bytes);
    return buf;
}

fn trainedTables() [tf4.tables_256_bytes]u8 {
    var b: [tf4.tables_256_bytes]u8 = @splat(0);
    std.mem.writeInt(u32, b[0..4], tf4.magic_tables, .little);
    std.mem.writeInt(u32, b[4..8], tf4.format_version, .little);
    std.mem.writeInt(u32, b[8..12], 256, .little);
    std.mem.writeInt(u32, b[12..16], 8, .little);
    std.mem.writeInt(u32, b[16..20], 0, .little);
    const lut = std.mem.bytesAsSlice(u16, b[64 .. 64 + 256 * 8 * 2]);
    for (lut, 0..) |*h, i| {
        const v: f16 = @floatCast(@as(f32, @floatFromInt((i * 19 + 5) % 29)) / 16.0);
        h.* = @bitCast(v);
    }
    return b;
}

fn synthTables3() [tfq3.tables_file_bytes]u8 {
    var b: [tfq3.tables_file_bytes]u8 = @splat(0);
    std.mem.writeInt(u32, b[0..4], tfq3.magic_tables, .little);
    std.mem.writeInt(u32, b[4..8], tfq3.format_version, .little);
    for (0..6) |i| std.mem.writeInt(u32, b[8 + i * 4 ..][0..4], @bitCast(@as(f32, 0.125)), .little);
    var off: usize = 64;
    while (off < tfq3.tables_bytes) : (off += 2) {
        const v: f16 = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((off * 13) % 31)) - 15)) / 32.0);
        std.mem.writeInt(u16, b[off..][0..2], @bitCast(v), .little);
    }
    return b;
}

fn cpuDenseRow(
    gpa: std.mem.Allocator,
    words: []const u32,
    scales: []const u16,
    lut: []align(1) const u16,
    table_kind: u32,
    x_bf16: []const u16,
    n: u32,
    k: u32,
    bits: u32,
    row_off: u32,
    sr_split: u32,
    out: []f64,
) !void {
    const n_bc = k / 128;
    const wpb: usize = if (bits == 225) 9 else 8;
    const wpr = n_bc * wpb;
    const x_rot = try gpa.alloc(f64, k);
    defer gpa.free(x_rot);
    for (0..n_bc) |bc| {
        var blk: [128]f64 = undefined;
        for (0..128) |c| blk[c] = bf16ToF64(x_bf16[bc * 128 + c]) * sgn(0x12345678, @intCast(bc * 128 + c));
        fwht128(&blk);
        for (0..128) |c| x_rot[bc * 128 + c] = blk[c];
    }
    var rb: u32 = 0;
    while (rb < n / 128) : (rb += 1) {
        var y_rot: [128]f64 = @splat(0);
        for (0..128) |lr| {
            const gr = rb * 128 + @as(u32, @intCast(lr));
            var acc: f64 = 0;
            for (0..n_bc) |bc| {
                var w128: [128]f64 = undefined;
                const woff = @as(usize, gr) * wpr + bc * wpb;
                const sc: f64 = @floatCast(@as(f16, @bitCast(scales[rb * n_bc + bc])));
                tf4.decodeRowBlock128(words[woff .. woff + wpb], bits, table_kind, lut, sc, &w128);
                var d: f64 = 0;
                for (0..128) |c| d += w128[c] * x_rot[bc * 128 + c];
                acc += d;
            }
            y_rot[lr] = acc;
        }
        fwht128(&y_rot);
        const r0 = rb * 128;
        const sr0 = if (sr_split != 0) (r0 + row_off) % sr_split else (r0 + row_off);
        for (0..128) |lr| out[r0 + lr] = y_rot[lr] * sgn(0x9E3779B9, sr0 + @as(u32, @intCast(lr)));
    }
}

fn runDenseCase(
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    pipes: *const tfq3.Pipelines,
    tbl_bytes: []const u8,
    table_kind: u32,
    bits: u32,
) !void {
    const rows: u32 = 4;
    const n: u32 = 256;
    const k: u32 = 256;
    const ldo: u32 = n + 128;
    const wc = try tf4.tfq4MatWords(n, k, bits);
    const words = try gpa.alloc(u32, @intCast(wc));
    defer gpa.free(words);
    var rng = std.Random.DefaultPrng.init(0x74667134 + table_kind * 17 + bits);
    for (words) |*w| w.* = rng.random().int(u32);
    const scales = try gpa.alloc(u16, (n / 128) * (k / 128));
    defer gpa.free(scales);
    @memset(scales, @as(u16, @bitCast(@as(f16, 0.125))));
    const x = try gpa.alloc(u16, rows * k);
    defer gpa.free(x);
    for (x, 0..) |*v, i| {
        const f: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + i / k) % 37)) - 18)) / 64.0;
        v.* = f32ToBf16(f);
    }
    const bw = try uploadBytes(device, std.mem.sliceAsBytes(words));
    defer bw.deinit();
    const bs = try uploadBytes(device, std.mem.sliceAsBytes(scales));
    defer bs.deinit();
    const bt = try uploadBytes(device, tbl_bytes);
    defer bt.deinit();
    const bx = try uploadBytes(device, std.mem.sliceAsBytes(x));
    defer bx.deinit();
    const lut = std.mem.bytesAsSlice(u16, tbl_bytes[64 .. 64 + (if (table_kind == 3) @as(usize, 1024) else 256) * 8 * 2]);
    const ref0 = try gpa.alloc(f64, n);
    defer gpa.free(ref0);
    try cpuDenseRow(gpa, words, scales, lut, table_kind, x[0..k], n, k, bits, 128, 256, ref0);

    for ([_]u32{ 0, 1 }) |out_f32| {
        const elem: usize = if (out_f32 != 0) 4 else 2;
        const out_len = rows * ldo * elem;
        const bo_serial = try device.buffer(out_len, mtl.ResourceOptions.shared);
        defer bo_serial.deinit();
        const bo_batch = try device.buffer(out_len, mtl.ResourceOptions.shared);
        defer bo_batch.deinit();
        @memset(bo_serial.contents()[0..out_len], 0x5a);
        @memset(bo_batch.contents()[0..out_len], 0x5a);
        {
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            for (0..rows) |r| {
                try pipes.dispatchDense4(enc, bw, bs, bt, .{ .buf = bx, .off = r * k * 2 }, .{ .buf = bo_serial, .off = r * ldo * elem }, .{
                    .rows = 1,
                    .n = n,
                    .k = k,
                    .ldo = ldo,
                    .bits = bits,
                    .table_kind = table_kind,
                    .row_off = 128,
                    .sr_split = 256,
                    .out_f32 = out_f32,
                });
            }
            try pipes.dispatchDense4(enc, bw, bs, bt, .{ .buf = bx, .off = 0 }, .{ .buf = bo_batch, .off = 0 }, .{
                .rows = rows,
                .n = n,
                .k = k,
                .ldo = ldo,
                .bits = bits,
                .table_kind = table_kind,
                .row_off = 128,
                .sr_split = 256,
                .out_f32 = out_f32,
            });
            enc.end();
            cb.commit();
            cb.wait();
            try std.testing.expect(cb.failure() == null);
        }
        try std.testing.expectEqualSlices(u8, bo_serial.contents()[0..out_len], bo_batch.contents()[0..out_len]);
        const got0 = try gpa.alloc(f64, n);
        defer gpa.free(got0);
        for (0..n) |c| {
            got0[c] = if (out_f32 != 0)
                @floatCast(@as(f32, @bitCast(std.mem.readInt(u32, bo_batch.contents()[c * 4 ..][0..4], .little))))
            else
                bf16ToF64(std.mem.readInt(u16, bo_batch.contents()[c * 2 ..][0..2], .little));
        }
        const cos = cosine(got0, ref0);
        const req: f64 = if (out_f32 != 0) 0.999999 else 0.9999;
        try std.testing.expect(cos >= req);
    }
}

test "Apple Silicon Metal TFQ4 dense (kind 0/2/3, 2b/225b) fp64 parity and width invariance" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const gpa = std.testing.allocator;
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var pipes = try tfq3.Pipelines.load(device);
    defer pipes.deinit();

    const tbl0 = trainedTables();
    var tbl2: [tf4.tables_256_bytes]u8 = undefined;
    try tf4.fillTrueE8pTables(&tbl2, 2);
    var tbl3: [tf4.tables_1024_bytes]u8 = undefined;
    try tf4.fillTrueE8pTables(&tbl3, 3);
    try runDenseCase(gpa, device, queue, &pipes, &tbl0, 0, 2);
    try runDenseCase(gpa, device, queue, &pipes, &tbl2, 2, 2);
    try runDenseCase(gpa, device, queue, &pipes, &tbl3, 3, 2);
    try runDenseCase(gpa, device, queue, &pipes, &tbl3, 3, 225);
}

test "Apple Silicon Metal Inkling-2 Hybrid TFPW MoE (2/225/3/4/5b) fp64 parity, 3/4/5b exactness, width invariance" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const gpa = std.testing.allocator;
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var pipes = try tfq3.Pipelines.load(device);
    defer pipes.deinit();

    const hidden: u32 = 2304;
    const inter: u32 = 1152;
    const rows: u32 = 4;
    const topk: u32 = 5;
    const bits_list = [_]u32{ 2, 225, 3, 4, 5 };
    const sc_gu = try tfq3.matScales(inter, hidden);
    const sc_dn = try tfq3.matScales(hidden, inter);
    var metas: [5]tfq3.ExpertMeta = undefined;
    var wc: u64 = 0;
    var sc: u32 = 0;
    for (&metas, bits_list) |*m, b| {
        const gu_w = try tfq3.matWords(inter, hidden, b);
        const dn_w = try tfq3.matWords(hidden, inter, b);
        m.* = std.mem.zeroes(tfq3.ExpertMeta);
        m.b_bits = b;
        m.gate_word_off = wc;
        wc += gu_w;
        m.up_word_off = wc;
        wc += gu_w;
        m.down_word_off = wc;
        wc += dn_w;
        m.gate_sc_off = sc;
        sc += @intCast(sc_gu);
        m.up_sc_off = sc;
        sc += @intCast(sc_gu);
        m.down_sc_off = sc;
        sc += @intCast(sc_dn);
    }
    const words = try gpa.alloc(u32, @intCast(wc));
    defer gpa.free(words);
    var rng = std.Random.DefaultPrng.init(0x696e6b32);
    for (words) |*w| w.* = rng.random().int(u32);
    const scales = try gpa.alloc(u16, sc);
    defer gpa.free(scales);
    @memset(scales, @as(u16, @bitCast(@as(f16, 0.0625))));
    const x = try gpa.alloc(u16, rows * hidden);
    defer gpa.free(x);
    for (x, 0..) |*v, i| {
        const f: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 19 + i / hidden) % 31)) - 15)) / 64.0;
        v.* = f32ToBf16(f);
    }
    var picks: [rows * topk]i32 = undefined;
    for (0..rows) |r| for (0..topk) |s| {
        picks[r * topk + s] = @intCast((r + s) % 5);
    };
    const tbl3_bytes = synthTables3();
    var tbl4_bytes: [tf4.tables_1024_bytes]u8 = undefined;
    try tf4.fillTrueE8pTables(&tbl4_bytes, 3);

    const bw = try uploadBytes(device, std.mem.sliceAsBytes(words));
    defer bw.deinit();
    const bs = try uploadBytes(device, std.mem.sliceAsBytes(scales));
    defer bs.deinit();
    const bm = try uploadBytes(device, std.mem.asBytes(&metas));
    defer bm.deinit();
    const bt3 = try uploadBytes(device, &tbl3_bytes);
    defer bt3.deinit();
    const bt4 = try uploadBytes(device, &tbl4_bytes);
    defer bt4.deinit();
    const bx = try uploadBytes(device, std.mem.sliceAsBytes(x));
    defer bx.deinit();
    const bp = try uploadBytes(device, std.mem.sliceAsBytes(&picks));
    defer bp.deinit();
    const act = try device.buffer(rows * topk * inter * 2, mtl.ResourceOptions.shared);
    defer act.deinit();
    const out_serial = try device.buffer(rows * topk * hidden * 2, mtl.ResourceOptions.shared);
    defer out_serial.deinit();
    const out_batch = try device.buffer(rows * topk * hidden * 2, mtl.ResourceOptions.shared);
    defer out_batch.deinit();
    const out_pure = try device.buffer(3 * hidden * 2, mtl.ResourceOptions.shared);
    defer out_pure.deinit();

    const hybrid_layer: tfq3.Layer = .{
        .layer_idx = 0,
        .num_experts = 5,
        .hidden = hidden,
        .inter = inter,
        .tables = bt3,
        .tables4 = bt4,
        .table4_kind = 3,
        .has_tfq4 = true,
        .meta = bm,
        .scales = bs,
        .weights = bw,
        .has_5b = true,
        .bytes = 0,
    };
    var pure_layer = hybrid_layer;
    pure_layer.has_tfq4 = false;
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        for (0..rows) |r| {
            try pipes.dispatchWith(
                enc,
                hybrid_layer,
                .{ .buf = bx, .off = r * hidden * 2 },
                .{ .buf = bp, .off = r * topk * 4 },
                .{ .buf = act, .off = 0 },
                .{ .buf = out_serial, .off = r * topk * hidden * 2 },
                1,
                topk,
                4.0,
                25.0,
            );
        }
        try pipes.dispatchWith(
            enc,
            hybrid_layer,
            .{ .buf = bx, .off = 0 },
            .{ .buf = bp, .off = 0 },
            .{ .buf = act, .off = 0 },
            .{ .buf = out_batch, .off = 0 },
            rows,
            topk,
            4.0,
            25.0,
        );
        try pipes.dispatchWith(
            enc,
            pure_layer,
            .{ .buf = bx, .off = 0 },
            .{ .buf = bp, .off = 2 * 4 },
            .{ .buf = act, .off = 0 },
            .{ .buf = out_pure, .off = 0 },
            1,
            3,
            4.0,
            25.0,
        );
        enc.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
    }
    const out_len = rows * topk * hidden * 2;
    const pure_len = 3 * hidden * 2;
    try std.testing.expectEqualSlices(u8, out_serial.contents()[0..out_len], out_batch.contents()[0..out_len]);
    try std.testing.expectEqualSlices(u8, out_batch.contents()[2 * hidden * 2 .. 5 * hidden * 2], out_pure.contents()[0..pure_len]);

    const lut4 = std.mem.bytesAsSlice(u16, tbl4_bytes[64 .. 64 + 1024 * 8 * 2]);
    const g_ref = try gpa.alloc(f64, inter);
    defer gpa.free(g_ref);
    const u_ref = try gpa.alloc(f64, inter);
    defer gpa.free(u_ref);
    const act_bf16 = try gpa.alloc(u16, inter);
    defer gpa.free(act_bf16);
    const y_ref = try gpa.alloc(f64, hidden);
    defer gpa.free(y_ref);
    const y_got = try gpa.alloc(f64, hidden);
    defer gpa.free(y_got);
    for ([_]usize{ 0, 1 }) |slot| {
        const em = metas[slot];
        const gu_w = try tfq3.matWords(inter, hidden, em.b_bits);
        const dn_w = try tfq3.matWords(hidden, inter, em.b_bits);
        try cpuDenseRow(gpa, words[@intCast(em.gate_word_off)..][0..gu_w], scales[em.gate_sc_off..][0..sc_gu], lut4, 3, x[0..hidden], inter, hidden, em.b_bits, 0, 0, g_ref);
        try cpuDenseRow(gpa, words[@intCast(em.up_word_off)..][0..gu_w], scales[em.up_sc_off..][0..sc_gu], lut4, 3, x[0..hidden], inter, hidden, em.b_bits, 0, 0, u_ref);
        for (0..inter) |i| {
            const gf: f32 = @floatCast(g_ref[i]);
            const uf: f32 = @floatCast(u_ref[i]);
            const gc = 4.0 * std.math.tanh(gf / 4.0);
            const uc = 25.0 * std.math.tanh(uf / 25.0);
            act_bf16[i] = f32ToBf16((gc / (1.0 + @exp(-gf))) * uc);
        }
        try cpuDenseRow(gpa, words[@intCast(em.down_word_off)..][0..dn_w], scales[em.down_sc_off..][0..sc_dn], lut4, 3, act_bf16, hidden, inter, em.b_bits, 0, 0, y_ref);
        for (0..hidden) |c| {
            const off = (slot * hidden + c) * 2;
            y_got[c] = bf16ToF64(std.mem.readInt(u16, out_batch.contents()[off..][0..2], .little));
        }
        const cos = cosine(y_got, y_ref);
        try std.testing.expect(cos >= 0.9999);
    }
}
