//! CUDA TFQ4 dense (table_kind 0/2/3, b_bits 2/225) and hybrid TFPW MoE (2/225/3/4/5b) parity & width tests.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const data = @import("tfq3_test_data.zig");
const fmt = cuda.tfq3_format;
const tf4fmt = fmt.tfq4;
const tf = cuda.tfq3;
const tf4 = cuda.tfq4;

const test_widths = [_]usize{ 1, 2, 3, 4, 5, 15, 16, 17, 32, 64, 513 };
const moe_widths = [_]usize{ 1, 2, 4, 15, 16, 17, 64 };
const dense_rows = 513;
const moe_rows = 64;

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
    const u: u32 = @as(u32, b) << 16;
    return @floatCast(@as(f32, @bitCast(u)));
}

fn f32FromBytes(buf: []const u8, idx: usize, f32_out: bool) f64 {
    if (f32_out) {
        const u = std.mem.readInt(u32, buf[idx * 4 ..][0..4], .little);
        return @floatCast(@as(f32, @bitCast(u)));
    }
    return bf16ToF64(std.mem.readInt(u16, buf[idx * 2 ..][0..2], .little));
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

fn trainedTables() [tf4fmt.tables_256_bytes]u8 {
    var b: [tf4fmt.tables_256_bytes]u8 = @splat(0);
    std.mem.writeInt(u32, b[0..4], tf4fmt.magic_tables, .little);
    std.mem.writeInt(u32, b[4..8], tf4fmt.format_version, .little);
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

fn cpuDenseRow(
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
) void {
    const n_bc = k / 128;
    const wpb: usize = if (bits == 225) 9 else 8;
    const wpr = n_bc * wpb;
    var x_rot: [256]f64 = undefined;
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
                tf4fmt.decodeRowBlock128(words[woff .. woff + wpb], bits, table_kind, lut, sc, &w128);
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

fn testDenseCase(
    gpu: check.Gpu,
    k4: *tf4.Kernels,
    stream: cuda.Stream,
    tbl_bytes: []const u8,
    table_kind: u32,
    bits: u32,
) !void {
    const a = gpu.gpa;
    const n: u32 = 128;
    const k: u32 = 256;
    const stride: u32 = n + 128;
    const row_off: u32 = 128;
    const sr_split: u32 = 256;
    var tbl_dev = try cuda.DeviceBuffer.fromHost(gpu.d, tbl_bytes);
    defer tbl_dev.free();
    const wc = try tf4fmt.tfq4MatWords(n, k, bits);
    const words = try data.words(a, @intCast(wc));
    defer a.free(words);
    const scales = try data.scales(a, @intCast(fmt.tensorScales(n, k)));
    defer a.free(scales);
    const x = try data.inputs(a, k);
    defer a.free(x);
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(words));
    defer dw.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scales));
    defer ds.free();
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x[0 .. dense_rows * k]));
    defer dx.free();
    const w: tf4.DenseWeight = .{
        .words = dw.ptr,
        .scales = ds.ptr,
        .lut = tbl_dev.ptr + 64,
        .table_kind = table_kind,
        .n = n,
        .k = k,
        .bits = @intCast(bits),
        .row_off = row_off,
        .sr_split = sr_split,
    };
    const lut = std.mem.bytesAsSlice(u16, tbl_bytes[64 .. 64 + (if (table_kind == 3) @as(usize, 1024) else 256) * 8 * 2]);
    var ref0: [128]f64 = undefined;
    cpuDenseRow(words, scales, lut, table_kind, x[0..k], n, k, bits, row_off, sr_split, &ref0);
    var cos_f32: f64 = 0;
    for ([_]bool{ false, true }) |f32_out| {
        const elem: usize = if (f32_out) 4 else 2;
        var serial = try cuda.DeviceBuffer.alloc(gpu.d, dense_rows * stride * elem);
        defer serial.free();
        var batch = try cuda.DeviceBuffer.alloc(gpu.d, serial.len);
        defer batch.free();
        try serial.fill8(0x5a, stream.handle);
        for (0..dense_rows) |row| {
            try k4.dense(stream, w, dx.ptr + row * k * 2, k, serial.ptr + row * stride * elem, stride, 1, f32_out);
        }
        try stream.synchronize();
        const want = try check.download(gpu, serial);
        defer a.free(want);
        var got0: [128]f64 = undefined;
        for (0..n) |c| got0[c] = f32FromBytes(want, c, f32_out);
        const cos = cosine(&got0, &ref0);
        if (f32_out) cos_f32 = cos;
        const req: f64 = if (f32_out) 0.999999 else 0.9999;
        try check.expect(cos >= req, "TFQ4 dense parity kind={d} b={d} f32={}: cos={d:.6}", .{ table_kind, bits, f32_out, cos });
        for (test_widths) |rows| {
            const used = rows * stride * elem;
            try batch.fill8(0x5a, stream.handle);
            try k4.dense(stream, w, dx.ptr, k, batch.ptr, stride, rows, f32_out);
            try stream.synchronize();
            const got = try a.alloc(u8, used);
            defer a.free(got);
            try batch.download(0, got);
            try check.sameBytes("TFQ4 dense serial/window/padding", got, want[0..used]);
        }
    }
    check.pass("TFQ4 dense kind={d} {d}-bit [{d},{d}]: fp32 cos={d:.7}, width-invariant", .{ table_kind, bits, n, k, cos_f32 });
}

fn cpuHybridMoeExpert(
    words: []const u32,
    scales: []const u16,
    meta: fmt.TfpwExpertMeta,
    lut: []align(1) const u16,
    table_kind: u32,
    x_bf16: []const u16,
    softcap: bool,
    out: *[128]f64,
) void {
    var g: [128]f64 = undefined;
    var u: [128]f64 = undefined;
    const wpb: usize = if (meta.b_bits == 225) 9 else 8;
    const nw: usize = 128 * wpb;
    cpuDenseRow(words[@intCast(meta.gate_word_off)..][0..nw], scales[meta.gate_sc_off..][0..1], lut, table_kind, x_bf16, 128, 128, meta.b_bits, 0, 0, &g);
    cpuDenseRow(words[@intCast(meta.up_word_off)..][0..nw], scales[meta.up_sc_off..][0..1], lut, table_kind, x_bf16, 128, 128, meta.b_bits, 0, 0, &u);
    var act_bf16: [128]u16 = undefined;
    for (0..128) |i| {
        const gf: f32 = @floatCast(g[i]);
        const uf: f32 = @floatCast(u[i]);
        const y: f32 = if (softcap) blk: {
            const gc = 30.0 * std.math.tanh(gf / 30.0);
            const uc = 30.0 * std.math.tanh(uf / 30.0);
            break :blk (gc / (1.0 + @exp(-gf))) * uc;
        } else blk: {
            const cl_g = @min(gf, 10.0);
            const cl_u = @min(@max(uf, -10.0), 10.0);
            break :blk (cl_g / (1.0 + @exp(-cl_g))) * cl_u;
        };
        const bits32: u32 = @bitCast(y);
        act_bf16[i] = @intCast((bits32 +% 0x7FFF +% ((bits32 >> 16) & 1)) >> 16);
    }
    cpuDenseRow(words[@intCast(meta.down_word_off)..][0..nw], scales[meta.down_sc_off..][0..1], lut, table_kind, &act_bf16, 128, 128, meta.b_bits, 0, 0, out);
}

fn testHybridMoe(gpu: check.Gpu, kernels: *tf.Kernels, stream: cuda.Stream, tables3: u64, tbl4_bytes: []const u8) !void {
    const a = gpu.gpa;
    const bits_list = [_]u32{ 2, 225, 3, 4, 5 };
    var metas: [5]fmt.TfpwExpertMeta = undefined;
    var wc: u64 = 0;
    for (&metas, bits_list, 0..) |*m, bits, i| {
        const nw = fmt.tensorWords(128, 128, bits);
        m.* = std.mem.zeroes(fmt.TfpwExpertMeta);
        m.b_bits = bits;
        m.gate_word_off = wc;
        m.up_word_off = wc + nw;
        m.down_word_off = wc + 2 * nw;
        m.gate_sc_off = @intCast(i * 3);
        m.up_sc_off = @intCast(i * 3 + 1);
        m.down_sc_off = @intCast(i * 3 + 2);
        wc += 3 * nw;
    }
    const words = try data.words(a, @intCast(wc));
    defer a.free(words);
    const scales = try data.scales(a, 15);
    defer a.free(scales);
    const x = try data.inputs(a, 128);
    defer a.free(x);
    const picks = try a.alloc(i32, moe_rows * 5);
    defer a.free(picks);
    for (0..moe_rows) |r| for (0..5) |s| {
        picks[r * 5 + s] = @intCast((r + s) % 5);
    };
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(words));
    defer dw.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scales));
    defer ds.free();
    var dm = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&metas));
    defer dm.free();
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x[0 .. moe_rows * 128]));
    defer dx.free();
    var dp = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(picks));
    defer dp.free();
    var dt4 = try cuda.DeviceBuffer.fromHost(gpu.d, tbl4_bytes);
    defer dt4.free();
    const layer: tf.MoeLayer = .{
        .weights = dw.ptr,
        .scales = ds.ptr,
        .metas = dm.ptr,
        .tables = tables3,
        .tables4_lut = dt4.ptr + 64,
        .table4_kind = 3,
        .has_tfq4 = true,
        .experts = 5,
        .dims = 128,
        .inter = 128,
    };
    const lut4 = std.mem.bytesAsSlice(u16, tbl4_bytes[64 .. 64 + 1024 * 8 * 2]);
    var pure_layer = layer;
    pure_layer.has_tfq4 = false;
    var min_cos_f32: f64 = 1.0;
    for ([_]bool{ false, true }) |softcap| for ([_]bool{ false, true }) |f32_out| {
        const elem: usize = if (f32_out) 4 else 2;
        const row_bytes = 5 * 128 * elem;
        var serial = try cuda.DeviceBuffer.alloc(gpu.d, moe_rows * row_bytes);
        defer serial.free();
        var batch = try cuda.DeviceBuffer.alloc(gpu.d, serial.len);
        defer batch.free();
        try serial.fill8(0x5a, stream.handle);
        for (0..moe_rows) |row| {
            if (softcap) {
                try kernels.moeSoftcap(stream, layer, dx.ptr + row * 128 * 2, 128, dp.ptr + row * 5 * 4, 5, serial.ptr + row * row_bytes, 5, 1, 5, 30.0, 30.0, f32_out);
            } else {
                try kernels.moe(stream, layer, dx.ptr + row * 128 * 2, 128, dp.ptr + row * 5 * 4, 5, serial.ptr + row * row_bytes, 5, 1, 5, 10.0, 0, f32_out);
            }
        }
        try stream.synchronize();
        const want = try check.download(gpu, serial);
        defer a.free(want);
        for ([_]usize{ 0, 1 }) |slot| {
            var ref: [128]f64 = undefined;
            cpuHybridMoeExpert(words, scales, metas[slot], lut4, 3, x[0..128], softcap, &ref);
            var got_s: [128]f64 = undefined;
            for (0..128) |c| got_s[c] = f32FromBytes(want, slot * 128 + c, f32_out);
            const cos = cosine(&got_s, &ref);
            if (f32_out and cos < min_cos_f32) min_cos_f32 = cos;
            const req: f64 = if (f32_out) 0.99999 else 0.9999;
            try check.expect(cos >= req, "TFQ4 hybrid MoE parity softcap={} b={d} f32={}: cos={d:.6}", .{ softcap, metas[slot].b_bits, f32_out, cos });
        }
        var pure_out = try cuda.DeviceBuffer.alloc(gpu.d, 3 * 128 * elem);
        defer pure_out.free();
        if (softcap) {
            try kernels.moeSoftcap(stream, pure_layer, dx.ptr, 128, dp.ptr + 2 * 4, 5, pure_out.ptr, 3, 1, 3, 30.0, 30.0, f32_out);
        } else {
            try kernels.moe(stream, pure_layer, dx.ptr, 128, dp.ptr + 2 * 4, 5, pure_out.ptr, 3, 1, 3, 10.0, 0, f32_out);
        }
        try stream.synchronize();
        const pure_bytes = try check.download(gpu, pure_out);
        defer a.free(pure_bytes);
        try check.sameBytes("TFQ4 hybrid 3/4/5b slots == pure TFQ3", want[2 * 128 * elem .. 5 * 128 * elem], pure_bytes);
        for (moe_widths) |rows| {
            try batch.fill8(0x5a, stream.handle);
            if (softcap) {
                try kernels.moeSoftcap(stream, layer, dx.ptr, 128, dp.ptr, 5, batch.ptr, 5, rows, 5, 30.0, 30.0, f32_out);
            } else {
                try kernels.moe(stream, layer, dx.ptr, 128, dp.ptr, 5, batch.ptr, 5, rows, 5, 10.0, 0, f32_out);
            }
            try stream.synchronize();
            const got = try a.alloc(u8, rows * row_bytes);
            defer a.free(got);
            try batch.download(0, got);
            try check.sameBytes("TFQ4 hybrid MoE serial/window", got, want[0..got.len]);
        }
    };
    check.pass("TFQ4 hybrid MoE (2/225/3/4/5b): SwiGLU + softcap, fp32 min cos={d:.6}, 3/4/5b bit-exact, width-invariant", .{min_cos_f32});
}

pub fn run(gpu: check.Gpu, kernels: *tf.Kernels, stream: cuda.Stream, tables3: u64) !void {
    var k4 = try tf4.Kernels.load(gpu.d);
    defer k4.unload();
    try k4.reserve(dense_rows * 256);
    const tbl0 = trainedTables();
    _ = try tf4fmt.validatePwV4Tables(&tbl0);
    var tbl2: [tf4fmt.tables_256_bytes]u8 = undefined;
    try tf4fmt.fillTrueE8pTables(&tbl2, 2);
    _ = try tf4fmt.validatePwV4Tables(&tbl2);
    var tbl3: [tf4fmt.tables_1024_bytes]u8 = undefined;
    try tf4fmt.fillTrueE8pTables(&tbl3, 3);
    _ = try tf4fmt.validatePwV4Tables(&tbl3);
    try testDenseCase(gpu, &k4, stream, &tbl0, 0, 2);
    try testDenseCase(gpu, &k4, stream, &tbl2, 2, 2);
    try testDenseCase(gpu, &k4, stream, &tbl3, 3, 2);
    try testDenseCase(gpu, &k4, stream, &tbl3, 3, 225);
    try testHybridMoe(gpu, kernels, stream, tables3, &tbl3);
}
