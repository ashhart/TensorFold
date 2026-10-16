//! FP32 lane output before residual/rank combination: independent FP64, rounding and row/reduction invariance.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const reference = @import("quant_ref_tests.zig");
const Rig = reference.RigT;
const bf = reference.helpers.bfOf;
const bf16 = reference.helpers.bf16Of;
const Mode = cuda.qmmf.Mode;

/// One fixture dispatch starts with fresh destination and reduction poison, including a reused row prefix.
pub fn dispatch(lane: *const cuda.qmmf.Lane, s: cuda.Stream, x: u64, ldx: usize, rows: usize, w: cuda.qmmf.Weight, out: cuda.DeviceBuffer, part: cuda.DeviceBuffer, format: cuda.qmmf.Out) !void {
    try out.fill8(0xff, s.handle);
    try part.fill8(0xff, s.handle);
    try lane.matmulOut(s, x, ldx, rows, w, out.ptr, format, part.ptr);
}

pub fn run(gpu: check.Gpu) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0xf32_5eed);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    for ([_]Mode{ .fp4, .fp8, .mxfp8, .fp8g, .bf16 }) |mode| try projection(&r, &lane, mode);
}

fn projection(r: *Rig, lane: *cuda.qmmf.Lane, mode: Mode) !void {
    const n = 70;
    const k = 1024;
    const m = 256;
    const ldx = k + 64;
    const npad = cuda.fp8.padded(n);
    const a = r.a;
    const codes = try a.alloc(u8, n * k);
    for (codes) |*v| v.* = reference.helpers.fp8CodeOf(r);
    const scales = try a.alloc(u8, n * (k / 16));
    for (scales) |*v| v.* = if (mode == .mxfp8) 121 + r.rng.uintLessThan(u8, 8) else 0x20 + r.rng.uintLessThan(u8, 0x28);
    const inv = try a.alloc(f32, ((n + 127) / 128) * (k / 128));
    for (inv) |*v| v.* = r.rng.float(f32) * 0.01 + 0.001;
    const w16 = try r.bfs(n * k, 0.25);
    const packed16 = try a.alloc(u16, npad * k);
    const tiled_codes = std.mem.sliceAsBytes(packed16);
    const global: f32 = 0.0123;
    var w: cuda.qmmf.Weight = undefined;
    switch (mode) {
        .bf16 => {
            cuda.qmmf.packBf16(packed16, w16, n, k);
            w = cuda.modelopt.bf16Weight(try r.dev(tiled_codes), n, k);
        },
        .fp4 => {
            const words = try a.alloc(u32, npad * k / 8);
            cuda.nvfp4.packWords(words, codes[0 .. n * k / 2], n, k);
            const tiles = try a.alloc(u8, npad * (k / 16));
            cuda.nvfp4.packScales(tiles, scales, n, k);
            w = cuda.nvfp4.weight(try r.dev(std.mem.sliceAsBytes(words)), try r.dev(tiles), global, n, k);
        },
        .fp8, .fp8g, .mxfp8 => {
            cuda.fp8.packCodes(tiled_codes[0 .. npad * k], codes, n, k);
            const dw = try r.dev(tiled_codes[0 .. npad * k]);
            w = switch (mode) {
                .fp8 => cuda.modelopt.fp8Weight(dw, global, n, k),
                .fp8g => blk: {
                    const tiles = try a.alloc(f32, npad * (k / 64));
                    cuda.fp8.packScales(tiles, inv, n, k);
                    break :blk cuda.fp8.weight(dw, try r.dev(std.mem.sliceAsBytes(tiles)), n, k);
                },
                .mxfp8 => blk: {
                    const tiles = try a.alloc(u8, npad * (k / 32));
                    cuda.mxfp8.packScales(tiles, scales[0 .. n * k / 32], n, k);
                    break :blk cuda.mxfp8.weight(dw, try r.dev(tiles), n, k);
                },
                else => unreachable,
            };
        },
    }
    const x = try a.alloc(u16, m * ldx);
    @memset(x, 0x7fc0);
    for (0..m) |row| {
        for (x[row * ldx ..][0..k]) |*v| v.* = bf16((r.rng.float(f32) - 0.5) * 2);
    }
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    var out_buf = try cuda.DeviceBuffer.alloc(r.gpu.d, m * n * 4);
    defer out_buf.free();
    var bf16_buf = try cuda.DeviceBuffer.alloc(r.gpu.d, m * n * 2);
    defer bf16_buf.free();
    var part_buf = try cuda.DeviceBuffer.alloc(r.gpu.d, 8 * m * n * 4);
    defer part_buf.free();
    const out = out_buf.ptr;
    const out_bf16 = bf16_buf.ptr;
    const want = try a.alloc(f64, m * n);
    const tol = try a.alloc(f64, m * n);
    for (0..m) |row| for (0..n) |col| {
        var sum: f64 = 0;
        var magnitude: f64 = 0;
        for (0..k) |i| {
            const code = codes[col * k + i];
            const weight: f64 = switch (mode) {
                .bf16 => bf(w16[col * k + i]),
                .fp8 => core.quant.fp8.e4m3(code) * global,
                .fp8g => core.quant.fp8.e4m3(code) * inv[(col / 128) * (k / 128) + i / 128],
                .mxfp8 => core.quant.fp8.e4m3(code) * std.math.ldexp(@as(f64, 1), @as(i32, scales[col * (k / 32) + i / 32]) - 127),
                .fp4 => blk: {
                    const byte = codes[col * (k / 2) + i / 2];
                    break :blk core.quant.nvfp4.e2m1(if (i % 2 == 0) byte & 15 else byte >> 4) * core.quant.fp8.e4m3(scales[col * (k / 16) + i / 16]) * global;
                },
            };
            const v = bf(x[row * ldx + i]) * weight;
            sum += v;
            magnitude += @abs(v);
        }
        want[row * n + col] = sum;
        tol[row * n + col] = magnitude * 2e-6 + @abs(sum) * 1e-6 + 1e-30;
    };
    var baseline: ?[]u8 = null;
    for ([_]bool{ true, false }) |clusters| {
        lane.clusters = clusters and (try r.gpu.ctx.capability()) >= 90;
        for ([_]usize{ 256, 1, 7, 17, 33 }) |rows| {
            try dispatch(lane, r.s, dx, ldx, rows, w, out_buf, part_buf, .f32);
            try r.close("FP32 projection", out, true, want[0 .. rows * n], tol[0 .. rows * n]);
            const got = try r.back(out, rows * n * 4);
            if (baseline) |b| try check.sameBytes("FP32 row/reduction invariance", got, b[0..got.len]) else baseline = got;
            try dispatch(lane, r.s, dx, ldx, rows, w, bf16_buf, part_buf, .bf16);
            const rounded = try r.back(out_bf16, rows * n * 2);
            for (0..rows * n) |i| {
                const v = std.mem.readInt(u32, got[i * 4 ..][0..4], .little);
                try check.expect(bf16(@bitCast(v)) == std.mem.readInt(u16, rounded[i * 2 ..][0..2], .little), "FP32 rounds once to the existing BF16 result", .{});
            }
        }
    }
    check.pass("{t}: FP32 outputs preserve FP64 accuracy, BF16 rounding and row/reduction invariance", .{mode});
}
