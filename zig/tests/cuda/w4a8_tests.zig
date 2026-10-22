//! cuda.w4a8's prompt matmul against float64 of its own formula (int8 inputs a 64-input group), and row independence.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref = @import("quant_ref_tests.zig");
const Gpu = check.Gpu;

const qlinear = cuda.qlinear;
const Rig = quant_ref.RigT;
const bf = quant_ref.helpers.bfOf;
const bf16 = quant_ref.helpers.bf16Of;

const Matrix = struct { codes: []u8, scales: []u16, biases: []u16, n: usize, k: usize };

/// Random 4-bit codes, and per-64 scales and biases as a converter writes them (bias near -8 scale).
fn matrix(r: *Rig, n: usize, k: usize) !Matrix {
    const codes = try r.a.alloc(u8, n * k);
    for (codes) |*c| c.* = r.rng.uintLessThan(u8, 16);
    const scales = try r.a.alloc(u16, n * k / 64);
    const biases = try r.a.alloc(u16, n * k / 64);
    for (scales, biases) |*s, *b| {
        const sv = (r.rng.float(f32) * 0.9 + 0.1) * 0.0025;
        s.* = bf16(sv);
        b.* = bf16(-@as(f32, @floatCast(bf(s.*))) * 8 * (0.8 + 0.4 * r.rng.float(f32)));
    }
    return .{ .codes = codes, .scales = scales, .biases = biases, .n = n, .k = k };
}

/// __float2int_rn's rounding: to the nearest integer, a tie to the even one.
fn nearestEven(y: f32) f64 {
    const fl = @floor(y);
    const d = y - fl;
    const up = d > 0.5 or (d == 0.5 and @mod(fl, 2) != 0);
    return if (up) fl + 1 else fl;
}

/// MLX words: eight codes a word, low code first.
fn words(r: *Rig, m: Matrix) ![]u32 {
    const out = try r.a.alloc(u32, m.n * m.k / 8);
    @memset(out, 0);
    for (m.codes, 0..) |code, i| out[i / 8] |= @as(u32, code) << @intCast(4 * (i % 8));
    return out;
}

/// Rows of bf16 inputs: uniform in [-2, 2], one input in 97 twelve times larger, and one all-zero group a row.
fn inputs(r: *Rig, rows: usize, k: usize) ![]u16 {
    const x = try r.a.alloc(u16, rows * k);
    for (x, 0..) |*v, i| {
        var f = (r.rng.float(f32) * 2 - 1) * 2;
        if (r.rng.uintLessThan(u32, 97) == 0) f *= 12;
        if ((i % k) / 64 == (i / k) % (k / 64)) f = 0;
        v.* = bf16(f);
    }
    return x;
}

/// Float64 of the kernel's formula on tf_w4a8_quantize's codes; the bound: fp32 sums and chain, the offset, bf16 out.
fn reference(r: *Rig, x: []const u16, rows: usize, m: Matrix) !struct { want: []f64, tol: []f64 } {
    const groups = m.k / 64;
    const xq = try r.a.alloc(f64, rows * m.k);
    const xa = try r.a.alloc(f64, rows * groups);
    const xs = try r.a.alloc(f64, rows * groups);
    for (0..rows) |row| for (0..groups) |g| {
        var amax: f32 = 0;
        var sum: f64 = 0;
        for (0..64) |j| {
            const v: f32 = @floatCast(bf(x[row * m.k + g * 64 + j]));
            amax = @max(amax, @abs(v));
            sum += v;
        }
        const a: f32 = if (amax > 0) amax / 127.0 else 1.0;
        const inv: f32 = if (amax > 0) 127.0 / amax else 1.0;
        for (0..64) |j| {
            const v: f32 = @floatCast(bf(x[row * m.k + g * 64 + j]));
            xq[row * m.k + g * 64 + j] = nearestEven(if (std.math.isFinite(inv)) v * inv else (v / amax) * 127.0);
        }
        xa[row * groups + g] = a;
        xs[row * groups + g] = sum;
    };
    const want = try r.a.alloc(f64, rows * m.n);
    const tol = try r.a.alloc(f64, rows * m.n);
    for (0..rows) |row| for (0..m.n) |j| {
        var total: f64 = 0;
        var abs: f64 = 0;
        var offset: f64 = 0;
        for (0..groups) |g| {
            var dot: f64 = 0;
            for (0..64) |i| dot += @as(f64, @floatFromInt(m.codes[j * m.k + g * 64 + i])) * xq[row * m.k + g * 64 + i];
            const s = bf(m.scales[j * groups + g]);
            const b = bf(m.biases[j * groups + g]);
            const term = s * xa[row * groups + g] * dot + b * xs[row * groups + g];
            total += term;
            abs += @abs(s * xa[row * groups + g] * dot) + @abs(b * xs[row * groups + g]);
            offset += 2 * @abs(s) * xa[row * groups + g]; // the -12582912 a constant's fp32 rounding: a dot unit at most
        }
        want[row * m.n + j] = total;
        tol[row * m.n + j] = @abs(total) * std.math.pow(f64, 2, -8) + abs * std.math.pow(f64, 2, -18) + offset + 1e-30;
    };
    return .{ .want = want, .tol = tol };
}

fn shape(r: *Rig, a4: *const qlinear.Affine4Kernels, kk: cuda.w4a8.Kernels, n: usize, k: usize) !void {
    var name: [128]u8 = undefined;
    const m = try matrix(r, n, k);
    const l = qlinear.Affine4.layout(n, k);
    const q: qlinear.Affine4 = .{ .w = try r.zeros(l.words), .s = try r.zeros(l.scales), .b = try r.zeros(l.scales), .n = n, .k = k, .npad = l.npad };
    try a4.pack(r.s, try r.dev(std.mem.sliceAsBytes(try words(r, m))), try r.dev(std.mem.sliceAsBytes(m.scales)), try r.dev(std.mem.sliceAsBytes(m.biases)), q);
    const rows = 200; // a 128-row tile and a partial one
    const x = try inputs(r, rows, k);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const ws = try r.zeros(cuda.w4a8.Kernels.scratch(rows, k));
    const out = try r.zeros(rows * n * 2);
    try kk.prompt(r.s, dx, q, out, rows, ws);
    const ref = try reference(r, x, rows, m);
    try r.close(try std.fmt.bufPrint(&name, "w4a8 [{d}, {d}], {d} rows, against float64 of its int8 formula", .{ n, k, rows }), out, false, ref.want, ref.tol);
    // a row's bits are its own: the first 37 rows alone equal their place in the 200
    const few = 37;
    const out_few = try r.zeros(few * n * 2);
    try kk.prompt(r.s, dx, q, out_few, few, ws);
    try check.sameBytes(try std.fmt.bufPrint(&name, "w4a8 [{d}, {d}]: {d} rows equal the 200-row call's", .{ n, k, few }), try r.back(out_few, few * n * 2), try r.back(out, few * n * 2));
}

pub fn run(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(23);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    const kk_ = cuda.kernels;
    const images = [_][]const u8{ kk_.qmm_group, kk_.qmm_prefill, kk_.lane_gemv, kk_.affine4_pack, kk_.w4a8_prefill };
    var mods: [images.len]cuda.Module = undefined;
    var loaded: usize = 0;
    defer for (mods[0..loaded]) |*md| md.unload();
    for (images, &mods) |img, *md| {
        md.* = try cuda.Module.load(gpu.d, img);
        loaded += 1;
    }
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var a4 = try qlinear.Affine4Kernels.resolve(gpu.d, mods[0], mods[1], mods[2], mods[3], sms, false);
    defer a4.deinit();
    const kk = try cuda.w4a8.Kernels.resolve(mods[4]);
    // the 27B's prompt projections' (N, K), and a ragged N past the 128 padding
    const shapes = [_][2]usize{ .{ 6144, 5120 }, .{ 5120, 6144 }, .{ 17408, 5120 }, .{ 5120, 17408 }, .{ 200, 1024 } };
    for (shapes) |nk| try shape(&r, &a4, kk, nk[0], nk[1]);
    check.pass("w4a8: {d} shapes within {d:.3} of the float64 bound of their int8 formula, rows independent", .{ shapes.len, r.worst });
}
