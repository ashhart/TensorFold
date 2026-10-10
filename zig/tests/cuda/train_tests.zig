//! Sliding Weights' training kernels (train.cu) against f64 references: products, packed weights, the change, the loss.

const std = @import("std");
const cuda = @import("cuda");
const nemotron = @import("nemotron");
const check = @import("check.zig");
const mixers = @import("train_mixer_tests.zig");
const Gpu = check.Gpu;
const Rig = mixers.Rig;
const bf = mixers.bf;
const gm = nemotron.glue_math;
const dims = nemotron.slide_dims;
const kern = nemotron.kernels;

pub fn run(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var k = try kern.Kernels.load(gpu.gpa, gpu.io, gpu.ctx, null);
    try k.loadTrain();
    defer k.deinit();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(5);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .t = .{ .f = &k.train, .s = s, .d = gpu.d }, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    for ([_][4]usize{ .{ 37, 192, 96, 1 }, .{ 130, 2688, 2688, 3 }, .{ 5, 64, 8192, 4 } }) |shape| try gemm(&r, shape);
    try dequant(&r, &k, 200, 128);
    try experts(&r, &k);
    try adapter(&r, 23);
    try adapter(&r, 5);
    try lora(&r, 19);
    try loss(&r, 7, 5000);
    try rmsBack(&r, 9);
    try sketches(&r, 11);
    try small(&r);
    check.pass("train: every kernel within {e:.2} of its f64 reference", .{r.worst});
}

/// C (+)= A B^T on the tensor cores, its K cut into slices that add in: in slice order, so twice gives the same bits.
fn gemm(r: *Rig, shape: [4]usize) !void {
    const m, const n, const kk, const split = shape;
    const a = try r.bfs(m * kk, 1.0);
    const b = try r.bfs(n * kk, 1.0);
    const c0 = try r.f32s(m * n, -1, 1);
    const want = try r.a.alloc(f64, m * n);
    for (0..m) |i| for (0..n) |j| {
        var sum: f64 = c0[i * n + j];
        for (0..kk) |x| sum += bf(a[i * kk + x]) * bf(b[j * kk + x]);
        want[i * n + j] = sum;
    };
    const da = try r.dev(u16, a);
    const db = try r.dev(u16, b);
    const parts = try r.zeros(f32, split * m * n);
    var c: [2]u64 = undefined;
    for (&c) |*x| {
        x.* = try r.dev(f32, c0);
        try r.t.gemm(da, kk, db, kk, x.*, n, m, n, kk, split, parts);
    }
    var name: [64]u8 = undefined;
    const what = try std.fmt.bufPrint(&name, "gemm {d}x{d}x{d} in {d} slices", .{ m, n, kk, split });
    try r.close(what, c[0], want, 1e-5);
    const first = try r.back(f32, c[0], m * n);
    const second = try r.back(f32, c[1], m * n);
    var moved: usize = 0;
    for (first, second) |x, y| moved += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
    std.debug.print("RESULT {s} twice: {d} of {d} values differ in any bit\n", .{ what, moved, m * n });
    try check.expect(moved == 0, "{s} gives the same bits twice ({d} values differ)", .{ what, moved });
}

/// MLX 4-bit words, scales and biases for `rows` rows of `cols` inputs.
fn mlx(r: *Rig, rows: usize, cols: usize) ![3][]const u8 {
    const words = try r.a.alloc(u32, rows * cols / 8);
    for (words) |*w| w.* = r.rng.int(u32);
    return .{ std.mem.sliceAsBytes(words), std.mem.sliceAsBytes(try r.bfs(rows * cols / 64, 0.05)), std.mem.sliceAsBytes(try r.bfs(rows * cols / 64, 0.05)) };
}

fn value(w: [3][]const u8, cols: usize, row: usize, i: usize) f64 {
    const words = std.mem.bytesAsSlice(u32, @as([]align(4) const u8, @alignCast(w[0])));
    const sb = [2][]align(1) const u16{ std.mem.bytesAsSlice(u16, w[1]), std.mem.bytesAsSlice(u16, w[2]) };
    const q: f64 = @floatFromInt((words[(row * cols + i) / 8] >> @intCast(4 * (i % 8))) & 0xF);
    const g = (row * cols + i) / 64;
    return bf(sb[0][g]) * q + bf(sb[1][g]);
}

/// A projection packed as the loader tiles it (qlinear's Affine4Kernels.pack), dequantized transposed.
fn dequant(r: *Rig, k: *const kern.Kernels, n: usize, kk: usize) !void {
    const w = try mlx(r, n, kk);
    const lay = cuda.qlinear.Affine4.layout(n, kk);
    const q: cuda.qlinear.Affine4 = .{ .w = try r.zeros(u8, lay.words), .s = try r.zeros(u8, lay.scales), .b = try r.zeros(u8, lay.scales), .n = n, .k = kk, .npad = lay.npad };
    try k.affine.pack(r.s, try r.dev(u8, w[0]), try r.dev(u8, w[1]), try r.dev(u8, w[2]), q);
    const out = try r.zeros(u16, kk * n);
    try r.t.dequantT(.{ .affine4 = q }, out);
    const got = try r.back(u16, out, kk * n);
    var bad: usize = 0;
    for (0..kk) |i| for (0..n) |j| {
        bad += @intFromBool(got[i * n + j] != gm.f32ToBf16(@floatCast(value(w, kk, j, i))));
    };
    std.debug.print("RESULT dequant_t {d}x{d}: {d} of {d} values differ from the MLX dequantization\n", .{ n, kk, bad, n * kk });
    try check.expect(bad == 0, "dequant_t reads the tiled layout as MLX's codes", .{});
}

/// Pairs' input gradients through packed experts (tf_experts_pack's blocks), plan item by plan item.
fn experts(r: *Rig, k: *const kern.Kernels) !void {
    const count = 3;
    const n = 64;
    const kk = 128;
    const w = try mlx(r, count * n, kk);
    const nb = n / 32;
    const kg = kk / 64;
    const packed_w = try r.zeros(u32, count * nb * kg * 288);
    try k.experts.pack(r.s, try r.dev(u8, w[0]), try r.dev(u8, w[1]), try r.dev(u8, w[2]), packed_w, count, n, kk);
    const pick = [_]usize{ 0, 2, 2, 1, 0, 2, 1 };
    const items = [_]i32{ 0, 0, 2, 1, 2, 2, 2, 4, 3 };
    const members = [_]i32{ 0, 4, 3, 6, 1, 2, 5 };
    const x = try r.f32s(pick.len * n, -1, 1);
    const want = try r.a.alloc(f64, pick.len * kk);
    for (pick, 0..) |e, p| for (0..kk) |i| {
        var sum: f64 = 0;
        for (0..n) |j| sum += x[p * n + j] * value(w, kk, e * n + j, i);
        want[p * kk + i] = sum;
    };
    const out = try r.zeros(f32, pick.len * kk);
    try r.t.expertsBack(try r.dev(f32, x), n, packed_w, try r.dev(i32, &items), try r.dev(i32, &[_]i32{3}), try r.dev(i32, &members), out, kk, n, kk, 3);
    try r.close("experts back", out, want, 1e-5);
}

/// The forward's change: xa over every block in use, then y += scale xa b where each block is open on the row.
fn adapter(r: *Rig, rows: usize) !void {
    const in = 3712;
    const out = 2688;
    const stride = 8 * 1856;
    const rank = 48;
    const x = try r.bfs(rows * stride, 1.0);
    const a = try r.f32s(dims.max_rank * in, -0.02, 0.02);
    const b = try r.f32s(dims.max_rank * out, -0.01, 0.01);
    var tau: [dims.max_blocks]f32 = @splat(dims.shut);
    tau[0] = -std.math.inf(f32);
    tau[1] = 0;
    const xa = try r.a.alloc(f64, rows * rank);
    for (0..rows) |t| for (0..rank) |q| {
        var sum: f64 = 0;
        var n: f64 = 0;
        for (0..in) |i| {
            sum += bf(x[t * stride + 6 * 1856 + i]) * a[q * in + i];
            n += bf(x[t * stride + 6 * 1856 + i]) * bf(x[t * stride + 6 * 1856 + i]);
        }
        xa[t * rank + q] = sum;
        if (q % 16 == 15) {
            const bl = q / 16;
            const open = n * dims.unit > 0 and xa[t * rank + bl * 16] >= tau[bl] * @sqrt(n * dims.unit);
            if (!open) for (xa[t * rank + bl * 16 ..][0..16]) |*v| {
                v.* = 0;
            };
        }
    };
    const want = try r.a.alloc(f64, rows * out);
    for (0..rows) |t| for (0..out) |j| {
        var sum: f64 = 0;
        for (0..rank) |q| sum += xa[t * rank + q] * b[q * out + j];
        want[t * out + j] = dims.scale * sum;
    };
    var word = try nemotron.sites.Shared.init(r.gpu.d, u32, 1);
    defer word.free();
    word.slice(u32, 1)[0] = rank;
    const ad: nemotron.weights.Adapter = .{ .a = try r.dev(f32, a), .b = try r.dev(f32, b), .tau = try r.dev(f32, &tau), .rank = word.dev, .xa = try r.zeros(f32, rows * dims.max_rank), .xn = try r.zeros(f32, rows), .in = in, .out = out };
    const y = try r.zeros(f32, rows * 8 * out);
    try r.t.adapt(ad, try r.dev(u16, x) + 6 * 1856 * 2, stride, y, true, out, 8, 6, rows);
    const got = try r.back(f32, y, rows * 8 * out);
    const rowsd = try r.a.alloc(f64, rows * out);
    for (0..rows) |t| for (0..out) |j| {
        rowsd[t * out + j] = got[(t * 8 + 6) * out + j];
    };
    const picked = try r.dev(f32, try toF32(r, rowsd));
    var name: [64]u8 = undefined;
    try r.close(try std.fmt.bufPrint(&name, "the change's forward at {d} rows", .{rows}), picked, want, 1e-4);
}

fn toF32(r: *Rig, xs: []const f64) ![]f32 {
    const out = try r.a.alloc(f32, xs.len);
    for (out, xs) |*o, x| o.* = @floatCast(x);
    return out;
}

/// The change's backward: gates, the open block's gradient, and every block's part of its site's input gradient.
fn lora(r: *Rig, rows: usize) !void {
    const in = 4096;
    const out = 2688;
    const rank = 48;
    const first = 32;
    const a = try r.f32s(dims.max_rank * in, -0.02, 0.02);
    const b = try r.f32s(dims.max_rank * out, -0.01, 0.01);
    const xa0 = try r.f32s(rows * dims.max_rank, -1, 1);
    const xn = try r.f32s(rows, 0.5, 2);
    var tau: [dims.max_blocks]f32 = @splat(dims.shut);
    tau[0] = -std.math.inf(f32);
    tau[1] = 0;
    tau[2] = -std.math.inf(f32);
    const g = try r.f32s(rows * out, -1, 1);
    const db0 = try r.f32s(16 * out, -1, 1);
    const dx0 = try r.f32s(rows * in, -1, 1);
    const xa = try r.a.alloc(f64, rows * rank);
    const gates = try r.a.alloc(f64, rows * 3);
    for (0..rows) |t| for (0..3) |bl| {
        const open = xn[t] * dims.unit > 0 and xa0[t * dims.max_rank + bl * 16] >= tau[bl] * @sqrt(xn[t] * dims.unit);
        gates[t * 3 + bl] = if (open) 1 else 0;
        for (0..16) |q| xa[t * rank + bl * 16 + q] = if (open) xa0[t * dims.max_rank + bl * 16 + q] else 0;
    };
    const db = try r.a.alloc(f64, 16 * out);
    for (0..16) |q| for (0..out) |j| {
        var sum: f64 = 0;
        for (0..rows) |t| sum += xa[t * rank + first + q] * g[t * out + j];
        db[q * out + j] = db0[q * out + j] + dims.scale * sum;
    };
    const dx = try r.a.alloc(f64, rows * in);
    for (0..rows) |t| {
        var dxa: [rank]f64 = undefined;
        for (0..rank) |q| {
            var sum: f64 = 0;
            for (0..out) |j| sum += g[t * out + j] * b[q * out + j];
            dxa[q] = dims.scale * sum * gates[t * 3 + q / 16];
        }
        for (0..in) |i| {
            var sum: f64 = dx0[t * in + i];
            for (0..rank) |q| sum += dxa[q] * a[q * in + i];
            dx[t * in + i] = sum;
        }
    }
    var word = try nemotron.sites.Shared.init(r.gpu.d, u32, 1);
    defer word.free();
    const ad: nemotron.weights.Adapter = .{ .a = try r.dev(f32, a), .b = try r.dev(f32, b), .tau = try r.dev(f32, &tau), .rank = word.dev, .xa = 0, .xn = 0, .in = in, .out = out };
    const xad = try r.dev(f32, xa0);
    const gated = try r.zeros(f32, rows * dims.max_blocks);
    const gd = try r.dev(f32, g);
    const dbd = try r.dev(f32, db0);
    const dxa = try r.zeros(f32, rows * dims.max_rank);
    const dxd = try r.dev(f32, dx0);
    try r.t.gate(ad, xad, try r.dev(f32, xn), gated, rows, rank);
    try r.t.loraDb(xad, gd, dbd, rows, out, first);
    try r.t.loraDxa(gd, ad.b, gated, dxa, rows, out, rank);
    try r.t.loraDx(dxa, ad.a, dxd, in, rows, in, rank);
    try r.close("the open block's gradient", dbd, db, 1e-5);
    try r.close("the change's input gradient", dxd, dx, 1e-5);
}

/// Each row's loss and target probability, and its logits replaced by (p - onehot) weight.
fn loss(r: *Rig, rows: usize, vocab: usize) !void {
    const logits = try r.bfs(rows * vocab, 6.0);
    const targets = try r.a.alloc(u32, rows);
    for (targets) |*t| t.* = r.rng.uintLessThan(u32, @intCast(vocab));
    const wts = try r.f32s(rows, 0, 1);
    const stats = try r.a.alloc(f64, rows * 2);
    const grad = try r.a.alloc(f64, rows * vocab);
    for (0..rows) |t| {
        var top: f64 = -std.math.inf(f64);
        for (logits[t * vocab ..][0..vocab]) |v| top = @max(top, bf(v));
        var sum: f64 = 0;
        for (logits[t * vocab ..][0..vocab]) |v| sum += @exp(bf(v) - top);
        const lt = bf(logits[t * vocab + targets[t]]);
        stats[2 * t] = @log(sum) + top - lt;
        stats[2 * t + 1] = @exp(lt - top) / sum;
        for (0..vocab) |v| grad[t * vocab + v] = (@exp(bf(logits[t * vocab + v]) - top) / sum - @as(f64, if (v == targets[t]) 1 else 0)) * wts[t];
    }
    const ld = try r.dev(u16, logits);
    const sd = try r.zeros(f32, rows * 2);
    try r.t.softmax(ld, try r.dev(u32, targets), try r.dev(f32, wts), sd, rows, vocab);
    try r.close("softmax loss and probability", sd, stats, 1e-5);
    const got = try r.back(u16, ld, rows * vocab);
    var diff: f64 = 0;
    for (got, grad) |x, y| diff = @max(diff, @abs(bf(x) - y));
    std.debug.print("RESULT softmax gradient: largest difference {e:.3} (bf16 grain at 1: 3.9e-3)\n", .{diff});
    try check.expect(diff < 4e-3, "the logits' gradient within bf16's grain", .{});
}

/// The input RMS norm undone onto the residual's gradient.
fn rmsBack(r: *Rig, rows: usize) !void {
    const dim = 2688;
    const h = try r.bfs(rows * dim, 2.0);
    const w = try r.bfs(dim, 1.0);
    const dx = try r.f32s(rows * dim, -1, 1);
    const g0 = try r.f32s(rows * dim, -1, 1);
    const want = try r.a.alloc(f64, rows * dim);
    for (0..rows) |t| {
        var sq: f64 = 0;
        var dot: f64 = 0;
        for (0..dim) |j| {
            sq += bf(h[t * dim + j]) * bf(h[t * dim + j]);
            dot += bf(w[j]) * dx[t * dim + j] * bf(h[t * dim + j]);
        }
        const s = 1 / @sqrt(sq / dim + 1e-5);
        for (0..dim) |j| want[t * dim + j] = g0[t * dim + j] + s * bf(w[j]) * dx[t * dim + j] - s * s * s * dot / dim * bf(h[t * dim + j]);
    }
    const g = try r.dev(f32, g0);
    try r.t.rmsBack(try r.dev(u16, h), try r.dev(u16, w), try r.dev(f32, dx), g, rows, dim, 1e-5);
    try r.close("rms norm back", g, want, 1e-5);
}

fn coin(row: u32, j: u32, seed: u32) f64 {
    var h = row *% 0x9E3779B9 +% j *% 0x7FEB352D +% seed;
    h ^= h >> 16;
    h *%= 0x7FEB352D;
    h ^= h >> 15;
    h *%= 0x846CA68B;
    h ^= h >> 16;
    return if (h & 1 == 1) 1 else -1;
}

/// Site inputs into random sketches, and their unit rows along candidate directions.
fn sketches(r: *Rig, rows: usize) !void {
    const in = 3712;
    const stride = 8 * 1856;
    const k = 64;
    const x = try r.bfs(rows * stride, 1.0);
    const y0 = try r.f32s(k * in, -1, 1);
    const first = 1000;
    const want = try r.a.alloc(f64, k * in);
    for (0..k) |j| for (0..in) |i| {
        var sum: f64 = 0;
        for (0..rows) |t| sum += coin(@intCast(first + t), @intCast(j), 2) * bf(x[t * stride + i]);
        want[j * in + i] = y0[j * in + i] + 1.5 * sum;
    };
    const y = try r.dev(f32, y0);
    const xd = try r.dev(u16, x);
    try r.t.sketch(xd, stride, y, rows, in, k, first, 2, 1.5);
    try r.close("sketch", y, want, 1e-5);
    const f = try r.f32s(k * in, -1, 1);
    const proj = try r.a.alloc(f64, rows * k);
    for (0..rows) |t| for (0..k) |j| {
        var sum: f64 = 0;
        var n: f64 = 0;
        for (0..in) |i| {
            sum += f[j * in + i] * bf(x[t * stride + i]);
            n += bf(x[t * stride + i]) * bf(x[t * stride + i]);
        }
        proj[t * k + j] = sum / @sqrt(n);
    };
    const p = try r.zeros(f32, rows * k);
    try r.t.project(xd, stride, try r.dev(f32, f), p, rows, in, k);
    try r.close("project", p, proj, 1e-5);
}

/// relu^2 undone from its output, the pairs' gradients in and out, and Adam.
fn small(r: *Rig) !void {
    const rows = 5;
    const slots = 8;
    const dim = 2688;
    const act = try r.a.alloc(u16, rows * slots * 64);
    for (act) |*v| v.* = gm.f32ToBf16(@max(0, r.rng.float(f32) * 2 - 0.5));
    const da = try r.f32s(act.len, -1, 1);
    const du = try r.a.alloc(f64, act.len);
    for (du, act, da) |*o, v, d| o.* = d * 2 * @sqrt(bf(v));
    const dud = try r.zeros(f32, act.len);
    try r.t.relu2Back(try r.dev(u16, act), try r.dev(f32, da), dud, act.len);
    try r.close("relu^2 back", dud, du, 1e-6);
    const wt = try r.f32s(rows * slots, 0, 1);
    const g = try r.f32s(rows * dim, -1, 1);
    const dy = try r.a.alloc(f64, rows * slots * dim);
    for (0..rows * slots) |p| for (0..dim) |j| {
        dy[p * dim + j] = wt[p] * g[p / slots * dim + j];
    };
    const dyd = try r.zeros(f32, dy.len);
    try r.t.pairsIn(try r.dev(f32, wt), try r.dev(f32, g), dyd, rows * slots, slots, dim);
    try r.close("pairs in", dyd, dy, 1e-6);
    const dxp = try r.f32s(rows * slots * dim, -1, 1);
    const dx0 = try r.f32s(rows * dim, -1, 1);
    const dx = try r.a.alloc(f64, rows * dim);
    for (0..rows) |t| for (0..dim) |j| {
        var sum: f64 = dx0[t * dim + j];
        for (0..slots) |q| sum += dxp[(t * slots + q) * dim + j];
        dx[t * dim + j] = sum;
    };
    const dxd = try r.dev(f32, dx0);
    try r.t.pairsOut(try r.dev(f32, dxp), dxd, rows, slots, dim);
    try r.close("pairs out", dxd, dx, 1e-6);
    const n = 1000;
    const p0 = try r.f32s(n, -1, 1);
    const g0 = try r.f32s(n, -1, 1);
    const m0 = try r.f32s(n, -0.1, 0.1);
    const v0 = try r.f32s(n, 0, 0.1);
    const h = dims.hyper;
    const corr = [2]f32{ 2, 3 };
    const want = try r.a.alloc(f64, n);
    for (want, p0, g0, m0, v0) |*o, p, gg, m, v| {
        const mi = h[1] * m + (1 - h[1]) * gg;
        const vi = h[2] * v + (1 - h[2]) * gg * gg;
        o.* = p - h[0] * (mi * corr[0]) / (@sqrt(vi * corr[1]) + h[3]);
    }
    const pd = try r.dev(f32, p0);
    try r.t.adam(pd, try r.dev(f32, g0), try r.dev(f32, m0), try r.dev(f32, v0), n, h, corr);
    try r.close("adam", pd, want, 1e-5);
}
