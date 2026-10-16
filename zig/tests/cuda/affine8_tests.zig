//! MLX affine widths on qlinear's tiles: tiles equal host-widened codes, products in float64 bounds, bits row-stable.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref = @import("quant_ref_tests.zig");
const Gpu = check.Gpu;

const qlinear = cuda.qlinear;
const Rig = quant_ref.RigT;
const bf = quant_ref.helpers.bfOf;
const bf16 = quant_ref.helpers.bf16Of;

/// `codes` packed `bits` wide, low code first, into little-endian words (MLX's layout; a code may cross a word).
fn packCodes(bits: u8, codes: []const u8, words: []u32) void {
    @memset(words, 0);
    for (codes, 0..) |code, i| {
        const bit = i * bits;
        const v: u64 = @as(u64, code) << @intCast(bit & 31);
        words[bit >> 5] |= @truncate(v);
        if ((bit & 31) + bits > 32) words[(bit >> 5) + 1] |= @truncate(v >> 32);
    }
}

const Matrix = struct { codes: []u8, scales: []u16, biases: []u16, n: usize, k: usize, bits: u8 };

/// Random codes of `bits`, and per-64 scales and biases as a converter writes them (bias near -scale * half range).
fn matrix(r: *Rig, n: usize, k: usize, bits: u8) !Matrix {
    const codes = try r.a.alloc(u8, n * k);
    for (codes) |*c| c.* = @intCast(r.rng.uintLessThan(u16, @as(u16, 1) << @intCast(bits)));
    const scales = try r.a.alloc(u16, n * k / 64);
    const biases = try r.a.alloc(u16, n * k / 64);
    const half: f32 = @floatFromInt(@as(u16, 1) << @intCast(bits - 1));
    for (scales, biases) |*s, *b| {
        const sv = (r.rng.float(f32) * 0.9 + 0.1) * 0.02 / half;
        s.* = bf16(sv);
        b.* = bf16(-bf32(s.*) * half * (0.8 + 0.4 * r.rng.float(f32)));
    }
    return .{ .codes = codes, .scales = scales, .biases = biases, .n = n, .k = k, .bits = bits };
}

fn bf32(x: u16) f32 {
    return @floatCast(bf(x));
}

/// The matrix's MLX words at `bits` (its own width, or a wider one its codes fit).
fn mlxWords(r: *Rig, m: Matrix, bits: u8) ![]u32 {
    const out = try r.a.alloc(u32, m.n * m.k * bits / 32);
    packCodes(bits, m.codes, out);
    return out;
}

/// y = x (s q + b)^T in float64, and its bound: a bf16 ulp of |y| and 2^-12 of sum |x| (|s q| + |b|).
fn reference(r: *Rig, x: []const u16, rows: usize, m: Matrix) !struct { want: []f64, tol: []f64 } {
    const want = try r.a.alloc(f64, rows * m.n);
    const tol = try r.a.alloc(f64, rows * m.n);
    for (0..rows) |row| for (0..m.n) |j| {
        var sum: f64 = 0;
        var abs: f64 = 0;
        for (0..m.k) |i| {
            const g = j * (m.k / 64) + i / 64;
            const s = bf(m.scales[g]);
            const b = bf(m.biases[g]);
            const q: f64 = @floatFromInt(m.codes[j * m.k + i]);
            const xv = bf(x[row * m.k + i]);
            sum += xv * (s * q + b);
            abs += @abs(xv) * (@abs(s * q) + @abs(b));
        }
        want[row * m.n + j] = sum;
        tol[row * m.n + j] = @abs(sum) * std.math.pow(f64, 2, -7) + abs * std.math.pow(f64, 2, -12) + 1e-30;
    };
    return .{ .want = want, .tol = tol };
}

/// tf_q35_sums's 64-group sums: fp32, in order.
fn sums(r: *Rig, x: []const u16, rows: usize, k: usize) ![]f32 {
    const out = try r.a.alloc(f32, rows * k / 64);
    for (0..rows) |row| for (0..k / 64) |g| {
        var s: f32 = 0;
        for (0..64) |j| s += bf32(x[row * k + g * 64 + j]);
        out[row * (k / 64) + g] = s;
    };
    return out;
}

const Kernels = struct { a4: *const qlinear.Affine4Kernels, a8: *const qlinear.Affine8Kernels };

/// One width at one shape: tiles against the host-widened codes' tiles, products against float64, rows and paths.
fn width(r: *Rig, k_: Kernels, n: usize, k: usize, bits: u8) !void {
    var name: [128]u8 = undefined;
    const m = try matrix(r, n, k, bits);
    const served = qlinear.servedBits(bits);
    const own = try r.dev(std.mem.sliceAsBytes(try mlxWords(r, m, bits)));
    const wide = try r.dev(std.mem.sliceAsBytes(try mlxWords(r, m, served)));
    const ds = try r.dev(std.mem.sliceAsBytes(m.scales));
    const db = try r.dev(std.mem.sliceAsBytes(m.biases));
    const l4 = qlinear.Affine4.layout(n, k);
    const l8 = qlinear.Affine8.layout(n, k);
    const words_len = if (served == 4) l4.words else l8.words;
    const npad = l4.npad;
    const t_own = try r.zeros(words_len);
    const t_wide = try r.zeros(words_len);
    const ts = try r.zeros(l4.scales);
    const tb = try r.zeros(l4.scales);
    try k_.a8.packWords(r.s, own, bits, n, k, t_own);
    if (served == 4) {
        // the affine-4 pack of the codes widened to 4 bits: the tiles a 4-bit checkpoint of these codes gets
        try k_.a4.pack(r.s, wide, ds, db, .{ .w = t_wide, .s = ts, .b = tb, .n = n, .k = k, .npad = npad });
    } else {
        try k_.a8.packWords(r.s, wide, 8, n, k, t_wide);
        try k_.a8.packTable(r.s, ds, n, k, npad, ts);
        try k_.a8.packTable(r.s, db, n, k, npad, tb);
    }
    try check.sameBytes(try std.fmt.bufPrint(&name, "{d}-bit codes [{d}, {d}]: tiles equal the {d}-bit widened codes' tiles", .{ bits, n, k, served }), try r.back(t_own, words_len), try r.back(t_wide, words_len));
    // the embedding's widen: rows of `bits` codes into rows of the served width, word for word
    const widened = try r.zeros(n * k * served / 8);
    try k_.a8.widen(r.s, own, bits, n, k, widened, served);
    try check.sameBytes(try std.fmt.bufPrint(&name, "{d}-bit rows [{d}, {d}] widened to {d} bits", .{ bits, n, k, served }), try r.back(widened, n * k * served / 8), try r.back(wide, n * k * served / 8));
    // products: 16 rows against float64, fewer rows byte-equal to the 16-row call's first rows
    const most = 16;
    const x = try r.bfs(most * k, 1.0);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const dxs = try r.dev(std.mem.sliceAsBytes(try sums(r, x, most, k)));
    const ref = try reference(r, x, most, m);
    const counts = [_]usize{ 1, 3, 9, most };
    var outs: [counts.len]u64 = undefined;
    for (counts, &outs) |rows, *o| {
        o.* = try r.zeros(rows * n * 2);
        if (served == 4) {
            try k_.a4.decode(r.s, dx, dxs, .{ .w = t_own, .s = ts, .b = tb, .n = n, .k = k, .npad = npad }, o.*, rows);
        } else {
            try k_.a8.decode(r.s, dx, dxs, .{ .w = t_own, .s = ts, .b = tb, .n = n, .k = k, .npad = npad }, o.*, rows);
        }
    }
    try r.close(try std.fmt.bufPrint(&name, "{d}-bit [{d}, {d}] decode, {d} rows", .{ bits, n, k, most }), outs[counts.len - 1], false, ref.want, ref.tol);
    for (counts[0 .. counts.len - 1], outs[0 .. counts.len - 1]) |rows, o| {
        const a = try r.back(outs[counts.len - 1], rows * n * 2);
        const b = try r.back(o, rows * n * 2);
        try check.sameBytes(try std.fmt.bufPrint(&name, "{d}-bit [{d}, {d}]: {d} rows equal the 16-row call's", .{ bits, n, k, rows }), b, a);
    }
    // the 8-bit tiles' two paths: split-K CTAs and lane_gemv's loop over slices give one row the same bits
    if (served == 8) if (k_.a8.split != null and qlinear.splitK(n, k) > 1) {
        var plain = k_.a8.*;
        plain.split = null;
        const o = try r.zeros(most * n * 2);
        try plain.decode(r.s, dx, dxs, .{ .w = t_own, .s = ts, .b = tb, .n = n, .k = k, .npad = npad }, o, most);
        try check.sameBytes(try std.fmt.bufPrint(&name, "{d}-bit [{d}, {d}]: split-K equals lane_gemv's slices", .{ bits, n, k }), try r.back(o, most * n * 2), try r.back(outs[counts.len - 1], most * n * 2));
    };
}

pub fn run(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(17);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    const kk = cuda.kernels;
    const images = [_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack, kk.affine8 };
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
    var a8 = try qlinear.Affine8Kernels.resolve(gpu.d, mods[4], mods[3], sms, false);
    defer a8.deinit();
    const k_: Kernels = .{ .a4 = &a4, .a8 = &a8 };
    // shapes: one K slice and several (split-K), rows short of a tile, a 64-row tile past the 128 padding
    const shapes = [_][2]usize{ .{ 320, 2560 }, .{ 2560, 512 }, .{ 200, 1024 }, .{ 48, 2048 } };
    var cases: usize = 0;
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| for (shapes) |nk| {
        try width(&r, k_, nk[0], nk[1], bits);
        cases += 1;
    };
    check.pass("affine8: {d} cases over 2, 3, 4, 5, 6 and 8 bits: tiles equal the widened codes', products within {d:.3} of the float64 bound, every row count and path byte-equal", .{ cases, r.worst });
}
