//! LFM2-VL's vision kernels against host references, then pictures' features against transformers' modules on the CPU.

const std = @import("std");
const cuda = @import("cuda");
const lfm2 = @import("lfm2");
const nemotron = @import("nemotron");
const check = @import("check.zig");
const Gpu = check.Gpu;
const gm = nemotron.glue_math;
const bf = gm.bf16ToF32;
const vision = lfm2.vision;

const hidden = 1152;
const heads = 16;
const dim = 72;
const ffn = 4304;
const eps: f32 = 1e-6;
const nt = 256; // lfm2_vision.cu's NT
const row_counts = [_]usize{ 1, 7, 308, 1024 }; // 308: a thumbnail's 22 x 14 patches; 1024: a whole tile

// Features against vision.json (torch bf16, CPU), relative L2: twice a CPU emulation of these roundings, reordered.
const norm_tolerance = 0.06; // every row's norm
const sum_tolerance = 0.17; // every row's sum
const row_tolerance = 0.035; // the picked rows whole
const cos_least = 0.998; // each picked row's cosine similarity
const gelu_share = 0.15; // GELU elements off: 2-ulp tanhf and erff flip bf16 roundings in the cancelling negative tail
const linear_share = 0.01; // spread values a bf16 step off: the two plans may sum the products in other orders

extern fn erf(x: f64) f64;

/// warp_sum: xor butterflies over 32 lanes, lane 0's sum.
fn warpSum(v: *[32]f32) f32 {
    var off: usize = 16;
    while (off > 0) : (off >>= 1) {
        var n: [32]f32 = undefined;
        for (&n, 0..) |*x, l| x.* = v[l] + v[l ^ off];
        v.* = n;
    }
    return v[0];
}

/// block_sum over NT lanes: each warp's butterfly, then the warps in order.
fn blockSum(lanes: *[nt]f32) f32 {
    var s: f32 = 0;
    for (0..nt / 32) |w| {
        const ws = warpSum(lanes[w * 32 ..][0..32]);
        s = if (w == 0) ws else s + ws;
    }
    return s;
}

fn layerNorm(x: []const u16, w: []const u16, b: []const u16, y: []u16, rows: usize, width: usize) void {
    var vals: [8 * nt]f32 = undefined;
    for (0..rows) |r| {
        const at = r * width;
        var lanes: [nt]f32 = @splat(0);
        for (0..width) |i| {
            vals[i] = bf(x[at + i]);
            lanes[i % nt] += vals[i];
        }
        const mean = blockSum(&lanes) / @as(f32, @floatFromInt(width));
        lanes = @splat(0);
        for (0..width) |i| {
            vals[i] -= mean;
            lanes[i % nt] += vals[i] * vals[i];
        }
        const rstd = 1.0 / @sqrt(blockSum(&lanes) / @as(f32, @floatFromInt(width)) + eps);
        for (0..width) |i| y[at + i] = gm.f32ToBf16(vals[i] * rstd * bf(w[i]) + bf(b[i]));
    }
}

/// torch's tanh GELU in fp32 as lfm2_vision.cu runs it, tanh correctly rounded.
fn geluTanh(x: f32) f32 {
    const beta: f32 = @floatCast(std.math.sqrt2 * std.math.two_sqrtpi * 0.5);
    const inner = beta * (x + 0.044715 * (x * x * x));
    return 0.5 * x * (1 + @as(f32, @floatCast(std.math.tanh(@as(f64, inner)))));
}

/// torch's exact GELU in fp32 as lfm2_vision.cu runs it, erf correctly rounded.
fn geluErf(x: f32) f32 {
    const alpha: f32 = @floatCast(std.math.sqrt1_2);
    return x * 0.5 * (1 + @as(f32, @floatCast(erf(@as(f64, x * alpha)))));
}

/// tf_lfm2v_pointwise's element: bf16(gelu(v)) through `act`, then bf16(r + v).
fn finish(v: f32, r: ?f32, act: vision.Act) u16 {
    const x = switch (act) {
        .none => v,
        .gelu_tanh => gm.rbf(geluTanh(v)),
        .gelu => gm.rbf(geluErf(v)),
    };
    return gm.f32ToBf16(if (r) |res| res + x else x);
}

/// SDPA in f64 over qkv [rows, 3 H D] -> [rows, H D]; bound: one bf16 rounding of each output, two of each probability.
fn attention(qkv: []const u16, out: []f64, bound: []f64, rows: usize) void {
    const w = 3 * heads * dim;
    const scale = 1.0 / @sqrt(@as(f64, dim));
    var e: [1024]f64 = undefined;
    for (0..heads) |h| for (0..rows) |i| {
        var m = -std.math.inf(f64);
        for (0..rows) |j| {
            var dot: f64 = 0;
            for (0..dim) |d| dot += @as(f64, bf(qkv[i * w + h * dim + d])) * bf(qkv[j * w + heads * dim + h * dim + d]);
            e[j] = dot * scale;
            m = @max(m, e[j]);
        }
        var sum: f64 = 0;
        for (e[0..rows]) |*x| {
            x.* = @exp(x.* - m);
            sum += x.*;
        }
        for (0..dim) |d| {
            var acc: f64 = 0;
            var mag: f64 = 0;
            for (0..rows) |j| {
                const v = bf(qkv[j * w + 2 * heads * dim + h * dim + d]);
                acc += e[j] * v;
                mag += e[j] * @abs(v);
            }
            const at = i * heads * dim + h * dim + d;
            out[at] = acc / sum;
            bound[at] = (@abs(out[at]) + 2 * mag / sum) / 256;
        }
    };
}

const Rig = struct {
    gpu: Gpu,
    a: std.mem.Allocator, // host data, freed together at the end
    s: cuda.Stream,
    k: vision.Kernels,
    at: lfm2.attention.Attention,
    g: vision.Gemms,
    rng: std.Random,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bad: usize = 0,
    cases: usize = 0,

    /// A device copy of host values, landed before any kernel on the rig's stream reads it.
    fn dev(r: *Rig, comptime T: type, host: []const T) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(host));
        try r.bufs.append(r.gpu.gpa, b);
        try r.gpu.ctx.synchronize();
        return b.ptr;
    }

    fn zeros(r: *Rig, comptime T: type, n: usize) !u64 {
        const host = try r.a.alloc(T, n);
        @memset(std.mem.sliceAsBytes(host), 0);
        return r.dev(T, host);
    }

    fn bfs(r: *Rig, n: usize, scale: f32) ![]u16 {
        const out = try r.a.alloc(u16, n);
        for (out) |*v| v.* = gm.f32ToBf16((r.rng.float(f32) * 2 - 1) * scale);
        return out;
    }

    fn f32s(r: *Rig, n: usize, scale: f32) ![]f32 {
        const out = try r.a.alloc(f32, n);
        for (out) |*v| v.* = (r.rng.float(f32) * 2 - 1) * scale;
        return out;
    }

    fn get(r: *Rig, comptime T: type, ptr: u64, n: usize) ![]T {
        try r.s.synchronize();
        const out = try r.a.alloc(T, n);
        try r.gpu.d.check(r.gpu.d.api.cuMemcpyDtoH_v2(out.ptr, ptr, n * @sizeOf(T)), "cuMemcpyDtoH");
        return out;
    }

    /// bf16 rows at `ptr` against `want`: equal (share 0), else `share` of them off by a bf16 step or 2^-20; NaN fails.
    fn near(r: *Rig, what: []const u8, rows: usize, ptr: u64, want: []const u16, share: f64) !void {
        const got = try r.get(u16, ptr, want.len);
        r.cases += 1;
        var worst: f64 = 0;
        var differ: usize = 0;
        var far: usize = 0;
        var live: usize = 0;
        for (got, want) |g, w| {
            const err = @abs(@as(f64, bf(g)) - bf(w));
            worst = @max(worst, err);
            differ += @intFromBool(g != w);
            far += @intFromBool(!(err <= 0x1p-7 * @abs(bf(w)) + 0x1p-20));
            live += @intFromBool(bf(w) != 0);
        }
        std.debug.print("RESULT {s} rows {d}: max abs error {e}, {d} of {d} elements differ, {d} past a bf16 step\n", .{ what, rows, worst, differ, want.len, far });
        if (@as(f64, @floatFromInt(differ)) > share * @as(f64, @floatFromInt(want.len)) or far != 0 or live == 0) r.bad += 1;
    }
};

fn normCases(r: *Rig, rows: usize) !void {
    const x = try r.bfs(rows * hidden, 3.0);
    const w = try r.bfs(hidden, 1.0);
    const b = try r.bfs(hidden, 0.5);
    const yd = try r.zeros(u16, rows * hidden);
    try r.k.layerNorm(r.s, try r.dev(u16, x), try r.dev(u16, w), try r.dev(u16, b), yd, rows, hidden, eps);
    const y = try r.a.alloc(u16, rows * hidden);
    layerNorm(x, w, b, y, rows, hidden);
    try r.near("layer_norm", rows, yd, y, 0);
}

fn pointwiseCases(r: *Rig, rows: usize) !void {
    for ([_]vision.Act{ .none, .gelu_tanh, .gelu }, [_]usize{ hidden, ffn, 2048 }) |act, n| {
        const x = try r.bfs(rows * n, 4.0);
        const res = try r.bfs(rows * n, 2.0);
        const with_r = act == .none;
        const od = if (with_r) try r.dev(u16, res) else try r.zeros(u16, rows * n);
        try r.k.pointwise(r.s, try r.dev(u16, x), if (with_r) od else null, od, rows * n, act);
        const o = try r.a.alloc(u16, rows * n);
        for (o, x, res) |*y, v, rv| y.* = finish(bf(v), if (with_r) bf(rv) else null, act);
        try r.near(if (with_r) "residual add in place" else @tagName(act), rows, od, o, if (with_r) 0 else gelu_share);
    }
    const acc = try r.f32s(rows * hidden, 4.0);
    const b = try r.bfs(hidden, 1.0);
    const od = try r.zeros(u16, rows * hidden);
    try r.k.bias(r.s, try r.dev(f32, acc), try r.dev(u16, b), od, rows, hidden);
    const o = try r.a.alloc(u16, rows * hidden);
    for (o, acc, 0..) |*y, v, i| y.* = gm.f32ToBf16(v + bf(b[i % hidden]));
    try r.near("bias kernel", rows, od, o, 0);
}

/// The tower's and projector's Linear shapes (n outputs, k inputs).
const linears = [_]struct { name: []const u8, n: usize, k: usize }{
    .{ .name = "patch", .n = hidden, .k = 768 },
    .{ .name = "qkv", .n = 3 * hidden, .k = hidden },
    .{ .name = "out_proj", .n = hidden, .k = hidden },
    .{ .name = "fc1", .n = ffn, .k = hidden },
    .{ .name = "fc2", .n = hidden, .k = ffn },
    .{ .name = "linear_1", .n = 2048, .k = 4 * hidden },
    .{ .name = "linear_2", .n = 2048, .k = 2048 },
};

/// Each Linear, epilogue or fallback, against bf16(f32 product + bias): exact products byte for byte, then spread ones.
fn linearCases(r: *Rig, rows: usize) !void {
    for (linears) |l| {
        const name = try std.fmt.allocPrint(r.a, "{s} ({s})", .{ l.name, if (try r.g.fused(rows, l.n, l.k)) "bias in the epilogue" else "bias kernel" });
        const b = try r.bfs(l.n, 1.0);
        const bd = try r.dev(u16, b);
        const yd = try r.zeros(u16, rows * l.n);
        const spare = try r.zeros(f32, rows * l.n);
        // multiples of 2^-6 up to 1/8: every partial sum is exact in f32, so any summation order gives the same product
        if (rows < 1024) {
            const xi = try r.a.alloc(i8, rows * l.k);
            const wi = try r.a.alloc(i8, l.n * l.k);
            for ([_][]i8{ xi, wi }) |m| for (m) |*v| {
                v.* = r.rng.intRangeAtMost(i8, -8, 8);
            };
            try vision.linear(&r.g, &r.k, r.s, spare, try r.dev(u16, try sixtyFourths(r.a, xi)), try r.dev(u16, try sixtyFourths(r.a, wi)), bd, yd, rows, l.n, l.k);
            const want = try r.a.alloc(u16, rows * l.n);
            for (0..rows) |i| for (0..l.n) |j| {
                var dot: i32 = 0;
                for (xi[i * l.k ..][0..l.k], wi[j * l.k ..][0..l.k]) |p, q| dot += @as(i32, p) * q;
                const acc: f32 = @as(f32, @floatFromInt(dot)) * 0x1p-12;
                want[i * l.n + j] = gm.f32ToBf16(acc + bf(b[j]));
            };
            try r.near(name, rows, yd, want, 0);
        }
        const x = try r.bfs(rows * l.k, 2.0);
        const xd = try r.dev(u16, x);
        const wd = try r.dev(u16, try r.bfs(l.n * l.k, 0.05));
        const accd = try r.zeros(f32, rows * l.n);
        try vision.linear(&r.g, &r.k, r.s, spare, xd, wd, bd, yd, rows, l.n, l.k);
        try r.g.run(r.s, xd, wd, accd, rows, l.n, l.k, .f32);
        const acc = try r.get(f32, accd, rows * l.n);
        const got = try r.get(u16, yd, rows * l.n);
        var differ: usize = 0;
        var far: usize = 0;
        for (got, acc, 0..) |g, a, i| {
            const want = gm.f32ToBf16(a + bf(b[i % l.n]));
            differ += @intFromBool(g != want);
            far += @intFromBool(!(@abs(bf(g) - bf(want)) <= 0x1p-7 * @abs(bf(want)) + 0x1p-12));
        }
        r.cases += 1;
        std.debug.print("RESULT {s} rows {d} spread values: {d} of {d} elements differ from the f32 GEMM's sums + bias, {d} past a bf16 step\n", .{ name, rows, differ, got.len, far });
        if (far != 0 or @as(f64, @floatFromInt(differ)) > linear_share * @as(f64, @floatFromInt(got.len))) r.bad += 1;
    }
}

/// `v` as bf16 multiples of 2^-6.
fn sixtyFourths(a: std.mem.Allocator, v: []const i8) ![]u16 {
    const out = try a.alloc(u16, v.len);
    for (out, v) |*o, x| o.* = gm.f32ToBf16(@as(f32, @floatFromInt(x)) * 0x1p-6);
    return out;
}

/// The tower's attention call (Attention.full on its qkv rows) against f64 SDPA within each output's bound.
fn attentionCases(r: *Rig, rows: usize) !void {
    const qkv = try r.bfs(rows * 3 * heads * dim, 2.0);
    const ad = try r.zeros(u16, rows * heads * dim);
    try vision.attention(&r.at, r.s, try r.dev(u16, qkv), ad, rows, heads, dim);
    const got = try r.get(u16, ad, rows * heads * dim);
    const want = try r.a.alloc(f64, rows * heads * dim);
    const bound = try r.a.alloc(f64, rows * heads * dim);
    attention(qkv, want, bound, rows);
    var worst: f64 = 0;
    var far: usize = 0;
    for (got, want, bound) |g, w, b| {
        const err = @abs(bf(g) - w);
        worst = @max(worst, err / b);
        far += @intFromBool(!(err <= b));
    }
    r.cases += 1;
    std.debug.print("RESULT attention rows {d}: worst error {d:.3} of its bound against f64, {d} past it\n", .{ rows, worst, far });
    if (far != 0) r.bad += 1;
}

fn unshuffleCase(r: *Rig, h: usize, w: usize) !void {
    const x = try r.bfs(h * w * hidden, 1.0);
    const yd = try r.zeros(u16, h * w * hidden);
    try r.k.unshuffle(r.s, try r.dev(u16, x), yd, h, w, hidden);
    const y = try r.a.alloc(u16, h * w * hidden);
    for (0..h / 2) |i| for (0..w / 2) |j| for (0..4) |q| {
        const src = ((2 * i + q / 2) * w + 2 * j + q % 2) * hidden;
        @memcpy(y[((i * (w / 2) + j) * 4 + q) * hidden ..][0..hidden], x[src..][0..hidden]);
    };
    try r.near("unshuffle", h * w, yd, y, 0);
}

fn json(a: std.mem.Allocator, io: std.Io, path: []const u8) !std.json.Value {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22));
    return std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
}

fn number(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => std.math.nan(f64),
    };
}

/// Each golden picture through image.zig and the tower: norms and sums of every row, the picked rows whole.
fn features(r: *Rig, model: []const u8) !void {
    const gpa = r.gpu.gpa;
    const io = r.gpu.io;
    var problem: []const u8 = "";
    var cfg = try lfm2.config.Config.load(gpa, io, model, &problem) orelse {
        std.debug.print("FAIL lfm2-vision: {s}\n", .{problem});
        return error.TestFailed;
    };
    defer cfg.deinit(gpa);
    var w = try lfm2.weights.Weights.load(gpa, io, r.gpu.d, model, &cfg);
    defer w.deinit(gpa);
    var v = try vision.Vision.init(gpa, r.gpu.d, &cfg, &(w.vision orelse return error.NoVisionTower));
    defer v.deinit();
    const urls = (try json(r.a, io, "zig/tests/fixtures/lfm2/images.json")).object.get("cases").?.array.items;
    const golden = (try json(r.a, io, "zig/tests/fixtures/lfm2/vision.json")).object.get("cases").?.array.items;
    const pics = try r.a.alloc(lfm2.image.Picture, golden.len);
    const outs = try r.a.alloc([]u16, golden.len);
    for (golden, 0..) |case, ci| {
        const name = case.object.get("name").?.string;
        const url = for (urls) |u| {
            if (std.mem.eql(u8, u.object.get("name").?.string, name)) break u.object.get("url").?.string;
        } else return error.MissingField;
        var rgb = (try lfm2.image.load(r.a, url, &problem)) orelse return error.TestFailed;
        defer rgb.deinit(r.a);
        const pic = &pics[ci];
        pic.* = try lfm2.image.prepare(r.a, rgb);
        const rows: usize = @intCast(case.object.get("rows").?.integer);
        const width = cfg.hidden;
        const outd = try r.zeros(u16, rows * width);
        const got_rows = try v.features(r.s, pic, outd);
        try check.expect(got_rows == rows, "lfm2-vision {s}: {d} feature rows, the golden has {d}", .{ name, got_rows, rows });
        const got = try r.get(u16, outd, rows * width);
        outs[ci] = got;
        var nonfinite: usize = 0;
        for (got) |g| nonfinite += @intFromBool(!std.math.isFinite(bf(g)));
        var norm_err = [2]f64{ 0, 0 }; // squared differences, squared golden values
        var sum_err = [2]f64{ 0, 0 };
        for (case.object.get("norms").?.array.items, case.object.get("sums").?.array.items, 0..) |nv, sv, i| {
            var norm: f64 = 0;
            var sum: f64 = 0;
            for (got[i * width ..][0..width]) |g| {
                norm += @as(f64, bf(g)) * bf(g);
                sum += bf(g);
            }
            norm_err[0] += std.math.pow(f64, @sqrt(norm) - number(nv), 2);
            norm_err[1] += number(nv) * number(nv);
            sum_err[0] += std.math.pow(f64, sum - number(sv), 2);
            sum_err[1] += number(sv) * number(sv);
        }
        const encoded = case.object.get("values").?.string;
        const bytes = try r.a.alloc(u8, try std.base64.standard.Decoder.calcSizeForSlice(encoded));
        try std.base64.standard.Decoder.decode(bytes, encoded);
        var row_err = [2]f64{ 0, 0 };
        var cos_worst: f64 = 1;
        for (case.object.get("picked").?.array.items, 0..) |pv, j| {
            const row: usize = @intCast(pv.integer);
            var dot: f64 = 0;
            var gg: f64 = 0;
            var ww: f64 = 0;
            for (got[row * width ..][0..width], 0..) |g, e| {
                const want: f64 = bf(std.mem.readInt(u16, bytes[(j * width + e) * 2 ..][0..2], .little));
                dot += bf(g) * want;
                gg += @as(f64, bf(g)) * bf(g);
                ww += want * want;
                row_err[0] += std.math.pow(f64, bf(g) - want, 2);
            }
            row_err[1] += ww;
            const cos = dot / @sqrt(gg * ww);
            if (!(cos >= cos_worst)) cos_worst = cos;
        }
        const norms = @sqrt(norm_err[0] / norm_err[1]);
        const sums = @sqrt(sum_err[0] / sum_err[1]);
        const picked = @sqrt(row_err[0] / row_err[1]);
        r.cases += 1;
        std.debug.print("RESULT features {s}: {d} tiles, {d} rows, {d} non-finite; relative L2 error of row norms {d:.4}, row sums {d:.4}, picked rows {d:.4}; least picked cosine {d:.5}\n", .{ name, pic.tiles.len, rows, nonfinite, norms, sums, picked, cos_worst });
        if (nonfinite != 0 or !(norms <= norm_tolerance and sums <= sum_tolerance and picked <= row_tolerance and cos_worst >= cos_least)) r.bad += 1;
    }
    // all pictures again, queued with no host wait between them: each refills the staging while earlier tiles run
    var total: usize = 0;
    for (outs) |o| total += o.len;
    const all = try r.zeros(u16, total);
    var at: usize = 0;
    for (pics, outs) |*pic, o| {
        _ = try v.features(r.s, pic, all + 2 * at);
        at += o.len;
    }
    at = 0;
    for (golden, outs) |case, o| {
        try r.near(try std.fmt.allocPrint(r.a, "features {s} queued back to back", .{case.object.get("name").?.string}), o.len / cfg.hidden, all + 2 * at, o, 0);
        at += o.len;
    }
}

/// The vision kernels on seeded rows against host references; with MODEL, the goldens' features too.
pub fn run(gpu: Gpu, model: ?[]const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x5167_1192);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = try cuda.Stream.init(gpu.d, true), .k = try vision.Kernels.load(gpu.d), .at = try lfm2.attention.Attention.load(gpu.d), .g = try vision.Gemms.init(gpu.gpa, gpu.d), .rng = prng.random() };
    defer {
        r.g.deinit();
        r.at.unload();
        r.k.unload();
        r.s.deinit();
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    for (row_counts) |rows| {
        try normCases(&r, rows);
        try pointwiseCases(&r, rows);
        try linearCases(&r, rows);
        try attentionCases(&r, rows);
    }
    try unshuffleCase(&r, 32, 32);
    try unshuffleCase(&r, 22, 14);
    if (model) |m| try features(&r, m) else std.debug.print("SKIP lfm2-vision features: no MODEL\n", .{});
    try check.expect(r.bad == 0, "lfm2-vision: {d} of {d} comparisons past their references", .{ r.bad, r.cases });
    check.pass("lfm2-vision: {d} comparisons within their references (kernels byte-equal, GELU within a bf16 step, attention and features within stated bounds)", .{r.cases});
}
