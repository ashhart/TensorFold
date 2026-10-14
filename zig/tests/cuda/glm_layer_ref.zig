//! glm.layer.kda and glm.layer.mlp's float64 part: every stage of the layer recomputed in f64 from its fixture inputs.
const std = @import("std");
const cuda = @import("cuda");
const ref = @import("glm_kda_ref.zig");
const exl3fmt = @import("core").exl3_format;
const lay = cuda.glm_layer;

const Obj = std.json.ObjectMap;
const id = "glm.layer.kda";

/// One EXL3 linear's host bytes: words (the engine's strips layout, or stored), fp16 suh/svh, shape and width.
pub const Lin = struct { words: []const u8, strips: bool = true, suh: []const u8, svh: []const u8, k: usize, n: usize, halves: u16 };

/// One plain bf16 matrix [n, k].
pub const Mat = struct { w: []const u8, n: usize, k: usize };

pub const KdaHost = struct {
    qkv: Lin,
    o: Lin,
    fa: Mat,
    ga: Mat,
    b: Mat,
    fb: Mat,
    gb: Mat,
    conv: []const u8,
    a_log: []const u8,
    dt_bias: []const u8,
    norm: []const u8,
    heads: u32,
    fa_off: u32,
    ga_off: u32,
    b_off: u32,
    p_stride: u32,
};

/// The routed experts: per projection one file of every expert's stored trellis, byte offsets, widths, scales.
pub const ExpertsHost = struct {
    proj: [3][]const u8,
    offs: [3][]u64,
    k2: [3][]i32,
    suh_g: []const u8,
    suh_u: []const u8,
    svh_g: []const u8,
    svh_u: []const u8,
    suh_d: []const u8,
    svh_d: []const u8,
    count: usize,
    dims: usize,
    width: usize,

    /// Expert e's projection j (gate, up, down) as a stored-layout linear.
    pub fn lin(x: *const ExpertsHost, j: usize, e: usize) Lin {
        const k = if (j == 2) x.width else x.dims;
        const n = if (j == 2) x.dims else x.width;
        const halves: u16 = @intCast(x.k2[j][e]);
        const len = (k / 16) * (n / 16) * @as(usize, halves) * 16;
        const scales = [3][2][]const u8{ .{ x.suh_g, x.svh_g }, .{ x.suh_u, x.svh_u }, .{ x.suh_d, x.svh_d } };
        return .{ .words = x.proj[j][x.offs[j][e]..][0..len], .strips = false, .suh = scales[j][0][e * k * 2 ..][0 .. k * 2], .svh = scales[j][1][e * n * 2 ..][0 .. n * 2], .k = k, .n = n, .halves = halves };
    }
};

/// Rank 0's layer tensors from the fixture (manifest section's "weights"), read once; ffn MoE or dense MLP.
pub const Host = struct {
    attn_hc: [3][]const u8,
    ffn_hc: [3][]const u8,
    in_norm: []const u8,
    post_norm: []const u8,
    router: []const u8,
    bias: []const u8,
    kda: ?KdaHost,
    ex: ExpertsHost, // undefined for a dense layer
    sh: [3]Lin, // the shared expert's gate, up, down; a dense layer's MLP
    dense: bool,

    pub fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8, o: Obj) !Host {
        const R = struct {
            a: std.mem.Allocator,
            io: std.Io,
            dir: []const u8,
            fn f(s: @This(), obj: Obj, key: []const u8) ![]const u8 {
                return ref.readFile(s.a, s.io, s.dir, ref.str(obj, key));
            }
            fn lin(s: @This(), obj: Obj) !Lin {
                if (!std.mem.eql(u8, ref.str(obj, "codebook"), "mul1")) return error.GlmLayerCodebook;
                return .{ .words = try s.f(obj, "words"), .suh = try s.f(obj, "suh"), .svh = try s.f(obj, "svh"), .k = @intCast(ref.int(obj, "k")), .n = @intCast(ref.int(obj, "n")), .halves = @intCast(ref.int(obj, "k2")) };
            }
            fn mat(s: @This(), obj: Obj) !Mat {
                return .{ .w = try s.f(obj, "w"), .n = @intCast(ref.int(obj, "n")), .k = @intCast(ref.int(obj, "k")) };
            }
            fn hc(s: @This(), obj: Obj) ![3][]const u8 {
                return .{ try s.f(obj, "fn"), try s.f(obj, "base"), try s.f(obj, "scale") };
            }
        };
        const s: R = .{ .a = a, .io = io, .dir = dir };
        const mo = o.get("moe");
        const m = (mo orelse o.get("mlp").?).object;
        const sh = if (mo != null) m.get("shared").?.object else m;
        var h: Host = .{
            .attn_hc = try s.hc(o.get("attn_hc").?.object),
            .ffn_hc = try s.hc(o.get("ffn_hc").?.object),
            .in_norm = try s.f(o, "in_norm"),
            .post_norm = try s.f(o, "post_norm"),
            .router = if (mo != null) try s.f(m, "router") else "",
            .bias = if (mo != null) try s.f(m, "bias") else "",
            .kda = if (o.get("kda")) |kv| kda: { // a DSA layer's fixture has none
                const k = kv.object;
                break :kda .{
                    .qkv = try s.lin(k.get("qkv").?.object),
                    .o = try s.lin(k.get("o").?.object),
                    .fa = try s.mat(k.get("fa").?.object),
                    .ga = try s.mat(k.get("ga").?.object),
                    .b = try s.mat(k.get("b").?.object),
                    .fb = try s.mat(k.get("fb").?.object),
                    .gb = try s.mat(k.get("gb").?.object),
                    .conv = try s.f(k, "conv"),
                    .a_log = try s.f(k, "a_log"),
                    .dt_bias = try s.f(k, "dt_bias"),
                    .norm = try s.f(k, "norm"),
                    .heads = @intCast(ref.int(k, "heads")),
                    .fa_off = @intCast(ref.int(k, "fa_off")),
                    .ga_off = @intCast(ref.int(k, "ga_off")),
                    .b_off = @intCast(ref.int(k, "b_off")),
                    .p_stride = @intCast(ref.int(k, "p_stride")),
                };
            } else null,
            .ex = undefined,
            .sh = .{ try s.lin(sh.get("gate").?.object), try s.lin(sh.get("up").?.object), try s.lin(sh.get("down").?.object) },
            .dense = mo == null,
        };
        if (h.dense) return h;
        const ex = m.get("experts").?.object;
        if (ref.int(ex, "cb") != 2) return error.GlmLayerCodebook;
        const count: usize = @intCast(ref.int(ex, "count"));
        h.ex = .{ .proj = undefined, .offs = undefined, .k2 = undefined, .suh_g = try s.f(ex, "suh_g"), .suh_u = try s.f(ex, "suh_u"), .svh_g = try s.f(ex, "svh_g"), .svh_u = try s.f(ex, "svh_u"), .suh_d = try s.f(ex, "suh_d"), .svh_d = try s.f(ex, "svh_d"), .count = count, .dims = @intCast(ref.int(ex, "dims")), .width = @intCast(ref.int(ex, "width")) };
        const names = [3][3][]const u8{ .{ "gate", "gate_offs", "gate_k2" }, .{ "up", "up_offs", "up_k2" }, .{ "down", "down_offs", "down_k2" } };
        for (names, 0..) |nm, j| {
            h.ex.proj[j] = try s.f(ex, nm[0]);
            h.ex.offs[j] = try a.alloc(u64, count);
            h.ex.k2[j] = try a.alloc(i32, count);
            for (ex.get(nm[1]).?.array.items, ex.get(nm[2]).?.array.items, 0..) |off, k2, e| {
                h.ex.offs[j][e] = @intCast(off.integer);
                h.ex.k2[j][e] = @intCast(k2.integer);
            }
        }
        return h;
    }
};

pub fn bfs(a: std.mem.Allocator, bytes: []const u8) ![]f64 {
    const u = std.mem.bytesAsSlice(u16, bytes);
    const out = try a.alloc(f64, u.len);
    for (u, 0..) |v, i| out[i] = ref.bf(v);
    return out;
}

pub fn f32s(bytes: []const u8) []const f32 {
    return ref.f32s(bytes);
}

fn f16At(bytes: []const u8, i: usize) f64 {
    return @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little))));
}

fn bf16r(v: f64) f64 {
    const u: u32 = @bitCast(@as(f32, @floatCast(v)));
    return @as(f32, @bitCast((u +% (0x7FFF + ((u >> 16) & 1))) & 0xFFFF0000));
}

fn sigmoid(x: f64) f64 {
    return 1.0 / (1.0 + @exp(-x));
}

/// The worst |got - want| / tol of one stage, NaN counting as infinite.
const Bound = struct {
    worst: f64 = 0,
    at: usize = 0,

    fn add(b: *Bound, got: f64, want: f64, tol: f64, i: usize) void {
        const q = @abs(got - want) / tol;
        if (!(q <= b.worst)) {
            b.worst = if (std.math.isNan(q)) std.math.inf(f64) else q;
            b.at = i;
        }
    }

    fn rel(b: *Bound, got: f64, want: f64, scale: f64, floor: f64, i: usize) void {
        b.add(got, want, @abs(want) * scale + floor, i);
    }
};

/// The stages' verdicts: the first stage past its tolerance fails the window; `line` lists every stage's worst.
pub const Report = struct {
    a: std.mem.Allocator,
    tag: []const u8,
    id: []const u8 = id,
    worst: f64 = 0,
    line: []const u8 = "",

    fn stage(rp: *Report, name: []const u8, b: Bound) !void {
        if (b.worst > 1.0) {
            std.debug.print("FAIL {s} {s} {s} f64: worst {d:.4} of tolerance at element {d}\n", .{ rp.id, rp.tag, name, b.worst, b.at });
            return error.TestFailed;
        }
        rp.worst = @max(rp.worst, b.worst);
        rp.line = try std.fmt.allocPrint(rp.a, "{s} {s} {d:.3};", .{ rp.line, name, b.worst });
    }
};

/// y [m <= 16, n] of x [m, k] through an EXL3 linear (exl3_dense_tests' reference) and each output's |svh| sum |xh wq|.
pub fn exl3Ref(a: std.mem.Allocator, l: Lin, x: []const f64, m: usize) !struct { want: []f64, s1: []f64 } {
    if (m == 0 or m > 16) return error.GlmLayerRows;
    const kt = l.k / 16;
    const nt = l.n / 16;
    const tw: usize = @as(usize, l.halves) * 8;
    const src: []const i16 = @alignCast(std.mem.bytesAsSlice(i16, l.words));
    var tr = src;
    if (l.strips) { // [N/128, K/16, 8, W] back to the stored [K/16, N/16, W]
        const st = try a.alloc(i16, src.len);
        for (0..nt / 8) |nb| for (0..kt) |ki| for (0..8) |j| {
            @memcpy(st[(ki * nt + nb * 8 + j) * tw ..][0..tw], src[((nb * kt + ki) * 8 + j) * tw ..][0..tw]);
        };
        tr = st;
    }
    const wq = try a.alloc(f16, l.k * l.n);
    exl3fmt.unpack(.{ .halves = l.halves }, .mul1, tr, kt, nt, wq);
    const xs = try a.alloc(f64, m * l.k);
    for (0..m) |row| for (0..l.k) |i| {
        xs[row * l.k + i] = x[row * l.k + i] * f16At(l.suh, i);
    };
    exl3fmt.rotate(xs, m, l.k, 1);
    const ypre = try a.alloc(f64, m * l.n);
    const s1 = try a.alloc(f64, m * l.n);
    @memset(ypre, 0);
    @memset(s1, 0);
    var xv: [16]f64 = undefined;
    for (0..l.k) |ki| {
        for (0..m) |row| xv[row] = @floatCast(@as(f16, @floatCast(xs[row * l.k + ki])));
        for (0..l.n) |col| {
            const wv: f64 = @floatCast(wq[ki * l.n + col]);
            for (0..m) |row| {
                ypre[col * m + row] += xv[row] * wv;
                s1[col * m + row] += @abs(xv[row] * wv);
            }
        }
    }
    exl3fmt.rotate(ypre, l.n, m, 0);
    const want = try a.alloc(f64, m * l.n);
    const s = try a.alloc(f64, m * l.n);
    for (0..l.n) |col| {
        const sv = f16At(l.svh, col);
        for (0..m) |row| {
            want[row * l.n + col] = ypre[col * m + row] * sv;
            s[row * l.n + col] = s1[col * m + row] * @abs(sv);
        }
    }
    return .{ .want = want, .s1 = s };
}

/// The rows the EXL3 stages recompute: every decode row, four rows of a prompt chunk.
pub fn samples(rows: usize) []const usize {
    const all = [_]usize{ 0, 1, 2, 3, 4, 5, 6, 7 };
    if (rows <= all.len) return all[0..rows];
    const S = struct {
        var four: [4]usize = undefined;
    };
    S.four = .{ 0, 1, rows / 2, rows - 1 };
    return &S.four;
}

/// x's chosen rows [m, k].
fn pick(a: std.mem.Allocator, x: []const f64, k: usize, stride: usize, off: usize, rows: []const usize) ![]f64 {
    const out = try a.alloc(f64, rows.len * k);
    for (rows, 0..) |row, i| @memcpy(out[i * k ..][0..k], x[row * stride + off ..][0..k]);
    return out;
}

/// An EXL3 stage on sample rows: got (bf16 or fp32 [rows, stride] at column off) vs the f64 reference.
fn exl3Stage(rp: *Report, a: std.mem.Allocator, name: []const u8, l: Lin, x: []const f64, rows: []const usize, got: []const f64, stride: usize, off: usize, f32_out: bool) !void {
    const ref_ = try exl3Ref(a, l, try pick(a, x, l.k, l.k, 0, rows), rows.len);
    var b = Bound{};
    for (rows, 0..) |row, i| for (0..l.n) |col| {
        const want = ref_.want[i * l.n + col];
        const tol = @abs(want) / @as(f64, if (f32_out) 1024 else 128) + ref_.s1[i * l.n + col] / 2048 + 1e-30;
        b.add(got[row * stride + off + col], want, tol, row * stride + off + col);
    };
    try rp.stage(name, b);
}

/// A plain bf16 matmul over every row: x [rows, k] at (xs, xo), w [n, k], got bf16 [rows, n].
fn b16Stage(rp: *Report, a: std.mem.Allocator, name: []const u8, mt: Mat, x: []const f64, xs: usize, xo: usize, got: []const f64, gs: usize, go: usize, rows: usize) !void {
    const wv = try bfs(a, mt.w);
    var b = Bound{};
    for (0..rows) |row| for (0..mt.n) |col| {
        var acc: f64 = 0;
        var s: f64 = 0;
        for (0..mt.k) |j| {
            const t = x[row * xs + xo + j] * wv[col * mt.k + j];
            acc += t;
            s += @abs(t);
        }
        b.add(got[row * gs + go + col], acc, @abs(acc) / 128 + s / 1024 + 1e-30, row * gs + go + col);
    };
    try rp.stage(name, b);
}

/// glue.hc_pre in f64: the 24 mixes, gates, Sinkhorn, the collapsed row's norm (glm_glue_tests' reference).
pub fn hcPreStage(rp: *Report, a: std.mem.Allocator, name: []const u8, c: lay.Config, hc: [3][]const u8, nw_b: []const u8, x: []const f64, rows: usize, normed: []const f64, post: []const f32, comb: []const f32) !void {
    const d: usize = c.d;
    const wide = 4 * d;
    const fnv = f32s(hc[0]);
    const basev = f32s(hc[1]);
    const scv = f32s(hc[2]);
    const nwv = try bfs(a, nw_b);
    const hc_eps: f64 = c.hc_eps;
    var b = Bound{};
    var bp = Bound{};
    var bc = Bound{};
    const cv = try a.alloc(f64, d);
    for (0..rows) |i| {
        var mix: [24]f64 = @splat(0);
        var ss: f64 = 0;
        for (0..wide) |j| {
            const xj = x[i * wide + j];
            ss += xj * xj;
            for (0..24) |m| mix[m] += fnv[m * wide + j] * xj;
        }
        const rinv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(wide)) + c.eps);
        var p: [4]f64 = undefined;
        for (0..4) |sv| {
            p[sv] = sigmoid(mix[sv] * rinv * scv[0] + basev[sv]) + hc_eps;
            bp.rel(post[i * 4 + sv], 2.0 * sigmoid(mix[4 + sv] * rinv * scv[1] + basev[4 + sv]), 1.0 / 1024.0, 1e-6, i * 4 + sv);
        }
        var ce: [4][4]f64 = undefined;
        for (0..4) |r| {
            var cmax: f64 = -std.math.inf(f64);
            for (0..4) |q| {
                ce[r][q] = mix[8 + r * 4 + q] * rinv * scv[2] + basev[8 + r * 4 + q];
                cmax = @max(cmax, ce[r][q]);
            }
            var rs: f64 = 0;
            for (0..4) |q| {
                ce[r][q] = @exp(ce[r][q] - cmax);
                rs += ce[r][q];
            }
            for (0..4) |q| ce[r][q] = ce[r][q] / rs + hc_eps;
        }
        for (0..c.hc_iters) |it| {
            if (it > 0) for (0..4) |r| {
                var rs: f64 = 0;
                for (0..4) |q| rs += ce[r][q];
                for (0..4) |q| ce[r][q] /= rs + hc_eps;
            };
            for (0..4) |q| {
                var cs: f64 = 0;
                for (0..4) |r| cs += ce[r][q];
                for (0..4) |r| ce[r][q] /= cs + hc_eps;
            }
        }
        for (0..4) |r| for (0..4) |q| bc.rel(comb[i * 16 + r * 4 + q], ce[r][q], 1.0 / 512.0, 1e-6, i * 16 + r * 4 + q);
        var cs2: f64 = 0;
        for (0..d) |j| {
            cv[j] = p[0] * x[i * wide + j] + p[1] * x[i * wide + d + j] + p[2] * x[i * wide + 2 * d + j] + p[3] * x[i * wide + 3 * d + j];
            cs2 += cv[j] * cv[j];
        }
        const rinv2 = 1.0 / @sqrt(cs2 / @as(f64, @floatFromInt(d)) + c.eps);
        for (0..d) |j| b.rel(normed[i * d + j], nwv[j] * cv[j] * rinv2, 1.0 / 64.0, 3e-3, i * d + j);
    }
    try rp.stage(name, b);
    try rp.stage(try std.fmt.allocPrint(a, "{s} post", .{name}), bp);
    try rp.stage(try std.fmt.allocPrint(a, "{s} comb", .{name}), bc);
}

/// glue.hc_post in f64: bf16(rank 0 + rank 1) times the post gate plus the comb mix of the streams.
pub fn hcPostStage(rp: *Report, name: []const u8, d: usize, x: []const f64, g0: []const f32, g1: []const f32, post: []const f32, comb: []const f32, rows: usize, got: []const f64) !void {
    var b = Bound{};
    for (0..rows) |i| for (0..4) |sv| for (0..d) |j| {
        const branch = bf16r(@as(f64, g0[i * d + j]) + g1[i * d + j]);
        var mixed: f64 = 0;
        for (0..4) |q| mixed += comb[i * 16 + q * 4 + sv] * x[(i * 4 + q) * d + j];
        const at = (i * 4 + sv) * d + j;
        b.rel(got[at], post[i * 4 + sv] * branch + mixed, 1.0 / 64.0, 3e-3, at);
    };
    try rp.stage(name, b);
}

/// One file of the window (stage outputs are equal bits on both sides by now).
pub const Win = struct {
    a: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    e: Obj,

    pub fn raw(w: Win, site: ?[]const u8, key: []const u8) ![]u8 {
        const o = if (site) |s| w.e.get(s).?.object else w.e;
        return ref.readFile(w.a, w.io, w.dir, ref.str(o, key));
    }

    pub fn bf(w: Win, site: ?[]const u8, key: []const u8) ![]f64 {
        return bfs(w.a, try w.raw(site, key));
    }

    pub fn f(w: Win, site: ?[]const u8, key: []const u8) ![]const f32 {
        return f32s(try w.raw(site, key));
    }
};

/// The whole layer in f64, stage by stage from each stage's inputs; returns the worst fraction of tolerance.
pub fn layerF64(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: lay.Config, h: *const Host, tag: []const u8, e: Obj, lid: []const u8) !f64 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const w: Win = .{ .a = a, .io = io, .dir = dir, .e = e };
    var rp: Report = .{ .a = a, .tag = tag, .id = lid };
    const rows: usize = @intCast(ref.int(e, "rows"));
    const d: usize = c.d;
    const kd = if (h.kda) |*v| v else return error.GlmLayerKind;
    const ps: usize = kd.p_stride;
    const sm = samples(rows);
    // the attention site: hc_pre, the KDA block's projections, its chain, o_proj, hc_post
    const x0 = try w.bf(null, "x0");
    const n0 = try w.bf("s0", "normed");
    try hcPreStage(&rp, a, "attn hc_pre", c, h.attn_hc, h.in_norm, x0, rows, n0, try w.f("s0", "post"), try w.f("s0", "comb"));
    const p = try w.bf("s0", "p");
    try exl3Stage(&rp, a, "qkv", kd.qkv, n0, sm, p, ps, 0, false);
    try b16Stage(&rp, a, "f_a", kd.fa, n0, d, 0, p, ps, kd.fa_off, rows);
    try b16Stage(&rp, a, "g_a", kd.ga, n0, d, 0, p, ps, kd.ga_off, rows);
    try b16Stage(&rp, a, "b", kd.b, n0, d, 0, p, ps, kd.b_off, rows);
    try b16Stage(&rp, a, "f_b", kd.fb, p, ps, kd.fa_off, try w.bf("s0", "ka"), kd.fb.n, 0, rows);
    try b16Stage(&rp, a, "g_b", kd.gb, p, ps, kd.ga_off, try w.bf("s0", "kg"), kd.gb.n, 0, rows);
    try kdaStage(&rp, a, w, c, h, rows);
    const kout = try w.bf("s0", "kout");
    const part0 = try w.f("s0", "part");
    const part0v = try a.alloc(f64, part0.len);
    for (part0v, part0) |*o, v| o.* = v;
    try exl3Stage(&rp, a, "o_proj", kd.o, kout, sm, part0v, d, 0, true);
    const x1 = try w.bf("s1", "x");
    try hcPostStage(&rp, "attn hc_post", d, x0, part0, try w.f("s0", "peer"), try w.f("s0", "post"), try w.f("s0", "comb"), rows, x1);
    // the ffn site: hc_pre, the MoE (router, select, routed and shared experts, combine) or the dense MLP, hc_post
    const n1 = try w.bf("s1", "normed");
    try hcPreStage(&rp, a, "ffn hc_pre", c, h.ffn_hc, h.post_norm, x1, rows, n1, try w.f("s1", "post"), try w.f("s1", "comb"));
    if (h.dense) _ = try swigluStage(&rp, a, w, c, h.sh, n1, rows, "mlp", "part") else try moeStage(&rp, a, w, c, h, n1, rows);
    try hcPostStage(&rp, "ffn hc_post", d, x1, try w.f("s1", "part"), try w.f("s1", "peer"), try w.f("s1", "post"), try w.f("s1", "comb"), rows, try w.bf(null, "x"));
    std.debug.print("RESULT {s} {s} f64 per stage (fraction of tolerance):{s}\n", .{ lid, tag, rp.line });
    return rp.worst;
}

/// The chain (serial for decode windows, chunked at pos for a prompt chunk) from the engine's prep, f64.
fn kdaStage(rp: *Report, a: std.mem.Allocator, w: Win, c: lay.Config, h: *const Host, rows: usize) !void {
    const hd: usize = h.kda.?.heads;
    const per = hd * ref.DK;
    const out64 = try a.alloc(f64, rows * per);
    const st64 = try a.alloc(f64, per * ref.DK);
    const q = f32s(try w.raw("prep", "q"));
    const k = f32s(try w.raw("prep", "k"));
    const v = f32s(try w.raw("prep", "v"));
    const lg = f32s(try w.raw("prep", "lg"));
    const beta = f32s(try w.raw("prep", "beta"));
    const rec0 = try w.f(null, "rec0");
    const gate = ref.bf16s(try w.raw("s0", "kg"));
    const norm = ref.bf16s(h.kda.?.norm);
    if (w.e.get("prefill").?.bool) {
        try ref.chunkedRef(a, q, k, v, lg, beta, rec0, gate, norm, c.eps, hd, rows, @intCast(ref.int(w.e, "pos")), out64, st64);
    } else {
        try ref.serialRef(a, q, k, v, lg, beta, rec0, gate, norm, c.eps, hd, rows, out64, st64);
    }
    const kout = try w.bf("s0", "kout");
    var b = Bound{};
    for (out64, 0..) |want, i| b.rel(kout[i], want, 1.0 / 128.0, 2e-3, i);
    try rp.stage("kda out", b);
    const rec = try w.f(null, "rec");
    var bs = Bound{};
    for (st64, 0..) |want, i| bs.rel(rec[i], want, 2e-2, 2e-3, i);
    try rp.stage("kda state", bs);
}

/// MoE in f64: router dots, the biased top-k and its weights, routed experts, the shared expert, the combine.
pub fn moeStage(rp: *Report, a: std.mem.Allocator, w: Win, c: lay.Config, h: *const Host, n1: []const f64, rows: usize) !void {
    const d: usize = c.d;
    const ne: usize = c.experts;
    const slots: usize = c.top_k + 1;
    const rw = try bfs(a, h.router);
    const mlog = try w.f("s1", "mlog");
    var b = Bound{};
    for (0..rows) |i| for (0..ne) |ex| {
        var acc: f64 = 0;
        for (0..d) |j| acc += n1[i * d + j] * rw[ex * d + j];
        b.rel(mlog[i * ne + ex], acc, 1.0 / 1024.0, 1e-5, i * ne + ex);
    };
    try rp.stage("router", b);
    const bias = f32s(h.bias);
    const picks: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, try w.raw("s1", "pick")));
    const wts = try w.f("s1", "wts");
    var bw = Bound{};
    const score = try a.alloc(f64, ne);
    const used = try a.alloc(bool, ne);
    for (0..rows) |i| {
        for (0..ne) |ex| {
            score[ex] = sigmoid(mlog[i * ne + ex]);
            used[ex] = false;
        }
        var total: f64 = 0;
        for (0..c.top_k) |kk| { // the biased greedy pick, lowest id on ties; a different pick must be a near-tie
            var best: f64 = -std.math.inf(f64);
            var at: usize = ne;
            for (0..ne) |ex| if (!used[ex] and score[ex] + bias[ex] > best) {
                best = score[ex] + bias[ex];
                at = ex;
            };
            const got: usize = @intCast(picks[i * slots + kk]);
            if (got != at and !(got < ne and !used[got] and @abs(score[got] + bias[got] - best) < 1e-6)) {
                std.debug.print("FAIL {s} {s} select f64: row {d} slot {d} picks {d}, f64 picks {d}\n", .{ rp.id, rp.tag, i, kk, got, at });
                return error.TestFailed;
            }
            used[got] = true;
            total += score[got];
        }
        for (0..c.top_k) |kk| bw.rel(wts[i * slots + kk], score[@intCast(picks[i * slots + kk])] / total * c.routed_scale, 1.0 / 1024.0, 1e-6, i * slots + kk);
        if (picks[i * slots + c.top_k] != @as(i32, @intCast(ne)) or wts[i * slots + c.top_k] != 1.0) {
            std.debug.print("FAIL {s} {s} select: row {d}'s last slot is not the shared expert at weight 1\n", .{ rp.id, rp.tag, i });
            return error.TestFailed;
        }
    }
    try rp.stage("select", bw);
    try expertStage(rp, a, w, c, h, n1, picks, rows);
    _ = try swigluStage(rp, a, w, c, h.sh, n1, rows, "shared", "sy");
    const ey = try w.f("s1", "ey");
    const part = try w.f("s1", "part");
    var bc = Bound{};
    for (0..rows) |i| for (0..d) |j| {
        var acc: f64 = 0;
        for (0..slots) |kk| acc += @as(f64, wts[i * slots + kk]) * ey[(i * slots + kk) * d + j];
        bc.rel(part[i * d + j], acc, 1.0 / 1024.0, 1e-5, i * d + j);
    };
    try rp.stage("combine", bc);
}

/// A SwiGLU MLP (the shared expert, or a dense layer's): gate and up (EXL3), SwiGLU, down (EXL3, fp32 into `out`).
fn swigluStage(rp: *Report, a: std.mem.Allocator, w: Win, c: lay.Config, lin: [3]Lin, n1: []const f64, rows: usize, label: []const u8, out: []const u8) ![]const f32 {
    const sm = samples(rows);
    const sw: usize = lin[0].n;
    const smg = try w.bf("s1", "smg");
    const smu = try w.bf("s1", "smu");
    try exl3Stage(rp, a, try std.fmt.allocPrint(a, "{s} gate", .{label}), lin[0], n1, sm, smg, sw, 0, false);
    try exl3Stage(rp, a, try std.fmt.allocPrint(a, "{s} up", .{label}), lin[1], n1, sm, smu, sw, 0, false);
    const sact = try w.bf("s1", "sact");
    var bs = Bound{};
    for (0..rows * sw) |i| {
        const g = @min(smg[i], c.limit);
        const u = std.math.clamp(smu[i], -c.limit, c.limit);
        bs.rel(sact[i], bf16r(bf16r(g * sigmoid(g)) * u), 1.0 / 64.0, 3e-3, i);
    }
    try rp.stage(try std.fmt.allocPrint(a, "{s} swiglu", .{label}), bs);
    const y = try w.f("s1", out);
    const yv = try a.alloc(f64, y.len);
    for (yv, y) |*o, v| o.* = v;
    try exl3Stage(rp, a, try std.fmt.allocPrint(a, "{s} down", .{label}), lin[2], sact, sm, yv, c.d, 0, true);
    return y;
}

/// Routed slots of the first and last rows: gate/up, the bf16 SwiGLU the epilogue rounds, down, each slot's y.
fn expertStage(rp: *Report, a: std.mem.Allocator, w: Win, c: lay.Config, h: *const Host, n1: []const f64, picks: []const i32, rows: usize) !void {
    const d: usize = c.d;
    const slots: usize = c.top_k + 1;
    const ex = &h.ex;
    const ey = try w.f("s1", "ey");
    var b = Bound{};
    const which = [2]usize{ 0, rows - 1 };
    for (which[0..if (rows > 1) 2 else 1]) |i| for (0..c.top_k) |kk| {
        const e: usize = @intCast(picks[i * slots + kk]);
        const x = n1[i * d ..][0..d];
        const g = try exl3Ref(a, ex.lin(0, e), x, 1);
        const u = try exl3Ref(a, ex.lin(1, e), x, 1);
        const act = try a.alloc(f64, ex.width);
        for (act, g.want, u.want) |*o, gv, uv| {
            const gg = @min(bf16r(gv), c.limit);
            const uu = std.math.clamp(bf16r(uv), -c.limit, c.limit);
            o.* = bf16r(bf16r(gg * sigmoid(gg)) * uu);
        }
        const y = try exl3Ref(a, ex.lin(2, e), act, 1);
        for (0..d) |j| {
            const at = (i * slots + kk) * d + j;
            b.add(ey[at], y.want[j], @abs(y.want[j]) / 128 + y.s1[j] / 128 + 1e-30, at);
        }
    };
    try rp.stage("routed experts", b);
}
