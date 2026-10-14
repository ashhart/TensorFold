//! lfm2.cu's kernels at d1's shapes against host references that round where the bf16 torch modules round: every byte.

const std = @import("std");
const cuda = @import("cuda");
const lfm2 = @import("lfm2");
const nemotron = @import("nemotron");
const check = @import("check.zig");
const Gpu = check.Gpu;
const gm = nemotron.glue_math;
const bf = gm.bf16ToF32;

const hidden = 2048;
const ffn = 10752;
const heads = 32;
const kv_heads = 8;
const dim = 64;
const taps = 3;
const eps: f32 = 1e-5;
const theta: f32 = 1e6;
const vocab = 128000;
const table_rows = 1024; // label rows drawn from a table this tall: the sum order does not depend on the vocabulary
const nt = 256; // lfm2.cu's NT
const lt = 1024; // LT

/// The packed prompt: a trunk of `p` rows then branches, each row's position and predecessors (-1: none).
const Tree = struct { rows: usize, positions: []i32, prev: []i32 };

fn tree(a: std.mem.Allocator, p: usize, lengths: []const usize) !Tree {
    var rows = p;
    for (lengths) |n| rows += n;
    const positions = try a.alloc(i32, rows);
    for (positions[0..p], 0..) |*x, i| x.* = @intCast(i);
    var row = p;
    for (lengths) |n| for (0..n) |k| {
        positions[row] = @intCast(p + k);
        row += 1;
    };
    const prev = try a.alloc(i32, rows * (taps - 1));
    for (0..rows) |t| for (1..taps) |j| {
        const back = positions[t] - @as(i32, @intCast(j));
        prev[t * (taps - 1) + j - 1] = if (back < 0) -1 else if (back >= p) @intCast(t - j) else back;
    };
    return .{ .rows = rows, .positions = positions, .prev = prev };
}

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

/// block_sum: each warp's butterfly, then the warps in order.
fn blockSum(lanes: []f32) f32 {
    var s: f32 = 0;
    for (0..lanes.len / 32) |w| {
        const ws = warpSum(lanes[w * 32 ..][0..32]);
        s = if (w == 0) ws else s + ws;
    }
    return s;
}

fn invRms(ss: f32, n: usize) f32 {
    return 1.0 / @sqrt(ss / @as(f32, @floatFromInt(n)) + eps);
}

fn addRms(h: []u16, y: ?[]const u16, w: []const u16, x: []u16, rows: usize, width: usize) void {
    var vals: [nt * 16]f32 = undefined;
    for (0..rows) |r| {
        const at = r * width;
        var lanes: [nt]f32 = @splat(0);
        for (0..width) |i| {
            if (y) |yy| h[at + i] = gm.f32ToBf16(bf(h[at + i]) + bf(yy[at + i]));
            vals[i] = bf(h[at + i]);
            lanes[i % nt] += vals[i] * vals[i];
        }
        const inv = invRms(blockSum(&lanes), width);
        for (0..width) |i| x[at + i] = gm.f32ToBf16(gm.rbf(vals[i] * inv) * bf(w[i]));
    }
}

fn shortConv(proj: []const u16, weight: []const u16, prev: []const i32, out: []u16, rows: usize, c: usize) void {
    const c3 = 3 * c;
    for (0..rows) |t| for (0..c) |ch| {
        var acc: f32 = 0;
        for (0..taps - 1) |j| {
            const src = prev[t * (taps - 1) + taps - 2 - j];
            const s: usize = @intCast(@max(src, 0));
            const bx: f32 = if (src < 0) 0 else gm.rbf(bf(proj[s * c3 + ch]) * bf(proj[s * c3 + 2 * c + ch]));
            acc += bf(weight[ch * taps + j]) * bx;
        }
        acc += bf(weight[ch * taps + taps - 1]) * gm.rbf(bf(proj[t * c3 + ch]) * bf(proj[t * c3 + 2 * c + ch]));
        out[t * c + ch] = gm.f32ToBf16(bf(proj[t * c3 + c + ch]) * gm.rbf(acc));
    };
}

fn qkNormRope(qkv: []const u16, qw: []const u16, kw: []const u16, positions: []const i32, q: []u16, k: []u16, v: []u16) void {
    const half = dim / 2;
    const all = heads + 2 * kv_heads;
    for (positions, 0..) |pos, t| {
        var cs: [half]f32 = undefined;
        var sn: [half]f32 = undefined;
        for (0..half) |i| {
            const e = @as(f32, @floatFromInt(2 * i)) / @as(f32, dim);
            const inv = 1.0 / @as(f32, @floatCast(std.math.pow(f64, theta, e)));
            const ang: f64 = @as(f32, @floatFromInt(pos)) * inv;
            cs[i] = @floatCast(@cos(ang));
            sn[i] = @floatCast(@sin(ang));
        }
        for (0..all) |h| {
            const src = qkv[(t * all + h) * dim ..][0..dim];
            if (h >= heads + kv_heads) {
                @memcpy(v[(t * kv_heads + h - heads - kv_heads) * dim ..][0..dim], src);
                continue;
            }
            var lanes: [32]f32 = @splat(0);
            for (lanes[0..half], 0..) |*x, l| x.* = bf(src[l]) * bf(src[l]) + bf(src[l + half]) * bf(src[l + half]);
            const inv = invRms(warpSum(&lanes), dim);
            const w = if (h < heads) qw else kw;
            const dst = if (h < heads) q[(t * heads + h) * dim ..][0..dim] else k[(t * kv_heads + h - heads) * dim ..][0..dim];
            for (0..half) |l| {
                const na = gm.rbf(gm.rbf(bf(src[l]) * inv) * bf(w[l]));
                const nb = gm.rbf(gm.rbf(bf(src[l + half]) * inv) * bf(w[l + half]));
                dst[l] = gm.f32ToBf16(na * cs[l] + -nb * sn[l]);
                dst[l + half] = gm.f32ToBf16(nb * cs[l] + na * sn[l]);
            }
        }
    }
}

fn siluMul(gu: []const u16, out: []u16, rows: usize) void {
    for (0..rows) |t| for (0..ffn) |i| {
        const s = gm.rbf(gm.silu(bf(gu[t * 2 * ffn + i])));
        out[t * ffn + i] = gm.f32ToBf16(s * bf(gu[t * 2 * ffn + ffn + i]));
    };
}

fn labelLogits(hid: []const u16, embed: []const u16, labels: []const i32, out: []f32) void {
    for (labels, out) |label, *o| {
        const row = embed[@as(usize, @intCast(label)) * hidden ..][0..hidden];
        var lanes: [nt]f32 = @splat(0);
        for (0..hidden) |i| lanes[i % nt] = gm.fma(bf(hid[i]), bf(row[i]), lanes[i % nt]);
        o.* = blockSum(&lanes);
    }
}

fn logsumexp(logits: []const f32) f32 {
    var m = -std.math.inf(f32);
    for (logits) |l| m = @max(m, l);
    if (std.math.isInf(m)) m = 0; // torch.logsumexp: an infinite max shifts nothing
    var lanes: [lt]f32 = @splat(0);
    for (logits, 0..) |l, i| lanes[i % lt] += gm.exp(l - m);
    return m + @as(f32, @floatCast(@log(@as(f64, blockSum(&lanes)))));
}

const Rig = struct {
    gpu: Gpu,
    a: std.mem.Allocator, // host data, freed together at the end
    s: cuda.Stream,
    k: lfm2.kernels.Kernels,
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

    /// The device values at `ptr` against `want` (bf16 as u16, or f32): equal bytes, a reference not all zeros.
    fn same(r: *Rig, what: []const u8, rows: usize, comptime T: type, ptr: u64, want: []const T) !void {
        try r.s.synchronize();
        const got = try r.a.alloc(T, want.len);
        try r.gpu.d.check(r.gpu.d.api.cuMemcpyDtoH_v2(got.ptr, ptr, want.len * @sizeOf(T)), "cuMemcpyDtoH");
        r.cases += 1;
        var worst: f64 = 0;
        var differ: usize = 0;
        var live: usize = 0;
        for (got, want) |g, w| {
            const gv: f64 = if (T == u16) bf(g) else g;
            const wv: f64 = if (T == u16) bf(w) else w;
            worst = @max(worst, @abs(gv - wv));
            differ += @intFromBool(g != w);
            live += @intFromBool(wv != 0);
        }
        std.debug.print("RESULT {s} rows {d}: max abs error {e}, {d} of {d} elements differ\n", .{ what, rows, worst, differ, want.len });
        if (differ != 0 or live == 0) r.bad += 1;
    }
};

fn addRmsCases(r: *Rig, rows: usize) !void {
    for ([_]bool{ false, true }) |with_y| {
        const h = try r.bfs(rows * hidden, 2.0);
        const y = try r.bfs(rows * hidden, 1.0);
        const w = try r.bfs(hidden, 1.0);
        const hd = try r.dev(u16, h);
        const xd = try r.zeros(u16, rows * hidden);
        try r.k.addRms(r.s, hd, if (with_y) try r.dev(u16, y) else null, try r.dev(u16, w), xd, rows, hidden, eps);
        const x = try r.a.alloc(u16, rows * hidden);
        addRms(h, if (with_y) y else null, w, x, rows, hidden);
        try r.same(if (with_y) "add_rms h" else "rms h", rows, u16, hd, h);
        try r.same(if (with_y) "add_rms x" else "rms x", rows, u16, xd, x);
    }
}

fn shortConvCase(r: *Rig, t: Tree) !void {
    const proj = try r.bfs(t.rows * 3 * hidden, 2.0);
    const weight = try r.bfs(hidden * taps, 0.5);
    const outd = try r.zeros(u16, t.rows * hidden);
    try r.k.shortConv(r.s, try r.dev(u16, proj), try r.dev(u16, weight), try r.dev(i32, t.prev), outd, t.rows, hidden, taps);
    const out = try r.a.alloc(u16, t.rows * hidden);
    shortConv(proj, weight, t.prev, out, t.rows, hidden);
    try r.same("short_conv", t.rows, u16, outd, out);
}

fn qkNormRopeCase(r: *Rig, positions: []const i32) !void {
    const rows = positions.len;
    const qkv = try r.bfs(rows * (heads + 2 * kv_heads) * dim, 2.0);
    const qw = try r.bfs(dim, 1.0);
    const kw = try r.bfs(dim, 1.0);
    const qd = try r.zeros(u16, rows * heads * dim);
    const kd = try r.zeros(u16, rows * kv_heads * dim);
    const vd = try r.zeros(u16, rows * kv_heads * dim);
    try r.k.qkNormRope(r.s, try r.dev(u16, qkv), try r.dev(u16, qw), try r.dev(u16, kw), try r.dev(i32, positions), qd, kd, vd, rows, heads, kv_heads, dim, theta, eps);
    const q = try r.a.alloc(u16, rows * heads * dim);
    const k = try r.a.alloc(u16, rows * kv_heads * dim);
    const v = try r.a.alloc(u16, rows * kv_heads * dim);
    qkNormRope(qkv, qw, kw, positions, q, k, v);
    try r.same("qk_norm_rope q", rows, u16, qd, q);
    try r.same("qk_norm_rope k", rows, u16, kd, k);
    try r.same("qk_norm_rope v", rows, u16, vd, v);
}

fn siluMulCase(r: *Rig, rows: usize) !void {
    const gu = try r.bfs(rows * 2 * ffn, 6.0);
    const outd = try r.zeros(u16, rows * ffn);
    try r.k.siluMul(r.s, try r.dev(u16, gu), outd, rows, ffn);
    const out = try r.a.alloc(u16, rows * ffn);
    siluMul(gu, out, rows);
    try r.same("silu_mul", rows, u16, outd, out);
}

fn labelCase(r: *Rig, count: usize) !void {
    const hid = try r.bfs(hidden, 2.0);
    const embed = try r.bfs(table_rows * hidden, 0.1);
    const labels = try r.a.alloc(i32, count);
    for (labels) |*l| l.* = r.rng.intRangeLessThan(i32, 0, table_rows);
    const outd = try r.zeros(f32, count);
    try r.k.labelLogits(r.s, try r.dev(u16, hid), try r.dev(u16, embed), try r.dev(i32, labels), outd, count, hidden);
    const out = try r.a.alloc(f32, count);
    labelLogits(hid, embed, labels, out);
    try r.same("label_logits", count, f32, outd, out);
}

fn logsumexpCase(r: *Rig, count: usize) !void {
    const logits = try r.a.alloc(f32, count);
    for (logits) |*l| l.* = (r.rng.float(f32) * 2 - 1) * 20;
    const outd = try r.zeros(f32, 1);
    try r.k.logsumexp(r.s, try r.dev(f32, logits), outd, count);
    try r.same("logsumexp", count, f32, outd, &.{logsumexp(logits)});
}

/// torch.logsumexp's own results (CPU, fp32) on infinite and large finite logits, for the kernel and the reference.
fn logsumexpTorch(r: *Rig) !void {
    const inf = std.math.inf(f32);
    const Case = struct { logits: []const f32, want: f32 };
    const cases = [_]Case{
        .{ .logits = &.{-inf}, .want = -inf },
        .{ .logits = &.{ inf, 0 }, .want = inf },
        .{ .logits = &.{ -1000, -1001 }, .want = -999.686767578125 },
        .{ .logits = &.{ 1000, 999 }, .want = 1000.313232421875 },
    };
    for (cases) |c| {
        try check.expect(logsumexp(c.logits) == c.want, "logsumexp reference: {any} gives {d}, torch {d}", .{ c.logits, logsumexp(c.logits), c.want });
        const outd = try r.zeros(f32, 1);
        try r.k.logsumexp(r.s, try r.dev(f32, c.logits), outd, c.logits.len);
        try r.same("logsumexp torch", c.logits.len, f32, outd, &.{c.want});
    }
}

/// Every lfm2.cu kernel on seeded inputs at d1's shapes (rows 1, 7, 300 over trees), byte-compared with its reference.
pub fn run(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x1f_2d_3b);
    var r: Rig = .{ .gpu = gpu, .a = a, .s = try cuda.Stream.init(gpu.d, true), .k = try lfm2.kernels.Kernels.load(gpu.d), .rng = prng.random() };
    defer {
        r.k.unload();
        r.s.deinit();
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    const trees = [_]Tree{ try tree(a, 0, &.{1}), try tree(a, 4, &.{ 1, 2 }), try tree(a, 280, &.{ 5, 7, 1, 7 }) };
    for (trees) |t| {
        try addRmsCases(&r, t.rows);
        try shortConvCase(&r, t);
        try qkNormRopeCase(&r, t.positions);
        try siluMulCase(&r, t.rows);
        try labelCase(&r, t.rows);
        try logsumexpCase(&r, t.rows);
    }
    const far = try a.alloc(i32, trees[2].rows);
    for (far, trees[2].positions) |*f, p| f.* = p + 32000; // a long trunk's tail: large angles
    try qkNormRopeCase(&r, far);
    try logsumexpCase(&r, vocab);
    try logsumexpTorch(&r);
    try check.expect(r.bad == 0, "lfm2-kernels: {d} of {d} comparisons differ from the host references or have all-zero references", .{ r.bad, r.cases });
    check.pass("lfm2-kernels: {d} of {d} comparisons byte-equal to the host references (add_rms, short_conv, qk_norm_rope, silu_mul, label_logits, logsumexp) and torch's logsumexp values", .{ r.cases, r.cases });
}
