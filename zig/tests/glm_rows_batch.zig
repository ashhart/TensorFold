//! glm_rows_check's batched launches: a chunk's heads in one batch, its split stacks in one call, one shared GLU.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const wide = tf.glm.wide;
const affine = tf.glm.affine;
const affine_mm = tf.affine_mm;
const rc = @import("glm_rows_check.zig");
const Ref = affine.Ref;
const Gpu = rc.Gpu;
const Tally = rc.Tally;
const Mat = rc.Mat;
const buf = rc.buf;
const ref = rc.ref;
const eql = rc.eql;
const toBf = rc.toBf;
const dense = rc.dense;
const mmDense = rc.mmDense;
const swiglu = rc.swiglu;
const gluLaunch = rc.gluLaunch;
const rows = rc.rows;

fn headsLaunch(
    e: mtl.ComputeEncoder,
    p: *const wide.Pipes,
    q: affine.Affine,
    h: u32,
    io: wide.Io,
    kx: u32,
    sums: Ref,
) void {
    wide.headsBatch(p, e, q, h, io, kx, sums);
}

/// A chunk's embed_q / unembed_out (wide.headsBatch): the heads in one affine_mm batch == each head's own product.
pub fn headsCheck(g: *const Gpu, mm: affine_mm.Pipes, bits: u8, n: usize, k: usize, t: *Tally) !void {
    const nh = 4;
    const m = 33; // two 16-row steps and one more
    const ldx = nh * k + 96; // a row's slices, then columns no head reads (qp's indexer queries)
    const w = try Mat.init(g.device, nh * n, k, bits, 64, 4100 + n + bits);
    defer w.deinit();
    var prng = std.Random.DefaultPrng.init(4200 + k);
    const r = prng.random();
    const bs = [_]mtl.Buffer{
        try buf(g.device, m * ldx * 2),
        try buf(g.device, m * nh * (k / 64) * 4),
        try buf(g.device, m * nh * n * 2),
        try buf(g.device, m * k * 2),
        try buf(g.device, m * n * 2),
    };
    defer for (bs) |b| b.deinit();
    for (bs[0].slice(u16, m * ldx)) |*v| v.* = toBf((r.float(f32) - 0.5) * 2);
    try g.run(headsLaunch, .{
        &g.p,
        w.q,
        @as(u32, nh),
        wide.Io{ .in = ref(bs[0]), .ldx = @intCast(ldx), .out = ref(bs[2]), .ldo = @intCast(nh * n), .rows = m },
        @as(u32, @intCast(k)),
        ref(bs[1]),
    });
    const all = bs[2].slice(u16, m * nh * n);
    for (0..nh) |h| {
        for (0..m) |i| @memcpy(
            bs[3].slice(u16, m * k)[i * k ..][0..k],
            bs[0].slice(u16, m * ldx)[i * ldx + h * k ..][0..k],
        );
        try g.run(mmDense, .{ mm, w.part(h, nh), ref(bs[3]), ref(bs[1]), ref(bs[4]), @as(u32, m) });
        for (0..m) |i| try eql(all[(i * nh + h) * n ..][0..n], bs[4].slice(u16, m * n)[i * n ..][0..n]);
        t.pairs += m * n;
    }
}

fn denseMm(e: mtl.ComputeEncoder, p: *const wide.Pipes, q: affine.Affine, io: wide.Io, sums: Ref) void {
    wide.denseMm(p, e, q, io, sums);
}

/// A chunk's split stack (wide.denseMm) at the 64-padded pitch == each member alone (affine_mm or the row kernels).
pub fn splitMmCheck(g: *const Gpu, k: usize, ns: []const usize, bits: []const u8, t: *Tally) !usize {
    const m = 33;
    var ms: [affine.max_runs]Mat = undefined;
    for (ns, bits, 0..) |n, b, i| ms[i] = try Mat.init(g.device, n, k, b, 64, 900 + 37 * i + n);
    defer for (ms[0..ns.len]) |x| x.deinit();
    var q = ms[0].q;
    var at: u32 = @intCast(ns[0]);
    for (ms[1..ns.len], 0..) |x, i| {
        q.runs[i] = .{
            .w = x.q.w,
            .s = x.q.s,
            .b = x.q.b,
            .row = at,
            .n = x.q.n,
            .bits = x.q.bits,
            .group = x.q.group,
        };
        at += x.q.n;
    }
    q.more = @intCast(ns.len - 1);
    q.n = at;
    var prng = std.Random.DefaultPrng.init(9100 + k);
    const r = prng.random();
    const ld = std.mem.alignForward(usize, at, 64);
    const bs = [_]mtl.Buffer{
        try buf(g.device, m * k * 2),
        try buf(g.device, m * (k / 64) * 4),
        try buf(g.device, m * ld * 2),
        try buf(g.device, m * ld * 2),
    };
    defer for (bs) |b| b.deinit();
    for (bs[0].slice(u16, m * k)) |*v| v.* = toBf((r.float(f32) - 0.5) * 2);
    try g.run(denseMm, .{
        &g.p,
        q,
        wide.Io{ .in = ref(bs[0]), .ldx = @intCast(k), .out = ref(bs[2]), .ldo = @intCast(ld), .rows = m },
        ref(bs[1]),
    });
    var on_mm: usize = 0;
    for (ms[0..ns.len], 0..) |x, i| {
        const run = q.run(i);
        if (wide.mm(&g.p, x.q) != null and run.n % 64 == 0) {
            try g.run(mmDense, .{ wide.mm(&g.p, x.q).?, x.q, ref(bs[0]), ref(bs[1]), ref(bs[3]), @as(u32, m) });
            on_mm += 1;
        } else try g.run(
            dense,
            .{ &g.p, x.q, wide.Io{ .in = ref(bs[0]), .ldx = @intCast(k), .out = ref(bs[3]), .ldo = run.n, .rows = m } },
        );
        for (0..m) |y| try eql(
            bs[2].slice(u16, m * ld)[y * ld + run.row ..][0..run.n],
            bs[3].slice(u16, m * ld)[y * run.n ..][0..run.n],
        );
        t.pairs += m * run.n;
    }
    return on_mm;
}

/// The shared expert's gate and up at one width (wide.glu, one launch) == glm_swiglu(dense [gate; up]) bit for bit.
pub fn sharedGluCheck(g: *const Gpu) !usize {
    var pairs: usize = 0;
    for ([_]u8{ 8, 5, 4, 2 }) |bits| {
        const n = 2048;
        const m = try Mat.init(g.device, 2 * n, 4096, bits, 64, 8800 + @as(u64, bits));
        defer m.deinit();
        const bs = [_]mtl.Buffer{
            try buf(g.device, rows * 2 * n * 2),
            try buf(g.device, rows * n * 2),
            try buf(g.device, rows * n * 2),
        };
        defer for (bs) |b| b.deinit();
        for ([_]u32{ 1, 3, rows }) |r| { // a lone last row and pairs
            try g.run(
                dense,
                .{ &g.p, m.q, wide.Io{ .in = ref(m.x()), .ldx = 4096, .out = ref(bs[0]), .ldo = 2 * n, .rows = r } },
            );
            try g.run(swiglu, .{ g.swiglu, ref(bs[0]), ref(bs[1]), r, @as(u32, n), @as(f32, 0.75) });
            try g.run(gluLaunch, .{ &g.p, m.q, @as(u32, n), ref(m.x()), ref(bs[2]), r });
            try eql(bs[2].slice(u16, r * n), bs[1].slice(u16, r * n));
            pairs += r * n;
        }
    }
    std.debug.print(
        "shared expert gate|up at one width in one launch == glm_swiglu(dense [gate; up]): {d} bit-equal values\n",
        .{pairs},
    );
    return pairs;
}
