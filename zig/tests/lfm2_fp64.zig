//! LFM2's projection classes, GLU and attention on real weights: tensor units, tiles and twin, each against float64.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const tile = tf.tile_qmm;
const am = tf.affine_mm;
const q = tf.lfm2_moe;
const st = q.state;
const c = q.config;
const row = tf.row_projection;

const opts = mtl.ResourceOptions.shared;
/// Rows every kernel reads: the twin's most a call, so prompt and decode kernels see the same rows.
const rows = 32;
const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

fn fromBf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}

/// bf16 bits of `v` truncated (the random rows need no rounding).
fn bf16(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

/// Code `i` of a row's little-endian bit stream; the next byte is read only when the code runs into it.
fn code(bytes: [*]const u8, bits: u8, i: usize) u32 {
    const off = i * @as(usize, bits);
    var v: u32 = bytes[off / 8];
    if (off % 8 + bits > 8) v |= @as(u32, bytes[off / 8 + 1]) << 8;
    return (v >> @intCast(off % 8)) & ((@as(u32, 1) << @intCast(bits)) - 1);
}

/// Row r of x against column n of w in float64, and its terms' magnitude (the error's scale).
fn reference(w: row.Weights, x: []const u16, r: usize, n: usize) [2]f64 {
    const groups = w.k / 64;
    const row_bytes = w.k * @as(usize, w.bits) / 8;
    const bytes = w.w.contents() + w.w_off + n * row_bytes;
    const sc: [*]const u16 = @ptrCast(@alignCast(w.scales.contents() + w.s_off));
    const bi: [*]const u16 = @ptrCast(@alignCast(w.biases.contents() + w.b_off));
    var total: f64 = 0;
    var scale: f64 = 0;
    for (0..groups) |g| {
        var dot: f64 = 0;
        var sum: f64 = 0;
        for (0..64) |i| {
            const xv = fromBf(x[r * w.k + g * 64 + i]);
            dot += xv * @as(f64, @floatFromInt(code(bytes, w.bits, g * 64 + i)));
            sum += xv;
        }
        const sd = fromBf(sc[n * groups + g]) * dot;
        const bs = fromBf(bi[n * groups + g]) * sum;
        total += sd + bs;
        scale += @abs(sd) + @abs(bs);
    }
    return .{ total, scale };
}

const Err = struct { worst: f64 = 0, mean: f64 = 0, col: usize = 0, col_err: f64 = 0 };

/// y [rows, n] against the reference and its scale: the worst and mean error, and the column whose worst is worst.
fn errors(y: []const u16, ref: []const f64, mag: []const f64, n: usize) Err {
    var e = Err{};
    var total: f64 = 0;
    for (0..n) |col| {
        var col_worst: f64 = 0;
        for (0..rows) |r| {
            const err = @abs(fromBf(y[r * n + col]) - ref[r * n + col]) / @max(mag[r * n + col], 1e-30);
            col_worst = @max(col_worst, err);
            total += err;
        }
        if (col_worst > e.col_err) {
            e.col_err = col_worst;
            e.col = col;
        }
    }
    e.worst = e.col_err;
    e.mean = total / @as(f64, @floatFromInt(rows * n));
    return e;
}

fn differing(a: []const u16, b: []const u16) usize {
    var n: usize = 0;
    for (a, b) |u, v| n += @intFromBool(u != v);
    return n;
}

const Probe = struct {
    m: *q.Model,
    x: mtl.Buffer,
    sums: mtl.Buffer,
    y: [3]mtl.Buffer,
    ref: []f64,
    mag: []f64,

    fn run(p: Probe, body: anytype) !void {
        const cb = p.m.queue.commandBuffer();
        const e = cb.compute(.serial);
        try body.encode(e, p);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("GPU failed: {s}\n", .{msg});
            return error.GpuFailed;
        }
    }
};

fn affine(w: row.Weights) struct { w: Ref, s: Ref, b: Ref, n: u32, k: u32 } {
    return .{
        .w = .{ .buf = w.w, .off = w.w_off },
        .s = .{ .buf = w.scales, .off = w.s_off },
        .b = .{ .buf = w.biases, .off = w.b_off },
        .n = @intCast(w.n),
        .k = @intCast(w.k),
    };
}

const Tiles = struct {
    w: row.Weights,
    fn encode(t: Tiles, e: mtl.ComputeEncoder, p: Probe) !void {
        tile.encodeSums(e, &p.m.kernels.tiles, t.w.k, p.x, 0, p.sums, rows);
        try tile.encode(e, &p.m.kernels.tiles, q.forward.matrix(t.w), p.x, 0, p.sums, 0, p.y[0], 0, rows);
    }
};

const Nax = struct {
    w: row.Weights,
    fn encode(t: Nax, e: mtl.ComputeEncoder, p: Probe) !void {
        const pipes = p.m.kernels.naxPipes(t.w.bits) orelse return error.NoTensorPipelines; // the matrix's own width
        am.rowSums(e, pipes, 64, Ref{ .buf = p.x }, Ref{ .buf = p.sums }, rows, @intCast(t.w.k));
        am.dense(e, pipes, Ref{ .buf = p.x }, Ref{ .buf = p.sums }, affine(t.w), Ref{ .buf = p.y[1] }, rows);
    }
};

const Twin = struct {
    w: row.Weights,
    fn encode(t: Twin, e: mtl.ComputeEncoder, p: Probe) !void {
        try tile.encodeTwin(e, &p.m.kernels.tiles, q.forward.matrix(t.w), p.x, 0, p.y[2], 0, rows);
    }
};

const names = [3][]const u8{ "prompt tiles", "prompt tensor", "decode twin" };

fn report(name: []const u8, w: row.Weights, errs: [3]Err, y: [3][]const u16) void {
    std.debug.print("{s} {d}x{d} ({d}-bit):\n", .{ name, w.n, w.k, w.bits });
    for (names, errs) |which, e| {
        const line = "  {s}: worst {e:.3}, mean {e:.3}, worst column {d} at {e:.3}\n";
        std.debug.print(line, .{ which, e.worst, e.mean, e.col, e.col_err });
    }
    const line = "  tensor differs from tiles on {d}/{d} outputs; twin differs from tiles on {d}\n";
    std.debug.print(line, .{ differing(y[1], y[0]), rows * w.n, differing(y[2], y[0]) });
}

/// One projection class: the three kernels on the same 32 random rows, each against float64 on every output.
fn class(p: Probe, name: []const u8, w: row.Weights, worse: *usize) !void {
    const x = p.x.slice(u16, rows * w.k);
    for (0..rows) |r| for (0..w.n) |n| {
        const ref = reference(w, x, r, n);
        p.ref[r * w.n + n] = ref[0];
        p.mag[r * w.n + n] = ref[1];
    };
    try p.run(Tiles{ .w = w });
    try p.run(Nax{ .w = w });
    try p.run(Twin{ .w = w });
    var y: [3][]const u16 = undefined;
    var errs: [3]Err = undefined;
    for (0..3) |i| {
        y[i] = p.y[i].slice(u16, rows * w.n);
        errs[i] = errors(y[i], p.ref[0 .. rows * w.n], p.mag[0 .. rows * w.n], w.n);
    }
    report(name, w, errs, y);
    if (errs[1].worst > errs[0].worst) worse.* += 1;
}

/// The experts' gate and up SwiGLU (every slot on expert 0): tiles, tensor gather and twin against float64 silu(g) u.
fn glu(p: Probe, x: q.weights.Moe, worse: *usize) !void {
    const gate = x.gate.one;
    const up = x.up.one;
    const dev = p.m.device;
    const xs = p.x.slice(u16, rows * gate.k);
    const n = gate.n;
    for (0..rows) |r| for (0..n) |col| {
        const g = reference(gate, xs, r, col)[0];
        const u = reference(up, xs, r, col)[0];
        p.ref[r * n + col] = g / (1.0 + @exp(-g)) * u;
        p.mag[r * n + col] = 1.0; // absolute error: the outputs are activations of order one
    };
    const starts = try dev.buffer(65 * 4, opts);
    defer starts.deinit();
    for (starts.slice(u32, 65), 0..) |*v, i| v.* = if (i == 0) 0 else rows;
    const work = try dev.buffer((tile.items(rows, 64) + 1) * 4, opts);
    defer work.deinit();
    const ident = try dev.buffer(rows * 4, opts);
    defer ident.deinit();
    for (ident.slice(u32, rows), 0..) |*v, i| v.* = @intCast(i);
    const zeros = try dev.buffer(rows * 4, opts);
    defer zeros.deinit();
    @memset(zeros.slice(u32, rows), 0);
    const Body = struct {
        gate: row.Weights,
        up: row.Weights,
        starts: mtl.Buffer,
        work: mtl.Buffer,
        ident: mtl.Buffer,
        zeros: mtl.Buffer,
        fn encode(b: @This(), e: mtl.ComputeEncoder, pr: Probe) !void {
            const t = &pr.m.kernels.tiles;
            const g = q.forward.matrix(b.gate);
            const u = q.forward.matrix(b.up);
            tile.encodeSums(e, t, b.gate.k, pr.x, 0, pr.sums, rows);
            tile.encodeWork(e, t, b.starts, 64, b.work);
            try tile.encodeGroupedGlu(e, t, g, u, 64, pr.x, pr.sums, b.ident, pr.y[0], rows, b.starts, b.work);
            const sums = Ref{ .buf = pr.sums };
            const pipes = pr.m.kernels.naxPipes(b.gate.bits) orelse return error.NoTensorPipelines;
            am.rowSums(e, pipes, 64, Ref{ .buf = pr.x }, sums, rows, @intCast(b.gate.k));
            const offsets = Ref{ .buf = b.starts };
            const order = Ref{ .buf = b.ident };
            const y = Ref{ .buf = pr.y[1] };
            const ga = affine(b.gate);
            const ua = affine(b.up);
            am.gatherGlu(e, pipes, Ref{ .buf = pr.x }, sums, ga, ua, offsets, order, y, rows, 64, 1, 0);
            try tile.encodeTwinGlu(e, t, g, u, pr.x, pr.y[2], b.zeros, null, rows, 1);
        }
    };
    try p.run(Body{ .gate = gate, .up = up, .starts = starts, .work = work, .ident = ident, .zeros = zeros });
    var y: [3][]const u16 = undefined;
    var errs: [3]Err = undefined;
    for (0..3) |i| {
        y[i] = p.y[i].slice(u16, rows * n);
        errs[i] = errors(y[i], p.ref[0 .. rows * n], p.mag[0 .. rows * n], n);
    }
    report("experts gate+up SwiGLU (absolute error)", gate, errs, y);
    if (errs[1].worst > errs[0].worst) worse.* += 1;
}

/// Attention on 64 rows (the decode kernel on the first 8, its own sub-span cuts) against float64 on sampled pairs.
fn attention(p: Probe, len: usize, worse: *usize) !void {
    const m = p.m;
    const dev = m.device;
    const arows = 64;
    var sc = try st.Scratch.init(m.gpa, dev, &m.config.attention, arows);
    defer sc.deinit();
    var cache = try st.Cache.init(m.gpa, dev, &m.config.attention, len + arows + 64);
    defer cache.deinit();
    const kv = cache.blocks[2].attention;
    const kv_count = c.kv_heads * cache.capacity * c.head_dim;
    var prng = std.Random.DefaultPrng.init(0x7173_0003);
    const rnd = prng.random();
    for (kv.keys.slice(u16, kv_count)) |*v| v.* = bf16(rnd.float(f32) * 2 - 1);
    for (kv.values.slice(u16, kv_count)) |*v| v.* = bf16(rnd.float(f32) * 2 - 1);
    for (sc.queries.slice(u16, arows * c.hidden)) |*v| v.* = bf16(rnd.float(f32) * 2 - 1);
    const scale: f32 = 1.0 / @sqrt(@as(f32, c.head_dim));
    var ys: [3]mtl.Buffer = undefined;
    for (&ys) |*y| y.* = try dev.buffer(arows * c.hidden * 2, opts);
    defer for (ys) |y| y.deinit();
    const kinds = [3][]const u8{ "prompt lfm2_attn_mma_r2", "prompt lfm2_attn_nax", "decode lfm2_attn_mma (8 rows)" };
    for (0..3) |which| {
        const decode = which == 2;
        const r: usize = if (decode) 8 else arows;
        const subs: u32 = if (decode) st.subs else 1;
        const args = [4]u32{ @intCast(len), @intCast(cache.capacity), @bitCast(scale), subs };
        const used = (len + r + st.chunk_keys - 1) / st.chunk_keys * subs;
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        if (which == 0) e.setPipeline(m.kernels.get("lfm2_attn_mma_r2"));
        if (which == 1) e.setPipeline(m.kernels.attn_nax.?);
        if (which == 2) e.setPipeline(m.kernels.get("lfm2_attn_mma"));
        e.setBuffer(sc.queries, 0, 0);
        e.setBuffer(kv.keys, 0, 1);
        e.setBuffer(kv.values, 0, 2);
        e.setBytes(std.mem.asBytes(&args), 3);
        e.setBuffer(sc.pm, 0, 4);
        e.setBuffer(sc.pl, 0, 5);
        e.setBuffer(sc.po, 0, 6);
        e.setValue(@as(u32, @intCast(r)), 7);
        const per: usize = if (decode) 8 else 16;
        e.dispatchGroups(mtl.Size.of((r + per - 1) / per, c.kv_heads, used), mtl.Size.of(128, 1, 1));
        e.setPipeline(m.kernels.get("lfm2_attn_merge"));
        e.setBuffer(sc.pm, 0, 0);
        e.setBuffer(sc.pl, 0, 1);
        e.setBuffer(sc.po, 0, 2);
        e.setBytes(std.mem.asBytes(&args), 3);
        e.setBuffer(ys[which], 0, 4);
        e.dispatchGroups(mtl.Size.of(c.query_heads, r, 1), mtl.Size.of(32, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("attention: {s}\n", .{msg});
            return error.GpuFailed;
        }
    }
    const keys = kv.keys.slice(u16, kv_count);
    const vals = kv.values.slice(u16, kv_count);
    const qs = sc.queries.slice(u16, arows * c.hidden);
    const pr = try m.gpa.alloc(f64, len + arows);
    defer m.gpa.free(pr);
    var worst = [3]f64{ 0, 0, 0 };
    var count: usize = 0;
    for (0..arows) |r| for (0..c.query_heads) |h| {
        if (r >= 8 and (r % 4 != 0 or h % 4 != 1)) continue; // rows 0-7 whole (the decode kernel's), the rest sampled
        const kh = h / (c.query_heads / c.kv_heads);
        const visible = len + r + 1;
        var mx: f64 = -std.math.inf(f64);
        for (0..visible) |j| {
            var dot: f64 = 0;
            const qrow = qs[(r * c.query_heads + h) * c.head_dim ..][0..c.head_dim];
            const krow = keys[(kh * cache.capacity + j) * c.head_dim ..][0..c.head_dim];
            for (qrow, krow) |qv, kv_| dot += fromBf(qv) * fromBf(kv_);
            pr[j] = dot * scale;
            mx = @max(mx, pr[j]);
        }
        var sum: f64 = 0;
        for (0..visible) |j| {
            pr[j] = @exp(pr[j] - mx);
            sum += pr[j];
        }
        count += 1;
        for (0..c.head_dim) |d| {
            var o: f64 = 0;
            for (0..visible) |j| o += pr[j] * fromBf(vals[(kh * cache.capacity + j) * c.head_dim + d]);
            o /= sum;
            for (0..3) |which| {
                if (which == 2 and r >= 8) continue;
                const got = fromBf(ys[which].slice(u16, arows * c.hidden)[(r * c.query_heads + h) * c.head_dim + d]);
                worst[which] = @max(worst[which], @abs(got - o));
            }
        }
    };
    const head_line = "attention, 64 rows at {d} cached keys, {d} (row, head) pairs against float64, absolute error:\n";
    std.debug.print(head_line, .{ len, count });
    for (kinds, worst) |kind, w| std.debug.print("  {s}: worst {e:.3}\n", .{ kind, w });
    const a = ys[0].slice(u16, arows * c.hidden);
    const b = ys[1].slice(u16, arows * c.hidden);
    const d = ys[2].slice(u16, 8 * c.hidden);
    const diff_line = "  tensor differs from mma_r2 on {d}/{d} outputs; decode kernel differs from mma_r2 on {d}/{d}\n";
    const decode_differs = differing(d, a[0 .. 8 * c.hidden]);
    std.debug.print(diff_line, .{ differing(b, a), arows * c.hidden, decode_differs, 8 * c.hidden });
    if (worst[1] > worst[0]) worse.* += 1;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.ExpectedModel;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try q.Model.load(init.gpa, init.io, args[1]);
    defer m.deinit();
    if (m.kernels.nax == null) {
        std.debug.print("no tensor units: nothing to compare\n", .{});
        return error.NoTensorUnits;
    }
    const dev = m.device;
    const x = try dev.buffer(rows * c.intermediate * 2, opts);
    defer x.deinit();
    var prng = std.Random.DefaultPrng.init(0x7173_0001);
    for (x.slice(u16, rows * c.intermediate)) |*v| v.* = bf16(prng.random().float(f32) * 2 - 1);
    const sums = try dev.buffer(rows * (c.intermediate / 64) * 4, opts);
    defer sums.deinit();
    var ys: [3]mtl.Buffer = undefined;
    for (&ys) |*y| y.* = try dev.buffer(rows * c.intermediate * 2, opts);
    defer for (ys) |y| y.deinit();
    const ref = try init.gpa.alloc(f64, rows * c.intermediate);
    defer init.gpa.free(ref);
    const mag = try init.gpa.alloc(f64, rows * c.intermediate);
    defer init.gpa.free(mag);
    const p = Probe{ .m = m, .x = x, .sums = sums, .y = ys, .ref = ref, .mag = mag };
    var worse: usize = 0;
    const b0 = m.weights.blocks[0];
    const attn = m.weights.blocks[2].mixer.attention;
    const moe = m.weights.blocks[10].ffn.moe;
    try class(p, "conv in", b0.mixer.conv.in, &worse);
    try class(p, "conv out", b0.mixer.conv.out, &worse);
    try class(p, "attention q", attn.q, &worse);
    try class(p, "attention k", attn.k, &worse);
    try class(p, "attention v", attn.v, &worse);
    try class(p, "attention out", attn.out, &worse);
    try class(p, "dense gate", b0.ffn.dense.gate, &worse);
    try class(p, "dense up", b0.ffn.dense.up, &worse);
    try class(p, "dense down", b0.ffn.dense.down, &worse);
    try class(p, "expert gate", moe.gate.one, &worse);
    try class(p, "expert up", moe.up.one, &worse);
    try class(p, "expert down", moe.down.one, &worse);
    try glu(p, moe, &worse);
    try attention(p, 0, &worse);
    try attention(p, 4000, &worse);
    std.debug.print("{d} classes where the tensor path's worst error exceeds the tiles'\n", .{worse});
    if (worse != 0) return error.TensorPathWorse;
}
