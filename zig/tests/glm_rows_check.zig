//! GLM-5.3-Flash's products at every MLX width (glm/wide.zig) against fp64, each row's bits in every window and launch.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const wide = tf.glm.wide;
const affine = tf.glm.affine;
const mlx = tf.quant.mlx;
const row = tf.row_projection;
const affine_mm = tf.affine_mm;
const batch = @import("glm_rows_batch.zig");

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const a = std.heap.page_allocator;
pub const rows = 16;
const pad = 32; // a strided x row's and out row's extra values
const Ref = affine.Ref;

pub fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}
fn fromBf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}
pub fn buf(device: mtl.Device, bytes: usize) !mtl.Buffer {
    return device.buffer(@max(bytes, 16), opts);
}
pub fn ref(b: mtl.Buffer) Ref {
    return .{ .buf = b };
}

/// A synthetic MLX matrix [n, k], its fp64 dequantization and its codes alone; x: `rows` rows of inputs.
pub const Mat = struct {
    q: affine.Affine,
    bufs: [4]mtl.Buffer,
    deq: []f64,
    codes: []f64,

    pub fn init(device: mtl.Device, n: usize, k: usize, bits: u8, group: usize, seed: u64) !Mat {
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        const words = n * k * bits / 32;
        const tables = n * (k / group);
        const bs = [4]mtl.Buffer{ try buf(device, words * 4), try buf(device, tables * 2), try buf(device, tables * 2), try buf(device, rows * k * 2) };
        for (bs[0].slice(u32, words)) |*v| v.* = r.int(u32);
        for (bs[1].slice(u16, tables)) |*v| v.* = toBf(r.float(f32) * 0.02 + 0.001);
        for (bs[2].slice(u16, tables)) |*v| v.* = toBf((r.float(f32) - 0.5) * 0.1);
        for (bs[3].slice(u16, rows * k)) |*v| v.* = toBf((r.float(f32) - 0.5) * 2);
        const m = Mat{ .q = .{ .w = ref(bs[0]), .s = ref(bs[1]), .b = ref(bs[2]), .n = @intCast(n), .k = @intCast(k), .bits = bits, .group = @intCast(group) }, .bufs = bs, .deq = try a.alloc(f64, n * k), .codes = try a.alloc(f64, n * k) };
        try mlx.reference(m.host(bs[1].slice(u16, tables), bs[2].slice(u16, tables)), m.deq);
        const ones = try a.alloc(u16, tables);
        defer a.free(ones);
        @memset(ones, toBf(1));
        const zeros = try a.alloc(u16, tables);
        defer a.free(zeros);
        @memset(zeros, 0);
        try mlx.reference(m.host(ones, zeros), m.codes);
        return m;
    }

    pub fn deinit(m: Mat) void {
        for (m.bufs) |b| b.deinit();
        a.free(m.deq);
        a.free(m.codes);
    }

    pub fn x(m: Mat) mtl.Buffer {
        return m.bufs[3];
    }

    fn host(m: Mat, scales: []const u16, biases: []const u16) mlx.Host {
        const q = m.q;
        const g = q.k / q.group;
        return .{
            .words = .{ .dtype = .i32, .rank = 2, .shape = .{ q.n, q.k * q.bits / 32, 1, 1, 1 }, .bytes = m.bufs[0].slice(u8, q.n * q.k * q.bits / 8) },
            .scales = .{ .dtype = .bf16, .rank = 2, .shape = .{ q.n, g, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(scales) },
            .biases = .{ .dtype = .bf16, .rank = 2, .shape = .{ q.n, g, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(biases) },
            .bits = q.bits,
            .group = q.group,
        };
    }

    /// Row i against column c in fp64, and the fp32 sums' bound (output rounding plus k-deep accumulation).
    fn reference(m: Mat, i: usize, c: usize) [2]f64 {
        const k: usize = m.q.k;
        const g_n = k / m.q.group;
        const sc = m.bufs[1].slice(u16, m.q.n * g_n);
        const bi = m.bufs[2].slice(u16, m.q.n * g_n);
        const xs = m.x().slice(u16, rows * k);
        var y: f64 = 0;
        var terms: f64 = 0;
        for (0..g_n) |g| {
            var dot: f64 = 0;
            var sum: f64 = 0;
            for (0..m.q.group) |j| {
                const kk = g * m.q.group + j;
                const xv = fromBf(xs[i * k + kk]);
                y += xv * m.deq[c * k + kk];
                dot += @abs(xv * m.codes[c * k + kk]);
                sum += @abs(xv);
            }
            terms += @abs(fromBf(sc[c * g_n + g])) * dot + @abs(fromBf(bi[c * g_n + g])) * sum;
        }
        const gamma = @import("glm_rows_classes.zig").gamma(k);
        return .{ y, std.math.ldexp(@abs(y) + gamma * terms, -8) + gamma * terms };
    }

    /// Expert (or head) e of `parts` stacked matrices: its rows, codes and tables.
    pub fn part(m: Mat, e: usize, parts: usize) affine.Affine {
        var q = m.q;
        const n = q.n / parts;
        const wb = affine.wBytes(n, q.k, q.width());
        const sb = affine.sBytes(n, q.k, q.width());
        q.w = q.w.at(e * wb);
        q.s = q.s.at(e * sb);
        q.b = q.b.at(e * sb);
        q.n = @intCast(n);
        return q;
    }
};

pub const Gpu = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    p: wide.Pipes,
    core: row.Pipelines,
    swiglu: mtl.Pipeline,

    /// One command buffer around `f(e, args)`.
    pub fn run(g: *const Gpu, comptime f: anytype, args: anytype) !void {
        const cb = g.queue.commandBuffer();
        const e = cb.compute(.serial);
        @call(.auto, f, .{e} ++ args);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("GPU failed: {s}\n", .{msg});
            return error.GpuFailed;
        }
    }
};

pub fn dense(e: mtl.ComputeEncoder, p: *const wide.Pipes, q: affine.Affine, io: wide.Io) void {
    wide.dense(p, e, q, io);
}

pub const Tally = struct { values: usize = 0, worst: f64 = 0, pairs: usize = 0 };

pub fn eql(got: []const u16, want: []const u16) !void {
    if (!std.mem.eql(u16, got, want)) {
        for (got, want, 0..) |x, y, i| if (x != y) {
            std.debug.print("first difference at {d}: {x} vs {x}\n", .{ i, x, y });
            break;
        };
        return error.BitsDiffer;
    }
}

/// One matrix: fp64 per column, windows and strided rows equal to each row alone, the core row kernel's bits.
fn denseCheck(g: *const Gpu, m: Mat, t: *Tally) !void {
    const n: usize = m.q.n;
    const k: usize = m.q.k;
    const alone = try buf(g.device, rows * n * 2);
    defer alone.deinit();
    for (0..rows) |i| try g.run(dense, .{ &g.p, m.q, wide.Io{ .in = ref(m.x()).at(i * k * 2), .ldx = m.q.k, .out = ref(alone).at(i * n * 2), .ldo = m.q.n, .rows = 1 } });
    const got = alone.slice(u16, rows * n);
    for (0..rows) |i| for (0..n) |c| {
        const r = m.reference(i, c);
        const err = @abs(fromBf(got[i * n + c]) - r[0]);
        t.worst = @max(t.worst, err / r[1]);
        if (err > r[1]) {
            std.debug.print("row {d} col {d}: {d} vs fp64 {d} (bound {e})\n", .{ i, c, fromBf(got[i * n + c]), r[0], r[1] });
            return error.PastBound;
        }
        t.values += 1;
    };
    if (n % 8 == 0) { // the core row kernel's sums (its tiles: 8 columns): the same bits for a row
        const core = try buf(g.device, n * 2);
        defer core.deinit();
        const call = try row.call(&g.core, .{ .w = m.bufs[0], .scales = m.bufs[1], .biases = m.bufs[2], .n = n, .k = k, .group = m.q.group, .bits = m.q.bits }, 1, false);
        try g.run(coreRow, .{ call, m, core });
        try eql(core.slice(u16, n), got[0..n]);
        t.pairs += n;
    }
    // windows, contiguous (two rows a weight read) and strided (one at a time, out rows apart with a lead offset)
    const xs = try buf(g.device, rows * (k + pad) * 2);
    defer xs.deinit();
    for (0..rows) |i| @memcpy(xs.slice(u16, rows * (k + pad))[i * (k + pad) ..][0..k], m.x().slice(u16, rows * k)[i * k ..][0..k]);
    const out = try buf(g.device, (rows * (n + pad) + 4) * 2);
    defer out.deinit();
    for ([_]usize{ 2, 3, 16 }) |w| for ([_]bool{ false, true }) |strided| {
        const ld: usize = if (strided) n + pad else n;
        const at: usize = if (strided) 4 else 0;
        try g.run(dense, .{ &g.p, m.q, wide.Io{ .in = ref(if (strided) xs else m.x()), .ldx = @intCast(if (strided) k + pad else k), .out = ref(out).at(at * 2), .ldo = @intCast(ld), .rows = @intCast(w) } });
        for (0..w) |i| try eql(out.slice(u16, rows * (n + pad) + 4)[at + i * ld ..][0..n], got[i * n ..][0..n]);
        t.pairs += w * n;
    };
}

fn coreRow(e: mtl.ComputeEncoder, c: row.Call, m: Mat, out: mtl.Buffer) void {
    e.setPipeline(c.pipeline);
    e.setBuffer(m.x(), 0, 0);
    e.setBuffer(m.bufs[0], 0, 1);
    e.setBuffer(m.bufs[1], 0, 2);
    e.setBuffer(m.bufs[2], 0, 3);
    e.setValue(c.dims, 4);
    e.setBuffer(out, 0, 5);
    e.dispatchGroups(mtl.Size.of(c.groups, c.slots, 1), mtl.Size.of(32 * row.groups, 1, 1));
}

/// Four heads (or experts): per head, by pick (GLU and plain) and sorted (8, 5, 2, 1 pairs), each against dense.
fn partsCheck(g: *const Gpu, m: Mat, up: Mat, t: *Tally) !void {
    const parts = 4;
    const k: usize = m.q.k;
    const r: usize = m.q.n / parts;
    const n_rows = 3; // x rows a head
    const repeat = 2; // picks an x row: 8 rows, so an expert's sorted pairs fill 4- and 8-pair passes and leave tails
    const slots = 8 * repeat;
    // per head: x [rows, heads, k], out [rows, heads, r]
    const ph = try buf(g.device, n_rows * parts * r * 2);
    defer ph.deinit();
    try g.run(perHead, .{ &g.p, m.q, @as(u32, parts), wide.Io{ .in = ref(m.x()), .ldx = @intCast(parts * k), .out = ref(ph), .ldo = @intCast(parts * r), .rows = n_rows }, @as(u32, @intCast(k)), @as(u32, @intCast(r)) });
    const one = try buf(g.device, n_rows * r * 2);
    defer one.deinit();
    for (0..parts) |h| {
        try g.run(dense, .{ &g.p, m.part(h, parts), wide.Io{ .in = ref(m.x()).at(h * k * 2), .ldx = @intCast(parts * k), .out = ref(one), .ldo = @intCast(r), .rows = n_rows } });
        for (0..n_rows) |i| try eql(ph.slice(u16, n_rows * parts * r)[(i * parts + h) * r ..][0..r], one.slice(u16, n_rows * r)[i * r ..][0..r]);
        t.pairs += n_rows * r;
    }
    // by pick: slot y through expert ids[y] on x row y / repeat
    const ids = try buf(g.device, slots * 4);
    defer ids.deinit();
    const picks = [slots]i32{ 2, 0, 0, 2, 2, 1, 3, 2, 2, 0, 1, 2, 2, 0, 0, 2 };
    @memcpy(ids.slice(i32, slots), &picks);
    var gate = m.q;
    gate.n = @intCast(r);
    var upq = up.q;
    upq.n = @intCast(r);
    const plain = try buf(g.device, slots * r * 2);
    defer plain.deinit();
    const glu = try buf(g.device, slots * r * 2);
    defer glu.deinit();
    const lim: f32 = 0.75;
    try g.run(byIds, .{ &g.p, wide.Picks{ .w = gate, .experts = parts, .repeat = repeat }, ref(m.x()), ref(ids), ref(plain), @as(u32, slots) });
    try g.run(byIds, .{ &g.p, wide.Picks{ .w = gate, .up = upq, .experts = parts, .repeat = repeat, .lim = lim }, ref(m.x()), ref(ids), ref(glu), @as(u32, slots) });
    const gu = try buf(g.device, slots * 2 * r * 2);
    defer gu.deinit();
    const act = try buf(g.device, slots * r * 2);
    defer act.deinit();
    for (0..slots) |y| {
        const e: usize = @intCast(picks[y]);
        const x_at = ref(m.x()).at(y / repeat * k * 2);
        try g.run(dense, .{ &g.p, m.part(e, parts), wide.Io{ .in = x_at, .ldx = @intCast(k), .out = ref(gu).at(y * 2 * r * 2), .ldo = @intCast(2 * r), .rows = 1 } });
        try g.run(dense, .{ &g.p, up.part(e, parts), wide.Io{ .in = x_at, .ldx = @intCast(k), .out = ref(gu).at((y * 2 * r + r) * 2), .ldo = @intCast(2 * r), .rows = 1 } });
        try eql(plain.slice(u16, slots * r)[y * r ..][0..r], gu.slice(u16, slots * 2 * r)[y * 2 * r ..][0..r]);
    }
    try g.run(swiglu, .{ g.swiglu, ref(gu), ref(act), @as(u32, slots), @as(u32, @intCast(r)), lim });
    try eql(glu.slice(u16, slots * r), act.slice(u16, slots * r));
    t.pairs += 2 * slots * r;
    // sorted: each expert's pairs in pick order; GLU out row i, down out row order[i] from x row i
    var offsets: [parts]i32 = undefined; // one per expert, as the route's sort writes them
    var order: [slots]i32 = undefined;
    var at: usize = 0;
    for (0..parts) |e| {
        offsets[e] = @intCast(at);
        for (picks, 0..) |pk, y| if (pk == e) {
            order[at] = @intCast(y);
            at += 1;
        };
    }
    const ob = try buf(g.device, (parts + slots) * 4);
    defer ob.deinit();
    @memcpy(ob.slice(i32, parts), &offsets);
    @memcpy(ob.slice(i32, parts + slots)[parts..], &order);
    const order_ref = ref(ob).at(parts * 4);
    const sorted = try buf(g.device, slots * r * 2);
    defer sorted.deinit();
    try g.run(bySorted, .{ &g.p, wide.Picks{ .w = gate, .up = upq, .experts = parts, .repeat = repeat, .lim = lim }, ref(m.x()), ref(ob), order_ref, ref(sorted), @as(u32, @intCast(r)), @as(u32, slots) });
    for (0..slots) |i| try eql(sorted.slice(u16, slots * r)[i * r ..][0..r], glu.slice(u16, slots * r)[@as(usize, @intCast(order[i])) * r ..][0..r]);
    // the down: x row i (the sorted pairs' inputs, here x rows by order) into out row order[i]
    const xs = try buf(g.device, slots * k * 2);
    defer xs.deinit();
    for (0..slots) |i| @memcpy(xs.slice(u16, slots * k)[i * k ..][0..k], m.x().slice(u16, rows * k)[@as(usize, @intCast(order[i])) * k ..][0..k]);
    try g.run(bySorted, .{ &g.p, wide.Picks{ .w = gate, .experts = parts, .repeat = 1 }, ref(xs), ref(ob), order_ref, ref(sorted), @as(u32, @intCast(r)), @as(u32, slots) });
    try g.run(byIds, .{ &g.p, wide.Picks{ .w = gate, .experts = parts, .repeat = 1 }, ref(m.x()), ref(ids), ref(plain), @as(u32, slots) });
    try eql(sorted.slice(u16, slots * r), plain.slice(u16, slots * r));
    t.pairs += 2 * slots * r;
}

pub fn mmDense(
    e: mtl.ComputeEncoder,
    mm: affine_mm.Pipes,
    q: affine.Affine,
    x: Ref,
    sums: Ref,
    y: Ref,
    n_rows: u32,
) void {
    affine_mm.rowSums(e, mm, 64, x, sums, n_rows, q.k);
    affine_mm.dense(e, mm, x, sums, q, y, n_rows);
}

/// dims: x rows, sorted pairs, experts, picks a row.
fn mmGlu(e: mtl.ComputeEncoder, mm: affine_mm.Pipes, gate: affine.Affine, up: affine.Affine, x: Ref, sums: Ref, sorted: [2]Ref, act: Ref, dims: [4]u32, lim: f32) void {
    affine_mm.rowSums(e, mm, 64, x, sums, dims[0], gate.k);
    affine_mm.gatherGlu(e, mm, x, sums, gate, up, sorted[0], sorted[1], act, dims[1], dims[2], dims[3], lim);
}

fn mmDown(e: mtl.ComputeEncoder, mm: affine_mm.Pipes, q: affine.Affine, x: Ref, sums: Ref, sorted: [2]Ref, y: Ref, slots: u32, experts: u32) void {
    affine_mm.rowSums(e, mm, 64, x, sums, slots, q.k);
    affine_mm.gatherTo(e, mm, x, sums, q, sorted[0], y, sorted[1], slots, experts);
}

/// oQ2's prompt experts, affine_mm at 2/64: gathered GLU == glm_swiglu(dense gate | up), gathered down == dense.
fn gatherCheck(g: *const Gpu, m: Mat, up: Mat, t: *Tally) !void {
    const parts = 4;
    const k: usize = m.q.k;
    const r: usize = m.q.n / parts;
    const repeat = 2;
    const slots = 8 * repeat;
    const picks = [slots]i32{ 2, 0, 0, 2, 2, 1, 3, 2, 2, 0, 1, 2, 2, 0, 0, 2 };
    const lim: f32 = 0.75;
    const mm = g.p.mm2;
    var order: [slots]i32 = undefined;
    var offsets: [parts]i32 = undefined;
    var at: usize = 0;
    for (0..parts) |e| {
        offsets[e] = @intCast(at);
        for (picks, 0..) |pk, y| if (pk == e) {
            order[at] = @intCast(y);
            at += 1;
        };
    }
    const ob = try buf(g.device, (parts + slots) * 4);
    defer ob.deinit();
    @memcpy(ob.slice(i32, parts), &offsets);
    @memcpy(ob.slice(i32, parts + slots)[parts..], &order);
    const sorted = [2]Ref{ ref(ob), ref(ob).at(parts * 4) };
    const bs = [_]mtl.Buffer{ try buf(g.device, rows * (k / 64) * 4), try buf(g.device, 2 * parts * rows * r * 2), try buf(g.device, slots * 2 * r * 2), try buf(g.device, 2 * slots * r * 2), try buf(g.device, slots * k * 2), try buf(g.device, (parts + 1) * slots * r * 2) };
    defer for (bs) |b| b.deinit();
    const sums = ref(bs[0]);
    var gate = m.q;
    gate.n = @intCast(r);
    var upq = up.q;
    upq.n = @intCast(r);
    // every expert's gate and up, dense on every x row: [2][parts][rows, r]
    for (0..parts) |e| for ([2]Mat{ m, up }, 0..) |w, j| {
        try g.run(mmDense, .{ mm, w.part(e, parts), ref(m.x()), sums, ref(bs[1]).at(((j * parts + e) * rows * r) * 2), @as(u32, rows) });
    };
    const dg = bs[1].slice(u16, 2 * parts * rows * r);
    const gu = bs[2].slice(u16, slots * 2 * r);
    for (order, 0..) |o, i| {
        const e: usize = @intCast(picks[@intCast(o)]);
        const xr: usize = @as(usize, @intCast(o)) / repeat;
        for (0..2) |j| @memcpy(gu[(2 * i + j) * r ..][0..r], dg[((j * parts + e) * rows + xr) * r ..][0..r]);
    }
    const act = ref(bs[3]);
    try g.run(mmGlu, .{ mm, gate, upq, ref(m.x()), sums, sorted, act, [4]u32{ rows, slots, parts, repeat }, lim });
    try g.run(swiglu, .{ g.swiglu, ref(bs[2]), act.at(slots * r * 2), @as(u32, slots), @as(u32, @intCast(r)), lim });
    const both = bs[3].slice(u16, 2 * slots * r);
    try eql(both[0 .. slots * r], both[slots * r ..]);
    // the down: sorted pair i's input (x row order[i]) into row order[i], == its expert's dense product of that row
    const xs = bs[4].slice(u16, slots * k);
    for (order, 0..) |o, i| @memcpy(xs[i * k ..][0..k], m.x().slice(u16, rows * k)[@as(usize, @intCast(o)) * k ..][0..k]);
    const y = bs[5].slice(u16, (parts + 1) * slots * r);
    try g.run(mmDown, .{ mm, gate, ref(bs[4]), sums, sorted, ref(bs[5]), @as(u32, slots), @as(u32, parts) });
    for (0..parts) |e| try g.run(mmDense, .{ mm, m.part(e, parts), ref(bs[4]), sums, ref(bs[5]).at((1 + e) * slots * r * 2), @as(u32, slots) });
    for (order, 0..) |o, i| {
        const e: usize = @intCast(picks[@intCast(o)]);
        try eql(y[@as(usize, @intCast(o)) * r ..][0..r], y[((1 + e) * slots + i) * r ..][0..r]);
    }
    t.pairs += 2 * slots * r;
}

/// A split stack (a run a member, each its width) in one call, at the decode's and the prompt's pitch == each alone.
fn splitCheck(g: *const Gpu, k: usize, ns: []const usize, bits: []const u8, t: *Tally) !void {
    var ms: [affine.max_runs]Mat = undefined;
    for (ns, bits, 0..) |n, b, i| ms[i] = try Mat.init(g.device, n, k, b, 64, 500 + 31 * i + n);
    defer for (ms[0..ns.len]) |m| m.deinit();
    var q = ms[0].q;
    var at: u32 = @intCast(ns[0]);
    for (ms[1..ns.len], 0..) |m, i| {
        q.runs[i] = .{ .w = m.q.w, .s = m.q.s, .b = m.q.b, .row = at, .n = m.q.n, .bits = m.q.bits, .group = m.q.group };
        at += m.q.n;
    }
    q.more = @intCast(ns.len - 1);
    q.n = at;
    const x = ref(ms[0].x());
    for ([_]usize{ at, std.mem.alignForward(usize, at, 64) }) |ld| {
        const out = try buf(g.device, rows * ld * 2);
        defer out.deinit();
        try g.run(dense, .{ &g.p, q, wide.Io{ .in = x, .ldx = @intCast(k), .out = ref(out), .ldo = @intCast(ld), .rows = rows } });
        for (ms[0..ns.len], 0..) |m, i| {
            const r = q.run(i);
            const one = try buf(g.device, rows * r.n * 2);
            defer one.deinit();
            try g.run(dense, .{ &g.p, m.q, wide.Io{ .in = x, .ldx = @intCast(k), .out = ref(one), .ldo = r.n, .rows = rows } });
            for (0..rows) |y| try eql(out.slice(u16, rows * ld)[y * ld + r.row ..][0..r.n], one.slice(u16, rows * r.n)[y * r.n ..][0..r.n]);
            t.pairs += rows * r.n;
        }
    }
}

pub fn gluLaunch(
    e: mtl.ComputeEncoder,
    p: *const wide.Pipes,
    gu: affine.Affine,
    n: u32,
    x: Ref,
    out: Ref,
    m: u32,
) void {
    wide.glu(p, e, gu, n, x, out, m, 0.75);
}

fn perHead(e: mtl.ComputeEncoder, p: *const wide.Pipes, q: affine.Affine, heads: u32, io: wide.Io, hsx: u32, hso: u32) void {
    wide.perHead(p, e, q, heads, io, hsx, hso);
}
pub fn byIds(e: mtl.ComputeEncoder, p: *const wide.Pipes, m: wide.Picks, x: Ref, ids: Ref, out: Ref, slots: u32) void {
    wide.byIds(p, e, m, x, ids, out, slots);
}
fn bySorted(e: mtl.ComputeEncoder, p: *const wide.Pipes, m: wide.Picks, x: Ref, offsets: Ref, order: Ref, out: Ref, ldo: u32, pairs: u32) void {
    wide.bySorted(p, e, m, x, offsets, order, out, ldo, pairs);
}
pub fn swiglu(e: mtl.ComputeEncoder, pipe: mtl.Pipeline, gu: Ref, act: Ref, n_rows: u32, width: u32, lim: f32) void {
    e.setPipeline(pipe);
    e.setBuffer(gu.buf, gu.off, 0);
    e.setBuffer(act.buf, act.off, 1);
    e.setValue(extern struct { rows: i32, width: i32, x_stride: i32, y_stride: i32, eps: f32 }{ .rows = @intCast(n_rows), .width = @intCast(width), .x_stride = @intCast(2 * width), .y_stride = @intCast(width), .eps = lim }, 2);
    e.dispatchThreads(mtl.Size.of(width, n_rows, 1), mtl.Size.of(256, 1, 1));
}

pub fn main() !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const glue = try mtl.Library.fromSource(device, sources.glm_glue, mtl.CompileOptions.mlx());
    defer glue.deinit();
    var g = Gpu{ .device = device, .queue = queue, .p = try wide.load(a, device), .core = try row.Pipelines.load(device), .swiglu = try mtl.Pipeline.init(device, glue, "glm_swiglu", false) };
    defer g.p.deinit();
    defer g.core.deinit();
    defer g.swiglu.deinit();
    std.debug.print("GPU {s}, tensor units: {}\n", .{ device.name(), device.tensorUnits() });
    var all = Tally{};
    const shapes = [_][2]usize{ .{ 132, 384 }, .{ 64, 2688 }, .{ 40, 1536 } };
    for (affine.row_bits) |bits| for ([_]usize{ 32, 64, 128 }) |group| {
        var t = Tally{};
        for (shapes, 0..) |s, si| {
            const m = try Mat.init(device, s[0], s[1], bits, group, 1000 * @as(u64, bits) + 10 * group + si);
            defer m.deinit();
            try denseCheck(&g, m, &t);
            if (s[0] == 64) {
                const up = try Mat.init(device, s[0], s[1], bits, group, 7 + 1000 * @as(u64, bits) + 10 * group + si);
                defer up.deinit();
                try partsCheck(&g, m, up, &t);
            }
        }
        // oQ2's routed experts (and oQ4's at 4 bits): four experts [2048, 4096] (gate/up) and [4096, 2048] (down)
        if (group == 64 and (bits == 2 or bits == 4)) for ([_][2]usize{ .{ 4 * 2048, 4096 }, .{ 4 * 4096, 2048 } }, 0..) |s, si| {
            const m = try Mat.init(device, s[0], s[1], bits, group, 77 + 1000 * @as(u64, bits) + si);
            defer m.deinit();
            const up = try Mat.init(device, s[0], s[1], bits, group, 78 + 1000 * @as(u64, bits) + si);
            defer up.deinit();
            try partsCheck(&g, m, up, &t);
            if (bits == 2) try gatherCheck(&g, m, up, &t); // the prompt's 2/64 experts on affine_mm
        };
        std.debug.print("q{d} g{d}: {d} values vs fp64, worst err/bound {d:.4}; {d} bit-equal pairs (windows, strides, heads, picks, sorted, core)\n", .{ bits, group, t.values, t.worst, t.pairs });
        all.values += t.values;
        all.pairs += t.pairs;
        all.worst = @max(all.worst, t.worst);
    };
    var split = Tally{};
    try splitCheck(&g, 256, &.{ 64, 64, 64, 16, 16, 8 }, &.{ 5, 8, 5, 8, 8, 8 }, &split); // oQ4 layer 6's KDA input
    try splitCheck(&g, 512, &.{ 48, 16, 8, 8 }, &.{ 6, 6, 5, 5 }, &split); // oQ4 layer 7's and the MTP's MLA x_proj
    std.debug.print("split stacks through one call == each member alone: {d} bit-equal values\n", .{split.pairs});
    var sm = Tally{};
    const kda_mm = try batch.splitMmCheck(
        &g,
        256,
        &.{ 128, 128, 128, 64, 64, 64 },
        &.{ 5, 8, 5, 8, 8, 8 },
        &sm,
    ); // oQ4 layer 6
    const x_mm = try batch.splitMmCheck(
        &g,
        512,
        &.{ 192, 64, 64, 32 },
        &.{ 6, 6, 5, 5 },
        &sm,
    ); // layer 7's x_proj: 32 on rows
    std.debug.print(
        "prompt split stacks == each member alone ({d} + {d} members on affine_mm): {d} bit-equal values\n",
        .{ kda_mm, x_mm, sm.pairs },
    );
    all.pairs += sm.pairs;
    var hd = Tally{};
    for ([_][2]usize{ .{ 512, 256 }, .{ 256, 512 } }) |s| try batch.headsCheck(
        &g,
        g.p.mm8,
        8,
        s[0],
        s[1],
        &hd,
    ); // embed_q, unembed_out
    std.debug.print(
        "prompt embed_q / unembed_out q8: the heads in one affine_mm batch == each head alone: {d} bit-equal values\n",
        .{hd.pairs},
    );
    all.pairs += hd.pairs;
    all.pairs += split.pairs;
    all.pairs += try @import("glm_rows_kda.zig").check(&g.p, device, queue);
    all.pairs += try @import("glm_rows_kda.zig").fusedCheck(&g.p, device, queue);
    all.pairs += try batch.sharedGluCheck(&g);
    all.pairs += try @import("glm_rows_moe.zig").check(&g);
    all.values += try @import("glm_rows_kda.zig").bf16Check(&g.p, device, queue, &all.worst);
    all.values += try @import("glm_rows_kda.zig").embedCheck(&g.p, device, queue, &all.worst);
    const cl = try @import("glm_rows_classes.zig").check(&g.p, device, queue);
    all.values += cl.values;
    all.pairs += cl.pairs;
    all.worst = @max(all.worst, cl.worst);
    std.debug.print("PASS {d} values within the fp32-sum bound (worst err/bound {d:.4}), {d} bit-equal pairs\n", .{ all.values, all.worst, all.pairs });
}
