//! The shared prompt products against float64 on synthetic weights, and every row's bits in every window width.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const mm = tf.affine_mm;
const sq = tf.row_projection.simd;

const opts = mtl.ResourceOptions.shared;
const a = std.heap.page_allocator;
const unit: f64 = 0x1p-24; // fp32's unit roundoff
const wide = 129; // rows of the reference call: two 64-row tiles and one more
const windows = [_]usize{ 1, 2, 3, 8, 16, 17, 33, 64, 65 }; // 17, 33 and 65 cross the 16-, 32- and 64-row tiles

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn fromBf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}

/// Synthetic MLX affine weights [n, k] (codes little-endian in bytes) and `wide` rows of x; `deq` s q + b exactly.
const Synth = struct {
    deqr: []f64, // bf16(fma(s, q, b)): the weights as affine_mm's tile dequantizes them
    n: usize,
    k: usize,
    bits: u8,
    group: usize,
    x: mtl.Buffer,
    w: mtl.Buffer,
    s: mtl.Buffer,
    b: mtl.Buffer,
    deq: []f64,

    fn init(device: mtl.Device, n: usize, k: usize, bits: u8, group: usize, seed: u64) !Synth {
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        const sn = n * (k / group);
        const self = Synth{ .n = n, .k = k, .bits = bits, .group = group, .x = try device.buffer(wide * k * 2, opts), .w = try device.buffer(n * k * bits / 8, opts), .s = try device.buffer(sn * 2, opts), .b = try device.buffer(sn * 2, opts), .deq = try a.alloc(f64, n * k), .deqr = try a.alloc(f64, n * k) };
        for (self.x.slice(u16, wide * k)) |*v| v.* = toBf(r.float(f32) * 2 - 1);
        r.bytes(self.w.slice(u8, n * k * bits / 8));
        for (self.s.slice(u16, sn), self.b.slice(u16, sn)) |*s, *b| {
            s.* = toBf((r.float(f32) + 0.5) / 64);
            b.* = toBf(-(r.float(f32) + 0.5) * 0.15);
        }
        for (0..n) |j| for (0..k) |i| {
            const g = j * (k / group) + i / group;
            const s32: f32 = @floatCast(fromBf(self.s.slice(u16, sn)[g]));
            const b32: f32 = @floatCast(fromBf(self.b.slice(u16, sn)[g]));
            self.deq[j * k + i] = @as(f64, s32) * @as(f64, @floatFromInt(self.code(j, i))) + @as(f64, b32);
            self.deqr[j * k + i] = fromBf(toBf(@mulAdd(f32, s32, @floatFromInt(self.code(j, i)), b32)));
        };
        return self;
    }

    fn deinit(self: Synth) void {
        for ([_]mtl.Buffer{ self.x, self.w, self.s, self.b }) |x| x.deinit();
        a.free(self.deq);
        a.free(self.deqr);
    }

    fn code(self: Synth, j: usize, i: usize) u32 {
        const bytes = self.w.slice(u8, self.n * self.k * self.bits / 8);
        const at = (j * self.k + i) * self.bits; // MLX's little-endian bit stream at every width
        var v: u32 = 0;
        for (0..3) |b| if (at / 8 + b < bytes.len) {
            v |= @as(u32, bytes[at / 8 + b]) << @intCast(8 * b);
        };
        return (v >> @intCast(at % 8)) & ((@as(u32, 1) << @intCast(self.bits)) - 1);
    }

    /// y = x W^T in float64 on deq or (`rounded`) deqr, the sizes its fp32 sums carry, and the scales' share.
    fn reference(self: Synth, rounded: bool) !Ref {
        const n = self.n;
        const k = self.k;
        const groups = k / self.group;
        const ref = Ref{ .want = try a.alloc(f64, wide * n), .mag = try a.alloc(f64, wide * n), .scaled = try a.alloc(f64, wide * n) };
        const x = self.x.slice(u16, wide * k);
        const sc = self.s.slice(u16, n * groups);
        const xd = try a.alloc(f64, k);
        defer a.free(xd);
        const bi = self.b.slice(u16, n * groups);
        const w = if (rounded) self.deqr else self.deq;
        for (0..wide) |r| {
            for (xd, 0..) |*v, i| v.* = fromBf(x[r * k + i]);
            for (0..n) |j| {
                var sum: f64 = 0;
                var mag: f64 = 0;
                var scaled: f64 = 0;
                for (0..groups) |g| {
                    var qx: f64 = 0;
                    var qxs: f64 = 0;
                    var xg: f64 = 0;
                    for (g * self.group..(g + 1) * self.group) |i| {
                        const q: f64 = @floatFromInt(self.code(j, i));
                        sum += xd[i] * w[j * k + i];
                        if (rounded) mag += @abs(xd[i] * w[j * k + i]);
                        qx += q * @abs(xd[i]);
                        qxs += q * xd[i];
                        xg += @abs(xd[i]);
                    }
                    const s = fromBf(sc[j * groups + g]);
                    if (!rounded) mag += s * qx + @abs(fromBf(bi[j * groups + g])) * xg;
                    scaled += s * qxs;
                }
                ref.want[r * n + j] = sum;
                ref.mag[r * n + j] = mag;
                ref.scaled[r * n + j] = scaled;
            }
        }
        return ref;
    }
};

/// The float64 product, its terms' fp32 magnitudes, and its scales' share (a control: those scales 1% off must fail).
const Ref = struct {
    want: []f64,
    mag: []f64,
    scaled: []f64,
    fn deinit(r: Ref) void {
        for ([_][]f64{ r.want, r.mag, r.scaled }) |v| a.free(v);
    }
};

/// Every check's outcome: failures, and the largest difference as a fraction of its bound.
const Tally = struct { failed: usize = 0, worst: f64 = 0 };

/// One product's output: `rows` rows of n bf16 or fp32 values.
const Out = struct {
    buf: mtl.Buffer,
    f32_out: bool,
    fn esize(o: Out) usize {
        return if (o.f32_out) 4 else 2;
    }
    fn get(o: Out, i: usize) f64 {
        return if (o.f32_out) o.buf.slice(f32, i + 1)[i] else fromBf(o.buf.slice(u16, i + 1)[i]);
    }
};

/// The bound: fp32 sums over 16-deep tensor-unit slices and their chain, or a group and a chain of groups; a bf16 ulp.
fn bound(want: f64, mag: f64, s: *const Synth, f32_out: bool, extra: usize, tensor: bool) f64 {
    const terms: f64 = @floatFromInt((if (tensor) 16 + s.k / 16 else s.group + s.k / s.group) + 8 + extra);
    return terms * unit * mag + (if (f32_out) 0 else @abs(want) * 0x1p-7) + 1e-30;
}

/// `rows` rows of `out` against the float64 reference: the worst column's max abs and relative error, and the bound.
fn compare(t: *Tally, name: []const u8, out: Out, rows: usize, ref: Ref, s: *const Synth, extra: usize, tensor: bool) void {
    const n = s.n;
    var col: usize = 0;
    var col_abs: f64 = -1;
    var col_rel: f64 = 0;
    var frac: f64 = 0;
    var ctl: f64 = 0;
    var at: usize = 0;
    for (0..n) |j| {
        var abs: f64 = 0;
        var top: f64 = 0;
        for (0..rows) |r| {
            const i = r * n + j;
            const d = @abs(out.get(i) - ref.want[i]);
            const tol = bound(ref.want[i], ref.mag[i], s, out.f32_out, extra, tensor);
            const q = d / tol;
            ctl = @max(ctl, @abs(out.get(i) - ref.want[i] - 0.01 * ref.scaled[i]) / tol);
            if (!(d <= abs)) abs = d;
            top = @max(top, @abs(ref.want[i]));
            if (!(q <= frac)) {
                frac = q;
                at = i;
            }
        }
        if (!(abs <= col_abs)) {
            col_abs = abs;
            col = j;
            col_rel = abs / top;
        }
    }
    t.worst = @max(t.worst, frac);
    const ok = frac <= 1 and ctl > 1;
    if (!ok) t.failed += 1;
    std.debug.print("RESULT {s}: worst column {d}: max abs {e:.3}, relative {e:.3}; {d:.4} of its bound (row {d} col {d}), " ++
        "scales 1% off {d:.1}{s}\n", .{ name, col, col_abs, col_rel, frac, at / n, at % n, ctl, if (ok) "" else " FAIL" });
}

/// Rows [0, rows) of `got` equal `want` byte for byte; the first difference printed.
fn sameRows(t: *Tally, name: []const u8, m: usize, want: Out, got: Out, rows: usize, n: usize) bool {
    const bytes = rows * n * want.esize();
    const w = want.buf.slice(u8, bytes);
    const g = got.buf.slice(u8, bytes);
    if (std.mem.eql(u8, w, g)) return true;
    const i = std.mem.indexOfDiff(u8, w, g).? / want.esize();
    var differ: usize = 0;
    var most: f64 = 0;
    for (0..rows) |r| {
        const row = r * n * want.esize();
        if (!std.mem.eql(u8, w[row..][0 .. n * want.esize()], g[row..][0 .. n * want.esize()])) differ += 1;
        for (r * n..(r + 1) * n) |e| most = @max(most, @abs(got.get(e) - want.get(e)));
    }
    std.debug.print("RESULT {s}: {d}-row windows DIFFER in {d} of {d} rows, by up to {e:.3}; first row {d} col {d}: {d} against {d} in the {d}-row call\n", .{ name, m, differ, rows, most, i / n, i % n, got.get(i), want.get(i), wide });
    t.failed += 1;
    return false;
}

fn newOut(device: mtl.Device, s: *const Synth, f32_out: bool) !Out {
    const o = Out{ .buf = try device.buffer(wide * s.n * 4, opts), .f32_out = f32_out };
    @memset(o.buf.slice(u8, wide * s.n * 4), 0xff); // NaN: a launch that writes nothing fails
    return o;
}

fn commit(cb: anytype, e: mtl.ComputeEncoder) !void {
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.debug.print("GPU failed: {s}\n", .{why});
        return error.GpuFailed;
    }
}

const Kernel = enum { dense, gather };

/// affine_mm over rows [start, start + m): the dense product or a gather with one expert.
fn affineWindow(e: mtl.ComputeEncoder, p: mm.Pipes, s: *const Synth, offs: mtl.Buffer, y: Out, kernel: Kernel, start: usize, m: usize) void {
    const x = .{ .buf = s.x, .off = start * s.k * 2 };
    const yo = .{ .buf = y.buf, .off = start * s.n * y.esize() };
    const z: usize = 0;
    const q = .{ .w = .{ .buf = s.w, .off = z }, .s = .{ .buf = s.s, .off = z }, .b = .{ .buf = s.b, .off = z }, .n = @as(u32, @intCast(s.n)), .k = @as(u32, @intCast(s.k)) };
    switch (kernel) {
        .dense => mm.dense(e, p, x, q, yo, @intCast(m)),
        .gather => mm.gather(e, p, x, q, .{ .buf = offs, .off = z }, yo, @intCast(m), 1),
    }
}

/// affine_mm at one format: the wide call against float64, then every window's rows against the wide call's.
fn affine(t: *Tally, device: mtl.Device, queue: mtl.Queue, s: *const Synth, ref: Ref, f32_out: bool) !void {
    const src = try mm.source(device, a, .{ .bits = s.bits, .group = @intCast(s.group), .out_f32 = f32_out });
    defer a.free(src);
    const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var p: mm.Pipes = undefined;
    for (mm.names, 0..) |name, i| p[i] = try mtl.Pipeline.init(device, lib, name, false);
    defer for (p) |x| x.deinit();
    const offs = try device.buffer(4, opts);
    defer offs.deinit();
    offs.slice(i32, 1)[0] = 0;
    var name: [96]u8 = undefined;
    for ([_]Kernel{ .dense, .gather }) |kernel| {
        const label = try std.fmt.bufPrint(&name, "affine_mm {s} q{d} g{d} {s} [{d}, {d}]", .{ @tagName(kernel), s.bits, s.group, if (f32_out) "fp32" else "bf16", s.n, s.k });
        const all = try newOut(device, s, f32_out);
        defer all.buf.deinit();
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        affineWindow(e, p, s, offs, all, kernel, 0, wide);
        try commit(cb, e);
        compare(t, label, all, wide, ref, s, 0, true);
        var same = true;
        for (windows) |m| {
            const got = try newOut(device, s, f32_out);
            defer got.buf.deinit();
            const wb = queue.commandBuffer();
            const we = wb.compute(.serial);
            var start: usize = 0;
            while (start + m <= wide) : (start += m) affineWindow(we, p, s, offs, got, kernel, start, m);
            try commit(wb, we);
            same = sameRows(t, label, m, all, got, wide / m * m, s.n) and same;
        }
        if (same) std.debug.print("RESULT {s}: every row's bits equal in windows of 1, 2, 3, 8, 16, 17, 33, 64 and 65 rows\n", .{label});
    }
}

/// simd_qmm over rows [start, start + m), the scalar twin when `scalar` and it takes m rows.
fn simdWindow(e: mtl.ComputeEncoder, p: *const sq.Pipelines, s: *const Synth, y: Out, scalar: bool, start: usize, m: usize) !void {
    const mat = sq.Matrix{ .w = s.w, .scales = s.s, .biases = s.b, .n = s.n, .k = s.k };
    sq.encode(e, try sq.call(p, s.n, s.k, m, scalar), mat, s.x, start * s.k * 2, y.buf, start * s.n * 2);
}

/// simd_qmm (4-bit, group 64): tiles at every window against float64 and the wide call; the twin at 1 and 2 rows.
fn simd(t: *Tally, device: mtl.Device, queue: mtl.Queue, p: *const sq.Pipelines, s: *const Synth, ref: Ref) !void {
    var name: [96]u8 = undefined;
    const label = try std.fmt.bufPrint(&name, "simd_qmm tiles q4 g64 bf16 [{d}, {d}]", .{ s.n, s.k });
    const all = try newOut(device, s, false);
    defer all.buf.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    try simdWindow(e, p, s, all, false, 0, wide);
    try commit(cb, e);
    const extra = std.math.log2_int(u32, sq.splits(s.n)) + 1; // the chunks' fixed tree
    compare(t, label, all, wide, ref, s, extra, false);
    var same = true;
    for (windows) |m| {
        const got = try newOut(device, s, false);
        defer got.buf.deinit();
        const wb = queue.commandBuffer();
        const we = wb.compute(.serial);
        var start: usize = 0;
        while (start + m <= wide) : (start += m) try simdWindow(we, p, s, got, false, start, m);
        try commit(wb, we);
        same = sameRows(t, label, m, all, got, wide / m * m, s.n) and same;
    }
    if (same) std.debug.print("RESULT {s}: every row's bits equal in windows of 1, 2, 3, 8, 16, 17, 33, 64 and 65 rows\n", .{label});
    // the twin: callers take it only where twinExact says it equals the tiles, so a difference fails only then
    const exact = try sq.twinExact(p, device, .{ .w = s.w, .scales = s.s, .biases = s.b, .n = s.n, .k = s.k });
    for ([_]usize{ 1, 2 }) |m| {
        const twin = try std.fmt.bufPrint(&name, "simd_qmm twin q4 g64 bf16 [{d}, {d}] at {d} rows", .{ s.n, s.k, m });
        const got = try newOut(device, s, false);
        defer got.buf.deinit();
        const wb = queue.commandBuffer();
        const we = wb.compute(.serial);
        var start: usize = 0;
        while (start + m <= wide) : (start += m) try simdWindow(we, p, s, got, true, start, m);
        try commit(wb, we);
        compare(t, twin, got, wide / m * m, ref, s, 0, false);
        const rows = wide / m * m;
        const eq = std.mem.eql(u8, all.buf.slice(u8, rows * s.n * 2), got.buf.slice(u8, rows * s.n * 2));
        std.debug.print("RESULT {s}: {s} the tiles' bits; twinExact says {s}{s}\n", .{ twin, if (eq) "equals" else "differs from", if (exact) "equal" else "differ", if (exact and !eq) " FAIL" else "" });
        if (exact and !eq) t.failed += 1;
    }
}

pub fn main(_: std.process.Init) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    std.debug.print("device {s}, tensor units {}\n", .{ device.name(), device.tensorUnits() });
    var t = Tally{};
    var seed: u64 = 0x716;
    for ([_][2]usize{ .{ 192, 1024 }, .{ 256, 4096 } }) |shape| for ([_]u8{ 4, 8 }) |bits| for ([_]usize{ 32, 64 }) |group| {
        seed += 1;
        const s = try Synth.init(device, shape[0], shape[1], bits, group, seed);
        defer s.deinit();
        const ref = try s.reference(true);
        defer ref.deinit();
        for ([_]bool{ false, true }) |f32_out| try affine(&t, device, queue, &s, ref, f32_out);
    };
    for ([_][2]usize{ .{ 192, 1024 }, .{ 256, 4096 } }) |shape| for ([_]usize{ 32, 64 }) |group| { // 2 bits: oQ2's experts
        const s = try Synth.init(device, shape[0], shape[1], 2, group, 0x2716 + shape[0] + group);
        defer s.deinit();
        const ref = try s.reference(true);
        defer ref.deinit();
        for ([_]bool{ false, true }) |f32_out| try affine(&t, device, queue, &s, ref, f32_out);
    };
    for ([_][2]usize{
        .{ 192, 1024 },
        .{ 256, 4096 },
    }) |shape| for ([_]u8{ 3, 5, 6 }) |bits| for ([_]usize{ 32, 64 }) |group| { // oQ4's 5 and 6
        const s = try Synth.init(
            device,
            shape[0],
            shape[1],
            bits,
            group,
            0x3716 + 100 * @as(usize, bits) + shape[0] + group,
        );
        defer s.deinit();
        const ref = try s.reference(true);
        defer ref.deinit();
        for ([_]bool{ false, true }) |f32_out| try affine(&t, device, queue, &s, ref, f32_out);
    };
    var p = try sq.Pipelines.load(device);
    defer p.deinit();
    for ([_][2]usize{ .{ 40, 512 }, .{ 256, 4096 }, .{ 6400, 256 } }) |shape| { // chunks of 32, 16 and 8
        seed += 1;
        const s = try Synth.init(device, shape[0], shape[1], 4, 64, seed);
        defer s.deinit();
        const ref = try s.reference(false);
        defer ref.deinit();
        try simd(&t, device, queue, &p, &s, ref);
    }
    if (t.failed > 0) {
        std.debug.print("FAIL {d} checks; the largest difference is {d:.4} of its float64 bound\n", .{ t.failed, t.worst });
        return error.CheckFailed;
    }
    std.debug.print("PASS every product within {d:.4} of its float64 bound, every row's bits equal at every window\n", .{t.worst});
}
