//! Bolt's Mamba scan under the contract: st = exp(A dt) st + x dt B, y = st . C, Y = bf16(silu(z)) * bf16(y + x D).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const bolt = @import("metal_bolt.zig");
const mamba = @import("metal_bolt_mamba.zig");

const c = contract.spec;
const g = contract.gates;
const nem = tf.nemotron;
const Mut = bolt.Mut;
const Range = bolt.Range;
const controls = mamba.controls;
const scanSpec = mamba.scanSpec;
const CD = mamba.CD;
const DH = mamba.DH;
const DS = mamba.DS;
const DTOFF = mamba.DTOFF;
const H = mamba.H;
const MW = mamba.MW;
const NG = mamba.NG;
const PROJ = mamba.PROJ;
const SSZ = mamba.SSZ;
const XD = mamba.XD;
const chain = mamba.chain;
const fast = mamba.fast;
const instantiated = mamba.instantiated;
const plus = mamba.plus;
const point = mamba.point;
const rid = mamba.rid;
const sameBits = mamba.sameBits;
const scanDepth = mamba.scanDepth;
const scan_inst = mamba.scan_inst;
const silu = mamba.silu;
const single = mamba.single;
const slot_in = mamba.slot_in;
const slot_out = mamba.slot_out;
const slots = mamba.slots;
const softplus = mamba.softplus;
const tableBytes = mamba.tableBytes;
const tree_source = mamba.tree_source;
const u = mamba.u;
const word = mamba.word;

pub const ScanProbe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    mut: ?Mut = null,
    rows: usize = 0,
    fill: g.Fill = .random,
    parity: u1 = 0,
    t: nem.layers.Tables = .{},
    p: []u16 = &.{},
    xbc: []u16 = &.{},
    s_in: []f32 = &.{},
    heads: []f32 = &.{}, // A_LOG, then D, then dt_bias: H each
    refs: [6]rig_.Ref = undefined,
    y: rig_.Ref = undefined,
    dtraw: rig_.Ref = undefined,
    s_out: rig_.Ref = undefined,
    tree: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, mut: ?Mut) !ScanProbe {
        try instantiated(scan_inst);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .mut = mut, .pipe = try bolt.build(a, device, tree_source, "tf_tree_scan_bolt", mut) };
    }

    fn free(t: *ScanProbe) void {
        for ([_][]u16{ t.p, t.xbc }) |b| t.a.free(b);
        for ([_][]f32{ t.s_in, t.heads }) |b| t.a.free(b);
        t.p = &.{};
        t.xbc = &.{};
        t.s_in = &.{};
        t.heads = &.{};
    }

    pub fn deinit(t: *ScanProbe) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }

    pub fn kernel(t: *ScanProbe) g.Kernel {
        return .{ .spec = bolt.under(scanSpec(), t.mut), .ctx = t, .vt = &vt, .control = if (t.mut) |m| m.gate else null };
    }

    const vt = g.hooks(ScanProbe);

    /// B's and C's partner states in gate 4's inputs: each of the first 64 pairs with one of the last, rotated.
    fn partner(s: usize) usize {
        return 64 + (s * 5 + 3) % 64;
    }

    pub fn prepare(t: *ScanProbe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != XD or shape.k != DS or shape.rows == 0 or shape.rows > 16) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.fill = fill;
        t.parity = @intCast(seed & 1);
        t.t = chain(t.rows, t.parity);
        t.p = try t.a.alloc(u16, t.rows * PROJ);
        t.xbc = try t.a.alloc(u16, t.rows * CD);
        t.s_in = try t.a.alloc(f32, slots * SSZ);
        t.heads = try t.a.alloc(f32, 3 * H);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x7363);
        const rng = prng.random();
        // z in [-3, 3), raw dt in [-2, 2); x, B, C in [-1, 1); A_LOG, D and dt_bias in [-1, 1) (so A dt >= -8.3)
        for (t.p) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (0..t.rows) |r| for (t.p[r * PROJ ..][0..XD]) |*v| {
            v.* = c.toBf16(rng.float(f32) * 6 - 3);
        };
        for (t.xbc) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (t.s_in) |*v| v.* = rng.float(f32) * 2 - 1;
        for (t.heads) |*v| v.* = rng.float(f32) * 2 - 1;
        switch (fill) {
            .random => {},
            .nonfinite => {
                // x NaN (row 0, channel 3), B +Inf (row 1, group 0), z -Inf (row 2, channel 7), C NaN (row 3, group 1)
                t.xbc[3] = c.toBf16(std.math.nan(f32));
                if (t.rows > 1) t.xbc[CD + XD + 5] = c.toBf16(std.math.inf(f32));
                if (t.rows > 2) t.p[2 * PROJ + 7] = c.toBf16(-std.math.inf(f32));
                if (t.rows > 3) t.xbc[3 * CD + XD + NG * DS + DS + 9] = c.toBf16(std.math.nan(f32));
            },
            .stress => {
                // A_LOG -inf, B 0, D 1, z 64; each state's term cancelled by its partner's: the dot is 0 in any order
                @memset(t.heads[0..H], -std.math.inf(f32));
                @memset(t.heads[H .. 2 * H], 1);
                for (0..t.rows) |r| {
                    @memset(t.p[r * PROJ ..][0..XD], c.toBf16(64));
                    const row = t.xbc[r * CD ..][0..CD];
                    for (row[0..XD]) |*v| v.* = c.toBf16(@floatFromInt(1 + rng.uintLessThan(u32, 7)));
                    @memset(row[XD .. XD + NG * DS], 0);
                    for (0..NG) |gi| for (0..64) |s| {
                        const cv: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
                        row[XD + NG * DS + gi * DS + s] = c.toBf16(cv);
                        row[XD + NG * DS + gi * DS + partner(s)] = c.toBf16(-cv);
                    };
                }
                for (0..XD) |cx| for (0..64) |s| {
                    const sv: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 2000));
                    t.s_in[slot_in * SSZ + cx * DS + s] = sv;
                    t.s_in[slot_in * SSZ + cx * DS + partner(s)] = sv;
                };
            },
        }
        t.refs[0] = try t.rig.input(std.mem.sliceAsBytes(t.p));
        t.refs[1] = try t.rig.input(std.mem.sliceAsBytes(t.xbc));
        t.refs[2] = try t.rig.input(std.mem.sliceAsBytes(t.s_in));
        t.refs[3] = try t.rig.input(std.mem.sliceAsBytes(t.heads[0..H]));
        t.refs[4] = try t.rig.input(std.mem.sliceAsBytes(t.heads[H .. 2 * H]));
        t.refs[5] = try t.rig.input(std.mem.sliceAsBytes(t.heads[2 * H ..]));
        t.y = try t.rig.output(t.rows * XD * 2);
        t.dtraw = try t.rig.output(@as(usize, @intCast(rid * 2 + 2)) * MW * H * 2);
        t.s_out = try t.rig.output(slots * SSZ * 4);
        t.tree = try t.rig.output(SSZ * 4);
        return true;
    }

    pub fn run(t: *ScanProbe) !void {
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        for (t.refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        // as layers.mamba binds it (its tables by value; one state buffer as 2 and 14 there, two buffers here)
        e.setValue([2]f32{ 0.0, std.math.inf(f32) }, 6);
        e.setValue([2]i32{ @intCast(t.t.virtual), MW }, 7);
        e.setBytes(tableBytes(&t.t.va, t.t.virtual), 8);
        e.setBytes(tableBytes(&t.t.vb, t.t.virtual), 9);
        e.setBytes(tableBytes(&t.t.vc, t.t.virtual), 10);
        e.setBytes(tableBytes(&t.t.segi, 8 * t.t.segs), 11);
        e.setBuffer(t.dtraw.buf, t.dtraw.off, 12);
        e.setBuffer(t.y.buf, t.y.off, 13);
        e.setBuffer(t.s_out.buf, t.s_out.off, 14);
        e.setBuffer(t.tree.buf, t.tree.off, 15);
        e.dispatchThreads(mtl.Size.of(32, DH, H), mtl.Size.of(32, 8, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *ScanProbe) usize {
        return t.rows * XD;
    }

    fn x(t: *const ScanProbe, v: usize, i: usize) f64 {
        return c.fromBf16(t.xbc[v * CD + i]);
    }

    /// B's and C's first index for channel cx's group.
    fn bAt(cx: usize) usize {
        return XD + cx / DH / (H / NG) * DS;
    }

    fn rawDt(t: *const ScanProbe, v: usize, h: usize) f64 {
        return c.fromBf16(t.p[v * PROJ + DTOFF + h]);
    }

    /// Channel cx's float64 recurrence: Y's rows; with an accumulator, the dot's sums as it rounds them in that order.
    fn exact(t: *const ScanProbe, cx: usize, sim: ?struct { acc: c.Accum, order: g.Order }, out: []f64) void {
        const h = cx / DH;
        const A = -@exp(@as(f64, t.heads[h]));
        const dskip: f64 = bolt.bf16of(t.heads[H + h]);
        var st: [DS]f64 = undefined;
        for (&st, 0..) |*v, s| v.* = t.s_in[slot_in * SSZ + cx * DS + s];
        const b0 = bAt(cx);
        var terms: [DS]f64 = undefined;
        for (0..t.rows) |v| {
            const dt = std.math.clamp(softplus(t.rawDt(v, h) + t.heads[2 * H + h]), 0, std.math.inf(f64));
            const dA = @exp(A * dt);
            const xv = t.x(v, cx);
            for (&st, &terms, 0..) |*sv, *term, s| {
                sv.* = dA * sv.* + xv * dt * t.x(v, b0 + s);
                term.* = sv.* * t.x(v, b0 + NG * DS + s);
            }
            var acc: f64 = 0;
            if (sim) |m| acc = contract.rows.sumIn(m.acc, m.order, &terms) else for (terms) |term| {
                acc += term;
            }
            out[v * XD + cx] = silu(c.fromBf16(t.p[v * PROJ + cx])) * (acc + xv * dskip);
        }
    }

    pub fn reference(t: *ScanProbe, want: []f64, mag: []f64) !void {
        for (0..XD) |cx| t.exact(cx, null, want);
        for (mag, want) |*m, w| m.* = @abs(w);
    }

    pub fn simulate(t: *ScanProbe, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..XD) |cx| t.exact(cx, .{ .acc = acc, .order = order }, out);
    }

    /// Channel cx's intervals: each f32 op widened by its rounding, the dot within gamma(depth), and the bf16 points.
    fn ranges(t: *const ScanProbe, cx: usize, low: []f64, high: []f64, state: ?[]Range) void {
        const h = cx / DH;
        const ea = @exp(@as(f64, t.heads[h]));
        const A = Range{ .lo = -ea * (1 + fast), .hi = -ea * (1 - fast) };
        const dskip: f64 = bolt.bf16of(t.heads[H + h]);
        var st: [DS]Range = undefined;
        for (&st, 0..) |*v, s| v.* = point(t.s_in[slot_in * SSZ + cx * DS + s]);
        const b0 = bAt(cx);
        for (0..t.rows) |v| {
            const xin = point(t.rawDt(v, h) + t.heads[2 * H + h]).widen(u);
            var dt = Range{ .lo = softplus(xin.lo), .hi = softplus(xin.hi) };
            const e = 2 * fast + 3 * u + u * @abs(dt.hi);
            dt = .{ .lo = @max(0, dt.lo - e), .hi = dt.hi + e };
            const adt = A.times(dt).widen(u);
            // A 0 (gate 4's A_LOG -inf): exp(0) is 1 exactly
            const dA = if (ea == 0) point(1) else (Range{ .lo = @exp(adt.lo), .hi = @exp(adt.hi) }).widen(fast);
            const xv = t.x(v, cx);
            const xdt = point(xv).times(dt).widen(u);
            // gate 4's integers (dA 1, B 0, products under 2^24) make every step exact in fp32
            const r: f64 = if (t.fill == .stress) 0 else u;
            var acc = point(0);
            var m: f64 = 0;
            for (&st, 0..) |*sv, s| {
                sv.* = plus(dA.times(sv.*).widen(r), xdt.scale(t.x(v, b0 + s)).widen(r)).widen(r);
                const term = sv.scale(t.x(v, b0 + NG * DS + s)).widen(r);
                acc = plus(acc, term);
                m += @max(@abs(term.lo), @abs(term.hi));
            }
            if (t.fill != .stress) {
                const e2 = c.gamma(@floatFromInt(scanDepth(.{ .rows = 0, .n = 0, .k = 0 })), u) * m;
                acc = .{ .lo = acc.lo - e2, .hi = acc.hi + e2 };
            }
            const y = plus(acc, point(xv * dskip).widen(u)).widen(u).bf16();
            const sz = point(silu(c.fromBf16(t.p[v * PROJ + cx]))).widen(2 * fast + 3 * u).bf16();
            const out = sz.times(y).widen(u).bf16();
            low[v * XD + cx] = out.lo;
            high[v * XD + cx] = out.hi;
        }
        if (state) |keep| @memcpy(keep, &st);
    }

    pub fn intervals(t: *ScanProbe, low: []f64, high: []f64) !void {
        for (0..XD) |cx| t.ranges(cx, low, high, null);
    }

    pub fn bounds(t: *ScanProbe, want: []const f64, _: []const f64, allow: []f64) !void {
        const buf = try t.a.alloc(f64, t.count() * 2);
        defer t.a.free(buf);
        const low = buf[0..t.count()];
        const high = buf[t.count()..];
        try t.intervals(low, high);
        for (allow, want, low, high) |*v, w, lo, hi| v.* = @max(@abs(w - lo), @abs(hi - w));
    }

    /// Y against float64; each row's raw dt saved bit for bit; the last row's state within its interval; nothing more.
    pub fn check(t: *ScanProbe, at: g.Gate) !?[]const u8 {
        const n = t.count();
        const out = t.rig.bytes(t.y, n * 2);
        const sout = t.rig.bytes(t.s_out, slots * SSZ * 4);
        {
            const buf = try t.a.alloc(f64, 2 * n);
            defer t.a.free(buf);
            const lo = buf[0..n];
            const hi = buf[n..];
            if (at == .nonfinite) {
                try t.reference(lo, hi);
                for (lo, 0..) |w, i| {
                    const got = c.read(.bf16, out, i);
                    if (c.classOf(got) != c.classOf(w)) return try std.fmt.bufPrint(&t.msg, "Y[{d}] = {d} is not of float64's class ({d})", .{ i, got, w });
                }
            } else {
                var st: [DS]Range = undefined;
                for (0..XD) |cx| {
                    t.ranges(cx, lo, hi, &st);
                    for (st, 0..) |r, s| {
                        const got = single(sout, @as(usize, @intCast(slot_out)) * SSZ + cx * DS + s);
                        if (!(got >= r.lo and got <= r.hi)) return try std.fmt.bufPrint(&t.msg, "state channel {d} index {d} = {d} outside float64's [{d}, {d}]", .{ cx, s, got, r.lo, r.hi });
                    }
                }
                for (lo, hi, 0..) |l, h, i| {
                    const got = c.read(.bf16, out, i);
                    if (!(got >= l and got <= h)) return try std.fmt.bufPrint(&t.msg, "Y[{d}] = {d} outside float64's [{d}, {d}] through its bf16 points", .{ i, got, l, h });
                }
            }
        }
        const dtraw = t.rig.bytes(t.dtraw, @as(usize, @intCast(rid * 2 + 2)) * MW * H * 2);
        const first: usize = @intCast((rid * 2 + 1 - @as(i32, t.parity)) * MW);
        for (0..@as(usize, @intCast(rid * 2 + 2)) * MW) |r| for (0..H) |h| {
            const want: u16 = if (r >= first and r < first + t.rows) t.p[(r - first) * PROJ + DTOFF + h] else 0xffff;
            if (!sameBits(word(dtraw, r * H + h), want)) return try std.fmt.bufPrint(&t.msg, "DTRAW row {d} head {d} = {x}, want {x}", .{ r, h, word(dtraw, r * H + h), want });
        };
        for (0..slots) |s| if (s != slot_out) for (0..SSZ) |i| {
            if (@as(u32, @bitCast(single(sout, s * SSZ + i))) != 0xffffffff) return try std.fmt.bufPrint(&t.msg, "state slot {d} index {d} written", .{ s, i });
        };
        const tree = t.rig.bytes(t.tree, SSZ * 4);
        for (0..SSZ) |i| if (@as(u32, @bitCast(single(tree, i))) != 0xffffffff) return try std.fmt.bufPrint(&t.msg, "tree state {d} written (a chain keeps none)", .{i});
        return null;
    }
};
