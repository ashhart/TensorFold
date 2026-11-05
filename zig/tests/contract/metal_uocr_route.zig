//! Unlimited-OCR-2's route kernels (decoder.metal): DASH block keys and picks with entmax bias, MoE gate and combine.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const uocr = @import("metal_uocr.zig");
const base = @import("metal_uocr_route_base.zig");

const c = contract.spec;
const g = contract.gates;
const block = base.block;
const local_blocks = base.local_blocks;
const ranked = base.ranked;
const reads = base.reads;
const max_keys = base.max_keys;
const max_blocks = base.max_blocks;
const chunk = base.chunk;
const scale = base.scale;
const idx_scale = base.idx_scale;
const floor_ = base.floor_;
const clamp_at = base.clamp_at;
const neg = base.neg;
const UdBlocks = base.UdBlocks;
const UdRoute = base.UdRoute;
const u = base.u;
const tiny = base.tiny;
const gamma = base.gamma;
const Range = base.Range;
const spread = base.spread;
const clampScore = base.clampScore;
const wanted = base.wanted;
const Edit = base.Edit;
const Control = base.Control;
const pipeline = base.pipeline;
const intIn = base.intIn;
const BlockKeys = base.BlockKeys;
const Route = @import("metal_uocr_route_pick.zig").Route;

// --- ud_moe_gate: softmax over 64 logits, the top 6 as ids and weights --------------------------------------------

const gate_rule = "fp32 logits, softmax with IEEE rules (its max ignores a NaN): a NaN or +Inf logit makes every score NaN (the sum takes it) and a -Inf one scores 0; six rounds of strict > from -Inf over the scores not yet taken, so ties go to the lower expert and a NaN is never taken; a round with nothing above -Inf takes expert 0 with weight -Inf; picks exact (the test's logits are exact in fp32 and distinct ones stay distinct), each weight within float64's range through the fp32 steps";

pub const MoeGate = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    label: []const u8,
    rows: usize = 0,
    x: []u16 = &.{},
    w: []u16 = &.{},
    ins: [2]rig_.Ref = undefined,
    wts: rig_.Ref = undefined,
    ids: rig_.Ref = undefined,
    msg: [240]u8 = undefined,

    const E = uocr.experts;
    const H = uocr.hidden;
    const K = uocr.topk;

    /// rows: tokens; n: the 6 picks; k: the hidden width.
    pub fn spec(label: []const u8) c.Spec {
        return .{
            .name = label,
            .accum = .f32,
            .depth = gateDepth,
            .out = .f32,
            .nan = .{ .stated = gate_rule },
            .served = &.{ .{ .rows = 33, .n = K, .k = H }, .{ .rows = 1, .n = K, .k = H }, .{ .rows = chunk, .n = K, .k = H }, .{ .rows = 7, .n = K, .k = H } },
        };
    }

    fn gateDepth(_: c.Shape) usize {
        return H;
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, label: []const u8, control: ?Control) !MoeGate {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(a, device, "ud_moe_gate", control), .label = label };
    }
    pub fn deinit(t: *MoeGate) void {
        t.a.free(t.x);
        t.a.free(t.w);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *MoeGate) g.Kernel {
        return .{ .spec = spec(t.label), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(MoeGate);

    pub fn prepare(t: *MoeGate, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != K or shape.k != H or shape.rows == 0) return false;
        t.rig.reset();
        t.a.free(t.x);
        t.x = &.{};
        t.a.free(t.w);
        t.w = &.{};
        t.rows = shape.rows;
        t.x = try t.a.alloc(u16, t.rows * H);
        t.w = try t.a.alloc(u16, E * H);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            // integers, dims 640.. mirror 0..639, dims 0..3 add n in [-1, 1]: each logit exact in [-4, 4]
            for (0..t.rows) |r| {
                const xr = t.x[r * H ..][0..H];
                for (0..H / 2) |d| {
                    const v: f32 = if (d < 4) 1 else intIn(rng, 1, 64);
                    xr[d] = c.toBf16(v);
                    xr[d + H / 2] = xr[d];
                }
            }
            for (0..E) |e| {
                const wr = t.w[e * H ..][0..H];
                for (0..H / 2) |d| {
                    const v = intIn(rng, 1, 64);
                    wr[d] = c.toBf16(v);
                    wr[d + H / 2] = c.toBf16(-v + (if (d < 4) intIn(rng, -1, 1) else 0));
                }
            }
        } else {
            // dyadic values: products are multiples of 2^-9 and partials stay below 2^7, so every logit is exact
            for (t.x) |*v| v.* = c.toBf16(intIn(rng, -8, 8) / 8);
            for (t.w) |*v| v.* = c.toBf16(intIn(rng, -6, 6) / 64);
            // experts 5 and 37 share one row and a large weight on a dim every token holds at 1: tied, both picked
            for (0..t.rows) |r| t.x[r * H + 1] = c.toBf16(1);
            t.w[5 * H + 1] = c.toBf16(4);
            @memcpy(t.w[37 * H ..][0..H], t.w[5 * H ..][0..H]);
        }
        if (fill == .nonfinite) {
            // rows 0-3: NaN, +Inf, -Inf, both at a dim (all scores NaN); expert 9 gets a -Inf logit, score 0
            for (0..t.rows) |r| t.x[r * H + 3] = c.toBf16(0.5);
            t.w[9 * H + 3] = c.toBf16(uocr.planted[2]);
            const put = [_][3]usize{ .{ 0, 100, 0 }, .{ 1, 200, 1 }, .{ 2, 300, 2 }, .{ 3, 400, 1 }, .{ 3, 401, 2 } };
            for (put) |p| if (p[0] < t.rows) {
                t.x[p[0] * H + p[1]] = c.toBf16(uocr.planted[p[2]]);
            };
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.wts = try t.rig.output(t.rows * K * 4);
        t.ids = try t.rig.output(t.rows * K * 4);
        return true;
    }

    pub fn run(t: *MoeGate) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        p.e.setBuffer(t.ids.buf, t.ids.off, 2);
        p.e.setBuffer(t.wts.buf, t.wts.off, 3);
        p.e.dispatchGroups(mtl.Size.of(t.rows, 1, 1), mtl.Size.of(E, 1, 1));
        try t.rig.finish(p);
    }

    /// The weights; the picks are checked by `check`.
    pub fn count(t: *MoeGate) usize {
        return t.rows * K;
    }

    const Dot = union(enum) { exact, sim: struct { acc: c.Accum, order: g.Order } };

    fn logits(t: *MoeGate, r: usize, how: Dot, out: *[E]f64) void {
        const xr = t.x[r * H ..][0..H];
        for (out, 0..) |*o, e| {
            const wr = t.w[e * H ..][0..H];
            switch (how) {
                .exact => {
                    var s: f64 = 0;
                    for (xr, wr) |a, b| s += @as(f64, c.fromBf16(a)) * c.fromBf16(b);
                    o.* = s;
                },
                .sim => |m| {
                    var terms: [H]f64 = undefined;
                    for (&terms, xr, wr) |*v, a, b| v.* = @as(f64, c.fromBf16(a)) * c.fromBf16(b);
                    o.* = contract.rows.sumIn(m.acc, m.order, &terms);
                },
            }
        }
    }

    /// Row r's picks and weights: points, or when `w` each weight's range through the kernel's fp32 steps.
    fn row(t: *MoeGate, r: usize, how: Dot, w: bool, ids: *[K]u32, out: *[K]Range) void {
        var l: [E]f64 = undefined;
        t.logits(r, how, &l);
        var m: f64 = -std.math.inf(f64);
        for (l) |v| if (v > m) {
            m = v;
        };
        var ex: [E]Range = undefined;
        var p: [E]f64 = undefined;
        var total = Range.point(0);
        var sum: f64 = 0;
        for (&ex, &p, l) |*e, *pv, v| {
            const a = Range.point(v - m).r(w, 1);
            e.* = (Range{ .lo = @exp(a.lo), .hi = @exp(a.hi) }).r(w, 8);
            total = total.add(e.*);
            pv.* = @exp(v - m);
            sum += pv.*;
        }
        // simd_sum's 5 levels, then the two simdgroups' sum through a second simd_sum
        total = total.r(w, 2 * gamma(10) / u);
        for (&p) |*pv| pv.* /= sum;
        // the picks on float64's scores; the weights' ranges beside them
        var taken: [E]bool = @splat(false);
        for (0..K) |k| {
            var best: usize = 0;
            var bv: f64 = -std.math.inf(f64);
            for (p, 0..) |s, i| if (!taken[i] and s > bv) {
                bv = s;
                best = i;
            };
            taken[best] = true;
            ids[k] = @intCast(best);
            out[k] = if (std.math.isInf(bv)) Range.point(bv) else ex[best].div(total).r(w, 6);
        }
    }

    pub fn reference(t: *MoeGate, want: []f64, mag: []f64) !void {
        for (0..t.rows) |r| {
            var ids: [K]u32 = undefined;
            var wv: [K]Range = undefined;
            t.row(r, .exact, false, &ids, &wv);
            for (wv, 0..) |x, k| want[r * K + k] = x.lo;
        }
        for (mag, want) |*m, x| m.* = @abs(x);
    }

    pub fn bounds(t: *MoeGate, want: []const f64, _: []const f64, allow: []f64) !void {
        for (0..t.rows) |r| {
            var ids: [K]u32 = undefined;
            var wv: [K]Range = undefined;
            t.row(r, .exact, true, &ids, &wv);
            for (wv, 0..) |x, k| allow[r * K + k] = spread(want[r * K + k], x);
        }
    }

    pub fn simulate(t: *MoeGate, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..t.rows) |r| {
            var ids: [K]u32 = undefined;
            var wv: [K]Range = undefined;
            t.row(r, .{ .sim = .{ .acc = acc, .order = order } }, false, &ids, &wv);
            for (wv, 0..) |x, k| out[r * K + k] = x.lo;
        }
    }

    pub fn check(t: *MoeGate, at: g.Gate) !?[]const u8 {
        const n = t.count();
        const ids = std.mem.bytesAsSlice(u32, t.rig.bytes(t.ids, n * 4));
        const wts = std.mem.bytesAsSlice(f32, t.rig.bytes(t.wts, n * 4));
        var degenerate: usize = 0;
        for (0..t.rows) |r| {
            var want: [K]u32 = undefined;
            var wv: [K]Range = undefined;
            t.row(r, .exact, true, &want, &wv);
            for (0..K) |k| {
                const i = r * K + k;
                if (ids[i] != want[k]) return try std.fmt.bufPrint(&t.msg, "row {d} pick {d} is expert {d} where the host picks {d}", .{ r, k, ids[i], want[k] });
                const v: f64 = wts[i];
                if (std.math.isInf(wv[k].lo)) {
                    degenerate += 1;
                    if (v != wv[k].lo) return try std.fmt.bufPrint(&t.msg, "row {d} pick {d} has weight {e} where the rule gives -Inf", .{ r, k, v });
                } else if (!wv[k].holds(v)) return try std.fmt.bufPrint(&t.msg, "row {d} pick {d} weight {e:.8} outside [{e:.8}, {e:.8}]", .{ r, k, v, wv[k].lo, wv[k].hi });
            }
        }
        if (at == .nonfinite and degenerate == 0) return try std.fmt.bufPrint(&t.msg, "no row with non-finite scores planted", .{});
        return null;
    }
};

// --- ud_moe_combine: y = bf16(bf16(sum_k w_k slot_k in fp32) + shared) ---------------------------------------------

pub const Combine = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    rows: usize = 0,
    exact: bool = false,
    slots: []u16 = &.{},
    wts: []f32 = &.{},
    shared: []u16 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    const H = uocr.hidden;
    const K = uocr.topk;
    const name = "metal uocr route ud_moe_combine top 6 hidden 1280";

    /// rows: tokens; n: the hidden width; k: the 6 slots each output sums.
    pub fn spec() c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = combineDepth,
            .mids = 1,
            .out = .bf16,
            .served = &.{ .{ .rows = 1, .n = H, .k = K }, .{ .rows = chunk, .n = H, .k = K }, .{ .rows = 5, .n = H, .k = K } },
        };
    }

    fn combineDepth(_: c.Shape) usize {
        return K;
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Combine {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try uocr.pipeline(device, .decoder, "ud_moe_combine") };
    }
    pub fn deinit(t: *Combine) void {
        t.a.free(t.slots);
        t.a.free(t.wts);
        t.a.free(t.shared);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Combine) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Combine);

    pub fn prepare(t: *Combine, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != H or shape.k != K or shape.rows == 0) return false;
        t.rig.reset();
        t.a.free(t.slots);
        t.slots = &.{};
        t.a.free(t.wts);
        t.wts = &.{};
        t.a.free(t.shared);
        t.shared = &.{};
        t.rows = shape.rows;
        t.exact = fill == .stress;
        t.slots = try t.a.alloc(u16, t.rows * K * H);
        t.wts = try t.a.alloc(f32, t.rows * K);
        t.shared = try t.a.alloc(u16, t.rows * H);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            // slots 3..5 repeat 0..2 nudged by n in [-1, 1], weights 3..5 negate 0..2: each sum exact
            for (0..t.rows) |r| {
                for (0..K / 2) |k| {
                    const v = intIn(rng, 1, 4096);
                    t.wts[r * K + k] = v;
                    t.wts[r * K + k + K / 2] = -v;
                }
                for (0..K / 2) |k| for (0..H) |d| {
                    const s = intIn(rng, 1, 255);
                    t.slots[(r * K + k) * H + d] = c.toBf16(s);
                    t.slots[(r * K + k + K / 2) * H + d] = c.toBf16(s + intIn(rng, -1, 1));
                };
                for (t.shared[r * H ..][0..H]) |*v| v.* = c.toBf16(intIn(rng, -8, 8));
            }
        } else {
            for (t.slots) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
            for (t.wts) |*v| v.* = rng.float(f32) * 0.3;
            for (t.shared) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        }
        if (fill == .nonfinite) {
            // NaN, +Inf, -Inf slots in rows 0-2, both infinities in row 3, row 4 a -Inf weight, a +Inf shared
            const put = [_][4]usize{ .{ 0, 0, 10, 0 }, .{ 1, 2, 20, 1 }, .{ 2, 5, 30, 2 }, .{ 3, 1, 40, 1 }, .{ 3, 4, 40, 2 } };
            for (put) |p| if (p[0] < t.rows) {
                t.slots[(p[0] * K + p[1]) * H + p[2]] = c.toBf16(uocr.planted[p[3]]);
            };
            if (t.rows > 4) {
                t.wts[4 * K] = uocr.planted[2];
                t.shared[4 * H + 50] = c.toBf16(uocr.planted[1]);
            }
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.slots));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.wts));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.shared));
        t.out = try t.rig.output(t.rows * H * 2);
        return true;
    }

    pub fn run(t: *Combine) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |ref, i| p.e.setBuffer(ref.buf, ref.off, i);
        p.e.setBuffer(t.out.buf, t.out.off, 3);
        p.e.dispatchThreads(mtl.Size.of(H, t.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Combine) usize {
        return t.rows * H;
    }

    fn terms(t: *Combine, i: usize, out: *[K]f64) void {
        const r = i / H;
        const d = i % H;
        for (out, 0..) |*v, k| v.* = @as(f64, c.fromBf16(t.slots[(r * K + k) * H + d])) * t.wts[r * K + k];
    }

    /// bf16(fp32(bf16(s) + shared)) for a sum s.
    fn finish(t: *Combine, i: usize, s: f64) f64 {
        const sum: f32 = @floatCast(uocr.bf(s));
        return uocr.bf(sum + c.fromBf16(t.shared[i]));
    }

    pub fn reference(t: *Combine, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            var x: [K]f64 = undefined;
            t.terms(i, &x);
            var s: f64 = 0;
            var a: f64 = 0;
            for (x) |v| {
                s += v;
                a += @abs(v);
            }
            w.* = t.finish(i, s);
            m.* = if (std.math.isFinite(a)) a else 0;
        }
    }

    /// The sum's chain of 6 (none on the exact stress inputs), its bf16 point, the fp32 add and the output's point.
    pub fn intervals(t: *Combine, low: []f64, high: []f64) !void {
        for (low, high, 0..) |*lo, *hi, i| {
            var x: [K]f64 = undefined;
            t.terms(i, &x);
            var s: f64 = 0;
            var a: f64 = 0;
            for (x) |v| {
                s += v;
                a += @abs(v);
            }
            const e = if (t.exact) 0 else gamma(K) * a + K * tiny;
            const sh = Range.point(c.fromBf16(t.shared[i]));
            const y = (Range{ .lo = s - e, .hi = s + e }).bf16().add(sh).r(true, 1).bf16();
            lo.* = y.lo;
            hi.* = y.hi;
        }
    }

    pub fn bounds(t: *Combine, want: []const f64, _: []const f64, allow: []f64) !void {
        const n = t.count();
        const buf = try t.a.alloc(f64, 2 * n);
        defer t.a.free(buf);
        try t.intervals(buf[0..n], buf[n..]);
        for (allow, want, buf[0..n], buf[n..]) |*l, w, lo, hi| l.* = spread(w, .{ .lo = lo, .hi = hi });
    }

    pub fn simulate(t: *Combine, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (out, 0..) |*o, i| {
            var x: [K]f64 = undefined;
            t.terms(i, &x);
            o.* = t.finish(i, contract.rows.sumIn(acc, order, &x));
        }
    }
};

// --- the group ---------------------------------------------------------------------------------------------------

const keys_name = "metal uocr route ud_block_keys block 32 router 128";
const route_name = "metal uocr route ud_route ranked 16 read 12 blocks to 4096";
const gate_name = "metal uocr route ud_moe_gate experts 64 top 6 hidden 1280";

/// Planted controls on the selection logic, each a decoder.metal edit the shapes gate must catch.
const route_controls = [_]Control{
    .{ .label = "control: metal uocr route ud_route, a tie goes to the higher block", .edits = &.{
        .{ .from = "(x[c] > best || (x[c] == best && int(c) < at))", .to = "(x[c] >= best)" },
        .{ .from = "simd_min(best == top ? at : 0x7fffffff)", .to = "simd_max(best == top ? at : -1)" },
    } },
    .{ .label = "control: metal uocr route ud_route, the local gap one block short", .edits = &.{
        .{ .from = "int(qpos[q] / 32) - 4", .to = "int(qpos[q] / 32) - 3" },
    } },
    .{ .label = "control: metal uocr route ud_route, a block's last page left out", .edits = &.{
        .{ .from = "qp <= bmax[c])", .to = "qp < bmax[c])" },
    } },
};
const gate_controls = [_]Control{
    .{ .label = "control: metal uocr route ud_moe_gate, a tie goes to the higher expert", .edits = &.{
        .{ .from = "if (!taken[i] && score[i] > bv)", .to = "if (!taken[i] && score[i] >= bv)" },
    } },
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    keys: std.ArrayList(*BlockKeys) = .empty,
    routes: std.ArrayList(*Route) = .empty,
    gates: std.ArrayList(*MoeGate) = .empty,
    combine: ?*Combine = null,

    fn add(s: *Suite, comptime T: type, list: *std.ArrayList(*T), label: []const u8, control: ?Control) !void {
        const x = try s.a.create(T);
        errdefer s.a.destroy(x);
        x.* = try T.init(s.a, s.device, s.queue, label, control);
        try list.append(s.a, x);
        var k = x.kernel();
        if (control != null) k.control = .shapes;
        try s.kernels.append(s.a, k);
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (wanted(only, keys_name)) try s.add(BlockKeys, &s.keys, keys_name, null);
        if (wanted(only, route_name)) try s.add(Route, &s.routes, route_name, null);
        for (&route_controls) |ctl| if (wanted(only, ctl.label)) try s.add(Route, &s.routes, ctl.label, ctl);
        if (wanted(only, gate_name)) try s.add(MoeGate, &s.gates, gate_name, null);
        for (&gate_controls) |ctl| if (wanted(only, ctl.label)) try s.add(MoeGate, &s.gates, ctl.label, ctl);
        if (wanted(only, Combine.name)) {
            const x = try s.a.create(Combine);
            errdefer s.a.destroy(x);
            x.* = try Combine.init(s.a, s.device, s.queue);
            s.combine = x;
            try s.kernels.append(s.a, x.kernel());
        }
    }

    fn drop(s: *Suite, comptime T: type, list: *std.ArrayList(*T)) void {
        for (list.items) |x| {
            x.deinit();
            s.a.destroy(x);
        }
        list.deinit(s.a);
    }

    pub fn deinit(s: *Suite) void {
        s.drop(BlockKeys, &s.keys);
        s.drop(Route, &s.routes);
        s.drop(MoeGate, &s.gates);
        if (s.combine) |x| {
            x.deinit();
            s.a.destroy(x);
        }
    }
};

/// Gate 5's texts for `uocr route ...`: decoder.metal as kernels.zig compiles it (no nax.h, so one text for all).
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, name: []const u8) !?uocr.Texts {
    if (!std.mem.startsWith(u8, name, "uocr route ")) return null;
    const text = uocr.source(.decoder);
    return .{ .here = text, .pre_m5 = text };
}
