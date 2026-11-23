//! DeepSeek V4.1 opt-in FP64-certified BF16 routed experts under the contract: decode, M5 tensor units, equality.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_experts_ref.zig");

const c = contract.spec;
const g = contract.gates;
const dec = tf.dsv41_metal.experts;
const xc = tf.dsv41_metal.xmma_cert;
const topk = ref.topk;
const backing_experts = 12;
const base = "dsv41 experts_cert ";

pub const Which = enum { up, down };
pub const Path = enum { decode, units };
pub const Variant = struct { which: Which, path: Path, wide: bool = false, ctl: u8 = 0 };

const served_up_dec = [_]c.Shape{
    .{ .rows = 1, .n = 96, .k = 5120 },
    .{ .rows = 4, .n = 96, .k = 5120 },
    .{ .rows = 16, .n = 96, .k = 5120 },
    .{ .rows = 1, .n = 2304, .k = 5120 },
    .{ .rows = 4, .n = 2304, .k = 5120 },
};

const served_up = [_]c.Shape{
    .{ .rows = 1, .n = 96, .k = 5120 },
    .{ .rows = 4, .n = 96, .k = 5120 },
    .{ .rows = 16, .n = 96, .k = 5120 },
    .{ .rows = 1, .n = 2304, .k = 5120 },
    .{ .rows = 4, .n = 2304, .k = 5120 },
    .{ .rows = 33, .n = 96, .k = 5120 },
};

const served_down_dec = [_]c.Shape{
    .{ .rows = 1, .n = 160, .k = 2304 },
    .{ .rows = 4, .n = 160, .k = 2304 },
    .{ .rows = 16, .n = 160, .k = 2304 },
    .{ .rows = 1, .n = 5120, .k = 2304 },
    .{ .rows = 4, .n = 5120, .k = 2304 },
};

const served_down = [_]c.Shape{
    .{ .rows = 1, .n = 160, .k = 2304 },
    .{ .rows = 4, .n = 160, .k = 2304 },
    .{ .rows = 16, .n = 160, .k = 2304 },
    .{ .rows = 1, .n = 5120, .k = 2304 },
    .{ .rows = 4, .n = 5120, .k = 2304 },
    .{ .rows = 33, .n = 160, .k = 2304 },
};

fn depthOf(s: c.Shape) usize {
    return ref.depth(s.k);
}

/// Direct round-to-nearest-even from f64 to BF16 without double rounding through f32.
fn f64ToBf16(x: f64) u16 {
    if (std.math.isNan(x)) return 0x7fc0;
    const f: f32 = @floatCast(x);
    if (!std.math.isFinite(f)) return c.toBf16(f);
    const u: u32 = @bitCast(f);
    if ((u & 0xffff) == 0x8000) {
        const diff = x - @as(f64, f);
        if (diff != 0) {
            const same_sign = (diff < 0) == ((u >> 31) != 0);
            const trunc: u16 = @intCast(u >> 16);
            return if (same_sign) trunc +% 1 else trunc;
        }
    }
    const lsb: u32 = (u >> 16) & 1;
    return @intCast((u +% 0x7fff +% lsb) >> 16);
}

fn bf64(x: f64) f32 {
    return c.fromBf16(f64ToBf16(x));
}

pub fn spec(v: Variant, name: []const u8) c.Spec {
    const served: []const c.Shape = if (v.path == .decode)
        (if (v.which == .up) &served_up_dec else &served_down_dec)
    else
        (if (v.which == .up) &served_up else &served_down);
    const max_r: usize = if (v.path == .decode) 16 else xc.row_tile;
    return .{
        .name = name,
        .accum = .f32,
        .depth = depthOf,
        .out = .bf16,
        .nan = if (v.which == .up) .{ .stated = "a non-finite value in a row's activation makes all of the row's outputs NaN" } else .propagate,
        .tiles = .{ .{ .size = max_r }, .{ .size = 32, .step = 32, .min = 32 }, .{ .size = 256, .step = 32, .min = 32 } },
        .served = served,
    };
}

const Refs = struct { ids: rig_.Ref, weights: rig_.Ref, slots: rig_.Ref, nslots: rig_.Ref, pairs: rig_.Ref, where: rig_.Ref };

pub const Proj = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    dp: ?dec.Pipelines = null,
    xp: ?xc.Pipelines = null,
    shape: c.Shape = undefined,
    plan: ?ref.Plan = null,
    m1: ?ref.Mats = null,
    m3: ?ref.Mats = null,
    x: []u16 = &.{},
    spread: []f64 = &.{},
    backing: [4]mtl.Buffer = undefined,
    nback: usize = 0,
    xr: rig_.Ref = undefined,
    table: rig_.Ref = undefined,
    rt: Refs = undefined,
    out: rig_.Ref = undefined,
    tiles: rig_.Ref = undefined,
    codes: rig_.Ref = undefined,
    scales: rig_.Ref = undefined,
    dq: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Proj {
        var p: Proj = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name };
        const head = if (v.ctl != 0) try std.fmt.allocPrint(a, "#define TF_CTL {d}\n", .{v.ctl}) else "";
        defer if (head.len > 0) a.free(head);
        if (v.path == .decode) {
            const text = try xc.decodeSource(a, head);
            defer a.free(text);
            p.dp = try dec.Pipelines.loadSourceCert(device, text);
        } else {
            const text = try xc.source(a, head);
            defer a.free(text);
            p.xp = try xc.Pipelines.loadSource(a, device, text);
        }
        return p;
    }

    pub fn deinit(p: *Proj) void {
        p.drop();
        if (p.dp) |*q| q.deinit();
        if (p.xp) |*q| q.deinit();
        p.rig.deinit();
    }

    fn drop(p: *Proj) void {
        if (p.plan) |q| q.deinit();
        if (p.m1) |m| m.deinit();
        if (p.m3) |m| m.deinit();
        if (p.x.len > 0) p.a.free(p.x);
        if (p.spread.len > 0) p.a.free(p.spread);
        p.plan = null;
        p.m1 = null;
        p.m3 = null;
        p.x = &.{};
        p.spread = &.{};
        p.rig.reset();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        const control: ?g.Gate = switch (p.v.ctl) {
            1 => .nonfinite,
            2, 5, 6 => .shapes,
            3 => .sums,
            4 => .repeat,
            else => null,
        };
        return .{ .spec = spec(p.v, p.name), .ctx = p, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Proj);

    fn pairs(p: *const Proj) usize {
        return p.shape.rows * topk;
    }

    fn xcall(p: *const Proj) xc.Call {
        return if (p.v.which == .up) .{ .rows = @intCast(p.shape.rows), .k = @intCast(p.shape.k), .n = @intCast(p.shape.n) } else .{ .rows = @intCast(p.shape.rows), .k = @intCast(p.shape.n), .n = @intCast(p.shape.k) };
    }

    fn dcall(p: *const Proj) dec.Call {
        return if (p.v.which == .up) .{ .rows = @intCast(p.shape.rows), .k = @intCast(p.shape.k), .n = @intCast(p.shape.n) } else .{ .rows = @intCast(p.shape.rows), .k = @intCast(p.shape.n), .n = @intCast(p.shape.k) };
    }

    fn address(r: rig_.Ref) u64 {
        return r.buf.gpuAddress() + r.off;
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        const up = p.v.which == .up;
        const max_r: usize = if (p.v.path == .decode) 16 else 64;
        if (shape.rows == 0 or shape.rows > max_r or shape.k % 32 != 0 or shape.n % 32 != 0 or shape.k == 0 or shape.n == 0 or shape.k > (if (up) @as(usize, 5120) else 3072)) return false;
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.plan = try ref.Plan.init(p.a, shape.rows, backing_experts, rng);
        const np = p.pairs();
        p.x = try p.a.alloc(u16, (if (up) shape.rows else np) * shape.k);
        p.m1 = try ref.Mats.init(p.a, backing_experts, shape.n, shape.k);
        if (up) p.m3 = try ref.Mats.init(p.a, backing_experts, shape.n, shape.k);
        if (fill == .stress) {
            for ([_]?ref.Mats{ p.m1, p.m3 }) |m| if (m) |mm| mm.stress(rng);
            ref.stressX(p.x, if (up) shape.rows else np, shape.k, up, rng);
        } else {
            for ([_]?ref.Mats{ p.m1, p.m3 }) |m| if (m) |mm| mm.fill(rng, p.v.wide);
            ref.fillX(p.x, rng);
            if (p.v.wide) p.wideX(rng);
            if (fill == .nonfinite) ref.plantNonfinite(p.x, if (up) shape.rows else np, shape.k);
            if (!up and !p.v.wide and fill == .random) p.plantUnsafe();
        }
        try p.upload();
        return true;
    }

    /// Pair 0, column 0: unsafe 32-element group whose sum is exact in FP64/dd but drops bits on raw tensor units.
    fn plantUnsafe(p: *Proj) void {
        const k = p.shape.k;
        const m = p.m1.?;
        const w = (p.plan.?.expertAt(0) % m.e) * m.n;
        @memset(p.x[0..k], 0);
        for ([_]f32{ 0x1p24, 0x1p23, 0x1p16, 2, 2, 2, 2 }, 0..) |v, i| p.x[i] = c.toBf16(v);
        @memset(m.codes[w * k / 2 ..][0..16], 0);
        @memcpy(m.codes[w * k / 2 ..][0..4], &[_]u8{ 0x22, 0x12, 0x11, 0x01 });
        m.scales[w * k / 32] = 127;
    }

    fn wideX(p: *Proj, rng: std.Random) void {
        const m = p.m1.?;
        for (0..p.x.len / m.k) |r| for (0..m.k / 32) |grp| {
            const e = m.srow(0, 0)[grp];
            const f: f32 = ref.pow2(if (e >= 128) -40 else 0);
            for (0..32) |j| p.x[r * m.k + 32 * grp + j] = c.toBf16((1 + rng.float(f32)) * f * if (rng.boolean()) @as(f32, 1) else -1);
        };
        for (1..m.n * m.e) |row| {
            const first = m.scales[0 .. m.k / 32];
            @memcpy(m.scales[row * m.k / 32 ..][0 .. m.k / 32], first);
        }
    }

    fn upload(p: *Proj) !void {
        const up = p.v.which == .up;
        const pl = p.plan.?;
        const m1 = p.m1.?;
        const w1p = try p.rig.input(m1.codes);
        const w1s = try p.rig.input(m1.scales);
        var refs = [4]rig_.Ref{ w1p, w1s, w1p, w1s };
        if (up) {
            refs[2] = try p.rig.input(p.m3.?.codes);
            refs[3] = try p.rig.input(p.m3.?.scales);
        }
        for (refs, 0..) |r, i| p.backing[i] = r.buf;
        p.nback = 4;
        var table: [backing_experts]xc.ExpertPtrs = undefined;
        const per = m1.n * m1.k / 2;
        for (&table, 0..) |*t, e| t.* = .{
            .w1p = address(refs[0]) + e * per,
            .w1s = address(refs[1]) + e * per / 16,
            .w3p = address(refs[2]) + e * per,
            .w3s = address(refs[3]) + e * per / 16,
            .w2p = address(refs[0]) + e * per,
            .w2s = address(refs[1]) + e * per / 16,
        };
        p.table = try p.rig.input(std.mem.sliceAsBytes(&table));
        p.xr = try p.rig.input(std.mem.sliceAsBytes(p.x));
        p.rt = .{
            .ids = try p.rig.input(std.mem.sliceAsBytes(pl.ids)),
            .weights = try p.rig.input(std.mem.sliceAsBytes(pl.weights)),
            .slots = try p.rig.input(std.mem.sliceAsBytes(pl.slots)),
            .nslots = try p.rig.input(std.mem.sliceAsBytes(&pl.nslots)),
            .pairs = try p.rig.input(std.mem.sliceAsBytes(pl.pairs)),
            .where = try p.rig.input(std.mem.sliceAsBytes(pl.where)),
        };
        p.out = try p.rig.output(p.count() * 2);
        if (p.v.path == .units) {
            const b = xc.bytes(p.xcall());
            p.tiles = try p.rig.scratch(b[0]);
            if (up) {
                p.codes = try p.rig.scratch(b[1]);
                p.scales = try p.rig.scratch(b[2]);
            } else p.dq = try p.rig.scratch(b[5]);
        } else if (up) {
            p.codes = try p.rig.scratch(p.shape.rows * p.shape.k * 2);
            p.scales = try p.rig.scratch(p.shape.rows * p.shape.k / 8);
        }
    }

    fn at(r: rig_.Ref) xc.Ref {
        return .{ .buf = r.buf, .off = r.off };
    }

    fn route(p: *const Proj) xc.Route {
        const r = p.rt;
        return .{ .ids = at(r.ids), .weights = at(r.weights), .slots = at(r.slots), .nslots = at(r.nslots), .pairs = at(r.pairs), .where = at(r.where) };
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        xc.use(pass.e, p.backing[0..p.nback]);
        if (p.v.path == .decode) {
            if (p.v.which == .up) {
                dec.quant(&p.dp.?, pass.e, at(p.xr), at(p.codes), at(p.scales), p.dcall());
                dec.upCoded(&p.dp.?, pass.e, at(p.codes), at(p.scales), at(p.table), p.route(), at(p.out), p.dcall());
            } else dec.down(&p.dp.?, pass.e, at(p.xr), at(p.table), p.route(), at(p.out), p.dcall());
        } else {
            xc.tiles(&p.xp.?, pass.e, p.route(), at(p.tiles));
            if (p.v.which == .up) {
                xc.quant(&p.xp.?, pass.e, at(p.xr), at(p.codes), at(p.scales), p.xcall());
                xc.up(&p.xp.?, pass.e, at(p.codes), at(p.scales), at(p.table), p.route(), at(p.tiles), at(p.out), p.xcall());
            } else xc.down(&p.xp.?, pass.e, at(p.xr), at(p.dq), at(p.table), at(p.tiles), at(p.out), p.xcall());
        }
        try p.rig.finish(pass);
    }

    pub fn count(p: *Proj) usize {
        return p.pairs() * p.shape.n;
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        const n = p.count();
        const k = p.shape.k;
        p.spread = try p.a.alloc(f64, n);
        @memset(p.spread, 0);
        if (p.v.which == .down) {
            const xv = try p.a.alloc(f32, k);
            defer p.a.free(xv);
            for (0..p.pairs()) |pos| {
                for (xv, p.x[pos * k ..][0..k]) |*y, h| y.* = c.fromBf16(h);
                const ex_i = p.plan.?.expertAt(pos);
                for (0..p.shape.n) |j| {
                    const idx = pos * p.shape.n + j;
                    const d = ref.dot(p.m1.?.row(ex_i, j), p.m1.?.srow(ex_i, j), xv, null, k);
                    const nom = bf64(d.exact);
                    want[idx] = nom;
                    mag[idx] = d.mag;
                    if (p.v.wide and std.math.isFinite(d.exact)) {
                        // 48-bit dd vs 53-bit FP64: wide E8M0 midpoints perturbed 49..52 bits down can tie-break.
                        const eps = 0x1p-45 * d.mag;
                        const lo = bf64(d.exact - eps);
                        const hi = bf64(d.exact + eps);
                        p.spread[idx] = @max(@abs(@as(f64, lo) - nom), @abs(@as(f64, hi) - nom));
                    }
                }
            }
            return;
        }
        const qv = try p.a.alloc(f32, p.shape.rows * k);
        defer p.a.free(qv);
        const qs = try p.a.alloc(f32, p.shape.rows * k / 32);
        defer p.a.free(qs);
        for (0..p.shape.rows) |r| ref.quantRow(p.x[r * k ..][0..k], qv[r * k ..][0..k], qs[r * k / 32 ..][0 .. k / 32]);
        const mults = [3]f64{ 1, 1 + ref.exp_spread, 1 - ref.exp_spread };
        for (0..p.pairs()) |pos| {
            const pr = p.plan.?.pairs[pos];
            const r = pr / topk;
            const ex_i = p.plan.?.ids[pr];
            const xv = qv[r * k ..][0..k];
            const xs = qs[r * k / 32 ..][0 .. k / 32];
            for (0..p.shape.n / 32) |b| {
                var v: [3][32]f32 = undefined;
                for (0..32) |j| {
                    const col = 32 * b + j;
                    const gd = ref.dot(p.m1.?.row(ex_i, col), p.m1.?.srow(ex_i, col), xv, xs, k);
                    const ud = ref.dot(p.m3.?.row(ex_i, col), p.m3.?.srow(ex_i, col), xv, xs, k);
                    for (&v, mults) |*m, mu| m[j] = ref.swiglu(bf64(gd.exact), bf64(ud.exact), p.plan.?.weights[pr], mu);
                }
                var q: [3][32]f32 = undefined;
                for (&q, v) |*o, m| ref.quantBlock(&m, o);
                for (0..32) |j| {
                    const idx = pos * p.shape.n + 32 * b + j;
                    want[idx] = q[0][j];
                    mag[idx] = @abs(q[0][j]);
                    p.spread[idx] = @max(@abs(@as(f64, q[1][j]) - q[0][j]), @abs(@as(f64, q[2][j]) - q[0][j]));
                }
            }
        }
    }

    /// Down matches exact FP64 at 0.0 ULP (wide E8M0 allows 48-bit dd tie-break); up sits inside Metal exp's spread.
    pub fn bounds(p: *Proj, _: []const f64, _: []const f64, allow: []f64) !void {
        if (p.v.which == .up or p.v.wide) @memcpy(allow, p.spread) else @memset(allow, 0);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        const m1 = p.m1.?;
        const k = p.shape.k;
        const t = try p.a.alloc(f64, 2 * k);
        defer p.a.free(t);
        if (p.v.which == .down) {
            const xv = try p.a.alloc(f32, k);
            defer p.a.free(xv);
            for (0..p.pairs()) |pos| {
                for (xv, p.x[pos * k ..][0..k]) |*y, h| y.* = c.fromBf16(h);
                const ex_i = p.plan.?.expertAt(pos);
                for (0..p.shape.n) |j| {
                    ref.terms(m1.row(ex_i, j), m1.srow(ex_i, j), xv, null, k, t);
                    out[pos * p.shape.n + j] = ref.bf(@floatCast(contract.rows.sumIn(acc, order, t[0..k])));
                }
            }
            return;
        }
        const qv = try p.a.alloc(f32, k);
        defer p.a.free(qv);
        const qs = try p.a.alloc(f32, k / 32);
        defer p.a.free(qs);
        for (0..p.pairs()) |pos| {
            const pr = p.plan.?.pairs[pos];
            const ex_i = p.plan.?.ids[pr];
            ref.quantRow(p.x[pr / topk * k ..][0..k], qv, qs);
            for (0..p.shape.n / 32) |b| {
                var v: [32]f32 = undefined;
                for (&v, 0..) |*y, j| {
                    const col = 32 * b + j;
                    ref.terms(m1.row(ex_i, col), m1.srow(ex_i, col), qv, qs, k, t[0..k]);
                    ref.terms(p.m3.?.row(ex_i, col), p.m3.?.srow(ex_i, col), qv, qs, k, t[k..]);
                    const gs: f32 = @floatCast(contract.rows.sumIn(acc, order, t[0..k]));
                    const us: f32 = @floatCast(contract.rows.sumIn(acc, order, t[k..]));
                    y.* = ref.swiglu(gs, us, p.plan.?.weights[pr], 1);
                }
                var q: [32]f32 = undefined;
                ref.quantBlock(&v, &q);
                for (q, 0..) |y, j| out[pos * p.shape.n + 32 * b + j] = y;
            }
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: std.ArrayList(*Proj) = .empty,

    const list = [_]struct { v: Variant, name: []const u8 }{
        .{ .v = .{ .which = .up, .path = .decode }, .name = base ++ "decode up" },
        .{ .v = .{ .which = .up, .path = .decode, .ctl = 1 }, .name = base ++ "decode control: up non-finite flushed" },
        .{ .v = .{ .which = .up, .path = .decode, .ctl = 2 }, .name = base ++ "decode control: up bf16 step dropped" },
        .{ .v = .{ .which = .up, .path = .decode, .ctl = 3 }, .name = base ++ "decode control: up sums rounded to fp16" },
        .{ .v = .{ .which = .up, .path = .decode, .ctl = 4 }, .name = base ++ "decode control: up writes past output" },
        .{ .v = .{ .which = .down, .path = .decode }, .name = base ++ "decode down" },
        .{ .v = .{ .which = .down, .path = .decode, .wide = true }, .name = base ++ "decode down: full E8M0 range" },
        .{ .v = .{ .which = .down, .path = .decode, .ctl = 1 }, .name = base ++ "decode control: down non-finite flushed" },
        .{ .v = .{ .which = .down, .path = .decode, .ctl = 2 }, .name = base ++ "decode control: down last K group dropped" },
        .{ .v = .{ .which = .down, .path = .decode, .ctl = 3 }, .name = base ++ "decode control: down sums rounded to fp16" },
        .{ .v = .{ .which = .down, .path = .decode, .ctl = 4 }, .name = base ++ "decode control: down writes past output" },
        .{ .v = .{ .which = .up, .path = .units }, .name = base ++ "units up" },
        .{ .v = .{ .which = .up, .path = .units, .ctl = 1 }, .name = base ++ "units control: up non-finite flushed" },
        .{ .v = .{ .which = .up, .path = .units, .ctl = 2 }, .name = base ++ "units control: up bf16 step dropped" },
        .{ .v = .{ .which = .up, .path = .units, .ctl = 3 }, .name = base ++ "units control: up sums rounded to fp16" },
        .{ .v = .{ .which = .up, .path = .units, .ctl = 4 }, .name = base ++ "units control: up writes past output" },
        .{ .v = .{ .which = .down, .path = .units }, .name = base ++ "units down" },
        .{ .v = .{ .which = .down, .path = .units, .wide = true }, .name = base ++ "units down: full E8M0 range" },
        .{ .v = .{ .which = .down, .path = .units, .ctl = 1 }, .name = base ++ "units control: down non-finite flushed" },
        .{ .v = .{ .which = .down, .path = .units, .ctl = 5 }, .name = base ++ "units control: down last K group dropped" },
        .{ .v = .{ .which = .down, .path = .units, .ctl = 3 }, .name = base ++ "units control: down sums rounded to fp16" },
        .{ .v = .{ .which = .down, .path = .units, .ctl = 4 }, .name = base ++ "units control: down writes past output" },
        .{ .v = .{ .which = .down, .path = .units, .ctl = 6 }, .name = base ++ "units control: down unsafe groups unpatched" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a, .device = device, .queue = queue };
        if (!xc.supported(device)) return s;
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            p.* = try Proj.init(a, device, queue, entry.v, entry.name);
            try s.probes.append(a, p);
            try kernels.append(a, p.kernel());
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }

    /// Prompt M5 tensor-unit experts equal decode certified experts byte-for-byte across row counts and NaN/Inf.
    pub fn equality(s: *Suite, seed: u64) !usize {
        if (!xc.supported(s.device) or s.probes.items.len == 0) return 0;
        const dsrc = try xc.decodeSource(s.a, "");
        defer s.a.free(dsrc);
        var dp = try dec.Pipelines.loadSourceCert(s.device, dsrc);
        defer dp.deinit();
        var xp = try xc.Pipelines.load(s.a, s.device);
        defer xp.deinit();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        var bad: usize = 0;
        var cells: usize = 0;
        for ([_]struct { rows: u32, nonfinite: bool }{ .{ .rows = 1, .nonfinite = false }, .{ .rows = 5, .nonfinite = false }, .{ .rows = 16, .nonfinite = false }, .{ .rows = 16, .nonfinite = true } }) |case| {
            var rig: rig_.Rig = .{ .a = s.a, .device = s.device, .queue = s.queue };
            defer rig.deinit();
            const e_cnt: u32 = 16;
            const hid: u32 = 5120;
            const width: u32 = 2304;
            const out_n: u32 = 256;
            const m1 = try ref.Mats.init(s.a, e_cnt, width, hid);
            defer m1.deinit();
            const m3 = try ref.Mats.init(s.a, e_cnt, width, hid);
            defer m3.deinit();
            const m2 = try ref.Mats.init(s.a, e_cnt, out_n, width);
            defer m2.deinit();
            m1.fill(rng, false);
            m3.fill(rng, false);
            m2.fill(rng, false);
            const r1p = try rig.input(m1.codes);
            const r1s = try rig.input(m1.scales);
            const r3p = try rig.input(m3.codes);
            const r3s = try rig.input(m3.scales);
            const r2p = try rig.input(m2.codes);
            const r2s = try rig.input(m2.scales);
            const backing = [6]mtl.Buffer{ r1p.buf, r1s.buf, r3p.buf, r3s.buf, r2p.buf, r2s.buf };
            var table: [16]xc.ExpertPtrs = undefined;
            for (&table, 0..) |*t, i| t.* = .{
                .w1p = r1p.buf.gpuAddress() + r1p.off + i * width * hid / 2,
                .w1s = r1s.buf.gpuAddress() + r1s.off + i * width * hid / 32,
                .w3p = r3p.buf.gpuAddress() + r3p.off + i * width * hid / 2,
                .w3s = r3s.buf.gpuAddress() + r3s.off + i * width * hid / 32,
                .w2p = r2p.buf.gpuAddress() + r2p.off + i * out_n * width / 2,
                .w2s = r2s.buf.gpuAddress() + r2s.off + i * out_n * width / 32,
            };
            const tbl = try rig.input(std.mem.sliceAsBytes(&table));
            const plan = try ref.Plan.init(s.a, case.rows, e_cnt, rng);
            defer plan.deinit();
            const xv = try s.a.alloc(u16, case.rows * hid);
            defer s.a.free(xv);
            ref.fillX(xv, rng);
            if (case.nonfinite) ref.plantNonfinite(xv, case.rows, hid);
            const xr = try rig.input(std.mem.sliceAsBytes(xv));
            const ids = try rig.input(std.mem.sliceAsBytes(plan.ids));
            const wts = try rig.input(std.mem.sliceAsBytes(plan.weights));
            const np = case.rows * topk;
            const slots = try rig.scratch(12 * np);
            const nslots = try rig.scratch(4);
            const pairs = try rig.scratch(4 * np);
            const where = try rig.scratch(4 * np);
            const act_d = try rig.output(2 * np * width);
            const out_d = try rig.output(2 * np * out_n);
            const act_x = try rig.output(2 * np * width);
            const out_x = try rig.output(2 * np * out_n);
            const call: xc.Call = .{ .rows = case.rows, .k = hid, .n = width, .experts = e_cnt };
            const dcall: dec.Call = .{ .rows = case.rows, .k = hid, .n = width, .experts = e_cnt };
            const b = xc.bytes(call);
            const tiles = try rig.scratch(b[0]);
            const codes = try rig.scratch(b[1]);
            const scales = try rig.scratch(b[2]);
            const dq = try rig.scratch(b[5]);
            const rt: xc.Route = .{ .ids = .{ .buf = ids.buf, .off = ids.off }, .weights = .{ .buf = wts.buf, .off = wts.off }, .slots = .{ .buf = slots.buf, .off = slots.off }, .nslots = .{ .buf = nslots.buf, .off = nslots.off }, .pairs = .{ .buf = pairs.buf, .off = pairs.off }, .where = .{ .buf = where.buf, .off = where.off } };
            const pass = rig.begin();
            xc.use(pass.e, &backing);
            dec.plan(&dp, pass.e, rt, dcall);
            dec.up(&dp, pass.e, .{ .buf = xr.buf, .off = xr.off }, .{ .buf = tbl.buf, .off = tbl.off }, rt, .{ .buf = act_d.buf, .off = act_d.off }, dcall);
            dec.down(&dp, pass.e, .{ .buf = act_d.buf, .off = act_d.off }, .{ .buf = tbl.buf, .off = tbl.off }, rt, .{ .buf = out_d.buf, .off = out_d.off }, .{ .rows = case.rows, .k = out_n, .n = width, .experts = e_cnt });
            xc.tiles(&xp, pass.e, rt, .{ .buf = tiles.buf, .off = tiles.off });
            xc.quant(&xp, pass.e, .{ .buf = xr.buf, .off = xr.off }, .{ .buf = codes.buf, .off = codes.off }, .{ .buf = scales.buf, .off = scales.off }, call);
            xc.up(&xp, pass.e, .{ .buf = codes.buf, .off = codes.off }, .{ .buf = scales.buf, .off = scales.off }, .{ .buf = tbl.buf, .off = tbl.off }, rt, .{ .buf = tiles.buf, .off = tiles.off }, .{ .buf = act_x.buf, .off = act_x.off }, call);
            xc.down(&xp, pass.e, .{ .buf = act_x.buf, .off = act_x.off }, .{ .buf = dq.buf, .off = dq.off }, .{ .buf = tbl.buf, .off = tbl.off }, .{ .buf = tiles.buf, .off = tiles.off }, .{ .buf = out_x.buf, .off = out_x.off }, .{ .rows = case.rows, .k = out_n, .n = width, .experts = e_cnt });
            try rig.finish(pass);
            bad += differ(rig.bytes(act_d, 2 * np * width), rig.bytes(act_x, 2 * np * width));
            bad += differ(rig.bytes(out_d, 2 * np * out_n), rig.bytes(out_x, 2 * np * out_n));
            cells += np * (width + out_n);
        }
        std.debug.print("{s} {s}prompt/decode equality: rows 1/5/16/16(non-finite), {d} values, {d} differ\n", .{ if (bad == 0) "PASS" else "FAIL", base, cells, bad });
        return bad;
    }
};

fn differ(a: []const u8, b: []const u8) usize {
    const au: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, a));
    const bu: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, b));
    var n: usize = 0;
    for (au, bu) |x, y| {
        const both_nan = (x & 0x7fff) > 0x7f80 and (y & 0x7fff) > 0x7f80;
        if (x != y and !both_nan) n += 1;
    }
    return n;
}
